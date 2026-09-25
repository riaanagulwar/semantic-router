"""
Thin wrapper around the Gemini API, used only as a second opinion when the
embedding router (app/router.py) isn't confident about a query's intent.

Gemini is used here strictly as a CLASSIFIER, never as a free-text answer
generator: it's asked to pick one of the app's own known route names (or
say none fit) and nothing else. That output is then validated against the
same fixed set of routes the embedding router itself is constrained to
(see the "Fixed route set" guardrail in router.py) — Gemini's raw text
response is never trusted directly as a route name, only checked for
membership in the allow-list it was given. If it doesn't clearly pick one,
the caller (app/core.py) falls through to a fixed, non-generated message.
This avoids handing an ungrounded, unscoped LLM answer straight to the
user for a domain (banking) where a confidently wrong made-up answer is
worse than an honest "I can't help with that."

Guardrails implemented here:
  1. Token-bucket rate limiter - caps calls per minute so quota is never
                                  exceeded (and never silently hammered
                                  when traffic spikes).
  2. Capped retries + backoff  - at most MAX_RETRIES attempts, with a short
                                  sleep between them. Never loops forever.
  3. Per-call timeout          - a hung request doesn't hang the whole router.
  4. Fail-safe on any error    - quota exceeded, network error, timeout,
                                  retries exhausted, or an unrecognized
                                  reply all resolve to "no match" (None)
                                  rather than raising or guessing.

Set GEMINI_API_KEY before running (get one from https://aistudio.google.com).
"""

import threading
import time
import logging

from google import genai
from google.genai import types

from app.config.settings import settings

logger = logging.getLogger("gemini_client")

# Defaults below match the project's original hardcoded values; all are
# now overridable via app/config/settings.py (env vars). Check
# https://ai.google.dev/gemini-api/docs/models for current model names and
# update GEMINI_MODEL_NAME if it's changed.
MODEL_NAME = settings.gemini_model_name
MAX_RPM = settings.max_rpm          # stay comfortably under the provider's per-minute limit
MAX_RETRIES = settings.max_retries

# Google's API rejects any deadline below 10s with a 400 INVALID_ARGUMENT
# ("Manually set deadline Ns is too short") — that's a permanent
# misconfiguration, not a flaky failure, so every retry would fail
# identically. Clamp up defensively (with a loud warning) rather than let
# a low TIMEOUT_S silently degrade every real call to a "no match" result.
MIN_TIMEOUT_S = 10
if settings.timeout_s < MIN_TIMEOUT_S:
    logger.warning(
        "TIMEOUT_S=%s is below the Gemini API's minimum deadline of %ds; "
        "using %ds instead.",
        settings.timeout_s, MIN_TIMEOUT_S, MIN_TIMEOUT_S,
    )
TIMEOUT_S = max(settings.timeout_s, MIN_TIMEOUT_S)

SAFE_FALLBACK_MESSAGE = (
    "I'm not able to answer that right now — please try again in a moment "
    "or rephrase your question."
)

NO_MATCH_TOKEN = "none"


class TokenBucket:
    """Simple per-minute rate limiter, thread-safe."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = float(max_per_minute)
        self.lock = threading.Lock()
        self.last_refill = time.monotonic()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_per_minute,
                self.tokens + elapsed * (self.max_per_minute / 60.0),
            )
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class GeminiFallback:
    def __init__(self):
        api_key = settings.gemini_api_key
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get a key at https://aistudio.google.com and export it."
            )
        self.client = genai.Client(api_key=api_key)
        self.bucket = TokenBucket(MAX_RPM)

    def classify_intent(self, query: str, candidate_routes: list[str]) -> str | None:
        """Ask Gemini to pick one of candidate_routes, or return None.

        The reply is only ever accepted if it's an exact (case-insensitive)
        match to one of the names it was given — never trusted as arbitrary
        text. Any failure (rate limit, network, timeout, retries exhausted,
        or a reply that isn't one of the given names) returns None, which
        the caller treats identically to "Gemini declined to match" — this
        never raises into the caller.
        """
        if not self.bucket.allow():
            logger.warning("rate limit reached, refusing LLM classification for query=%r", query)
            return None

        instruction = (
            "You are an intent classifier for a banking assistant. Decide "
            "whether the user's message clearly matches exactly one of "
            f"these categories: {', '.join(candidate_routes)}. "
            f"Reply with ONLY the matching category name, or \"{NO_MATCH_TOKEN}\" "
            "if none of them clearly fit. Do not explain your answer, add "
            "punctuation, or invent a category that isn't in the list."
        )

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=query,
                    config=types.GenerateContentConfig(
                        system_instruction=instruction,
                        max_output_tokens=20,
                        temperature=0.0,
                        # Newer Gemini models spend "thinking" tokens before
                        # producing visible output, and those count against
                        # max_output_tokens — without this, a low cap (fine
                        # for a one-word classification reply) gets consumed
                        # entirely by internal reasoning, leaving 0 tokens
                        # for the actual answer (finish_reason=MAX_TOKENS,
                        # response.text=None, even though the call
                        # succeeded). This is a one-label classification;
                        # it doesn't need a reasoning pass.
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000),
                    ),
                )
                raw = (response.text or "").strip().strip(".").lower()
                for route in candidate_routes:
                    if raw == route.lower():
                        return route
                return None  # "none", empty, or anything unrecognized
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "gemini classification failed (attempt %d/%d): %s",
                    attempt, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)  # small backoff before retrying

        logger.error("gemini classification exhausted retries: %s", last_error)
        return None


# Lazy, thread-safe singleton. GeminiFallback.__init__ raises RuntimeError
# if GEMINI_API_KEY isn't set — constructing it eagerly at import time
# would mean `import app.gemini_client` (or anything importing it
# transitively, like app.core) crashes in any environment without the key,
# breaking pytest collection and any process that imports it before the
# key is available. Deferring construction to first real use means import
# is always safe; only an actual classification attempt touches the env
# var.
_instance: GeminiFallback | None = None
_instance_lock = threading.Lock()


def get_fallback() -> GeminiFallback:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = GeminiFallback()
    return _instance
