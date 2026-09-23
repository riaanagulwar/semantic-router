"""
Thin wrapper around the Gemini API, used only as the router's fallback
path for queries that didn't confidently match a known route.

Guardrails implemented here:
  1. Token-bucket rate limiter - caps calls per minute so quota is never
                                  exceeded (and never silently hammered
                                  when traffic spikes).
  2. Capped retries + backoff  - at most MAX_RETRIES attempts, with a short
                                  sleep between them. Never loops forever.
  3. Per-call timeout          - a hung request doesn't hang the whole router.
  4. Fail-safe on any error    - quota exceeded, network error, timeout, or
                                  retries exhausted all return a canned safe
                                  message instead of raising into the caller.

Set GEMINI_API_KEY before running (get one from https://aistudio.google.com).
"""

import time
import logging
import threading

from google import genai
from google.genai import types

from app.config.settings import settings

logger = logging.getLogger("gemini_client")

# Defaults below match the project's original hardcoded values; all are
# now overridable via config/settings.py (env vars). Check
# https://ai.google.dev/gemini-api/docs/models for current model names and
# update GEMINI_MODEL_NAME if it's changed.
MODEL_NAME = settings.gemini_model_name
MAX_RPM = settings.max_rpm          # stay comfortably under the provider's per-minute limit
MAX_RETRIES = settings.max_retries

# Google's API rejects any deadline below 10s with a 400 INVALID_ARGUMENT
# ("Manually set deadline Ns is too short") — that's a permanent
# misconfiguration, not a flaky failure, so every retry would fail
# identically. Clamp up defensively (with a loud warning) rather than let
# a low TIMEOUT_S silently degrade every real call to the safe fallback.
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

    def answer(self, query: str) -> str:
        # Guardrail: refuse before attempting the call if we're over budget —
        # fail cheap and safe instead of risking a 429 from the provider.
        if not self.bucket.allow():
            logger.warning("rate limit reached, refusing LLM call for query=%r", query)
            return SAFE_FALLBACK_MESSAGE

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=MODEL_NAME,
                    contents=query,
                    config=types.GenerateContentConfig(
                        max_output_tokens=200,
                        temperature=0.2,
                        http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000),
                    ),
                )
                return (response.text or "").strip() or SAFE_FALLBACK_MESSAGE
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "gemini call failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)  # small backoff before retrying

        logger.error("gemini call exhausted retries: %s", last_error)
        return SAFE_FALLBACK_MESSAGE
