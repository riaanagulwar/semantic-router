"""
Route handlers. Each is a plain function registered in the HANDLERS dict
below — the only place routes are wired to code.

Guardrail: main.py/api.py dispatch by looking a route name up in HANDLERS.
They never do getattr()/eval() on a route name or on model output, so a
route can only ever call a function a developer explicitly registered
here.

The non-LLM handlers below are mocked (no real backend) — swap in real API
calls when you have something to call. The point of the project is the
routing and guardrails, not these stubs.
"""

import logging
import threading

from app.gemini_client import SAFE_FALLBACK_MESSAGE, GeminiFallback
from app.metrics import llm_fallback_errors_total, llm_fallback_total

logger = logging.getLogger("handlers")

# Lazy, thread-safe singleton. GeminiFallback.__init__ raises RuntimeError
# if GEMINI_API_KEY isn't set — instantiating it eagerly at import time (as
# this module used to) meant `import handlers` itself crashed in any
# environment without the key, which broke pytest collection and any
# process (API, CLI) that imports this module before the key is available.
# Deferring construction to first real use means import is always safe;
# only an actual llm_fallback query touches the env var.
_gemini: GeminiFallback | None = None
_gemini_lock = threading.Lock()


def _get_gemini() -> GeminiFallback:
    global _gemini
    if _gemini is None:
        with _gemini_lock:
            if _gemini is None:
                _gemini = GeminiFallback()
    return _gemini


def balance_handler(query: str) -> str:
    return "[balance_handler] Your current balance is $1,240.50 (mocked)."


def refund_handler(query: str) -> str:
    return "[refund_handler] Refund initiated for your most recent order (mocked)."


def history_handler(query: str) -> str:
    return "[history_handler] Last 3 transactions: -$40 Swiggy, -$12 Uber, +$1000 Salary (mocked)."


def small_talk_handler(query: str) -> str:
    return "Hey! How can I help with your account today?"


def llm_fallback_handler(query: str) -> str:
    """The only handler that reaches an LLM call — used when the router
    isn't confident a query matches a known route. Rate-limited and
    retried inside GeminiFallback (see gemini_client.py).

    If GEMINI_API_KEY isn't configured, fails safe the same way
    GeminiFallback.answer() already does for quota/network/timeout errors
    — return the canned safe message rather than raising into the caller.
    """
    llm_fallback_total.inc()
    try:
        answer = _get_gemini().answer(query)
    except RuntimeError:
        logger.error("llm_fallback unavailable: GEMINI_API_KEY not configured")
        answer = SAFE_FALLBACK_MESSAGE

    # answer() never raises — returning SAFE_FALLBACK_MESSAGE is its
    # documented signal that the call didn't produce a real answer (rate
    # limited, timed out, errored, or retries exhausted), so that's what
    # the error counter tracks.
    if answer == SAFE_FALLBACK_MESSAGE:
        llm_fallback_errors_total.inc()
    return answer


HANDLERS = {
    "balance_query": balance_handler,
    "refund_request": refund_handler,
    "transaction_history": history_handler,
    "small_talk": small_talk_handler,
    "llm_fallback": llm_fallback_handler,
}
