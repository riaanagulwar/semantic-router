"""
Route handlers. Each is a plain function registered in the HANDLERS dict
below — the only place routes are wired to code.

Guardrail: main.py/api.py dispatch by looking a route name up in HANDLERS.
They never do getattr()/eval() on a route name or on model output, so a
route can only ever call a function a developer explicitly registered
here.

The handlers below are all mocked (no real backend) — swap in real API
calls when you have something to call. The point of the project is the
routing and guardrails, not these stubs.

No handler here calls Gemini. The embedding router's low-confidence case
gets a second opinion from Gemini's intent classifier first (see
app/core.py's handle_query) — if that succeeds, dispatch lands on one of
the real handlers below just like a confident match would. no_match_handler
is only reached once both the embedding router AND Gemini's classification
attempt have failed to find a matching route, and it never touches the
network — it's always the same static message.
"""

from app.gemini_client import SAFE_FALLBACK_MESSAGE


def balance_handler(query: str) -> str:
    return "[balance_handler] Your current balance is $1,240.50 (mocked)."


def refund_handler(query: str) -> str:
    return "[refund_handler] Refund initiated for your most recent order (mocked)."


def history_handler(query: str) -> str:
    return "[history_handler] Last 3 transactions: -$40 Swiggy, -$12 Uber, +$1000 Salary (mocked)."


def small_talk_handler(query: str) -> str:
    return "Hey! How can I help with your account today?"


def no_match_handler(query: str) -> str:
    """Terminal fallback — reached only when neither the embedding router
    nor Gemini's classification attempt could confidently match a known
    route. Always the same static message; no LLM call happens here."""
    return SAFE_FALLBACK_MESSAGE


HANDLERS = {
    "balance_query": balance_handler,
    "refund_request": refund_handler,
    "transaction_history": history_handler,
    "small_talk": small_talk_handler,
    "no_match": no_match_handler,
}
