"""
Handler regression tests. Importing handlers.py must never crash — that's
the whole point of the lazy GeminiFallback singleton fix — so none of
these tests need GEMINI_API_KEY set.
"""

import yaml

from app.handlers import HANDLERS, balance_handler, refund_handler, history_handler, small_talk_handler
from app.gemini_client import SAFE_FALLBACK_MESSAGE


def test_every_configured_route_has_a_registered_handler():
    with open("app/config/routes.yaml") as f:
        cfg = yaml.safe_load(f)
    for route in cfg["routes"]:
        assert route["name"] in HANDLERS, (
            f"route {route['name']!r} is defined in app/config/routes.yaml but "
            "has no matching entry in handlers.HANDLERS"
        )


def test_mocked_handlers_return_unchanged_strings():
    # Locks in "don't touch business logic" as an actual regression guard.
    assert balance_handler("q") == "[balance_handler] Your current balance is $1,240.50 (mocked)."
    assert refund_handler("q") == "[refund_handler] Refund initiated for your most recent order (mocked)."
    assert history_handler("q") == "[history_handler] Last 3 transactions: -$40 Swiggy, -$12 Uber, +$1000 Salary (mocked)."
    assert small_talk_handler("q") == "Hey! How can I help with your account today?"


def test_llm_fallback_handler_fails_safe_without_api_key(monkeypatch):
    monkeypatch.setattr("app.config.settings.settings.gemini_api_key", None)
    import app.handlers as handlers

    monkeypatch.setattr(handlers, "_gemini", None)
    result = handlers.llm_fallback_handler("anything")
    assert result == SAFE_FALLBACK_MESSAGE
