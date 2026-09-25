"""
Handler regression tests. Importing handlers.py must never crash — none
of these need GEMINI_API_KEY set, since no handler here calls Gemini
directly (that only happens in core.handle_query, before dispatch reaches
a handler at all — see tests/test_core.py).
"""

import yaml

from app.gemini_client import SAFE_FALLBACK_MESSAGE
from app.handlers import (
    HANDLERS,
    balance_handler,
    history_handler,
    no_match_handler,
    refund_handler,
    small_talk_handler,
)


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


def test_no_match_handler_returns_the_static_message_without_any_network_call():
    # No monkeypatching needed here at all — that's the point: this
    # handler never touches Gemini, so it can't fail on a missing key or
    # a network error either.
    assert no_match_handler("anything") == SAFE_FALLBACK_MESSAGE
