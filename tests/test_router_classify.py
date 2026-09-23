"""
Exercises the real SemanticRouter against the real app/config/routes.yaml.
Loads the actual sentence-transformers model, so it's slower than a pure
unit test, but deterministic — cosine similarity against a fixed model and
fixed utterances doesn't vary between runs.
"""

import pytest

from app.router import SemanticRouter


@pytest.fixture(scope="module")
def semantic_router():
    return SemanticRouter(config_path="app/config/routes.yaml")


def test_clear_balance_query_is_confident(semantic_router):
    match = semantic_router.classify("What's my account balance?")
    assert match.confident is True
    assert match.route == "balance_query"
    assert match.reason == "ok"


def test_nonsense_query_falls_below_threshold(semantic_router):
    match = semantic_router.classify("asdkjhasdkjh qwopqwop zzzxxx")
    assert match.confident is False
    assert match.reason == "below_threshold"


def test_straddling_query_is_ambiguous_or_below_threshold(semantic_router):
    # Deliberately spans balance/refund/history vocabulary — should never
    # be trusted as a single confident route.
    match = semantic_router.classify("balance transfer refund history")
    assert match.confident is False
    assert match.reason in {"ambiguous", "below_threshold"}


def test_every_route_in_config_is_reachable(semantic_router):
    # Regression guard: each route's own example utterances should route
    # back to that same route confidently.
    for name, utterances in semantic_router.routes.items():
        match = semantic_router.classify(utterances[0])
        assert match.confident is True
        assert match.route == name
