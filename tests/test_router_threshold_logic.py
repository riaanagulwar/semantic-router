"""
Fast unit-level check of the threshold/margin arithmetic in
SemanticRouter.classify(), independent of the real ML model's actual
similarity behavior — SentenceTransformer.encode is stubbed out entirely,
so this runs in milliseconds and can't be flaky due to model drift.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from app.router import SemanticRouter


@pytest.fixture
def router_with_fake_model(monkeypatch):
    # Skip the real (slow) SentenceTransformer load entirely. _load_config
    # calls .encode() once per route during __init__, so the stub needs a
    # dummy implementation too (the real vectors get overwritten below).
    fake_model = SimpleNamespace(encode=lambda texts, normalize_embeddings=True: np.zeros((len(texts), 2)))
    monkeypatch.setattr("app.router.SentenceTransformer", lambda *_a, **_kw: fake_model)
    r = SemanticRouter(config_path="app/config/routes.yaml")
    r._route_embeddings = {
        "a": np.array([[1.0, 0.0]]),
        "b": np.array([[0.0, 1.0]]),
    }
    r.routes = {"a": ["x"], "b": ["y"]}
    return r


def _stub_query_vec(r, vec):
    r.model.encode = lambda queries, normalize_embeddings=True: np.array([vec])


def test_confident_when_clearly_above_threshold_and_margin(router_with_fake_model):
    r = router_with_fake_model
    _stub_query_vec(r, [1.0, 0.0])
    match = r.classify("irrelevant, encode is stubbed")
    assert match.route == "a"
    assert match.confident is True
    assert match.reason == "ok"


def test_below_threshold_when_similarity_is_weak(router_with_fake_model):
    r = router_with_fake_model
    _stub_query_vec(r, [0.1, 0.1])
    match = r.classify("weak match")
    assert match.confident is False
    assert match.reason == "below_threshold"


def test_ambiguous_when_top_two_routes_are_tied(router_with_fake_model):
    r = router_with_fake_model
    r._route_embeddings = {
        "a": np.array([[0.9, 0.1]]),
        "b": np.array([[0.89, 0.11]]),
    }
    _stub_query_vec(r, [1.0, 0.0])
    match = r.classify("ambiguous")
    assert match.confident is False
    assert match.reason == "ambiguous"
