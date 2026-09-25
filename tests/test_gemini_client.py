"""
gemini_client.py tests — no real network call or API key needed. The
google.genai client is mocked throughout.
"""

import importlib
from unittest.mock import MagicMock

import pytest

import app.gemini_client as gemini_client_module
from app.gemini_client import GeminiFallback, TokenBucket

ROUTES = ["balance_query", "refund_request", "transaction_history", "small_talk"]


@pytest.fixture
def fake_client():
    return MagicMock()


@pytest.fixture
def fallback(monkeypatch, fake_client):
    monkeypatch.setattr("app.config.settings.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("app.gemini_client.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("app.gemini_client.time.sleep", lambda *_: None)  # skip backoff
    return GeminiFallback()


def _respond(fake_client, text):
    fake_client.models.generate_content.return_value = MagicMock(text=text)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr("app.config.settings.settings.gemini_api_key", None)
    with pytest.raises(RuntimeError):
        GeminiFallback()


def test_token_bucket_allows_up_to_capacity():
    bucket = TokenBucket(max_per_minute=3)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]


def test_matching_reply_returns_the_route(fallback, fake_client):
    _respond(fake_client, "balance_query")
    assert fallback.classify_intent("what's my balance", ROUTES) == "balance_query"


def test_thinking_is_disabled_for_the_classification_call(fallback, fake_client):
    """Regression test: newer Gemini models spend "thinking" tokens before
    producing visible output, and those count against max_output_tokens.
    Without thinking_budget=0, a low cap (fine for a one-word reply) gets
    consumed entirely by internal reasoning — finish_reason=MAX_TOKENS,
    response.text=None, even though the call itself succeeds. Caught live:
    a real query returned an empty reply and silently fell through to
    "no match" with no error raised anywhere."""
    _respond(fake_client, "balance_query")
    fallback.classify_intent("what's my balance", ROUTES)

    _, kwargs = fake_client.models.generate_content.call_args
    thinking_config = kwargs["config"].thinking_config
    assert thinking_config is not None
    assert thinking_config.thinking_budget == 0


def test_reply_is_case_and_punctuation_insensitive(fallback, fake_client):
    _respond(fake_client, "Balance_Query.")
    assert fallback.classify_intent("what's my balance", ROUTES) == "balance_query"


def test_explicit_none_reply_returns_none(fallback, fake_client):
    _respond(fake_client, "none")
    assert fallback.classify_intent("what's the weather", ROUTES) is None


def test_unrecognized_reply_returns_none_not_an_invented_route(fallback, fake_client):
    # Guardrail: a reply that isn't one of the given candidates is never
    # trusted, even if it looks plausible — this is what stops Gemini's
    # raw text from ever being used as an arbitrary route/dict key.
    _respond(fake_client, "some_route_i_made_up")
    assert fallback.classify_intent("anything", ROUTES) is None


def test_capped_retries_then_none(fallback, fake_client):
    fake_client.models.generate_content.side_effect = RuntimeError("boom")

    assert fallback.classify_intent("hi", ROUTES) is None
    from app.gemini_client import MAX_RETRIES

    assert fake_client.models.generate_content.call_count == MAX_RETRIES


def test_rate_limit_exhausted_returns_none_without_calling_api(fallback, fake_client):
    fallback.bucket = TokenBucket(max_per_minute=0)  # never allows

    assert fallback.classify_intent("hi", ROUTES) is None
    fake_client.models.generate_content.assert_not_called()


def test_timeout_below_provider_minimum_is_clamped_at_import(monkeypatch):
    """Regression test: Google's API rejects any deadline under 10s with a
    permanent 400 INVALID_ARGUMENT (not a flaky failure — every retry fails
    identically). A misconfigured low TIMEOUT_S must never reach the API
    as-is."""
    monkeypatch.setattr("app.config.settings.settings.timeout_s", 5)
    try:
        importlib.reload(gemini_client_module)
        assert gemini_client_module.TIMEOUT_S >= gemini_client_module.MIN_TIMEOUT_S
        assert gemini_client_module.TIMEOUT_S == 10
    finally:
        importlib.reload(gemini_client_module)  # restore real settings for later tests


def test_get_fallback_is_a_lazy_singleton(monkeypatch, fake_client):
    monkeypatch.setattr("app.config.settings.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("app.gemini_client.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("app.gemini_client._instance", None)

    from app.gemini_client import get_fallback

    first = get_fallback()
    second = get_fallback()
    assert first is second
