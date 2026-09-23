"""
gemini_client.py tests — no real network call or API key needed. The
google.genai client is mocked throughout.
"""

import importlib
from unittest.mock import MagicMock

import pytest

import app.gemini_client as gemini_client_module
from app.gemini_client import (
    MAX_RETRIES,
    SAFE_FALLBACK_MESSAGE,
    GeminiFallback,
    TokenBucket,
)


@pytest.fixture
def fake_client():
    return MagicMock()


@pytest.fixture
def fallback(monkeypatch, fake_client):
    monkeypatch.setattr("app.config.settings.settings.gemini_api_key", "fake-key")
    monkeypatch.setattr("app.gemini_client.genai.Client", lambda api_key: fake_client)
    monkeypatch.setattr("app.gemini_client.time.sleep", lambda *_: None)  # skip backoff
    return GeminiFallback()


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr("app.config.settings.settings.gemini_api_key", None)
    with pytest.raises(RuntimeError):
        GeminiFallback()


def test_token_bucket_allows_up_to_capacity():
    bucket = TokenBucket(max_per_minute=3)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]


def test_successful_call_returns_response_text(fallback, fake_client):
    fake_client.models.generate_content.return_value = MagicMock(text="hello there")
    assert fallback.answer("hi") == "hello there"


def test_blank_response_falls_back_to_safe_message(fallback, fake_client):
    fake_client.models.generate_content.return_value = MagicMock(text="   ")
    assert fallback.answer("hi") == SAFE_FALLBACK_MESSAGE


def test_capped_retries_then_safe_fallback(fallback, fake_client):
    fake_client.models.generate_content.side_effect = RuntimeError("boom")

    assert fallback.answer("hi") == SAFE_FALLBACK_MESSAGE
    assert fake_client.models.generate_content.call_count == MAX_RETRIES


def test_rate_limit_exhausted_returns_safe_fallback_without_calling_api(
    fallback, fake_client
):
    fallback.bucket = TokenBucket(max_per_minute=0)  # never allows

    assert fallback.answer("hi") == SAFE_FALLBACK_MESSAGE
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
