"""
Tests for the two-stage routing decision in core.handle_query: confident
embedding match -> dispatch directly; not confident -> ask Gemini to
classify -> dispatch to whatever it matched, or the static no-match
message if it didn't. Gemini itself is mocked throughout — these tests
check the decision logic, not the model's actual judgment.
"""

from unittest.mock import MagicMock

import app.core as core_module
from app.core import handle_query


def test_confident_embedding_match_never_calls_gemini(monkeypatch):
    fake_reclassify = MagicMock()
    monkeypatch.setattr(core_module, "_try_llm_reclassify", fake_reclassify)

    result = handle_query("what's my account balance?")

    assert result.route == "balance_query"
    assert result.reclassified_by_llm is False
    fake_reclassify.assert_not_called()


def test_llm_reclassification_rescues_an_unsure_query(monkeypatch):
    monkeypatch.setattr(core_module, "_try_llm_reclassify", lambda query: "refund_request")

    result = handle_query("asdkjhasdkjh")  # nonsense -> embedding router unsure

    assert result.route == "refund_request"
    assert result.reclassified_by_llm is True
    assert "refund_handler" in result.response


def test_no_match_when_llm_also_cannot_classify(monkeypatch):
    monkeypatch.setattr(core_module, "_try_llm_reclassify", lambda query: None)

    result = handle_query("asdkjhasdkjh")

    assert result.route == "no_match"
    assert result.reclassified_by_llm is False


def test_llm_reclassify_treats_missing_api_key_as_no_match(monkeypatch):
    def raise_runtime_error():
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    # get_fallback is imported into core's namespace with `from ... import
    # get_fallback`, so it must be patched there, not on app.gemini_client
    # — patching the origin module wouldn't affect core's already-bound
    # name.
    monkeypatch.setattr(core_module, "get_fallback", raise_runtime_error)

    route = core_module._try_llm_reclassify("asdkjhasdkjh")

    assert route is None


def test_llm_reclassify_only_offers_configured_routes_as_candidates(monkeypatch):
    captured = {}

    def fake_classify(query, candidates):
        captured["candidates"] = candidates
        return None

    monkeypatch.setattr(
        core_module, "get_fallback", lambda: MagicMock(classify_intent=fake_classify)
    )

    core_module._try_llm_reclassify("anything")

    assert set(captured["candidates"]) == set(core_module.router.routes.keys())
