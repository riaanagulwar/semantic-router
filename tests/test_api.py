import pytest
from fastapi.testclient import TestClient

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("app.config.settings.settings.api_key", API_KEY)
    import app.api as api

    # Rate-limit state is module-level and would otherwise leak between
    # tests (and make them order-dependent).
    api.limiter.reset()
    return TestClient(api.app)


def test_health_always_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_routes_loaded(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["routes_loaded"] == 4


def test_query_without_api_key_rejected(client):
    resp = client.post("/query", json={"query": "what's my balance"})
    assert resp.status_code == 401


def test_query_with_wrong_api_key_rejected(client):
    resp = client.post(
        "/query", json={"query": "what's my balance"}, headers={"X-API-Key": "nope"}
    )
    assert resp.status_code == 401


def test_query_with_correct_api_key_succeeds(client):
    resp = client.post("/query", json={"query": "what's my balance"}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "balance_query"
    assert body["confident"] is True
    assert "mocked" in body["response"]


def test_empty_query_rejected_with_422(client):
    resp = client.post("/query", json={"query": ""}, headers=AUTH)
    assert resp.status_code == 422


def test_oversized_query_rejected_with_422(client):
    from app.config.settings import settings

    resp = client.post(
        "/query", json={"query": "a" * (settings.max_query_len + 1)}, headers=AUTH
    )
    assert resp.status_code == 422


def test_request_id_echoed_back(client):
    resp = client.post(
        "/query",
        json={"query": "what's my balance"},
        headers={**AUTH, "X-Request-ID": "abc-123"},
    )
    assert resp.headers["X-Request-ID"] == "abc-123"


def test_metrics_endpoint_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "requests_total" in resp.text


def test_rate_limit_returns_429_past_the_configured_limit(client):
    from app.config.settings import settings

    # "hello" routes to small_talk, so none of these touch the LLM path.
    limit = int(settings.rate_limit.split("/")[0])
    statuses = [
        client.post("/query", json={"query": "hello"}, headers=AUTH).status_code
        for _ in range(limit + 1)
    ]

    assert statuses[:limit] == [200] * limit
    assert statuses[-1] == 429
