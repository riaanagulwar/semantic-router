"""
Prometheus metrics, defined once here so both api.py (which exposes them at
GET /metrics) and core.py / handlers.py (which increment them inline, pure
instrumentation around existing calls — no change to routing/response
behavior) import the same instances.
"""

from prometheus_client import Counter, Histogram

requests_total = Counter(
    "requests_total", "Total routed requests", ["route"]
)
llm_classification_attempts_total = Counter(
    "llm_classification_attempts_total",
    "Times the embedding router wasn't confident and Gemini was asked to "
    "classify the query into a known route",
)
llm_reclassified_total = Counter(
    "llm_reclassified_total",
    "Times Gemini's classification attempt matched a known route, so the "
    "query was routed there instead of the static no-match message",
    ["route"],
)
llm_no_match_total = Counter(
    "llm_no_match_total",
    "Times Gemini's classification attempt did not resolve to a known "
    "route (declined, rate-limited, missing key, or error) and the query "
    "got the static no-match message",
)
request_latency_seconds = Histogram(
    "request_latency_seconds", "End-to-end request latency", ["route"]
)
rate_limit_rejections_total = Counter(
    "rate_limit_rejections_total", "Requests rejected by the API-level rate limiter"
)
