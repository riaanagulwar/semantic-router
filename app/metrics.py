"""
Prometheus metrics, defined once here so both api.py (which exposes them at
GET /metrics) and handlers.py (which increments a couple of them inline,
pure instrumentation around existing calls — no change to return values)
import the same instances.
"""

from prometheus_client import Counter, Histogram

requests_total = Counter(
    "requests_total", "Total routed requests", ["route"]
)
llm_fallback_total = Counter(
    "llm_fallback_total", "Total invocations of the llm_fallback handler"
)
llm_fallback_errors_total = Counter(
    "llm_fallback_errors_total",
    "llm_fallback invocations that returned the safe-fallback message "
    "instead of a real Gemini answer (rate-limited, missing key, or error)",
)
request_latency_seconds = Histogram(
    "request_latency_seconds", "End-to-end request latency", ["route"]
)
rate_limit_rejections_total = Counter(
    "rate_limit_rejections_total", "Requests rejected by the API-level rate limiter"
)
