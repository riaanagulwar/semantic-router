"""
HTTP API — the production entry point (`uvicorn api:app`). Wraps the same
validate -> classify -> dispatch pipeline as main.py's CLI (both call
core.handle_query) behind FastAPI, adding: request-ID propagation,
structured JSON logging, Prometheus metrics, API-key auth, and per-client
rate limiting.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import logging
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config.logging_config import configure_logging
from app.config.request_context import set_request_id
from app.config.settings import settings
from app.core import handle_query, router
from app.metrics import rate_limit_rejections_total, request_latency_seconds, requests_total
from app.router import InputRejected

configure_logging()
logger = logging.getLogger("api")

if not settings.gemini_api_key:
    logger.warning(
        "GEMINI_API_KEY not set — llm_fallback route will return the safe "
        "fallback message instead of a real answer until it's configured."
    )
if not settings.api_key:
    logger.warning("API_KEY not set — every request to POST /query will be rejected.")

app = FastAPI(title="Semantic API Router")

# Rate limiting protects the whole service (embedding included), unlike
# gemini_client.TokenBucket which only throttles the outbound LLM call.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    rate_limit_rejections_total.inc()
    return _rate_limit_exceeded_handler(request, exc)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Tag the request (and every log line it produces) with an ID, reusing
    the caller's X-Request-ID when they sent one so traces join up."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guardrail: POST /query is closed by default — an unset API_KEY
    rejects every request rather than silently allowing them all."""
    expected = settings.api_key
    if not expected or not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.exception_handler(InputRejected)
async def _input_rejected_handler(request: Request, exc: InputRejected):
    logger.warning("input rejected: %s", exc)
    return JSONResponse(
        status_code=400, content={"error": "invalid input", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """Log the full detail, return none of it. A handler lookup miss
    (config/code mismatch) lands here as a 500 for that one request
    instead of taking the worker down."""
    logger.exception("unhandled error handling request")
    return JSONResponse(status_code=500, content={"error": "internal error"})


class QueryRequest(BaseModel):
    # Bounded here as well as in router.validate_input so oversized input
    # is rejected before any embedding work is spent on it.
    query: str = Field(..., min_length=1, max_length=settings.max_query_len)


class QueryResponse(BaseModel):
    route: str
    response: str
    score: float
    runner_up: str | None
    runner_up_score: float
    confident: bool
    reason: str


@app.post(
    "/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)]
)
@limiter.limit(settings.rate_limit)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    start = time.monotonic()
    # handle_query does CPU-bound local embedding work (and, on the
    # fallback path, a blocking network call) — run it off the event loop
    # so one slow request doesn't stall every other connection.
    result = await run_in_threadpool(handle_query, body.query)
    request_latency_seconds.labels(route=result.route).observe(time.monotonic() - start)
    requests_total.labels(route=result.route).inc()

    return QueryResponse(
        route=result.route,
        response=result.response,
        score=result.match.score,
        runner_up=result.match.runner_up,
        runner_up_score=result.match.runner_up_score,
        confident=result.match.confident,
        reason=result.match.reason,
    )


@app.get("/health")
async def health() -> dict:
    """Liveness: the process is up. Deliberately checks nothing else, so a
    dependency being unhappy never gets the container restarted."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict:
    """Readiness: the model and routes are loaded and we can serve. A
    missing GEMINI_API_KEY is reported but is not a failure — the four
    configured routes serve fine without it."""
    if router.model is None or not router.routes:
        raise HTTPException(status_code=503, detail="router not ready")
    return {
        "status": "ready",
        "routes_loaded": len(router.routes),
        "gemini_configured": bool(settings.gemini_api_key),
    }


@app.get("/metrics")
async def metrics() -> Response:
    # A plain route rather than a mounted sub-app, so scrapers pointed at
    # /metrics get the payload directly instead of a 307 to /metrics/.
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
