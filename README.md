# Semantic API Router

Routes a free-text query to a handler using local embedding similarity — no
LLM call in the hot path. Only queries the router isn't confident about fall
through to Gemini.

## Why

Most user queries map to a small set of known intents ("what's my balance",
"refund this"). Calling an LLM to classify every request is wasted spend —
this routes those by cosine similarity against a handful of example phrases
per route, and reserves the LLM for the genuinely ambiguous tail.

## How it works

```
query --> validate (length/empty/control chars)
      --> embed locally (sentence-transformers, no API call)
      --> compare against each route's example utterances (cosine similarity)
      --> best match above threshold AND clearly ahead of runner-up?
             yes --> dispatch to that route's handler
             no  --> dispatch to llm_fallback (rate-limited Gemini call)
```

`app/core.py` holds this pipeline (`handle_query`) once, shared by both
entry points below.

## Project layout

```
app/                  # everything importable
├── main.py           # CLI entry point (python -m app.main)
├── api.py            # HTTP entry point (uvicorn app.api:app)
├── core.py           # shared validate -> classify -> dispatch pipeline
├── router.py          # embedding-based classifier + input validation
├── handlers.py        # per-route handlers (mocked, + the LLM fallback)
├── gemini_client.py   # rate-limited, retried, fail-safe Gemini wrapper
├── metrics.py          # Prometheus counters/histogram
└── config/
    ├── settings.py         # all env-driven configuration
    ├── logging_config.py   # structured JSON logging setup
    ├── request_context.py  # per-request ID (contextvars)
    └── routes.yaml         # intents + their example utterances

scripts/eval_routes.py  # manual tuning tool — routing decisions only
tests/                  # pytest suite
```

## Guardrails

| Guardrail | Where | Why |
|---|---|---|
| Input validation | `app.router.validate_input` | Rejects empty/oversized/control-char input before it reaches embedding or a handler |
| Similarity threshold | `app.router.SemanticRouter.classify` | A weak best-match isn't trusted — falls back instead of guessing |
| Ambiguity margin | `app.router.SemanticRouter.classify` | If the top two routes are nearly tied, treated as ambiguous rather than picking one |
| Fixed handler dispatch | `app.handlers.HANDLERS` / `app.core.handle_query` | Routes are dispatched via a plain dict lookup — never `getattr`/`eval` on a route name |
| Rate-limited LLM fallback | `app.gemini_client.TokenBucket` | Caps Gemini calls per minute so quota is never exceeded |
| Capped retries + timeout | `app.gemini_client.GeminiFallback.answer` | At most 2 attempts, each with a 10s timeout (Google's API minimum) — never hangs, never loops forever |
| Fail-safe on error | `app.gemini_client.GeminiFallback.answer` | Quota/network/timeout failures, or a missing `GEMINI_API_KEY`, return a safe canned message, never a crash |
| API auth | `app.api.require_api_key` | `POST /query` requires a matching `X-API-Key` header |
| API-level rate limiting | `app/api.py` (`slowapi`) | Caps requests per client to the whole service, not just the Gemini leg |
| No internals leaked on error | `app/api.py` exception handlers | Unexpected errors are logged in full and return a generic 500 to the client |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY and API_KEY
```

For running tests locally, also install `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

## Run

All commands run from the repo root, as modules, so `app`'s internal
imports resolve correctly. Two entry points share the same
routing/dispatch logic (`app/core.py`):

```bash
python -m app.main             # interactive CLI — local debugging/demo only
python -m scripts.eval_routes  # eval set — routing decisions only, no Gemini calls

uvicorn app.api:app --reload   # HTTP service — this is what gets deployed
# or, for a prod-like local run:
docker compose up --build
```

### HTTP API

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/query` | POST | `X-API-Key` header | `{"query": str}` → `{route, response, score, runner_up, runner_up_score, confident, reason}` |
| `/health` | GET | none | Liveness — 200 if the process is up |
| `/ready` | GET | none | Readiness — 200 once the router/model/config are loaded, else 503 |
| `/metrics` | GET | none | Prometheus exposition format |

Every response carries an `X-Request-ID` header (echoed from the request if
provided, generated otherwise), which is also attached to every log line
for that request.

## Configuration

All tuning knobs and secrets are environment variables, read via
`app/config/settings.py` (see `.env.example` for the full list with
defaults). Nothing below requires a code change to adjust:

- `SIMILARITY_THRESHOLD` (default 0.55) — raise it to send more traffic to
  the LLM fallback (safer, more calls); lower it to trust the router more
  (fewer calls, more risk of a wrong route).
- `AMBIGUITY_MARGIN` (default 0.05) — raise it to be stricter about ties
  between two close routes.
- `MAX_QUERY_LEN`, `MAX_RPM`, `MAX_RETRIES`, `TIMEOUT_S` — see
  `gemini_client.py` / `router.py` docstrings for what each guards.
  `TIMEOUT_S` has a 10s floor enforced in code — Google's API rejects
  anything lower outright.
- `EMBEDDING_MODEL_NAME` (default `all-MiniLM-L6-v2`) — the local model
  used for matching. The Dockerfile takes the same value as a build arg so
  the model baked into the image can't drift from the one loaded at
  runtime.
- `RATE_LIMIT` (default `30/minute`) — per-client cap on `POST /query`.
- `GEMINI_API_KEY` — required only for the `llm_fallback` route to return
  real answers; if unset, that route returns a safe canned message instead
  of crashing (checked lazily, on first use — importing the app never
  requires it).
- `API_KEY` — required for `POST /query` to accept any request at all;
  every request is rejected with 401 until this is set.

Run `python -m scripts.eval_routes` after any threshold/margin change — it
prints score and margin per query so you can see exactly what moved.

## Extending

Add a route by adding a block to `app/config/routes.yaml` (name, handler,
5-10 example utterances) and a matching function in `app/handlers.py`. No
other code changes needed — embeddings are (re)built from the YAML at
startup. `tests/test_handlers.py` will fail the build if a route is added
to the YAML without a matching handler registered.

## Testing & CI

```bash
pytest tests/            # unit + integration tests (no GEMINI_API_KEY needed)
pip-audit -r requirements.txt
```

`.github/workflows/ci.yml` runs both on every push/PR, plus a Docker build.
`tests/test_router_classify.py` loads the real embedding model (slower);
`tests/test_router_threshold_logic.py` covers the same threshold/margin
arithmetic with a stubbed model for a fast unit-level check.

Known gap: `pip-audit` currently flags two advisories in transitive build
dependencies pulled in by `sentence-transformers` (`torch`, `setuptools`)
with fix versions not yet published on PyPI at time of writing — not
directly pinnable from this project's `requirements.txt`. Re-run
`pip-audit` periodically and bump `sentence-transformers`/`torch` once a
fixed release is available.

## Deployment

`Dockerfile` bakes the sentence-transformers model into the image at build
time (`RUN python -c "... SentenceTransformer(...)"`), so a fresh container
never hits the network for the ~80MB model download on cold start — only
`docker build` does. Runs as a non-root user.

```bash
docker build -t semantic-router .
docker run --rm -p 8000:8000 --env-file .env semantic-router
# or
docker compose up --build
```

No Kubernetes manifests are included — this project's scale doesn't need
orchestration, but the image containerizes cleanly onto Kubernetes, ECS, or
Cloud Run later if that changes.
