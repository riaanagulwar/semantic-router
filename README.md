# Semantic API Router

Routes a free-text query to a handler using local embedding similarity — no
LLM call in the hot path for the common case. Gemini is only ever
consulted as a second-opinion CLASSIFIER for queries the embedding router
isn't confident about; it never generates the user-facing answer itself.

## Why

Most user queries map to a small set of known intents ("what's my balance",
"refund this"). Calling an LLM to classify every request is wasted spend —
this routes those by cosine similarity against a handful of example phrases
per route, and reserves the LLM for the genuinely ambiguous tail.

Gemini is deliberately *not* used to generate free-text answers for
queries outside the known intents. An ungrounded LLM with no context about
this app could confidently make up an answer to something like "how is
interest calculated" — in a banking context, a wrong answer stated
confidently is worse than an honest "I can't help with that." So instead,
Gemini gets exactly one job: try to fit the query into one of the app's
own known routes. If it can, the query is dispatched to that route's real
handler, same as a confident embedding match. If it can't, the user gets a
fixed, non-generated message — never Gemini's raw prose.

## How it works

```
query --> validate (length/empty/control chars)
      --> embed locally (sentence-transformers, no API call)
      --> compare against each route's example utterances (cosine similarity)
      --> best match above threshold AND clearly ahead of runner-up?
             yes --> dispatch to that route's handler
             no  --> ask Gemini to classify into one of the known routes
                        matched a known route? --> dispatch there too
                        no match / any failure? --> fixed no-match message
```

`app/core.py` holds this pipeline (`handle_query`) once, shared by both
entry points below. Gemini's reply is never trusted directly as a route
name — it's only accepted if it's an exact match to one of the routes
loaded from `app/config/routes.yaml`, the same fixed set the embedding
router itself is constrained to. This is the same "fixed route set"
guardrail applied twice: neither the embedding match nor Gemini's guess
can ever cause a route/handler to run that a developer didn't explicitly
register.

## Project layout

```
app/                  # everything importable
├── main.py           # CLI entry point (python -m app.main)
├── api.py            # HTTP entry point (uvicorn app.api:app)
├── core.py           # shared validate -> classify -> dispatch pipeline
├── router.py          # embedding-based classifier + input validation
├── handlers.py        # per-route handlers (mocked) + the static no-match handler
├── gemini_client.py   # rate-limited, retried, fail-safe Gemini intent classifier
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
| LLM output never trusted directly | `app.gemini_client.GeminiFallback.classify_intent` | Gemini's reply is only accepted if it exactly matches one of the routes it was given — it can't invent a route or return anything else |
| Rate-limited LLM classification | `app.gemini_client.TokenBucket` | Caps Gemini calls per minute so quota is never exceeded |
| Capped retries + timeout | `app.gemini_client.GeminiFallback.classify_intent` | At most 2 attempts, each with a 10s timeout (Google's API minimum) — never hangs, never loops forever |
| Fail-safe on error | `app.gemini_client.GeminiFallback.classify_intent` | Quota/network/timeout failures, a missing `GEMINI_API_KEY`, or an unrecognized reply all resolve to "no match," never a crash and never Gemini's raw text |
| API auth | `app.api.require_api_key` | `POST /query` requires a matching `X-API-Key` header |
| API-level rate limiting | `app/api.py` (`slowapi`) | Caps requests per client to the whole service, not just the Gemini leg |
| No internals leaked on error | `app/api.py` exception handlers | Unexpected errors are logged in full and return a generic 500 to the client |

## Design decisions

The reasoning behind the architecture, written down once so it doesn't
need reconstructing from scratch later.

**Why local embeddings before any LLM call.** Most real user queries map
to a small, fixed set of intents. Calling an LLM to classify every single
request is unnecessary cost and latency when a cheap, local, deterministic
comparison (cosine similarity against a handful of example phrases) gets
the common case right instantly and for free. The LLM is reserved for the
tail of queries that don't confidently match anything known.

**Why Gemini classifies rather than generates a free-text answer.** An
earlier version of this project let Gemini answer ambiguous queries
directly. The problem: with no context about what this app actually is,
an LLM can confidently produce a wrong answer to something that sounds
like it should have a factual one ("how is interest calculated"). In a
banking context, a wrong answer stated confidently is worse than an
honest "I can't help with that." So Gemini's role was narrowed to one
much smaller, more checkable task — decide whether the query fits one of
the app's own known intents — rather than open-ended generation.

**Why Gemini's answer is never trusted directly.** Even narrowed to
classification, an LLM's reply is still just text, and text can be wrong,
malformed, or (in principle) manipulated by adversarial input. Its reply
is only ever accepted if it's an exact match to a route name from
`app/config/routes.yaml` — the same fixed set the embedding router itself
is constrained to. Anything else (a made-up category, an explanation
instead of a label, an empty reply) is treated identically to "no match."
This is the project's one guardrail — never let untrusted output pick
what code runs — applied to both classification paths, not just one.

**Why a static message on total no-match, not free generation or human
handoff.** Once both the embedding router and Gemini's classification
attempt fail, there are three honest options: generate a free-text answer
anyway (rejected — see above), hand the query to a real support queue for
a person to answer, or return a fixed, non-generated message. Human
handoff is the pattern real production support bots use for exactly this
case, and it's arguably the safest of the three — but it needs a real
destination (a queue, a ticketing system, staffing) that doesn't exist for
a local project like this one. Building that infrastructure here would be
complexity for its own sake with no one on the other end of it. The static
message is the honest choice given that constraint: it never pretends to
know something it doesn't, and it costs nothing to maintain.

**Why the routing set is a plain dict, not something dynamic.** `HANDLERS`
in `app/handlers.py` only ever contains functions a developer explicitly
wrote and registered. Nothing in the pipeline — not the embedding match,
not Gemini's classification — ever does `getattr()`/`eval()` on a route
name or otherwise turns model output into arbitrary code execution. Every
route that can run is one that was reviewed and committed.

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
| `/query` | POST | `X-API-Key` header | `{"query": str}` → `{route, response, score, runner_up, runner_up_score, confident, reason, reclassified_by_llm}` |
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
  Gemini's classification attempt (safer, more calls); lower it to trust
  the embedding router more (fewer calls, more risk of a wrong route).
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
- `GEMINI_API_KEY` — required only for the classification attempt on
  low-confidence queries; if unset, those queries skip straight to the
  static no-match message instead of crashing (checked lazily, on first
  use — importing the app never requires it).
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
