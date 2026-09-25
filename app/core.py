"""
Shared query-handling logic used by both main.py (CLI) and api.py (HTTP
service), so the two entry points don't duplicate the validate -> classify
-> reclassify -> dispatch pipeline.

Guardrail: dispatch is always a plain dict lookup against HANDLERS (see
handlers.py) — never getattr()/eval() on a route name or on model output.
This applies just as much to Gemini's classification attempt below as it
does to the embedding router: Gemini only ever gets to pick from
router.routes (the same fixed set loaded from config/routes.yaml), and its
reply is checked for exact membership in that set before it's ever used as
a dict key. It can't invent a route, and it can't cause anything to run
beyond the handlers a developer already registered.
"""

import logging
from dataclasses import dataclass

from app.gemini_client import get_fallback
from app.handlers import HANDLERS
from app.metrics import (
    llm_classification_attempts_total,
    llm_no_match_total,
    llm_reclassified_total,
)
from app.router import RouteMatch, SemanticRouter, validate_input

logger = logging.getLogger("core")

router = SemanticRouter()


@dataclass
class HandleResult:
    response: str
    route: str
    match: RouteMatch
    reclassified_by_llm: bool


def _try_llm_reclassify(query: str) -> str | None:
    """Second opinion for a query the embedding router wasn't confident
    about. Returns a route name from router.routes, or None if Gemini
    couldn't confidently pick one (declined, rate-limited, missing key, or
    an error) — callers treat None as "no route matched", never as a
    reason to crash."""
    llm_classification_attempts_total.inc()
    try:
        route = get_fallback().classify_intent(query, list(router.routes.keys()))
    except RuntimeError:
        logger.warning("llm reclassification unavailable: GEMINI_API_KEY not configured")
        route = None

    if route:
        llm_reclassified_total.labels(route=route).inc()
    else:
        llm_no_match_total.inc()
    return route


def handle_query(query: str) -> HandleResult:
    """Raises InputRejected if the query fails validation. Raises KeyError
    if the final route has no registered handler (config/code mismatch)
    — that should surface loudly rather than silently fall through;
    callers decide how to translate it (main.py lets it crash the REPL
    turn, api.py turns it into a logged 500)."""
    clean = validate_input(query)

    match = router.classify(clean)
    reclassified_by_llm = False

    if match.confident:
        route = match.route
    else:
        llm_route = _try_llm_reclassify(clean)
        if llm_route:
            route = llm_route
            reclassified_by_llm = True
        else:
            route = "no_match"

    # The query itself is logged because tuning SIMILARITY_THRESHOLD /
    # AMBIGUITY_MARGIN means correlating scores with what was actually
    # asked. If this ever handled real customer data, redact or hash it
    # here — this is the one place raw user input reaches the logs.
    logger.info(
        "query routed",
        extra={
            "query": clean,
            "route": route,
            "score": match.score,
            "runner_up": match.runner_up,
            "runner_up_score": match.runner_up_score,
            "confident": match.confident,
            "reason": match.reason,
            "reclassified_by_llm": reclassified_by_llm,
        },
    )

    handler = HANDLERS[route]
    response = handler(clean)
    return HandleResult(
        response=response, route=route, match=match, reclassified_by_llm=reclassified_by_llm
    )
