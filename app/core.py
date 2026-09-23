"""
Shared query-handling logic used by both main.py (CLI) and api.py (HTTP
service), so the two entry points don't duplicate the validate -> classify
-> dispatch pipeline.

Guardrail: dispatch is a plain dict lookup against HANDLERS (see
handlers.py) — never getattr()/eval() on a route name or on model output.
"""

import logging
from dataclasses import dataclass

from app.handlers import HANDLERS
from app.router import RouteMatch, SemanticRouter, validate_input

logger = logging.getLogger("core")

router = SemanticRouter()


@dataclass
class HandleResult:
    response: str
    route: str
    match: RouteMatch


def handle_query(query: str) -> HandleResult:
    """Raises InputRejected if the query fails validation. Raises KeyError
    if the classified route has no registered handler (config/code
    mismatch) — that should surface loudly rather than silently fall
    through; callers decide how to translate it (main.py lets it crash the
    REPL turn, api.py turns it into a logged 500)."""
    clean = validate_input(query)

    match = router.classify(clean)
    route = match.route if match.confident else "llm_fallback"

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
        },
    )

    handler = HANDLERS[route]
    response = handler(clean)
    return HandleResult(response=response, route=route, match=match)
