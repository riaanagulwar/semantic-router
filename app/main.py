"""
CLI entry point. Reads a query interactively, classifies it, dispatches to
the matching handler, and logs the routing decision (route, similarity
score, margin, confidence) as structured JSON — useful both for debugging
thresholds and as evidence of how the router behaves when you write this
up.

For a long-running service, use api.py (`uvicorn api:app`) instead — this
file stays a local debugging/demo REPL and is not what gets deployed.

Run interactively:
    python main.py

Run the eval set instead:
    python test_queries.py
"""

import logging

from app.config.logging_config import configure_logging
from app.config.request_context import set_request_id
from app.core import handle_query
from app.router import InputRejected

configure_logging()
set_request_id("cli")
logger = logging.getLogger("main")


def handle(query: str) -> str:
    try:
        result = handle_query(query)
    except InputRejected as e:
        logger.warning("input rejected: %s", e)
        return f"Sorry, I can't process that: {e}"
    return result.response


if __name__ == "__main__":
    print("Semantic router demo. Type a query ('quit' to exit).\n")
    while True:
        q = input("> ")
        if q.strip().lower() in {"quit", "exit"}:
            break
        print(handle(q), "\n")
