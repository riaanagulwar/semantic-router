"""
Structured JSON logging setup. Call configure_logging() once, at process
startup (main.py's CLI entry point and api.py both do), instead of the old
logging.basicConfig(...) text formatter.

Routing fields (route, score, runner_up, runner_up_score, confident,
reason) are passed via logger.info(msg, extra={...}) at the call site and
come out as top-level JSON keys, so they're queryable in a log aggregator
instead of being buried in a formatted string. The per-request request_id
is attached automatically by RequestIdFilter, so it doesn't have to be
threaded through every function signature in router.py / handlers.py.
"""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter

from app.config.request_context import get_request_id
from app.config.settings import settings


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s")
    )
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)
