"""
Per-request request ID, threaded through contextvars so any log call made
anywhere during a request (in router.py, handlers.py, core.py, ...) can be
tagged with it via RequestIdFilter without passing it through every
function signature. api.py's middleware sets it for the duration of one
request; main.py's CLI sets a fixed "cli" value since there's no request
boundary there.
"""

from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()
