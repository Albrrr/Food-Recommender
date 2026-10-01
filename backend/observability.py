"""Structured logging, request correlation, and per-stage timing.

Every request emits exactly one JSON log line carrying a request id, the time
spent in each pipeline stage, and token usage. Timing the stages separately is
the point: when the service is slow, the log says *which* stage is slow rather
than just that the request took three seconds.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging(level: str) -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "foodify"):
    return structlog.get_logger(name)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id to every request, its logs, and its response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class StageTimer:
    """Collects millisecond timings for named pipeline stages."""

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self.marks: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.marks[f"{name}_ms"] = round((time.perf_counter() - started) * 1000, 1)

    def finish(self) -> dict[str, float]:
        return {
            **self.marks,
            "total_ms": round((time.perf_counter() - self._started) * 1000, 1),
        }
