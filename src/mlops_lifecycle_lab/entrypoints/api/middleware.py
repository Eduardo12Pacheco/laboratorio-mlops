"""Request-id middleware with an access log line and latency (S18, D21)."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

LOGGER = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate/echo `X-Request-ID` and bind it to the structlog context."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=round(elapsed * 1000.0, 3),
            )
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        return response


__all__ = ["RequestIDMiddleware"]
