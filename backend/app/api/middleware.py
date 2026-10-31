"""Request correlation id + structured access logging."""

from __future__ import annotations

import re
import time
import uuid

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.api.errors import internal_error_response
from app.core.logging import bind_context, get_logger

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")

logger = get_logger("app.api.access")


def resolve_request_id(incoming: str | None) -> str:
    """Reuse a well-formed client/proxy id, otherwise mint a new one (avoids log injection)."""
    if incoming and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds ``request_id`` to the logging context and logs one line per request.

    Also acts as the last line of defence: any exception escaping the routers
    becomes a generic 500 envelope that still carries the request id.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        start = time.perf_counter()
        with bind_context(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("unhandled exception while processing request")
                response = internal_error_response(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            response.headers[REQUEST_ID_HEADER] = request_id
            log = logger.warning if response.status_code >= 500 else logger.info
            log(
                "%s %s -> %s",
                request.method,
                request.url.path,
                response.status_code,
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
        return response


def register_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)
