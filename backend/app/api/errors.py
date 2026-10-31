"""Map exceptions to a uniform JSON error envelope.

Every error response has the shape
``{"error": {"code": str, "message": str, "request_id": str | None}}``.
Unexpected exceptions become a generic 500: internal details are logged,
never returned to the client.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from app.domain.exceptions import (
    CircuitOpenError,
    DocumentParsingError,
    DomainError,
    ExternalServiceError,
    NotFoundError,
    ValidationError,
)

logger = get_logger(__name__)

# Literal codes: Starlette renamed the 422/413 constants across versions.
HTTP_422_UNPROCESSABLE = 422
HTTP_413_TOO_LARGE = 413

INTERNAL_ERROR_MESSAGE = "An unexpected error occurred. Please try again later."


class ApiError(Exception):
    """HTTP-level error raised by routes (e.g. 413 payload too large)."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# Ordered from most to least specific: the first matching class wins.
_DOMAIN_ERROR_MAP: tuple[tuple[type[DomainError], int, str, str | None], ...] = (
    (NotFoundError, status.HTTP_404_NOT_FOUND, "not_found", None),
    (ValidationError, HTTP_422_UNPROCESSABLE, "validation_error", None),
    (DocumentParsingError, HTTP_422_UNPROCESSABLE, "document_parsing_error", None),
    (
        CircuitOpenError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "service_unavailable",
        "A required service is temporarily unavailable. Please retry shortly.",
    ),
    (
        ExternalServiceError,
        status.HTTP_502_BAD_GATEWAY,
        "upstream_error",
        "An upstream service failed to process the request.",
    ),
)


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_of(request)}
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status_code, content={"error": body}, headers=headers)


def internal_error_response(request: Request) -> JSONResponse:
    return error_response(
        request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", INTERNAL_ERROR_MESSAGE
    )


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    for exc_type, status_code, code, public_message in _DOMAIN_ERROR_MAP:
        if isinstance(exc, exc_type):
            # Upstream failures may carry provider internals: log them, return a generic text.
            if public_message is not None:
                logger.warning("upstream failure: %s", exc, extra={"error_type": type(exc).__name__})
            return error_response(request, status_code, code, public_message or str(exc))
    logger.warning("unmapped domain error", extra={"error_type": type(exc).__name__})
    return error_response(request, status.HTTP_400_BAD_REQUEST, "bad_request", str(exc))


async def _api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(request, exc.status_code, exc.code, exc.message)


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = "not_found" if exc.status_code == status.HTTP_404_NOT_FOUND else "http_error"
    return error_response(
        request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None)
    )


async def _request_validation_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {"loc": list(err.get("loc", ())), "msg": err.get("msg", ""), "type": err.get("type", "")}
        for err in exc.errors()
    ]
    return error_response(
        request,
        HTTP_422_UNPROCESSABLE,
        "validation_error",
        "Request validation failed.",
        details=details,
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception", exc_info=exc)
    return internal_error_response(request)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _domain_error_handler)
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
