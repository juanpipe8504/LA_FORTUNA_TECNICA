"""Translate ``openai`` SDK exceptions into domain exceptions (shared by LLM + embeddings)."""

from __future__ import annotations

import openai

from app.domain.exceptions import ExternalServiceError
from app.infrastructure.resilience.retry import parse_retry_after, transient_error

_TRANSIENT_STATUS = {408, 409, 425, 429}


def _retry_after(exc: openai.APIError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_ms = headers.get("retry-after-ms")
    if retry_ms:
        try:
            return max(0.0, float(retry_ms) / 1000.0)
        except ValueError:
            pass
    return parse_retry_after(headers.get("retry-after"))


def map_openai_error(exc: Exception, operation: str) -> ExternalServiceError:
    """Return the domain error to raise (caller does ``raise ... from exc``)."""
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return transient_error(f"OpenAI {operation}: connection/timeout error: {exc}")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status in _TRANSIENT_STATUS or status >= 500:
            return transient_error(f"OpenAI {operation}: HTTP {status}: {exc.message}", _retry_after(exc))
        return ExternalServiceError(f"OpenAI {operation}: HTTP {status}: {exc.message}")
    return ExternalServiceError(f"OpenAI {operation} failed: {exc}")
