"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.dependencies import HealthChecksDep
from app.api.schemas import HealthResponse, ReadinessResponse
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe (Redis + Qdrant)",
    responses={503: {"model": ReadinessResponse}},
)
def ready(checks: HealthChecksDep, response: Response) -> ReadinessResponse:
    results: dict[str, bool] = {}
    for name, check in checks.items():
        try:
            results[name] = bool(check())
        except Exception as exc:  # noqa: BLE001 - a failing probe must not raise
            logger.warning("readiness check failed", extra={"check": name, "error": str(exc)})
            results[name] = False
    is_ready = all(results.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if is_ready else "unavailable", checks=results)
