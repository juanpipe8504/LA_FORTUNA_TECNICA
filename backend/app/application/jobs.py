"""Job status read model with a stale-job watchdog.

A job is PROCESSING only while a worker keeps reporting progress (every
progress update refreshes ``updated_at``, including one heartbeat per image
described). If the worker is lost – process killed, OOM, host asleep, hard
time limit – nothing would ever mark the job FAILED and clients would poll a
job that looks alive forever. ``GetJobUseCase`` therefore *reports* a
PROCESSING job whose last update is older than ``stale_after_seconds`` as
FAILED.

The verdict is derived on read and NOT persisted, on purpose:

- ``GET /jobs/{id}`` stays side-effect free (no write races with a worker).
- If the worker was merely slow, its next progress update makes ``updated_at``
  fresh again and the job is reported as PROCESSING once more – stored state
  always reflects what the worker actually did.
- If the broker re-delivers the task (worker genuinely lost), the stored
  status is still PROCESSING, so ``ProcessDocumentUseCase`` restarts it from
  scratch (logged as ``job_redelivered_restarting``). Jobs that really ended
  FAILED (error / time limit) are terminal and never re-run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

from app.core.logging import get_logger
from app.domain.models import IngestionJob, JobStatus, utcnow
from app.domain.ports import JobRepositoryPort

logger = get_logger(__name__)

STALE_JOB_ERROR = (
    "El procesamiento se interrumpió: el worker dejó de informar progreso "
    "(reinicio, falta de memoria o suspensión del equipo). Vuelve a subir el documento."
)


def is_stale(job: IngestionJob, now: datetime, stale_after: timedelta) -> bool:
    return job.status == JobStatus.PROCESSING and now - job.updated_at > stale_after


def apply_stale_watchdog(job: IngestionJob, now: datetime, stale_after: timedelta) -> IngestionJob:
    """Return ``job`` – or a FAILED copy of it when it is a stale PROCESSING job."""
    if not is_stale(job, now, stale_after):
        return job
    logger.warning(
        "job_reported_stale",
        extra={
            "job_id": job.job_id,
            "progress": job.progress,
            "last_update_age_s": round((now - job.updated_at).total_seconds()),
        },
    )
    return replace(job, status=JobStatus.FAILED, error=STALE_JOB_ERROR, message="Failed")


class GetJobUseCase:
    """Reads a job for status polling, applying the stale-job watchdog."""

    def __init__(
        self,
        job_repository: JobRepositoryPort,
        stale_after_seconds: float = 900,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")
        self._jobs = job_repository
        self._stale_after = timedelta(seconds=stale_after_seconds)
        self._clock = clock

    def execute(self, job_id: str) -> IngestionJob:
        """Raises ``NotFoundError`` for unknown jobs (mapped to 404 by the API)."""
        return apply_stale_watchdog(self._jobs.get(job_id), self._clock(), self._stale_after)
