"""Document catalogue use cases."""

from __future__ import annotations

from app.domain.models import IngestionJob, JobStatus
from app.domain.ports import JobRepositoryPort


class ListDocumentsUseCase:
    """Lists successfully ingested documents (one per COMPLETED job), newest first."""

    def __init__(self, job_repository: JobRepositoryPort, limit: int = 1000) -> None:
        self._jobs = job_repository
        self._limit = limit

    def execute(self) -> list[IngestionJob]:
        completed = [j for j in self._jobs.list(limit=self._limit) if j.status == JobStatus.COMPLETED]
        return sorted(completed, key=lambda j: j.created_at, reverse=True)
