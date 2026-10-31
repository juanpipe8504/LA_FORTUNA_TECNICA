"""Thread-safe in-memory ``JobRepositoryPort`` (tests / local dev without Redis)."""

from __future__ import annotations

import copy
import threading

from app.domain.exceptions import NotFoundError
from app.domain.models import IngestionJob, utcnow
from app.domain.ports import JobRepositoryPort


class InMemoryJobRepository(JobRepositoryPort):
    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = threading.Lock()

    def create(self, job: IngestionJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = copy.deepcopy(job)

    def get(self, job_id: str) -> IngestionJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFoundError(f"Job {job_id} not found")
            # Return a copy so callers cannot mutate stored state without update().
            return copy.deepcopy(job)

    def update(self, job: IngestionJob) -> None:
        with self._lock:
            if job.job_id not in self._jobs:
                raise NotFoundError(f"Job {job.job_id} not found")
            job.updated_at = utcnow()
            self._jobs[job.job_id] = copy.deepcopy(job)

    def list(self, *, limit: int = 100) -> list[IngestionJob]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [copy.deepcopy(j) for j in jobs[:limit]]
