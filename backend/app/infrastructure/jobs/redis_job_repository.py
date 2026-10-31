"""Redis-backed ``JobRepositoryPort``.

Each job is a JSON string under ``ingestion:job:{job_id}`` (with a TTL) and its
id is indexed in the sorted set ``ingestion:jobs`` scored by ``created_at`` so
listing newest-first is a single ``ZREVRANGE``. Entries whose key expired are
lazily pruned from the index while listing.
"""

from __future__ import annotations

import redis

from app.core.logging import get_logger
from app.domain.exceptions import NotFoundError
from app.domain.models import IngestionJob, utcnow
from app.domain.ports import JobRepositoryPort
from app.infrastructure.jobs.serialization import job_from_json, job_to_json

logger = get_logger(__name__)

_KEY_PREFIX = "ingestion:job:"
_INDEX_KEY = "ingestion:jobs"


class RedisJobRepository(JobRepositoryPort):
    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 7 * 24 * 3600) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    @staticmethod
    def _key(job_id: str) -> str:
        return f"{_KEY_PREFIX}{job_id}"

    def create(self, job: IngestionJob) -> None:
        pipe = self._redis.pipeline()
        pipe.set(self._key(job.job_id), job_to_json(job), ex=self._ttl)
        pipe.zadd(_INDEX_KEY, {job.job_id: job.created_at.timestamp()})
        pipe.execute()

    def get(self, job_id: str) -> IngestionJob:
        raw = self._redis.get(self._key(job_id))
        if raw is None:
            raise NotFoundError(f"Job {job_id} not found")
        return job_from_json(raw)

    def update(self, job: IngestionJob) -> None:
        job.updated_at = utcnow()
        # XX: only update an existing job, never resurrect an expired/unknown one.
        updated = self._redis.set(self._key(job.job_id), job_to_json(job), ex=self._ttl, xx=True)
        if not updated:
            raise NotFoundError(f"Job {job.job_id} not found")

    def list(self, *, limit: int = 100) -> list[IngestionJob]:
        if limit <= 0:
            return []
        ids = [self._decode(i) for i in self._redis.zrevrange(_INDEX_KEY, 0, limit - 1)]
        if not ids:
            return []
        raws = self._redis.mget([self._key(i) for i in ids])
        jobs: list[IngestionJob] = []
        stale: list[str] = []
        for job_id, raw in zip(ids, raws, strict=True):
            if raw is None:
                stale.append(job_id)
                continue
            jobs.append(job_from_json(raw))
        if stale:
            self._redis.zrem(_INDEX_KEY, *stale)
            logger.debug("pruned_stale_job_index_entries", extra={"count": len(stale)})
        return jobs

    @staticmethod
    def _decode(value: bytes | str) -> str:
        return value.decode() if isinstance(value, bytes) else value
