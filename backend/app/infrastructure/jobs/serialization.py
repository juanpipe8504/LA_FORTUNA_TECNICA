"""(De)serialization of ``IngestionJob`` shared by the job repositories."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from app.domain.models import IngestionJob, JobStatus


def job_to_dict(job: IngestionJob) -> dict[str, Any]:
    data = asdict(job)
    data["status"] = job.status.value
    data["created_at"] = job.created_at.isoformat()
    data["updated_at"] = job.updated_at.isoformat()
    return data


def job_from_dict(data: dict[str, Any]) -> IngestionJob:
    payload = dict(data)
    payload["status"] = JobStatus(payload["status"])
    payload["created_at"] = datetime.fromisoformat(payload["created_at"])
    payload["updated_at"] = datetime.fromisoformat(payload["updated_at"])
    known = IngestionJob.__dataclass_fields__.keys()
    return IngestionJob(**{k: v for k, v in payload.items() if k in known})


def job_to_json(job: IngestionJob) -> str:
    return json.dumps(job_to_dict(job), ensure_ascii=False)


def job_from_json(raw: str | bytes) -> IngestionJob:
    return job_from_dict(json.loads(raw))
