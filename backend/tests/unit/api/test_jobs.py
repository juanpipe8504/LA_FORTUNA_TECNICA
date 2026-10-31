from __future__ import annotations

from app.domain.exceptions import NotFoundError
from app.domain.models import JobStatus
from tests.unit.api.conftest import API, make_job


def test_get_job_returns_full_status(client, container):
    container.job_repository.get.return_value = make_job(
        status=JobStatus.PROCESSING, progress=40, message="Embedding chunks", page_count=5
    )

    response = client.get(f"{API}/jobs/job-1")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-1",
        "document_id": "doc-1",
        "filename": "manual.pdf",
        "status": "processing",
        "progress": 40,
        "message": "Embedding chunks",
        "error": None,
        "page_count": 5,
        "chunk_count": None,
        "image_count": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    container.job_repository.get.assert_called_once_with("job-1")


def test_failed_job_exposes_error(client, container):
    container.job_repository.get.return_value = make_job(status=JobStatus.FAILED, error="Corrupt PDF")

    body = client.get(f"{API}/jobs/job-1").json()

    assert body["status"] == "failed"
    assert body["error"] == "Corrupt PDF"


def test_unknown_job_is_404_with_request_id(client, container):
    container.job_repository.get.side_effect = NotFoundError("Job 'nope' not found")

    response = client.get(f"{API}/jobs/nope", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "not_found", "message": "Job 'nope' not found", "request_id": "req-123"}
    }
