from __future__ import annotations

import threading
from datetime import timedelta
from unittest.mock import MagicMock, patch

import fakeredis
import pytest

from app.domain.exceptions import ExternalServiceError, NotFoundError, ValidationError
from app.domain.models import BoundingBox, ImageAsset, IngestionJob, JobStatus
from app.infrastructure.jobs.in_memory_job_repository import InMemoryJobRepository
from app.infrastructure.jobs.redis_job_repository import RedisJobRepository
from app.infrastructure.jobs.serialization import job_from_json, job_to_json
from app.infrastructure.queue.celery_app import celery_app
from app.infrastructure.queue.celery_task_queue import CeleryTaskQueue
from app.infrastructure.storage.local_blob_storage import LocalBlobStorage


def make_job(job_id: str = "job-1", minutes: int = 0, **kwargs) -> IngestionJob:
    job = IngestionJob(job_id, f"doc-{job_id}", "manual.pdf", "/data/uploads/x.pdf", **kwargs)
    job.created_at = job.created_at + timedelta(minutes=minutes)
    return job


# --------------------------------------------------------------------------- #
# Job serialization + repositories
# --------------------------------------------------------------------------- #
def test_job_json_roundtrip_preserves_types():
    job = make_job(status=JobStatus.FAILED, progress=40, error="boom", page_count=3)
    restored = job_from_json(job_to_json(job))
    assert restored == job
    assert restored.created_at.tzinfo is not None


@pytest.fixture(params=["redis", "memory"])
def repository(request):
    if request.param == "redis":
        return RedisJobRepository(fakeredis.FakeRedis(), ttl_seconds=60)
    return InMemoryJobRepository()


def test_repository_create_get_update(repository):
    job = make_job()
    repository.create(job)
    loaded = repository.get("job-1")
    assert loaded == job

    loaded.status, loaded.progress = JobStatus.PROCESSING, 50
    before = loaded.updated_at
    repository.update(loaded)
    again = repository.get("job-1")
    assert (again.status, again.progress) == (JobStatus.PROCESSING, 50)
    assert again.updated_at >= before


def test_repository_missing_job(repository):
    with pytest.raises(NotFoundError):
        repository.get("nope")
    with pytest.raises(NotFoundError):
        repository.update(make_job("nope"))


def test_repository_lists_newest_first_with_limit(repository):
    for i in range(5):
        repository.create(make_job(f"job-{i}", minutes=i))
    assert [j.job_id for j in repository.list(limit=3)] == ["job-4", "job-3", "job-2"]
    assert repository.list(limit=0) == []


def test_redis_repository_sets_ttl_and_prunes_expired_index_entries():
    client = fakeredis.FakeRedis()
    repo = RedisJobRepository(client, ttl_seconds=120)
    repo.create(make_job("a"))
    repo.create(make_job("b", minutes=1))
    assert 0 < client.ttl("ingestion:job:a") <= 120

    client.delete("ingestion:job:b")  # simulate TTL expiry
    assert [j.job_id for j in repo.list()] == ["a"]
    assert client.zcard("ingestion:jobs") == 1


def test_in_memory_repository_returns_copies_and_is_thread_safe():
    repo = InMemoryJobRepository()
    repo.create(make_job())
    repo.get("job-1").progress = 99
    assert repo.get("job-1").progress == 0

    def worker(n: int) -> None:
        for i in range(50):
            repo.create(make_job(f"t{n}-{i}"))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(repo.list(limit=1000)) == 201


# --------------------------------------------------------------------------- #
# Blob storage
# --------------------------------------------------------------------------- #
@pytest.fixture
def storage(tmp_path) -> LocalBlobStorage:
    return LocalBlobStorage(str(tmp_path / "uploads"), str(tmp_path / "images"))


def test_blob_storage_roundtrip(storage, tmp_path):
    path = storage.save_upload("doc-1", "../../evil name.pdf", b"%PDF-1.7")
    assert path.endswith("doc-1.pdf")
    assert (tmp_path / "uploads" / "doc-1.pdf").read_bytes() == b"%PDF-1.7"

    asset = ImageAsset("doc-1_p1_i0", "doc-1", 1, BoundingBox(0, 0, 1, 1))
    storage.save_image(asset, b"\x89PNG data")
    assert (tmp_path / "images" / "doc-1" / "doc-1_p1_i0.png").exists()
    assert storage.get_image("doc-1", "doc-1_p1_i0") == (b"\x89PNG data", "image/png")


def test_blob_storage_missing_image(storage):
    with pytest.raises(NotFoundError):
        storage.get_image("doc-1", "doc-1_p1_i9")


@pytest.mark.parametrize(
    "bad_id", ["../etc", "..", "a/b", "a\\b", "", "C:\\windows", "doc 1", "%2e%2e", "x" * 200, ".hidden"]
)
def test_blob_storage_rejects_path_traversal(storage, bad_id):
    with pytest.raises(ValidationError):
        storage.get_image(bad_id, "img")
    with pytest.raises(ValidationError):
        storage.get_image("doc", bad_id)
    with pytest.raises(ValidationError):
        storage.save_upload(bad_id, "a.pdf", b"x")
    with pytest.raises(ValidationError):
        storage.save_image(ImageAsset(bad_id, "doc", 1, BoundingBox(0, 0, 1, 1)), b"x")


# --------------------------------------------------------------------------- #
# Celery
# --------------------------------------------------------------------------- #
def test_celery_app_reliability_configuration():
    conf = celery_app.conf
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.task_serializer == "json"
    assert conf.task_routes["ingestion.*"] == {"queue": "ingestion"}
    assert "app.infrastructure.queue.tasks" in conf.include


def test_celery_task_queue_sends_task_by_name():
    with patch.object(celery_app, "send_task", return_value=MagicMock(id="task-1")) as send:
        CeleryTaskQueue().enqueue_ingestion("job-42")
    send.assert_called_once_with("ingestion.ingest_document", args=["job-42"], queue="ingestion")


def test_celery_task_queue_wraps_broker_errors():
    with patch.object(celery_app, "send_task", side_effect=ConnectionError("redis down")):
        with pytest.raises(ExternalServiceError):
            CeleryTaskQueue().enqueue_ingestion("job-42")


def test_ingest_task_delegates_to_process_document_use_case():
    from app.infrastructure.queue.tasks import ingest_document

    container = MagicMock()
    fake_module = MagicMock(get_container=MagicMock(return_value=container))
    with patch.dict("sys.modules", {"app.bootstrap.container": fake_module}):
        ingest_document.apply(args=["job-7"]).get()
    from celery.exceptions import SoftTimeLimitExceeded

    container.process_document.execute.assert_called_once_with(
        "job-7", interrupt_errors=(SoftTimeLimitExceeded,)
    )
    assert ingest_document.name == "ingestion.ingest_document"
