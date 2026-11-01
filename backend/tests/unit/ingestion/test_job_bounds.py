"""Bounded / self-healing ingestion: time limits, image budget, watchdog, redelivery."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pymupdf
import pytest
from celery.exceptions import SoftTimeLimitExceeded

from app.application.ingestion import TIME_LIMIT_ERROR, ProcessDocumentUseCase
from app.application.jobs import STALE_JOB_ERROR, GetJobUseCase
from app.core.config import BROKER_VISIBILITY_MARGIN_SECONDS, Settings, get_settings
from app.domain.exceptions import CircuitOpenError, ExternalServiceError, NotFoundError
from app.domain.models import ChunkType, IngestionJob, JobStatus, utcnow
from app.domain.ports import VectorStorePort
from app.infrastructure.chunking.layout_chunker import LayoutAwareChunker
from app.infrastructure.jobs.in_memory_job_repository import InMemoryJobRepository
from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser
from tests.unit.ingestion.conftest import make_png
from tests.unit.ingestion.test_ingestion_use_cases import FakeEmbeddings, make_vision_llm


class MessageRecordingJobRepository(InMemoryJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.updates: list[tuple[int, JobStatus, str]] = []

    def update(self, job: IngestionJob) -> None:
        self.updates.append((job.progress, job.status, job.message))
        super().update(job)

    @property
    def messages(self) -> list[str]:
        return [m for _, _, m in self.updates]


class FakeInterrupt(Exception):
    """Stands in for a framework signal such as Celery's SoftTimeLimitExceeded."""


def build_pdf_with_images(path: Path, n: int) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "1. Diagrams", fontsize=18, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(72, 70, 523, 110), "Several diagrams follow.", fontsize=11)
    for i in range(n):
        y = 120 + i * 150
        page.insert_image(pymupdf.Rect(72, y, 232, y + 100), stream=make_png(color=(30 * i, 90, 200)))
        page.insert_text((72, y + 115), f"Figura {i + 1}: Diagrama {i + 1}", fontsize=9)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def make_job(tmp_path) -> Callable[..., MessageRecordingJobRepository]:
    def factory(images: int = 2, status: JobStatus = JobStatus.PENDING, progress: int = 0):
        jobs = MessageRecordingJobRepository()
        path = build_pdf_with_images(tmp_path / "doc.pdf", images)
        jobs.create(IngestionJob("job-1", "doc1", "doc.pdf", str(path), status=status, progress=progress))
        jobs.updates.clear()
        return jobs

    return factory


def build(jobs, blob_storage, llm=None, store=None, parser=None, **kwargs) -> ProcessDocumentUseCase:
    return ProcessDocumentUseCase(
        parser=parser or PyMuPDFParser(blob_storage),
        chunker=LayoutAwareChunker(),
        embeddings=FakeEmbeddings(),
        vector_store=store or create_autospec(VectorStorePort, instance=True),
        job_repository=jobs,
        llm=llm,
        blob_storage=blob_storage,
        describe_images=llm is not None,
        **kwargs,
    )


def indexed_image_texts(store) -> list[str]:
    return [
        c.text
        for call in store.upsert.call_args_list
        for c in call.args[0]
        if c.chunk_type == ChunkType.IMAGE
    ]


# --------------------------------------------------------------------------- #
# Time limits
# --------------------------------------------------------------------------- #
def test_interrupt_error_during_image_description_propagates_not_swallowed(make_job, blob_storage):
    jobs = make_job()
    llm = make_vision_llm()
    llm.describe_image.side_effect = FakeInterrupt()
    use_case = build(jobs, blob_storage, llm=llm)

    with pytest.raises(FakeInterrupt):
        use_case.execute("job-1", interrupt_errors=(FakeInterrupt,))

    assert llm.describe_image.call_count == 1  # not treated as a per-image failure
    assert jobs.get("job-1").status == JobStatus.PROCESSING  # the caller decides the outcome
    use_case.mark_timed_out("job-1")
    job = jobs.get("job-1")
    assert job.status == JobStatus.FAILED
    assert job.error == TIME_LIMIT_ERROR
    assert job.error.startswith("Tiempo máximo de procesamiento excedido")


def test_interrupt_error_in_pipeline_step_propagates(make_job, blob_storage):
    jobs = make_job()
    parser = MagicMock()
    parser.parse.side_effect = FakeInterrupt()
    with pytest.raises(FakeInterrupt):
        build(jobs, blob_storage, parser=parser).execute("job-1", interrupt_errors=(FakeInterrupt,))


def test_without_interrupt_errors_unknown_exceptions_still_fail_the_job(make_job, blob_storage):
    jobs = make_job()
    parser = MagicMock()
    parser.parse.side_effect = FakeInterrupt()
    build(jobs, blob_storage, parser=parser).execute("job-1")  # must not raise
    assert jobs.get("job-1").status == JobStatus.FAILED


def test_mark_failed_never_overwrites_a_completed_job(make_job, blob_storage):
    jobs = make_job(status=JobStatus.COMPLETED, progress=100)
    build(jobs, blob_storage).mark_timed_out("job-1")
    assert jobs.get("job-1").status == JobStatus.COMPLETED


def test_mark_failed_unknown_job_does_not_raise(blob_storage):
    build(InMemoryJobRepository(), blob_storage).mark_timed_out("missing")


def test_celery_task_marks_job_failed_on_soft_time_limit(make_job, blob_storage):
    """End to end through the real task: the signal hits inside the LLM call."""
    from app.infrastructure.queue.tasks import ingest_document

    jobs = make_job()
    llm = make_vision_llm()
    llm.describe_image.side_effect = SoftTimeLimitExceeded()
    container = MagicMock(process_document=build(jobs, blob_storage, llm=llm))
    fake_module = MagicMock(get_container=MagicMock(return_value=container))
    with patch.dict("sys.modules", {"app.bootstrap.container": fake_module}):
        result = ingest_document.apply(args=["job-1"])

    assert result.successful()  # acknowledged, not re-delivered
    job = jobs.get("job-1")
    assert job.status == JobStatus.FAILED
    assert job.error == TIME_LIMIT_ERROR


def test_celery_time_limits_and_visibility_timeout():
    from app.infrastructure.queue.celery_app import celery_app
    from app.infrastructure.queue.tasks import ingest_document

    settings = get_settings()
    conf = celery_app.conf
    fields = Settings.model_fields
    assert fields["ingestion_soft_time_limit_seconds"].default == 1800
    assert fields["ingestion_hard_time_limit_seconds"].default == 1900
    assert fields["describe_images_budget_seconds"].default == 600
    assert fields["job_stale_after_seconds"].default == 900
    assert conf.task_soft_time_limit == settings.ingestion_soft_time_limit_seconds
    assert conf.task_time_limit == settings.ingestion_hard_time_limit_seconds
    assert ingest_document.soft_time_limit == settings.ingestion_soft_time_limit_seconds
    assert ingest_document.time_limit == settings.ingestion_hard_time_limit_seconds
    visibility = conf.broker_transport_options["visibility_timeout"]
    assert visibility == settings.ingestion_hard_time_limit_seconds + BROKER_VISIBILITY_MARGIN_SECONDS
    assert visibility > conf.task_time_limit > conf.task_soft_time_limit


def test_settings_reject_hard_limit_not_above_soft_limit():
    with pytest.raises(ValueError, match="HARD_TIME_LIMIT"):
        Settings(ingestion_soft_time_limit_seconds=100, ingestion_hard_time_limit_seconds=100)


# --------------------------------------------------------------------------- #
# Image description budget / circuit breaking
# --------------------------------------------------------------------------- #
def test_budget_exceeded_skips_remaining_images_and_completes(make_job, blob_storage):
    jobs = make_job(images=3)
    store = create_autospec(VectorStorePort, instance=True)
    llm = make_vision_llm("A diagram.")
    ticks = iter(range(0, 10_000, 400))  # every clock read advances 400 s
    use_case = build(
        jobs, blob_storage, llm=llm, store=store,
        describe_images_budget_seconds=600, clock=lambda: next(ticks),
    )

    use_case.execute("job-1")

    # start=0, before #1 -> 400 (< 600, describe), before #2 -> 800 (>= 600, stop)
    assert llm.describe_image.call_count == 1
    job = jobs.get("job-1")
    assert job.status == JobStatus.COMPLETED
    assert any("Skipping 2 image description(s) (time budget exceeded)" in m for m in jobs.messages)
    texts = indexed_image_texts(store)
    assert len(texts) == 3 and sum("Description:" in t for t in texts) == 1


def test_stops_after_consecutive_failures(make_job, blob_storage):
    jobs = make_job(images=4)
    store = create_autospec(VectorStorePort, instance=True)
    llm = make_vision_llm()
    llm.describe_image.side_effect = ExternalServiceError("Ollama HTTP 500: llama-server timed out")

    use_case = build(jobs, blob_storage, llm=llm, store=store, describe_images_max_consecutive_failures=2)
    use_case.execute("job-1")

    assert llm.describe_image.call_count == 2
    assert jobs.get("job-1").status == JobStatus.COMPLETED
    assert any("Skipping 2 image description(s) (2 consecutive failures)" in m for m in jobs.messages)
    # Images are still indexed with the captions from the PDF.
    texts = indexed_image_texts(store)
    assert len(texts) == 4 and all("Figura" in t for t in texts)


def test_success_resets_the_consecutive_failure_counter(make_job, blob_storage):
    jobs = make_job(images=4)
    llm = make_vision_llm()
    llm.describe_image.side_effect = [ExternalServiceError("x"), "ok", ExternalServiceError("y"), "ok"]
    build(jobs, blob_storage, llm=llm, describe_images_max_consecutive_failures=2).execute("job-1")
    assert llm.describe_image.call_count == 4


def test_open_circuit_stops_describing_immediately(make_job, blob_storage):
    jobs = make_job(images=3)
    store = create_autospec(VectorStorePort, instance=True)
    llm = make_vision_llm()
    llm.describe_image.side_effect = CircuitOpenError("Circuit 'llm' is open")

    use_case = build(jobs, blob_storage, llm=llm, store=store, describe_images_max_consecutive_failures=5)
    use_case.execute("job-1")

    assert llm.describe_image.call_count == 1
    assert jobs.get("job-1").status == JobStatus.COMPLETED
    assert any("vision model unavailable" in m for m in jobs.messages)
    assert len(indexed_image_texts(store)) == 3


def test_heartbeat_message_before_each_image_and_progress_monotonic(make_job, blob_storage):
    jobs = make_job(images=3)
    llm = make_vision_llm()
    seen: list[str] = []
    llm.describe_image.side_effect = lambda *a, **k: seen.append(jobs.get("job-1").message) or "desc"

    build(jobs, blob_storage, llm=llm).execute("job-1")

    assert seen == ["Describing image 1/3", "Describing image 2/3", "Describing image 3/3"]
    progress = [p for p, _, _ in jobs.updates]
    assert progress == sorted(progress)


def test_invalid_max_consecutive_failures(blob_storage):
    with pytest.raises(ValueError):
        build(InMemoryJobRepository(), blob_storage, describe_images_max_consecutive_failures=0)


# --------------------------------------------------------------------------- #
# Redelivery semantics
# --------------------------------------------------------------------------- #
def test_redelivered_processing_job_restarts_from_scratch(make_job, blob_storage):
    jobs = make_job(status=JobStatus.PROCESSING, progress=55)
    build(jobs, blob_storage).execute("job-1")
    job = jobs.get("job-1")
    assert job.status == JobStatus.COMPLETED and job.progress == 100
    assert jobs.updates[0][0] == 10  # progress reset, first milestone is parsing


def test_redelivered_failed_job_is_not_resurrected(make_job, blob_storage):
    jobs = make_job(status=JobStatus.FAILED)
    parser = MagicMock()
    build(jobs, blob_storage, parser=parser).execute("job-1")
    parser.parse.assert_not_called()
    assert jobs.get("job-1").status == JobStatus.FAILED


def test_job_reported_stale_is_retried_on_redelivery(make_job, blob_storage):
    """The watchdog verdict is read-only: the stored job is still PROCESSING, so
    a re-delivered task (worker genuinely lost) can finish it."""
    jobs = make_job(status=JobStatus.PROCESSING, progress=55)
    later = utcnow() + timedelta(hours=3)
    assert GetJobUseCase(jobs, 900, clock=lambda: later).execute("job-1").status == JobStatus.FAILED

    build(jobs, blob_storage).execute("job-1")

    assert GetJobUseCase(jobs, 900).execute("job-1").status == JobStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Stale-job watchdog
# --------------------------------------------------------------------------- #
@pytest.fixture
def stored_job() -> Callable[..., InMemoryJobRepository]:
    def factory(status: JobStatus) -> InMemoryJobRepository:
        jobs = InMemoryJobRepository()
        jobs.create(IngestionJob("job-1", "doc1", "a.pdf", "/a", status=status, progress=55, message="m"))
        return jobs

    return factory


def test_watchdog_reports_stale_processing_job_as_failed(stored_job):
    jobs = stored_job(JobStatus.PROCESSING)
    updated = jobs.get("job-1").updated_at
    use_case = GetJobUseCase(jobs, stale_after_seconds=900, clock=lambda: updated + timedelta(seconds=901))

    job = use_case.execute("job-1")

    assert job.status == JobStatus.FAILED
    assert job.error == STALE_JOB_ERROR and job.error.startswith("El procesamiento se interrumpió")
    assert job.progress == 55
    assert jobs.get("job-1").status == JobStatus.PROCESSING  # not persisted


def test_watchdog_keeps_fresh_processing_job(stored_job):
    jobs = stored_job(JobStatus.PROCESSING)
    updated = jobs.get("job-1").updated_at
    use_case = GetJobUseCase(jobs, stale_after_seconds=900, clock=lambda: updated + timedelta(seconds=899))
    assert use_case.execute("job-1").status == JobStatus.PROCESSING


@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.FAILED])
def test_watchdog_ignores_non_processing_jobs(stored_job, status):
    jobs = stored_job(status)
    later = utcnow() + timedelta(days=1)
    job = GetJobUseCase(jobs, stale_after_seconds=900, clock=lambda: later).execute("job-1")
    assert job.status == status and job.error is None


def test_progress_updates_refresh_the_watchdog(stored_job):
    jobs = stored_job(JobStatus.PROCESSING)
    clock_now = [utcnow() + timedelta(seconds=1000)]
    use_case = GetJobUseCase(jobs, stale_after_seconds=900, clock=lambda: clock_now[0])
    assert use_case.execute("job-1").status == JobStatus.FAILED

    job = jobs.get("job-1")
    job.message = "Describing image 2/5"
    jobs.update(job)  # refreshes updated_at
    clock_now[0] = utcnow()
    assert use_case.execute("job-1").status == JobStatus.PROCESSING


def test_get_job_unknown_raises_not_found():
    with pytest.raises(NotFoundError):
        GetJobUseCase(InMemoryJobRepository()).execute("missing")


def test_get_job_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        GetJobUseCase(InMemoryJobRepository(), stale_after_seconds=0)
