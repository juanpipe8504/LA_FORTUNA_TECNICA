from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from unittest.mock import MagicMock, create_autospec

import pytest

from app.application.documents import ListDocumentsUseCase
from app.application.ingestion import ProcessDocumentUseCase, SubmitDocumentUseCase
from app.domain.exceptions import DocumentParsingError, ExternalServiceError, ValidationError
from app.domain.models import ChunkType, IngestionJob, JobStatus
from app.domain.ports import EmbeddingPort, LLMPort, TaskQueuePort, VectorStorePort
from app.infrastructure.chunking.layout_chunker import LayoutAwareChunker
from app.infrastructure.jobs.in_memory_job_repository import InMemoryJobRepository
from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser

PDF_BYTES = b"%PDF-1.7\n%fake but has the magic header\n"


class FakeEmbeddings(EmbeddingPort):
    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.calls: list[int] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        return [[float(len(t)), 1.0, 0.0, 0.5][: self._dimension] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


class RecordingJobRepository(InMemoryJobRepository):
    """Remembers the (progress, status) history of every update."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[tuple[int, JobStatus]] = []

    def update(self, job: IngestionJob) -> None:
        self.history.append((job.progress, job.status))
        super().update(job)


def make_vision_llm(description: str = "A red block diagram.") -> MagicMock:
    llm = create_autospec(LLMPort, instance=True)
    type(llm).supports_vision = property(lambda self: True)
    llm.describe_image.return_value = description
    return llm


# --------------------------------------------------------------------------- #
# SubmitDocumentUseCase
# --------------------------------------------------------------------------- #
@pytest.fixture
def queue() -> MagicMock:
    return create_autospec(TaskQueuePort, instance=True)


def test_submit_stores_upload_creates_pending_job_and_enqueues(blob_storage, queue):
    jobs = InMemoryJobRepository()
    job = SubmitDocumentUseCase(blob_storage, jobs, queue).execute("C:\\fakepath\\manual.pdf", PDF_BYTES)

    assert job.status == JobStatus.PENDING
    assert job.filename == "manual.pdf"
    assert blob_storage.uploads[job.document_id] == PDF_BYTES
    assert job.file_path == f"/uploads/{job.document_id}.pdf"
    assert jobs.get(job.job_id).document_id == job.document_id
    queue.enqueue_ingestion.assert_called_once_with(job.job_id)


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("a.pdf", b"", "empty"),
        ("a.pdf", b"PK\x03\x04 zip file", "PDF"),
        ("a.exe", PDF_BYTES, "PDF"),
        ("a.pdf", b"%PDF-" + b"x" * (1024 * 1024 + 1), "maximum size"),
    ],
)
def test_submit_validation_errors(blob_storage, queue, filename, data, message):
    use_case = SubmitDocumentUseCase(blob_storage, InMemoryJobRepository(), queue, max_upload_mb=1)
    with pytest.raises(ValidationError, match=message):
        use_case.execute(filename, data)
    queue.enqueue_ingestion.assert_not_called()
    assert blob_storage.uploads == {}


def test_submit_marks_job_failed_when_broker_is_down(blob_storage, queue):
    jobs = InMemoryJobRepository()
    queue.enqueue_ingestion.side_effect = ExternalServiceError("broker down")

    with pytest.raises(ExternalServiceError):
        SubmitDocumentUseCase(blob_storage, jobs, queue).execute("a.pdf", PDF_BYTES)

    [job] = jobs.list()
    assert job.status == JobStatus.FAILED
    assert job.error


# --------------------------------------------------------------------------- #
# ProcessDocumentUseCase
# --------------------------------------------------------------------------- #
@pytest.fixture
def pending_job(pdf_factory) -> tuple[RecordingJobRepository, IngestionJob]:
    jobs = RecordingJobRepository()
    job = IngestionJob("job-1", "doc1", "doc.pdf", str(pdf_factory(with_vector_diagram=True)))
    jobs.create(job)
    return jobs, job


def build_process(
    jobs,
    blob_storage,
    *,
    llm=None,
    describe=False,
    batch_size=32,
    vector_store=None,
    embeddings=None,
    parser=None,
):
    return ProcessDocumentUseCase(
        parser=parser or PyMuPDFParser(blob_storage),
        chunker=LayoutAwareChunker(),
        embeddings=embeddings or FakeEmbeddings(),
        vector_store=vector_store or create_autospec(VectorStorePort, instance=True),
        job_repository=jobs,
        llm=llm,
        blob_storage=blob_storage,
        describe_images=describe,
        batch_size=batch_size,
    )


def test_process_full_pipeline_indexes_chunks_and_completes_job(pending_job, blob_storage):
    jobs, job = pending_job
    store = create_autospec(VectorStorePort, instance=True)
    embeddings = FakeEmbeddings()

    use_case = build_process(jobs, blob_storage, vector_store=store, embeddings=embeddings, batch_size=3)
    use_case.execute("job-1")

    done = jobs.get("job-1")
    assert done.status == JobStatus.COMPLETED and done.progress == 100 and done.error is None
    assert (done.page_count, done.image_count) == (2, 2)
    upserted = [c for call in store.upsert.call_args_list for c in call.args[0]]
    vectors = [v for call in store.upsert.call_args_list for v in call.args[1]]
    assert done.chunk_count == len(upserted) == len(vectors) > 0
    assert all(n <= 3 for n in embeddings.calls)  # batched
    # Order matters: collection exists, old points removed, then new ones written.
    names = [c[0] for c in store.method_calls]
    assert names[0] == "ensure_collection" and names[1] == "delete_document" and names[2] == "upsert"
    store.ensure_collection.assert_called_once_with(4)
    store.delete_document.assert_called_once_with("doc1")


def test_process_reports_monotonic_progress_milestones(pending_job, blob_storage):
    jobs, _ = pending_job
    build_process(jobs, blob_storage, llm=make_vision_llm(), describe=True, batch_size=2).execute("job-1")

    progress = [p for p, _ in jobs.history]
    assert progress == sorted(progress)
    for milestone in (10, 40, 50, 60, 90, 95, 100):
        assert milestone in progress
    assert jobs.history[0][1] == JobStatus.PROCESSING
    assert jobs.history[-1] == (100, JobStatus.COMPLETED)


def test_process_enriches_image_chunks_and_refs_with_vision_descriptions(pending_job, blob_storage):
    jobs, _ = pending_job
    store = create_autospec(VectorStorePort, instance=True)
    llm = make_vision_llm("Three stages connected by arrows.")

    build_process(jobs, blob_storage, llm=llm, describe=True, vector_store=store).execute("job-1")

    assert llm.describe_image.call_count == 2
    image_bytes = llm.describe_image.call_args_list[0].args[0]
    assert image_bytes.startswith(b"\x89PNG")
    assert "Caption: Figura 1" in llm.describe_image.call_args_list[0].kwargs["context"]
    chunks = [c for call in store.upsert.call_args_list for c in call.args[0]]
    image_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMAGE]
    assert all("Description: Three stages connected by arrows." in c.text for c in image_chunks)
    refs = [r for c in chunks for r in c.images if r.image_id == "doc1_p1_i0"]
    assert refs and all(
        r.caption == "Figura 1: Arquitectura del sistema de ingesta — Three stages connected by arrows."
        for r in refs
    )


def test_failed_image_description_is_skipped_not_fatal(pending_job, blob_storage):
    jobs, _ = pending_job
    store = create_autospec(VectorStorePort, instance=True)
    llm = make_vision_llm()
    llm.describe_image.side_effect = [ExternalServiceError("vision timeout"), "Second image description."]

    build_process(jobs, blob_storage, llm=llm, describe=True, vector_store=store).execute("job-1")

    assert jobs.get("job-1").status == JobStatus.COMPLETED
    texts = [
        c.text
        for call in store.upsert.call_args_list
        for c in call.args[0]
        if c.chunk_type == ChunkType.IMAGE
    ]
    assert sum("Description:" in t for t in texts) == 1


@pytest.mark.parametrize("describe, vision", [(False, True), (True, False)])
def test_image_description_disabled_or_unsupported(pending_job, blob_storage, describe, vision):
    jobs, _ = pending_job
    llm = make_vision_llm()
    type(llm).supports_vision = property(lambda self: vision)
    build_process(jobs, blob_storage, llm=llm, describe=describe).execute("job-1")
    llm.describe_image.assert_not_called()
    assert jobs.get("job-1").status == JobStatus.COMPLETED


def test_parsing_error_marks_job_failed_with_message(blob_storage):
    jobs = RecordingJobRepository()
    jobs.create(IngestionJob("job-1", "doc1", "doc.pdf", "/nope.pdf"))
    parser = MagicMock()
    parser.parse.side_effect = DocumentParsingError("Cannot open PDF: broken xref")

    build_process(jobs, blob_storage, parser=parser).execute("job-1")  # must not raise

    job = jobs.get("job-1")
    assert job.status == JobStatus.FAILED
    assert job.error == "Cannot open PDF: broken xref"


def test_unexpected_error_is_not_leaked_verbatim(pending_job, blob_storage):
    jobs, _ = pending_job
    store = create_autospec(VectorStorePort, instance=True)
    store.upsert.side_effect = RuntimeError("password=hunter2 in connection string")

    build_process(jobs, blob_storage, vector_store=store).execute("job-1")

    job = jobs.get("job-1")
    assert job.status == JobStatus.FAILED
    assert job.error == "Internal error (RuntimeError)"


def test_embedding_count_mismatch_fails_job(pending_job, blob_storage):
    jobs, _ = pending_job
    embeddings = MagicMock(spec=EmbeddingPort)
    embeddings.embed_documents.return_value = [[0.1]]
    build_process(jobs, blob_storage, embeddings=embeddings, batch_size=50).execute("job-1")
    job = jobs.get("job-1")
    assert job.status == JobStatus.FAILED and "vectors" in job.error


def test_document_without_content_fails(tmp_path, blob_storage):
    import pymupdf

    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    jobs = RecordingJobRepository()
    jobs.create(IngestionJob("job-1", "doc1", "blank.pdf", str(path)))

    build_process(jobs, blob_storage).execute("job-1")

    job = jobs.get("job-1")
    assert job.status == JobStatus.FAILED and "No extractable" in job.error


def test_completed_job_is_not_reprocessed(pending_job, blob_storage):
    jobs, job = pending_job
    job.status = JobStatus.COMPLETED
    jobs.update(job)
    parser = MagicMock()
    build_process(jobs, blob_storage, parser=parser).execute("job-1")
    parser.parse.assert_not_called()


def test_unknown_job_is_ignored(blob_storage):
    parser = MagicMock()
    build_process(InMemoryJobRepository(), blob_storage, parser=parser).execute("missing")
    parser.parse.assert_not_called()


def test_invalid_batch_size(blob_storage):
    with pytest.raises(ValueError):
        build_process(InMemoryJobRepository(), blob_storage, batch_size=0)


# --------------------------------------------------------------------------- #
# ListDocumentsUseCase
# --------------------------------------------------------------------------- #
def test_list_documents_returns_only_completed_newest_first():
    jobs = InMemoryJobRepository()
    base = IngestionJob("j0", "d0", "a.pdf", "/a").created_at
    statuses = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.COMPLETED, JobStatus.PENDING]
    for i, status in enumerate(statuses):
        job = IngestionJob(
            f"j{i}", f"d{i}", f"{i}.pdf", "/x", status=status, created_at=base + timedelta(minutes=i)
        )
        jobs.create(job)

    documents = ListDocumentsUseCase(jobs).execute()

    assert [d.document_id for d in documents] == ["d2", "d0"]
