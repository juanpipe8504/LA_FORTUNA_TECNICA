"""Ingestion use cases.

``SubmitDocumentUseCase`` runs in the API process: it validates and stores the
upload, creates a PENDING job and enqueues it – it never does heavy work, so
the HTTP request returns immediately (202).

``ProcessDocumentUseCase`` runs in the Celery worker: parse -> chunk ->
(optionally) describe images with a vision LLM -> embed -> index, reporting
progress on the job so clients can poll ``GET /jobs/{id}``.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from app.core.logging import bind_context, get_logger
from app.domain.exceptions import DomainError, ExternalServiceError, ValidationError
from app.domain.models import (
    Chunk,
    ChunkType,
    ImageAsset,
    IngestionJob,
    JobStatus,
    ParsedDocument,
)
from app.domain.ports import (
    BlobStoragePort,
    ChunkerPort,
    DocumentParserPort,
    EmbeddingPort,
    JobRepositoryPort,
    LLMPort,
    TaskQueuePort,
    VectorStorePort,
)

logger = get_logger(__name__)

_PDF_MAGIC = b"%PDF-"
_MAX_FILENAME_LENGTH = 255
_MAX_ERROR_LENGTH = 500


class SubmitDocumentUseCase:
    def __init__(
        self,
        blob_storage: BlobStoragePort,
        job_repository: JobRepositoryPort,
        task_queue: TaskQueuePort,
        max_upload_mb: int = 50,
    ) -> None:
        self._blob_storage = blob_storage
        self._jobs = job_repository
        self._queue = task_queue
        self._max_bytes = max_upload_mb * 1024 * 1024

    def execute(self, filename: str, data: bytes) -> IngestionJob:
        safe_name = self._sanitise_filename(filename)
        self._validate(data)

        document_id = str(uuid.uuid4())
        file_path = self._blob_storage.save_upload(document_id, safe_name, data)
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            filename=safe_name,
            file_path=file_path,
            message="Queued for processing",
        )
        self._jobs.create(job)
        try:
            self._queue.enqueue_ingestion(job.job_id)
        except Exception:
            # The job must not stay PENDING forever if the broker is down.
            job.status = JobStatus.FAILED
            job.error = "Could not enqueue the document for processing"
            job.message = "Failed"
            self._jobs.update(job)
            raise
        logger.info(
            "document_submitted",
            extra={
                "job_id": job.job_id,
                "document_id": document_id,
                "doc_filename": safe_name,
                "bytes": len(data),
            },
        )
        return job

    def _validate(self, data: bytes) -> None:
        if not data:
            raise ValidationError("The uploaded file is empty")
        if len(data) > self._max_bytes:
            raise ValidationError(f"File exceeds the maximum size of {self._max_bytes // (1024 * 1024)} MB")
        # The PDF spec allows the header anywhere in the first 1024 bytes.
        if _PDF_MAGIC not in data[:1024]:
            raise ValidationError("Only PDF files are supported")

    @staticmethod
    def _sanitise_filename(filename: str) -> str:
        # Keep only the base name (browsers may send full paths) and strip controls.
        name = os.path.basename((filename or "").replace("\\", "/")).strip()
        name = "".join(ch for ch in name if ch.isprintable())[:_MAX_FILENAME_LENGTH]
        if not name:
            return "document.pdf"
        if not name.lower().endswith(".pdf"):
            raise ValidationError("Only PDF files are supported")
        return name


class ProcessDocumentUseCase:
    # Progress milestones reported on the job (0..100).
    PARSING, CHUNKING, DESCRIBING_START, EMBEDDING_START, EMBEDDING_END, INDEXING = 10, 40, 50, 60, 90, 95

    def __init__(
        self,
        parser: DocumentParserPort,
        chunker: ChunkerPort,
        embeddings: EmbeddingPort,
        vector_store: VectorStorePort,
        job_repository: JobRepositoryPort,
        llm: LLMPort | None = None,
        blob_storage: BlobStoragePort | None = None,
        describe_images: bool = False,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._jobs = job_repository
        self._llm = llm
        self._blob_storage = blob_storage
        self._describe_images = describe_images
        self._batch_size = batch_size

    def execute(self, job_id: str) -> None:
        """Process a job. Never raises: failures are recorded on the job."""
        with bind_context(job_id=job_id):
            try:
                job = self._jobs.get(job_id)
            except Exception:
                logger.exception("job_not_loadable")
                return
            if job.status == JobStatus.COMPLETED:
                # At-least-once delivery (acks_late) can redeliver a finished job.
                logger.info("job_already_completed_skipping")
                return
            with bind_context(document_id=job.document_id):
                self._run(job)

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def _run(self, job: IngestionJob) -> None:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        job.status, job.error = JobStatus.PROCESSING, None
        try:
            self._progress(job, self.PARSING, "Parsing PDF layout")
            with self._timed(timings, "parse_ms"):
                document = self._parser.parse(
                    job.file_path, document_id=job.document_id, filename=job.filename
                )
            job.page_count = document.page_count
            job.image_count = len(document.images)

            self._progress(job, self.CHUNKING, "Building layout-aware chunks")
            with self._timed(timings, "chunk_ms"):
                chunks = self._chunker.chunk(document)
            if not chunks:
                raise ValidationError("No extractable text, tables or images were found in the document")

            if self._should_describe_images(document):
                self._progress(job, self.DESCRIBING_START, "Describing images")
                with self._timed(timings, "describe_ms"):
                    self._enrich_images(job, document, chunks)

            self._progress(job, self.EMBEDDING_START, "Generating embeddings")
            with self._timed(timings, "embed_ms"):
                vectors = self._embed(job, chunks)

            self._progress(job, self.INDEXING, "Indexing in the vector store")
            with self._timed(timings, "index_ms"):
                self._index(job.document_id, chunks, vectors)

            job.chunk_count = len(chunks)
            job.status = JobStatus.COMPLETED
            self._progress(job, 100, "Completed")
            logger.info(
                "ingestion_completed",
                extra={
                    "pages": job.page_count,
                    "chunks": job.chunk_count,
                    "images": job.image_count,
                    "chunk_types": self._count_types(chunks),
                    **{k: round(v, 1) for k, v in timings.items()},
                    "total_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        except Exception as exc:
            self._fail(job, exc, timings)

    def _fail(self, job: IngestionJob, exc: Exception, timings: dict[str, float]) -> None:
        logger.exception("ingestion_failed", extra={"progress": job.progress, **timings})
        # Domain errors carry user-facing messages; unexpected ones are not leaked verbatim.
        message = str(exc) if isinstance(exc, DomainError) else f"Internal error ({type(exc).__name__})"
        job.status = JobStatus.FAILED
        job.error = message[:_MAX_ERROR_LENGTH] or type(exc).__name__
        job.message = "Failed"
        try:
            self._jobs.update(job)
        except Exception:
            logger.exception("job_failure_not_persisted")

    def _progress(self, job: IngestionJob, progress: int, message: str) -> None:
        job.progress = progress
        job.message = message
        self._jobs.update(job)
        logger.info("ingestion_progress", extra={"progress": progress, "step": message})

    @staticmethod
    @contextmanager
    def _timed(timings: dict[str, float], key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            timings[key] = (time.perf_counter() - start) * 1000

    @staticmethod
    def _count_types(chunks: Sequence[Chunk]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.chunk_type.value] = counts.get(chunk.chunk_type.value, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # Image descriptions (optional, vision LLM)
    # ------------------------------------------------------------------ #
    def _should_describe_images(self, document: ParsedDocument) -> bool:
        return bool(
            self._describe_images
            and document.images
            and self._llm is not None
            and self._blob_storage is not None
            and self._llm.supports_vision
        )

    def _enrich_images(self, job: IngestionJob, document: ParsedDocument, chunks: list[Chunk]) -> None:
        """Caption every image with the vision LLM. A failing image is logged and
        skipped – a missing description must never fail the whole document."""
        if self._llm is None or self._blob_storage is None:
            return
        image_chunks = self._image_chunks_by_id(chunks)
        total = len(document.images)
        span = self.EMBEDDING_START - self.DESCRIBING_START
        described = 0
        for index, asset in enumerate(document.images, start=1):
            chunk = image_chunks.get(asset.image_id)
            try:
                data, _ = self._blob_storage.get_image(asset.document_id, asset.image_id)
                context = self._image_context(asset, chunk)
                description = self._llm.describe_image(data, context=context).strip()
            except Exception as exc:
                logger.warning(
                    "image_description_failed",
                    extra={"image_id": asset.image_id, "error": f"{type(exc).__name__}: {exc}"},
                )
                description = ""
            if description:
                self._apply_description(asset, description, chunk, chunks)
                described += 1
            new_progress = self.DESCRIBING_START + int(span * index / total)
            if new_progress > job.progress:
                self._progress(job, new_progress, f"Describing images ({index}/{total})")
        logger.info("images_described", extra={"described": described, "total": total})

    @staticmethod
    def _image_chunks_by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
        """IMAGE chunk of each image: the ref whose bbox is the chunk's own bbox."""
        result: dict[str, Chunk] = {}
        for chunk in chunks:
            if chunk.chunk_type != ChunkType.IMAGE:
                continue
            for ref in chunk.images:
                if ref.page == chunk.page_start and ref.bbox == chunk.bbox:
                    result.setdefault(ref.image_id, chunk)
                    break
        return result

    @staticmethod
    def _image_context(asset: ImageAsset, chunk: Chunk | None) -> str:
        parts = []
        if chunk and chunk.section_path:
            parts.append(f"Section: {' > '.join(chunk.section_path)}")
        if asset.caption:
            parts.append(f"Caption: {asset.caption}")
        parts.append(f"Page {asset.page} of a technical document")
        return "\n".join(parts)

    @staticmethod
    def _apply_description(
        asset: ImageAsset, description: str, image_chunk: Chunk | None, chunks: list[Chunk]
    ) -> None:
        asset.caption = f"{asset.caption} — {description}" if asset.caption else description
        if image_chunk is not None:
            image_chunk.text = f"{image_chunk.text}\nDescription: {description}"
        for chunk in chunks:
            for ref in chunk.images:
                if ref.image_id == asset.image_id:
                    ref.caption = asset.caption

    # ------------------------------------------------------------------ #
    # Embedding + indexing
    # ------------------------------------------------------------------ #
    def _batches(self, items: Sequence[Chunk]) -> Iterator[Sequence[Chunk]]:
        for start in range(0, len(items), self._batch_size):
            yield items[start : start + self._batch_size]

    def _embed(self, job: IngestionJob, chunks: list[Chunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        span = self.EMBEDDING_END - self.EMBEDDING_START
        for batch in self._batches(chunks):
            batch_vectors = self._embeddings.embed_documents([c.text for c in batch])
            if len(batch_vectors) != len(batch):
                raise ExternalServiceError(
                    f"Embedding provider returned {len(batch_vectors)} vectors for {len(batch)} chunks"
                )
            vectors.extend(batch_vectors)
            progress = self.EMBEDDING_START + int(span * len(vectors) / len(chunks))
            if progress > job.progress:
                self._progress(job, progress, f"Generating embeddings ({len(vectors)}/{len(chunks)})")
        return vectors

    def _index(self, document_id: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._vector_store.ensure_collection(self._embeddings.dimension)
        # Idempotency: re-processing a document replaces its previous points.
        self._vector_store.delete_document(document_id)
        for start in range(0, len(chunks), self._batch_size):
            end = start + self._batch_size
            self._vector_store.upsert(chunks[start:end], vectors[start:end])
