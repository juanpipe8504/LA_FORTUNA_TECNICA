"""Composition root: the ONLY module that knows concrete adapter classes.

``build_container(settings)`` wires ports to adapters according to
configuration and injects them into the application use cases. Infrastructure
modules are imported lazily inside each factory, so choosing Ollama never
imports ``openai`` and importing this module stays cheap (the API and the
Celery worker both import it).

Building the container never opens network connections: Redis/Qdrant clients
connect lazily on first use, and embedding models load lazily.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.core.config import EmbeddingProvider, LLMProvider, Settings, get_settings
from app.core.logging import get_logger
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.application.documents import ListDocumentsUseCase
    from app.application.ingestion import ProcessDocumentUseCase, SubmitDocumentUseCase
    from app.application.query import AnswerQuestionUseCase

logger = get_logger(__name__)

HealthCheck = Callable[[], bool]

_REDIS_CONNECT_TIMEOUT_SECONDS = 2.0
_REDIS_SOCKET_TIMEOUT_SECONDS = 5.0
_QDRANT_TIMEOUT_SECONDS = 10


class ConfigurationError(RuntimeError):
    """The settings are inconsistent (e.g. OpenAI selected without an API key)."""


@dataclass
class Container:
    """Holds the fully wired object graph (ports + use cases)."""

    settings: Settings
    llm: LLMPort
    embeddings: EmbeddingPort
    vector_store: VectorStorePort
    blob_storage: BlobStoragePort
    job_repository: JobRepositoryPort
    task_queue: TaskQueuePort
    parser: DocumentParserPort
    chunker: ChunkerPort
    submit_document: SubmitDocumentUseCase
    process_document: ProcessDocumentUseCase
    answer_question: AnswerQuestionUseCase
    list_documents: ListDocumentsUseCase
    health_checks: dict[str, HealthCheck] = field(default_factory=dict)
    closers: list[Callable[[], None]] = field(default_factory=list)

    def close(self) -> None:
        """Release external resources (connection pools). Never raises."""
        for closer in self.closers:
            try:
                closer()
            except Exception:  # noqa: BLE001 - best effort on shutdown
                logger.warning("resource close failed", exc_info=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _secret(value: Any) -> str | None:
    if value is None:
        return None
    raw = value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)
    return raw or None


def _require_openai_key(settings: Settings, purpose: str) -> str:
    key = _secret(settings.openai_api_key)
    if not key:
        raise ConfigurationError(
            f"{purpose} provider is 'openai' but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or choose another provider (e.g. LLM_PROVIDER=ollama, "
            "EMBEDDING_PROVIDER=fastembed)."
        )
    return key


def _ensure_data_dirs(settings: Settings) -> None:
    for path in (settings.data_dir, settings.uploads_dir, settings.images_dir):
        os.makedirs(path, exist_ok=True)


def _retry_policy(settings: Settings) -> Any:
    from app.infrastructure.resilience.retry import RetryPolicy

    return RetryPolicy(
        max_attempts=settings.retry_max_attempts,
        base_delay=settings.retry_base_delay_seconds,
        max_delay=settings.retry_max_delay_seconds,
    )


def _circuit_breaker(name: str, settings: Settings) -> Any:
    from app.infrastructure.resilience.circuit_breaker import CircuitBreaker

    return CircuitBreaker(
        name=name,
        failure_threshold=settings.circuit_failure_threshold,
        recovery_timeout=settings.circuit_recovery_seconds,
    )


# --------------------------------------------------------------------------- #
# Adapter factories (one per concern)
# --------------------------------------------------------------------------- #
def _build_raw_llm(settings: Settings) -> LLMPort:
    if settings.llm_provider is LLMProvider.OPENAI:
        api_key = _require_openai_key(settings, "LLM")
        from app.infrastructure.llm.openai_llm import OpenAILLM

        return OpenAILLM(
            api_key=api_key,
            model=settings.openai_chat_model,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider is LLMProvider.OLLAMA:
        from app.infrastructure.llm.ollama_llm import OllamaLLM

        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            timeout=settings.llm_timeout_seconds,
        )
    raise ConfigurationError(f"Unsupported LLM provider: {settings.llm_provider!r}")


def _build_llm(settings: Settings) -> LLMPort:
    from app.infrastructure.resilience.resilient import ResilientLLM

    return ResilientLLM(
        _build_raw_llm(settings),
        breaker=_circuit_breaker("llm", settings),
        policy=_retry_policy(settings),
    )


def _build_raw_embeddings(settings: Settings) -> EmbeddingPort:
    provider = settings.embedding_provider
    if provider is EmbeddingProvider.OPENAI:
        api_key = _require_openai_key(settings, "Embedding")
        from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(
            api_key=api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
        )
    if provider is EmbeddingProvider.OLLAMA:
        from app.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
            timeout=settings.llm_timeout_seconds,
        )
    if provider is EmbeddingProvider.FASTEMBED:
        from app.infrastructure.embeddings.fastembed_embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=settings.fastembed_model)
    raise ConfigurationError(f"Unsupported embedding provider: {provider!r}")


def _build_embeddings(settings: Settings) -> EmbeddingPort:
    from app.infrastructure.resilience.resilient import ResilientEmbeddings

    return ResilientEmbeddings(
        _build_raw_embeddings(settings),
        breaker=_circuit_breaker("embeddings", settings),
        policy=_retry_policy(settings),
    )


def _build_qdrant_client(settings: Settings) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=_secret(settings.qdrant_api_key),
        timeout=_QDRANT_TIMEOUT_SECONDS,
        check_compatibility=False,  # avoid a network round-trip at build time
    )


def _build_vector_store(settings: Settings, client: Any) -> VectorStorePort:
    from app.infrastructure.vectorstore.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        client=client,
        collection=settings.qdrant_collection,
        bm25_language=settings.bm25_language,
    )


def _build_redis_client(settings: Settings) -> Any:
    import redis

    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=_REDIS_CONNECT_TIMEOUT_SECONDS,
        socket_timeout=_REDIS_SOCKET_TIMEOUT_SECONDS,
        health_check_interval=30,
    )


def _build_job_repository(settings: Settings, redis_client: Any) -> JobRepositoryPort:
    from app.core.config import Environment

    if settings.app_env is Environment.TEST:
        from app.infrastructure.jobs.in_memory_job_repository import InMemoryJobRepository

        return InMemoryJobRepository()
    from app.infrastructure.jobs.redis_job_repository import RedisJobRepository

    return RedisJobRepository(redis_client=redis_client)


def _build_blob_storage(settings: Settings) -> BlobStoragePort:
    from app.infrastructure.storage.local_blob_storage import LocalBlobStorage

    return LocalBlobStorage(uploads_dir=settings.uploads_dir, images_dir=settings.images_dir)


def _build_task_queue() -> TaskQueuePort:
    from app.infrastructure.queue.celery_task_queue import CeleryTaskQueue

    return CeleryTaskQueue()


def _build_parser(settings: Settings, blob_storage: BlobStoragePort) -> DocumentParserPort:
    from app.infrastructure.parsing.pymupdf_parser import PyMuPDFParser

    return PyMuPDFParser(blob_storage=blob_storage, min_image_size_px=settings.min_image_size_px)


def _build_chunker(settings: Settings) -> ChunkerPort:
    from app.infrastructure.chunking.layout_chunker import LayoutAwareChunker

    return LayoutAwareChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_sentences=settings.chunk_overlap_sentences,
        image_link_max_distance=settings.image_link_max_distance,
    )


def _build_health_checks(redis_client: Any, qdrant_client: Any) -> dict[str, HealthCheck]:
    def redis_ok() -> bool:
        return bool(redis_client.ping())

    def qdrant_ok() -> bool:
        qdrant_client.get_collections()
        return True

    return {"redis": redis_ok, "qdrant": qdrant_ok}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_container(settings: Settings) -> Container:
    """Wire every port to its adapter and build the use cases.

    Raises ``ConfigurationError`` when the settings are inconsistent, so a
    misconfigured deployment fails fast at startup instead of on first request.
    """
    from app.application.documents import ListDocumentsUseCase
    from app.application.ingestion import ProcessDocumentUseCase, SubmitDocumentUseCase
    from app.application.query import AnswerQuestionUseCase

    # Validate provider configuration before touching anything else.
    llm = _build_llm(settings)
    embeddings = _build_embeddings(settings)

    _ensure_data_dirs(settings)
    redis_client = _build_redis_client(settings)
    qdrant_client = _build_qdrant_client(settings)

    vector_store = _build_vector_store(settings, qdrant_client)
    blob_storage = _build_blob_storage(settings)
    job_repository = _build_job_repository(settings, redis_client)
    task_queue = _build_task_queue()
    parser = _build_parser(settings, blob_storage)
    chunker = _build_chunker(settings)

    container = Container(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        vector_store=vector_store,
        blob_storage=blob_storage,
        job_repository=job_repository,
        task_queue=task_queue,
        parser=parser,
        chunker=chunker,
        submit_document=SubmitDocumentUseCase(
            blob_storage=blob_storage,
            job_repository=job_repository,
            task_queue=task_queue,
            max_upload_mb=settings.max_upload_mb,
        ),
        process_document=ProcessDocumentUseCase(
            parser=parser,
            chunker=chunker,
            embeddings=embeddings,
            vector_store=vector_store,
            job_repository=job_repository,
            llm=llm,
            blob_storage=blob_storage,
            describe_images=settings.describe_images,
            batch_size=settings.embedding_batch_size,
        ),
        answer_question=AnswerQuestionUseCase(
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
            max_images=settings.max_images_per_answer,
            history_max_turns=settings.history_max_turns,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ),
        list_documents=ListDocumentsUseCase(job_repository=job_repository),
        health_checks=_build_health_checks(redis_client, qdrant_client),
        closers=[redis_client.close, qdrant_client.close],
    )
    logger.info(
        "container built",
        extra={
            "llm_provider": settings.llm_provider.value,
            "embedding_provider": settings.embedding_provider.value,
            "app_env": settings.app_env.value,
        },
    )
    return container


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Process-wide singleton used by the API lifespan and the Celery worker."""
    return build_container(get_settings())
