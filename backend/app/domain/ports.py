"""Ports (interfaces) of the hexagon.

Application use cases depend only on these abstractions. Infrastructure
adapters implement them and are wired in ``app.bootstrap.container``, so
swapping OpenAI for Ollama or Qdrant for another store never touches the core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from app.domain.models import (
    ChatMessage,
    Chunk,
    ImageAsset,
    IngestionJob,
    ParsedDocument,
    RetrievedChunk,
)


class LLMPort(ABC):
    """Chat-completion style generative model (optionally vision capable)."""

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield the answer incrementally (text deltas).

        Default (non-streaming providers / test doubles): the whole ``generate``
        result as a single chunk. Adapters override it with real token streaming.
        """
        text = self.generate(messages, temperature=temperature, max_tokens=max_tokens)
        if text:
            yield text

    @abstractmethod
    def describe_image(self, image: bytes, *, context: str = "") -> str:
        """Return a short technical description of an image (diagram, chart...)."""

    @property
    @abstractmethod
    def supports_vision(self) -> bool: ...


class EmbeddingPort(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...


class VectorStorePort(ABC):
    """Stores chunks and performs hybrid (dense + keyword/sparse) retrieval."""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...


class DocumentParserPort(ABC):
    @abstractmethod
    def parse(self, file_path: str, *, document_id: str, filename: str) -> ParsedDocument:
        """Extract text blocks, tables and images (with bounding boxes).

        Extracted image bytes are persisted through ``BlobStoragePort`` by the
        adapter; ``ParsedDocument.images`` only carries metadata.
        """


class ChunkerPort(ABC):
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...


class BlobStoragePort(ABC):
    @abstractmethod
    def save_upload(self, document_id: str, filename: str, data: bytes) -> str:
        """Persist the original PDF and return its path/key."""

    @abstractmethod
    def save_image(self, asset: ImageAsset, data: bytes) -> None: ...

    @abstractmethod
    def get_image(self, document_id: str, image_id: str) -> tuple[bytes, str]:
        """Return (bytes, mime_type). Raises ``NotFoundError``."""


class JobRepositoryPort(ABC):
    @abstractmethod
    def create(self, job: IngestionJob) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> IngestionJob:
        """Raises ``NotFoundError``."""

    @abstractmethod
    def update(self, job: IngestionJob) -> None: ...

    @abstractmethod
    def list(self, *, limit: int = 100) -> list[IngestionJob]: ...


class TaskQueuePort(ABC):
    @abstractmethod
    def enqueue_ingestion(self, job_id: str) -> None: ...
