"""Qdrant vector store with hybrid retrieval (dense + BM25 sparse, fused with RRF).

Collection layout:
* named dense vector ``dense`` (cosine) produced by the configured EmbeddingPort;
* named sparse vector ``bm25`` with the ``IDF`` modifier: Qdrant computes IDF
  server-side, the client only sends term frequencies (fastembed ``Qdrant/bm25``,
  with a language-specific Snowball stemmer + stop-word list, ``spanish`` by
  default). The language determines the token ids, so documents and queries must
  be encoded with the same one: changing it requires re-ingesting;
* payload = the full serialized chunk (+ keyword index on ``document_id``).

Hybrid search uses the Query API: two prefetches (dense / sparse, ``top_k * 4``
candidates each) fused with Reciprocal Rank Fusion. RRF is rank-based, so the
incomparable cosine and BM25 score scales never need normalising. Note the
returned scores are RRF scores (roughly ``sum 1/(k + rank)``), not cosine.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.logging import get_logger
from app.domain.exceptions import DomainError, ExternalServiceError
from app.domain.models import BoundingBox, Chunk, ChunkType, ImageRef, RetrievedChunk
from app.domain.ports import VectorStorePort
from app.infrastructure.resilience.retry import transient_error

logger = get_logger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
_POINT_NAMESPACE = uuid.UUID("6f1c9a52-6d3e-4b8e-9d0a-2f4b1e7c3a10")
_UPSERT_BATCH = 128

T = TypeVar("T")
SparseTuple = tuple[list[int], list[float]]


# --------------------------------------------------------------------------- #
# Sparse encoding
# --------------------------------------------------------------------------- #
class SparseEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> list[SparseTuple]: ...

    def encode_query(self, text: str) -> SparseTuple: ...


DEFAULT_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_BM25_LANGUAGE = "spanish"


class FastEmbedSparseEncoder:
    """BM25 term-frequency encoder from fastembed, loaded lazily (thread-safe).

    ``language`` selects fastembed's Snowball stemmer and stop-word list
    (e.g. ``spanish`` folds plurals/inflections such as "rodamiento"/"rodamientos";
    verb/noun pairs like "relubricar"/"relubricación" still stem differently).
    """

    def __init__(
        self, model_name: str = DEFAULT_SPARSE_MODEL, language: str = DEFAULT_BM25_LANGUAGE
    ) -> None:
        self._model_name = model_name
        self._language = language
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import SparseTextEmbedding

                    logger.info(
                        "sparse_model_loading",
                        extra={"model": self._model_name, "language": self._language},
                    )
                    self._model = SparseTextEmbedding(
                        model_name=self._model_name, language=self._language
                    )
        return self._model

    @staticmethod
    def _to_tuple(embedding: Any) -> SparseTuple:
        return [int(i) for i in embedding.indices], [float(v) for v in embedding.values]

    def encode_documents(self, texts: Sequence[str]) -> list[SparseTuple]:
        return [self._to_tuple(e) for e in self._get_model().embed(list(texts))]

    def encode_query(self, text: str) -> SparseTuple:
        return self._to_tuple(next(iter(self._get_model().query_embed(text))))


# --------------------------------------------------------------------------- #
# Chunk <-> payload (pure functions)
# --------------------------------------------------------------------------- #
def chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "bbox": chunk.bbox.to_list() if chunk.bbox else None,
        "chunk_type": chunk.chunk_type.value,
        "section_path": list(chunk.section_path),
        "images": [
            {
                "image_id": ref.image_id,
                "page": ref.page,
                "bbox": ref.bbox.to_list(),
                "distance": ref.distance,
                "caption": ref.caption,
            }
            for ref in chunk.images
        ],
    }


def payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    bbox = payload.get("bbox")
    page_start = int(payload.get("page_start", 1))
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        document_name=str(payload.get("document_name", "")),
        text=str(payload.get("text", "")),
        page_start=page_start,
        page_end=int(payload.get("page_end", page_start)),
        bbox=BoundingBox.from_list(bbox) if bbox else None,
        chunk_type=ChunkType(payload.get("chunk_type", ChunkType.TEXT.value)),
        section_path=[str(s) for s in payload.get("section_path", [])],
        images=[
            ImageRef(
                image_id=str(img["image_id"]),
                page=int(img["page"]),
                bbox=BoundingBox.from_list(img["bbox"]),
                distance=float(img.get("distance", 0.0)),
                caption=str(img.get("caption", "")),
            )
            for img in payload.get("images", [])
        ],
    )


def point_id(chunk_id: str) -> str:
    """Deterministic UUID so re-ingesting a document overwrites instead of duplicating."""
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def _document_filter(document_ids: Sequence[str] | None) -> models.Filter | None:
    if not document_ids:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=list(document_ids)))]
    )


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class QdrantVectorStore(VectorStorePort):
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        sparse_model: str = DEFAULT_SPARSE_MODEL,
        *,
        sparse_encoder: SparseEncoder | None = None,
        bm25_language: str = DEFAULT_BM25_LANGUAGE,
    ) -> None:
        self._client = client
        self._collection = collection
        self._sparse: SparseEncoder = sparse_encoder or FastEmbedSparseEncoder(
            sparse_model, language=bm25_language
        )
        self._ready = False
        self._lock = threading.Lock()

    # --------------------------------------------------------------- errors
    def _guard(self, operation: str, fn: Callable[[], T]) -> T:
        try:
            return fn()
        except DomainError:
            raise
        except UnexpectedResponse as exc:
            status = exc.status_code or 0
            message = f"Qdrant {operation}: HTTP {status}: {exc.reason_phrase}"
            if status == 429 or status >= 500:
                raise transient_error(message) from exc
            raise ExternalServiceError(message) from exc
        except (ResponseHandlingException, httpx.TransportError, ConnectionError, TimeoutError) as exc:
            raise transient_error(f"Qdrant {operation}: connection error: {exc}") from exc
        except Exception as exc:
            raise ExternalServiceError(f"Qdrant {operation} failed: {exc}") from exc

    # ----------------------------------------------------------- collection
    def ensure_collection(self, dimension: int) -> None:
        with self._lock:
            self._guard("ensure_collection", lambda: self._ensure(dimension))
            self._ready = True

    def _ensure(self, dimension: int) -> None:
        if self._client.collection_exists(self._collection):
            self._check_dimension(dimension)
            return
        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        except Exception:
            # Another worker may have created it concurrently: re-validate instead of failing.
            if not self._client.collection_exists(self._collection):
                raise
            self._check_dimension(dimension)
            return
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "qdrant_collection_created",
            extra={"collection": self._collection, "dimension": dimension},
        )

    def _check_dimension(self, dimension: int) -> None:
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        params = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if params is None:
            raise ExternalServiceError(
                f"Qdrant collection '{self._collection}' exists but has no '{DENSE_VECTOR}' vector; "
                "drop it or configure a different QDRANT_COLLECTION."
            )
        if params.size != dimension:
            raise ExternalServiceError(
                f"Qdrant collection '{self._collection}' has dimension {params.size} but the "
                f"embedding model produces {dimension}. Changing the embedding model requires a new "
                "collection (set QDRANT_COLLECTION) or re-indexing."
            )

    def _collection_ready(self) -> bool:
        if not self._ready:
            self._ready = self._client.collection_exists(self._collection)
        return self._ready

    # ---------------------------------------------------------------- write
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        sparse = self._guard("sparse_encode", lambda: self._sparse.encode_documents([c.text for c in chunks]))
        points = [
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR: [float(x) for x in vector],
                    SPARSE_VECTOR: models.SparseVector(indices=indices, values=values),
                },
                payload=chunk_to_payload(chunk),
            )
            for chunk, vector, (indices, values) in zip(chunks, vectors, sparse, strict=True)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            batch = points[start : start + _UPSERT_BATCH]
            self._guard(
                "upsert",
                lambda batch=batch: self._client.upsert(
                    collection_name=self._collection, points=batch, wait=True
                ),
            )

    def delete_document(self, document_id: str) -> None:
        def _delete() -> None:
            if not self._collection_ready():
                return
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=_document_filter([document_id])),
                wait=True,
            )

        self._guard("delete_document", _delete)

    # ----------------------------------------------------------------- read
    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        query_filter = _document_filter(document_ids)
        indices, values = self._guard("sparse_encode", lambda: self._sparse.encode_query(query_text))
        candidates = top_k * 4
        prefetch = [
            models.Prefetch(
                query=[float(x) for x in query_vector],
                using=DENSE_VECTOR,
                limit=candidates,
                filter=query_filter,
            )
        ]
        if indices:  # a query made only of stop-words has no BM25 terms
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR,
                    limit=candidates,
                    filter=query_filter,
                )
            )

        def _query() -> list[RetrievedChunk]:
            if not self._collection_ready():
                return []
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            return [
                RetrievedChunk(chunk=payload_to_chunk(point.payload or {}), score=float(point.score))
                for point in response.points
                if point.payload
            ]

        return self._guard("hybrid_search", _query)
