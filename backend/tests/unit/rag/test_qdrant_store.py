from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.domain.exceptions import ExternalServiceError, TransientError
from app.domain.models import ChunkType
from app.infrastructure.vectorstore.qdrant_store import (
    FastEmbedSparseEncoder,
    QdrantVectorStore,
    chunk_to_payload,
    payload_to_chunk,
    point_id,
)


@pytest.fixture
def store(sparse_encoder) -> QdrantVectorStore:
    warnings.filterwarnings("ignore", message="Payload indexes have no effect")
    return QdrantVectorStore(QdrantClient(":memory:"), "test", sparse_encoder=sparse_encoder)


@pytest.fixture
def indexed(store, make_chunk, make_image_ref) -> QdrantVectorStore:
    store.ensure_collection(3)
    chunks = [
        make_chunk("a", "hydraulic pump maximum pressure 250 bar", document_id="d1"),
        make_chunk("b", "valve tightening torque 45 Nm", document_id="d1",
                   images=[make_image_ref("d1_p1_i0", 12.5, "Figure 2: valve")]),
        make_chunk("c", "electric motor nominal speed 1450 rpm", document_id="d2"),
    ]
    store.upsert(chunks, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return store


def test_payload_round_trip(make_chunk, make_image_ref) -> None:
    original = make_chunk(
        "c1", "text", chunk_type=ChunkType.TABLE, section_path=["1 Intro", "1.2 Specs"],
        images=[make_image_ref("img", 3.0, "Fig. 1")],
    )
    payload = chunk_to_payload(original)
    assert payload["bbox"] == [10.0, 20.0, 300.0, 80.0]
    assert payload["images"][0]["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert payload["chunk_type"] == "table"
    assert payload_to_chunk(payload) == original


def test_payload_round_trip_without_bbox(make_chunk) -> None:
    original = make_chunk("c2")
    original.bbox = None
    assert payload_to_chunk(chunk_to_payload(original)) == original


def test_point_id_is_deterministic_uuid() -> None:
    assert point_id("doc_c1") == point_id("doc_c1") != point_id("doc_c2")


def test_hybrid_search_fuses_dense_and_sparse(indexed) -> None:
    results = indexed.hybrid_search("electric motor speed", [0, 0, 1], top_k=2)
    assert results[0].chunk.chunk_id == "c"
    assert results[0].score >= results[-1].score
    assert len(results) == 2


def test_sparse_branch_finds_exact_keywords_when_dense_disagrees(indexed) -> None:
    # Dense vector points at "a" but the keyword query only matches "b" -> both appear.
    ids = [r.chunk.chunk_id for r in indexed.hybrid_search("torque Nm", [1, 0, 0], top_k=3)]
    assert {"a", "b"} <= set(ids)


def test_hybrid_search_document_filter_and_payload(indexed) -> None:
    results = indexed.hybrid_search("motor", [0, 0, 1], top_k=5, document_ids=["d1"])
    assert {r.chunk.document_id for r in results} == {"d1"}
    b = next(r.chunk for r in results if r.chunk.chunk_id == "b")
    assert b.images[0].image_id == "d1_p1_i0"
    assert b.images[0].distance == 12.5


def test_upsert_is_idempotent(indexed, make_chunk) -> None:
    indexed.upsert([make_chunk("a", "replaced text", document_id="d1")], [[1, 0, 0]])
    results = indexed.hybrid_search("x", [1, 0, 0], top_k=10)
    assert len(results) == 3
    assert next(r for r in results if r.chunk.chunk_id == "a").chunk.text == "replaced text"


def test_delete_document(indexed) -> None:
    indexed.delete_document("d1")
    ids = {r.chunk.chunk_id for r in indexed.hybrid_search("pump valve motor", [1, 1, 1], top_k=10)}
    assert ids == {"c"}


def test_search_before_collection_exists_returns_empty(store) -> None:
    assert store.hybrid_search("anything", [0, 0, 1], top_k=3) == []
    store.delete_document("d1")  # no-op, no error


def test_stopword_only_query_uses_dense_only(indexed, sparse_encoder) -> None:
    sparse_encoder.encode_query = lambda text: ([], [])
    assert indexed.hybrid_search("", [0, 1, 0], top_k=1)[0].chunk.chunk_id == "b"


def test_ensure_collection_is_idempotent_and_rejects_dimension_mismatch(store) -> None:
    store.ensure_collection(3)
    store.ensure_collection(3)
    with pytest.raises(ExternalServiceError, match="dimension 3"):
        store.ensure_collection(384)


def test_upsert_length_mismatch(store, make_chunk) -> None:
    with pytest.raises(ValueError):
        store.upsert([make_chunk("a")], [])


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (UnexpectedResponse(503, "Service Unavailable", b"", httpx.Headers()), TransientError),
        (ResponseHandlingException(httpx.ConnectError("refused")), TransientError),
        (UnexpectedResponse(400, "Bad Request", b"", httpx.Headers()), ExternalServiceError),
        (RuntimeError("boom"), ExternalServiceError),
    ],
)
def test_client_errors_are_wrapped(sparse_encoder, error, expected) -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = error
    store = QdrantVectorStore(client, "test", sparse_encoder=sparse_encoder)

    with pytest.raises(expected) as info:
        store.hybrid_search("q", [0.0], top_k=2)
    if expected is ExternalServiceError:
        assert not isinstance(info.value, TransientError)


def test_query_uses_rrf_fusion_with_two_prefetches(sparse_encoder) -> None:
    client = MagicMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(points=[])
    store = QdrantVectorStore(client, "test", sparse_encoder=sparse_encoder)

    store.hybrid_search("pump pressure", [0.1, 0.2], top_k=5, document_ids=["d1", "d2"])

    kwargs = client.query_points.call_args.kwargs
    dense, sparse = kwargs["prefetch"]
    assert (dense.using, sparse.using) == ("dense", "bm25")
    assert dense.limit == sparse.limit == 20
    assert kwargs["query"].fusion.value == "rrf"
    assert kwargs["query_filter"].must[0].match.any == ["d1", "d2"]
    assert kwargs["limit"] == 5


# --------------------------------------------------------------------------- BM25 language
class _FakeSparseTextEmbedding:
    created: list[dict] = []

    def __init__(self, **kwargs) -> None:
        type(self).created.append(kwargs)

    def embed(self, texts):
        return [MagicMock(indices=[1, 2], values=[1.0, 2.0]) for _ in texts]

    def query_embed(self, text):
        return iter([MagicMock(indices=[3], values=[1.0])])


@pytest.fixture
def fake_fastembed(monkeypatch):
    import fastembed

    _FakeSparseTextEmbedding.created = []
    monkeypatch.setattr(fastembed, "SparseTextEmbedding", _FakeSparseTextEmbedding)
    return _FakeSparseTextEmbedding


def test_sparse_encoder_defaults_to_spanish_and_loads_lazily(fake_fastembed) -> None:
    encoder = FastEmbedSparseEncoder()
    assert fake_fastembed.created == []  # nothing loaded at construction

    assert encoder.encode_query("relubricar rodamientos") == ([3], [1.0])
    assert encoder.encode_documents(["a", "b"]) == [([1, 2], [1.0, 2.0])] * 2
    assert fake_fastembed.created == [{"model_name": "Qdrant/bm25", "language": "spanish"}]


def test_sparse_encoder_passes_configured_language(fake_fastembed) -> None:
    FastEmbedSparseEncoder(language="english").encode_query("q")

    assert fake_fastembed.created == [{"model_name": "Qdrant/bm25", "language": "english"}]


def test_store_builds_sparse_encoder_with_bm25_language(fake_fastembed) -> None:
    client = MagicMock()
    client.collection_exists.return_value = False
    store = QdrantVectorStore(client, "test", bm25_language="english")

    store.hybrid_search("q", [0.0], top_k=1)  # encodes the query, collection missing -> []

    assert fake_fastembed.created == [{"model_name": "Qdrant/bm25", "language": "english"}]


def test_store_positional_sparse_model_is_backward_compatible(fake_fastembed) -> None:
    client = MagicMock()
    client.collection_exists.return_value = False
    QdrantVectorStore(client, "test", "Qdrant/bm25").hybrid_search("q", [0.0], top_k=1)

    assert fake_fastembed.created == [{"model_name": "Qdrant/bm25", "language": "spanish"}]
