from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from app.domain.exceptions import ExternalServiceError, TransientError
from app.infrastructure.embeddings.fastembed_embeddings import FastEmbedEmbeddings
from app.infrastructure.embeddings.ollama_embeddings import OllamaEmbeddings
from app.infrastructure.embeddings.openai_embeddings import OpenAIEmbeddings


# --------------------------------------------------------------------------- OpenAI
def _openai_response(n: int, dim: int = 3, reverse: bool = False):
    items = [SimpleNamespace(index=i, embedding=[float(i)] * dim) for i in range(n)]
    return SimpleNamespace(data=list(reversed(items)) if reverse else items)


def test_openai_embeddings_batches_and_orders_by_index() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = lambda model, input: _openai_response(len(input), reverse=True)
    emb = OpenAIEmbeddings("sk", "text-embedding-3-small", batch_size=2, client=client)

    vectors = emb.embed_documents(["a", "b", "c"])

    assert vectors == [[0.0] * 3, [1.0] * 3, [0.0] * 3]
    assert client.embeddings.create.call_count == 2
    assert emb.dimension == 1536  # known table, no API call
    assert client.embeddings.create.call_count == 2


def test_openai_embeddings_unknown_model_probes_dimension_once() -> None:
    client = MagicMock()
    client.embeddings.create.side_effect = lambda model, input: _openai_response(len(input), dim=7)
    emb = OpenAIEmbeddings("sk", "custom-embedder", client=client)
    assert emb.dimension == 7
    assert emb.dimension == 7
    assert client.embeddings.create.call_count == 1


def test_openai_embeddings_error_mapping() -> None:
    client = MagicMock()
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    client.embeddings.create.side_effect = openai.APITimeoutError(request=request)
    emb = OpenAIEmbeddings("sk", "text-embedding-3-small", client=client)
    with pytest.raises(TransientError):
        emb.embed_query("q")


def test_openai_embeddings_count_mismatch() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = _openai_response(1)
    emb = OpenAIEmbeddings("sk", "text-embedding-3-small", client=client)
    with pytest.raises(ExternalServiceError):
        emb.embed_documents(["a", "b"])


# --------------------------------------------------------------------------- Ollama
def _ollama(handler, **kwargs) -> OllamaEmbeddings:
    client = httpx.Client(base_url="http://ollama:11434", transport=httpx.MockTransport(handler))
    return OllamaEmbeddings("http://ollama:11434", "nomic-embed-text", client=client, **kwargs)


def test_ollama_embeddings_payload_batches_and_dimension_probe() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3, 0.4]] * len(body["input"])})

    emb = _ollama(handler, batch_size=2)
    assert len(emb.embed_documents(["a", "b", "c"])) == 3
    assert [b["input"] for b in bodies] == [["a", "b"], ["c"]]
    assert bodies[0]["model"] == "nomic-embed-text"

    assert emb.dimension == 4
    assert emb.dimension == 4
    assert len(bodies) == 3  # probed exactly once


def test_ollama_embeddings_errors() -> None:
    with pytest.raises(TransientError):
        _ollama(lambda r: httpx.Response(503)).embed_query("q")
    with pytest.raises(ExternalServiceError):
        _ollama(lambda r: httpx.Response(200, json={"embeddings": []})).embed_query("q")


# --------------------------------------------------------------------------- FastEmbed
class FakeTextEmbedding:
    instances = 0

    def __init__(self, model_name: str) -> None:
        FakeTextEmbedding.instances += 1
        self.model_name = model_name
        self.query_calls = 0

    def embed(self, texts):
        for text in texts:
            yield [float(len(text)), 0.0]

    def query_embed(self, texts):
        self.query_calls += 1
        yield from self.embed(texts)


def test_fastembed_lazy_load_and_query_embed() -> None:
    FakeTextEmbedding.instances = 0
    emb = FastEmbedEmbeddings("my/custom-model", model_factory=FakeTextEmbedding)
    assert FakeTextEmbedding.instances == 0  # lazy

    assert emb.embed_documents(["abc", "de"]) == [[3.0, 0.0], [2.0, 0.0]]
    assert emb.embed_query("hello") == [5.0, 0.0]
    assert emb.embed_documents([]) == []
    assert emb.dimension == 2  # unknown model -> probe
    assert FakeTextEmbedding.instances == 1


def test_fastembed_known_dimension_without_loading() -> None:
    FakeTextEmbedding.instances = 0
    emb = FastEmbedEmbeddings("BAAI/bge-small-en-v1.5", model_factory=FakeTextEmbedding)
    assert emb.dimension == 384
    assert FakeTextEmbedding.instances == 0


def test_fastembed_load_failure_is_external_error() -> None:
    def broken(name: str):
        raise OSError("no network")

    with pytest.raises(ExternalServiceError, match="no network"):
        FastEmbedEmbeddings("x", model_factory=broken).embed_query("q")
