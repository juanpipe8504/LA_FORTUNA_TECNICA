"""Composition-root tests.

Adapter/use-case modules are replaced by recording fakes in ``sys.modules`` so
the wiring logic is verified deterministically, offline, and independently of
the adapters' own behaviour (covered by their own test suites).
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest

from app.bootstrap import container as container_module
from app.bootstrap.container import ConfigurationError, build_container
from app.core.config import Settings

OPENAI_LLM = "app.infrastructure.llm.openai_llm"
OPENAI_EMB = "app.infrastructure.embeddings.openai_embeddings"


class _Recorder:
    """Stand-in for any adapter/use case: remembers its constructor arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


def _fake_class(name: str) -> type:
    return type(name, (_Recorder,), {})


FAKE_MODULES: dict[str, list[str]] = {
    "app.infrastructure.resilience.retry": ["RetryPolicy"],
    "app.infrastructure.resilience.circuit_breaker": ["CircuitBreaker"],
    "app.infrastructure.resilience.resilient": ["ResilientLLM", "ResilientEmbeddings"],
    OPENAI_LLM: ["OpenAILLM"],
    "app.infrastructure.llm.ollama_llm": ["OllamaLLM"],
    OPENAI_EMB: ["OpenAIEmbeddings"],
    "app.infrastructure.embeddings.ollama_embeddings": ["OllamaEmbeddings"],
    "app.infrastructure.embeddings.fastembed_embeddings": ["FastEmbedEmbeddings"],
    "app.infrastructure.vectorstore.qdrant_store": ["QdrantVectorStore"],
    "app.infrastructure.storage.local_blob_storage": ["LocalBlobStorage"],
    "app.infrastructure.jobs.redis_job_repository": ["RedisJobRepository"],
    "app.infrastructure.jobs.in_memory_job_repository": ["InMemoryJobRepository"],
    "app.infrastructure.queue.celery_task_queue": ["CeleryTaskQueue"],
    "app.infrastructure.parsing.pymupdf_parser": ["PyMuPDFParser"],
    "app.infrastructure.chunking.layout_chunker": ["LayoutAwareChunker"],
    "app.application.ingestion": ["SubmitDocumentUseCase", "ProcessDocumentUseCase"],
    "app.application.documents": ["ListDocumentsUseCase"],
    "app.application.query": ["AnswerQuestionUseCase"],
}


@pytest.fixture
def fake_modules(monkeypatch) -> dict[str, types.ModuleType]:
    modules: dict[str, types.ModuleType] = {}
    for module_name, class_names in FAKE_MODULES.items():
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, _fake_class(class_name))
        monkeypatch.setitem(sys.modules, module_name, module)
        modules[module_name] = module
    return modules


@pytest.fixture
def no_network_clients(monkeypatch) -> dict[str, Any]:
    """Avoid even constructing real Redis/Qdrant clients."""
    clients = {"redis": _Recorder(), "qdrant": _Recorder()}
    clients["redis"].close = lambda: None  # type: ignore[attr-defined]
    clients["qdrant"].close = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(container_module, "_build_redis_client", lambda s: clients["redis"])
    monkeypatch.setattr(container_module, "_build_qdrant_client", lambda s: clients["qdrant"])
    return clients


def _settings(tmp_path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        _env_file=None,
        app_env="development",
        data_dir=str(tmp_path / "data"),
        llm_provider="ollama",
        embedding_provider="fastembed",
        openai_api_key=None,
    )
    values.update(overrides)
    return Settings(**values)


def _unwrap(resilient: Any) -> Any:
    return resilient.args[0]


def test_ollama_and_fastembed_never_import_openai(tmp_path, fake_modules, no_network_clients, monkeypatch):
    # Make any import of the OpenAI adapters fail loudly.
    monkeypatch.setitem(sys.modules, OPENAI_LLM, None)
    monkeypatch.setitem(sys.modules, OPENAI_EMB, None)
    settings = _settings(tmp_path, ollama_base_url="http://ollama:11434", ollama_chat_model="llava")

    c = build_container(settings)

    assert type(c.llm).__name__ == "ResilientLLM"
    inner_llm = _unwrap(c.llm)
    assert type(inner_llm).__name__ == "OllamaLLM"
    assert inner_llm.kwargs == {"base_url": "http://ollama:11434", "model": "llava", "timeout": 60.0}
    assert type(_unwrap(c.embeddings)).__name__ == "FastEmbedEmbeddings"
    assert _unwrap(c.embeddings).kwargs == {"model_name": settings.fastembed_model}


def test_separate_breakers_and_retry_policy(tmp_path, fake_modules, no_network_clients):
    settings = _settings(
        tmp_path,
        retry_max_attempts=7,
        retry_base_delay_seconds=0.1,
        retry_max_delay_seconds=2.0,
        circuit_failure_threshold=3,
        circuit_recovery_seconds=9.0,
    )

    c = build_container(settings)

    llm_breaker, emb_breaker = c.llm.kwargs["breaker"], c.embeddings.kwargs["breaker"]
    assert llm_breaker is not emb_breaker
    assert llm_breaker.kwargs == {"name": "llm", "failure_threshold": 3, "recovery_timeout": 9.0}
    assert emb_breaker.kwargs["name"] == "embeddings"
    assert c.llm.kwargs["policy"].kwargs == {"max_attempts": 7, "base_delay": 0.1, "max_delay": 2.0}


def test_openai_selection_passes_secret_value(tmp_path, fake_modules, no_network_clients):
    settings = _settings(
        tmp_path,
        llm_provider="openai",
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_chat_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
    )

    c = build_container(settings)

    llm, emb = _unwrap(c.llm), _unwrap(c.embeddings)
    assert type(llm).__name__ == "OpenAILLM"
    assert llm.kwargs["api_key"] == "sk-test"
    assert llm.kwargs["model"] == "gpt-4o-mini"
    assert type(emb).__name__ == "OpenAIEmbeddings"
    assert emb.kwargs["model"] == "text-embedding-3-small"


def test_ollama_embeddings_selection(tmp_path, fake_modules, no_network_clients):
    c = build_container(_settings(tmp_path, embedding_provider="ollama", ollama_embedding_model="nomic"))

    emb = _unwrap(c.embeddings)
    assert type(emb).__name__ == "OllamaEmbeddings"
    assert emb.kwargs["model"] == "nomic"


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_provider": "openai"},
        {"embedding_provider": "openai"},
        {"llm_provider": "openai", "openai_api_key": ""},
    ],
    ids=["llm", "embeddings", "empty-key"],
)
def test_openai_without_key_fails_fast(tmp_path, fake_modules, no_network_clients, overrides):
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_container(_settings(tmp_path, **overrides))


def test_use_cases_are_wired_with_shared_ports(tmp_path, fake_modules, no_network_clients):
    settings = _settings(tmp_path, max_upload_mb=7, retrieval_top_k=9, describe_images=False)

    c = build_container(settings)

    submit = c.submit_document.kwargs
    assert submit["blob_storage"] is c.blob_storage
    assert submit["job_repository"] is c.job_repository
    assert submit["task_queue"] is c.task_queue
    assert submit["max_upload_mb"] == 7

    process = c.process_document.kwargs
    assert process["embeddings"] is c.embeddings
    assert process["vector_store"] is c.vector_store
    assert process["llm"] is c.llm
    assert process["parser"] is c.parser
    assert process["chunker"] is c.chunker
    assert process["describe_images"] is False

    answer = c.answer_question.kwargs
    assert answer["embeddings"] is c.embeddings
    assert answer["llm"] is c.llm
    assert answer["top_k"] == 9
    assert answer["max_tokens"] == settings.llm_max_tokens
    assert c.list_documents.kwargs == {"job_repository": c.job_repository}

    assert c.parser.kwargs["blob_storage"] is c.blob_storage
    assert c.vector_store.kwargs == {
        "client": no_network_clients["qdrant"],
        "collection": settings.qdrant_collection,
    }


def test_job_repository_redis_in_dev_in_memory_in_test(tmp_path, fake_modules, no_network_clients):
    dev = build_container(_settings(tmp_path, app_env="development"))
    test = build_container(_settings(tmp_path, app_env="test"))

    assert type(dev.job_repository).__name__ == "RedisJobRepository"
    assert dev.job_repository.kwargs["redis_client"] is no_network_clients["redis"]
    assert type(test.job_repository).__name__ == "InMemoryJobRepository"


def test_creates_data_directories(tmp_path, fake_modules, no_network_clients):
    settings = _settings(tmp_path)

    build_container(settings)

    assert os.path.isdir(settings.uploads_dir)
    assert os.path.isdir(settings.images_dir)


def test_health_checks_use_clients(tmp_path, fake_modules, no_network_clients):
    no_network_clients["redis"].ping = lambda: True
    no_network_clients["qdrant"].get_collections = lambda: []

    c = build_container(_settings(tmp_path))

    assert set(c.health_checks) == {"redis", "qdrant"}
    assert all(check() for check in c.health_checks.values())


def test_real_clients_are_built_without_network(tmp_path):
    """Redis/Qdrant clients connect lazily: constructing them must not hit the network."""
    settings = _settings(tmp_path, redis_url="redis://127.0.0.1:1/0", qdrant_url="http://127.0.0.1:1")

    redis_client = container_module._build_redis_client(settings)
    qdrant_client = container_module._build_qdrant_client(settings)

    assert redis_client is not None and qdrant_client is not None
    redis_client.close()
    qdrant_client.close()


def test_close_swallows_errors(tmp_path, fake_modules, no_network_clients):
    c = build_container(_settings(tmp_path))

    def boom() -> None:
        raise RuntimeError("already closed")

    c.closers.insert(0, boom)
    c.close()  # must not raise


def test_real_resilient_llm_wraps_real_ollama_when_available(tmp_path):
    """With workstream B's real modules, the factory wiring matches their signatures."""
    pytest.importorskip("app.infrastructure.resilience.resilient")
    from app.infrastructure.resilience.resilient import ResilientEmbeddings, ResilientLLM

    settings = _settings(tmp_path)

    assert isinstance(container_module._build_llm(settings), ResilientLLM)
    assert isinstance(container_module._build_embeddings(settings), ResilientEmbeddings)


def test_real_ollama_adapter_when_available(tmp_path):
    """Once workstream B lands, verify the real adapter accepts the wired kwargs."""
    pytest.importorskip("app.infrastructure.llm.ollama_llm")
    from app.infrastructure.llm.ollama_llm import OllamaLLM

    llm = container_module._build_raw_llm(_settings(tmp_path))

    assert isinstance(llm, OllamaLLM)
