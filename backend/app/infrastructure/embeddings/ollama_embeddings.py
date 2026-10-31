"""Ollama embeddings adapter (``POST /api/embed``)."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import httpx

from app.domain.exceptions import ExternalServiceError
from app.domain.ports import EmbeddingPort
from app.infrastructure.llm.ollama_http import OllamaHttpClient


class OllamaEmbeddings(EmbeddingPort):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        *,
        batch_size: int = 64,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._batch_size = max(1, batch_size)
        self._http = OllamaHttpClient(base_url, timeout, client=client)
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        data = self._http.post_json("/api/embed", {"model": self._model, "input": texts})
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ExternalServiceError(
                f"Ollama /api/embed returned an unexpected payload for {len(texts)} inputs"
            )
        return [[float(x) for x in vector] for vector in embeddings]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        vectors: list[list[float]] = []
        for start in range(0, len(items), self._batch_size):
            vectors.extend(self._embed_batch(items[start : start + self._batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    @property
    def dimension(self) -> int:
        """Unknown up-front for arbitrary Ollama models: probe once and cache."""
        if self._dimension is None:
            with self._lock:
                if self._dimension is None:
                    self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension
