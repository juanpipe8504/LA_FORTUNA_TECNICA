"""OpenAI embeddings adapter (batched)."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import openai
from openai import OpenAI

from app.domain.exceptions import ExternalServiceError
from app.domain.ports import EmbeddingPort
from app.infrastructure.llm.openai_errors import map_openai_error

KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings(EmbeddingPort):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        *,
        batch_size: int = 256,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        self._batch_size = max(1, batch_size)
        self._client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
        self._dimension: int | None = KNOWN_DIMENSIONS.get(model)
        self._lock = threading.Lock()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=texts)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, "embeddings") from exc
        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(texts):
            raise ExternalServiceError(
                f"OpenAI embeddings returned {len(data)} vectors for {len(texts)} inputs"
            )
        return [list(item.embedding) for item in data]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        items = [t if t.strip() else " " for t in texts]  # the API rejects empty strings
        for start in range(0, len(items), self._batch_size):
            vectors.extend(self._embed_batch(items[start : start + self._batch_size]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text if text.strip() else " "])[0]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            with self._lock:
                if self._dimension is None:
                    self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension
