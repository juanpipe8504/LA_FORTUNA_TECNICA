"""Local CPU embeddings with ``fastembed`` (ONNX) - no API key, no network at query time."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from app.core.logging import get_logger
from app.domain.exceptions import ExternalServiceError
from app.domain.ports import EmbeddingPort

logger = get_logger(__name__)

ModelFactory = Callable[[str], Any]


def _default_factory(model_name: str) -> Any:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def known_dimension(model_name: str) -> int | None:
    """Look the dimension up in fastembed's model registry (no download)."""
    try:
        from fastembed import TextEmbedding

        for description in TextEmbedding.list_supported_models():
            if str(description.get("model", "")).lower() == model_name.lower():
                return int(description["dim"])
    except Exception:  # registry format is a fastembed implementation detail
        logger.debug("fastembed_registry_lookup_failed", exc_info=True)
    return None


class FastEmbedEmbeddings(EmbeddingPort):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._model_name = model_name
        self._factory = model_factory or _default_factory
        self._model: Any | None = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("fastembed_model_loading", extra={"model": self._model_name})
                    try:
                        self._model = self._factory(self._model_name)
                    except Exception as exc:
                        raise ExternalServiceError(
                            f"Could not load fastembed model '{self._model_name}': {exc}"
                        ) from exc
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        return [[float(x) for x in vector] for vector in model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        embed = getattr(model, "query_embed", None) or model.embed
        return [float(x) for x in next(iter(embed([text])))]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            dim = known_dimension(self._model_name)
            self._dimension = dim if dim is not None else len(self.embed_query("dimension probe"))
        return self._dimension
