from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from app.domain.models import BoundingBox, Chunk, ChunkType, ImageRef


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FakeSparseEncoder:
    """Deterministic offline replacement for fastembed's BM25 (term-frequency vectors)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _encode(self, text: str) -> tuple[list[int], list[float]]:
        counts: dict[int, float] = {}
        for word in text.lower().split():
            word = word.strip(".,;:?!¿¡")
            if not word:
                continue
            idx = self.vocab.setdefault(word, len(self.vocab))
            counts[idx] = counts.get(idx, 0.0) + 1.0
        indices = sorted(counts)
        return indices, [counts[i] for i in indices]

    def encode_documents(self, texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
        return [self._encode(t) for t in texts]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._encode(text)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_sleep() -> FakeSleep:
    return FakeSleep()


@pytest.fixture
def sparse_encoder() -> FakeSparseEncoder:
    return FakeSparseEncoder()


def image_ref(image_id: str, distance: float = 0.0, caption: str = "", page: int = 1) -> ImageRef:
    return ImageRef(image_id=image_id, page=page, bbox=BoundingBox(0, 0, 100, 100), distance=distance,
                    caption=caption)


def chunk(
    chunk_id: str,
    text: str = "some text",
    *,
    document_id: str = "doc1",
    chunk_type: ChunkType = ChunkType.TEXT,
    images: list[ImageRef] | None = None,
    page: int = 1,
    section_path: list[str] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        text=text,
        page_start=page,
        page_end=page,
        bbox=BoundingBox(10, 20, 300, 80),
        chunk_type=chunk_type,
        section_path=section_path or [],
        images=images or [],
    )


@pytest.fixture
def make_chunk() -> Callable[..., Chunk]:
    return chunk


@pytest.fixture
def make_image_ref() -> Callable[..., ImageRef]:
    return image_ref
