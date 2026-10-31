"""Domain entities and value objects.

Pure Python (dataclasses + enums): no framework or third-party imports, so the
core can be reused by the API, the worker and the tests without coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Spatial primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Rectangle in PDF points, origin at the top-left of the page."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def union(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def distance_to(self, other: BoundingBox) -> float:
        """Shortest euclidean gap between two rectangles (0 if they overlap)."""
        dx = max(0.0, max(self.x0, other.x0) - min(self.x1, other.x1))
        dy = max(0.0, max(self.y0, other.y0) - min(self.y1, other.y1))
        return (dx**2 + dy**2) ** 0.5

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values: list[float] | tuple[float, ...]) -> BoundingBox:
        x0, y0, x1, y1 = values
        return cls(float(x0), float(y0), float(x1), float(y1))


# --------------------------------------------------------------------------- #
# Parsed document (output of the PDF parser, input of the chunker)
# --------------------------------------------------------------------------- #
class ElementType(str, Enum):
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass(slots=True)
class PageElement:
    """A layout element on a page (text block, heading, table or image)."""

    element_type: ElementType
    page: int  # 1-based
    bbox: BoundingBox
    text: str = ""  # markdown for tables, caption/alt text for images
    image_id: str | None = None  # only for IMAGE elements
    font_size: float | None = None
    is_bold: bool = False


@dataclass(slots=True)
class ImageAsset:
    """An image extracted from a PDF and persisted in blob storage."""

    image_id: str
    document_id: str
    page: int
    bbox: BoundingBox
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0
    caption: str = ""  # figure caption found in the PDF and/or vision-LLM description


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    filename: str
    page_count: int
    elements: list[PageElement] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Chunks and retrieval
# --------------------------------------------------------------------------- #
class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"  # chunk whose content is an image caption/description


@dataclass(slots=True)
class ImageRef:
    """Pointer from a chunk to a spatially related image."""

    image_id: str
    page: int
    bbox: BoundingBox
    distance: float  # gap in PDF points between chunk bbox and image bbox
    caption: str = ""


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    bbox: BoundingBox | None  # bbox on page_start
    chunk_type: ChunkType = ChunkType.TEXT
    section_path: list[str] = field(default_factory=list)  # heading hierarchy
    images: list[ImageRef] = field(default_factory=list)  # sorted by distance asc


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


# --------------------------------------------------------------------------- #
# Conversation / answers
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str
    images: list[bytes] = field(default_factory=list)  # for multimodal LLM calls


@dataclass(slots=True)
class SourceCitation:
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    score: float
    snippet: str
    bbox: BoundingBox | None = None


@dataclass(slots=True)
class ImageCitation:
    image_id: str
    document_id: str
    document_name: str
    page: int
    caption: str
    bbox: BoundingBox
    related_chunk_id: str
    relevance: float


@dataclass(slots=True)
class Answer:
    text: str
    has_sufficient_context: bool
    sources: list[SourceCitation] = field(default_factory=list)
    images: list[ImageCitation] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Ingestion jobs
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class IngestionJob:
    job_id: str
    document_id: str
    filename: str
    file_path: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0..100
    message: str = ""
    error: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    image_count: int | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
