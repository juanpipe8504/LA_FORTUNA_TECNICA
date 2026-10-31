"""HTTP DTOs (Pydantic v2) and mappers from domain objects.

The API never serialises domain dataclasses directly: these DTOs are the
public contract, so domain refactors don't leak into the HTTP surface.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.models import (
    Answer,
    BoundingBox,
    ChatMessage,
    ImageCitation,
    IngestionJob,
    JobStatus,
    Role,
    SourceCitation,
)

MAX_QUESTION_CHARS = 2000
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_MESSAGE_CHARS = 8000
MAX_DOCUMENT_FILTERS = 50

BBox = list[float]


def bbox_to_list(bbox: BoundingBox | None) -> BBox | None:
    return bbox.to_list() if bbox is not None else None


def image_url(api_prefix: str, document_id: str, image_id: str) -> str:
    return f"{api_prefix.rstrip('/')}/documents/{document_id}/images/{image_id}"


# --------------------------------------------------------------------------- #
# Documents / jobs
# --------------------------------------------------------------------------- #
class DocumentSubmittedResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: JobStatus

    @classmethod
    def from_domain(cls, job: IngestionJob) -> DocumentSubmittedResponse:
        return cls(job_id=job.job_id, document_id=job.document_id, filename=job.filename, status=job.status)


class JobResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    message: str
    error: str | None
    page_count: int | None
    chunk_count: int | None
    image_count: int | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, job: IngestionJob) -> JobResponse:
        return cls(
            job_id=job.job_id,
            document_id=job.document_id,
            filename=job.filename,
            status=job.status,
            progress=max(0, min(100, job.progress)),
            message=job.message,
            error=job.error,
            page_count=job.page_count,
            chunk_count=job.chunk_count,
            image_count=job.image_count,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int | None
    chunk_count: int | None
    image_count: int | None
    created_at: datetime

    @classmethod
    def from_domain(cls, job: IngestionJob) -> DocumentResponse:
        return cls(
            document_id=job.document_id,
            filename=job.filename,
            page_count=job.page_count,
            chunk_count=job.chunk_count,
            image_count=job.image_count,
            created_at=job.created_at,
        )


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ChatRole(StrEnum):
    """Roles a client may send; ``system`` is reserved for the server."""

    USER = "user"
    ASSISTANT = "assistant"


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str = Field(min_length=1, max_length=MAX_HISTORY_MESSAGE_CHARS)

    def to_domain(self) -> ChatMessage:
        return ChatMessage(role=Role(self.role.value), content=self.content)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "question": "What is the maximum operating temperature?",
                    "history": [],
                    "document_ids": None,
                }
            ]
        },
    )

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    document_ids: list[str] | None = Field(default=None, max_length=MAX_DOCUMENT_FILTERS)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    def history_to_domain(self) -> list[ChatMessage]:
        return [turn.to_domain() for turn in self.history]


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    score: float
    snippet: str
    bbox: BBox | None

    @classmethod
    def from_domain(cls, source: SourceCitation) -> SourceResponse:
        return cls(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            document_name=source.document_name,
            page=source.page,
            score=source.score,
            snippet=source.snippet,
            bbox=bbox_to_list(source.bbox),
        )


class ImageResponse(BaseModel):
    image_id: str
    document_id: str
    document_name: str
    page: int
    caption: str
    bbox: BBox
    related_chunk_id: str
    relevance: float
    url: str

    @classmethod
    def from_domain(cls, image: ImageCitation, api_prefix: str) -> ImageResponse:
        return cls(
            image_id=image.image_id,
            document_id=image.document_id,
            document_name=image.document_name,
            page=image.page,
            caption=image.caption,
            bbox=image.bbox.to_list(),
            related_chunk_id=image.related_chunk_id,
            relevance=image.relevance,
            url=image_url(api_prefix, image.document_id, image.image_id),
        )


class ChatResponse(BaseModel):
    answer: str
    has_sufficient_context: bool
    sources: list[SourceResponse]
    images: list[ImageResponse]

    @classmethod
    def from_domain(cls, answer: Answer, api_prefix: str) -> ChatResponse:
        return cls(
            answer=answer.text,
            has_sufficient_context=answer.has_sufficient_context,
            sources=[SourceResponse.from_domain(s) for s in answer.sources],
            images=[ImageResponse.from_domain(i, api_prefix) for i in answer.images],
        )


# --------------------------------------------------------------------------- #
# Health / errors
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "unavailable"
    checks: dict[str, bool]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None


class ErrorResponse(BaseModel):
    error: ErrorBody
