"""Document upload, listing and extracted-image serving."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Path, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import BlobStorageDep, ListDocumentsDep, SettingsDep, SubmitDocumentDep
from app.api.errors import HTTP_413_TOO_LARGE, HTTP_422_UNPROCESSABLE, ApiError
from app.api.schemas import DocumentResponse, DocumentSubmittedResponse, ErrorResponse
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_CHUNK_BYTES = 1024 * 1024
_PDF_CONTENT_TYPES = frozenset({"application/pdf", "application/x-pdf"})
_GENERIC_CONTENT_TYPES = frozenset({"", "application/octet-stream", "binary/octet-stream"})
# Ids are used to build file paths: restrict them to a safe alphabet (no traversal).
_SAFE_ID = r"^[A-Za-z0-9_\-]{1,128}$"
IMAGE_CACHE_CONTROL = "public, max-age=31536000, immutable"


def _ensure_pdf(file: UploadFile) -> str:
    filename = (file.filename or "").strip()
    if not filename:
        raise ApiError(HTTP_422_UNPROCESSABLE, "validation_error", "A filename is required.")
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    looks_like_pdf = content_type in _PDF_CONTENT_TYPES or (
        content_type in _GENERIC_CONTENT_TYPES and filename.lower().endswith(".pdf")
    )
    if not looks_like_pdf:
        raise ApiError(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            "Only PDF files are supported.",
        )
    return filename


async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read the upload in chunks, aborting as soon as it exceeds ``max_bytes``."""
    buffer = bytearray()
    while chunk := await file.read(_READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise ApiError(
                HTTP_413_TOO_LARGE,
                "payload_too_large",
                f"File exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB.",
            )
    return bytes(buffer)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentSubmittedResponse,
    summary="Upload a PDF; ingestion runs asynchronously",
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File(description="PDF document")],
    use_case: SubmitDocumentDep,
    settings: SettingsDep,
) -> DocumentSubmittedResponse:
    try:
        filename = _ensure_pdf(file)
        data = await _read_limited(file, settings.max_upload_mb * 1024 * 1024)
    finally:
        await file.close()
    # The use case does blocking I/O (disk, Redis, broker): keep it off the event loop.
    job = await run_in_threadpool(use_case.execute, filename, data)
    response.headers["Location"] = f"{settings.api_prefix.rstrip('/')}/jobs/{job.job_id}"
    logger.info(
        "document submitted",
        extra={"job_id": job.job_id, "document_id": job.document_id, "size_bytes": len(data)},
    )
    return DocumentSubmittedResponse.from_domain(job)


@router.get("", response_model=list[DocumentResponse], summary="List ingested documents")
def list_documents(use_case: ListDocumentsDep) -> list[DocumentResponse]:
    return [DocumentResponse.from_domain(job) for job in use_case.execute()]


@router.get(
    "/{document_id}/images/{image_id}",
    summary="Get an image extracted from a document",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "Image bytes"},
        404: {"model": ErrorResponse},
    },
)
def get_image(
    storage: BlobStorageDep,
    document_id: Annotated[str, Path(pattern=_SAFE_ID)],
    image_id: Annotated[str, Path(pattern=_SAFE_ID)],
) -> Response:
    data, mime_type = storage.get_image(document_id, image_id)
    return Response(
        content=data,
        media_type=mime_type,
        headers={"Cache-Control": IMAGE_CACHE_CONTROL},
    )
