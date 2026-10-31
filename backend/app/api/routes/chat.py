"""Question answering over the indexed documents."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import AnswerQuestionDep, SettingsDep
from app.api.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Ask a question; returns a grounded answer with sources and related images",
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def chat(body: ChatRequest, use_case: AnswerQuestionDep, settings: SettingsDep) -> ChatResponse:
    # Sync ``def``: the use case does blocking I/O, so FastAPI runs it in its threadpool.
    answer = use_case.execute(
        question=body.question,
        history=body.history_to_domain(),
        document_ids=body.document_ids,
    )
    logger.info(
        "question answered",
        extra={
            "has_sufficient_context": answer.has_sufficient_context,
            "source_count": len(answer.sources),
            "image_count": len(answer.images),
        },
    )
    return ChatResponse.from_domain(answer, settings.api_prefix)
