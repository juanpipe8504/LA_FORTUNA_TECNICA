"""Question answering over the indexed documents (JSON and Server-Sent Events)."""

from __future__ import annotations

import contextvars
import json
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import AnswerQuestionDep, SettingsDep
from app.api.errors import describe_error, request_id_of
from app.api.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.application.query import FinalEvent, StreamEvent, TokenEvent
from app.core.logging import bind_context, get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

SSE_MEDIA_TYPE = "text/event-stream"
# no-cache: never store a live answer; X-Accel-Buffering: tell nginx not to buffer the stream.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# CPU models can take a long time to emit the first token: keep proxies from timing out.
KEEPALIVE_SECONDS = 15.0
SSE_PING = ": ping\n\n"


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


@router.post(
    "/stream",
    response_class=StreamingResponse,
    summary="Ask a question; streams the answer as Server-Sent Events",
    description=(
        "Same request body as `POST /chat`. Response `text/event-stream` with events "
        '`token` (`{"text": delta}`), one final `final` (same JSON as `POST /chat`) or '
        '`error` (`{"error": {"code", "message", "request_id"}}`), plus `: ping` keep-alive '
        "comments while waiting. Request validation errors are still plain JSON 422."
    ),
    responses={
        200: {"content": {SSE_MEDIA_TYPE: {}}},
        422: {"model": ErrorResponse},
    },
)
def chat_stream(
    body: ChatRequest, request: Request, use_case: AnswerQuestionDep, settings: SettingsDep
) -> StreamingResponse:
    question, history, document_ids = body.question, body.history_to_domain(), body.document_ids
    events = sse_events(
        lambda: use_case.stream(question=question, history=history, document_ids=document_ids),
        api_prefix=settings.api_prefix,
        request_id=request_id_of(request),
    )
    return StreamingResponse(events, media_type=SSE_MEDIA_TYPE, headers=SSE_HEADERS)


# --------------------------------------------------------------------------- #
# SSE plumbing
# --------------------------------------------------------------------------- #
def format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))  # single line
    return f"event: {event}\ndata: {payload}\n\n"


class _Failure:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_DONE = object()


def sse_events(
    open_events: Callable[[], Iterable[StreamEvent]],
    *,
    api_prefix: str,
    request_id: str | None,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
    clock: Callable[[], float] = time.perf_counter,
) -> Iterator[str]:
    """Run the use-case stream in a worker thread and render it as SSE frames.

    A producer thread is needed for keep-alive pings: the consumer waits on a
    queue with a timeout and emits ``: ping`` while the model is still thinking.
    If the client disconnects, the generator is closed, ``stop`` is set and the
    producer closes the LLM stream at its next event.
    """
    items: queue.Queue[object] = queue.Queue()
    stop = threading.Event()
    context = contextvars.copy_context()

    def produce() -> None:
        events: Iterable[StreamEvent] | None = None
        with bind_context(request_id=request_id):
            try:
                events = open_events()
                for event in events:
                    if stop.is_set():
                        break
                    items.put(event)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the client as an error event
                items.put(_Failure(exc))
            finally:
                close = getattr(events, "close", None)
                if callable(close):
                    close()
                items.put(_DONE)

    threading.Thread(target=context.run, args=(produce,), name="chat-stream", daemon=True).start()

    started = clock()
    first_token_at: float | None = None
    token_events = 0
    outcome = "incomplete"
    sufficient: bool | None = None
    try:
        while True:
            try:
                item = items.get(timeout=keepalive_seconds)
            except queue.Empty:
                yield SSE_PING
                continue
            if item is _DONE:
                break
            if isinstance(item, _Failure):
                with bind_context(request_id=request_id):
                    _, code, message = describe_error(item.exc)
                outcome = code
                error = {"code": code, "message": message, "request_id": request_id}
                yield format_sse("error", {"error": error})
                break
            if isinstance(item, TokenEvent):
                if first_token_at is None:
                    first_token_at = clock()
                token_events += 1
                yield format_sse("token", {"text": item.text})
            elif isinstance(item, FinalEvent):
                outcome = "ok"
                sufficient = item.answer.has_sufficient_context
                response = ChatResponse.from_domain(item.answer, api_prefix)
                yield format_sse("final", response.model_dump(mode="json"))
    except GeneratorExit:
        outcome = "client_disconnected"
        raise
    finally:
        stop.set()
        now = clock()
        with bind_context(request_id=request_id):
            logger.info(
                "chat stream finished",
                extra={
                    "outcome": outcome,
                    "latency_ms": round((now - started) * 1000, 1),
                    "ttft_ms": round((first_token_at - started) * 1000, 1) if first_token_at else None,
                    "token_events": token_events,
                    "has_sufficient_context": sufficient,
                },
            )
