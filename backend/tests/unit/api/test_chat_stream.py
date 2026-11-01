from __future__ import annotations

import json
import threading

import pytest

from app.api.routes.chat import SSE_PING, sse_events
from app.application.query import FinalEvent, TokenEvent
from app.domain.exceptions import CircuitOpenError, ExternalServiceError, ValidationError
from app.domain.models import Answer
from tests.unit.api.conftest import API
from tests.unit.api.test_chat import _answer

URL = f"{API}/chat/stream"


def parse_sse(text: str) -> list[tuple[str, object]]:
    """Return [(event, data)] in order; comment lines (": ping") become ("comment", text)."""
    events: list[tuple[str, object]] = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        if frame.startswith(":"):
            events.append(("comment", frame[1:].strip()))
            continue
        fields = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_stream_emits_tokens_then_final_identical_to_chat(client, container):
    answer = _answer()
    container.answer_question.stream.return_value = iter(
        [TokenEvent("The limit is "), TokenEvent("**85 C** [1]."), FinalEvent(answer)]
    )
    container.answer_question.execute.return_value = answer

    response = client.post(
        URL,
        json={
            "question": " max temp? ",
            "history": [{"role": "user", "content": "Hi"}],
            "document_ids": ["doc-1"],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_sse(response.text)
    assert events[:2] == [("token", {"text": "The limit is "}), ("token", {"text": "**85 C** [1]."})]
    assert events[-1][0] == "final"
    assert events[-1][1] == client.post(f"{API}/chat", json={"question": "max temp?"}).json()
    kwargs = container.answer_question.stream.call_args.kwargs
    assert kwargs["question"] == "max temp?"
    assert kwargs["document_ids"] == ["doc-1"]
    assert [m.content for m in kwargs["history"]] == ["Hi"]


def test_stream_tokens_are_json_escaped_single_line(client, container):
    container.answer_question.stream.return_value = iter(
        [TokenEvent("línea 1\n\nlínea 2 \"x\""), FinalEvent(Answer(text="t", has_sufficient_context=True))]
    )
    events = parse_sse(client.post(URL, json={"question": "q"}).text)
    assert events[0] == ("token", {"text": "línea 1\n\nlínea 2 \"x\""})


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (CircuitOpenError("circuit 'llm' is open"), "service_unavailable"),
        (ExternalServiceError("openai: secret-key-abc invalid"), "upstream_error"),
        (ValidationError("The question must not be empty"), "validation_error"),
        (RuntimeError("db password=hunter2"), "internal_error"),
    ],
)
def test_errors_become_error_event_with_request_id(client, container, error, code):
    def events(**_):
        yield TokenEvent("partial ")
        raise error

    container.answer_question.stream.side_effect = events

    response = client.post(URL, json={"question": "hi"}, headers={"X-Request-ID": "req-42"})

    assert response.status_code == 200  # headers were already sent when the error happened
    parsed = parse_sse(response.text)
    assert parsed[0] == ("token", {"text": "partial "})
    name, data = parsed[-1]
    assert name == "error"
    assert data["error"]["code"] == code
    assert data["error"]["request_id"] == "req-42"
    assert "secret-key-abc" not in response.text and "hunter2" not in response.text
    assert all(event != "final" for event, _ in parsed)


def test_invalid_payload_is_plain_json_422(client, container):
    response = client.post(URL, json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    container.answer_question.stream.assert_not_called()


def test_keepalive_ping_while_waiting_for_first_token():
    release = threading.Event()

    def slow_events():
        release.wait(5)
        yield TokenEvent("hola")
        yield FinalEvent(Answer(text="hola", has_sufficient_context=True))

    frames = sse_events(slow_events, api_prefix="/api/v1", request_id="r1", keepalive_seconds=0.01)
    assert next(frames) == SSE_PING
    release.set()
    rest = [f for f in frames if f != SSE_PING]
    parsed = parse_sse("".join(rest))
    assert [name for name, _ in parsed] == ["token", "final"]
    assert parsed[1][1]["answer"] == "hola"


def test_client_disconnect_stops_and_closes_the_producer():
    closed = threading.Event()
    gate = threading.Event()

    def events():
        try:
            yield TokenEvent("a")
            gate.wait(5)
            yield TokenEvent("b")
            yield TokenEvent("c")
        finally:
            closed.set()

    frames = sse_events(events, api_prefix="/api/v1", request_id=None, keepalive_seconds=5)
    assert parse_sse(next(frames)) == [("token", {"text": "a"})]
    frames.close()  # what Starlette does when the client goes away
    gate.set()
    assert closed.wait(5)
