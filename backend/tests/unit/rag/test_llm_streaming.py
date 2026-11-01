"""Streaming in the LLM adapters (offline: mocked openai client / httpx.MockTransport)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from app.domain.exceptions import ExternalServiceError, TransientError
from app.domain.models import ChatMessage, Role
from app.infrastructure.llm.ollama_llm import OllamaLLM
from app.infrastructure.llm.openai_llm import OpenAILLM

MSG = [ChatMessage(Role.USER, "q")]
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


# --------------------------------------------------------------------------- OpenAI
def _chunk(content: str | None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


class FakeStream:
    def __init__(self, items) -> None:
        self.items = items
        self.closed = False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self.closed = True


def _openai(create) -> tuple[OpenAILLM, MagicMock]:
    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return OpenAILLM(api_key="sk-test", model="gpt-4o-mini", client=client), client


def test_openai_stream_yields_deltas_and_closes() -> None:
    stream = FakeStream([_chunk("Hola"), _chunk(None), SimpleNamespace(choices=[]), _chunk(" [1].")])
    llm, client = _openai(lambda **kw: stream)

    assert list(llm.stream(MSG, temperature=0.2, max_tokens=50)) == ["Hola", " [1]."]
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["max_tokens"] == 50 and kwargs["temperature"] == 0.2
    assert stream.closed


def test_openai_stream_open_errors_are_mapped() -> None:
    def create(**kw):
        raise openai.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)

    llm, _ = _openai(create)
    with pytest.raises(TransientError):
        list(llm.stream(MSG))


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (httpx.ReadTimeout("slow"), True),
        (httpx.RemoteProtocolError("reset"), True),
        (openai.APIError("boom", request=REQUEST, body=None), False),
    ],
)
def test_openai_stream_mid_stream_errors_are_mapped(error, transient) -> None:
    stream = FakeStream([_chunk("a"), error])
    llm, _ = _openai(lambda **kw: stream)
    out: list[str] = []
    with pytest.raises(ExternalServiceError) as info:
        out.extend(llm.stream(MSG))
    assert out == ["a"]
    assert isinstance(info.value, TransientError) is transient
    assert stream.closed


# --------------------------------------------------------------------------- Ollama
def _ollama(handler, model: str = "gemma3:4b") -> OllamaLLM:
    client = httpx.Client(base_url="http://ollama:11434", transport=httpx.MockTransport(handler))
    return OllamaLLM("http://ollama:11434", model, client=client)


def _ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def test_ollama_stream_parses_ndjson_and_passes_num_predict() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        body = _ndjson(
            {"message": {"role": "assistant", "content": "Ho"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": False},
            {"message": {"role": "assistant", "content": "la [1]"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 3},
        )
        return httpx.Response(200, content=body, headers={"content-type": "application/x-ndjson"})

    out = list(_ollama(handler).stream(MSG, temperature=0.1, max_tokens=256))

    assert out == ["Ho", "la [1]"]
    assert seen["body"]["stream"] is True
    assert seen["body"]["options"] == {"temperature": 0.1, "num_predict": 256}


def test_ollama_stream_without_max_tokens_has_no_num_predict() -> None:
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=_ndjson({"message": {"content": "x"}, "done": True}))

    assert list(_ollama(handler).stream(MSG)) == ["x"]
    assert "num_predict" not in seen["body"]["options"]


def test_ollama_stream_error_line_and_status_mapping() -> None:
    mid_error = _ollama(lambda r: httpx.Response(
        200, content=_ndjson({"message": {"content": "a"}, "done": False}, {"error": "model crashed"})))
    out: list[str] = []
    with pytest.raises(ExternalServiceError, match="model crashed"):
        out.extend(mid_error.stream(MSG))
    assert out == ["a"]

    busy = _ollama(lambda r: httpx.Response(503, text="busy", headers={"retry-after": "3"}))
    with pytest.raises(TransientError) as info:
        list(busy.stream(MSG))
    assert info.value.retry_after == 3.0

    missing = _ollama(lambda r: httpx.Response(404, json={"error": "model 'x' not found"}))
    with pytest.raises(ExternalServiceError) as info:
        list(missing.stream(MSG))
    assert not isinstance(info.value, TransientError)
    assert "ollama pull" in str(info.value)

    garbage = _ollama(lambda r: httpx.Response(200, content=b"not json\n"))
    with pytest.raises(ExternalServiceError):
        list(garbage.stream(MSG))


def test_ollama_stream_transport_errors_are_transient() -> None:
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(TransientError):
        list(_ollama(handler).stream(MSG))


def test_ollama_describe_image_caps_generation() -> None:
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "a diagram"}})

    assert _ollama(handler).describe_image(b"\x89PNG....") == "a diagram"
    assert seen["body"]["options"]["num_predict"] == 300
    assert seen["body"]["stream"] is False
