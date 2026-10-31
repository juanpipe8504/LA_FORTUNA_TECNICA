from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from app.domain.exceptions import ExternalServiceError, TransientError, ValidationError
from app.domain.models import ChatMessage, Role
from app.infrastructure.llm.ollama_llm import OllamaLLM, is_vision_model
from app.infrastructure.llm.openai_llm import OpenAILLM, image_data_url, sniff_image_mime, to_openai_message

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 16
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status: int, headers: dict | None = None):
    response = httpx.Response(status, headers=headers or {}, request=REQUEST)
    return cls(f"HTTP {status}", response=response, body=None)


def _completion(content: str | None = "hello"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )


def _openai_llm(create):
    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return OpenAILLM(api_key="sk-test", model="gpt-4o-mini", client=client), client


# --------------------------------------------------------------------------- OpenAI
def test_openai_message_mapping_text_and_images() -> None:
    assert to_openai_message(ChatMessage(Role.SYSTEM, "rules")) == {"role": "system", "content": "rules"}

    mapped = to_openai_message(ChatMessage(Role.USER, "what is this?", images=[PNG, JPEG]))
    assert mapped["role"] == "user"
    text, img1, img2 = mapped["content"]
    assert text == {"type": "text", "text": "what is this?"}
    assert img1["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert img2["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_sniff_mime() -> None:
    assert sniff_image_mime(b"GIF89a....") == "image/gif"
    assert sniff_image_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_mime(b"unknown") == "image/png"
    assert image_data_url(JPEG).startswith("data:image/jpeg")


def test_openai_generate_passes_parameters() -> None:
    llm, client = _openai_llm(lambda **kw: _completion("answer [1]"))
    out = llm.generate([ChatMessage(Role.USER, "q")], temperature=0.2, max_tokens=99)

    assert out == "answer [1]"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 99
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]
    assert llm.supports_vision is True


def test_openai_none_content_and_empty_choices() -> None:
    llm, _ = _openai_llm(lambda **kw: _completion(None))
    assert llm.generate([ChatMessage(Role.USER, "q")]) == ""

    llm, _ = _openai_llm(lambda **kw: SimpleNamespace(choices=[], usage=None))
    with pytest.raises(ExternalServiceError):
        llm.generate([ChatMessage(Role.USER, "q")])


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.InternalServerError, 503),
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
    ],
)
def test_openai_transient_errors(error) -> None:
    llm, _ = _openai_llm(error)
    with pytest.raises(TransientError) as info:
        llm.generate([ChatMessage(Role.USER, "q")])
    assert info.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.BadRequestError, 400),
        _status_error(openai.AuthenticationError, 401),
        _status_error(openai.NotFoundError, 404),
    ],
)
def test_openai_permanent_errors_are_not_transient(error) -> None:
    llm, _ = _openai_llm(error)
    with pytest.raises(ExternalServiceError) as info:
        llm.generate([ChatMessage(Role.USER, "q")])
    assert not isinstance(info.value, TransientError)


def test_openai_retry_after_header_is_parsed() -> None:
    llm, _ = _openai_llm(_status_error(openai.RateLimitError, 429, {"retry-after": "7"}))
    with pytest.raises(TransientError) as info:
        llm.generate([ChatMessage(Role.USER, "q")])
    assert info.value.retry_after == 7.0

    llm, _ = _openai_llm(_status_error(openai.RateLimitError, 429, {"retry-after-ms": "1500"}))
    with pytest.raises(TransientError) as info:
        llm.generate([ChatMessage(Role.USER, "q")])
    assert info.value.retry_after == 1.5


def test_openai_describe_image_sends_image_and_context() -> None:
    llm, client = _openai_llm(lambda **kw: _completion("  Block diagram of a pump.  "))
    assert llm.describe_image(PNG, context="Sección 3: Bomba hidráulica") == "Block diagram of a pump."

    message = client.chat.completions.create.call_args.kwargs["messages"][0]
    text_part, image_part = message["content"]
    assert "Bomba hidráulica" in text_part["text"]
    assert image_part["type"] == "image_url"


# --------------------------------------------------------------------------- Ollama
def _ollama(handler, model: str = "llama3.2-vision") -> OllamaLLM:
    client = httpx.Client(base_url="http://ollama:11434", transport=httpx.MockTransport(handler))
    return OllamaLLM("http://ollama:11434", model, client=client)


def test_ollama_generate_payload_and_response() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hola [1]"}})

    llm = _ollama(handler)
    out = llm.generate([ChatMessage(Role.USER, "q", images=[PNG])], temperature=0.1, max_tokens=64)

    assert out == "hola [1]"
    assert seen["path"] == "/api/chat"
    body = seen["body"]
    assert body["stream"] is False
    assert body["model"] == "llama3.2-vision"
    assert body["options"] == {"temperature": 0.1, "num_predict": 64}
    assert body["messages"][0]["images"] == [base64.b64encode(PNG).decode()]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_ollama_transient_status(status) -> None:
    llm = _ollama(lambda r: httpx.Response(status, text="busy", headers={"retry-after": "2"}))
    with pytest.raises(TransientError) as info:
        llm.generate([ChatMessage(Role.USER, "q")])
    assert info.value.retry_after == 2.0


@pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_ollama_transport_errors_are_transient(exc) -> None:
    def handler(request):
        raise exc

    with pytest.raises(TransientError):
        _ollama(handler).generate([ChatMessage(Role.USER, "q")])


def test_ollama_client_errors_are_permanent() -> None:
    llm = _ollama(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(ExternalServiceError) as info:
        llm.generate([ChatMessage(Role.USER, "q")])
    assert not isinstance(info.value, TransientError)
    assert "ollama pull" in str(info.value)


def test_ollama_malformed_response() -> None:
    with pytest.raises(ExternalServiceError):
        _ollama(lambda r: httpx.Response(200, json={"done": True})).generate([ChatMessage(Role.USER, "q")])


def test_ollama_describe_image() -> None:
    llm = _ollama(lambda r: httpx.Response(200, json={"message": {"content": " a chart "}}))
    assert llm.describe_image(PNG) == "a chart"

    text_only = _ollama(lambda r: httpx.Response(200, json={}), model="llama3.1:8b")
    assert text_only.supports_vision is False
    with pytest.raises(ValidationError):
        text_only.describe_image(PNG)


@pytest.mark.parametrize(
    ("model", "expected"),
    [("llama3.2-vision", True), ("llava:13b", True), ("moondream", True), ("gemma3:4b", True),
     ("qwen2.5vl:7b", True), ("minicpm-v", True), ("llama3.1", False), ("mistral", False)],
)
def test_is_vision_model(model, expected) -> None:
    assert is_vision_model(model) is expected
