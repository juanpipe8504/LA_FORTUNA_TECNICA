"""OpenAI (or any OpenAI-compatible endpoint) chat adapter with vision support."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import openai
from openai import OpenAI

from app.application.prompts import build_image_description_prompt
from app.core.logging import get_logger
from app.domain.exceptions import ExternalServiceError
from app.domain.models import ChatMessage, Role
from app.domain.ports import LLMPort
from app.infrastructure.llm.openai_errors import map_openai_error

logger = get_logger(__name__)


def sniff_image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def image_data_url(data: bytes) -> str:
    return f"data:{sniff_image_mime(data)};base64,{base64.b64encode(data).decode('ascii')}"


def to_openai_message(message: ChatMessage) -> dict[str, Any]:
    """Map a domain message to the Chat Completions format (multimodal when images exist)."""
    if not message.images:
        return {"role": message.role.value, "content": message.content}
    parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
    parts.extend(
        {"type": "image_url", "image_url": {"url": image_data_url(img), "detail": "auto"}}
        for img in message.images
    )
    return {"role": message.role.value, "content": parts}


class OpenAILLM(LLMPort):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self._model = model
        # max_retries=0: retries are owned by our resilience layer (single policy, observable).
        self._client = client or OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    @property
    def supports_vision(self) -> bool:
        return True

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "model": self._model,
            "messages": [to_openai_message(m) for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        try:
            response = self._client.chat.completions.create(**params)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, "chat completion") from exc
        if not response.choices:
            raise ExternalServiceError("OpenAI chat completion returned no choices")
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.debug(
                "openai_usage",
                extra={
                    "model": self._model,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                },
            )
        return response.choices[0].message.content or ""

    def describe_image(self, image: bytes, *, context: str = "") -> str:
        message = ChatMessage(Role.USER, build_image_description_prompt(context), images=[image])
        return self.generate([message], temperature=0.0, max_tokens=300).strip()
