"""Ollama chat adapter (``POST /api/chat``), multimodal for vision models."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import httpx

from app.application.prompts import build_image_description_prompt
from app.domain.exceptions import ExternalServiceError, ValidationError
from app.domain.models import ChatMessage, Role
from app.domain.ports import LLMPort
from app.infrastructure.llm.ollama_http import OllamaHttpClient

VISION_MODEL_HINTS = ("vision", "llava", "bakllava", "moondream", "gemma3", "qwen2.5vl", "minicpm-v")


def is_vision_model(model: str) -> bool:
    name = model.lower()
    return any(hint in name for hint in VISION_MODEL_HINTS)


def to_ollama_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.images:
        payload["images"] = [base64.b64encode(img).decode("ascii") for img in message.images]
    return payload


class OllamaLLM(LLMPort):
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._http = OllamaHttpClient(base_url, timeout, client=client)

    @property
    def supports_vision(self) -> bool:
        return is_vision_model(self._model)

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        data = self._http.post_json(
            "/api/chat",
            {
                "model": self._model,
                "messages": [to_ollama_message(m) for m in messages],
                "stream": False,
                "options": options,
            },
        )
        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ExternalServiceError("Ollama /api/chat: response without message content")
        return str(message["content"])

    def describe_image(self, image: bytes, *, context: str = "") -> str:
        if not self.supports_vision:
            raise ValidationError(f"Ollama model '{self._model}' does not support images")
        message = ChatMessage(Role.USER, build_image_description_prompt(context), images=[image])
        return self.generate([message], temperature=0.0, max_tokens=300).strip()
