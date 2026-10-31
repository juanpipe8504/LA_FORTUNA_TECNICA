"""Small JSON-over-HTTP helper for Ollama with domain error mapping (LLM + embeddings)."""

from __future__ import annotations

from typing import Any

import httpx

from app.domain.exceptions import ExternalServiceError
from app.infrastructure.resilience.retry import parse_retry_after, transient_error

_TRANSIENT_STATUS = {408, 425, 429}


class OllamaHttpClient:
    def __init__(self, base_url: str, timeout: float = 60.0, *, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise transient_error(f"Ollama {path}: timeout: {exc}") from exc
        except httpx.TransportError as exc:  # connect errors, resets, protocol errors
            raise transient_error(f"Ollama {path}: connection error: {exc}") from exc

        status = response.status_code
        if status in _TRANSIENT_STATUS or status >= 500:
            raise transient_error(
                f"Ollama {path}: HTTP {status}: {response.text[:300]}",
                parse_retry_after(response.headers.get("retry-after")),
            )
        if status >= 400:
            hint = " (is the model pulled? `ollama pull <model>`)" if status == 404 else ""
            raise ExternalServiceError(f"Ollama {path}: HTTP {status}: {response.text[:300]}{hint}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(f"Ollama {path}: invalid JSON response") from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(f"Ollama {path}: unexpected response shape")
        if "error" in data:
            raise ExternalServiceError(f"Ollama {path}: {data['error']}")
        return data
