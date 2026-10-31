"""Typed configuration loaded from environment variables / ``.env``.

``APP_ENV`` selects an optional overlay file (``.env.<env>``) on top of
``.env``, so development, test and production can differ without code changes.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    FASTEMBED = "fastembed"  # local CPU model, no API key needed


def _env_files() -> tuple[str, ...]:
    env = os.getenv("APP_ENV", Environment.DEVELOPMENT.value)
    return (".env", f".env.{env}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_files(), env_file_encoding="utf-8", extra="ignore"
    )

    # --- App -------------------------------------------------------------
    app_env: Environment = Environment.DEVELOPMENT
    app_name: str = "Multimodal RAG"
    log_level: str = "INFO"
    log_json: bool = True
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])
    max_upload_mb: int = 50
    data_dir: str = "./data"

    # --- Infrastructure --------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"

    # --- LLM -------------------------------------------------------------
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma3:4b"
    ollama_embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # --- Embeddings ------------------------------------------------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.FASTEMBED
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    # --- Ingestion -------------------------------------------------------
    chunk_max_tokens: int = 400
    chunk_overlap_sentences: int = 1
    min_image_size_px: int = 64  # skip icons / decorations
    image_link_max_distance: float = 150.0  # PDF points
    describe_images: bool = True  # caption images with the vision LLM if available

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = 6
    retrieval_min_score: float = 0.0
    max_images_per_answer: int = 2
    history_max_turns: int = 6

    # --- Resilience ------------------------------------------------------
    retry_max_attempts: int = 4
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.data_dir, "images")


@lru_cache
def get_settings() -> Settings:
    return Settings()
