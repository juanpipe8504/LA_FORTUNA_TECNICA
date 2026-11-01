"""Typed configuration loaded from environment variables / ``.env``.

``APP_ENV`` selects an optional overlay file (``.env.<env>``) on top of
``.env``, so development, test and production can differ without code changes.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator, model_validator
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


# Snowball stemmers + stop-word lists shipped with fastembed's ``Qdrant/bm25``.
BM25_LANGUAGES = frozenset(
    {
        "arabic", "danish", "dutch", "english", "finnish", "french", "german", "greek",
        "hungarian", "italian", "norwegian", "portuguese", "romanian", "russian",
        "spanish", "swedish", "tamil", "turkish",
    }
)


# Extra seconds on top of the hard time limit before the broker re-delivers a task.
BROKER_VISIBILITY_MARGIN_SECONDS = 300


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
    # Stemmer/stop-words of the BM25 sparse vectors. Must match the documents'
    # language; changing it changes the sparse token ids -> re-ingest.
    bm25_language: str = "spanish"

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
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_batch_size: int = 32

    # --- Ingestion -------------------------------------------------------
    chunk_max_tokens: int = 400
    chunk_overlap_sentences: int = 1
    min_image_size_px: int = 64  # skip icons / decorations
    image_link_max_distance: float = 150.0  # PDF points
    describe_images: bool = True  # caption images with the vision LLM if available
    # Wall-clock budget for the optional describe-images step of ONE document;
    # once exceeded the remaining images keep their PDF captions only.
    describe_images_budget_seconds: float = Field(default=600.0, gt=0)
    # Stop describing after this many consecutive failures (e.g. Ollama cannot
    # load the vision model); an open circuit breaker stops it immediately.
    describe_images_max_consecutive_failures: int = Field(default=2, ge=1)

    # --- Ingestion job bounds (Celery worker) -----------------------------
    # Soft limit raises inside the task so the job is marked FAILED cleanly;
    # the hard limit kills the worker process if that cleanup itself hangs.
    ingestion_soft_time_limit_seconds: int = Field(default=1800, gt=0)
    ingestion_hard_time_limit_seconds: int = Field(default=1900, gt=0)
    # A PROCESSING job whose last progress update is older than this is
    # reported as FAILED (worker lost / host slept). Must exceed the longest
    # single step without a progress update (one LLM call incl. retries).
    job_stale_after_seconds: int = Field(default=900, gt=0)

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

    @field_validator("bm25_language")
    @classmethod
    def _check_bm25_language(cls, value: str) -> str:
        language = value.strip().lower()
        if language not in BM25_LANGUAGES:
            raise ValueError(
                f"Unsupported BM25_LANGUAGE '{value}'. Use one of: {', '.join(sorted(BM25_LANGUAGES))}"
            )
        return language

    @model_validator(mode="after")
    def _check_time_limits(self) -> Settings:
        if self.ingestion_hard_time_limit_seconds <= self.ingestion_soft_time_limit_seconds:
            raise ValueError(
                "INGESTION_HARD_TIME_LIMIT_SECONDS must be greater than INGESTION_SOFT_TIME_LIMIT_SECONDS "
                "(the gap is the time left to mark the job as failed)"
            )
        return self

    @property
    def broker_visibility_timeout_seconds(self) -> int:
        """Redis re-delivers an unacked (acks_late) task after this long. It must
        exceed the hard time limit, otherwise a long but healthy task would be
        handed to a second worker while the first is still running it."""
        return self.ingestion_hard_time_limit_seconds + BROKER_VISIBILITY_MARGIN_SECONDS

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.data_dir, "images")


@lru_cache
def get_settings() -> Settings:
    return Settings()
