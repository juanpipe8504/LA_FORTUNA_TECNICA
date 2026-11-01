"""Fixtures for API tests: a fake container whose use cases are MagicMocks."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.application.jobs import GetJobUseCase
from app.bootstrap.container import Container
from app.core.config import Settings
from app.domain.models import IngestionJob, JobStatus

FIXTURE_TS = datetime(2026, 1, 1, tzinfo=UTC)

API = "/api/v1"


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Iterator[None]:
    """The app lifespan reconfigures root logging; undo it after each test."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=str(tmp_path / "data"),
        max_upload_mb=1,
        llm_provider="ollama",
        embedding_provider="fastembed",
        log_json=False,
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    job_repository = MagicMock(name="job_repository")
    return Container(
        settings=settings,
        llm=MagicMock(name="llm"),
        embeddings=MagicMock(name="embeddings"),
        vector_store=MagicMock(name="vector_store"),
        blob_storage=MagicMock(name="blob_storage"),
        job_repository=job_repository,
        task_queue=MagicMock(name="task_queue"),
        parser=MagicMock(name="parser"),
        chunker=MagicMock(name="chunker"),
        submit_document=MagicMock(name="submit_document"),
        process_document=MagicMock(name="process_document"),
        answer_question=MagicMock(name="answer_question"),
        list_documents=MagicMock(name="list_documents"),
        # Clock pinned to the fixtures' timestamp so the stale-job watchdog stays quiet.
        get_job=GetJobUseCase(job_repository=job_repository, clock=lambda: FIXTURE_TS),
        health_checks={"redis": lambda: True, "qdrant": lambda: True},
    )


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    # No ``with``: lifespan (logging setup) is not needed with an injected container.
    yield TestClient(create_app(container=container), raise_server_exceptions=False)


def make_job(**overrides) -> IngestionJob:
    ts = FIXTURE_TS
    values = dict(
        job_id="job-1",
        document_id="doc-1",
        filename="manual.pdf",
        file_path="/data/uploads/doc-1.pdf",
        status=JobStatus.PENDING,
        created_at=ts,
        updated_at=ts,
    )
    values.update(overrides)
    return IngestionJob(**values)
