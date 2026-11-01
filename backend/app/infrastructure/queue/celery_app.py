"""Celery application used by the ingestion worker.

Run with: ``celery -A app.infrastructure.queue.celery_app worker -Q ingestion``.

Reliability settings: tasks are acknowledged *after* they finish
(``acks_late``) and re-queued if the worker process dies
(``task_reject_on_worker_lost``); prefetch is 1 because ingestion tasks are
long and CPU-heavy, so a worker never hoards jobs another worker could run.

Bounded execution: every ingestion task has a soft time limit (raises
``SoftTimeLimitExceeded`` inside the task -> the job is marked FAILED) and a
hard one (the child process is killed). The Redis ``visibility_timeout`` is
larger than the hard limit, so an unacknowledged (acks_late) task is only
re-delivered when its worker was genuinely lost, never while it still runs.
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import worker_process_init

from app.core.config import get_settings
from app.core.logging import configure_logging

INGESTION_QUEUE = "ingestion"
INGEST_DOCUMENT_TASK = "ingestion.ingest_document"


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "multimodal_rag",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.infrastructure.queue.tasks"],
    )
    app.conf.update(
        task_routes={"ingestion.*": {"queue": INGESTION_QUEUE}},
        task_default_queue=INGESTION_QUEUE,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        result_expires=24 * 3600,
        broker_connection_retry_on_startup=True,
        timezone="UTC",
        enable_utc=True,
        # Celery would otherwise replace our structured log handlers.
        worker_hijack_root_logger=False,
        # Time limits (also set on the task itself, see tasks.py).
        task_soft_time_limit=settings.ingestion_soft_time_limit_seconds,
        task_time_limit=settings.ingestion_hard_time_limit_seconds,
        broker_transport_options={"visibility_timeout": settings.broker_visibility_timeout_seconds},
        result_backend_transport_options={"visibility_timeout": settings.broker_visibility_timeout_seconds},
    )
    return app


celery_app = create_celery_app()


@worker_process_init.connect
def _configure_worker_logging(**_: Any) -> None:
    """Each forked worker process gets the same structured logging as the API."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
