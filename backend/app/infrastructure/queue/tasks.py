"""Celery tasks (thin adapters: all business logic lives in the use cases)."""

from __future__ import annotations

from celery import Task

from app.core.logging import bind_context, get_logger
from app.infrastructure.queue.celery_app import INGEST_DOCUMENT_TASK, celery_app

logger = get_logger(__name__)


@celery_app.task(name=INGEST_DOCUMENT_TASK, bind=True)
def ingest_document(self: Task, job_id: str) -> None:
    """Process an uploaded document. Failures are recorded on the job by the use case."""
    # Imported lazily so importing the tasks module (e.g. by the API process) does
    # not build the whole container with its heavy adapters.
    from app.bootstrap.container import get_container

    with bind_context(job_id=job_id, task_id=self.request.id):
        logger.info("ingestion_task_started")
        get_container().process_document.execute(job_id)
        logger.info("ingestion_task_finished")
