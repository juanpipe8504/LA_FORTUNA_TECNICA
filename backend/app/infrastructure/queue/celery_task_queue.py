"""``TaskQueuePort`` adapter that publishes ingestion jobs to Celery."""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.exceptions import ExternalServiceError
from app.domain.ports import TaskQueuePort
from app.infrastructure.queue.celery_app import INGEST_DOCUMENT_TASK, INGESTION_QUEUE, celery_app

logger = get_logger(__name__)


class CeleryTaskQueue(TaskQueuePort):
    def enqueue_ingestion(self, job_id: str) -> None:
        try:
            # send_task by name: the API never imports the worker-side task code.
            result = celery_app.send_task(INGEST_DOCUMENT_TASK, args=[job_id], queue=INGESTION_QUEUE)
        except Exception as exc:  # kombu raises various connection errors
            logger.exception("enqueue_failed", extra={"job_id": job_id})
            raise ExternalServiceError(f"Could not enqueue ingestion job {job_id}") from exc
        logger.info("ingestion_enqueued", extra={"job_id": job_id, "task_id": result.id})
