"""Celery tasks (thin adapters: all business logic lives in the use cases)."""

from __future__ import annotations

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import get_settings
from app.core.logging import bind_context, get_logger
from app.infrastructure.queue.celery_app import INGEST_DOCUMENT_TASK, celery_app

logger = get_logger(__name__)

_settings = get_settings()


@celery_app.task(
    name=INGEST_DOCUMENT_TASK,
    bind=True,
    soft_time_limit=_settings.ingestion_soft_time_limit_seconds,
    time_limit=_settings.ingestion_hard_time_limit_seconds,
)
def ingest_document(self: Task, job_id: str) -> None:
    """Process an uploaded document. Failures are recorded on the job by the use case.

    ``SoftTimeLimitExceeded`` is raised asynchronously by Celery anywhere inside
    the use case. The use case is told to let it propagate untouched (it stays
    framework-agnostic: it only receives an opaque tuple of "interrupt" exception
    types); here it is translated into an explicit ``mark_timed_out`` so the job
    ends FAILED with a clear message instead of staying PROCESSING. The task then
    returns normally, so the message is acknowledged and NOT re-delivered.
    """
    # Imported lazily so importing the tasks module (e.g. by the API process) does
    # not build the whole container with its heavy adapters.
    from app.bootstrap.container import get_container

    with bind_context(job_id=job_id, task_id=self.request.id):
        redelivered = bool((self.request.delivery_info or {}).get("redelivered"))
        logger.info("ingestion_task_started", extra={"redelivered": redelivered})
        use_case = get_container().process_document
        try:
            use_case.execute(job_id, interrupt_errors=(SoftTimeLimitExceeded,))
        except SoftTimeLimitExceeded:
            logger.error(
                "ingestion_task_time_limit_exceeded",
                extra={"soft_time_limit_s": _settings.ingestion_soft_time_limit_seconds},
            )
            use_case.mark_timed_out(job_id)
            return
        logger.info("ingestion_task_finished")
