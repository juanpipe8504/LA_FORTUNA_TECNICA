"""Ingestion job status (polled by the frontend)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import GetJobDep
from app.api.schemas import ErrorResponse, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get the status/progress of an ingestion job",
    responses={404: {"model": ErrorResponse}},
)
def get_job(job_id: str, get_job: GetJobDep) -> JobResponse:
    # Stale PROCESSING jobs are reported as FAILED by the use case (watchdog);
    # NotFoundError is mapped to 404 by the error handlers.
    return JobResponse.from_domain(get_job.execute(job_id))
