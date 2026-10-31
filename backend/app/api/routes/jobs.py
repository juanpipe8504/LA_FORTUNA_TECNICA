"""Ingestion job status (polled by the frontend)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import JobRepositoryDep
from app.api.schemas import ErrorResponse, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get the status/progress of an ingestion job",
    responses={404: {"model": ErrorResponse}},
)
def get_job(job_id: str, jobs: JobRepositoryDep) -> JobResponse:
    # NotFoundError from the repository is mapped to 404 by the error handlers.
    return JobResponse.from_domain(jobs.get(job_id))
