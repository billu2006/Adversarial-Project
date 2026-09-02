"""``/v1/jobs`` - submission, polling, results, listing and cancellation."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Header, Query, Response, status
from sqlalchemy.orm import Session

from service import jobs as job_service
from service.config import Settings
from service.database import get_db
from service.dependencies import queue_dependency, settings_dependency
from service.logging_config import request_id_var
from service.models import JobStatus
from service.queue import JobQueue
from service.schemas import (
    AttackResultResource,
    JobCreateRequest,
    JobListResponse,
    JobResource,
    JobResultsResponse,
)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.post(
    "",
    response_model=JobResource,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a benchmark job",
)
def submit_job(
    payload: JobCreateRequest,
    response: Response,
    session: Session = Depends(get_db),
    queue: JobQueue = Depends(queue_dependency),
    settings: Settings = Depends(settings_dependency),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResource:
    """Accept the job and return immediately - the benchmark runs on a worker.

    A replayed ``Idempotency-Key`` returns the original job with ``200 OK``
    rather than ``202 Accepted``, so a client can tell "I created this" from
    "this already existed" without comparing timestamps.
    """
    job, replayed = job_service.create_job(
        session,
        payload,
        queue=queue,
        idempotency_key=idempotency_key,
        # Carried into the worker so both processes log the same id.
        request_id=request_id_var.get(),
        settings=settings,
    )

    if replayed:
        response.status_code = status.HTTP_200_OK
        response.headers["Idempotency-Replayed"] = "true"
    else:
        # Where to poll. Standard for a 202, and saves the client string-building.
        response.headers["Location"] = f"/v1/jobs/{job.id}"

    return JobResource.from_job(job)


@router.get("", response_model=JobListResponse, summary="List jobs, newest first")
def list_jobs(
    session: Session = Depends(get_db),
    settings: Settings = Depends(settings_dependency),
    job_status: JobStatus | None = Query(default=None, alias="status"),
    limit: int | None = Query(default=None, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> JobListResponse:
    page, next_cursor = job_service.list_jobs(
        session,
        status=job_status,
        limit=min(limit or settings.default_page_size, settings.max_page_size),
        cursor=cursor,
    )
    return JobListResponse(
        items=[JobResource.from_job(job) for job in page], next_cursor=next_cursor
    )


@router.get("/{job_id}", response_model=JobResource, summary="Poll job status")
def get_job(job_id: uuid.UUID, session: Session = Depends(get_db)) -> JobResource:
    return JobResource.from_job(job_service.get_job(session, job_id))


@router.get(
    "/{job_id}/results",
    response_model=JobResultsResponse,
    summary="Fetch results (409 until the job has succeeded)",
)
def get_results(job_id: uuid.UUID, session: Session = Depends(get_db)) -> JobResultsResponse:
    job, results = job_service.get_job_results(session, job_id)
    return JobResultsResponse(
        job_id=job.id,
        status=job.status,
        model_name=job.model_name,
        epsilon=float(job.epsilon),
        max_iterations=job.max_iterations,
        results=[AttackResultResource.from_row(row) for row in results],
        defence_score=job_service.defence_score(results),
    )


@router.delete("/{job_id}", response_model=JobResource, summary="Cancel a queued job")
def cancel_job(
    job_id: uuid.UUID,
    session: Session = Depends(get_db),
    queue: JobQueue = Depends(queue_dependency),
) -> JobResource:
    return JobResource.from_job(job_service.cancel_job(session, job_id, queue=queue))
