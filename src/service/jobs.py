"""Job lifecycle logic, kept out of the routers.

The routers translate HTTP to function calls; everything that decides what a
job *is* lives here, which is what lets the worker and the reaper reuse it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

# The torch-free catalogue: the API validates names, it never loads a model.
from benchmark.catalog import ATTACK_NAMES, MODEL_NAMES, is_supported_model
from service.config import Settings, get_settings
from service.errors import (
    CapacityError,
    IdempotencyKeyReuseError,
    InvalidRequestError,
    JobNotCancellableError,
    NotFoundError,
    ResultsNotReadyError,
    UnsupportedAttackError,
    UnsupportedModelError,
)
from service.models import TERMINAL_STATUSES, Job, JobResult, JobStatus
from service.queue import JobQueue
from service.schemas import JobCreateRequest

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = (JobStatus.QUEUED, JobStatus.RUNNING)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_submission(payload: JobCreateRequest, settings: Settings) -> None:
    """Reject anything the schema could not express, with an actionable error."""
    if not is_supported_model(payload.model_name):
        raise UnsupportedModelError(
            f"Model {payload.model_name!r} is not supported.",
            {"supported": list(MODEL_NAMES)},
        )

    unknown = [name for name in payload.attacks if name not in ATTACK_NAMES]
    if unknown:
        raise UnsupportedAttackError(
            f"Unsupported attack(s): {', '.join(sorted(unknown))}.",
            {"supported": list(ATTACK_NAMES)},
        )

    # The schema caps the list at 8; the deployment may cap it lower.
    if len(payload.attacks) > settings.max_attacks_per_job:
        raise InvalidRequestError(
            f"A job may request at most {settings.max_attacks_per_job} attacks.",
            {"requested": len(payload.attacks), "limit": settings.max_attacks_per_job},
        )
    if payload.max_iterations > settings.max_iterations_limit:
        raise InvalidRequestError(
            f"max_iterations may not exceed {settings.max_iterations_limit}.",
            {"limit": settings.max_iterations_limit},
        )
    if payload.epsilon > settings.max_epsilon:
        raise InvalidRequestError(
            f"epsilon may not exceed {settings.max_epsilon}.",
            {"limit": settings.max_epsilon},
        )


def request_fingerprint(payload: JobCreateRequest) -> str:
    """Stable hash of the submission, used to police idempotency-key reuse."""
    canonical = json.dumps(
        {
            "model_name": payload.model_name,
            "attacks": list(payload.attacks),
            "epsilon": f"{payload.epsilon:.4f}",
            "max_iterations": payload.max_iterations,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _assert_capacity(session: Session, settings: Settings) -> None:
    """Crude backpressure: refuse work when the backlog is already deep.

    Without this a client can enqueue faster than the workers drain, and the
    only signal is latency climbing until something falls over.
    """
    active = session.scalar(
        select(func.count()).select_from(Job).where(Job.status.in_(ACTIVE_STATUSES))
    )
    if active is not None and active >= settings.max_active_jobs:
        raise CapacityError(
            "Too many jobs are queued or running; retry shortly.",
            {"active_jobs": active, "limit": settings.max_active_jobs},
        )


# --------------------------------------------------------------------------- #
# Creation
# --------------------------------------------------------------------------- #
def create_job(
    session: Session,
    payload: JobCreateRequest,
    *,
    queue: JobQueue,
    idempotency_key: str | None = None,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> tuple[Job, bool]:
    """Create and enqueue a job. Returns ``(job, replayed)``.

    ``replayed`` is True when an ``Idempotency-Key`` matched an existing job, in
    which case nothing new was created and nothing new was enqueued.

    Idempotency is enforced by the UNIQUE constraint on ``idempotency_key``: we
    insert optimistically and catch the integrity error. A read-then-write check
    would look equivalent under a single client and duplicate the job under two
    concurrent ones - the exact scenario a retrying client produces.
    """
    settings = settings or get_settings()
    validate_submission(payload, settings)
    _assert_capacity(session, settings)

    fingerprint = request_fingerprint(payload)
    job = Job(
        id=uuid.uuid4(),
        idempotency_key=idempotency_key,
        request_fingerprint=fingerprint,
        model_name=payload.model_name,
        attacks=list(payload.attacks),
        epsilon=payload.epsilon,
        max_iterations=payload.max_iterations,
        status=JobStatus.QUEUED,
    )
    session.add(job)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        if idempotency_key is None:
            raise  # Not an idempotency collision; let the 500 handler log it.

        existing = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
        if existing is None:
            raise

        if existing.request_fingerprint != fingerprint:
            # Same key, different body. Returning the original job would be a
            # lie about what we ran; creating a new one would break the key's
            # promise. Refuse instead.
            raise IdempotencyKeyReuseError(
                "This Idempotency-Key was already used with a different request body.",
                {"job_id": str(existing.id)},
            ) from None

        logger.info(
            "Idempotent replay",
            extra={"job_id": str(existing.id), "idempotency_key": idempotency_key},
        )
        return existing, True

    try:
        queue.enqueue(job.id, request_id=request_id)
    except Exception as exc:
        # The row is already committed, so the alternative to failing it here is
        # a job that sits in `queued` forever with nothing to pick it up. A
        # transactional outbox would be the robust fix; see the README.
        logger.exception("Enqueue failed", extra={"job_id": str(job.id)})
        mark_failed(session, job, f"Could not enqueue job: {exc}")
        return job, False

    logger.info(
        "Job created",
        extra={"job_id": str(job.id), "model_name": job.model_name, "attacks": job.attacks},
    )
    return job, False


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def get_job(session: Session, job_id: uuid.UUID) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"No job with id {job_id}.")
    return job


def get_job_results(session: Session, job_id: uuid.UUID) -> tuple[Job, Sequence[JobResult]]:
    """Return a finished job's results, or explain why they are not there yet."""
    job = session.scalar(select(Job).options(selectinload(Job.results)).where(Job.id == job_id))
    if job is None:
        raise NotFoundError(f"No job with id {job_id}.")

    if job.status != JobStatus.SUCCEEDED:
        raise ResultsNotReadyError(
            f"Job {job_id} is {job.status.value}; results are only available for a succeeded job.",
            {
                "status": job.status.value,
                "error_message": job.error_message,
                "poll": f"/v1/jobs/{job_id}",
            },
        )
    return job, job.results


def encode_cursor(job: Job) -> str:
    """Keyset cursor: the sort key of the last row on the page.

    Keyset rather than OFFSET because jobs are created while a client pages
    through them, and OFFSET would silently skip or repeat rows when that
    happens (and gets slower the deeper you page).
    """
    created = job.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    raw = f"{created.isoformat()}|{job.id.hex}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        timestamp, job_id = raw.split("|", 1)
        return datetime.fromisoformat(timestamp), uuid.UUID(job_id)
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise InvalidRequestError("The cursor is malformed.", {"cursor": cursor}) from exc


def list_jobs(
    session: Session,
    *,
    status: JobStatus | None = None,
    limit: int = 20,
    cursor: str | None = None,
) -> tuple[list[Job], str | None]:
    """Newest-first page of jobs plus the cursor for the next page (or None)."""
    query = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    if status is not None:
        query = query.where(Job.status == status)

    if cursor is not None:
        created_at, job_id = decode_cursor(cursor)
        # Compare the full sort key, so jobs sharing a created_at (a burst of
        # submissions) still page deterministically.
        query = query.where(
            or_(
                Job.created_at < created_at,
                and_(Job.created_at == created_at, Job.id < job_id),
            )
        )

    # Fetch one extra row: its existence, not a COUNT(*), is what tells us
    # whether another page exists.
    rows = list(session.scalars(query.limit(limit + 1)))
    has_more = len(rows) > limit
    page = rows[:limit]
    return page, encode_cursor(page[-1]) if has_more and page else None


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #
def cancel_job(session: Session, job_id: uuid.UUID, *, queue: JobQueue) -> Job:
    """Cancel a queued job.

    The state change is a conditional UPDATE rather than read-modify-write, so a
    worker picking the job up at the same instant cannot lose the race and start
    a cancelled job. Running jobs are not cancellable: killing a worker mid-batch
    is a bigger hammer than this endpoint should hold.
    """
    job = get_job(session, job_id)

    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
        .values(status=JobStatus.CANCELLED, finished_at=datetime.now(UTC))
    )
    session.commit()

    if result.rowcount == 0:
        session.refresh(job)
        raise JobNotCancellableError(
            f"Job {job_id} is {job.status.value}; only a queued job can be cancelled.",
            {"status": job.status.value},
        )

    # Best effort: if the worker already popped it, the status check at the top
    # of the task is what stops the benchmark from running.
    queue.cancel(job_id)

    session.refresh(job)
    logger.info("Job cancelled", extra={"job_id": str(job_id)})
    return job


def mark_running(session: Session, job_id: uuid.UUID) -> Job | None:
    """Claim a job for execution.

    Returns None if the job is no longer claimable (cancelled while queued, or
    already picked up), which is the worker's signal to drop it silently.
    """
    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.status == JobStatus.QUEUED)
        .values(status=JobStatus.RUNNING, started_at=datetime.now(UTC))
    )
    session.commit()
    if result.rowcount == 0:
        return None
    return session.get(Job, job_id)


def mark_succeeded(session: Session, job: Job, results: Sequence[JobResult]) -> Job:
    """Persist results and complete the job in one transaction.

    Results and the status change must land together: a client that sees
    `succeeded` and then gets an empty results list has been lied to.
    """
    for result in results:
        session.add(result)
    job.status = JobStatus.SUCCEEDED
    job.finished_at = datetime.now(UTC)
    job.error_message = None
    session.commit()
    return job


def mark_failed(session: Session, job: Job, message: str) -> Job:
    job.status = JobStatus.FAILED
    job.finished_at = datetime.now(UTC)
    # Truncated because the message is client-visible and an unbounded traceback
    # is neither useful to them nor safe to echo.
    job.error_message = message[:1000]
    session.commit()
    return job


def sweep_stale_jobs(session: Session, *, settings: Settings | None = None) -> int:
    """Fail jobs that have been `running` longer than any job is allowed to run.

    This is the answer to "what happens if the worker dies mid-job?". A SIGKILL
    gives the task no chance to write `failed`, so the row would sit in
    `running` forever and the client would poll forever. Every job has a hard
    timeout, so anything past that timeout plus a grace period is by definition
    abandoned. Returns the number of jobs reaped.
    """
    settings = settings or get_settings()
    cutoff = datetime.now(UTC) - timedelta(
        seconds=settings.job_timeout_seconds + settings.stale_job_grace_seconds
    )

    result = session.execute(
        update(Job)
        .where(
            Job.status == JobStatus.RUNNING,
            Job.started_at.is_not(None),
            Job.started_at < cutoff,
        )
        .values(
            status=JobStatus.FAILED,
            finished_at=datetime.now(UTC),
            error_message=(
                "Job exceeded its timeout without reporting a result; the worker "
                "running it is presumed dead."
            ),
        )
    )
    session.commit()
    if result.rowcount:
        logger.warning("Reaped stale jobs", extra={"reaped": result.rowcount})
    return result.rowcount


def defence_score(results: Sequence[JobResult]) -> float | None:
    """Mean robust accuracy across attacks - the framework's headline metric.

    Rounded to the precision the stored values actually have (NUMERIC(5,4)),
    rather than published with the float noise an average introduces.
    """
    if not results:
        return None
    total = sum(Decimal(str(row.robust_accuracy)) for row in results)
    return float((total / len(results)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "cancel_job",
    "create_job",
    "decode_cursor",
    "defence_score",
    "encode_cursor",
    "get_job",
    "get_job_results",
    "list_jobs",
    "mark_failed",
    "mark_running",
    "mark_succeeded",
    "request_fingerprint",
    "sweep_stale_jobs",
    "validate_submission",
]
