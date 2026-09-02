"""The RQ task: run one benchmark and record what happened.

The contract with the API is entirely through the database. Whatever happens in
here - success, exception, timeout, cancellation - the job must end in a
terminal state with an explanation, because a client polling a status that never
changes has no way to recover.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence

from service.database import SessionLocal
from service.jobs import mark_failed, mark_running, mark_succeeded
from service.logging_config import bind_request_id
from service.models import Job, JobResult

logger = logging.getLogger(__name__)


def _build_result_rows(job_id: uuid.UUID, results: Sequence) -> list[JobResult]:
    """Map engine ``AttackResult`` dataclasses onto ``job_results`` rows."""
    return [
        JobResult(
            job_id=job_id,
            attack_name=result.attack_name,
            robust_accuracy=result.robust_accuracy,
            mean_nll=result.mean_nll,
            perturbation_norm=result.perturbation_norm,
            runtime_ms=result.runtime_ms,
        )
        for result in results
    ]


def run_benchmark_job(*, job_id: str, request_id: str | None = None) -> dict:
    """Execute the benchmark for ``job_id``.

    Returns a small summary dict (handy in the RQ dashboard); the authoritative
    record is the rows written to Postgres.
    """
    # Adopt the API's correlation id so the whole submission reads as one trace.
    bind_request_id(request_id)
    identifier = uuid.UUID(job_id)

    # Imported inside the function, not at module scope: importing torch costs
    # seconds, and deferring it keeps worker startup fast and turns an engine
    # import error into a failed job rather than a crash-looping container.
    from benchmark.engine import run_benchmark
    from service.config import get_settings

    settings = get_settings()
    session = SessionLocal()

    try:
        # Claim the job. None means it is no longer queued - cancelled while
        # waiting, or a duplicate delivery lost the race - so there is nothing
        # to do and nothing to report.
        job = mark_running(session, identifier)
        if job is None:
            logger.info("Job not claimable, skipping", extra={"job_id": job_id})
            return {"job_id": job_id, "skipped": True}

        logger.info(
            "Benchmark started",
            extra={
                "job_id": job_id,
                "model_name": job.model_name,
                "attacks": job.attacks,
                "epsilon": float(job.epsilon),
            },
        )

        def _progress(attack_name: str, done: int, total: int) -> None:
            logger.info(
                "Attack finished",
                extra={"job_id": job_id, "attack": attack_name, "progress": f"{done}/{total}"},
            )

        results = run_benchmark(
            model_name=job.model_name,
            attacks=list(job.attacks),
            epsilon=float(job.epsilon),
            max_iterations=job.max_iterations,
            max_samples=settings.benchmark_max_samples,
            batch_size=settings.benchmark_batch_size,
            progress=_progress,
        )

        mark_succeeded(session, job, _build_result_rows(identifier, results))
        logger.info(
            "Benchmark succeeded", extra={"job_id": job_id, "attacks_completed": len(results)}
        )
        return {"job_id": job_id, "attacks": len(results)}

    except Exception as exc:
        # Catches engine errors, missing weights, and RQ's timeout exception,
        # which is raised inside this frame when the job outruns job_timeout. A
        # hard kill (SIGKILL, OOM, node loss) escapes even this; sweep_stale_jobs
        # is the backstop for those.
        logger.exception("Benchmark failed", extra={"job_id": job_id})
        session.rollback()
        job = session.get(Job, identifier)
        if job is not None:
            mark_failed(session, job, f"{type(exc).__name__}: {exc}")
        # Deliberately not re-raised. The database is the record of truth, and
        # re-raising would additionally surface as a 500 when the inline queue
        # backend runs this in the API process.
        return {"job_id": job_id, "failed": True, "error": type(exc).__name__}
    finally:
        session.close()
