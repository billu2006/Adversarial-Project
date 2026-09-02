"""The worker task: state transitions, failure recovery and cancellation.

The engine itself is stubbed here. What is under test is the promise the worker
makes to the API - that a job always reaches a terminal state - not PyTorch.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

import benchmark.engine as engine_module
from benchmark.engine import AttackResult
from service.database import SessionLocal
from service.jobs import cancel_job, create_job, get_job, mark_running, sweep_stale_jobs
from service.models import Job, JobStatus
from service.worker.tasks import run_benchmark_job


@pytest.fixture
def queued_job(session, job_request, job_queue) -> Job:
    job, _ = create_job(session, job_request, queue=job_queue)
    return job


def _stub_engine(monkeypatch, results=None, error: Exception | None = None):
    """Replace the benchmark with something instant and predictable."""

    def _fake_run_benchmark(**kwargs):
        if error is not None:
            raise error
        return results or [
            AttackResult(
                attack_name=name,
                robust_accuracy=0.5,
                mean_nll=1.5,
                perturbation_norm=kwargs["epsilon"],
                runtime_ms=42,
                samples=128,
            )
            for name in kwargs["attacks"]
        ]

    monkeypatch.setattr(engine_module, "run_benchmark", _fake_run_benchmark)


def test_successful_job_persists_results_and_timestamps(monkeypatch, queued_job, session):
    _stub_engine(monkeypatch)

    run_benchmark_job(job_id=str(queued_job.id))

    session.expire_all()
    job = get_job(session, queued_job.id)
    assert job.status is JobStatus.SUCCEEDED
    assert job.started_at is not None and job.finished_at is not None
    assert job.finished_at >= job.started_at
    assert [row.attack_name for row in job.results] == ["fgsm", "pgd"]
    assert float(job.results[0].robust_accuracy) == 0.5


def test_a_crashing_benchmark_leaves_the_job_failed(monkeypatch, queued_job, session):
    """The definition-of-done case: a job must never be stuck in `running`."""
    _stub_engine(monkeypatch, error=RuntimeError("CUDA is on fire"))

    run_benchmark_job(job_id=str(queued_job.id))

    session.expire_all()
    job = get_job(session, queued_job.id)
    assert job.status is JobStatus.FAILED
    assert "CUDA is on fire" in job.error_message
    assert job.finished_at is not None
    assert job.results == []


def test_a_missing_checkpoint_is_a_failed_job_not_a_crash(monkeypatch, queued_job, session):
    _stub_engine(monkeypatch, error=FileNotFoundError("weights missing"))

    run_benchmark_job(job_id=str(queued_job.id))

    session.expire_all()
    assert get_job(session, queued_job.id).status is JobStatus.FAILED


def test_a_cancelled_job_is_never_executed(monkeypatch, queued_job, session, job_queue):
    """Cancellation must hold even if the worker already had the job in hand."""
    _stub_engine(monkeypatch)
    cancel_job(session, queued_job.id, queue=job_queue)

    result = run_benchmark_job(job_id=str(queued_job.id))

    assert result["skipped"] is True
    session.expire_all()
    job = get_job(session, queued_job.id)
    assert job.status is JobStatus.CANCELLED
    assert job.results == []


def test_a_job_is_claimed_only_once(monkeypatch, queued_job, session):
    """Duplicate queue delivery must not run the benchmark twice."""
    _stub_engine(monkeypatch)

    first = run_benchmark_job(job_id=str(queued_job.id))
    second = run_benchmark_job(job_id=str(queued_job.id))

    assert first.get("skipped") is None
    assert second["skipped"] is True


def test_mark_running_is_atomic(queued_job, session):
    with SessionLocal() as other:
        claimed = mark_running(session, queued_job.id)
        lost = mark_running(other, queued_job.id)

    assert claimed is not None
    assert lost is None


def test_the_reaper_fails_jobs_abandoned_by_a_dead_worker(queued_job, session, settings):
    """A SIGKILLed worker cannot write `failed` itself; the reaper does it."""
    mark_running(session, queued_job.id)
    # Backdate the claim past the timeout to simulate a worker that never
    # came back.
    job = get_job(session, queued_job.id)
    job.started_at = datetime.now(UTC) - timedelta(
        seconds=settings.job_timeout_seconds + settings.stale_job_grace_seconds + 60
    )
    session.commit()

    reaped = sweep_stale_jobs(session)

    session.expire_all()
    assert reaped == 1
    assert get_job(session, queued_job.id).status is JobStatus.FAILED
    assert "presumed dead" in get_job(session, queued_job.id).error_message


def test_the_reaper_leaves_healthy_jobs_alone(queued_job, session):
    mark_running(session, queued_job.id)

    assert sweep_stale_jobs(session) == 0
    session.expire_all()
    assert get_job(session, queued_job.id).status is JobStatus.RUNNING


def test_worker_handles_an_unknown_job_id(monkeypatch):
    _stub_engine(monkeypatch)

    result = run_benchmark_job(job_id=str(uuid.uuid4()))

    assert result["skipped"] is True


def test_request_id_propagates_into_the_worker(monkeypatch, queued_job):
    """One correlation id across both processes, or the logs are unjoinable."""
    _stub_engine(monkeypatch)
    from service.logging_config import request_id_var

    run_benchmark_job(job_id=str(queued_job.id), request_id="abc123")

    assert request_id_var.get() == "abc123"
