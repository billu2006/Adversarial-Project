"""Idempotent submission.

The interesting property is not "the same key returns the same job" but *how*:
the uniqueness is enforced by the database, so two requests racing each other
still produce one job. These tests drive that path directly with two
independent sessions, which is the same code path a genuine race takes.
"""

from __future__ import annotations

import uuid

import pytest

from service.database import SessionLocal
from service.errors import IdempotencyKeyReuseError
from service.jobs import create_job
from service.models import Job

KEY = "8f14e45f-ea1a-4e5e-9c2b-1d3b7c6a9e01"


def test_replayed_key_returns_the_original_job(client, submission, job_queue):
    first = client.post("/v1/jobs", json=submission, headers={"Idempotency-Key": KEY})
    second = client.post("/v1/jobs", json=submission, headers={"Idempotency-Key": KEY})

    assert first.status_code == 202
    # 200 rather than 202: nothing new was accepted, so the client can tell a
    # replay from a fresh submission.
    assert second.status_code == 200
    assert second.headers["Idempotency-Replayed"] == "true"
    assert first.json()["job_id"] == second.json()["job_id"]
    # And - the point of the whole exercise - the benchmark is queued once.
    assert len(job_queue.enqueued) == 1


def test_concurrent_duplicates_create_one_job(job_queue, job_request, session):
    """Two sessions inserting the same key: the loser reads the winner's job.

    This is the read-then-write bug made impossible. Nothing here checks
    "does a job with this key exist?" first; the second insert simply violates
    the UNIQUE constraint and the handler recovers.
    """
    with SessionLocal() as other_session:
        first, first_replayed = create_job(
            session, job_request, queue=job_queue, idempotency_key=KEY
        )
        second, second_replayed = create_job(
            other_session, job_request, queue=job_queue, idempotency_key=KEY
        )

    assert first_replayed is False
    assert second_replayed is True
    assert first.id == second.id
    assert len(job_queue.enqueued) == 1

    with SessionLocal() as verify:
        assert verify.query(Job).count() == 1


def test_same_key_with_a_different_body_is_a_conflict(client, submission, job_queue):
    """Reusing a key for different work is a client bug, not a replay."""
    client.post("/v1/jobs", json=submission, headers={"Idempotency-Key": KEY})
    response = client.post(
        "/v1/jobs",
        json={**submission, "epsilon": 0.05},
        headers={"Idempotency-Key": KEY},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_key_reuse"
    assert len(job_queue.enqueued) == 1


def test_key_reuse_raises_at_the_service_layer(job_queue, job_request, session):
    create_job(session, job_request, queue=job_queue, idempotency_key=KEY)

    altered = job_request.model_copy(update={"max_iterations": 5})
    with SessionLocal() as other, pytest.raises(IdempotencyKeyReuseError):
        create_job(other, altered, queue=job_queue, idempotency_key=KEY)


def test_submissions_without_a_key_are_independent(client, submission, job_queue):
    """A NULL idempotency_key must not collide with another NULL."""
    first = client.post("/v1/jobs", json=submission)
    second = client.post("/v1/jobs", json=submission)

    assert first.json()["job_id"] != second.json()["job_id"]
    assert len(job_queue.enqueued) == 2
    assert all(isinstance(job_id, uuid.UUID) for job_id in job_queue.enqueued)
