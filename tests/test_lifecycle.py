"""Polling, results, cancellation - the states a job moves through."""

from __future__ import annotations

import uuid

from service.jobs import get_job, mark_running, mark_succeeded
from service.models import JobResult, JobStatus


def _submit(client, submission) -> str:
    return client.post("/v1/jobs", json=submission).json()["job_id"]


def test_poll_returns_status_and_timestamps(client, submission):
    job_id = _submit(client, submission)

    response = client.get(f"/v1/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["created_at"]
    # Not started, so these must be absent rather than zero-valued.
    assert body["started_at"] is None
    assert body["finished_at"] is None


def test_unknown_job_is_404(client):
    response = client.get(f"/v1/jobs/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_malformed_job_id_is_a_400_not_a_500(client):
    response = client.get("/v1/jobs/not-a-uuid")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_results_before_completion_are_409(client, submission):
    job_id = _submit(client, submission)

    response = client.get(f"/v1/jobs/{job_id}/results")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "results_not_ready"
    # Tell the client what to do next instead of just refusing.
    assert error["details"]["status"] == "queued"
    assert error["details"]["poll"] == f"/v1/jobs/{job_id}"


def test_results_of_a_failed_job_are_409_with_the_reason(client, submission, session):
    job_id = _submit(client, submission)
    job = get_job(session, uuid.UUID(job_id))
    from service.jobs import mark_failed

    mark_failed(session, job, "RuntimeError: the worker exploded")

    response = client.get(f"/v1/jobs/{job_id}/results")

    assert response.status_code == 409
    details = response.json()["error"]["details"]
    assert details["status"] == "failed"
    assert "exploded" in details["error_message"]


def test_results_after_success(client, submission, session):
    job_id = _submit(client, submission)
    job = mark_running(session, uuid.UUID(job_id))
    mark_succeeded(
        session,
        job,
        [
            JobResult(
                job_id=job.id,
                attack_name="fgsm",
                robust_accuracy=0.4823,
                mean_nll=1.234567,
                perturbation_norm=0.1,
                runtime_ms=1500,
            ),
            JobResult(
                job_id=job.id,
                attack_name="pgd",
                robust_accuracy=0.3211,
                mean_nll=2.5,
                perturbation_norm=0.1,
                runtime_ms=9000,
            ),
        ],
    )

    response = client.get(f"/v1/jobs/{job_id}/results")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert [row["attack_name"] for row in body["results"]] == ["fgsm", "pgd"]
    # NUMERIC round-trips exactly; a float column would not guarantee this.
    assert body["results"][0]["robust_accuracy"] == 0.4823
    # Rounded to the stored precision rather than carrying float noise.
    assert body["defence_score"] == 0.4017


def test_cancel_a_queued_job(client, submission, job_queue, session):
    job_id = _submit(client, submission)

    response = client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["finished_at"] is not None
    # The queue entry is removed too, so a worker never picks it up.
    assert uuid.UUID(job_id) in job_queue.cancelled


def test_cancelling_a_running_job_is_409(client, submission, session):
    job_id = _submit(client, submission)
    mark_running(session, uuid.UUID(job_id))

    response = client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "job_not_cancellable"
    assert error["details"]["status"] == "running"


def test_cancelling_twice_is_409(client, submission):
    job_id = _submit(client, submission)
    client.delete(f"/v1/jobs/{job_id}")

    response = client.delete(f"/v1/jobs/{job_id}")

    assert response.status_code == 409


def test_cancel_is_a_conditional_update(client, submission, session):
    """The transition must not be read-modify-write.

    A worker that claims the job between the read and the write would otherwise
    end up running a job the client believes was cancelled.
    """
    job_id = _submit(client, submission)
    identifier = uuid.UUID(job_id)

    # Simulate the worker winning the race.
    mark_running(session, identifier)

    response = client.delete(f"/v1/jobs/{job_id}")
    assert response.status_code == 409
    assert get_job(session, identifier).status is JobStatus.RUNNING
