"""Submitting a job: the accepted path and every way it can be rejected."""

from __future__ import annotations

import uuid

from benchmark.catalog import ATTACK_NAMES, MODEL_NAMES


def test_submit_returns_202_with_a_queued_job(client, submission, job_queue):
    response = client.post("/v1/jobs", json=submission)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert uuid.UUID(body["job_id"])
    assert body["model_name"] == submission["model_name"]
    assert body["attacks"] == submission["attacks"]
    assert body["links"]["self"] == f"/v1/jobs/{body['job_id']}"
    assert body["links"]["results"] == f"/v1/jobs/{body['job_id']}/results"
    # 202 means "accepted, not done": the client is told where to poll, and the
    # work really was handed to the worker.
    assert response.headers["Location"] == f"/v1/jobs/{body['job_id']}"
    assert job_queue.enqueued == [uuid.UUID(body["job_id"])]


def test_submit_is_not_blocking(client, submission, job_queue):
    """The response must not wait on the benchmark - nothing ran in-process."""
    client.post("/v1/jobs", json=submission)
    assert len(job_queue.enqueued) == 1


def test_unknown_model_is_rejected_with_the_whitelist(client, submission, job_queue):
    response = client.post("/v1/jobs", json={**submission, "model_name": "foo"})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_model"
    # The error tells the client what it *can* ask for, so discovering the
    # whitelist does not require reading the source.
    assert error["details"]["supported"] == list(MODEL_NAMES)
    assert job_queue.enqueued == []


def test_unknown_attack_is_rejected(client, submission):
    response = client.post("/v1/jobs", json={**submission, "attacks": ["fgsm", "nope"]})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_attack"
    assert error["details"]["supported"] == list(ATTACK_NAMES)


def test_epsilon_above_the_constraint_is_rejected(client, submission):
    # 0.11 is exactly the framework's forbidden boundary; anything at or above
    # it would invalidate the benchmark.
    response = client.post("/v1/jobs", json={**submission, "epsilon": 0.11})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_iteration_cap_is_enforced(client, submission):
    response = client.post("/v1/jobs", json={**submission, "max_iterations": 100_000})

    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "invalid_request"
    assert body["details"]["fields"][0]["field"] == "max_iterations"


def test_duplicate_attacks_are_rejected(client, submission):
    response = client.post("/v1/jobs", json={**submission, "attacks": ["fgsm", "fgsm"]})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_unknown_fields_are_rejected(client, submission):
    """extra="forbid" keeps a typo'd parameter from being silently ignored."""
    response = client.post("/v1/jobs", json={**submission, "epsilonn": 0.1})

    assert response.status_code == 400


def test_api_key_is_required(client, submission):
    response = client.post("/v1/jobs", json=submission, headers={"X-API-Key": "wrong"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_every_response_carries_a_request_id(client, submission):
    response = client.post("/v1/jobs", json=submission)
    assert response.headers["X-Request-ID"]


def test_a_queue_outage_fails_the_job_rather_than_stranding_it(
    client, submission, job_queue, session
):
    """A job nothing can consume must not sit in `queued` forever."""
    job_queue.fail_on_enqueue = True

    response = client.post("/v1/jobs", json=submission)

    assert response.status_code == 202
    body = response.json()
    from service.jobs import get_job

    job = get_job(session, uuid.UUID(body["job_id"]))
    assert job.status.value == "failed"
    assert "enqueue" in job.error_message
