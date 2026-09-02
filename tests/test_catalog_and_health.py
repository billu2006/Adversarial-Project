"""Discovery endpoints and the health check."""

from __future__ import annotations

from benchmark.catalog import ATTACK_NAMES, MODEL_NAMES


def test_models_endpoint_publishes_the_whitelist(client):
    body = client.get("/v1/models").json()

    assert [item["name"] for item in body["items"]] == list(MODEL_NAMES)
    # The checkpoints ship with the repo, so they should be reported present.
    assert all(item["available"] for item in body["items"])


def test_attacks_endpoint_publishes_limits(client, settings):
    body = client.get("/v1/attacks").json()

    assert [item["name"] for item in body["items"]] == list(ATTACK_NAMES)
    # Limits are advertised rather than discovered by trial and error.
    assert body["constraints"]["max_epsilon"] == settings.max_epsilon
    assert body["constraints"]["max_iterations"] == settings.max_iterations_limit


def test_catalog_requires_the_api_key(client):
    assert client.get("/v1/models", headers={"X-API-Key": "nope"}).status_code == 401


def test_healthz_reports_dependencies(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "database": "ok", "queue": "ok", "version": body["version"]}


def test_healthz_is_unauthenticated(client):
    """An orchestrator's probe must not need the shared secret."""
    response = client.get("/healthz", headers={"X-API-Key": ""})

    assert response.status_code == 200


def test_healthz_is_503_when_the_queue_is_down(client, job_queue):
    job_queue.healthy = False

    response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "database": "ok",
        "queue": "unavailable",
        "version": response.json()["version"],
    }
