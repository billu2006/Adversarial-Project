"""Cursor-paginated listing."""

from __future__ import annotations

import pytest

from service.jobs import mark_running


def _submit_many(client, submission, count: int) -> list[str]:
    return [client.post("/v1/jobs", json=submission).json()["job_id"] for _ in range(count)]


def test_listing_is_newest_first(client, submission):
    ids = _submit_many(client, submission, 3)

    items = client.get("/v1/jobs").json()["items"]

    assert [item["job_id"] for item in items] == list(reversed(ids))


def test_pagination_walks_every_job_exactly_once(client, submission):
    ids = _submit_many(client, submission, 5)

    seen: list[str] = []
    cursor = None
    for _page in range(5):  # bounded so a cursor bug cannot hang the suite
        params = {"limit": 2}
        if cursor:
            params["cursor"] = cursor
        body = client.get("/v1/jobs", params=params).json()
        seen.extend(item["job_id"] for item in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break

    assert cursor is None
    assert seen == list(reversed(ids))
    assert len(set(seen)) == 5


def test_last_page_has_no_cursor(client, submission):
    _submit_many(client, submission, 2)

    body = client.get("/v1/jobs", params={"limit": 10}).json()

    assert body["next_cursor"] is None


def test_status_filter(client, submission, session):
    import uuid

    ids = _submit_many(client, submission, 3)
    mark_running(session, uuid.UUID(ids[0]))

    running = client.get("/v1/jobs", params={"status": "running"}).json()["items"]
    queued = client.get("/v1/jobs", params={"status": "queued"}).json()["items"]

    assert [item["job_id"] for item in running] == [ids[0]]
    assert len(queued) == 2


def test_unknown_status_is_a_400(client):
    response = client.get("/v1/jobs", params={"status": "elsewhere"})

    assert response.status_code == 400


@pytest.mark.parametrize("cursor", ["not-base64", "Zm9v", ""])
def test_malformed_cursor_is_a_400(client, submission, cursor):
    _submit_many(client, submission, 1)

    response = client.get("/v1/jobs", params={"cursor": cursor})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
