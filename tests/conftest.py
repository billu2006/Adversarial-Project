"""Test fixtures.

The suite runs against SQLite by default so a clean clone can run ``pytest``
with no Docker daemon; point ``TEST_DATABASE_URL`` at a Postgres instance (as CI
does) to exercise the real dialect, the native enum and ``ON DELETE CASCADE``.

Environment variables are set *before* the service modules are imported, because
``service.database`` builds its engine at import time from the settings.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="benchmark-tests-"))

os.environ.setdefault("ENVIRONMENT", "ci")
os.environ.setdefault(
    "DATABASE_URL", os.environ.get("TEST_DATABASE_URL", f"sqlite:///{_TEST_DB_DIR}/test.db")
)
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("JSON_LOGS", "false")
# Never let a test reach a real Redis; every test supplies its own queue double.
os.environ.setdefault("QUEUE_BACKEND", "inline")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from service import queue as queue_module  # noqa: E402
from service.config import get_settings  # noqa: E402
from service.database import SessionLocal, engine  # noqa: E402
from service.main import create_app  # noqa: E402
from service.models import Base  # noqa: E402
from service.schemas import JobCreateRequest  # noqa: E402

API_KEY = os.environ["API_KEY"]
AUTH_HEADERS = {"X-API-Key": API_KEY}


class RecordingQueue:
    """Queue double: remembers what was enqueued instead of running it.

    Using this rather than a live Redis keeps the API tests fast and, more
    usefully, lets them assert on *how many* enqueues happened - which is the
    whole point of the idempotency tests.
    """

    def __init__(self) -> None:
        self.enqueued: list[uuid.UUID] = []
        self.cancelled: list[uuid.UUID] = []
        self.healthy = True
        self.fail_on_enqueue = False

    def enqueue(self, job_id: uuid.UUID, request_id: str | None = None) -> None:
        if self.fail_on_enqueue:
            raise RuntimeError("redis is down")
        self.enqueued.append(job_id)

    def cancel(self, job_id: uuid.UUID) -> bool:
        self.cancelled.append(job_id)
        return True

    def ping(self) -> bool:
        return self.healthy


@pytest.fixture(scope="session", autouse=True)
def _database_schema() -> Iterator[None]:
    """Create the schema once for the whole session.

    On Postgres this could run Alembic instead; ``create_all`` is used so the
    ORM metadata is what is under test, and CI runs the migration separately to
    prove the two agree.
    """
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    """Truncate between tests so ordering and listing assertions are stable."""
    yield
    with SessionLocal() as session:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def job_queue() -> Iterator[RecordingQueue]:
    """Install a recording queue for the duration of a test."""
    recording = RecordingQueue()
    queue_module.set_queue(recording)
    yield recording
    queue_module.set_queue(None)


@pytest.fixture
def session() -> Iterator[Session]:
    with SessionLocal() as db:
        yield db


@pytest.fixture
def client(job_queue: RecordingQueue) -> Iterator[TestClient]:
    """Authenticated-by-default test client against a fresh app instance."""
    app = create_app()
    with TestClient(app) as test_client:
        test_client.headers.update(AUTH_HEADERS)
        yield test_client


@pytest.fixture
def submission() -> dict:
    """A valid submission body, used as the base for most requests."""
    return {
        "model_name": "fmnist-mlp-defender-0",
        "attacks": ["fgsm", "pgd"],
        "epsilon": 0.1,
        "max_iterations": 10,
    }


@pytest.fixture
def job_request(submission: dict) -> JobCreateRequest:
    return JobCreateRequest(**submission)
