"""Handing work to the worker.

The API only ever talks to this module, never to Redis or RQ directly, for two
reasons: the tests can swap in a recording implementation without a Redis
server, and a single-container demo can run jobs inline. The Redis
implementation is the one that ships.

The RQ job id is set to the database job id. That is not cosmetic - it is what
makes cancellation possible without a second lookup table, and it means a
requeue can never produce two queue entries for one job.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from service.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Import path of the worker entrypoint. Referenced as a string so the API
#: process never imports torch just to enqueue a job.
TASK_PATH = "service.worker.tasks.run_benchmark_job"


class JobQueue(Protocol):
    """What the API needs from a queue, and nothing more."""

    def enqueue(self, job_id: uuid.UUID, request_id: str | None = None) -> None: ...

    def cancel(self, job_id: uuid.UUID) -> bool:
        """Remove a not-yet-started job. True if it was still in the queue."""

    def ping(self) -> bool:
        """Liveness check for /healthz."""


class RedisQueue:
    """The production path: RQ over Redis."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._connection = None
        self._queue = None

    @property
    def connection(self):
        # Connect lazily so importing the app does not require a live Redis.
        if self._connection is None:
            import redis

            self._connection = redis.Redis.from_url(self._settings.redis_url)
        return self._connection

    @property
    def queue(self):
        if self._queue is None:
            from rq import Queue

            self._queue = Queue(
                self._settings.queue_name,
                connection=self.connection,
                default_timeout=self._settings.job_timeout_seconds,
            )
        return self._queue

    def enqueue(self, job_id: uuid.UUID, request_id: str | None = None) -> None:
        self.queue.enqueue(
            TASK_PATH,
            kwargs={"job_id": str(job_id), "request_id": request_id},
            job_id=str(job_id),
            # Hard wall-clock cap. RQ raises inside the worker at this point,
            # which the task turns into a `failed` job rather than a hang.
            job_timeout=self._settings.job_timeout_seconds,
            # Keep finished/failed metadata around briefly for debugging; the
            # database, not Redis, is the record of truth.
            result_ttl=3600,
            failure_ttl=86400,
        )
        logger.info("Job enqueued", extra={"job_id": str(job_id)})

    def cancel(self, job_id: uuid.UUID) -> bool:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job

        try:
            job = Job.fetch(str(job_id), connection=self.connection)
        except NoSuchJobError:
            # Already consumed, expired, or never made it to Redis. The database
            # transition is what actually cancels the job; this is best effort.
            return False
        job.cancel()
        return True

    def ping(self) -> bool:
        try:
            return bool(self.connection.ping())
        except Exception:
            logger.warning("Redis ping failed", exc_info=True)
            return False


class InlineQueue:
    """Runs the job synchronously in the calling process.

    Only for tests and for a laptop demo without Redis: it makes ``POST /v1/jobs``
    block for the entire benchmark, which is precisely the behaviour the whole
    project exists to avoid.
    """

    def enqueue(self, job_id: uuid.UUID, request_id: str | None = None) -> None:
        from service.worker.tasks import run_benchmark_job

        run_benchmark_job(job_id=str(job_id), request_id=request_id)

    def cancel(self, job_id: uuid.UUID) -> bool:
        return False

    def ping(self) -> bool:
        return True


_queue: JobQueue | None = None


def get_queue() -> JobQueue:
    """Process-wide queue singleton, chosen by configuration."""
    global _queue
    if _queue is None:
        settings = get_settings()
        _queue = InlineQueue() if settings.queue_backend == "inline" else RedisQueue(settings)
    return _queue


def set_queue(queue: JobQueue | None) -> None:
    """Override the singleton. Used by the test suite and by the worker's tests."""
    global _queue
    _queue = queue
