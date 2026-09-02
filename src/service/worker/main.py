"""Worker entrypoint: ``python -m service.worker.main``.

Starts an RQ worker against the configured queue. Before consuming anything it
reaps jobs abandoned by a previous worker, which is what stops a container
restart from leaving rows stuck in ``running`` forever.
"""

from __future__ import annotations

import logging
import sys
import threading

import redis
from rq import SimpleWorker, Worker

from service.config import get_settings
from service.database import session_scope
from service.jobs import sweep_stale_jobs
from service.logging_config import configure_logging

logger = logging.getLogger(__name__)


def resolve_worker_class(preference: str = "auto", platform: str | None = None):
    """Choose between RQ's forking worker and its in-process one.

    RQ's default ``Worker`` forks a work-horse per job. That is what we want in
    the container: the benchmark gets a fresh process, a leaked tensor cannot
    accumulate across jobs, and a job that overruns its timeout can be killed
    outright.

    It cannot be used on macOS. PyTorch initialises Objective-C state on import,
    and Apple's runtime deliberately aborts in a forked child that has not
    exec'd - the worker dies with "+[MPSGraphObject initialize] may have been in
    progress in another thread when fork() was called" and the job is left
    behind for the reaper. So a native macOS run uses ``SimpleWorker``, which
    runs the job in the worker process itself.

    The trade-off is real and worth naming: SimpleWorker gives up per-job
    process isolation, and a job that wedges hard takes the worker with it.
    That is acceptable for local development and is not how the project runs in
    Docker.
    """
    if preference == "fork":
        return Worker
    if preference == "simple":
        return SimpleWorker

    platform = platform or sys.platform
    return Worker if platform.startswith("linux") else SimpleWorker


def reap_stale_jobs() -> int:
    """Fail anything left ``running`` by a worker that did not come back."""
    return sweep_stale_jobs_safely()


def start_reaper_thread(interval_seconds: int) -> threading.Thread:
    """Sweep for abandoned jobs on a timer, not just at startup.

    A worker pool where one member is killed would otherwise leave that job
    stuck in ``running`` until some other worker happened to restart. The thread
    is a daemon and only ever issues one small UPDATE, so it costs nothing while
    the worker is busy benchmarking.
    """

    def _loop() -> None:
        while True:
            try:
                reaped = sweep_stale_jobs_safely()
                if reaped:
                    logger.warning("Reaper failed abandoned jobs", extra={"reaped": reaped})
            except Exception:
                logger.exception("Reaper sweep failed")
            stop.wait(interval_seconds)

    stop = threading.Event()
    thread = threading.Thread(target=_loop, name="stale-job-reaper", daemon=True)
    thread.start()
    return thread


def sweep_stale_jobs_safely() -> int:
    with session_scope() as session:
        return sweep_stale_jobs(session)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.json_logs)

    reaped = reap_stale_jobs()
    logger.info(
        "Worker starting",
        extra={"queue": settings.queue_name, "reaped_stale_jobs": reaped},
    )

    start_reaper_thread(settings.stale_job_grace_seconds)

    connection = redis.Redis.from_url(settings.redis_url)
    worker_class = resolve_worker_class(settings.worker_class)
    logger.info("Worker class selected", extra={"worker_class": worker_class.__name__})
    worker = worker_class([settings.queue_name], connection=connection)
    # with_scheduler enables RQ's periodic machinery; the maintenance interval
    # also gives us a natural point to re-run the reaper.
    worker.work(with_scheduler=True, logging_level=settings.log_level)


if __name__ == "__main__":
    main()
