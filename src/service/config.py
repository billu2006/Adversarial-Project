"""Runtime configuration, read once from the environment.

Every knob that differs between Docker Compose, CI and a bare `uvicorn` run
lives here, and every resource limit the security model depends on is a setting
rather than a literal buried in a handler.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from benchmark.constants import MAX_EPSILON, MAX_ITERATIONS_LIMIT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["local", "ci", "production"] = "local"
    log_level: str = "INFO"
    #: Emit one JSON object per log line. Off locally is easier to read, but the
    #: containers default to on so Cloud Logging can parse them.
    json_logs: bool = True

    database_url: str = "postgresql+psycopg://benchmark:benchmark@localhost:5432/benchmark"
    redis_url: str = "redis://localhost:6379/0"

    #: Single static key, checked on every /v1 route. Real user accounts are out
    #: of scope (see the README); this only keeps the deployment from being an
    #: open compute endpoint.
    api_key: str = "local-development-key"
    require_api_key: bool = True

    #: How the worker executes a job. RQ's default forks a "work-horse" process
    #: per job, which is what gives each benchmark a clean process and a
    #: killable timeout - the right default in the Linux container. It is not
    #: usable on macOS: once torch is imported, the Objective-C runtime aborts
    #: in any forked child that has not exec'd. "auto" picks fork on Linux and
    #: simple (in-process) elsewhere; set it explicitly to override.
    worker_class: Literal["auto", "fork", "simple"] = "auto"

    #: How the API hands work to the worker. "redis" is the real path; "inline"
    #: runs the job synchronously in-process and exists for tests and for
    #: single-container demos.
    queue_backend: Literal["redis", "inline"] = "redis"
    queue_name: str = "benchmarks"

    # --- Resource limits (see "Security considerations" in the README) --------
    #: Hard wall-clock cap on a single job. The worker kills the job at this
    #: point and the reaper fails anything that outlives it.
    job_timeout_seconds: int = 900
    #: Grace period past the timeout before the reaper declares a job abandoned.
    stale_job_grace_seconds: int = 120
    #: Ceiling on the per-request iteration count.
    max_iterations_limit: int = MAX_ITERATIONS_LIMIT
    #: Ceiling on the per-request epsilon. The engine clamps to this too.
    max_epsilon: float = MAX_EPSILON
    #: Number of attacks a single job may request.
    max_attacks_per_job: int = 5
    #: Evaluation-set size. Bounds the runtime of any single attack.
    benchmark_max_samples: int = 2048
    benchmark_batch_size: int = 128
    #: Refuse new submissions once this many jobs are queued or running. Crude
    #: backpressure, but it beats an unbounded queue.
    max_active_jobs: int = 50

    #: Cursor-paginated listing defaults.
    default_page_size: int = 20
    max_page_size: int = 100


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process."""
    return Settings()
