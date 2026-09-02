"""``/healthz`` - liveness that actually checks the dependencies.

A health check that only proves the process is running is worse than none: the
orchestrator keeps routing traffic to an instance that cannot reach its
database. This one touches both Postgres and Redis and reports 503 when either
is down, since the API cannot accept a job without both.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from service import __version__
from service.database import get_db
from service.dependencies import queue_dependency
from service.queue import JobQueue
from service.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness and readiness")
def healthz(
    response: Response,
    session: Session = Depends(get_db),
    queue: JobQueue = Depends(queue_dependency),
) -> HealthResponse:
    try:
        session.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        logger.warning("Database health check failed", exc_info=True)
        database_ok = False

    queue_ok = queue.ping()

    healthy = database_ok and queue_ok
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if healthy else "degraded",
        database="ok" if database_ok else "unavailable",
        queue="ok" if queue_ok else "unavailable",
        version=__version__,
    )
