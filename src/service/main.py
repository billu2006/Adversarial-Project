"""FastAPI application factory.

Assembles configuration, logging, middleware, error handlers and routers. The
factory form (rather than a module-level ``app = FastAPI()`` with side effects)
is what lets the tests build an app against a throwaway database.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI

from service import __version__
from service.config import get_settings
from service.errors import register_exception_handlers
from service.logging_config import configure_logging
from service.middleware import RequestContextMiddleware
from service.routers import catalog, health, jobs
from service.security import require_api_key

logger = logging.getLogger(__name__)

DESCRIPTION = """
Submit adversarial-robustness benchmarks against a whitelist of pretrained
Fashion-MNIST models and collect the results asynchronously.

A benchmark takes minutes, so `POST /v1/jobs` returns `202 Accepted` with a job
id; poll `GET /v1/jobs/{id}` and fetch `GET /v1/jobs/{id}/results` once the job
has succeeded.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.json_logs)

    app = FastAPI(
        title="Adversarial Robustness Benchmarking Service",
        description=DESCRIPTION,
        version=__version__,
        # The path is versioned from the first commit: adding /v2 later is
        # cheap, retrofitting a version onto published unversioned URLs is not.
        openapi_url="/openapi.json",
        docs_url="/docs",
    )

    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    # /healthz is deliberately unauthenticated: an orchestrator probing it
    # should not need the shared key, and it exposes nothing but up/down.
    app.include_router(health.router)
    app.include_router(jobs.router, dependencies=[Depends(require_api_key)])
    app.include_router(catalog.router, dependencies=[Depends(require_api_key)])

    logger.info(
        "Application started",
        extra={"environment": settings.environment, "queue_backend": settings.queue_backend},
    )
    return app


app = create_app()
