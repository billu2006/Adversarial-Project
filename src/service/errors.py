"""One error shape for every failure path.

A client should never have to parse two different error formats, and should
never see a bare 500 with an HTML body. Every handler below renders::

    {"error": {"code": "...", "message": "...", "details": {...}}}

``code`` is a stable machine-readable string; ``message`` is for humans;
``details`` carries whatever would otherwise force the caller to guess (the list
of supported models, which fields failed validation, the job's current status).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base class for errors we raise deliberately and can describe precisely."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "bad_request"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(APIError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class UnsupportedModelError(APIError):
    code = "unsupported_model"


class UnsupportedAttackError(APIError):
    code = "unsupported_attack"


class InvalidRequestError(APIError):
    code = "invalid_request"


class ConflictError(APIError):
    """State conflict: results asked for too early, cancel of a running job."""

    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class IdempotencyKeyReuseError(ConflictError):
    code = "idempotency_key_reuse"


class ResultsNotReadyError(ConflictError):
    code = "results_not_ready"


class JobNotCancellableError(ConflictError):
    code = "job_not_cancellable"


class UnauthorizedError(APIError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"


class CapacityError(APIError):
    """Backpressure: the queue is full enough that we refuse new work."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "capacity_exceeded"


def error_body(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


#: Error responses worth advertising in the OpenAPI document. Without these the
#: /docs page lists only the 200, and a caller has to discover the failure
#: shapes by triggering them.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Invalid request, or a model/attack outside the whitelist"},
    401: {"description": "Missing or invalid API key"},
    404: {"description": "No such job"},
    409: {"description": "State conflict: results not ready, or job not cancellable"},
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so no route can leak a non-envelope error response."""

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default 422 body is a bare list; flatten it into the
        # envelope so clients only ever parse one shape.
        fields = [
            {
                "field": ".".join(str(part) for part in err["loc"][1:]) or str(err["loc"][0]),
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_body(
                "invalid_request", "The request body failed validation.", {"fields": fields}
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(
            exc.status_code, "http_error"
        )
        return JSONResponse(status_code=exc.status_code, content=error_body(code, str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the detail, return none of it: an unexpected exception message can
        # contain connection strings or row data.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("internal_error", "An unexpected error occurred."),
        )
