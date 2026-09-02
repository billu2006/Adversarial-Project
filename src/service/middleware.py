"""Request-scoped middleware: correlation ids and access logs."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from service.logging_config import bind_request_id, request_id_var

logger = logging.getLogger("service.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id for the life of the request and log the outcome.

    An inbound ``X-Request-ID`` is honoured so a trace started by a load
    balancer or a calling service survives into our logs; otherwise we mint one.
    It is echoed back on the response so a client can quote it in a bug report.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = bind_request_id(request.headers.get(REQUEST_ID_HEADER))
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # The exception handler produces the body; this makes sure the
            # failed request still appears in the access log with its timing.
            logger.exception(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            raise
        finally:
            request_id_var.set(None)

        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "Request handled",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
            },
        )
        return response
