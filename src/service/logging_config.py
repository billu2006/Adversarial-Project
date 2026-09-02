"""Structured JSON logging with a request id that survives the queue hop.

A benchmark's life spans two processes, so "which log lines belong to this
submission?" is only answerable if the id travels with the job. The API mints
one per request (or honours an inbound ``X-Request-ID``), stores it in a
contextvar, and enqueues it alongside the job id; the worker sets the same
contextvar before it starts. Every line from either process then carries it.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Attributes LogRecord always has; anything else was passed via `extra=` and is
#: worth promoting to a top-level JSON field.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {
    "asctime",
    "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, which is what Cloud Logging wants."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Install a single stdout handler. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own colourised handlers; drop them so every line in
    # the container goes through one formatter.
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # Silence uvicorn's own access log: RequestContextMiddleware already emits
    # one structured line per request, with the request id and the duration
    # attached. Two access logs per request is noise, and uvicorn's fires after
    # the request context has been torn down, so its line has no id to show.
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_request_id(request_id: str | None) -> str:
    """Set the ambient request id, minting one if the caller did not supply it."""
    resolved = request_id or new_request_id()
    request_id_var.set(resolved)
    return resolved
