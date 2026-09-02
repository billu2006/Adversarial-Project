"""Shared FastAPI dependencies.

Wrapping the queue and settings in dependencies (rather than importing them in
the routers) is what lets the test suite substitute a recording queue and an
in-memory database with ``app.dependency_overrides``.
"""

from __future__ import annotations

from service.config import Settings, get_settings
from service.queue import JobQueue, get_queue


def settings_dependency() -> Settings:
    return get_settings()


def queue_dependency() -> JobQueue:
    return get_queue()
