"""Static API-key authentication.

User accounts are explicitly out of scope (see the README). What is *not*
optional is that a public deployment of a compute-heavy endpoint needs some
gate, so every /v1 route requires a shared key. The comparison is
constant-time; a plain ``==`` on a secret is a timing oracle, and it costs
nothing to avoid.
"""

from __future__ import annotations

import secrets

from fastapi import Header

from service.config import Settings, get_settings
from service.errors import UnauthorizedError

API_KEY_HEADER = "X-API-Key"


def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency guarding the versioned routes."""
    settings: Settings = get_settings()
    if not settings.require_api_key:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise UnauthorizedError("A valid API key is required.", {"header": API_KEY_HEADER})
