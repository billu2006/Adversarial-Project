"""Portable column types.

Production is PostgreSQL and the migrations emit native ``uuid``/``jsonb``. The
test suite defaults to SQLite so a clean clone can run ``pytest`` without Docker,
and these decorators let one set of ORM models serve both. Anything genuinely
Postgres-specific (the ``job_status`` enum type, ``ON DELETE CASCADE``) is
exercised by the CI job that runs the same tests against a real Postgres.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CHAR, JSON, Dialect, types
from sqlalchemy.dialects.postgresql import JSONB, UUID


class GUID(types.TypeDecorator):
    """UUID column: native ``uuid`` on Postgres, 32-char hex elsewhere."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(32))

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        return value if dialect.name == "postgresql" else value.hex

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


#: JSONB on Postgres (indexable, typed), plain JSON text elsewhere.
JSONColumn = JSON().with_variant(JSONB(), "postgresql")
