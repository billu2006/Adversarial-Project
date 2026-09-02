"""SQLAlchemy engine, session factory and the FastAPI session dependency."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from service.config import get_settings

_settings = get_settings()


def _engine_kwargs(url: str) -> dict:
    """Connection arguments, which differ between Postgres and SQLite.

    Production is Postgres; SQLite appears only when the test suite runs without
    Docker. Rather than branch on the environment, branch on the URL - the code
    then behaves the same however it was pointed at a database.
    """
    if url.startswith("sqlite"):
        # The test client dispatches handlers on a threadpool, and SQLite
        # refuses cross-thread connection reuse unless told otherwise.
        return {"connect_args": {"check_same_thread": False}}
    return {
        # Guards against connections silently killed by a cloud proxy while
        # idle, which otherwise shows up as a failure on the first request after
        # a quiet period.
        "pool_pre_ping": True,
        "pool_size": 5,
        "max_overflow": 10,
    }


engine = create_engine(
    _settings.database_url, future=True, **_engine_kwargs(_settings.database_url)
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session for code outside a request (the worker, the reaper)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
