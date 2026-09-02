"""ORM models for the job lifecycle.

Schema notes worth defending:

* ``NUMERIC`` rather than ``FLOAT`` for accuracy, epsilon and perturbation norm.
  Results are the product here; two clients comparing scores must get bit-identical
  values, and binary floats do not round-trip through JSON and SQL predictably.
* ``idempotency_key`` carries a UNIQUE constraint, so duplicate submission is
  rejected by the database rather than by a read-then-write check in Python that
  loses to a concurrent request.
* ``ON DELETE CASCADE`` on results, so a result can never outlive its job.
* ``(status, created_at DESC)`` composite index. Status is the equality
  predicate and created_at the range/sort key, so status must come first for the
  index to serve both "oldest queued job" and the filtered listing endpoint.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from service.db_types import GUID, JSONColumn


class Base(DeclarativeBase):
    pass


class JobStatus(enum.StrEnum):
    """The job lifecycle. Terminal states are succeeded/failed/cancelled."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATUSES


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})

#: Stored as the lowercase values above, not the Python member names, so the
#: database enum reads the same as the API.
JobStatusType = Enum(
    JobStatus,
    name="job_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(GUID, primary_key=True, default=uuid.uuid4)

    # Client-supplied replay protection. NULL is allowed and does not collide,
    # so idempotency stays opt-in.
    idempotency_key: Mapped[str | None] = mapped_column(Text, unique=True)
    # Hash of the request body that first used the key. A replay with the same
    # key but a different body is a client bug, and we surface it as a 409
    # rather than silently returning someone else's job.
    request_fingerprint: Mapped[str | None] = mapped_column(String(64))

    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    attacks: Mapped[list[str]] = mapped_column(JSONColumn, nullable=False)
    epsilon: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=20)

    status: Mapped[JobStatus] = mapped_column(
        JobStatusType, nullable=False, default=JobStatus.QUEUED, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text)

    # Set in Python (always UTC) rather than relying on the server default, so
    # the value is available on the returned object without a refresh and reads
    # back identically on Postgres and SQLite. The server default stays as a
    # backstop for rows inserted outside the ORM.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    results: Mapped[list[JobResult]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobResult.id"
    )

    __table_args__ = (
        Index("idx_jobs_status_created", "status", created_at.desc()),
        CheckConstraint("max_iterations > 0", name="ck_jobs_max_iterations_positive"),
        CheckConstraint("epsilon > 0", name="ck_jobs_epsilon_positive"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Job {self.id} {self.status.value} {self.model_name}>"


class JobResult(Base):
    __tablename__ = "job_results"

    # BIGSERIAL on Postgres. SQLite only auto-increments a plain INTEGER
    # primary key, so the variant keeps the test dialect working.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        GUID, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )

    attack_name: Mapped[str] = mapped_column(Text, nullable=False)
    robust_accuracy: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    mean_nll: Mapped[float | None] = mapped_column(Numeric(10, 6))
    perturbation_norm: Mapped[float | None] = mapped_column(Numeric(8, 6))
    runtime_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    job: Mapped[Job] = relationship(back_populates="results")

    __table_args__ = (
        # One row per attack per job: makes the worker's writes naturally
        # idempotent if a job is ever retried.
        UniqueConstraint("job_id", "attack_name", name="uq_job_results_job_attack"),
        Index("idx_results_job", "job_id"),
    )
