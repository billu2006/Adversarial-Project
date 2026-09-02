"""Initial schema: jobs and job_results.

Revision ID: 0001
Revises:
Create Date: 2026-09-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Declared once and reused, with create_type=False on the column so the enum is
# created exactly once (by the explicit create() below) rather than racing.
job_status = postgresql.ENUM(
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    name="job_status",
    create_type=False,
)


def upgrade() -> None:
    job_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # UNIQUE, so duplicate submission is rejected by the database rather
        # than by an application-level check that loses to concurrency.
        sa.Column("idempotency_key", sa.Text(), nullable=True, unique=True),
        sa.Column("request_fingerprint", sa.String(64), nullable=True),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("attacks", postgresql.JSONB(), nullable=False),
        # NUMERIC, not FLOAT: these values are the product, and must compare exactly.
        sa.Column("epsilon", sa.Numeric(6, 4), nullable=False),
        sa.Column("max_iterations", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("status", job_status, nullable=False, server_default="queued"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("max_iterations > 0", name="ck_jobs_max_iterations_positive"),
        sa.CheckConstraint("epsilon > 0", name="ck_jobs_epsilon_positive"),
    )

    # Status first, created_at second: status is the equality predicate and
    # created_at the ordering key, which is the order the worker's "oldest
    # queued job" lookup and the filtered listing endpoint both need. Reversed,
    # neither query could use the index without a full scan of one status.
    op.create_index(
        "idx_jobs_status_created", "jobs", ["status", sa.text("created_at DESC")]
    )

    op.create_table(
        "job_results",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            # CASCADE, so a result can never outlive the job that produced it.
            sa.ForeignKey("jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("attack_name", sa.Text(), nullable=False),
        sa.Column("robust_accuracy", sa.Numeric(5, 4), nullable=False),
        sa.Column("mean_nll", sa.Numeric(10, 6), nullable=True),
        sa.Column("perturbation_norm", sa.Numeric(8, 6), nullable=True),
        sa.Column("runtime_ms", sa.Integer(), nullable=False),
        sa.UniqueConstraint("job_id", "attack_name", name="uq_job_results_job_attack"),
    )
    op.create_index("idx_results_job", "job_results", ["job_id"])


def downgrade() -> None:
    op.drop_index("idx_results_job", table_name="job_results")
    op.drop_table("job_results")
    op.drop_index("idx_jobs_status_created", table_name="jobs")
    op.drop_table("jobs")
    job_status.drop(op.get_bind(), checkfirst=True)
