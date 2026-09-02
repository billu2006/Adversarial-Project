"""Pydantic request/response models - the wire contract.

Request models carry the limits that protect the worker (epsilon ceiling,
iteration cap, attack-count cap) so a hostile payload is rejected at the edge,
before a database row or a queue entry exists. Membership checks against the
model/attack whitelists happen in the service layer instead, because they
produce a richer error (the list of what *is* supported) than a schema
violation can.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchmark.constants import MAX_EPSILON, MAX_ITERATIONS_LIMIT
from service.models import JobStatus

Epsilon = Annotated[float, Field(gt=0, le=MAX_EPSILON)]
MaxIterations = Annotated[int, Field(ge=1, le=MAX_ITERATIONS_LIMIT)]


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        ..., description="A model from GET /v1/models.", examples=["fmnist-mlp-defender-0"]
    )
    attacks: list[str] = Field(
        ..., min_length=1, max_length=8, description="Attacks from GET /v1/attacks."
    )
    epsilon: Epsilon = Field(
        ...,
        description=(
            f"L-inf perturbation budget. Must be <= {MAX_EPSILON}; the framework's "
            "constraint is a strict < 0.11."
        ),
        examples=[0.1],
    )
    max_iterations: MaxIterations = Field(
        default=20, description="Iteration budget for iterative attacks."
    )

    @field_validator("attacks")
    @classmethod
    def _reject_duplicates(cls, value: list[str]) -> list[str]:
        # job_results is unique on (job_id, attack_name), so a duplicate attack
        # would fail on insert deep inside the worker. Catch it at the edge.
        if len(set(value)) != len(value):
            raise ValueError("attacks must not contain duplicates")
        return value


def _as_utc(value: datetime | None) -> datetime | None:
    """Stamp UTC on a naive timestamp.

    Postgres hands back timezone-aware values; SQLite drops the offset. Every
    timestamp we write is UTC, so re-attaching it here means the API emits the
    same ``...+00:00`` form whichever database is underneath.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


class JobLinks(BaseModel):
    self: str
    results: str


class JobResource(BaseModel):
    """The job as the client sees it. Never exposes internal columns."""

    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID
    status: JobStatus
    model_name: str
    attacks: list[str]
    epsilon: float
    max_iterations: int
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    links: JobLinks

    @classmethod
    def from_job(cls, job: Any) -> JobResource:
        return cls(
            job_id=job.id,
            status=job.status,
            model_name=job.model_name,
            attacks=list(job.attacks),
            # Numeric comes back as Decimal; float is what JSON can carry.
            epsilon=float(job.epsilon),
            max_iterations=job.max_iterations,
            error_message=job.error_message,
            created_at=_as_utc(job.created_at),
            started_at=_as_utc(job.started_at),
            finished_at=_as_utc(job.finished_at),
            links=JobLinks(self=f"/v1/jobs/{job.id}", results=f"/v1/jobs/{job.id}/results"),
        )


class AttackResultResource(BaseModel):
    attack_name: str
    robust_accuracy: float
    mean_nll: float | None = None
    perturbation_norm: float | None = None
    runtime_ms: int

    @classmethod
    def from_row(cls, row: Any) -> AttackResultResource:
        def _float(value: Decimal | float | None) -> float | None:
            return None if value is None else float(value)

        return cls(
            attack_name=row.attack_name,
            robust_accuracy=float(row.robust_accuracy),
            mean_nll=_float(row.mean_nll),
            perturbation_norm=_float(row.perturbation_norm),
            runtime_ms=row.runtime_ms,
        )


class JobResultsResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    model_name: str
    epsilon: float
    max_iterations: int
    results: list[AttackResultResource]
    #: Mean robust accuracy across attacks - the "defence score" the assignment
    #: graded a defence on.
    defence_score: float | None = None


class JobListResponse(BaseModel):
    items: list[JobResource]
    #: Opaque keyset cursor; pass back as ?cursor= for the next page. Null on
    #: the last page.
    next_cursor: str | None = None


class ModelResource(BaseModel):
    name: str
    architecture: str
    dataset: str
    description: str
    available: bool = Field(
        description="False when the checkpoint is missing from this deployment."
    )


class ModelListResponse(BaseModel):
    items: list[ModelResource]


class AttackResource(BaseModel):
    name: str
    description: str
    uses_iterations: bool
    relative_cost: str


class AttackListResponse(BaseModel):
    items: list[AttackResource]
    constraints: dict[str, float | int]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]
    queue: Literal["ok", "unavailable"]
    version: str
