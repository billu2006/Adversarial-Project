"""``/v1/models`` and ``/v1/attacks`` - what a client is allowed to ask for.

These are not decoration: the whitelist *is* the security model, so publishing
it is how a client discovers what exists instead of probing with guesses.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from benchmark.catalog import list_attack_info, list_models
from service.config import Settings
from service.dependencies import settings_dependency
from service.schemas import (
    AttackListResponse,
    AttackResource,
    ModelListResponse,
    ModelResource,
)

router = APIRouter(prefix="/v1", tags=["catalog"])


@router.get("/models", response_model=ModelListResponse, summary="Supported models")
def get_models() -> ModelListResponse:
    return ModelListResponse(
        items=[
            ModelResource(
                name=spec.name,
                architecture=spec.architecture,
                dataset=spec.dataset,
                description=spec.description,
                # Reported per deployment: the checkpoints are mounted, not
                # baked in, so a misconfigured volume shows up here rather than
                # as a failed job ten minutes later.
                available=spec.weights_path.is_file(),
            )
            for spec in list_models()
        ]
    )


@router.get("/attacks", response_model=AttackListResponse, summary="Available attacks")
def get_attacks(settings: Settings = Depends(settings_dependency)) -> AttackListResponse:
    return AttackListResponse(
        items=[
            AttackResource(
                name=spec.name,
                description=spec.description,
                uses_iterations=spec.uses_iterations,
                relative_cost=spec.relative_cost,
            )
            for spec in list_attack_info()
        ],
        constraints={
            "max_epsilon": settings.max_epsilon,
            "max_iterations": settings.max_iterations_limit,
            "max_attacks_per_job": settings.max_attacks_per_job,
            "evaluation_samples": settings.benchmark_max_samples,
        },
    )
