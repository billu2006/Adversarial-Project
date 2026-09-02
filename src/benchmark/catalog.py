"""What models and attacks exist - deliberately free of any PyTorch import.

The API validates a submission against these names, renders ``/v1/models`` and
``/v1/attacks`` from them, and never loads a model or runs an attack. Keeping
the catalogue separate from the implementations is what lets the API image ship
without PyTorch at all: importing ~800MB of tensor library to answer "is
'fgsm' a valid attack name?" would be absurd, and it would slow every API start
down to the worker's.

``benchmark.models`` and ``benchmark.attacks`` import from here and add the
parts that need torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Where the pretrained defender checkpoints live. Overridable so the worker
#: container can mount them elsewhere, but never taken from a request.
WEIGHTS_DIR = Path(os.environ.get("BENCHMARK_WEIGHTS_DIR", "models/defenders")).resolve()


# --------------------------------------------------------------------------- #
# Models - the whitelist. See benchmark.models for why this is a security
# boundary rather than a convenience.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """A whitelisted model: the public name and the file it resolves to."""

    name: str
    #: Path relative to WEIGHTS_DIR. Never influenced by user input.
    weights_filename: str
    architecture: str
    dataset: str
    description: str

    @property
    def weights_path(self) -> Path:
        return WEIGHTS_DIR / self.weights_filename


def _defender(index: int) -> ModelSpec:
    return ModelSpec(
        name=f"fmnist-mlp-defender-{index}",
        weights_filename=f"{index}.pt",
        architecture="mlp-784-128-64-32-10",
        dataset="fashion-mnist",
        description=(
            f"Reference defender #{index}: adversarially trained Fashion-MNIST MLP "
            "from the source framework's defender pool."
        ),
    )


MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in (_defender(i) for i in range(10))}

#: Whitelisted model names, in a stable order (used by the API catalogue).
MODEL_NAMES: tuple[str, ...] = tuple(MODELS)


def list_models() -> list[ModelSpec]:
    return list(MODELS.values())


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError as exc:
        raise KeyError(f"Unsupported model: {name!r}") from exc


def is_supported_model(name: str) -> bool:
    return name in MODELS


# --------------------------------------------------------------------------- #
# Attacks - metadata only; the implementations live in benchmark.attacks.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttackInfo:
    """Everything ``GET /v1/attacks`` needs to describe an attack."""

    name: str
    description: str
    #: False for single-step attacks, where max_iterations has no effect.
    uses_iterations: bool
    #: Rough cost per batch relative to FGSM. Documentation, not a guarantee.
    relative_cost: str


ATTACK_CATALOG: dict[str, AttackInfo] = {
    info.name: info
    for info in (
        AttackInfo(
            name="fgsm",
            description="Fast Gradient Sign Method - single-step, cheapest baseline.",
            uses_iterations=False,
            relative_cost="1x",
        ),
        AttackInfo(
            name="pgd",
            description="Projected Gradient Descent - iterative FGSM with a random start.",
            uses_iterations=True,
            relative_cost="~iterations x",
        ),
        AttackInfo(
            name="cw",
            description="Carlini & Wagner - margin-minimising attack optimised in tanh space.",
            uses_iterations=True,
            relative_cost="~2x iterations",
        ),
        AttackInfo(
            name="lbfgs",
            description="L-BFGS - second-order optimisation, strong but expensive.",
            uses_iterations=True,
            relative_cost="~5x iterations",
        ),
        AttackInfo(
            name="ensemble",
            description="Best-of PGD, C&W and L-BFGS per batch. Strongest and slowest.",
            uses_iterations=True,
            relative_cost="~8x iterations",
        ),
    )
}

#: Attack names in a stable, documented order (used by the API catalogue).
ATTACK_NAMES: tuple[str, ...] = tuple(ATTACK_CATALOG)


def list_attack_info() -> list[AttackInfo]:
    return list(ATTACK_CATALOG.values())


def is_supported_attack(name: str) -> bool:
    return name in ATTACK_CATALOG
