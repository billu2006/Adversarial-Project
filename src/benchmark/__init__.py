"""Adversarial-robustness benchmarking engine.

This package is the *library* half of the project: the pure-PyTorch code that
loads a whitelisted model, runs adversarial attacks against it and reports
robustness metrics. It knows nothing about HTTP, databases or queues, which is
what makes it callable from a worker process, a notebook or the tutorial
scripts in ``scripts/`` alike.

The names below are resolved lazily (PEP 562). ``benchmark.catalog`` and
``benchmark.constants`` are importable without PyTorch installed - which is what
the API process does - and eagerly re-exporting the engine here would break
that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.catalog import (
    ATTACK_CATALOG,
    ATTACK_NAMES,
    MODEL_NAMES,
    MODELS,
    AttackInfo,
    ModelSpec,
    get_model_spec,
    is_supported_attack,
    is_supported_model,
    list_attack_info,
    list_models,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from benchmark.attacks import ATTACKS, AttackSpec, get_attack, list_attacks
    from benchmark.engine import AttackResult, run_benchmark
    from benchmark.models import DefenderNet, load_model

#: Attribute name -> the torch-dependent module that defines it.
_LAZY_EXPORTS = {
    "ATTACKS": "benchmark.attacks",
    "AttackSpec": "benchmark.attacks",
    "get_attack": "benchmark.attacks",
    "list_attacks": "benchmark.attacks",
    "AttackResult": "benchmark.engine",
    "run_benchmark": "benchmark.engine",
    "DefenderNet": "benchmark.models",
    "load_model": "benchmark.models",
}

__all__ = [
    "ATTACKS",
    "ATTACK_CATALOG",
    "ATTACK_NAMES",
    "MODELS",
    "MODEL_NAMES",
    "AttackInfo",
    "AttackResult",
    "AttackSpec",
    "DefenderNet",
    "ModelSpec",
    "get_attack",
    "get_model_spec",
    "is_supported_attack",
    "is_supported_model",
    "list_attack_info",
    "list_attacks",
    "list_models",
    "load_model",
    "run_benchmark",
]


def __getattr__(name: str):
    """Import the torch-dependent modules only when something asks for them."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    return getattr(importlib.import_module(module_path), name)
