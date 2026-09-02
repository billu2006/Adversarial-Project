"""The benchmark itself: run attacks against a whitelisted model, report metrics.

One call to :func:`run_benchmark` is one job. It is deliberately synchronous and
blocking - making it asynchronous would not make it faster, since it is CPU
bound - and the service handles the slowness by running it on a worker instead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmark.attacks import get_attack
from benchmark.constants import INPUT_FEATURES
from benchmark.data import load_test_loader, resolve_device
from benchmark.models import load_model


@dataclass(frozen=True)
class AttackResult:
    """One row of ``job_results``: how a model held up against one attack."""

    attack_name: str
    #: Fraction of the evaluation set still classified correctly under attack.
    #: Lower means a stronger attack / weaker defence.
    robust_accuracy: float
    #: Mean negative log-likelihood of the true class on adversarial inputs.
    mean_nll: float
    #: Largest L-inf perturbation actually applied. Sanity-checks the epsilon
    #: constraint: this must stay below 0.11 or the benchmark is invalid.
    perturbation_norm: float
    runtime_ms: int
    samples: int


class EpsilonConstraintViolation(RuntimeError):
    """Raised when an attack perturbs further than the agreed epsilon budget."""


def _evaluate_attack(
    model: torch.nn.Module,
    loader: Iterable,
    attack_name: str,
    *,
    epsilon: float,
    max_iterations: int,
    device: torch.device,
) -> AttackResult:
    attack = get_attack(attack_name)
    started = time.perf_counter()

    correct = 0
    total = 0
    nll_sum = 0.0
    max_perturbation = 0.0

    for images, targets in loader:
        images = images.to(device).view(-1, INPUT_FEATURES)
        targets = targets.to(device)

        adversarial = attack.fn(
            model,
            images,
            targets,
            epsilon=epsilon,
            max_iterations=max_iterations,
            device=device,
        )

        with torch.no_grad():
            outputs = model(adversarial)
            # size-agnostic sum so the last (short) batch is not over-weighted.
            nll_sum += F.nll_loss(outputs, targets, reduction="sum").item()
            correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)
            batch_norm = (adversarial - images).abs().max().item()
            max_perturbation = max(max_perturbation, batch_norm)

    runtime_ms = int((time.perf_counter() - started) * 1000)

    if total == 0:  # pragma: no cover - only reachable with an empty dataset
        raise RuntimeError("Evaluation set was empty")

    return AttackResult(
        attack_name=attack_name,
        robust_accuracy=correct / total,
        mean_nll=nll_sum / total,
        perturbation_norm=max_perturbation,
        runtime_ms=runtime_ms,
        samples=total,
    )


def run_benchmark(
    *,
    model_name: str,
    attacks: Sequence[str],
    epsilon: float,
    max_iterations: int = 20,
    max_samples: int = 2048,
    batch_size: int = 128,
    device: torch.device | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[AttackResult]:
    """Benchmark ``model_name`` against each attack in ``attacks``.

    ``progress`` is called as ``(attack_name, completed, total)`` after each
    attack so the worker can log partial progress on a job that may run for
    minutes. Attacks run in the order given and results come back in that order.
    """
    device = device or resolve_device()
    model = load_model(model_name, device=device)
    loader = load_test_loader(batch_size=batch_size, max_samples=max_samples)

    results: list[AttackResult] = []
    for index, attack_name in enumerate(attacks, start=1):
        result = _evaluate_attack(
            model,
            loader,
            attack_name,
            epsilon=epsilon,
            max_iterations=max_iterations,
            device=device,
        )
        # A perturbation above the requested budget means an attack is broken;
        # publishing that number as a robustness score would be worse than
        # failing the job.
        if result.perturbation_norm > epsilon + 1e-6:
            raise EpsilonConstraintViolation(
                f"Attack {attack_name!r} perturbed by {result.perturbation_norm:.6f}, "
                f"which exceeds the requested epsilon of {epsilon:.6f}"
            )
        results.append(result)
        if progress is not None:
            progress(attack_name, index, len(attacks))

    return results
