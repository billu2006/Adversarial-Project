"""Score one attack against the whole pool of reference defenders.

The standalone counterpart to the service: same engine, no HTTP. Use it to
iterate on an attack locally, then submit the finished thing through the API.

    python scripts/evaluate_attack.py --attack pgd --epsilon 0.1

A lower robust accuracy means a stronger attack; the attack score is the mean
of 1/accuracy across the defenders, as in the original assignment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from benchmark.attacks import ATTACKS
from benchmark.constants import INPUT_FEATURES, MAX_EPSILON
from benchmark.data import load_test_loader, resolve_device
from benchmark.models import MODELS, load_model


def evaluate(model, loader, attack_fn, *, epsilon, max_iterations, device) -> float:
    """Fraction of the evaluation set still classified correctly under attack."""
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device).view(-1, INPUT_FEATURES)
        targets = targets.to(device)

        adversarial = attack_fn(
            model, images, targets, epsilon=epsilon, max_iterations=max_iterations, device=device
        )

        with torch.no_grad():
            predictions = model(adversarial).argmax(dim=1)
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", default="pgd", choices=sorted(ATTACKS))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-samples", type=int, default=2048, help="Evaluation-set size (0 for all)"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.epsilon > MAX_EPSILON:
        parser.error(f"epsilon must be <= {MAX_EPSILON} (the framework's constraint is < 0.11)")

    torch.manual_seed(args.seed)
    device = resolve_device()
    loader = load_test_loader(batch_size=args.batch_size, max_samples=args.max_samples or None)
    attack = ATTACKS[args.attack]

    print(f"Attack: {attack.name} | epsilon: {args.epsilon} | device: {device}")
    print("-" * 62)
    print(f"{'Model':<28}{'Robust accuracy':>18}{'Score (1/acc)':>16}")
    print("-" * 62)

    accuracies = []
    for name in MODELS:
        model = load_model(name, device=device)
        accuracy = evaluate(
            model,
            loader,
            attack.fn,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations,
            device=device,
        )
        accuracies.append(accuracy)
        # Guard the reciprocal: a perfect attack would otherwise divide by zero.
        print(f"{name:<28}{accuracy * 100:>17.2f}%{1.0 / max(accuracy, 1e-10):>16.2f}")

    mean_accuracy = sum(accuracies) / len(accuracies)
    attack_score = sum(1.0 / max(a, 1e-10) for a in accuracies) / len(accuracies)
    print("-" * 62)
    print(f"{'Mean':<28}{mean_accuracy * 100:>17.2f}%{attack_score:>16.2f}")


if __name__ == "__main__":
    main()
