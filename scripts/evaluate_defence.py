"""Score one defender against a suite of attacks.

The mirror image of ``evaluate_attack.py``: fix the model, vary the attack.

    python scripts/evaluate_defence.py --model fmnist-mlp-defender-0
    python scripts/evaluate_defence.py --weights ./my-model.pt --attacks fgsm pgd

``--weights`` loads a checkpoint straight off your disk, which is exactly what
the *service* refuses to do (``torch.load`` on an untrusted file is remote code
execution - see the README). It is fine here because you are running it on your
own machine against your own file; ``weights_only=True`` is still used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from benchmark.attacks import ATTACKS
from benchmark.constants import INPUT_FEATURES, MAX_EPSILON
from benchmark.data import load_test_loader, resolve_device
from benchmark.models import MODELS, DefenderNet, load_model


def evaluate(model, loader, attack_fn, *, epsilon, max_iterations, device):
    """Return (robust accuracy, largest L-inf perturbation applied)."""
    correct = 0
    total = 0
    max_perturbation = 0.0
    for images, targets in loader:
        images = images.to(device).view(-1, INPUT_FEATURES)
        targets = targets.to(device)

        adversarial = attack_fn(
            model, images, targets, epsilon=epsilon, max_iterations=max_iterations, device=device
        )

        with torch.no_grad():
            correct += model(adversarial).argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)
            max_perturbation = max(max_perturbation, (adversarial - images).abs().max().item())
    return correct / total, max_perturbation


def load_local_checkpoint(path: Path, device) -> torch.nn.Module:
    model = DefenderNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default="fmnist-mlp-defender-0", choices=sorted(MODELS))
    source.add_argument("--weights", type=Path, help="Path to your own .pt checkpoint")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "cw"])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.epsilon > MAX_EPSILON:
        parser.error(f"epsilon must be <= {MAX_EPSILON} (the framework's constraint is < 0.11)")
    unknown = [name for name in args.attacks if name not in ATTACKS]
    if unknown:
        parser.error(f"unknown attack(s): {', '.join(unknown)}. Choose from {sorted(ATTACKS)}")

    torch.manual_seed(args.seed)
    device = resolve_device()
    loader = load_test_loader(batch_size=args.batch_size, max_samples=args.max_samples or None)

    if args.weights:
        model = load_local_checkpoint(args.weights, device)
        label = str(args.weights)
    else:
        model = load_model(args.model, device=device)
        label = args.model

    print(f"Defence: {label} | epsilon: {args.epsilon} | device: {device}")
    print("-" * 62)
    print(f"{'Attack':<20}{'Robust accuracy':>18}{'Max L-inf':>16}")
    print("-" * 62)

    accuracies = []
    for name in args.attacks:
        accuracy, perturbation = evaluate(
            model,
            loader,
            ATTACKS[name].fn,
            epsilon=args.epsilon,
            max_iterations=args.max_iterations,
            device=device,
        )
        accuracies.append(accuracy)
        # Flag any attack that broke the budget: its number is not comparable.
        flag = "  <- exceeds epsilon!" if perturbation > args.epsilon + 1e-6 else ""
        print(f"{name:<20}{accuracy * 100:>17.2f}%{perturbation:>16.6f}{flag}")

    print("-" * 62)
    print(f"{'Mean (defence score)':<20}{sum(accuracies) / len(accuracies) * 100:>17.2f}%")


if __name__ == "__main__":
    main()
