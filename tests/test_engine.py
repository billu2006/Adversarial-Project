"""The real benchmarking engine.

Marked ``engine`` and skipped when the checkpoints or the Fashion-MNIST cache
are absent, so the fast suite stays runnable anywhere. Run them with
``pytest -m engine``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmark.attacks import ATTACKS  # noqa: E402
from benchmark.constants import INPUT_FEATURES  # noqa: E402
from benchmark.data import DATA_DIR  # noqa: E402
from benchmark.engine import EpsilonConstraintViolation, run_benchmark  # noqa: E402
from benchmark.models import MODELS, load_model  # noqa: E402

pytestmark = pytest.mark.engine

WEIGHTS_PRESENT = MODELS["fmnist-mlp-defender-0"].weights_path.is_file()
DATASET_PRESENT = (DATA_DIR / "FashionMNIST").exists()

requires_weights = pytest.mark.skipif(not WEIGHTS_PRESENT, reason="defender weights not present")
requires_dataset = pytest.mark.skipif(not DATASET_PRESENT, reason="Fashion-MNIST cache not present")


@requires_weights
def test_only_whitelisted_models_load():
    with pytest.raises(KeyError):
        load_model("../../etc/passwd")
    with pytest.raises(KeyError):
        load_model("resnet18-cifar10")


@requires_weights
@pytest.mark.parametrize("attack_name", list(ATTACKS))
def test_every_attack_respects_the_epsilon_budget(attack_name):
    """The one invariant that makes a robustness number meaningful."""
    torch.manual_seed(0)
    model = load_model("fmnist-mlp-defender-0")
    images = torch.rand(8, INPUT_FEATURES)
    labels = torch.randint(0, 10, (8,))
    epsilon = 0.1

    adversarial = ATTACKS[attack_name].fn(
        model, images, labels, epsilon=epsilon, max_iterations=3, device=torch.device("cpu")
    )

    assert adversarial.shape == images.shape
    assert (adversarial - images).abs().max().item() <= epsilon + 1e-6
    # Still a valid image.
    assert adversarial.min() >= 0.0 and adversarial.max() <= 1.0


@requires_weights
@requires_dataset
def test_benchmark_produces_plausible_metrics():
    results = run_benchmark(
        model_name="fmnist-mlp-defender-0",
        attacks=["fgsm", "pgd"],
        epsilon=0.1,
        max_iterations=3,
        max_samples=256,
        batch_size=128,
    )

    assert [row.attack_name for row in results] == ["fgsm", "pgd"]
    for row in results:
        assert 0.0 <= row.robust_accuracy <= 1.0
        assert row.mean_nll >= 0.0
        assert row.perturbation_norm <= 0.1 + 1e-6
        assert row.runtime_ms >= 0
        assert row.samples == 256

    # PGD is FGSM run iteratively, so it should be at least as damaging.
    assert results[1].robust_accuracy <= results[0].robust_accuracy + 0.05


@requires_weights
@requires_dataset
def test_a_misbehaving_attack_fails_the_job(monkeypatch):
    """A result that violated the epsilon budget must never be published."""
    from benchmark import attacks as attacks_module

    def _cheating_attack(model, X, y, **kwargs):
        return torch.clamp(X + 0.5, 0.0, 1.0)

    monkeypatch.setitem(
        attacks_module.ATTACKS,
        "fgsm",
        attacks_module.AttackSpec(info=attacks_module.ATTACK_CATALOG["fgsm"], fn=_cheating_attack),
    )

    with pytest.raises(EpsilonConstraintViolation):
        run_benchmark(
            model_name="fmnist-mlp-defender-0",
            attacks=["fgsm"],
            epsilon=0.1,
            max_samples=128,
            batch_size=128,
        )
