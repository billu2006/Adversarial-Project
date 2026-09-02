# The benchmarking framework

The library underneath the service, and how to use it on its own.

It started as a COMP219 university assignment, which supplied three things: an
evaluation harness, 15 reference attacks as compiled `.pyc` files, and 10
pretrained defender checkpoints. The attacks, the defence, the metrics and the
structure here are mine — see [Provenance](../README.md#provenance). Since then
the attacks have been parametrised by `epsilon` and `max_iterations` rather than
hard-coding them, the model whitelist has replaced ad-hoc file paths, and the
supplied harness has been replaced by `engine.py` plus the CLI tools in
`scripts/`.

The service ([README.md](../README.md)) is a wrapper around exactly this code.

---

## What it does

- Runs adversarial attacks (FGSM, PGD, C&W, L-BFGS, and a best-of ensemble)
  against Fashion-MNIST classifiers.
- Scores an **attack** by how far it drives robust accuracy down across the pool
  of reference defenders.
- Scores a **defence** by how much robust accuracy it retains across a suite of
  attacks.

Every attack must satisfy the assignment's constraint: the L-inf perturbation
must be **strictly less than 0.11**. The library clamps to `0.109`
(`benchmark.constants.MAX_EPSILON`) and the engine fails a benchmark whose
measured perturbation exceeds the requested budget, so a broken attack produces
an error rather than a flattering score.

---

## Layout

```
src/benchmark/
├── attacks.py            # FGSM, PGD, C&W, L-BFGS, ensemble implementations
├── catalog.py            # the model whitelist + attack metadata (imports no torch)
├── constants.py          # epsilon and iteration limits, input shape
├── data.py               # Fashion-MNIST test-split loader
├── engine.py             # run_benchmark(): the thing a job actually executes
├── extended_attacks.py   # optional pre-compiled attack suite (not used by the service)
└── models.py             # DefenderNet + guarded checkpoint loading

models/defenders/         # 0.pt ... 9.pt, the ten reference defenders
scripts/                  # standalone CLI evaluation tools
```

---

## Using it directly

```python
from benchmark import run_benchmark

results = run_benchmark(
    model_name="fmnist-mlp-defender-0",
    attacks=["fgsm", "pgd"],
    epsilon=0.1,
    max_iterations=20,
    max_samples=2048,
)

for row in results:
    print(row.attack_name, row.robust_accuracy, row.runtime_ms)
```

### Evaluating an attack across every defender

```bash
python scripts/evaluate_attack.py --attack pgd --epsilon 0.1 --max-iterations 20
```

```
Attack: pgd | epsilon: 0.1 | device: cpu
--------------------------------------------------------------
Model                          Robust accuracy   Score (1/acc)
--------------------------------------------------------------
fmnist-mlp-defender-0                    52.30%            1.91
...
```

Lower robust accuracy means a stronger attack.

### Evaluating a defence against a suite of attacks

```bash
python scripts/evaluate_defence.py --model fmnist-mlp-defender-0 --attacks fgsm pgd cw
python scripts/evaluate_defence.py --weights ./my-model.pt --attacks fgsm pgd
```

`--weights` loads a checkpoint from your own disk. The *service* deliberately
refuses to do this — see [Security considerations](../README.md#security-considerations)
— but it is fine locally, against a file you produced yourself.

---

## Adding an attack

Attacks share one signature and are registered in `ATTACKS`:

```python
# src/benchmark/attacks.py
def my_attack(model, X, y, *, epsilon, max_iterations=20, **_):
    """X is (N, 784) in [0, 1]; return the same shape, clipped to the budget."""
    perturbed = X + epsilon * torch.randn_like(X).sign()
    return _clip(perturbed, X, epsilon)  # keeps the L-inf constraint
```

Register it in two places — an `AttackInfo` in `catalog.py` (the published
description) and an entry in `ATTACK_FUNCTIONS` in `attacks.py` (the
implementation) — and it appears automatically in `GET /v1/attacks`, in both CLI
scripts, and as a valid value in a job submission. The two are cross-checked at
import: a catalogued attack with no implementation, or the reverse, raises
immediately rather than failing a job later.

---

## Training a defender

`notebooks/competition.ipynb` holds my adversarial-training loop: progressive
epsilon, a mixed FGSM/PGD/L-BFGS attack schedule during training, and a 25/75
clean/adversarial loss split. It saves a checkpoint you can then score with
`scripts/evaluate_defence.py --weights`.

Note that this is *not* where `models/defenders/*.pt` came from — those ten are
the reference defenders supplied with the assignment, kept as a fixed pool so
attack scores are comparable. The notebook trains a defence to submit against
that pool. It stays a notebook because that is what it is: exploratory work, not
part of the service.

---

## The extended attack suite

The assignment also shipped 15 attacks as `.pyc` files compiled for CPython
3.13, loaded by `benchmark/extended_attacks.py` from an `_attacks_internal/`
directory. They are optional, are not included in this repository, and are
**not** reachable from the service: importing bytecode from disk executes it,
which is the same trust problem as loading an untrusted checkpoint.

---

## Dataset

Fashion-MNIST is downloaded on first use into `data/` (override with
`BENCHMARK_DATA_DIR`). Docker Compose mounts a volume there so the worker
downloads it once.
