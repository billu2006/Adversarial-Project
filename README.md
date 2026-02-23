This repository lets you:
- Experiment with adversarial attacks
- Evaluate defence robustness of neural networks
- Compare baseline vs advanced attack suites
- Add and test your own custom attacks

Originally developed as part of a university assignment, this project is shared here for **learning and exploration**.

---

## Prerequisites

### Python

- **Required:** Python **3.13**
- Extended attacks rely on `.pyc` files compiled specifically for Python 3.13

Check your version:

```bash
python --version
```

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/billu2006/Adversarial-Project.git
cd Adversarial-Project
```

---

## Required Model Files

This repository does not include pretrained models.

**Download reference models:**
1. Go to Canvas → COMP219 → Lecture 14
2. Download `Defenders.zip`
3. Extract it into the project root

After extraction, you should have:

```
Defenders/
├── 0.pt
├── 1.pt
└── ... (10 models total)
```

---

## File Structure

```
Competition/
├── Competition.ipynb          # Main experimentation notebook
├── Reference_attacks.py       # Baseline attacks (testing only)
├── Evaluate_attack.py         # Attack evaluation script
├── Evaluate_defence.py        # Defence evaluation script
├── Extended_attacks.py        # Extended attack loader (optional)
├── _attacks_internal/         # Compiled attack files (optional)
│   ├── attack_001.pyc
│   └── ... (15 attacks total)
└── Defenders/                 # Reference models (downloaded separately)
    ├── 0.pt
    └── ... (10 models total)
```

---

## Tutorial 1: Testing an Attack

Evaluate how strong an adversarial attack is.

**Run:**

```bash
python Evaluate_attack.py
```

**Inside `Evaluate_attack.py`:**

```python
defenders_path = "./Defenders"

def your_attack(model, X, y, device):
    return X_adv
```

**Example Output:**

```
ATTACK MATRIX
Model     Robust Accuracy   Attack Score (1/acc)
0.pt      52.30%            1.91
1.pt      48.15%            2.08
Mean      49.82%            2.15
```

**Interpretation:**
- Lower robust accuracy → stronger attack
- Higher attack score → more effective attack

---

## Tutorial 2: Testing a Defence

Evaluate how robust a trained model is.

**Run:**

```bash
python Evaluate_defence.py
```

**Inside `Evaluate_defence.py`:**

```python
your_model_file = "0.pt"
attack_mode = "baseline"  # or "extended"
```

### Attack Modes

**Baseline Mode**
- FGSM
- PGD-5
- PGD-20
- Fast and ideal for experimentation

**Extended Mode**
- 15 advanced attacks
- Slower but more comprehensive
- Requires Python 3.13

**Example Defence Output:**

```
DEFENCE EVALUATION RESULTS
Attack   Robust Accuracy
FGSM     45.23%
PGD-5    38.67%
PGD-20   32.15%
Mean (Defence Score): 38.68%
```

Higher accuracy indicates a stronger defence.

---

## Adding a Custom Attack

**Inside `Evaluate_defence.py`:**

```python
def my_custom_attack(model, X, y, device):
    epsilon = 0.1
    X_adv = X + epsilon * torch.randn_like(X)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)
    return X_adv

test_attacks.append(("MyAttack", my_custom_attack))
```

Re-run the evaluation to observe the effect.

---

## Critical Constraint: Epsilon Limit

All attacks must satisfy:

> **L∞ norm STRICTLY < 0.11**

Recommended safe value:

```
epsilon ≤ 0.109
```

---

## Academic Note

This project is adapted from a university assignment and shared for **educational purposes only**.
