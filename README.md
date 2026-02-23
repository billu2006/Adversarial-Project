This repository lets you:
Experiment with adversarial attacks
Evaluate defence robustness of neural networks
Compare baseline vs advanced attack suites
Add and test your own custom attacks
Originally developed as part of a university assignment, this project is shared here for learning and exploration.

Prerequisites
Python
Required: Python 3.13
Extended attacks rely on .pyc files compiled specifically for Python 3.13
Check your version:
python --version
Getting Started
Clone the repository
git clone https://github.com/billu2006/Adversarial-Project.git
cd Adversarial-Project

After extraction, you should have:

Defenders/
├── 0.pt
├── 1.pt
├── ...
└── 9.pt
These models are used as reference defences for evaluation.
Project Structure
.
├── Competition.ipynb
├── Reference_attacks.py
├── Evaluate_attack.py
├── Evaluate_defence.py
├── Extended_attacks.py          # optional
├── _attacks_internal/           # optional
│   ├── attack_001.pyc
│   └── ...
└── Defenders/                   # downloaded separately

Tutorial 1: Testing an Attack
This evaluates how strong an adversarial attack is.
Run the attack evaluation
python Evaluate_attack.py
Inside Evaluate_attack.py, configure:
defenders_path = "./Defenders"

def your_attack(model, X, y, device):
    # Your attack logic here
    return X_adv
Example output:

ATTACK MATRIX
Model          Robust Accuracy      Attack Score (1/acc)
0.pt                 52.30%              1.91
1.pt                 48.15%              2.08
...
Mean                 49.82%              2.15

How to interpret results:
Lower robust accuracy → stronger attack
Higher attack score → more effective attack

Tutorial 2: Testing a Defence:
This evaluates how robust a trained model is.
Run the defence evaluation
python Evaluate_defence.py
Inside Evaluate_defence.py, configure:
your_model_file = "0.pt"
attack_mode = "baseline"  # or "extended"
Attack Modes
Baseline Mode (recommended for learning)
FGSM
PGD-5
PGD-20
Fast execution
Ideal for experimentation
Extended Mode (advanced)
15 compiled adversarial attacks
Slower but more comprehensive
Requires Python 3.13
Switch modes by setting:
attack_mode = "extended"
Example Defence Output
DEFENCE EVALUATION RESULTS

Attack               Robust Accuracy
------------------------------------------------
FGSM                  45.23%
PGD-5                 38.67%
PGD-20                32.15%
------------------------------------------------
Mean (Defence Score)  38.68%
Higher accuracy indicates a stronger defence.
Tutorial 3: Adding a Custom Attack
You can easily test your own ideas.
Add the following to Evaluate_defence.py:
def my_custom_attack(model, X, y, device):
    epsilon = 0.1
    X_adv = X + epsilon * torch.randn_like(X)
    X_adv = torch.clamp(X_adv, 0.0, 1.0)
    return X_adv

test_attacks.append(("MyAttack", my_custom_attack))
Re-run the evaluation to see how it performs.
Critical Constraint: Epsilon Limit
All attacks must satisfy:
L∞ norm STRICTLY < 0.11
Recommended safe value:
epsilon ≤ 0.109
Violating this constraint invalidates results.
Troubleshooting
Defenders folder not found
→ Ensure Defenders.zip has been downloaded and extracted correctly.
Bad magic number / .pyc errors
→ Ensure you are using Python 3.13.
Extended attacks not loading
→ Missing optional files or wrong Python version.
Model file not found
→ Check the path in Evaluate_defence.py.
Suggested Exploration Path
Run baseline defence evaluation
Modify an attack slightly
Observe how scores change
Switch to extended mode
Add a custom attack
Compare robustness across models
Academic Note
This project is adapted from a university assignment and is shared for educational purposes only.
Baseline attacks are provided for evaluation
They are not intended as final solutions
Experimentation and independent development are encouraged
