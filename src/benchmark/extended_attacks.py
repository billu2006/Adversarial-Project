"""Loader for the optional pre-compiled extended attack suite.

The original assignment shipped 15 additional attacks as ``.pyc`` files built for
CPython 3.13. They are *not* wired into the service: importing bytecode from
disk executes it, which is exactly the trust boundary the model whitelist exists
to protect (see ``benchmark.models``). This module is kept so the standalone
tutorial script ``scripts/evaluate_defence.py`` still works for local
experimentation, where the operator already trusts the files.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

ATTACKS_DIR = Path(__file__).resolve().parent.parent.parent / "_attacks_internal"


def _load_attack_function(filepath: Path) -> Callable | None:
    """Import one compiled attack module and return its ``attack_*`` function."""
    spec = importlib.util.spec_from_file_location("attack_module", filepath)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in dir(module):
        if name.startswith("attack_") and callable(getattr(module, name)):
            return getattr(module, name)
    return None


def get_all_attacks() -> list[Callable]:
    """Return every loadable extended attack, in filename order.

    Each has the framework's original signature ``(model, X, y, device)``.
    """
    if not ATTACKS_DIR.exists():
        raise FileNotFoundError(
            f"Attack functions directory not found: {ATTACKS_DIR}\n"
            "The extended attack suite is an optional download; see the README."
        )

    attack_files = sorted(
        p for p in ATTACKS_DIR.iterdir() if p.name.startswith("attack_") and p.suffix == ".pyc"
    )
    if not attack_files:
        raise RuntimeError(f"No attack functions found in {ATTACKS_DIR}")

    attacks: list[Callable] = []
    for attack_file in attack_files:
        try:
            attack_fn = _load_attack_function(attack_file)
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the suite
            print(f"Warning: could not load {attack_file.name}: {exc}", file=sys.stderr)
            continue
        if attack_fn is not None:
            attacks.append(attack_fn)

    return attacks
