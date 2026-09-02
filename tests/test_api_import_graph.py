"""The API must import without PyTorch installed.

Two images are built from one Dockerfile, and only the worker gets PyTorch. That
saves ~800MB and most of the API's cold-start time - but it is an invisible
constraint: adding ``from benchmark.attacks import ...`` to a router would break
the deployed API while every local test still passed, because a development
machine has torch installed.

So the constraint is asserted here. The subprocess blocks ``torch`` at the
import system level and then imports the application.
"""

from __future__ import annotations

import subprocess
import sys

# Refuse to import torch (or any submodule of it), then build the app.
PROBE = """
import sys


class TorchBlocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith(("torch.", "torchvision")):
            raise ImportError(f"{name} is not available in the API image")
        return None


sys.meta_path.insert(0, TorchBlocker())

from service.main import create_app

create_app()

assert "torch" not in sys.modules, "the API imported torch"
print("OK")
"""


def test_the_api_imports_without_torch():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_catalogue_is_importable_without_torch():
    """The whitelist the API validates against must not drag torch in either."""
    probe = PROBE.replace(
        "from service.main import create_app\n\ncreate_app()",
        "from benchmark.catalog import ATTACK_NAMES, MODEL_NAMES\n"
        "assert MODEL_NAMES and ATTACK_NAMES",
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
