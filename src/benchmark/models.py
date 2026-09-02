"""Loading a whitelisted model - the project's central security boundary.

``torch.load`` on an untrusted checkpoint is remote code execution, because
PyTorch checkpoints are pickles and unpickling runs arbitrary code. Rather than
trying to sanitise uploads, the service only ever loads weights it shipped
itself: a request names a model from the whitelist in ``benchmark.catalog``, and
that registry maps the name to a file inside a directory we control.

Two further defences sit behind that:
  * ``weights_only=True`` on every ``torch.load`` call, so even a tampered local
    checkpoint cannot execute code, and
  * a containment check that the resolved path really is inside the weights
    directory, so a registry entry can never escape it via ``..``.

See "Security considerations" in the README.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark.catalog import (
    MODEL_NAMES,
    MODELS,
    WEIGHTS_DIR,
    ModelSpec,
    get_model_spec,
    is_supported_model,
    list_models,
)
from benchmark.constants import INPUT_FEATURES, NUM_CLASSES

# Re-exported so callers that need both the registry and the loader have one
# import; the catalogue itself stays importable without torch.
__all__ = [
    "MODELS",
    "MODEL_NAMES",
    "WEIGHTS_DIR",
    "DefenderNet",
    "ModelSpec",
    "get_model_spec",
    "is_supported_model",
    "list_models",
    "load_model",
]


class DefenderNet(nn.Module):
    """The fully-connected Fashion-MNIST classifier the defenders were trained as.

    Unchanged from the original assignment (784-128-64-32-10 with ReLU and a
    log-softmax head); the checkpoints in ``models/defenders`` only load into
    this exact architecture.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_FEATURES, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.fc4(x), dim=1)


def load_model(name: str, device: torch.device | None = None) -> nn.Module:
    """Load a whitelisted model in eval mode.

    Raises ``KeyError`` for a name outside the whitelist and ``FileNotFoundError``
    if the checkpoint is missing from the image/volume.
    """
    spec = get_model_spec(name)
    path = spec.weights_path.resolve()

    # Defence in depth: even though the filename comes from our own registry,
    # assert it resolves inside the weights directory before touching the disk.
    if not path.is_relative_to(WEIGHTS_DIR):
        raise ValueError(f"Refusing to load weights outside {WEIGHTS_DIR}: {path}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights for {name!r} are missing at {path}. "
            "See the README for how to fetch the defender checkpoints."
        )

    device = device or torch.device("cpu")
    model = DefenderNet().to(device)
    # weights_only=True refuses to unpickle anything but tensors and plain data,
    # which neutralises the pickle-RCE class of attack even for local files.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
