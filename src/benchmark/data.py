"""Evaluation dataset loading.

The benchmark always scores against the Fashion-MNIST test split, the same data
the defenders were graded on. The number of samples is capped rather than fixed
so an operator can trade accuracy of the estimate against runtime - an
unbounded evaluation set is a denial-of-service vector as surely as an unbounded
iteration count.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

#: Where torchvision caches the dataset. Mounted as a volume in Compose so the
#: download happens once rather than on every worker restart.
DATA_DIR = Path(os.environ.get("BENCHMARK_DATA_DIR", "data"))


def load_test_loader(
    *,
    batch_size: int = 128,
    max_samples: int | None = 2048,
    download: bool = True,
) -> DataLoader:
    """Return a loader over the first ``max_samples`` Fashion-MNIST test images."""
    # Imported lazily: torchvision pulls in PIL and friends, and importing it at
    # module scope would slow down anything that only wants DATA_DIR.
    import torchvision
    from torchvision import transforms

    dataset = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=download,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    if max_samples is not None and max_samples < len(dataset):
        # A deterministic prefix, not a random sample: two jobs with the same
        # configuration must produce comparable numbers.
        dataset = Subset(dataset, range(max_samples))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
