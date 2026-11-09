"""Global seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) torch; pin torch to deterministic CPU kernels."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency, but stay safe
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
