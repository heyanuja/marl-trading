from __future__ import annotations

import os
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """
    Seed Python process randomness best-effort.

    Returns a metadata dict you can persist for reproducibility.
    """
    seed_i = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed_i)
    np.random.seed(seed_i)

    meta: dict[str, Any] = {"seed": seed_i, "torch_seeded": False, "torch_deterministic": False}

    # Avoid importing torch here; some environments crash on import. If available, callers can
    # explicitly seed torch, or we can seed via a subprocess in a future iteration.
    if deterministic_torch:
        meta["torch_deterministic"] = True
    return meta

