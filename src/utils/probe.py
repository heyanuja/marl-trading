from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable


def subprocess_imports_ok(
    modules: Iterable[str],
    *,
    timeout_s: float = 10.0,
) -> bool:
    """
    Check whether `modules` can be imported in a fresh Python subprocess.

    This is intentionally subprocess-based: some native stacks hard-crash the interpreter on import
    (e.g., OpenMP SHM errors). A non-zero return code is treated as "unavailable".
    """
    mods = [m.strip() for m in modules if str(m).strip()]
    if not mods:
        raise ValueError("modules must be non-empty")

    code = "; ".join([f"import {m}" for m in mods]) + "; print('ok')"
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_s),
        )
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def numpy_torch_import_works(*, trials: int = 2) -> bool:
    """
    Best-effort probe for `numpy` + `torch` in the same process.

    Some restricted environments crash intermittently; requiring multiple successful trials
    reduces false positives for "auto" algorithm selection.
    """
    n = int(trials)
    if n <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    for _ in range(n):
        if not subprocess_imports_ok(["numpy", "torch"]):
            return False
    return True
