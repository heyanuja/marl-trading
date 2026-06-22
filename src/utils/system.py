from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def git_metadata(cwd: str | None = None) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    head = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {
        "head": head,
        "branch": branch,
        "dirty": bool(dirty),
    }


def system_metadata(*, cwd: str | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "gymnasium": package_version("gymnasium"),
            "torch": package_version("torch"),
            "matplotlib": package_version("matplotlib"),
        },
        "git": git_metadata(cwd=cwd),
    }

