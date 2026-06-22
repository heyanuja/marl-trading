from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_plot_caches(run_dir: Path) -> None:
    """
    Avoid noisy warnings and slow imports when HOME isn't writable in sandboxes.
    Must run before importing matplotlib.
    """
    mpl_cache = run_dir / ".mplcache"
    xdg_cache = run_dir / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def main() -> None:
    p = argparse.ArgumentParser(description="Generate plots for a run directory.")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--no-gif", action="store_true", help="Skip GIF creation.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    _configure_plot_caches(run_dir)

    # Import after setting MPLCONFIGDIR/XDG_CACHE_HOME.
    from src.visualization.visualize import TradingVisualizer

    metrics = _load_jsonl(run_dir / "metrics.jsonl")
    tracked = _load_json(run_dir / "tracked.json")

    viz = TradingVisualizer()
    viz.plot_training_dashboard(metrics, save_path=str(run_dir / "analysis_dashboard.png"))

    if tracked:
        prices = np.asarray(tracked[0]["prices"], dtype=np.float32)
        equity = np.asarray(tracked[0]["equity"], dtype=np.float32)
        viz.plot_episode_diagnostics(prices, equity, save_path=str(run_dir / "episode_diagnostics.png"))
        viz.plot_price_evolution_storyboard(prices, save_path=str(run_dir / "price_storyboard.png"))
        if not args.no_gif:
            try:
                viz.create_trading_animation(prices, save_path=str(run_dir / "price_evolution.gif"))
            except Exception as e:
                print(f"Skipping GIF creation: {e}")


if __name__ == "__main__":
    main()
