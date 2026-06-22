from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.sparkline import sparkline


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_float(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.6f}"


def write_run_card(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    cfg = _load_json(run_path / "config.json") or {}
    meta = _load_json(run_path / "meta.json") or {}
    eval_results = _load_json(run_path / "eval.json")

    # Metrics
    rewards: list[float] = []
    final_equity: list[float] = []
    fees_total: list[float] = []
    slippage_total: list[float] = []
    notional_total: list[float] = []
    turnover_total: list[float] = []
    metrics_path = run_path / "metrics.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "episode_reward_mean" in row:
                rewards.append(float(row["episode_reward_mean"]))
            if "final_equity_mean" in row:
                final_equity.append(float(row["final_equity_mean"]))
            if "fees_total_mean" in row:
                fees_total.append(float(row["fees_total_mean"]))
            if "slippage_total_mean" in row:
                slippage_total.append(float(row["slippage_total_mean"]))
            if "notional_total_mean" in row:
                notional_total.append(float(row["notional_total_mean"]))
            if "turnover_total_mean" in row:
                turnover_total.append(float(row["turnover_total_mean"]))

    reward_last = float(rewards[-1]) if rewards else float("nan")
    reward_best = float(np.max(rewards)) if rewards else float("nan")
    reward_last100 = float(np.mean(rewards[-100:])) if rewards else float("nan")
    equity_last = float(final_equity[-1]) if final_equity else float("nan")
    fees_last = float(fees_total[-1]) if fees_total else float("nan")
    slippage_last = float(slippage_total[-1]) if slippage_total else float("nan")
    notional_last = float(notional_total[-1]) if notional_total else float("nan")
    turnover_last = float(turnover_total[-1]) if turnover_total else float("nan")

    git = meta.get("git", {}) if isinstance(meta.get("git", {}), dict) else {}
    head = git.get("head", None)
    head_short = str(head)[:8] if head else None
    branch = git.get("branch", None)
    dirty = git.get("dirty", None)

    lines: list[str] = []
    lines.append(f"# Run: `{run_path.as_posix()}`")
    lines.append("")
    lines.append("## Snapshot")
    lines.append(f"- Timestamp (UTC): `{meta.get('timestamp_utc', 'unknown')}`")
    if branch:
        lines.append(f"- Git branch: `{branch}`")
    if head_short:
        lines.append(f"- Git commit: `{head_short}`")
    if dirty is not None:
        lines.append(f"- Git dirty: `{bool(dirty)}`")
    algo = cfg.get("algo_resolved", cfg.get("algo", "unknown"))
    lines.append(f"- Algo: `{algo}`")
    preset = cfg.get("preset", None)
    if preset:
        lines.append(f"- Preset: `{preset}`")
    if str(algo).lower() == "baseline":
        lines.append(f"- Baseline policy: `{cfg.get('baseline_policy', 'unknown')}`")
    lines.append(f"- Seed: `{cfg.get('seed', 'unknown')}`")
    lines.append(f"- Agents/assets/steps: `{cfg.get('agents', '?')}/{cfg.get('resources', '?')}/{cfg.get('steps', '?')}`")
    lines.append("")

    lines.append("## Key Metrics")
    lines.append(f"- Reward mean (last): `{_fmt_float(reward_last)}`")
    lines.append(f"- Reward mean (best): `{_fmt_float(reward_best)}`")
    lines.append(f"- Reward mean (last 100): `{_fmt_float(reward_last100)}`")
    lines.append(f"- Final equity mean (last): `{_fmt_float(equity_last)}`")
    lines.append(f"- Fees total mean (last): `{_fmt_float(fees_last)}`")
    lines.append(f"- Slippage total mean (last): `{_fmt_float(slippage_last)}`")
    lines.append(f"- Notional total mean (last): `{_fmt_float(notional_last)}`")
    lines.append(f"- Turnover total mean (last): `{_fmt_float(turnover_last)}`")
    if rewards:
        lines.append(f"- Reward sparkline: `{sparkline(rewards, width=48)}`")
    lines.append("")

    if eval_results:
        lines.append("## Evaluation")
        for name, stats in eval_results.items():
            if not isinstance(stats, dict):
                continue
            total_ret = stats.get("total_return", {})
            sharpe = stats.get("sharpe", {})
            turnover = stats.get("turnover_total", {})
            fees = stats.get("fees_total", {})
            lines.append(
                f"- `{name}`: total_return μ={_fmt_float(float(total_ret.get('mean', float('nan'))))}, "
                f"sharpe μ={_fmt_float(float(sharpe.get('mean', float('nan'))))}, "
                f"turnover μ={_fmt_float(float(turnover.get('mean', float('nan'))))}, "
                f"fees μ={_fmt_float(float(fees.get('mean', float('nan'))))}"
            )
        lines.append("")

    lines.append("## Artifacts")
    artifacts = [
        "config.json",
        "meta.json",
        "metrics.jsonl",
        "ppo_agent.pt",
        "cem_policy.npz",
        "analysis_dashboard.png",
        "episode_diagnostics.png",
        "price_storyboard.png",
        "price_evolution.gif",
        "eval.json",
    ]
    for a in artifacts:
        p = run_path / a
        if p.exists():
            lines.append(f"- `{p.as_posix()}`")
    lines.append("")

    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    out_path = run_path / "RUN.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
