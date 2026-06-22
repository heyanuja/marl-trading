from __future__ import annotations

from typing import Any

TRAIN_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    # Designed for demos: strong autocorrelation (trend), high trading friction, and enough horizon
    # for a low-turnover learner to look clever vs naive baselines.
    "wow": {
        "algo": "cem",
        "episodes": 512,
        "steps": 200,
        "save_dir": "results/wow",
        "cem_evals": 3,
        "eval_episodes": 100,
        "return_autocorr": 0.6,
        "volatility": 0.03,
        "transaction_cost": 0.005,
        "spread": 0.005,
        "temporary_impact": 0.08,
        "liquidity": 800.0,
        "permanent_impact": 0.0,
    },
}


def list_train_presets() -> list[str]:
    return sorted(TRAIN_PRESETS.keys())


def train_preset_overrides(name: str) -> dict[str, Any]:
    preset = str(name or "default").strip().lower()
    if preset not in TRAIN_PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {list_train_presets()}")
    return dict(TRAIN_PRESETS[preset])
