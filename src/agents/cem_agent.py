from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LinearTanhPolicy:
    weights: np.ndarray  # (obs_dim, action_dim)
    bias: np.ndarray  # (action_dim,)

    @property
    def obs_dim(self) -> int:
        return int(self.weights.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.weights.shape[1])

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        logits = obs_arr @ self.weights + self.bias
        return np.tanh(logits).astype(np.float32)

    def to_flat(self) -> np.ndarray:
        flat = np.concatenate([self.weights.reshape(-1), self.bias.reshape(-1)], axis=0)
        return np.asarray(flat, dtype=np.float32)

    @classmethod
    def from_flat(cls, theta: np.ndarray, *, obs_dim: int, action_dim: int) -> LinearTanhPolicy:
        theta_arr = np.asarray(theta, dtype=np.float32).reshape(-1)
        expected = int(obs_dim) * int(action_dim) + int(action_dim)
        if theta_arr.size != expected:
            raise ValueError(f"Expected {expected} params, got {theta_arr.size}")

        w_end = int(obs_dim) * int(action_dim)
        weights = theta_arr[:w_end].reshape(int(obs_dim), int(action_dim))
        bias = theta_arr[w_end:]
        return cls(weights=weights.copy(), bias=bias.copy())


def save_linear_policy(path: str | Path, policy: LinearTanhPolicy, *, extra: dict[str, Any] | None = None) -> Path:
    out_path = Path(path)
    payload: dict[str, Any] = {
        "weights": np.asarray(policy.weights, dtype=np.float32),
        "bias": np.asarray(policy.bias, dtype=np.float32),
        "obs_dim": int(policy.obs_dim),
        "action_dim": int(policy.action_dim),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(out_path, **payload)
    return out_path


def load_linear_policy(path: str | Path) -> LinearTanhPolicy:
    data = np.load(str(path))
    weights = np.asarray(data["weights"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32)
    return LinearTanhPolicy(weights=weights, bias=bias)


@dataclass(frozen=True)
class CEMConfig:
    population_size: int = 32
    elite_frac: float = 0.2
    smoothing: float = 0.25
    init_std: float = 0.50
    min_std: float = 0.05


class CEMOptimizer:
    def __init__(
        self,
        *,
        param_dim: int,
        config: CEMConfig,
        rng: np.random.Generator,
    ) -> None:
        if param_dim <= 0:
            raise ValueError(f"param_dim must be positive, got {param_dim}")
        if config.population_size <= 1:
            raise ValueError(f"population_size must be > 1, got {config.population_size}")
        if not (0.0 < config.elite_frac <= 1.0):
            raise ValueError(f"elite_frac must be in (0, 1], got {config.elite_frac}")
        if not (0.0 < config.smoothing <= 1.0):
            raise ValueError(f"smoothing must be in (0, 1], got {config.smoothing}")
        if config.init_std <= 0 or config.min_std <= 0:
            raise ValueError("init_std/min_std must be positive")

        self.param_dim = int(param_dim)
        self.config = config
        self.rng = rng

        self.mean = np.zeros(self.param_dim, dtype=np.float32)
        self.std = np.full(self.param_dim, float(config.init_std), dtype=np.float32)

        self.best_theta = self.mean.copy()
        self.best_score = -float("inf")

        self._batch_thetas: list[np.ndarray] = []
        self._batch_scores: list[float] = []

    def sample(self) -> np.ndarray:
        return self.rng.normal(loc=self.mean, scale=self.std).astype(np.float32)

    def tell(self, theta: np.ndarray, score: float) -> dict[str, float] | None:
        theta_arr = np.asarray(theta, dtype=np.float32).reshape(-1)
        if theta_arr.size != self.param_dim:
            raise ValueError(f"Expected theta size {self.param_dim}, got {theta_arr.size}")

        score_f = float(score)
        self._batch_thetas.append(theta_arr.copy())
        self._batch_scores.append(score_f)

        if score_f > self.best_score:
            self.best_score = score_f
            self.best_theta = theta_arr.copy()

        if len(self._batch_scores) >= self.config.population_size:
            return self._update_distribution()
        return None

    def flush(self) -> dict[str, float] | None:
        if not self._batch_scores:
            return None
        return self._update_distribution()

    def _update_distribution(self) -> dict[str, float]:
        scores = np.asarray(self._batch_scores, dtype=np.float64)
        thetas = np.asarray(self._batch_thetas, dtype=np.float32)

        elite_count = max(1, int(np.ceil(float(self.config.elite_frac) * float(scores.size))))
        elite_idx = np.argsort(scores)[::-1][:elite_count]
        elites = thetas[elite_idx]

        mean_new = elites.mean(axis=0)
        std_new = elites.std(axis=0)
        std_new = np.maximum(std_new, float(self.config.min_std)).astype(np.float32)

        alpha = float(self.config.smoothing)
        self.mean = ((1.0 - alpha) * self.mean + alpha * mean_new).astype(np.float32)
        self.std = ((1.0 - alpha) * self.std + alpha * std_new).astype(np.float32)

        self._batch_scores.clear()
        self._batch_thetas.clear()

        return {
            "cem_elite_count": float(elite_count),
            "cem_mean_norm": float(np.linalg.norm(self.mean)),
            "cem_std_mean": float(np.mean(self.std)),
            "cem_std_min": float(np.min(self.std)),
        }

