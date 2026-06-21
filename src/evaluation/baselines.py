from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.environment.trading_env import TradingEnv

Policy = Callable[[np.ndarray], np.ndarray]
PolicyFactory = Callable[[int], Policy]


def _obs_slices(env: TradingEnv) -> dict[str, slice]:
    r = env.num_resources
    return {
        "prices_scaled": slice(0, r),
        "log_returns": slice(r, 2 * r),
        "pos_frac": slice(2 * r, 3 * r),
        "last_actions": slice(3 * r, 4 * r),
        "cash_frac": slice(-2, -1),
        "time_frac": slice(-1, None),
    }


def hold_policy_factory(env: TradingEnv) -> PolicyFactory:
    action_dim = env.single_action_space.shape[0]

    def factory(_seed: int) -> Policy:
        def policy(obs: np.ndarray) -> np.ndarray:
            obs = np.asarray(obs, dtype=np.float32)
            if obs.ndim == 1:
                return np.zeros(action_dim, dtype=np.float32)
            return np.zeros((obs.shape[0], action_dim), dtype=np.float32)

        return policy

    return factory


def random_policy_factory(env: TradingEnv) -> PolicyFactory:
    action_dim = env.single_action_space.shape[0]

    def factory(seed: int) -> Policy:
        rng = np.random.default_rng(int(seed))

        def policy(obs: np.ndarray) -> np.ndarray:
            obs = np.asarray(obs, dtype=np.float32)
            if obs.ndim == 1:
                return rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            return rng.uniform(-1.0, 1.0, size=(obs.shape[0], action_dim)).astype(np.float32)

        return policy

    return factory


def momentum_policy_factory(env: TradingEnv, *, scale: float = 10.0) -> PolicyFactory:
    s = _obs_slices(env)["log_returns"]

    def factory(_seed: int) -> Policy:
        def policy(obs: np.ndarray) -> np.ndarray:
            obs = np.asarray(obs, dtype=np.float32)
            if obs.ndim == 1:
                return np.clip(scale * obs[s], -1.0, 1.0).astype(np.float32)
            return np.clip(scale * obs[:, s], -1.0, 1.0).astype(np.float32)

        return policy

    return factory


def contrarian_policy_factory(env: TradingEnv, *, scale: float = 10.0) -> PolicyFactory:
    mom = momentum_policy_factory(env, scale=scale)

    def factory(seed: int) -> Policy:
        base = mom(seed)

        def policy(obs: np.ndarray) -> np.ndarray:
            return -base(obs)

        return policy

    return factory


def rebalance_policy_factory(env: TradingEnv, *, target_cash: float = 0.10, k: float = 3.0) -> PolicyFactory:
    r = env.num_resources
    s_pos = _obs_slices(env)["pos_frac"]

    def factory(_seed: int) -> Policy:
        def policy(obs: np.ndarray) -> np.ndarray:
            obs = np.asarray(obs, dtype=np.float32)
            if obs.ndim == 1:
                pos = obs[s_pos]
                cash = float(obs[-2])
                target_pos = (1.0 - target_cash) / float(r)
                delta = target_pos - pos
                if cash > target_cash:
                    delta = np.maximum(delta, 0.0)
                return np.clip(k * delta, -1.0, 1.0).astype(np.float32)

            pos = obs[:, s_pos]
            cash = obs[:, -2]
            target_pos = (1.0 - target_cash) / float(r)
            delta = target_pos - pos
            delta = np.where((cash[:, None] > target_cash), np.maximum(delta, 0.0), delta)
            return np.clip(k * delta, -1.0, 1.0).astype(np.float32)

        return policy

    return factory


def build_baselines(env: TradingEnv, names: list[str]) -> dict[str, PolicyFactory]:
    baselines: dict[str, PolicyFactory] = {}
    wanted = {n.strip().lower() for n in names if n.strip()}

    if "hold" in wanted:
        baselines["hold"] = hold_policy_factory(env)
    if "random" in wanted:
        baselines["random"] = random_policy_factory(env)
    if "momentum" in wanted:
        baselines["momentum"] = momentum_policy_factory(env)
    if "contrarian" in wanted:
        baselines["contrarian"] = contrarian_policy_factory(env)
    if "rebalance" in wanted:
        baselines["rebalance"] = rebalance_policy_factory(env)

    unknown = wanted - set(baselines.keys())
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    return baselines
