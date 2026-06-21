from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.environment.trading_env import PriceProcess, TradingEnv
from src.evaluation.baselines import Policy, PolicyFactory, build_baselines
from src.evaluation.execution import execution_costs_from_info
from src.evaluation.metrics import summarize_equity_curve
from src.utils.io import write_json
from src.utils.probe import numpy_torch_import_works


def _env_from_train_config(cfg: dict[str, Any], *, max_steps: int) -> TradingEnv:
    return TradingEnv(
        num_agents=int(cfg.get("agents", 1)),
        num_resources=int(cfg.get("resources", 3)),
        max_steps=int(max_steps),
        reward_mode=str(cfg.get("reward_mode", "log_return")),  # type: ignore[arg-type]
        risk_aversion=float(cfg.get("risk_aversion", 0.0)),
        transaction_cost=float(cfg.get("transaction_cost", 1e-3)),
        spread=float(cfg.get("spread", 1e-3)),
        temporary_impact=float(cfg.get("temporary_impact", 2e-2)),
        permanent_impact=float(cfg.get("permanent_impact", 5e-3)),
        liquidity=float(cfg.get("liquidity", 1_000.0)),
        max_trade_fraction=float(cfg.get("max_trade_fraction", 0.20)),
        price_process=PriceProcess(
            drift=float(cfg.get("drift", 0.0)),
            volatility=float(cfg.get("volatility", 0.02)),
            return_autocorr=float(cfg.get("return_autocorr", 0.10)),
            corr=float(cfg.get("corr", 0.3)),
        ),
    )


def _run_episode(env: TradingEnv, policy: Policy, *, seed: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)

    equity = [env.portfolio_values().astype(np.float64)]
    fees_total = np.zeros(env.num_agents, dtype=np.float64)
    slippage_total = np.zeros(env.num_agents, dtype=np.float64)
    notional_total = np.zeros(env.num_agents, dtype=np.float64)
    for _ in range(env.max_steps):
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        equity.append(env.portfolio_values().astype(np.float64))
        costs = execution_costs_from_info(info)
        fees_total += np.asarray(costs["fees"], dtype=np.float64)
        slippage_total += np.asarray(costs["slippage"], dtype=np.float64)
        notional_total += np.asarray(costs["notional"], dtype=np.float64)
        if terminated or truncated:
            break

    equity_arr = np.asarray(equity, dtype=np.float64)  # (T+1, A)
    initial = np.maximum(equity_arr[0], 1e-8)
    turnover_total = notional_total / initial

    per_agent: list[dict[str, float]] = []
    for a in range(equity_arr.shape[1]):
        m = summarize_equity_curve(equity_arr[:, a])
        m.update(
            {
                "fees_total": float(fees_total[a]),
                "slippage_total": float(slippage_total[a]),
                "notional_total": float(notional_total[a]),
                "turnover_total": float(turnover_total[a]),
            }
        )
        per_agent.append(m)

    keys = [
        "pnl",
        "total_return",
        "sharpe",
        "max_drawdown",
        "final_value",
        "fees_total",
        "slippage_total",
        "notional_total",
        "turnover_total",
    ]
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in per_agent]))
    return out


def _summarize_runs(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [
        "pnl",
        "total_return",
        "sharpe",
        "max_drawdown",
        "final_value",
        "fees_total",
        "slippage_total",
        "notional_total",
        "turnover_total",
    ]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        xs = np.array([r[k] for r in rows], dtype=np.float64)
        finite = np.isfinite(xs)
        out[k] = {
            "mean": float(np.mean(xs[finite])) if finite.any() else float("nan"),
            "std": float(np.std(xs[finite], ddof=1)) if finite.sum() >= 2 else float("nan"),
        }
    return out


def evaluate(
    *,
    env: TradingEnv,
    policies: dict[str, PolicyFactory],
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, factory in policies.items():
        rows = []
        for i in range(int(episodes)):
            ep_seed = int(seed) + i
            policy = factory(ep_seed)
            rows.append(_run_episode(env, policy, seed=ep_seed))
        results[name] = _summarize_runs(rows)
    return results


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent and baselines.")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--steps", type=int, default=None, help="Defaults to config steps (or 100).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional agent checkpoint (.pt for PPO, .npz for CEM).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (loads config.json and auto-detects agent checkpoint if present).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="hold,random,momentum,contrarian,rebalance",
        help="Comma-separated list: hold,random,momentum,contrarian,rebalance",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional path to save results.")
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    run_dir: Path | None = None

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.checkpoint:
        run_dir = Path(args.checkpoint).resolve().parent

    if run_dir is not None:
        cfg_path = run_dir / "config.json"
        if cfg_path.exists():
            cfg = _load_json(cfg_path)

    max_steps = int(args.steps) if args.steps is not None else int(cfg.get("steps", 100))
    env = _env_from_train_config(cfg, max_steps=max_steps)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    policies: dict[str, PolicyFactory] = build_baselines(env, baselines)

    agent_path: Path | None = None
    if args.checkpoint:
        agent_path = Path(args.checkpoint)
    elif run_dir is not None:
        ppo = run_dir / "ppo_agent.pt"
        cem = run_dir / "cem_policy.npz"
        if ppo.exists() and numpy_torch_import_works():
            agent_path = ppo
        elif cem.exists():
            agent_path = cem
        elif ppo.exists():
            print(
                "Note: found ppo_agent.pt but torch import failed; evaluating baselines only.",
                file=sys.stderr,
            )

    if agent_path is not None:
        if not agent_path.exists():
            raise FileNotFoundError(agent_path)

        if agent_path.suffix == ".npz":
            from src.agents.cem_agent import load_linear_policy

            policy_obj = load_linear_policy(agent_path)

            def agent_factory(_seed: int) -> Policy:
                def policy(obs: np.ndarray) -> np.ndarray:
                    return policy_obj.act(obs)

                return policy

            policies = {"agent": agent_factory, **policies}
        elif agent_path.suffix == ".pt":
            if not numpy_torch_import_works():
                print(
                    "Note: torch import failed; skipping PPO checkpoint and evaluating baselines only.",
                    file=sys.stderr,
                )
            else:
                # Import lazily so baseline-only evaluation works even when torch is unavailable.
                from src.agents.ppo_agent import PPOAgent

                agent = PPOAgent(
                    input_dim=env.single_observation_space.shape[0],
                    action_dim=env.single_action_space.shape[0],
                )
                agent.load(str(agent_path))

                def agent_factory(_seed: int) -> Policy:
                    def policy(obs: np.ndarray) -> np.ndarray:
                        actions, _, _ = agent.select_action(obs, deterministic=True)
                        return np.asarray(actions, dtype=np.float32)

                    return policy

                policies = {"agent": agent_factory, **policies}
        else:
            raise ValueError(f"Unsupported checkpoint type: {agent_path}")

    results = evaluate(env=env, policies=policies, episodes=args.episodes, seed=args.seed)
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.json:
        write_json(args.json, results)
    elif run_dir is not None:
        write_json(run_dir / "eval.json", results)
        try:
            from src.experiments.run_card import write_run_card

            write_run_card(run_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
