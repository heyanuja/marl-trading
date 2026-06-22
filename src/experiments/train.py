from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.environment.trading_env import PriceProcess, TradingEnv
from src.evaluation.baselines import build_baselines
from src.evaluation.execution import execution_costs_from_info
from src.evaluation.harness import evaluate as evaluate_policies
from src.experiments.presets import list_train_presets, train_preset_overrides
from src.experiments.run_card import write_run_card
from src.utils.io import write_json, write_jsonl
from src.utils.probe import numpy_torch_import_works
from src.utils.seed import seed_everything
from src.utils.system import system_metadata


def _seed_torch(seed: int, *, deterministic: bool = True) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 1_000
    steps: int = 100
    seed: int = 0
    save_dir: str = "results/run0"
    agents: int = 1
    resources: int = 3
    preset: str = "wow"

    reward_mode: str = "log_return"
    risk_aversion: float = 0.0

    transaction_cost: float = 1e-3
    spread: float = 1e-3
    temporary_impact: float = 2e-2
    permanent_impact: float = 5e-3
    liquidity: float = 1_000.0
    max_trade_fraction: float = 0.20

    drift: float = 0.0
    volatility: float = 0.02
    return_autocorr: float = 0.10
    corr: float = 0.3

    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 3e-4
    ppo_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    log_every: int = 10
    save_every: int = 50
    track_episodes: int = 1
    no_plots: bool = False
    eval_episodes: int = 25
    eval_baselines: str = "hold,random,momentum,contrarian,rebalance"
    no_eval: bool = False
    algo: str = "auto"  # auto|ppo|cem|baseline
    baseline_policy: str = "momentum"
    overwrite: bool = False

    cem_population: int = 32
    cem_elite_frac: float = 0.2
    cem_smoothing: float = 0.25
    cem_init_std: float = 0.50
    cem_min_std: float = 0.05
    cem_evals: int = 3


def _prepare_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not any(run_dir.iterdir()):
        return
    if not overwrite:
        raise FileExistsError(
            f"{run_dir} is not empty; pass --overwrite or choose a new --save-dir."
        )

    results_root = (Path.cwd() / "results").resolve()
    run_dir_resolved = run_dir.resolve()
    if run_dir_resolved == results_root or not run_dir_resolved.is_relative_to(results_root):
        raise ValueError(f"Refusing to overwrite outside `{results_root}`: {run_dir_resolved}")

    for p in run_dir.iterdir():
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _make_env(cfg: TrainConfig) -> TradingEnv:
    return TradingEnv(
        num_agents=cfg.agents,
        num_resources=cfg.resources,
        max_steps=cfg.steps,
        reward_mode=cfg.reward_mode,  # type: ignore[arg-type]
        risk_aversion=cfg.risk_aversion,
        transaction_cost=cfg.transaction_cost,
        spread=cfg.spread,
        temporary_impact=cfg.temporary_impact,
        permanent_impact=cfg.permanent_impact,
        liquidity=cfg.liquidity,
        max_trade_fraction=cfg.max_trade_fraction,
        price_process=PriceProcess(
            drift=cfg.drift,
            volatility=cfg.volatility,
            return_autocorr=cfg.return_autocorr,
            corr=cfg.corr,
        ),
    )


def _rollout_policy(
    env: TradingEnv,
    policy,
    *,
    num_steps: int,
    seed: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obs, _ = env.reset(seed=seed)

    prices_path = [env.prices.copy()]
    equity_path = [env.portfolio_values().copy()]
    rewards: list[np.ndarray] = []
    fees_path: list[np.ndarray] = []
    slippage_path: list[np.ndarray] = []
    notional_path: list[np.ndarray] = []

    for _ in range(num_steps):
        action = policy(obs)
        obs, rew, terminated, truncated, info = env.step(action)
        rewards.append(np.asarray(rew, dtype=np.float32))
        prices_path.append(info["prices"].copy())
        equity_path.append(info["portfolio_values"].copy())

        costs = execution_costs_from_info(info)
        fees_path.append(np.asarray(costs["fees"], dtype=np.float32))
        slippage_path.append(np.asarray(costs["slippage"], dtype=np.float32))
        notional_path.append(np.asarray(costs["notional"], dtype=np.float32))

        if terminated or truncated:
            break

    rewards_arr = np.asarray(rewards, dtype=np.float32) if rewards else np.zeros((0, env.num_agents), dtype=np.float32)
    fees_arr = np.asarray(fees_path, dtype=np.float32) if fees_path else np.zeros((0, env.num_agents), dtype=np.float32)
    slippage_arr = (
        np.asarray(slippage_path, dtype=np.float32)
        if slippage_path
        else np.zeros((0, env.num_agents), dtype=np.float32)
    )
    notional_arr = (
        np.asarray(notional_path, dtype=np.float32)
        if notional_path
        else np.zeros((0, env.num_agents), dtype=np.float32)
    )
    traces = {
        "prices": np.asarray(prices_path, dtype=np.float32),
        "equity": np.asarray(equity_path, dtype=np.float32),
        "rewards": rewards_arr,
        "fees": fees_arr,
        "slippage": slippage_arr,
        "notional": notional_arr,
    }
    return {}, traces


def _collect_rollout_ppo(
    env: TradingEnv,
    agent,
    *,
    num_steps: int,
    seed: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from src.agents.ppo_agent import compute_gae

    obs, _ = env.reset(seed=seed)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    fees_path: list[np.ndarray] = []
    slippage_path: list[np.ndarray] = []
    notional_path: list[np.ndarray] = []

    prices_path = [env.prices.copy()]
    equity_path = [env.portfolio_values().copy()]

    for _ in range(num_steps):
        act, log_p, val = agent.select_action(obs)
        next_obs, rew, terminated, truncated, info = env.step(act)

        states.append(obs)
        actions.append(np.asarray(act, dtype=np.float32))
        log_probs.append(np.asarray(log_p, dtype=np.float32))
        values.append(np.asarray(val, dtype=np.float32))
        rewards.append(np.asarray(rew, dtype=np.float32))

        done = float(bool(terminated))
        dones.append(np.full(env.num_agents, done, dtype=np.float32))

        prices_path.append(info["prices"].copy())
        equity_path.append(info["portfolio_values"].copy())
        costs = execution_costs_from_info(info)
        fees_path.append(np.asarray(costs["fees"], dtype=np.float32))
        slippage_path.append(np.asarray(costs["slippage"], dtype=np.float32))
        notional_path.append(np.asarray(costs["notional"], dtype=np.float32))

        obs = next_obs
        if terminated or truncated:
            break

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    log_probs_arr = np.asarray(log_probs, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.float32)
    fees_arr = np.asarray(fees_path, dtype=np.float32) if fees_path else np.zeros((0, env.num_agents), dtype=np.float32)
    slippage_arr = (
        np.asarray(slippage_path, dtype=np.float32)
        if slippage_path
        else np.zeros((0, env.num_agents), dtype=np.float32)
    )
    notional_arr = (
        np.asarray(notional_path, dtype=np.float32)
        if notional_path
        else np.zeros((0, env.num_agents), dtype=np.float32)
    )

    last_values = agent.value(obs).astype(np.float32)
    adv, returns = compute_gae(
        rewards_arr,
        values_arr,
        dones_arr,
        last_values=last_values,
        gamma=agent.config.gamma,
        gae_lambda=agent.config.gae_lambda,
    )

    t_steps, num_agents, obs_dim = states_arr.shape
    act_dim = actions_arr.shape[-1]

    batch = {
        "states": states_arr.reshape(t_steps * num_agents, obs_dim),
        "actions": actions_arr.reshape(t_steps * num_agents, act_dim),
        "old_log_probs": log_probs_arr.reshape(t_steps * num_agents),
        "advantages": adv.reshape(t_steps * num_agents),
        "returns": returns.reshape(t_steps * num_agents),
    }

    traces = {
        "prices": np.asarray(prices_path, dtype=np.float32),
        "equity": np.asarray(equity_path, dtype=np.float32),
        "rewards": rewards_arr,
        "fees": fees_arr,
        "slippage": slippage_arr,
        "notional": notional_arr,
    }
    return batch, traces


def train(cfg: TrainConfig) -> list[dict[str, Any]]:
    seed_everything(cfg.seed)

    save_dir = Path(cfg.save_dir)
    _prepare_run_dir(save_dir, overwrite=bool(cfg.overwrite))

    algo = cfg.algo.lower().strip()
    if algo == "auto":
        algo = "ppo" if numpy_torch_import_works() else "cem"

    env = _make_env(cfg)
    agent = None
    baseline_policy = None
    cem = None
    cem_spec: dict[str, Any] | None = None

    if algo == "ppo":
        _seed_torch(cfg.seed, deterministic=True)
        from src.agents.ppo_agent import PPOAgent, PPOConfig

        ppo_cfg = PPOConfig(
            hidden_sizes=cfg.hidden_sizes,
            lr=cfg.lr,
            ppo_epochs=cfg.ppo_epochs,
            minibatch_size=cfg.minibatch_size,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_param=cfg.clip_param,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
        )
        agent = PPOAgent(
            input_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            config=ppo_cfg,
        )
    elif algo == "cem":
        from src.agents.cem_agent import CEMConfig, CEMOptimizer, LinearTanhPolicy

        obs_dim = int(env.single_observation_space.shape[0])
        action_dim = int(env.single_action_space.shape[0])
        param_dim = obs_dim * action_dim + action_dim
        cem_cfg = CEMConfig(
            population_size=int(cfg.cem_population),
            elite_frac=float(cfg.cem_elite_frac),
            smoothing=float(cfg.cem_smoothing),
            init_std=float(cfg.cem_init_std),
            min_std=float(cfg.cem_min_std),
        )
        cem = CEMOptimizer(
            param_dim=param_dim,
            config=cem_cfg,
            rng=np.random.default_rng(int(cfg.seed)),
        )
        cem_spec = {"obs_dim": obs_dim, "action_dim": action_dim, "param_dim": int(param_dim)}
    elif algo == "baseline":
        baseline_policy = cfg.baseline_policy.lower().strip()
        policies = build_baselines(env, [baseline_policy])
        factory = policies[baseline_policy]
        baseline_policy = factory(cfg.seed)
    else:
        raise ValueError(f"Unknown algo: {cfg.algo!r} (use auto|ppo|cem|baseline)")

    cfg_payload = asdict(cfg)
    cfg_payload["algo_resolved"] = algo
    write_json(save_dir / "config.json", cfg_payload)
    write_json(save_dir / "meta.json", system_metadata(cwd=str(Path.cwd())))

    metrics: list[dict[str, Any]] = []
    final_prices: list[np.ndarray] = []
    tracked: list[dict[str, Any]] = []

    for episode in range(cfg.episodes):
        update_metrics: dict[str, Any] = {}
        if algo == "ppo":
            assert agent is not None
            batch, traces = _collect_rollout_ppo(env, agent, num_steps=cfg.steps, seed=cfg.seed + episode)
            update_metrics = agent.update(batch)
        elif algo == "cem":
            assert cem is not None and cem_spec is not None
            from src.agents.cem_agent import LinearTanhPolicy

            if int(cfg.cem_evals) <= 0:
                raise ValueError(f"cem_evals must be positive, got {cfg.cem_evals}")

            theta = cem.sample()
            policy_obj = LinearTanhPolicy.from_flat(
                theta,
                obs_dim=int(cem_spec["obs_dim"]),
                action_dim=int(cem_spec["action_dim"]),
            )

            def policy(obs: np.ndarray, _policy: LinearTanhPolicy = policy_obj) -> np.ndarray:
                return _policy.act(obs)

            batch_id = int(episode) // int(cfg.cem_population)
            seed_base = int(cfg.seed) + batch_id * 10_000

            reward_means: list[float] = []
            final_equities: list[np.ndarray] = []
            fees_totals: list[np.ndarray] = []
            slippage_totals: list[np.ndarray] = []
            notional_totals: list[np.ndarray] = []
            turnover_totals: list[np.ndarray] = []

            traces = {}
            for rep in range(int(cfg.cem_evals)):
                _, rep_traces = _rollout_policy(env, policy, num_steps=cfg.steps, seed=seed_base + rep)
                traces = rep_traces

                rep_episode_rewards = rep_traces["rewards"].sum(axis=0)
                reward_means.append(float(np.mean(rep_episode_rewards)))

                rep_final_equity = np.asarray(rep_traces["equity"][-1], dtype=np.float64)
                final_equities.append(rep_final_equity)

                rep_fees_total = np.asarray(rep_traces["fees"], dtype=np.float64).sum(axis=0)
                rep_slippage_total = np.asarray(rep_traces["slippage"], dtype=np.float64).sum(axis=0)
                rep_notional_total = np.asarray(rep_traces["notional"], dtype=np.float64).sum(axis=0)
                rep_initial_equity = np.asarray(rep_traces["equity"][0], dtype=np.float64)
                rep_turnover_total = rep_notional_total / np.maximum(rep_initial_equity, 1e-8)

                fees_totals.append(rep_fees_total)
                slippage_totals.append(rep_slippage_total)
                notional_totals.append(rep_notional_total)
                turnover_totals.append(rep_turnover_total)

            episode_reward_mean = float(np.mean(reward_means)) if reward_means else float("nan")
            final_equity = np.mean(np.stack(final_equities, axis=0), axis=0) if final_equities else np.zeros(env.num_agents)
            final_equity_mean = float(np.mean(final_equity))
            fees_total = np.mean(np.stack(fees_totals, axis=0), axis=0) if fees_totals else np.zeros(env.num_agents)
            slippage_total = (
                np.mean(np.stack(slippage_totals, axis=0), axis=0) if slippage_totals else np.zeros(env.num_agents)
            )
            notional_total = (
                np.mean(np.stack(notional_totals, axis=0), axis=0) if notional_totals else np.zeros(env.num_agents)
            )
            turnover_total = (
                np.mean(np.stack(turnover_totals, axis=0), axis=0) if turnover_totals else np.zeros(env.num_agents)
            )
        else:
            assert baseline_policy is not None
            update_metrics, traces = _rollout_policy(
                env, baseline_policy, num_steps=cfg.steps, seed=cfg.seed + episode
            )

        if algo != "cem":
            episode_rewards = traces["rewards"].sum(axis=0)
            episode_reward_mean = float(np.mean(episode_rewards))
            final_equity = traces["equity"][-1]
            final_equity_mean = float(np.mean(final_equity))

            fees_total = np.asarray(traces.get("fees", np.zeros((0, env.num_agents))), dtype=np.float64).sum(axis=0)
            slippage_total = np.asarray(
                traces.get("slippage", np.zeros((0, env.num_agents))), dtype=np.float64
            ).sum(axis=0)
            notional_total = np.asarray(
                traces.get("notional", np.zeros((0, env.num_agents))), dtype=np.float64
            ).sum(axis=0)
            initial_equity = np.asarray(traces["equity"][0], dtype=np.float64)
            turnover_total = notional_total / np.maximum(initial_equity, 1e-8)

        if algo == "cem":
            assert cem is not None
            cem_update = cem.tell(theta, score=episode_reward_mean)
            update_metrics = {
                **update_metrics,
                "cem_score": episode_reward_mean,
                "cem_best_score": float(cem.best_score),
                **(cem_update or {}),
            }

        row: dict[str, Any] = {
            "episode": int(episode),
            "episode_reward_mean": episode_reward_mean,
            "final_equity_mean": final_equity_mean,
            "final_equity": final_equity.tolist(),
            "fees_total_mean": float(np.mean(fees_total)),
            "slippage_total_mean": float(np.mean(slippage_total)),
            "notional_total_mean": float(np.mean(notional_total)),
            "turnover_total_mean": float(np.mean(turnover_total)),
        }
        row.update(update_metrics)
        metrics.append(row)
        final_prices.append(env.prices.copy())

        if len(tracked) < cfg.track_episodes:
            tracked.append(
                {
                    "episode": int(episode),
                    "prices": traces["prices"].tolist(),
                    "equity": traces["equity"].tolist(),
                }
            )

        if cfg.log_every and (episode + 1) % cfg.log_every == 0:
            window = min(cfg.log_every, len(metrics))
            avg = float(np.mean([m["episode_reward_mean"] for m in metrics[-window:]]))
            print(f"Episode {episode + 1}/{cfg.episodes} | mean reward: {avg:.6f}")

        if algo == "ppo" and cfg.save_every and (episode + 1) % cfg.save_every == 0:
            assert agent is not None
            agent.save(str(save_dir / "ppo_agent.pt"))

    if algo == "ppo":
        assert agent is not None
        agent.save(str(save_dir / "ppo_agent.pt"))
    elif algo == "cem":
        assert cem is not None and cem_spec is not None
        from src.agents.cem_agent import LinearTanhPolicy, save_linear_policy

        cem.flush()
        best = LinearTanhPolicy.from_flat(
            cem.best_theta,
            obs_dim=int(cem_spec["obs_dim"]),
            action_dim=int(cem_spec["action_dim"]),
        )
        save_linear_policy(
            save_dir / "cem_policy.npz",
            best,
            extra={
                "algo": "cem",
                "best_score": float(cem.best_score),
                **cem_spec,
                "cem_population": int(cfg.cem_population),
                "cem_elite_frac": float(cfg.cem_elite_frac),
                "cem_smoothing": float(cfg.cem_smoothing),
                "cem_init_std": float(cfg.cem_init_std),
                "cem_min_std": float(cfg.cem_min_std),
                "cem_evals": int(cfg.cem_evals),
            },
        )

    write_jsonl(save_dir / "metrics.jsonl", metrics)
    np.save(save_dir / "final_prices.npy", np.asarray(final_prices, dtype=np.float32))

    if algo == "cem":
        assert cem is not None and cem_spec is not None
        from src.agents.cem_agent import LinearTanhPolicy

        best = LinearTanhPolicy.from_flat(
            cem.best_theta,
            obs_dim=int(cem_spec["obs_dim"]),
            action_dim=int(cem_spec["action_dim"]),
        )

        def best_policy(obs: np.ndarray) -> np.ndarray:
            return best.act(obs)

        tracked = []
        for i in range(int(cfg.track_episodes)):
            _, traces = _rollout_policy(env, best_policy, num_steps=cfg.steps, seed=cfg.seed + 10_000 + i)
            tracked.append(
                {
                    "episode": int(i),
                    "prices": traces["prices"].tolist(),
                    "equity": traces["equity"].tolist(),
                }
            )

    write_json(save_dir / "tracked.json", tracked)

    if not cfg.no_plots:
        # Plotting happens in a clean subprocess to keep training dependencies minimal
        # and avoid torch <-> matplotlib OpenMP conflicts on some systems.
        cmd = [sys.executable, "-m", "src.experiments.plot", "--run-dir", str(save_dir)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print("Plot generation failed:")
            if res.stdout.strip():
                print(res.stdout.strip())
            if res.stderr.strip():
                print(res.stderr.strip())

    if not cfg.no_eval and int(cfg.eval_episodes) > 0:
        eval_env = _make_env(cfg)
        baseline_names = [b.strip() for b in cfg.eval_baselines.split(",") if b.strip()]
        policies = build_baselines(eval_env, baseline_names)

        if algo == "ppo":
            assert agent is not None

            def agent_factory(_seed: int):
                def policy(obs: np.ndarray) -> np.ndarray:
                    actions, _, _ = agent.select_action(obs, deterministic=True)
                    return np.asarray(actions, dtype=np.float32)

                return policy

            policies = {"agent": agent_factory, **policies}
        elif algo == "cem":
            assert cem is not None and cem_spec is not None
            from src.agents.cem_agent import LinearTanhPolicy

            best = LinearTanhPolicy.from_flat(
                cem.best_theta,
                obs_dim=int(cem_spec["obs_dim"]),
                action_dim=int(cem_spec["action_dim"]),
            )

            def agent_factory(_seed: int):
                def policy(obs: np.ndarray) -> np.ndarray:
                    return best.act(obs)

                return policy

            policies = {"agent": agent_factory, **policies}
        else:
            assert baseline_policy is not None

            def agent_factory(_seed: int):
                def policy(obs: np.ndarray) -> np.ndarray:
                    return baseline_policy(obs)

                return policy

            policies = {"agent": agent_factory, **policies}

        results = evaluate_policies(env=eval_env, policies=policies, episodes=int(cfg.eval_episodes), seed=cfg.seed)
        write_json(save_dir / "eval.json", results)

    write_run_card(save_dir)
    return metrics


def parse_args() -> TrainConfig:
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--preset", type=str, default=TrainConfig.preset, choices=list_train_presets())
    args0, _ = p0.parse_known_args()

    preset = str(args0.preset)
    default_cfg = replace(
        TrainConfig(),
        preset=preset,
        **train_preset_overrides(preset),
    )

    def hidden_sizes_to_str(xs: tuple[int, ...]) -> str:
        return ",".join(str(x) for x in xs)

    p = argparse.ArgumentParser(description="Train an agent in TradingEnv (PPO/CEM/baselines).")
    p.add_argument(
        "--preset",
        type=str,
        default=default_cfg.preset,
        choices=list_train_presets(),
        help="Preset that sets sensible defaults (all flags can still override).",
    )
    p.add_argument("--algo", type=str, default=default_cfg.algo, choices=["auto", "ppo", "cem", "baseline"])
    p.add_argument("--baseline-policy", type=str, default=default_cfg.baseline_policy)
    p.add_argument("--episodes", type=int, default=default_cfg.episodes)
    p.add_argument("--steps", type=int, default=default_cfg.steps)
    p.add_argument("--seed", type=int, default=default_cfg.seed)
    p.add_argument("--save-dir", type=str, default=default_cfg.save_dir)
    p.add_argument("--agents", type=int, default=default_cfg.agents)
    p.add_argument("--resources", type=int, default=default_cfg.resources)
    p.add_argument(
        "--reward-mode",
        type=str,
        default=default_cfg.reward_mode,
        choices=["pnl", "return", "log_return"],
    )
    p.add_argument("--risk-aversion", type=float, default=default_cfg.risk_aversion)

    p.add_argument("--transaction-cost", type=float, default=default_cfg.transaction_cost)
    p.add_argument("--spread", type=float, default=default_cfg.spread)
    p.add_argument("--temporary-impact", type=float, default=default_cfg.temporary_impact)
    p.add_argument("--permanent-impact", type=float, default=default_cfg.permanent_impact)
    p.add_argument("--liquidity", type=float, default=default_cfg.liquidity)
    p.add_argument("--max-trade-fraction", type=float, default=default_cfg.max_trade_fraction)

    p.add_argument("--drift", type=float, default=default_cfg.drift)
    p.add_argument("--volatility", type=float, default=default_cfg.volatility)
    p.add_argument("--return-autocorr", type=float, default=default_cfg.return_autocorr)
    p.add_argument("--corr", type=float, default=default_cfg.corr)

    p.add_argument("--hidden-sizes", type=str, default=hidden_sizes_to_str(default_cfg.hidden_sizes))
    p.add_argument("--lr", type=float, default=default_cfg.lr)
    p.add_argument("--ppo-epochs", type=int, default=default_cfg.ppo_epochs)
    p.add_argument("--minibatch-size", type=int, default=default_cfg.minibatch_size)
    p.add_argument("--gamma", type=float, default=default_cfg.gamma)
    p.add_argument("--gae-lambda", type=float, default=default_cfg.gae_lambda)
    p.add_argument("--clip-param", type=float, default=default_cfg.clip_param)
    p.add_argument("--value-coef", type=float, default=default_cfg.value_coef)
    p.add_argument("--entropy-coef", type=float, default=default_cfg.entropy_coef)

    p.add_argument("--log-every", type=int, default=default_cfg.log_every)
    p.add_argument("--save-every", type=int, default=default_cfg.save_every)
    p.add_argument("--track-episodes", type=int, default=default_cfg.track_episodes)
    p.add_argument("--no-plots", action="store_true", default=default_cfg.no_plots, help="Skip plot/GIF generation.")
    p.add_argument("--eval-episodes", type=int, default=default_cfg.eval_episodes)
    p.add_argument("--eval-baselines", type=str, default=default_cfg.eval_baselines)
    p.add_argument("--no-eval", action="store_true", default=default_cfg.no_eval, help="Skip end-of-training evaluation.")
    p.add_argument("--overwrite", action="store_true", default=default_cfg.overwrite, help="Overwrite --save-dir if it exists.")

    p.add_argument("--cem-population", type=int, default=default_cfg.cem_population)
    p.add_argument("--cem-elite-frac", type=float, default=default_cfg.cem_elite_frac)
    p.add_argument("--cem-smoothing", type=float, default=default_cfg.cem_smoothing)
    p.add_argument("--cem-init-std", type=float, default=default_cfg.cem_init_std)
    p.add_argument("--cem-min-std", type=float, default=default_cfg.cem_min_std)
    p.add_argument("--cem-evals", type=int, default=default_cfg.cem_evals, help="Rollouts per candidate.")
    args = p.parse_args()

    hidden = tuple(int(x) for x in args.hidden_sizes.split(",") if x.strip())
    return TrainConfig(
        preset=args.preset,
        algo=args.algo,
        baseline_policy=args.baseline_policy,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        save_dir=args.save_dir,
        agents=args.agents,
        resources=args.resources,
        reward_mode=args.reward_mode,
        risk_aversion=args.risk_aversion,
        transaction_cost=args.transaction_cost,
        spread=args.spread,
        temporary_impact=args.temporary_impact,
        permanent_impact=args.permanent_impact,
        liquidity=args.liquidity,
        max_trade_fraction=args.max_trade_fraction,
        drift=args.drift,
        volatility=args.volatility,
        return_autocorr=args.return_autocorr,
        corr=args.corr,
        hidden_sizes=hidden or default_cfg.hidden_sizes,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        log_every=args.log_every,
        save_every=args.save_every,
        track_episodes=args.track_episodes,
        no_plots=args.no_plots,
        eval_episodes=args.eval_episodes,
        eval_baselines=args.eval_baselines,
        no_eval=args.no_eval,
        overwrite=args.overwrite,
        cem_population=args.cem_population,
        cem_elite_frac=args.cem_elite_frac,
        cem_smoothing=args.cem_smoothing,
        cem_init_std=args.cem_init_std,
        cem_min_std=args.cem_min_std,
        cem_evals=args.cem_evals,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
