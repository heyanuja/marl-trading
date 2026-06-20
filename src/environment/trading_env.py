from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import make_constant_correlation, safe_cholesky

RewardMode = Literal["pnl", "return", "log_return"]


@dataclass(frozen=True)
class PriceProcess:
    drift: float = 0.0
    volatility: float = 0.02
    return_autocorr: float = 0.1
    corr: float = 0.3
    dt: float = 1.0


class TradingEnv(gym.Env):
    """
    A stylized multi-asset, multi-agent trading simulator.

    - Actions: per-agent vector in [-1, 1] per asset.
      Positive -> buy, negative -> sell.
    - Execution: includes bid/ask spread, transaction fees, and temporary impact (slippage).
    - Prices: evolve with an exogenous log-return process + permanent impact from net order flow.

    Observations are returned as a float32 array of shape (num_agents, obs_dim).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        num_agents: int = 1,
        num_resources: int = 3,
        max_steps: int = 100,
        max_trade_fraction: float = 0.20,
        initial_cash: float = 1_000.0,
        initial_inventory: float = 0.0,
        initial_price_range: tuple[float, float] = (8.0, 12.0),
        min_price: float = 1.0,
        max_price: float = 100.0,
        transaction_cost: float = 1e-3,
        spread: float = 1e-3,
        temporary_impact: float = 2e-2,
        permanent_impact: float = 5e-3,
        liquidity: float = 1_000.0,
        reward_mode: RewardMode = "log_return",
        risk_aversion: float = 0.0,
        price_process: PriceProcess | None = None,
    ):
        super().__init__()

        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        if num_resources <= 0:
            raise ValueError(f"num_resources must be positive, got {num_resources}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if not (0.0 < max_trade_fraction <= 1.0):
            raise ValueError(f"max_trade_fraction must be in (0, 1], got {max_trade_fraction}")
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")
        if initial_inventory < 0:
            raise ValueError(f"initial_inventory must be non-negative, got {initial_inventory}")
        if liquidity <= 0:
            raise ValueError(f"liquidity must be positive, got {liquidity}")
        if transaction_cost < 0:
            raise ValueError(f"transaction_cost must be non-negative, got {transaction_cost}")
        if spread < 0:
            raise ValueError(f"spread must be non-negative, got {spread}")
        if temporary_impact < 0 or permanent_impact < 0:
            raise ValueError("Impact coefficients must be non-negative")
        if risk_aversion < 0:
            raise ValueError(f"risk_aversion must be non-negative, got {risk_aversion}")
        low, high = initial_price_range
        if low <= 0 or high <= 0 or high <= low:
            raise ValueError(f"initial_price_range must be positive and ordered, got {initial_price_range}")

        self.num_agents = int(num_agents)
        self.num_resources = int(num_resources)
        self.max_steps = int(max_steps)

        self.max_trade_fraction = float(max_trade_fraction)
        self.initial_cash = float(initial_cash)
        self.initial_inventory = float(initial_inventory)
        self.initial_price_range = (float(low), float(high))

        self.min_price = float(min_price)
        self.max_price = float(max_price)

        self.transaction_cost = float(transaction_cost)
        self.spread = float(spread)
        self.temporary_impact = float(temporary_impact)
        self.permanent_impact = float(permanent_impact)
        self.liquidity = float(liquidity)

        self.reward_mode: RewardMode = reward_mode
        self.risk_aversion = float(risk_aversion)

        self.price_process = price_process or PriceProcess()
        corr = make_constant_correlation(self.num_resources, float(self.price_process.corr))
        self._chol = safe_cholesky(corr)

        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_resources,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, self.num_resources),
            dtype=np.float32,
        )

        # Per-agent observation features:
        # [prices_scaled(R), log_returns(R), position_value_frac(R), last_actions(R), cash_frac(1), time_frac(1)]
        self._obs_dim = self.num_resources * 4 + 2
        self.single_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_agents, self._obs_dim),
            dtype=np.float32,
        )

        self.current_step = 0
        self.cash = np.zeros(self.num_agents, dtype=np.float32)
        self.inventories = np.zeros((self.num_agents, self.num_resources), dtype=np.float32)
        self.prices = np.zeros(self.num_resources, dtype=np.float32)
        self._prev_exo_log_returns = np.zeros(self.num_resources, dtype=np.float32)
        self.last_log_returns = np.zeros(self.num_resources, dtype=np.float32)
        self.last_actions = np.zeros((self.num_agents, self.num_resources), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = 0

        self.cash = np.full(self.num_agents, self.initial_cash, dtype=np.float32)
        self.inventories = np.full(
            (self.num_agents, self.num_resources),
            self.initial_inventory,
            dtype=np.float32,
        )
        low, high = self.initial_price_range
        self.prices = self.np_random.uniform(low, high, size=self.num_resources).astype(np.float32)

        self._prev_exo_log_returns = np.zeros(self.num_resources, dtype=np.float32)
        self.last_log_returns = np.zeros(self.num_resources, dtype=np.float32)
        self.last_actions = np.zeros((self.num_agents, self.num_resources), dtype=np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, actions):
        if self.current_step >= self.max_steps:
            raise RuntimeError("step() called on a terminated episode; call reset()")

        self.current_step += 1
        actions_arr = self._coerce_actions(actions)
        self.last_actions = actions_arr.copy()

        prices_before = self.prices.copy()
        portfolio_before = self._portfolio_values(prices_before)

        executed_units = np.zeros((self.num_agents, self.num_resources), dtype=np.float32)
        execution_prices = np.tile(prices_before, (self.num_agents, 1)).astype(np.float32, copy=False)
        fees = np.zeros(self.num_agents, dtype=np.float32)

        for agent in range(self.num_agents):
            units, px, fee = self._execute_agent_trades(agent, actions_arr[agent], prices_before)
            executed_units[agent] = units
            execution_prices[agent] = px
            fees[agent] = fee

        net_units = executed_units.sum(axis=0)
        self._advance_prices(prices_before, net_units)

        portfolio_after = self._portfolio_values(self.prices)
        rewards = self._compute_rewards(portfolio_before, portfolio_after)

        if self.risk_aversion > 0:
            rewards = rewards - self.risk_aversion * self._concentration_penalty()

        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info(
            prices_before=prices_before,
            executed_units=executed_units,
            execution_prices=execution_prices,
            fees=fees,
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
        )
        return obs, rewards.astype(np.float32), terminated, truncated, info

    def portfolio_values(self) -> np.ndarray:
        return self._portfolio_values(self.prices).copy()

    # -------------------
    # Mechanics

    def _coerce_actions(self, actions) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.shape == (self.num_resources,):
            arr = arr[None, :]
        if arr.shape != (self.num_agents, self.num_resources):
            raise ValueError(
                f"Expected actions shape {(self.num_agents, self.num_resources)}, got {arr.shape}"
            )
        return np.clip(arr, -1.0, 1.0)

    def _portfolio_values(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float32)
        return self.cash + (self.inventories * prices).sum(axis=1)

    def _execute_agent_trades(
        self,
        agent: int,
        action: np.ndarray,
        mid_prices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        action = np.asarray(action, dtype=np.float32)
        mid_prices = np.asarray(mid_prices, dtype=np.float32)

        pv = float(self.cash[agent] + np.dot(self.inventories[agent], mid_prices))
        pv = max(pv, 1e-8)
        desired_notional = action.astype(np.float64) * (self.max_trade_fraction * pv)
        desired_units = desired_notional / mid_prices.astype(np.float64)

        executed_units = np.zeros(self.num_resources, dtype=np.float32)
        execution_prices = mid_prices.astype(np.float32, copy=True)
        total_fee = 0.0

        # Sells first (increase cash), capped by inventory and per-step trade fraction.
        sell_units = np.clip(-desired_units, 0.0, np.inf)
        max_sell = self.inventories[agent].astype(np.float64) * self.max_trade_fraction
        sell_units = np.minimum(sell_units, max_sell)

        for i in range(self.num_resources):
            units = float(sell_units[i])
            if units <= 0:
                continue
            px = float(self._execution_price(float(mid_prices[i]), side=-1.0, notional=units * float(mid_prices[i])))
            notional = units * px
            fee = self.transaction_cost * notional
            self.inventories[agent, i] -= units
            self.cash[agent] += notional - fee
            executed_units[i] -= units
            execution_prices[i] = px
            total_fee += fee

        # Buys (spend cash), scaled down if needed.
        buy_units = np.clip(desired_units, 0.0, np.inf)
        if np.any(buy_units > 0):
            buy_units = buy_units.astype(np.float64)
            cash_avail = float(self.cash[agent])

            def total_buy_cost(units_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
                pxs = np.empty(self.num_resources, dtype=np.float64)
                costs = np.empty(self.num_resources, dtype=np.float64)
                fees = np.empty(self.num_resources, dtype=np.float64)
                for j in range(self.num_resources):
                    units_j = float(units_vec[j])
                    if units_j <= 0:
                        pxs[j] = float(mid_prices[j])
                        costs[j] = 0.0
                        fees[j] = 0.0
                        continue
                    mid = float(mid_prices[j])
                    px = float(self._execution_price(mid, side=+1.0, notional=units_j * mid))
                    cost = units_j * px
                    fee = self.transaction_cost * cost
                    pxs[j] = px
                    costs[j] = cost
                    fees[j] = fee
                return pxs, costs, float(costs.sum() + fees.sum())

            pxs, costs, gross = total_buy_cost(buy_units)
            if gross > cash_avail and gross > 0:
                scale = cash_avail / gross
                buy_units *= scale
                pxs, costs, gross = total_buy_cost(buy_units)

            if gross > 0:
                self.cash[agent] -= gross
                self.inventories[agent] += buy_units.astype(np.float32)
                executed_units += buy_units.astype(np.float32)
                execution_prices = np.where(buy_units > 0, pxs.astype(np.float32), execution_prices)
                total_fee += float((self.transaction_cost * costs).sum())

        return executed_units, execution_prices.astype(np.float32), float(total_fee)

    def _execution_price(self, mid_price: float, *, side: float, notional: float) -> float:
        """
        Side: +1 buy, -1 sell.
        Temporary impact is always paid against the trader (slippage).
        """
        side = 1.0 if side >= 0 else -1.0
        impact = self.temporary_impact * (abs(notional) / self.liquidity)
        px = mid_price * (1.0 + side * (self.spread / 2.0 + impact))
        return float(max(px, self.min_price))

    def _advance_prices(self, prices_before: np.ndarray, net_units: np.ndarray) -> None:
        prices_before = np.asarray(prices_before, dtype=np.float32)
        net_units = np.asarray(net_units, dtype=np.float32)

        shock = self.np_random.standard_normal(self.num_resources).astype(np.float64)
        corr_shock = (self._chol @ shock).astype(np.float64)
        exo = (
            float(self.price_process.drift) * float(self.price_process.dt)
            + float(self.price_process.return_autocorr) * self._prev_exo_log_returns.astype(np.float64)
            + float(self.price_process.volatility) * np.sqrt(float(self.price_process.dt)) * corr_shock
        )

        net_notional = net_units.astype(np.float64) * prices_before.astype(np.float64)
        impact = float(self.permanent_impact) * (net_notional / float(self.liquidity))

        log_prices = np.log(prices_before.astype(np.float64)) + exo + impact
        prices = np.exp(log_prices).astype(np.float32)
        prices = np.clip(prices, self.min_price, self.max_price)

        self._prev_exo_log_returns = exo.astype(np.float32)
        self.last_log_returns = np.log(np.maximum(prices, 1e-8) / np.maximum(prices_before, 1e-8)).astype(np.float32)
        self.prices = prices

    def _compute_rewards(self, portfolio_before: np.ndarray, portfolio_after: np.ndarray) -> np.ndarray:
        pv0 = np.asarray(portfolio_before, dtype=np.float64)
        pv1 = np.asarray(portfolio_after, dtype=np.float64)
        eps = 1e-8
        if self.reward_mode == "pnl":
            return (pv1 - pv0).astype(np.float32)
        if self.reward_mode == "return":
            return (pv1 / (pv0 + eps) - 1.0).astype(np.float32)
        if self.reward_mode == "log_return":
            return np.log((pv1 + eps) / (pv0 + eps)).astype(np.float32)
        raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

    def _concentration_penalty(self) -> np.ndarray:
        pv = self._portfolio_values(self.prices).astype(np.float64)
        pv = np.maximum(pv, 1e-8)
        position_value = self.inventories.astype(np.float64) * self.prices.astype(np.float64)
        weights = position_value / pv[:, None]
        return (weights**2).sum(axis=1).astype(np.float32)

    # -------------------
    # Observations / info

    def _get_obs(self) -> np.ndarray:
        prices_scaled = (self.prices / max(self.max_price, 1e-8)).astype(np.float32)
        time_frac = float(self.current_step) / float(self.max_steps)

        pv = self._portfolio_values(self.prices).astype(np.float64)
        pv = np.maximum(pv, 1e-8)
        pos_value = self.inventories.astype(np.float64) * self.prices.astype(np.float64)
        pos_frac = (pos_value / pv[:, None]).astype(np.float32)
        cash_frac = (self.cash.astype(np.float64) / pv).astype(np.float32)

        obs = np.concatenate(
            [
                np.tile(prices_scaled, (self.num_agents, 1)),
                np.tile(self.last_log_returns.astype(np.float32), (self.num_agents, 1)),
                pos_frac,
                self.last_actions.astype(np.float32),
                cash_frac[:, None],
                np.full((self.num_agents, 1), time_frac, dtype=np.float32),
            ],
            axis=1,
        )
        return obs.astype(np.float32, copy=False)

    def _get_info(self, **kwargs) -> dict[str, Any]:
        info: dict[str, Any] = {
            "step": int(self.current_step),
            "prices": self.prices.copy(),
            "cash": self.cash.copy(),
            "inventories": self.inventories.copy(),
            "portfolio_values": self.portfolio_values(),
            "log_returns": self.last_log_returns.copy(),
        }
        info.update(kwargs)
        return info
