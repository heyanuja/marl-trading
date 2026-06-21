from __future__ import annotations

from typing import Any

import numpy as np


def execution_costs_from_info(info: dict[str, Any], *, eps: float = 1e-8) -> dict[str, np.ndarray]:
    """
    Compute execution/cost metrics from TradingEnv step `info`.

    Returns per-agent arrays:
    - fees: transaction fee paid (>=0)
    - slippage: spread + temporary impact cost (>=0)
    - notional: total traded notional (>=0)
    - turnover: notional / portfolio_before
    """
    required = ["executed_units", "execution_prices", "prices_before", "fees", "portfolio_before"]
    missing = [k for k in required if k not in info]
    if missing:
        raise KeyError(f"Missing keys in info: {missing}")

    executed_units = np.asarray(info["executed_units"], dtype=np.float64)
    execution_prices = np.asarray(info["execution_prices"], dtype=np.float64)
    prices_before = np.asarray(info["prices_before"], dtype=np.float64)
    fees = np.asarray(info["fees"], dtype=np.float64)
    portfolio_before = np.asarray(info["portfolio_before"], dtype=np.float64)

    if executed_units.ndim != 2:
        raise ValueError(f"Expected executed_units shape (A, R), got {executed_units.shape}")
    if execution_prices.shape != executed_units.shape:
        raise ValueError(
            f"Expected execution_prices shape {executed_units.shape}, got {execution_prices.shape}"
        )
    if prices_before.ndim != 1 or prices_before.shape[0] != executed_units.shape[1]:
        raise ValueError(
            f"Expected prices_before shape ({executed_units.shape[1]},), got {prices_before.shape}"
        )
    if fees.ndim != 1 or fees.shape[0] != executed_units.shape[0]:
        raise ValueError(f"Expected fees shape ({executed_units.shape[0]},), got {fees.shape}")
    if portfolio_before.ndim != 1 or portfolio_before.shape[0] != executed_units.shape[0]:
        raise ValueError(
            f"Expected portfolio_before shape ({executed_units.shape[0]},), got {portfolio_before.shape}"
        )

    notional = np.sum(np.abs(executed_units) * execution_prices, axis=1)
    slippage = np.sum((execution_prices - prices_before[None, :]) * executed_units, axis=1)
    turnover = notional / np.maximum(portfolio_before, float(eps))

    return {
        "fees": fees,
        "slippage": slippage,
        "notional": notional,
        "turnover": turnover,
    }

