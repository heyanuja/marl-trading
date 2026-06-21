import numpy as np


def _to_1d_float_array(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def simple_returns(equity_curve):
    equity = _to_1d_float_array(equity_curve)
    if equity.size < 2:
        return np.array([], dtype=np.float64)
    return equity[1:] / equity[:-1] - 1.0


def sharpe_ratio(step_returns, *, eps=1e-12):
    r = _to_1d_float_array(step_returns)
    if r.size < 2:
        return float("nan")
    std = float(r.std(ddof=1))
    if std < eps:
        return float("nan")
    return float((r.mean() / std) * np.sqrt(r.size))


def max_drawdown(equity_curve):
    equity = _to_1d_float_array(equity_curve)
    if equity.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - (equity / peaks)
    return float(drawdowns.max(initial=0.0))


def summarize_equity_curve(equity_curve):
    equity = _to_1d_float_array(equity_curve)
    if equity.size == 0:
        raise ValueError("Empty equity_curve")

    r = simple_returns(equity)
    total_return = float(equity[-1] / equity[0] - 1.0) if equity[0] != 0 else float("nan")

    return {
        "start_value": float(equity[0]),
        "final_value": float(equity[-1]),
        "pnl": float(equity[-1] - equity[0]),
        "total_return": total_return,
        "mean_step_return": float(r.mean()) if r.size else float("nan"),
        "volatility": float(r.std(ddof=1)) if r.size >= 2 else float("nan"),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(equity),
        "num_steps": int(equity.size - 1),
    }

