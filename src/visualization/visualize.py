from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


class TradingVisualizer:
    def __init__(self, style: str = "darkgrid"):
        sns.set_theme(style=style)
        plt.rcParams["figure.figsize"] = [12, 8]
        plt.rcParams["font.size"] = 12
        self.colors = sns.color_palette("husl", 10)

    def plot_training_dashboard(self, metrics: list[dict[str, Any]], *, save_path: str) -> None:
        df = pd.DataFrame(metrics)
        if df.empty:
            raise ValueError("metrics is empty")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2)

        ax1 = fig.add_subplot(gs[0, :])
        self._plot_series_with_rolling(
            ax1,
            df["episode"],
            df["episode_reward_mean"],
            title="Episode Reward (Mean Across Agents)",
            ylabel="Reward",
        )

        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_series_with_rolling(
            ax2,
            df["episode"],
            df["final_equity_mean"],
            title="Final Equity (Mean Across Agents)",
            ylabel="Portfolio Value",
        )

        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_aux_panel(ax3, df)

        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_distribution(ax4, df["episode_reward_mean"], title="Reward Distribution")

        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_distribution(ax5, df["final_equity_mean"], title="Final Equity Distribution")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_episode_diagnostics(
        self,
        prices: np.ndarray,
        equity: np.ndarray,
        *,
        save_path: str,
    ) -> None:
        prices = np.asarray(prices, dtype=np.float32)
        equity = np.asarray(equity, dtype=np.float32)
        if prices.ndim != 2:
            raise ValueError(f"Expected prices shape (T, R), got {prices.shape}")
        if equity.ndim != 2:
            raise ValueError(f"Expected equity shape (T, A), got {equity.shape}")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, :])
        for i in range(prices.shape[1]):
            ax1.plot(prices[:, i], label=f"Asset {i+1}", color=self.colors[i], alpha=0.9)
        ax1.set_title("Prices (Single Episode)")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[1, 0])
        for a in range(equity.shape[1]):
            ax2.plot(equity[:, a], label=f"Agent {a+1}", color=self.colors[a], alpha=0.9)
        ax2.set_title("Equity Curve (Single Episode)")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Portfolio Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        if equity.shape[0] >= 2:
            step_returns = equity[1:] / np.clip(equity[:-1], 1e-8, None) - 1.0
            sns.histplot(step_returns.flatten(), bins=40, kde=True, ax=ax3, color=self.colors[4])
        ax3.set_title("Step Return Distribution")
        ax3.set_xlabel("Return")
        ax3.set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def create_trading_animation(
        self,
        prices: np.ndarray,
        *,
        save_path: str = "trading_animation.gif",
        max_frames: int = 200,
    ) -> None:
        prices_array = np.asarray(prices, dtype=np.float32)
        if prices_array.ndim != 2:
            raise ValueError(f"Expected prices shape (T, R), got {prices_array.shape}")

        fig, ax = plt.subplots(figsize=(12, 6))
        lines = [
            ax.plot([], [], label=f"Asset {i+1}", color=self.colors[i])[0]
            for i in range(prices_array.shape[1])
        ]

        ax.set_xlim(0, len(prices_array))
        ax.set_ylim(float(prices_array.min() * 0.95), float(prices_array.max() * 1.05))
        ax.set_title("Price Evolution (Single Episode)", fontsize=14, pad=20)
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        ax.legend()

        def animate(frame: int):
            x = np.arange(frame)
            for line, ys in zip(lines, prices_array.T, strict=True):
                line.set_data(x, ys[:frame])
            return lines

        frames = int(min(max_frames, len(prices_array)))
        anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
        try:
            anim.save(save_path, writer="pillow")
        except Exception as e:
            raise RuntimeError(
                f"Failed to save GIF to {save_path}. Install pillow or pass --no-plots. ({e})"
            ) from e
        finally:
            plt.close(fig)

    def plot_price_evolution_storyboard(
        self,
        prices: np.ndarray,
        *,
        save_path: str,
        panels: int = 12,
    ) -> None:
        prices_array = np.asarray(prices, dtype=np.float32)
        if prices_array.ndim != 2:
            raise ValueError(f"Expected prices shape (T, R), got {prices_array.shape}")

        total_steps, num_assets = prices_array.shape
        if total_steps < 2:
            raise ValueError("Need at least 2 timesteps for a storyboard")

        n_panels = int(max(1, panels))
        frames = np.unique(np.linspace(2, total_steps, n_panels, dtype=np.int32))
        frames = frames[frames >= 2]
        if frames.size == 0:
            frames = np.array([total_steps], dtype=np.int32)

        cols = min(4, int(frames.size))
        rows = int(np.ceil(float(frames.size) / float(cols)))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.25, rows * 3.25), sharey=True)
        axes_arr = np.atleast_1d(axes).reshape(rows, cols)

        y_min = float(prices_array.min() * 0.95)
        y_max = float(prices_array.max() * 1.05)

        for idx, frame in enumerate(frames.tolist()):
            r, c = divmod(idx, cols)
            ax = axes_arr[r, c]
            x = np.arange(frame)
            for asset in range(num_assets):
                ax.plot(
                    x,
                    prices_array[:frame, asset],
                    color=self.colors[asset],
                    linewidth=1.5,
                    alpha=0.9,
                )

            ax.set_xlim(0, total_steps - 1)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"t={frame - 1}", fontsize=11)
            ax.grid(True, alpha=0.25)

            if r < rows - 1:
                ax.set_xticklabels([])
            if c > 0:
                ax.set_yticklabels([])

            if idx == 0:
                handles = [
                    Line2D([0], [0], color=self.colors[a], lw=2, label=f"Asset {a+1}")
                    for a in range(num_assets)
                ]
                ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

        for j in range(int(frames.size), rows * cols):
            r, c = divmod(j, cols)
            axes_arr[r, c].axis("off")

        fig.suptitle("Price Evolution (Storyboard)", fontsize=16, y=0.98)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -----------------
    # Internals

    def _plot_series_with_rolling(self, ax, x, y, *, title: str, ylabel: str, window: int = 50) -> None:
        df = pd.DataFrame({"x": x, "y": y})
        rolling_mean = df["y"].rolling(window=window).mean()
        rolling_std = df["y"].rolling(window=window).std()

        ax.plot(df["x"], df["y"], alpha=0.25, color=self.colors[0], label="raw")
        ax.plot(df["x"], rolling_mean, color=self.colors[1], label=f"{window}-ep mean")
        ax.fill_between(
            df["x"],
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.15,
            color=self.colors[1],
        )

        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_aux_panel(self, ax, df: pd.DataFrame) -> None:
        cols = [c for c in ["policy_loss", "value_loss", "entropy", "approx_kl"] if c in df.columns]
        if not cols:
            cem_cols = [c for c in ["cem_best_score", "cem_score"] if c in df.columns]
            if cem_cols:
                for i, col in enumerate(cem_cols):
                    ax.plot(df["episode"], df[col], label=col, color=self.colors[i], alpha=0.9)
                ax.set_title("CEM Metrics")
                ax.set_xlabel("Episode")
                ax.legend()
                ax.grid(True, alpha=0.3)
                return

            cost_cols = [c for c in ["fees_total_mean", "slippage_total_mean"] if c in df.columns]
            turnover_col = "turnover_total_mean" if "turnover_total_mean" in df.columns else None
            if cost_cols or turnover_col:
                for i, col in enumerate(cost_cols):
                    ax.plot(df["episode"], df[col], label=col, color=self.colors[i], alpha=0.9)
                ax.set_title("Execution Costs")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Cost")
                ax.grid(True, alpha=0.3)

                if turnover_col:
                    ax2 = ax.twinx()
                    ax2.plot(
                        df["episode"],
                        df[turnover_col],
                        label=turnover_col,
                        color=self.colors[3],
                        alpha=0.85,
                        linestyle="--",
                    )
                    ax2.set_ylabel("Turnover")
                    h1, l1 = ax.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    ax2.legend(h1 + h2, l1 + l2, loc="upper left")
                else:
                    ax.legend(loc="upper left")
                return

            ax.set_title("Auxiliary Metrics (missing)")
            ax.axis("off")
            return

        for i, col in enumerate(cols):
            ax.plot(df["episode"], df[col], label=col, color=self.colors[i], alpha=0.9)
        ax.set_title("Optimization Metrics")
        ax.set_xlabel("Episode")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_distribution(self, ax, values: pd.Series, *, title: str) -> None:
        xs = values.to_numpy(dtype=np.float64)
        xs = xs[np.isfinite(xs)]
        if xs.size:
            sns.histplot(xs, bins=40, kde=True, ax=ax, color=self.colors[2])
        ax.set_title(title)
