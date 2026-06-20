from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from .models import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epochs: int = 10
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalized Advantage Estimation (GAE).

    rewards: (T, B)
    values:  (T, B) value estimates for states at t
    dones:   (T, B) 1.0 if terminal at t, else 0.0
    last_values: (B,) value estimates for state at T
    """
    r = np.asarray(rewards, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    d = np.asarray(dones, dtype=np.float32)
    lv = np.asarray(last_values, dtype=np.float32)

    if r.shape != v.shape or r.shape != d.shape:
        raise ValueError(f"Expected rewards/values/dones shapes to match, got {r.shape}, {v.shape}, {d.shape}")
    if lv.shape != (r.shape[1],):
        raise ValueError(f"Expected last_values shape {(r.shape[1],)}, got {lv.shape}")

    t_steps, batch = r.shape
    adv = np.zeros((t_steps, batch), dtype=np.float32)
    last_adv = np.zeros(batch, dtype=np.float32)

    for t in reversed(range(t_steps)):
        next_values = lv if t == t_steps - 1 else v[t + 1]
        next_nonterminal = 1.0 - d[t]
        delta = r[t] + gamma * next_values * next_nonterminal - v[t]
        last_adv = delta + gamma * gae_lambda * next_nonterminal * last_adv
        adv[t] = last_adv

    returns = adv + v
    return adv, returns


class PPOAgent:
    def __init__(
        self,
        *,
        input_dim: int,
        action_dim: int,
        config: PPOConfig | None = None,
        device: str | torch.device | None = None,
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.actor_critic = ActorCritic(
            int(input_dim),
            int(action_dim),
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config.lr)

    def select_action(self, obs: np.ndarray, *, deterministic: bool = False):
        """
        obs: (obs_dim,) or (B, obs_dim)
        returns: actions (B, act_dim), log_probs (B,), values (B,)
        """
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        if obs_arr.ndim != 2:
            raise ValueError(f"Expected obs with shape (obs_dim,) or (B, obs_dim), got {obs_arr.shape}")

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_arr).to(self.device)
            action_t, log_prob_t, value_t = self.actor_critic.get_action(
                obs_t, deterministic=deterministic
            )
        actions = action_t.cpu().numpy().astype(np.float32, copy=False)
        log_probs = log_prob_t.cpu().numpy().astype(np.float32, copy=False)
        values = value_t.cpu().numpy().astype(np.float32, copy=False)

        if np.asarray(obs).ndim == 1:
            return actions[0], float(log_probs[0]), float(values[0])
        return actions, log_probs, values

    def value(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_arr).to(self.device)
            v = self.actor_critic(obs_t)[2]
        out = v.cpu().numpy().astype(np.float32, copy=False)
        return out[0] if np.asarray(obs).ndim == 1 else out

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """
        batch keys: states, actions, old_log_probs, returns, advantages
        """
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = states.shape[0]
        minibatch_size = int(min(self.config.minibatch_size, batch_size))
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")

        metrics: dict[str, float] = {}
        for epoch in range(int(self.config.ppo_epochs)):
            perm = torch.randperm(batch_size, device=self.device)
            clip_fracs: list[float] = []
            kls: list[float] = []
            losses: list[float] = []
            pol_losses: list[float] = []
            val_losses: list[float] = []
            entropies: list[float] = []

            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                log_prob, entropy, values = self.actor_critic.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(log_prob - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_mean

                with torch.no_grad():
                    approx_kl = (mb_old_log - log_prob).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_param).float().mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                losses.append(float(loss.item()))
                pol_losses.append(float(policy_loss.item()))
                val_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))

            metrics = {
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "policy_loss": float(np.mean(pol_losses)) if pol_losses else float("nan"),
                "value_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
                "entropy": float(np.mean(entropies)) if entropies else float("nan"),
                "approx_kl": float(np.mean(kls)) if kls else float("nan"),
                "clip_frac": float(np.mean(clip_fracs)) if clip_fracs else float("nan"),
                "epoch": float(epoch + 1),
            }

        return metrics

    def save(self, path: str) -> None:
        payload: dict[str, Any] = {
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
