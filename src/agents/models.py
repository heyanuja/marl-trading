from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


def _init_layer(layer: nn.Module, *, gain: float = 1.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    return layer


def _mlp(
    sizes: Sequence[int],
    *,
    activation: type[nn.Module] = nn.Tanh,
    output_activation: type[nn.Module] | None = None,
    gain: float = 1.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:], strict=True):
        layers.append(_init_layer(nn.Linear(in_dim, out_dim), gain=gain))
        layers.append(activation())
    if output_activation is not None:
        layers[-1] = output_activation()
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        *,
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        trunk_sizes = (int(input_dim), *[int(h) for h in hidden_sizes])
        self.trunk = _mlp(trunk_sizes, gain=nn.init.calculate_gain("tanh"))

        last_dim = int(trunk_sizes[-1])
        self.actor_mean = _init_layer(nn.Linear(last_dim, int(action_dim)), gain=0.01)
        self.critic = _init_layer(nn.Linear(last_dim, 1), gain=1.0)

        self.actor_log_std = nn.Parameter(torch.full((int(action_dim),), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        value = self.critic(features).squeeze(-1)
        return mean, std, value

    def get_action(self, state: torch.Tensor, *, deterministic: bool = False):
        mean, std, value = self(state)
        dist = Normal(mean, std)

        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.sample()

        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob -= torch.log1p(-action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        mean, std, value = self(states)
        dist = Normal(mean, std)

        eps = 1e-6
        clipped_actions = torch.clamp(actions, -1.0 + eps, 1.0 - eps)
        pre_tanh_action = torch.atanh(clipped_actions)

        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        log_prob -= torch.log1p(-clipped_actions.pow(2) + eps).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value
