from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GRASPConfig:
    horizon: int = 16
    action_dim: int = 1
    n_candidates: int = 128
    n_iters: int = 20
    lr: float = 0.05


class GRASPPlanner:
    """Parallel stochastic gradient planner (educational reference)."""

    def __init__(self, config: GRASPConfig):
        self.config = config

    def initialize(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.config.n_candidates,
            self.config.horizon,
            self.config.action_dim,
            device=device,
            requires_grad=True,
        )

    def optimize(self, objective_fn, device: torch.device | None = None):
        device = device or torch.device("cpu")
        actions = self.initialize(device=device)
        optimizer = torch.optim.Adam([actions], lr=self.config.lr)

        for _ in range(self.config.n_iters):
            optimizer.zero_grad(set_to_none=True)
            scores = objective_fn(actions)
            loss = -scores.mean()
            loss.backward()
            optimizer.step()

        best_idx = int(torch.argmax(objective_fn(actions.detach())).item())
        return actions.detach()[best_idx]
