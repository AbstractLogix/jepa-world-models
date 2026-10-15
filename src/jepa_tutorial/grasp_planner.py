from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PlannerConfig:
    horizon: int = 16
    action_dim: int = 1
    n_candidates: int = 256
    n_iters: int = 10
    lr: float = 0.05


class SimpleGRASPPlanner:
    """Minimal gradient-based action sequence optimizer.

    The world model/value model callables are intentionally abstract so
    notebooks can plug in toy or real modules.
    """

    def __init__(self, config: PlannerConfig) -> None:
        self.config = config

    def plan(self, init_actions: torch.Tensor, objective_fn) -> torch.Tensor:
        """Optimize action sequences to maximize objective_fn(actions)."""
        actions = init_actions.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([actions], lr=self.config.lr)

        for _ in range(self.config.n_iters):
            optimizer.zero_grad(set_to_none=True)
            score = objective_fn(actions).mean()
            loss = -score
            loss.backward()
            optimizer.step()

        return actions.detach()
