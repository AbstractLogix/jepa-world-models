from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """Small value head for value-guided planning demos."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def value_loss(pred_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_values, returns)


def value_shaped_distance(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Distance used in value-shaping intuition demos.

    Lower is better for planning. Uses smooth L2 + scaling.
    """
    return alpha * ((z1 - z2) ** 2).sum(dim=-1).sqrt()
