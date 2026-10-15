from __future__ import annotations

import torch
import torch.nn.functional as F


def predictive_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE predictive objective between predicted and target embeddings."""
    return F.mse_loss(pred, target)


def lejepa_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigreg_term: torch.Tensor,
    lambda_sigreg: float = 0.1,
) -> torch.Tensor:
    """LeJEPA-style combined objective used in the educational notebooks.

    total = predictive_loss + lambda_sigreg * sigreg_term
    """
    return predictive_loss(pred, target) + lambda_sigreg * sigreg_term
