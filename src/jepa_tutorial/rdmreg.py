from __future__ import annotations

import torch


def rectified_distribution_matching_loss(
    z: torch.Tensor,
    sparsity_weight: float = 1e-3,
    entropy_weight: float = 1e-3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Educational Rectified LpJEPA proxy regularizer.

    - Rectify activations with ReLU (sparse, non-negative codes)
    - Encourage sparsity via L1
    - Encourage diversity via entropy over normalized activations
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z to be [batch, dim], got shape {tuple(z.shape)}")

    zr = torch.relu(z)

    l1_term = zr.mean()

    p = zr / (zr.sum(dim=1, keepdim=True) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=1).mean()

    return sparsity_weight * l1_term - entropy_weight * entropy
