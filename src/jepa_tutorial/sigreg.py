from __future__ import annotations

import torch


def _safe_var(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    return x.var(dim=dim, unbiased=False) + eps


def sigreg_loss(
    z: torch.Tensor,
    target_var: float = 1.0,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.05,
) -> torch.Tensor:
    """Simple SIGReg-inspired isotropy regularizer for teaching.

    This educational proxy encourages:
    1) per-dimension variance near `target_var`
    2) low off-diagonal covariance
    """
    if z.ndim != 2:
        raise ValueError(f"Expected z to be [batch, dim], got shape {tuple(z.shape)}")

    zc = z - z.mean(dim=0, keepdim=True)

    # Variance matching term.
    var = _safe_var(zc, dim=0)
    var_term = ((var - target_var) ** 2).mean()

    # Off-diagonal covariance penalty.
    cov = (zc.T @ zc) / max(1, zc.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = (off_diag**2).mean()

    return lambda_var * var_term + lambda_cov * cov_term
