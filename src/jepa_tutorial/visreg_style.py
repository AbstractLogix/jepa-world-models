"""Independent VISReg-style regularization from the published equations."""

from __future__ import annotations

import math

import torch


def visreg_style_regularization_loss(
    embeddings: torch.Tensor,
    *,
    num_projections: int = 128,
    scale_weight: float = 1.0,
    shape_weight: float = 1.0,
    center_weight: float = 1.0,
    epsilon: float = 1e-4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return scale, sliced-shape, and center penalties for a batch of embeddings.

    This is an independent implementation of the paper equations. It does not
    implement the paper's multi-view training recipe.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[1] < 1:
        raise ValueError("embeddings must have shape [batch, dimension] with at least 2 rows")
    if not embeddings.is_floating_point():
        raise ValueError("embeddings must use a floating-point dtype")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ValueError("embeddings must contain only finite values")
    if num_projections < 1:
        raise ValueError("num_projections must be at least 1")
    weights = (scale_weight, shape_weight, center_weight)
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("component weights must be finite and nonnegative")
    if not any(weight > 0 for weight in weights):
        raise ValueError("at least one component weight must be positive")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    batch_size, dimension = embeddings.shape
    mean = embeddings.mean(dim=0)
    centered = embeddings - mean
    spread = centered.std(dim=0, unbiased=False)
    scale_loss = (1.0 - spread).square().mean()

    normalized = centered / (spread.detach().unsqueeze(0) + epsilon)
    directions = torch.randn(
        (dimension, num_projections),
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    projected = normalized @ directions
    sorted_projected = projected.sort(dim=0).values

    probabilities = (
        torch.arange(batch_size, device=embeddings.device, dtype=embeddings.dtype) + 0.5
    ) / batch_size
    gaussian_quantiles = math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)
    shape_loss = (sorted_projected - gaussian_quantiles.unsqueeze(1)).square().mean()
    center_loss = mean.square().sum()

    components = {
        "scale_loss": scale_loss,
        "shape_loss": shape_loss,
        "center_loss": center_loss,
    }
    total_loss = (
        scale_weight * scale_loss
        + shape_weight * shape_loss
        + center_weight * center_loss
    )
    return total_loss, components
