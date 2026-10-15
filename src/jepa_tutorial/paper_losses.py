"""Small paper-aligned SIGReg and RDMReg losses for the teaching notebooks."""

from __future__ import annotations

import math

import torch


def _validate_matrix(
    embeddings: torch.Tensor,
    *,
    require_nonnegative: bool = False,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape [batch, dimension], got {tuple(embeddings.shape)}"
        )
    if embeddings.shape[0] < 2 or embeddings.shape[1] < 1:
        raise ValueError("Embeddings must have at least two rows and one feature")
    if not embeddings.is_floating_point():
        raise ValueError("Embeddings must use a floating-point dtype")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Embeddings must contain only finite values")
    if require_nonnegative and bool((embeddings < 0).any()):
        raise ValueError("Rectified LpJEPA embeddings must be nonnegative")


def _random_unit_directions(
    dimension: int,
    num_projections: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    directions = torch.randn(
        (dimension, num_projections),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)


def epps_pulley_sigreg_loss(
    embeddings: torch.Tensor,
    *,
    num_projections: int = 512,
    num_points: int = 17,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Match random projections to a standard Gaussian with Epps-Pulley SIGReg."""
    _validate_matrix(embeddings)
    if num_projections < 1:
        raise ValueError("num_projections must be at least 1")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    batch_size, dimension = embeddings.shape
    directions = _random_unit_directions(
        dimension,
        num_projections,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    frequencies = torch.linspace(
        -5.0,
        5.0,
        num_points,
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    projected_phases = (embeddings @ directions).unsqueeze(-1) * frequencies
    empirical_cf = torch.exp(1j * projected_phases).mean(dim=0)
    target_cf = torch.exp(-0.5 * frequencies.square())
    error = (empirical_cf - target_cf).abs().square() * target_cf
    per_projection = torch.trapezoid(error, frequencies, dim=-1) * batch_size
    return per_projection.mean()


def sample_rectified_laplace(
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ReLU(Laplace(0, 1/sqrt(2))) samples for the paper's p=1 target."""
    if len(shape) != 2 or any(size < 1 for size in shape):
        raise ValueError("shape must contain two positive dimensions")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")

    uniform = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    epsilon = torch.finfo(dtype).eps
    centered = uniform.clamp(epsilon, 1.0 - epsilon) - 0.5
    laplace = -math.sqrt(0.5) * torch.sign(centered) * torch.log1p(
        -2.0 * centered.abs()
    )
    return torch.relu(laplace)


def rectified_laplace_rdmreg_loss(
    embeddings: torch.Tensor,
    *,
    num_projections: int = 512,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Match nonnegative features to Rectified Laplace samples with sliced W2."""
    _validate_matrix(embeddings, require_nonnegative=True)
    if num_projections < 1:
        raise ValueError("num_projections must be at least 1")

    batch_size, dimension = embeddings.shape
    target = sample_rectified_laplace(
        (batch_size, dimension),
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    directions = _random_unit_directions(
        dimension,
        num_projections,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    projected = embeddings @ directions
    projected_target = target @ directions
    sorted_features = projected.sort(dim=0).values
    sorted_target = projected_target.sort(dim=0).values
    return (sorted_features - sorted_target).square().mean()
