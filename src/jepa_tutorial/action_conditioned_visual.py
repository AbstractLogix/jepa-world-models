"""Compact visual and state dynamics models for the action-conditioned lesson."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F

IMAGE_SIZE = (48, 128)
CONTEXT_CHANNELS = 6
EMBEDDING_DIM = 32


def select_device(prefer_cuda: bool = True) -> torch.device:
    """Use CUDA only when the installed build can run and sync a small kernel."""
    if not prefer_cuda or not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        probe = torch.ones((4, 4), device="cuda")
        result = probe @ probe
        torch.cuda.synchronize()
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("CUDA probe returned a non-finite value")
        return torch.device("cuda")
    except Exception as exc:  # CUDA availability can be true for an unsupported GPU build.
        warnings.warn(
            f"CUDA probe failed ({exc}); using CPU for this run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")


def _observation_batch(observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim != 4 or observations.shape[1] != CONTEXT_CHANNELS:
        raise ValueError("observations must have shape [batch, 6, height, width]")
    if not observations.is_floating_point():
        observations = observations.float().div(255.0)
    else:
        observations = observations.float()
    if observations.shape[-2:] != IMAGE_SIZE:
        observations = F.interpolate(
            observations, size=IMAGE_SIZE, mode="bilinear", align_corners=False
        )
    return observations.clamp(0.0, 1.0)


class VisualEncoder(nn.Module):
    """Small encoder for two consecutive rendered RGB frames."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(CONTEXT_CHANNELS, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 8)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 8, embedding_dim),
            nn.LayerNorm(embedding_dim, elementwise_affine=False),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.features(_observation_batch(observations))


class ActionPredictor(nn.Module):
    """Predict a next embedding from a current embedding and scalar action."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if action.shape != (embedding.shape[0], 1):
            raise ValueError("action must have shape [batch] or [batch, 1]")
        return self.network(torch.cat((embedding, action.to(embedding.dtype)), dim=-1))


class ActionConditionedJEPA(nn.Module):
    """Predict next-context embeddings without reconstructing pixels."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.encoder = VisualEncoder(embedding_dim)
        self.predictor = ActionPredictor(embedding_dim)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)

    def predict_embedding(
        self, observations: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(self.encoder(observations), action)

    def forward(
        self,
        observations: torch.Tensor,
        action: torch.Tensor,
        target_observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self.predict_embedding(observations, action)
        target = self.encoder(target_observations)
        return predicted, target


class ImageReconstructionDynamics(nn.Module):
    """Action-conditioned latent model with a decoder for future RGB frames."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.encoder = VisualEncoder(embedding_dim)
        self.predictor = ActionPredictor(embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 3 * 8),
            nn.GELU(),
            nn.Unflatten(1, (128, 3, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)

    def predict_embedding(
        self, observations: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(self.encoder(observations), action)

    def predict_image(
        self, observations: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(self.predict_embedding(observations, action))

    def forward(
        self,
        observations: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        predicted = self.predict_embedding(observations, action)
        return predicted, self.decoder(predicted)


class StateDynamics(nn.Module):
    """Privileged state-space reference; it does not consume image observations."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        if state.ndim != 2 or state.shape[1] != 2:
            raise ValueError("state must have shape [batch, 2] for position and velocity")
        if action.shape != (state.shape[0], 1):
            raise ValueError("action must have shape [batch] or [batch, 1]")
        return self.network(torch.cat((state, action.to(state.dtype)), dim=-1))


def embedding_collapse_diagnostics(
    embeddings: torch.Tensor,
    *,
    std_threshold: float = 1e-3,
    collapsed_fraction: float = 0.5,
) -> dict[str, float | int | bool]:
    """Report per-dimension spread and a predeclared near-constant collapse flag."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[1] < 1:
        raise ValueError("embeddings must have shape [batch, dimension] with at least 2 rows")
    finite = bool(torch.isfinite(embeddings).all().item())
    spread = embeddings.float().std(dim=0, unbiased=False)
    low_variance_dims = int((spread < std_threshold).sum().item())
    dimension = int(embeddings.shape[1])
    return {
        "finite": finite,
        "mean_std": float(spread.mean().item()),
        "low_variance_dims": low_variance_dims,
        "dimension": dimension,
        "collapsed": (not finite) or low_variance_dims / dimension > collapsed_fraction,
    }
