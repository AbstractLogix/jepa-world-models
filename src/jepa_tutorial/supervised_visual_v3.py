"""Small model and metric primitives for supervised visual study v3."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .action_conditioned_visual import EMBEDDING_DIM, VisualEncoder

V3_VERSION = "visual-jepa-supervised-positive-control-v3"
DATA_SEEDS = (9701, 9702, 9703)
MODEL_SEEDS = (9801, 9802, 9803)
TEST_SEEDS = (9911, 9912, 9913, 9914, 9915)

METHOD_STATE_ESTIMATOR = "visual_state_estimator"
METHOD_ABS_RAW = "supervised_visual_absolute_raw"
METHOD_ABS_STANDARDIZED = "supervised_visual_absolute_standardized"
METHOD_RESIDUAL = "supervised_visual_residual"
METHOD_ACTION_SHUFFLE = "supervised_visual_action_shuffle"
METHOD_HANDCRAFTED_EXACT = "handcrafted_state_exact_dynamics"
METHOD_RIDGE = "handcrafted_feature_ridge"

NEURAL_METHODS = (
    METHOD_STATE_ESTIMATOR,
    METHOD_ABS_RAW,
    METHOD_ABS_STANDARDIZED,
    METHOD_RESIDUAL,
    METHOD_ACTION_SHUFFLE,
)
DYNAMICS_CANDIDATES = (METHOD_ABS_RAW, METHOD_ABS_STANDARDIZED, METHOD_RESIDUAL)
ACTION_METHODS = (
    METHOD_HANDCRAFTED_EXACT,
    METHOD_RIDGE,
    METHOD_ABS_RAW,
    METHOD_ABS_STANDARDIZED,
    METHOD_RESIDUAL,
    METHOD_ACTION_SHUFFLE,
)

MAX_POSITION_MAE = 0.01
MAX_VELOCITY_MAE = 0.002
MAX_CONTRAST_ERROR = 0.25
MIN_ORDER_ACCURACY = 0.80
MIN_VALID_STARTS = 12
MAX_CLIPPED_FRACTION = 0.05
MIN_TARGET_SCALE = 1e-6
LOW_SPREAD_THRESHOLD = 1e-3
MAX_LOW_SPREAD_DIMS = 30
MIN_EFFECTIVE_RANK = 2.0
MIN_PROBE_R2 = 0.80


class VisualStateEstimator(nn.Module):
    """Estimate current position and velocity from two rendered frames."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.encoder = VisualEncoder(embedding_dim)
        self.head = nn.Linear(embedding_dim, 2)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(context))


class SupervisedVisualDynamics(nn.Module):
    """V2-matched absolute-state visual model used by the v3 controls."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.encoder = VisualEncoder(embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if actions.shape != (context.shape[0], 1):
            raise ValueError("actions must have shape [batch] or [batch, 1]")
        embedding = self.encoder(context)
        return self.predictor(torch.cat((embedding, actions.to(embedding.dtype)), dim=-1))


def context_tensor(
    frames: np.ndarray | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Stack previous then current RGB frames and scale pixels to [0, 1]."""
    values = frames
    if values.ndim == 4 and values.shape[0] in (2, 3):
        values = values[None, ...]
    if values.ndim != 5 or values.shape[1] not in (2, 3) or values.shape[-1] != 3:
        raise ValueError("frames must have shape [batch, 2 or 3, height, width, 3]")

    if isinstance(values, np.ndarray):
        if values.dtype != np.uint8:
            raise ValueError("NumPy frame inputs must use uint8 pixels")
        stacked = np.concatenate((values[:, 0], values[:, 1]), axis=-1)
        tensor = torch.from_numpy(np.ascontiguousarray(stacked)).permute(0, 3, 1, 2)
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
    else:
        if values.dtype == torch.uint8:
            stacked = torch.cat((values[:, 0], values[:, 1]), dim=-1)
            tensor = stacked.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        else:
            stacked = torch.cat((values[:, 0], values[:, 1]), dim=-1)
            tensor = stacked.permute(0, 3, 1, 2).contiguous().float()
    return tensor.to(device=device, non_blocking=True)


def fit_target_standardizer(
    targets: np.ndarray,
    *,
    minimum_scale: float = MIN_TARGET_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the fixed per-coordinate target transform on training rows only."""
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise ValueError("targets must be a non-empty [rows, 2] matrix")
    if not np.isfinite(values).all() or not math.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("targets and minimum_scale must be finite")
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0, ddof=0), minimum_scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def standardize_targets(
    targets: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    spread = np.asarray(scale, dtype=np.float32)
    if values.shape[-1:] != (2,) or center.shape != (2,) or spread.shape != (2,):
        raise ValueError("targets, mean, and scale must end in two coordinates")
    if not np.isfinite(values).all() or not np.isfinite(center).all():
        raise ValueError("targets and transform values must be finite")
    if not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError("target scales must be finite and positive")
    return ((values - center) / spread).astype(np.float32)


def unstandardize_targets(
    targets: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    spread = np.asarray(scale, dtype=np.float32)
    if values.shape[-1:] != (2,) or center.shape != (2,) or spread.shape != (2,):
        raise ValueError("targets, mean, and scale must end in two coordinates")
    if not np.isfinite(values).all() or not np.isfinite(center).all():
        raise ValueError("targets and transform values must be finite")
    if not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError("target scales must be finite and positive")
    return (values * spread + center).astype(np.float32)


def residual_target(next_state: np.ndarray, handcrafted_state: np.ndarray) -> np.ndarray:
    next_values = np.asarray(next_state, dtype=np.float32)
    base = np.asarray(handcrafted_state, dtype=np.float32)
    if next_values.shape != base.shape or next_values.shape[-1:] != (2,):
        raise ValueError("next_state and handcrafted_state must have matching final dimension 2")
    if not np.isfinite(next_values).all() or not np.isfinite(base).all():
        raise ValueError("residual inputs must be finite")
    return next_values - base


def residual_state(handcrafted_state: np.ndarray, predicted_residual: np.ndarray) -> np.ndarray:
    base = np.asarray(handcrafted_state, dtype=np.float32)
    residual = np.asarray(predicted_residual, dtype=np.float32)
    if base.shape != residual.shape or base.shape[-1:] != (2,):
        raise ValueError("handcrafted_state and residual must have matching final dimension 2")
    if not np.isfinite(base).all() or not np.isfinite(residual).all():
        raise ValueError("residual state inputs must be finite")
    return base + residual


def shuffle_training_actions(actions: np.ndarray, *, seed: int) -> np.ndarray:
    """Permute only the supplied training labels with a stable local RNG."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("training actions must be a finite vector with at least two values")
    return values[np.random.default_rng(int(seed)).permutation(values.size)].copy()


def fit_handcrafted_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """Fit a train-standardized three-feature multi-output ridge regressor."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or y.shape != (x.shape[0], 2) or x.shape[0] < 2:
        raise ValueError("features must be [rows, 3] and targets must be [rows, 2]")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge inputs and alpha must be finite; alpha must be positive")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0, ddof=0), MIN_TARGET_SCALE)
    z = (x - mean) / scale
    target_mean = y.mean(axis=0)
    centered_target = y - target_mean
    gram = z.T @ z / x.shape[0]
    cross = z.T @ centered_target / x.shape[0]
    coefficient = np.linalg.solve(gram + float(alpha) * np.eye(3), cross)
    return {
        "feature_mean": mean.astype(np.float32),
        "feature_scale": scale.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "coefficient": coefficient.astype(np.float32),
        "alpha": float(alpha),
    }


def predict_handcrafted_ridge(features: np.ndarray, model: Mapping[str, object]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    scale = np.asarray(model["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(model["target_mean"], dtype=np.float32)
    coefficient = np.asarray(model["coefficient"], dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or mean.shape != (3,) or scale.shape != (3,):
        raise ValueError("ridge inputs must have three features")
    if target_mean.shape != (2,) or coefficient.shape != (3, 2):
        raise ValueError("ridge model has invalid target dimensions")
    prediction = ((x - mean) / scale) @ coefficient + target_mean
    if not np.isfinite(prediction).all():
        raise FloatingPointError("ridge model produced non-finite predictions")
    return prediction.astype(np.float32)


def dynamics_seed_gate(record: Mapping[str, object]) -> dict[str, bool]:
    """Apply frozen v3 representation, state, action, and clipping thresholds."""
    required = (
        "finite",
        "position_mae",
        "velocity_mae",
        "position_median_normalized_contrast_error",
        "velocity_median_normalized_contrast_error",
        "position_order_accuracy",
        "velocity_order_accuracy",
        "position_valid_starts",
        "velocity_valid_starts",
        "unbounded_prediction_fraction",
    )
    present = all(key in record for key in required)
    finite_values = present and bool(record.get("finite"))
    if present:
        try:
            finite_values = finite_values and all(
                math.isfinite(float(record[key]))
                for key in required[1:]
            )
        except (TypeError, ValueError):
            finite_values = False
    finite_pass = bool(finite_values)
    representation_fields = (
        "low_variance_dims",
        "effective_rank",
        "position_probe_r2",
        "velocity_probe_r2",
    )
    try:
        representation_finite = all(
            math.isfinite(float(record[key])) for key in representation_fields
        )
    except (KeyError, TypeError, ValueError):
        representation_finite = False
    representation_pass = bool(
        representation_finite
        and int(record["low_variance_dims"]) <= MAX_LOW_SPREAD_DIMS
        and float(record["effective_rank"]) >= MIN_EFFECTIVE_RANK
        and float(record["position_probe_r2"]) >= MIN_PROBE_R2
        and float(record["velocity_probe_r2"]) >= MIN_PROBE_R2
    )
    position_state_pass = finite_pass and float(record["position_mae"]) <= MAX_POSITION_MAE
    velocity_state_pass = finite_pass and float(record["velocity_mae"]) <= MAX_VELOCITY_MAE
    contrast_pass = finite_pass and all(
        float(record[key]) <= MAX_CONTRAST_ERROR
        for key in (
            "position_median_normalized_contrast_error",
            "velocity_median_normalized_contrast_error",
        )
    )
    order_pass = finite_pass and all(
        float(record[key]) >= MIN_ORDER_ACCURACY
        for key in ("position_order_accuracy", "velocity_order_accuracy")
    )
    valid_start_pass = finite_pass and all(
        int(record[key]) >= MIN_VALID_STARTS
        for key in ("position_valid_starts", "velocity_valid_starts")
    )
    clipping_pass = finite_pass and float(record["unbounded_prediction_fraction"]) <= MAX_CLIPPED_FRACTION
    action_fidelity_pass = contrast_pass and order_pass and valid_start_pass
    return {
        "finite_pass": bool(finite_pass),
        "representation_finite": bool(representation_finite),
        "representation_pass": bool(representation_pass),
        "position_state_pass": bool(position_state_pass),
        "velocity_state_pass": bool(velocity_state_pass),
        "contrast_pass": bool(contrast_pass),
        "order_pass": bool(order_pass),
        "valid_start_pass": bool(valid_start_pass),
        "action_fidelity_pass": bool(action_fidelity_pass),
        "clipping_pass": bool(clipping_pass),
        "validation_pass": bool(
            finite_pass
            and representation_pass
            and position_state_pass
            and velocity_state_pass
            and action_fidelity_pass
            and clipping_pass
        ),
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def choose_candidate_method(
    records: Sequence[Mapping[str, object]],
    *,
    data_path_pass: bool,
    records_complete: bool,
) -> dict[str, object]:
    """Select one validation-qualified family and decide whether tests may open."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("method", ""))].append(row)

    candidate_summaries: dict[str, dict[str, object]] = {}
    for method in DYNAMICS_CANDIDATES:
        rows = grouped.get(method, [])
        by_seed = {int(row["model_seed"]): row for row in rows}
        complete = (
            set(by_seed) == set(MODEL_SEEDS)
            and len(rows) == len(MODEL_SEEDS)
            and all(str(row.get("status", "complete")) == "complete" for row in rows)
        )
        passing = [row for seed, row in by_seed.items() if seed in MODEL_SEEDS and _as_bool(row.get("validation_pass"))]
        eligible = complete and len(passing) >= 2 and all(
            "mean_normalized_contrast_error" in row and "normalized_state_error" in row
            for row in passing
        )
        if eligible:
            contrast = [float(row["mean_normalized_contrast_error"]) for row in passing]
            state_error = [float(row["normalized_state_error"]) for row in passing]
            eligible = all(math.isfinite(value) for value in (*contrast, *state_error))
        else:
            contrast = []
            state_error = []
        candidate_summaries[method] = {
            "complete": bool(complete),
            "passing_seed_count": len(passing),
            "eligible": bool(eligible),
            "median_contrast_error": float(np.median(contrast)) if eligible else None,
            "median_normalized_state_error": float(np.median(state_error)) if eligible else None,
        }

    shuffle_rows = grouped.get(METHOD_ACTION_SHUFFLE, [])
    shuffle_by_seed = {int(row["model_seed"]): row for row in shuffle_rows}
    shuffle_complete = (
        set(shuffle_by_seed) == set(MODEL_SEEDS)
        and len(shuffle_rows) == len(MODEL_SEEDS)
        and all(str(row.get("status", "complete")) == "complete" for row in shuffle_rows)
        and all("action_fidelity_pass" in row for row in shuffle_rows)
    )
    shuffle_failures = sum(
        not _as_bool(row.get("action_fidelity_pass"))
        for seed, row in shuffle_by_seed.items()
        if seed in MODEL_SEEDS
    )
    action_shuffle_valid = shuffle_complete and shuffle_failures >= 2

    eligible_methods = [method for method in DYNAMICS_CANDIDATES if candidate_summaries[method]["eligible"]]
    selected_method = None
    if eligible_methods:
        selected_method = min(
            eligible_methods,
            key=lambda method: (
                float(candidate_summaries[method]["median_contrast_error"]),
                float(candidate_summaries[method]["median_normalized_state_error"]),
                DYNAMICS_CANDIDATES.index(method),
            ),
        )
    heldout_open = bool(data_path_pass and records_complete and action_shuffle_valid and selected_method)
    if not data_path_pass:
        reason = "data-path positive controls failed"
    elif not records_complete:
        reason = "required validation records are incomplete"
    elif not action_shuffle_valid:
        reason = "action-shuffling negative control did not fail its action-fidelity gate"
    elif not selected_method:
        reason = "no learned visual dynamics family passed the validation gate on two seed pairs"
    else:
        reason = "frozen validation gate passed"
    return {
        "data_path_pass": bool(data_path_pass),
        "records_complete": bool(records_complete),
        "candidate_families": candidate_summaries,
        "action_shuffle_valid": bool(action_shuffle_valid),
        "action_shuffle_failure_count": int(shuffle_failures),
        "selected_method": selected_method,
        "heldout_open": bool(heldout_open),
        "reason": reason,
    }


def can_open_heldout(summary: Mapping[str, object]) -> bool:
    """Fail closed unless every frozen validation prerequisite is recorded."""
    return bool(
        _as_bool(summary.get("data_path_pass"))
        and _as_bool(summary.get("records_complete"))
        and _as_bool(summary.get("action_shuffle_valid"))
        and summary.get("selected_method") in DYNAMICS_CANDIDATES
        and _as_bool(summary.get("heldout_open"))
    )


def validate_method_run_coverage(rows: Sequence[Mapping[str, object]]) -> None:
    """Require exactly five neural arms and a ridge record for every seed pair."""
    expected = {
        (method, int(data_seed), int(model_seed))
        for data_seed, model_seed in zip(DATA_SEEDS, MODEL_SEEDS, strict=True)
        for method in NEURAL_METHODS
    }
    expected.update((METHOD_RIDGE, int(data_seed), 0) for data_seed in DATA_SEEDS)
    expected.update((METHOD_HANDCRAFTED_EXACT, int(data_seed), 0) for data_seed in DATA_SEEDS)
    observed: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (str(row["method"]), int(row["data_seed"]), int(row["model_seed"]))
        if key in observed:
            raise ValueError(f"duplicate model-run record: {key}")
        observed.add(key)
    if observed != expected:
        raise ValueError(
            f"model-run coverage mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
