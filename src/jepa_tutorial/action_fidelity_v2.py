"""Metrics and stable readouts for the visual JEPA action-fidelity v2 study."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

V2_VERSION = "visual-jepa-action-fidelity-v2"
ACTION_VALUES = (-1.0, 0.0, 1.0)
LOW_SPREAD_THRESHOLD = 1e-3
LOW_SPREAD_MAX_DIMS = 16
MIN_EFFECTIVE_RANK = 8.0
MIN_PROBE_FEATURE_SCALE = 1e-3
PROBE_ALPHAS = (0.01, 0.1, 1.0, 10.0)


def stable_seed(*parts: object) -> int:
    """Return a stable 32-bit seed from JSON-encoded parts."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def split_episode_indices(
    episode_ids: np.ndarray, *, train_episode_split: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by episode identity and reject empty or overlapping partitions."""
    episodes = np.asarray(episode_ids)
    if episodes.ndim != 1 or episodes.size < 2:
        raise ValueError("episode_ids must be a one-dimensional array with at least two rows")
    if not isinstance(train_episode_split, int) or train_episode_split < 1:
        raise ValueError("train_episode_split must be a positive integer")
    train = np.flatnonzero(episodes < train_episode_split)
    validation = np.flatnonzero(episodes >= train_episode_split)
    if not train.size or not validation.size:
        raise ValueError("episode-level train and validation partitions must both be non-empty")
    if np.intersect1d(train, validation).size:
        raise RuntimeError("episode-level partitions overlap")
    if set(np.unique(episodes[train])).intersection(set(np.unique(episodes[validation]))):
        raise ValueError("an episode appears in both train and validation partitions")
    return train, validation


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_rank_diagnostics(embeddings: torch.Tensor | np.ndarray) -> dict[str, float | int | bool]:
    """Report spread, covariance spectrum, effective rank, and embedding norms."""
    values = (
        embeddings.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(embeddings, torch.Tensor)
        else np.asarray(embeddings, dtype=np.float64)
    )
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("embeddings must be a finite [rows, dimensions] matrix")
    finite = bool(np.isfinite(values).all())
    if not finite:
        return {
            "finite": False,
            "rows": int(values.shape[0]),
            "dimension": int(values.shape[1]),
            "mean_std": math.nan,
            "low_variance_dims": int(values.shape[1]),
            "covariance_trace": math.nan,
            "effective_rank": 0.0,
            "mean_norm": math.nan,
            "rms_norm": math.nan,
            "max_norm": math.nan,
        }

    centered = values - values.mean(axis=0, keepdims=True)
    spread = centered.std(axis=0, ddof=0)
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    trace = float(eigenvalues.sum())
    if trace <= 1e-20:
        rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > trace * 1e-12] / trace
        rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    norms = np.linalg.norm(values, axis=1)
    return {
        "finite": True,
        "rows": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "mean_std": float(spread.mean()),
        "min_std": float(spread.min()),
        "max_std": float(spread.max()),
        "low_variance_dims": int(np.sum(spread < LOW_SPREAD_THRESHOLD)),
        "covariance_trace": trace,
        "effective_rank": rank,
        "mean_norm": float(norms.mean()),
        "rms_norm": float(np.sqrt(np.mean(norms**2))),
        "max_norm": float(norms.max()),
    }


def tensor_norm(values: torch.Tensor) -> float:
    """Return the global L2 norm of a tensor as a Python float."""
    return float(torch.linalg.vector_norm(values.detach().float()).cpu().item())


def parameter_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    """Return the global L2 norm of available parameter gradients."""
    squares = [parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu().item())


def gradient_vector(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    parts = [gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
    if not parts:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(parts)


def gradient_cosine(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-12) -> float:
    """Return cosine similarity between two flattened gradient vectors."""
    if first.shape != second.shape:
        raise ValueError("gradient vectors must have the same shape")
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= epsilon:
        return 0.0
    return float(torch.dot(first, second).div(denominator).cpu().item())


def fit_stable_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    alphas: Sequence[float] = PROBE_ALPHAS,
    minimum_feature_scale: float = MIN_PROBE_FEATURE_SCALE,
) -> dict[str, object]:
    """Fit a train-only standardized ridge probe and select alpha on validation."""
    x_train = np.asarray(train_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    x_validation = np.asarray(validation_x, dtype=np.float64)
    y_validation = np.asarray(validation_y, dtype=np.float64)
    if x_train.ndim != 2 or x_validation.ndim != 2 or x_train.shape[1] != x_validation.shape[1]:
        raise ValueError("train and validation features must have matching [rows, dimensions] shapes")
    if y_train.ndim != 2 or y_validation.ndim != 2 or y_train.shape[1] != y_validation.shape[1]:
        raise ValueError("train and validation targets must have matching [rows, outputs] shapes")
    if x_train.shape[0] != y_train.shape[0] or x_validation.shape[0] != y_validation.shape[0]:
        raise ValueError("feature and target row counts must match")
    if not all(np.isfinite(x).all() for x in (x_train, y_train, x_validation, y_validation)):
        raise ValueError("probe inputs must be finite")
    if not alphas or any(not math.isfinite(float(alpha)) or float(alpha) <= 0 for alpha in alphas):
        raise ValueError("probe alphas must be finite and positive")
    if not math.isfinite(minimum_feature_scale) or minimum_feature_scale <= 0:
        raise ValueError("minimum_feature_scale must be finite and positive")

    feature_mean = x_train.mean(axis=0)
    feature_scale = np.maximum(x_train.std(axis=0), minimum_feature_scale)
    standardized_train = (x_train - feature_mean) / feature_scale
    standardized_validation = (x_validation - feature_mean) / feature_scale
    target_mean = y_train.mean(axis=0)
    centered_target = y_train - target_mean
    sample_count = x_train.shape[0]
    gram = standardized_train.T @ standardized_train / sample_count
    cross = standardized_train.T @ centered_target / sample_count

    candidates: list[tuple[float, float, np.ndarray, float]] = []
    identity = np.eye(gram.shape[0], dtype=np.float64)
    for alpha in alphas:
        system = gram + float(alpha) * identity
        coefficient = np.linalg.solve(system, cross)
        prediction = standardized_validation @ coefficient + target_mean
        mse = float(np.mean(np.square(prediction - y_validation)))
        condition = float(np.linalg.cond(system))
        candidates.append((mse, float(alpha), coefficient, condition))
    mse, alpha, coefficient, condition = min(candidates, key=lambda row: (row[0], row[1]))
    raw_coefficient = coefficient / feature_scale[:, None]
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "coef": coefficient.astype(np.float32),
        "alpha": alpha,
        "validation_mse": mse,
        "gram_condition": condition,
        "standardized_coef_l2": float(np.linalg.norm(coefficient)),
        "standardized_coef_max_abs": float(np.max(np.abs(coefficient))),
        "raw_coef_l2": float(np.linalg.norm(raw_coefficient)),
        "raw_coef_max_abs": float(np.max(np.abs(raw_coefficient))),
        "minimum_observed_feature_scale": float(np.min(x_train.std(axis=0))),
    }


def apply_probe_numpy(features: np.ndarray, probe: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(probe["feature_mean"], dtype=np.float64)
    scale = np.asarray(probe["feature_scale"], dtype=np.float64)
    target_mean = np.asarray(probe["target_mean"], dtype=np.float64)
    coefficient = np.asarray(probe["coef"], dtype=np.float64)
    return ((values - mean) / scale) @ coefficient + target_mean


def probe_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("probe predictions and targets must have matching [rows, 2] shapes")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("probe predictions and targets must be finite")
    error = predicted - actual
    denominator = np.square(actual - actual.mean(axis=0)).sum(axis=0)
    numerator = np.square(error).sum(axis=0)
    r2 = np.where(denominator > 1e-15, 1.0 - numerator / denominator, 0.0)
    mae = np.abs(error).mean(axis=0)
    return {
        "position_mae": float(mae[0]),
        "velocity_mae": float(mae[1]),
        "position_r2": float(r2[0]),
        "velocity_r2": float(r2[1]),
    }


def probe_bootstrap_rmse(
    train_x: np.ndarray,
    train_y: np.ndarray,
    episode_ids: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
    seed: int,
    replicates: int = 20,
) -> dict[str, float]:
    """Measure readout variation under deterministic episode-block resampling."""
    features = np.asarray(train_x, dtype=np.float64)
    targets = np.asarray(train_y, dtype=np.float64)
    groups = np.asarray(episode_ids)
    if features.shape[0] != targets.shape[0] or groups.shape != (features.shape[0],):
        raise ValueError("episode IDs must align with probe training rows")
    unique_groups = np.unique(groups)
    if unique_groups.size < 2 or replicates < 2:
        raise ValueError("probe stability needs at least two episodes and two refits")
    validation = np.asarray(validation_x, dtype=np.float64)
    if validation.ndim != 2 or validation.shape[1] != features.shape[1]:
        raise ValueError("validation features must match the training feature dimensions")
    if not np.isfinite(features).all() or not np.isfinite(targets).all() or not np.isfinite(validation).all():
        raise ValueError("bootstrap probe inputs must be finite")
    rng = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    identity = np.eye(features.shape[1], dtype=np.float64)
    for _ in range(replicates):
        chosen = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        xb_raw = features[indices]
        mean = xb_raw.mean(axis=0)
        scale = np.maximum(xb_raw.std(axis=0), MIN_PROBE_FEATURE_SCALE)
        xb = (xb_raw - mean) / scale
        yb = targets[indices]
        target_mean = yb.mean(axis=0)
        gram = xb.T @ xb / len(indices)
        cross = xb.T @ (yb - target_mean) / len(indices)
        coefficient = np.linalg.solve(gram + alpha * identity, cross)
        predictions.append(((validation - mean) / scale) @ coefficient + target_mean)
    stack = np.stack(predictions)
    variation = stack.std(axis=0, ddof=0)
    return {
        "position_rmse": float(np.sqrt(np.mean(np.square(variation[:, 0])))),
        "velocity_rmse": float(np.sqrt(np.mean(np.square(variation[:, 1])))),
        "replicates": int(replicates),
    }


def action_effect_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize paired action contrasts, normalized errors, and pair ordering."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["start_id"]))].append(row)
    expected = set(ACTION_VALUES)
    per_start: list[dict[str, object]] = []
    for (method, start_id), group in sorted(grouped.items()):
        by_action = {float(row["action"]): row for row in group}
        if set(by_action) != expected or len(group) != 3:
            raise ValueError(f"action group {(method, start_id)} must contain actions -1, 0, and +1")
        metrics: dict[str, object] = {"method": method, "start_id": start_id}
        for coordinate in ("position", "velocity"):
            actual = {a: float(by_action[a][f"actual_{coordinate}"]) for a in ACTION_VALUES}
            predicted = {a: float(by_action[a][f"predicted_{coordinate}_raw"]) for a in ACTION_VALUES}
            if not all(math.isfinite(value) for value in (*actual.values(), *predicted.values())):
                raise ValueError(f"non-finite action effect in {(method, start_id, coordinate)}")
            actual_span = max(actual.values()) - min(actual.values())
            predicted_span = max(predicted.values()) - min(predicted.values())
            contrast_errors = [
                abs((predicted[action] - predicted[0.0]) - (actual[action] - actual[0.0]))
                for action in (-1.0, 1.0)
            ]
            normalized_error = float(np.mean(contrast_errors) / max(actual_span, 1e-4))
            comparisons = []
            correct = 0
            for left, right in ((-1.0, 0.0), (-1.0, 1.0), (0.0, 1.0)):
                actual_delta = actual[right] - actual[left]
                predicted_delta = predicted[right] - predicted[left]
                if abs(actual_delta) > 1e-5:
                    comparisons.append((actual_delta, predicted_delta))
                    if actual_delta * predicted_delta > 0:
                        correct += 1
            order_accuracy = correct / len(comparisons) if comparisons else math.nan
            metrics[f"actual_{coordinate}_span"] = actual_span
            metrics[f"predicted_{coordinate}_span"] = predicted_span
            metrics[f"{coordinate}_normalized_contrast_error"] = normalized_error
            metrics[f"{coordinate}_order_correct"] = correct
            metrics[f"{coordinate}_order_total"] = len(comparisons)
            metrics[f"{coordinate}_order_accuracy"] = order_accuracy
        per_start.append(metrics)

    summary: dict[str, object] = {"per_start": per_start, "starts": len(per_start)}
    for coordinate in ("position", "velocity"):
        accuracy_rows = [row for row in per_start if int(row[f"{coordinate}_order_total"]) > 0]
        errors = [float(row[f"{coordinate}_normalized_contrast_error"]) for row in accuracy_rows]
        correct = sum(int(row[f"{coordinate}_order_correct"]) for row in accuracy_rows)
        total = sum(int(row[f"{coordinate}_order_total"]) for row in accuracy_rows)
        summary[f"{coordinate}_median_normalized_contrast_error"] = float(np.median(errors)) if errors else math.nan
        summary[f"{coordinate}_order_correct"] = correct
        summary[f"{coordinate}_order_total"] = total
        summary[f"{coordinate}_order_accuracy"] = correct / total if total else math.nan
        summary[f"{coordinate}_valid_starts"] = len(accuracy_rows)
    return summary


def validation_representation_pass(diagnostics: Mapping[str, object]) -> bool:
    """Apply the preregistered representation-only checks."""
    return (
        bool(diagnostics.get("finite"))
        and int(diagnostics.get("low_variance_dims", 32)) <= LOW_SPREAD_MAX_DIMS
        and float(diagnostics.get("effective_rank", 0.0)) >= MIN_EFFECTIVE_RANK
    )


def unclipped_state_diagnostics(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a task-bounded copy and an elementwise clipping mask."""
    if state.shape[-1:] != (2,) or not state.is_floating_point():
        raise ValueError("state must be a floating-point tensor ending in position and velocity")
    limits = torch.tensor([[-1.2, -0.07], [0.6, 0.07]], device=state.device, dtype=state.dtype)
    if not bool(torch.isfinite(state).all().item()):
        raise ValueError("unclipped state must be finite")
    clipped = torch.maximum(torch.minimum(state, limits[1]), limits[0])
    clipped[..., 1] = torch.where(clipped[..., 0] <= -1.2, 0.0, clipped[..., 1])
    mask = clipped != state
    return clipped, mask


def validate_rollout_records(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject incomplete, duplicate, or non-finite per-horizon rollout records."""
    required = {"method", "model_seed", "start_id", "horizon", "actual_position", "actual_velocity", "predicted_position", "predicted_velocity", "status"}
    seen: set[tuple[str, str, str, int]] = set()
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"rollout row {index} missing fields: {sorted(missing)}")
        key = (str(row["method"]), str(row["model_seed"]), str(row["start_id"]), int(row["horizon"]))
        if key in seen:
            raise ValueError(f"duplicate rollout row: {key}")
        seen.add(key)
        if row["status"] == "complete":
            for field in ("actual_position", "actual_velocity", "predicted_position", "predicted_velocity"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"rollout row {index} has non-finite {field}")
        elif not str(row.get("error", "")).strip():
            raise ValueError(f"failed rollout row {index} needs an error reason")


def validate_action_triplet_records(rows: Sequence[Mapping[str, object]]) -> None:
    """Require complete, finite action triplets for every recorded context."""
    required = {
        "method", "objective", "model_seed", "data_seed", "start_id", "action",
        "actual_position", "actual_velocity", "predicted_position_raw", "predicted_velocity_raw",
    }
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"action row {index} missing fields: {sorted(missing)}")
        key = tuple(str(row[name]) for name in ("method", "objective", "model_seed", "data_seed", "start_id"))
        grouped[key].append(row)
    for key, group in grouped.items():
        actions = [float(row["action"]) for row in group]
        if len(group) != 3 or set(actions) != set(ACTION_VALUES):
            raise ValueError(f"action context {key} must have one row for each of -1, 0, and +1")
        for row in group:
            for name in ("actual_position", "actual_velocity", "predicted_position_raw", "predicted_velocity_raw"):
                if not math.isfinite(float(row[name])):
                    raise ValueError(f"action context {key} has non-finite {name}")


def validate_experiment_completeness(
    model_rows: Sequence[Mapping[str, object]],
    action_rows: Sequence[Mapping[str, object]],
    *,
    data_seeds: Sequence[int] = (9301, 9302, 9303),
    model_seeds: Sequence[int] = (9401, 9402, 9403),
    methods: Sequence[tuple[str, str]] = (
        ("jepa", "sigreg"), ("jepa", "visreg_style"), ("jepa", "variance_cov"),
        ("supervised_visual", "supervised_state_mse"), ("state_model", "privileged_state_mse"),
    ),
) -> None:
    """Validate paired fit coverage and the expected action-triplet rows."""
    expected = {
        (method, objective, int(model_seed), int(data_seed))
        for data_seed, model_seed in zip(data_seeds, model_seeds, strict=True)
        for method, objective in methods
    }
    observed: dict[tuple[str, str, int, int], Mapping[str, object]] = {}
    for row in model_rows:
        key = (str(row["method"]), str(row["objective"]), int(row["model_seed"]), int(row["data_seed"]))
        if key in observed:
            raise ValueError(f"duplicate model-fit record: {key}")
        observed[key] = row
    if set(observed) != expected:
        raise ValueError(
            f"model-fit coverage mismatch: missing={sorted(expected - set(observed))}, extra={sorted(set(observed) - expected)}"
        )
    validate_action_triplet_records(action_rows)
    action_groups: dict[tuple[str, str, int, int], set[str]] = defaultdict(set)
    for row in action_rows:
        key = (str(row["method"]), str(row["objective"]), int(row["model_seed"]), int(row["data_seed"]))
        action_groups[key].add(str(row["start_id"]))
    expected_action_groups = {
        key for key, fit in observed.items() if fit.get("status") == "complete"
    }
    for data_seed in data_seeds:
        expected_action_groups.update(
            {
                ("exact", "simulator_oracle", 0, int(data_seed)),
                ("handcrafted", "visual_state_plus_oracle", 0, int(data_seed)),
            }
        )
    if set(action_groups) != expected_action_groups:
        raise ValueError(
            f"action-triplet coverage mismatch: missing={sorted(expected_action_groups - set(action_groups))}, "
            f"extra={sorted(set(action_groups) - expected_action_groups)}"
        )
    for key, fit in observed.items():
        if fit.get("status") == "complete" and len(action_groups.get(key, set())) != 20:
            raise ValueError(f"complete fit {key} must have 20 action-triplet contexts")
        if fit.get("status") != "complete" and action_groups.get(key):
            raise ValueError(f"failed fit {key} must not have action-triplet predictions")
    for data_seed in data_seeds:
        for method, objective in (("exact", "simulator_oracle"), ("handcrafted", "visual_state_plus_oracle")):
            key = (method, objective, 0, int(data_seed))
            if len(action_groups.get(key, set())) != 20:
                raise ValueError(f"positive control {key} must have 20 action-triplet contexts")


def validate_file_manifest(manifest_path: str | Path, root: str | Path) -> None:
    """Verify byte counts and SHA-256 values for retained public evidence files."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    base = Path(root).resolve()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("evidence manifest must contain a non-empty files list")
    for entry in files:
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe evidence path in manifest: {relative}")
        path = (base / relative).resolve()
        if os.path.commonpath((str(base), str(path))) != str(base):
            raise ValueError(f"evidence path escapes manifest root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"evidence file is missing: {relative}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"evidence file size changed: {relative}")
        if sha256_file(str(path)) != str(entry["sha256"]):
            raise ValueError(f"evidence file hash changed: {relative}")
