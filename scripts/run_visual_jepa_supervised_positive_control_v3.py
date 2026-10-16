"""Run the preregistered supervised visual positive-control study v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_tutorial.action_conditioned_visual import select_device  # noqa: E402
from jepa_tutorial.action_fidelity_v2 import (  # noqa: E402
    ACTION_VALUES,
    action_effect_metrics,
    apply_probe_numpy,
    effective_rank_diagnostics,
    fit_stable_probe,
    probe_metrics,
    split_episode_indices,
    stable_seed,
    unclipped_state_diagnostics,
)
from jepa_tutorial.decision_study import (  # noqa: E402
    MAX_POSITION,
    MAX_SPEED,
    MIN_POSITION,
    mountaincar_step,
)
from jepa_tutorial.supervised_visual_v3 import (  # noqa: E402
    DATA_SEEDS,
    DYNAMICS_CANDIDATES,
    MAX_CLIPPED_FRACTION,
    METHOD_ABS_RAW,
    METHOD_ABS_STANDARDIZED,
    METHOD_ACTION_SHUFFLE,
    METHOD_HANDCRAFTED_EXACT,
    METHOD_RESIDUAL,
    METHOD_RIDGE,
    METHOD_STATE_ESTIMATOR,
    MODEL_SEEDS,
    NEURAL_METHODS,
    TEST_SEEDS,
    V3_VERSION,
    SupervisedVisualDynamics,
    VisualStateEstimator,
    can_open_heldout,
    choose_candidate_method,
    context_tensor,
    dynamics_seed_gate,
    fit_handcrafted_ridge,
    fit_target_standardizer,
    predict_handcrafted_ridge,
    residual_target,
    shuffle_training_actions,
    standardize_targets,
    validate_method_run_coverage,
)
from jepa_tutorial.visual_mountaincar import (  # noqa: E402
    TransitionData,
    collect_random_transitions,
    make_env,
    render_frame,
    resize_frame,
    visual_state,
)

PROTOCOL_PATH = ROOT / "docs/reproductions/protocols/visual-jepa-supervised-positive-control-v3.md"
EVIDENCE_DIR = ROOT / "docs/evidence/visual-jepa-supervised-positive-control-v3"
LOCK_PATH = EVIDENCE_DIR / "protocol-lock.json"
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/e/codex-research/jepa-world-models-tutorial/"
    "visual-jepa-supervised-positive-control-v3/validation"
)
DEFAULT_VALIDATION_DIR = DEFAULT_OUTPUT_DIR
TRAIN_EPISODE_SPLIT = 16
EPISODES = 20
VALIDATION_CONTEXTS = 20
TEST_STEPS = (40, 90, 140, 190)
ROLLOUT_HORIZON = 60
EPOCHS = 35
BATCH_SIZE = 256
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0001
GRADIENT_EPOCHS = (1, 5, 10, 15, 20, 25, 30, 35)
GPU_BUDGET_SECONDS = 60 * 60
GPU_MEMORY_LIMIT_MIB = 2048.0
STORAGE_LIMIT_BYTES = 5 * 1024**3

LOCKED_FILES = (
    "docs/reproductions/protocols/visual-jepa-supervised-positive-control-v3.md",
    "src/jepa_tutorial/action_conditioned_visual.py",
    "src/jepa_tutorial/action_fidelity_v2.py",
    "src/jepa_tutorial/decision_study.py",
    "src/jepa_tutorial/supervised_visual_v3.py",
    "src/jepa_tutorial/visual_mountaincar.py",
    "pyproject.toml",
    "uv.lock",
    "scripts/run_visual_jepa_supervised_positive_control_v3.py",
    "scripts/analyze_visual_jepa_supervised_positive_control_v3.py",
    "scripts/validate_visual_jepa_supervised_positive_control_v3.py",
    "tests/test_visual_jepa_supervised_positive_control_v3.py",
    "requirements-torch.txt",
)


class StudyError(RuntimeError):
    """A frozen study invariant or resource limit failed."""


class ComputeBudgetExceeded(StudyError):
    """The preregistered GPU time or memory limit was exceeded."""


@dataclass
class PreparedGroup:
    data_seed: int
    data: TransitionData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    handcrafted_states: np.ndarray
    intervention_indices: np.ndarray
    exact_label_max_error: float
    handcrafted_state_metrics: dict[str, float]
    target_scale_rows: list[dict[str, object]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fields})


def protocol_lock_payload() -> dict[str, object]:
    missing = [name for name in LOCKED_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"study files are missing before protocol lock: {missing}")
    return {
        "protocol_version": V3_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "files": {name: sha256_file(ROOT / name) for name in LOCKED_FILES},
    }


def freeze_protocol_lock() -> dict[str, object]:
    """Write the immutable protocol and source hash before the first fit."""
    payload = protocol_lock_payload()
    if LOCK_PATH.exists():
        current = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if current != payload and any(
            current.get(key) != payload.get(key)
            for key in ("protocol_version", "protocol_sha256", "files")
        ):
            raise FileExistsError(
                f"protocol lock already exists and differs: {LOCK_PATH}; do not overwrite it"
            )
        return current
    payload["locked_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(LOCK_PATH, payload)
    return payload


def verify_protocol_lock() -> dict[str, object]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"v3 protocol lock is missing: {LOCK_PATH}")
    recorded = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    current = protocol_lock_payload()
    for key in ("protocol_version", "protocol_sha256", "files"):
        if recorded.get(key) != current.get(key):
            raise StudyError(f"v3 protocol/code lock mismatch in {key}; stop and version the change")
    return recorded


def require_e_drive(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not str(resolved).startswith("/mnt/e/"):
        raise ValueError(f"transient data and checkpoints must stay on E:, got {resolved}")
    return resolved


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def check_storage_budget(path: Path) -> None:
    if directory_bytes(path) > STORAGE_LIMIT_BYTES:
        raise ComputeBudgetExceeded("transient study output exceeded the 5 GiB E: limit")


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "CPU"


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _prepare_group(data_seed: int) -> PreparedGroup:
    data = collect_random_transitions(data_seed, episodes=EPISODES)
    train_indices, validation_indices = split_episode_indices(
        data.episode_ids, train_episode_split=TRAIN_EPISODE_SPLIT
    )
    handcrafted = np.stack(
        [visual_state(frames[0], frames[1]) for frames in data.frames], axis=0
    ).astype(np.float32)
    selected_rng = np.random.default_rng(
        stable_seed(V3_VERSION, "validation-action-contexts", int(data_seed))
    )
    intervention_indices = np.sort(
        selected_rng.choice(validation_indices, size=VALIDATION_CONTEXTS, replace=False)
    ).astype(np.int64)
    exact = mountaincar_step(data.states, data.actions)
    exact_label_max_error = float(np.max(np.abs(exact - data.next_states)))
    state_error = handcrafted[validation_indices] - data.states[validation_indices]
    state_metrics = {
        "position_mae": float(np.mean(np.abs(state_error[:, 0]))),
        "velocity_mae": float(np.mean(np.abs(state_error[:, 1]))),
    }
    target_scales: list[dict[str, object]] = []
    for split_name, indices in (("train", train_indices), ("validation", validation_indices)):
        groups = {
            "current_state": data.states[indices],
            "next_state": data.next_states[indices],
            "delta_state": data.next_states[indices] - data.states[indices],
        }
        for target_name, values in groups.items():
            for coordinate, index in (("position", 0), ("velocity", 1)):
                column = values[:, index].astype(np.float64)
                target_scales.append(
                    {
                        "protocol_version": V3_VERSION,
                        "data_seed": data_seed,
                        "split": split_name,
                        "target": target_name,
                        "coordinate": coordinate,
                        "rows": int(len(column)),
                        "mean": float(column.mean()),
                        "std": float(column.std(ddof=0)),
                        "variance": float(column.var(ddof=0)),
                        "minimum": float(column.min()),
                        "maximum": float(column.max()),
                    }
                )
    return PreparedGroup(
        data_seed=data_seed,
        data=data,
        train_indices=train_indices,
        validation_indices=validation_indices,
        handcrafted_states=handcrafted,
        intervention_indices=intervention_indices,
        exact_label_max_error=exact_label_max_error,
        handcrafted_state_metrics=state_metrics,
        target_scale_rows=target_scales,
    )


def _feature_matrix(
    states: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        (np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32).reshape(-1, 1)),
        axis=1,
    )


def _coordinate_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("state predictions and targets must have matching shape [rows, 2]")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise FloatingPointError("state metrics received non-finite values")
    error = predicted - actual
    return {
        "position_mae": float(np.mean(np.abs(error[:, 0]))),
        "velocity_mae": float(np.mean(np.abs(error[:, 1]))),
        "position_mse": float(np.mean(np.square(error[:, 0]))),
        "velocity_mse": float(np.mean(np.square(error[:, 1]))),
    }


def _fit_complete_data_path(group: PreparedGroup) -> bool:
    return (
        group.exact_label_max_error <= 1e-6
        and group.handcrafted_state_metrics["position_mae"] <= 0.005
        and group.handcrafted_state_metrics["velocity_mae"] <= 0.0002
    )


def _model_objective(method: str) -> str:
    return {
        METHOD_STATE_ESTIMATOR: "current_state_raw_mse",
        METHOD_ABS_RAW: "next_state_raw_mse",
        METHOD_ABS_STANDARDIZED: "next_state_train_standardized_mse",
        METHOD_RESIDUAL: "next_state_residual_raw_mse",
        METHOD_ACTION_SHUFFLE: "next_state_raw_mse_shuffled_action",
    }[method]


def _model_for_method(method: str) -> torch.nn.Module:
    if method == METHOD_STATE_ESTIMATOR:
        return VisualStateEstimator()
    return SupervisedVisualDynamics()


def _fit_targets(
    method: str,
    group: PreparedGroup,
    indices: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> np.ndarray:
    data = group.data
    if method == METHOD_STATE_ESTIMATOR:
        return data.states[indices]
    if method == METHOD_RESIDUAL:
        return residual_target(data.next_states[indices], group.handcrafted_states[indices])
    if method == METHOD_ABS_STANDARDIZED:
        return standardize_targets(data.next_states[indices], target_mean, target_scale)
    return data.next_states[indices]


def _decode_model_output(
    method: str,
    raw: torch.Tensor,
    handcrafted_state: torch.Tensor,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    if method == METHOD_ABS_STANDARDIZED:
        return raw * target_scale + target_mean
    if method == METHOD_RESIDUAL:
        return handcrafted_state + raw
    return raw


def _objective_loss(
    method: str,
    raw: torch.Tensor,
    objective_target: torch.Tensor,
) -> torch.Tensor:
    del method
    return F.mse_loss(raw, objective_target)


def _gradient_norm(values: Sequence[torch.Tensor | None]) -> float:
    parts = [value.detach().float().reshape(-1) for value in values if value is not None]
    if not parts:
        return 0.0
    return float(torch.linalg.vector_norm(torch.cat(parts)).detach().cpu().item())


def _fixed_gradient_rows(
    model: torch.nn.Module,
    method: str,
    group: PreparedGroup,
    fixed_indices: np.ndarray,
    action_lookup: np.ndarray,
    device: torch.device,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    model_seed: int,
) -> list[dict[str, object]]:
    context = context_tensor(group.data.frames[fixed_indices], device=device)
    actions = torch.as_tensor(action_lookup[fixed_indices], dtype=torch.float32, device=device)
    if method == METHOD_STATE_ESTIMATOR:
        objective_target_np = group.data.states[fixed_indices]
    elif method == METHOD_RESIDUAL:
        objective_target_np = residual_target(
            group.data.next_states[fixed_indices], group.handcrafted_states[fixed_indices]
        )
    elif method == METHOD_ABS_STANDARDIZED:
        objective_target_np = standardize_targets(
            group.data.next_states[fixed_indices], target_mean, target_scale
        )
    else:
        objective_target_np = group.data.next_states[fixed_indices]
    objective_target = torch.as_tensor(objective_target_np, dtype=torch.float32, device=device)
    encoder_params = list(model.encoder.parameters())
    head = model.head if method == METHOD_STATE_ESTIMATOR else model.predictor
    head_params = list(head.parameters())
    rows: list[dict[str, object]] = []
    for coordinate, name in enumerate(("position", "velocity")):
        model.zero_grad(set_to_none=True)
        raw = model(context) if method == METHOD_STATE_ESTIMATOR else model(context, actions)
        loss = F.mse_loss(raw[:, coordinate], objective_target[:, coordinate])
        gradients = torch.autograd.grad(loss, encoder_params + head_params, allow_unused=True)
        encoder_gradients = gradients[: len(encoder_params)]
        head_gradients = gradients[len(encoder_params) :]
        rows.append(
            {
                "protocol_version": V3_VERSION,
                "method": method,
                "objective": _model_objective(method),
                "model_seed": model_seed,
                "data_seed": group.data_seed,
                "coordinate": name,
                "encoder_gradient_norm": _gradient_norm(encoder_gradients),
                "prediction_head_gradient_norm": _gradient_norm(head_gradients),
                "training_objective_coordinate_mse": float(loss.detach().cpu().item()),
                "fixed_batch_rows": int(len(fixed_indices)),
            }
        )
    return rows


def _evaluate_visual_model(
    model: torch.nn.Module,
    method: str,
    group: PreparedGroup,
    indices: np.ndarray,
    device: torch.device,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    outputs: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    encoder_norms: list[np.ndarray] = []
    predictor_norms: list[np.ndarray] = []
    model.eval()
    mean_tensor = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(target_scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            context = context_tensor(group.data.frames[batch_indices], device=device)
            action = torch.as_tensor(
                group.data.actions[batch_indices], dtype=torch.float32, device=device
            )
            embedding = model.encoder(context)
            if method == METHOD_STATE_ESTIMATOR:
                raw = model.head(embedding)
                decoded = raw
            else:
                raw = model.predictor(
                    torch.cat((embedding, action[:, None].to(embedding.dtype)), dim=-1)
                )
                handcrafted = torch.as_tensor(
                    group.handcrafted_states[batch_indices], dtype=torch.float32, device=device
                )
                decoded = _decode_model_output(method, raw, handcrafted, mean_tensor, scale_tensor)
            outputs.append(decoded.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
            encoder_norms.append(torch.linalg.vector_norm(embedding, dim=1).cpu().numpy())
            predictor_norms.append(torch.linalg.vector_norm(raw, dim=1).cpu().numpy())
    prediction = np.concatenate(outputs, axis=0).astype(np.float32)
    embedding_values = np.concatenate(embeddings, axis=0).astype(np.float32)
    diagnostics = effective_rank_diagnostics(embedding_values)
    diagnostics = {
        **{key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in diagnostics.items()},
        "mean_encoder_output_norm": float(np.mean(np.concatenate(encoder_norms))),
        "mean_prediction_head_output_norm": float(np.mean(np.concatenate(predictor_norms))),
    }
    return prediction, embedding_values, np.concatenate(predictor_norms), diagnostics


def _train_one(
    method: str,
    group: PreparedGroup,
    model_seed: int,
    device: torch.device,
    output_dir: Path,
    stage_gpu_seconds_used: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], torch.nn.Module, np.ndarray, np.ndarray, float]:
    """Fit one preregistered neural arm and return raw per-epoch diagnostics."""
    _set_seeds(model_seed)
    model = _model_for_method(method).to(device)
    target_mean, target_scale = fit_target_standardizer(group.data.next_states[group.train_indices])
    if method == METHOD_STATE_ESTIMATOR:
        target_mean, target_scale = fit_target_standardizer(group.data.states[group.train_indices])

    train_targets = _fit_targets(
        method, group, group.train_indices, target_mean, target_scale
    )
    training_actions = group.data.actions[group.train_indices].copy()
    shuffle_seed = stable_seed(V3_VERSION, "action-label-permutation", group.data_seed, model_seed)
    if method == METHOD_ACTION_SHUFFLE:
        training_actions = shuffle_training_actions(training_actions, seed=shuffle_seed)
    action_lookup = np.zeros(len(group.data.actions), dtype=np.float32)
    action_lookup[group.train_indices] = training_actions
    action_lookup[group.validation_indices] = group.data.actions[group.validation_indices]

    order_rng = torch.Generator(device="cpu")
    order_rng.manual_seed(
        stable_seed(V3_VERSION, "training-row-order", group.data_seed, model_seed)
    )
    train_count = len(group.train_indices)
    gradient_count = min(BATCH_SIZE, train_count)
    gradient_rng = np.random.default_rng(
        stable_seed(V3_VERSION, "fixed-gradient-batch", group.data_seed)
    )
    fixed_indices = np.sort(
        gradient_rng.choice(group.train_indices, size=gradient_count, replace=False)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mean_tensor = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(target_scale, dtype=torch.float32, device=device)
    history: list[dict[str, object]] = []
    gradients: list[dict[str, object]] = []
    fit_started = time.perf_counter()
    fit_device_seconds = 0.0
    peak_memory = 0.0
    status = "complete"
    error_message = ""
    epochs_completed = 0

    for epoch in range(1, EPOCHS + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        model.train()
        permutation = torch.randperm(train_count, generator=order_rng).numpy()
        objective_loss_sum = 0.0
        position_mse_sum = 0.0
        velocity_mse_sum = 0.0
        for start in range(0, train_count, BATCH_SIZE):
            selected = group.train_indices[permutation[start : start + BATCH_SIZE]]
            context = context_tensor(group.data.frames[selected], device=device)
            actions = torch.as_tensor(action_lookup[selected], dtype=torch.float32, device=device)
            objective_target = torch.as_tensor(
                train_targets[permutation[start : start + BATCH_SIZE]],
                dtype=torch.float32,
                device=device,
            )
            handcrafted = torch.as_tensor(
                group.handcrafted_states[selected], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            raw = model(context) if method == METHOD_STATE_ESTIMATOR else model(context, actions)
            loss = _objective_loss(method, raw, objective_target)
            if not bool(torch.isfinite(loss).item()):
                status = "failed"
                error_message = f"non-finite training loss at epoch {epoch}"
                break
            decoded = _decode_model_output(method, raw, handcrafted, mean_tensor, scale_tensor)
            state_targets = (
                torch.as_tensor(group.data.states[selected], dtype=torch.float32, device=device)
                if method == METHOD_STATE_ESTIMATOR
                else torch.as_tensor(group.data.next_states[selected], dtype=torch.float32, device=device)
            )
            loss.backward()
            optimizer.step()
            selected_count = len(selected)
            objective_loss_sum += float(loss.detach().cpu().item()) * selected_count
            raw_error = (decoded.detach() - state_targets).square().mean(dim=0)
            position_mse_sum += float(raw_error[0].cpu().item()) * selected_count
            velocity_mse_sum += float(raw_error[1].cpu().item()) * selected_count

        if status != "complete":
            break

        validation_prediction, _, _, _ = _evaluate_visual_model(
            model,
            method,
            group,
            group.validation_indices,
            device,
            target_mean,
            target_scale,
        )
        validation_target_state = (
            group.data.states[group.validation_indices]
            if method == METHOD_STATE_ESTIMATOR
            else group.data.next_states[group.validation_indices]
        )
        validation_raw_metrics = _coordinate_metrics(validation_prediction, validation_target_state)
        validation_objective_sum = 0.0
        validation_objective_count = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(group.validation_indices), BATCH_SIZE):
                selected = group.validation_indices[start : start + BATCH_SIZE]
                context = context_tensor(group.data.frames[selected], device=device)
                actions = torch.as_tensor(group.data.actions[selected], dtype=torch.float32, device=device)
                if method == METHOD_STATE_ESTIMATOR:
                    target_np = group.data.states[selected]
                elif method == METHOD_RESIDUAL:
                    target_np = residual_target(group.data.next_states[selected], group.handcrafted_states[selected])
                elif method == METHOD_ABS_STANDARDIZED:
                    target_np = standardize_targets(group.data.next_states[selected], target_mean, target_scale)
                else:
                    target_np = group.data.next_states[selected]
                target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
                raw = model(context) if method == METHOD_STATE_ESTIMATOR else model(context, actions)
                loss = _objective_loss(method, raw, target)
                validation_objective_sum += float(loss.cpu().item()) * len(selected)
                validation_objective_count += len(selected)
        history.append(
            {
                "protocol_version": V3_VERSION,
                "method": method,
                "objective": _model_objective(method),
                "model_seed": model_seed,
                "data_seed": group.data_seed,
                "epoch": epoch,
                "train_objective_loss": objective_loss_sum / train_count,
                "validation_objective_loss": validation_objective_sum / max(1, validation_objective_count),
                "train_position_mse_raw": position_mse_sum / train_count,
                "train_velocity_mse_raw": velocity_mse_sum / train_count,
                "validation_position_mse_raw": validation_raw_metrics["position_mse"],
                "validation_velocity_mse_raw": validation_raw_metrics["velocity_mse"],
                "validation_position_mae_raw": validation_raw_metrics["position_mae"],
                "validation_velocity_mae_raw": validation_raw_metrics["velocity_mae"],
                "training_rows": train_count,
                "validation_rows": len(group.validation_indices),
            }
        )
        if epoch in GRADIENT_EPOCHS:
            gradients.extend(
                _fixed_gradient_rows(
                    model,
                    method,
                    group,
                    fixed_indices,
                    action_lookup,
                    device,
                    target_mean,
                    target_scale,
                    model_seed,
                )
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device) / (1024**2))
        epoch_elapsed = time.perf_counter() - epoch_started
        fit_device_seconds += epoch_elapsed
        stage_gpu_seconds_used += epoch_elapsed
        epochs_completed = epoch
        if stage_gpu_seconds_used > GPU_BUDGET_SECONDS:
            status = "failed"
            error_message = "training exceeded the 60-minute compute budget"
        elif device.type == "cuda" and peak_memory > GPU_MEMORY_LIMIT_MIB:
            status = "failed"
            error_message = "peak allocated CUDA memory exceeded 2 GiB"
        check_storage_budget(output_dir)
        if status != "complete":
            break

    if status == "complete":
        prediction, embeddings, predictor_norms, rep_diagnostics = _evaluate_visual_model(
            model,
            method,
            group,
            group.validation_indices,
            device,
            target_mean,
            target_scale,
        )
        target_state = (
            group.data.states[group.validation_indices]
            if method == METHOD_STATE_ESTIMATOR
            else group.data.next_states[group.validation_indices]
        )
        metrics: dict[str, object] = _coordinate_metrics(prediction, target_state)
        finite = bool(
            np.isfinite(prediction).all()
            and all(math.isfinite(float(row["train_objective_loss"])) for row in history)
        )
        if not finite:
            status = "failed"
            error_message = "non-finite prediction or learning-curve value"
    else:
        rep_diagnostics = {
            "mean_std": None,
            "min_std": None,
            "max_std": None,
            "low_variance_dims": None,
            "covariance_trace": None,
            "effective_rank": None,
            "mean_encoder_output_norm": None,
            "mean_prediction_head_output_norm": None,
        }
        metrics = {"position_mae": None, "velocity_mae": None, "position_mse": None, "velocity_mse": None}
        finite = False
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{method}-model-{model_seed}-data-{group.data_seed}.pt"
    if status == "complete":
        torch.save(
            {
                "method": method,
                "model_seed": model_seed,
                "data_seed": group.data_seed,
                "model_state_dict": model.state_dict(),
                "target_mean": target_mean,
                "target_scale": target_scale,
                "protocol_version": V3_VERSION,
                "protocol_lock_sha256": sha256_file(LOCK_PATH),
            },
            checkpoint_path,
        )
    else:
        checkpoint_path = None
    fit_elapsed = time.perf_counter() - fit_started
    run = {
        "protocol_version": V3_VERSION,
        "stage": "validation",
        "method": method,
        "objective": _model_objective(method),
        "model_seed": model_seed,
        "data_seed": group.data_seed,
        "status": status,
        "epochs": epochs_completed,
        "train_rows": int(train_count),
        "validation_rows": int(len(group.validation_indices)),
        "train_seconds": fit_device_seconds,
        "fit_elapsed_seconds": fit_elapsed,
        "device": device.type,
        "device_name": _device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "precision": "float32",
        "peak_allocated_mib": peak_memory,
        "finite": finite,
        **metrics,
        "target_position_mean": float(target_mean[0]),
        "target_velocity_mean": float(target_mean[1]),
        "target_position_scale": float(target_scale[0]),
        "target_velocity_scale": float(target_scale[1]),
        "mean_std": rep_diagnostics["mean_std"],
        "min_std": rep_diagnostics["min_std"],
        "max_std": rep_diagnostics["max_std"],
        "low_variance_dims": rep_diagnostics["low_variance_dims"],
        "covariance_trace": rep_diagnostics["covariance_trace"],
        "effective_rank": rep_diagnostics["effective_rank"],
        "mean_encoder_output_norm": rep_diagnostics["mean_encoder_output_norm"],
        "mean_prediction_head_output_norm": rep_diagnostics["mean_prediction_head_output_norm"],
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
        "error": error_message,
    }
    check_storage_budget(output_dir)
    return run, history, gradients, model, target_mean, target_scale, stage_gpu_seconds_used


def _predict_neural_triplets(
    model: torch.nn.Module,
    method: str,
    group: PreparedGroup,
    device: torch.device,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw next states and d(predicted state)/d(action at zero)."""
    indices = group.intervention_indices
    frames = context_tensor(group.data.frames[indices], device=device)
    count = len(indices)
    actions = torch.tensor(ACTION_VALUES, dtype=torch.float32, device=device).repeat(count)
    contexts = frames.repeat_interleave(len(ACTION_VALUES), dim=0)
    handcrafted = torch.as_tensor(
        group.handcrafted_states[indices], dtype=torch.float32, device=device
    ).repeat_interleave(len(ACTION_VALUES), dim=0)
    mean_tensor = torch.as_tensor(target_mean, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(target_scale, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        raw = model(contexts, actions)
        decoded = _decode_model_output(method, raw, handcrafted, mean_tensor, scale_tensor)
    predictions = decoded.reshape(count, len(ACTION_VALUES), 2).cpu().numpy()
    zero_actions = torch.zeros((count, 1), dtype=torch.float32, device=device, requires_grad=True)
    zero_raw = model(frames, zero_actions)
    zero_decoded = _decode_model_output(
        method,
        zero_raw,
        torch.as_tensor(group.handcrafted_states[indices], dtype=torch.float32, device=device),
        mean_tensor,
        scale_tensor,
    )
    derivatives = []
    for coordinate in range(2):
        derivative = torch.autograd.grad(
            zero_decoded[:, coordinate].sum(),
            zero_actions,
            retain_graph=coordinate == 0,
        )[0]
        derivatives.append(derivative[:, 0].detach().cpu().numpy())
    return predictions.astype(np.float32), np.stack(derivatives, axis=1).astype(np.float32)


def _build_action_records(
    *,
    method: str,
    objective: str,
    model_seed: int,
    group: PreparedGroup,
    predictions: np.ndarray,
    derivatives: np.ndarray,
    stage: str = "validation",
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Create one raw row per context/action and one summary per context."""
    indices = group.intervention_indices
    actual = np.stack(
        [
            mountaincar_step(
                np.repeat(group.data.states[index][None, :], 3, axis=0),
                np.asarray(ACTION_VALUES, dtype=np.float32),
            )
            for index in indices
        ]
    )
    if predictions.shape != actual.shape or derivatives.shape != (len(indices), 2):
        raise ValueError("action predictions and derivatives have invalid dimensions")
    raw_rows: list[dict[str, object]] = []
    for context_index, source_index in enumerate(indices):
        start_id = f"d{group.data_seed}-row{int(source_index)}"
        for action_index, action in enumerate(ACTION_VALUES):
            predicted = predictions[context_index, action_index]
            target = actual[context_index, action_index]
            clipped_state, clipping_applied = _clip_rollout_state(predicted)
            raw_rows.append(
                {
                    "protocol_version": V3_VERSION,
                    "stage": stage,
                    "method": method,
                    "objective": objective,
                    "model_seed": model_seed,
                    "data_seed": group.data_seed,
                    "start_id": start_id,
                    "source_row": int(source_index),
                    "action": action,
                    "actual_position": float(target[0]),
                    "actual_velocity": float(target[1]),
                    "predicted_position_raw": float(predicted[0]),
                    "predicted_velocity_raw": float(predicted[1]),
                    "predicted_position_clipped": float(clipped_state[0]),
                    "predicted_velocity_clipped": float(clipped_state[1]),
                    "position_outside_bounds": bool(
                        predicted[0] < MIN_POSITION or predicted[0] > MAX_POSITION
                    ),
                    "velocity_outside_bounds": bool(
                        predicted[1] < -MAX_SPEED or predicted[1] > MAX_SPEED
                    ),
                    "environment_clip_applied": clipping_applied,
                    "position_error_raw": float(predicted[0] - target[0]),
                    "velocity_error_raw": float(predicted[1] - target[1]),
                }
            )
    effect = action_effect_metrics(raw_rows)
    summary_rows: list[dict[str, object]] = []
    clipped_predictions = 0
    total_predictions = int(predictions.shape[0] * predictions.shape[1])
    for context_index, row in enumerate(effect["per_start"]):
        values = predictions[context_index]
        outside = (
            (values[:, 0] < MIN_POSITION)
            | (values[:, 0] > MAX_POSITION)
            | (values[:, 1] < -MAX_SPEED)
            | (values[:, 1] > MAX_SPEED)
        )
        clipped_predictions += int(outside.sum())
        action_slope = (values[2] - values[0]) / 2.0
        actual_slope = (actual[context_index, 2] - actual[context_index, 0]) / 2.0
        summary_rows.append(
            {
                "protocol_version": V3_VERSION,
                "stage": stage,
                "method": method,
                "objective": objective,
                "model_seed": model_seed,
                "data_seed": group.data_seed,
                "start_id": row["start_id"],
                **{key: value for key, value in row.items() if key not in {"method", "start_id"}},
                "position_action_slope": float(action_slope[0]),
                "velocity_action_slope": float(action_slope[1]),
                "position_actual_action_slope": float(actual_slope[0]),
                "velocity_actual_action_slope": float(actual_slope[1]),
                "position_sensitivity_ratio": float(
                    action_slope[0] / max(abs(actual_slope[0]), 1e-8)
                ),
                "velocity_sensitivity_ratio": float(
                    action_slope[1] / max(abs(actual_slope[1]), 1e-8)
                ),
                "position_action_derivative_zero": float(derivatives[context_index, 0]),
                "velocity_action_derivative_zero": float(derivatives[context_index, 1]),
                "clipped_prediction_count": int(outside.sum()),
                "position_clipped_fraction": float(
                    np.mean((values[:, 0] < MIN_POSITION) | (values[:, 0] > MAX_POSITION))
                ),
                "velocity_clipped_fraction": float(
                    np.mean((values[:, 1] < -MAX_SPEED) | (values[:, 1] > MAX_SPEED))
                ),
            }
        )
    summary = {
        "method": method,
        "objective": objective,
        "model_seed": model_seed,
        "data_seed": group.data_seed,
        "starts": len(indices),
        "position_median_normalized_contrast_error": effect[
            "position_median_normalized_contrast_error"
        ],
        "velocity_median_normalized_contrast_error": effect[
            "velocity_median_normalized_contrast_error"
        ],
        "position_order_accuracy": effect["position_order_accuracy"],
        "velocity_order_accuracy": effect["velocity_order_accuracy"],
        "position_valid_starts": effect["position_valid_starts"],
        "velocity_valid_starts": effect["velocity_valid_starts"],
        "unbounded_prediction_fraction": clipped_predictions / max(1, total_predictions),
        "position_clipped_fraction": float(
            np.mean((predictions[..., 0] < MIN_POSITION) | (predictions[..., 0] > MAX_POSITION))
        ),
        "velocity_clipped_fraction": float(
            np.mean((predictions[..., 1] < -MAX_SPEED) | (predictions[..., 1] > MAX_SPEED))
        ),
        "mean_normalized_contrast_error": float(
            np.mean(
                [
                    effect["position_median_normalized_contrast_error"],
                    effect["velocity_median_normalized_contrast_error"],
                ]
            )
        ),
    }
    summary_rows = [
        {**row, "unbounded_prediction_fraction": summary["unbounded_prediction_fraction"]}
        for row in summary_rows
    ]
    return raw_rows, summary_rows, summary


def _baseline_action_predictions(
    method: str,
    group: PreparedGroup,
    indices: np.ndarray,
    ridge: Mapping[str, object] | None,
) -> np.ndarray:
    states = group.handcrafted_states[indices]
    if method == METHOD_HANDCRAFTED_EXACT:
        return mountaincar_step(states, group.data.actions[indices])
    if method == METHOD_RIDGE:
        if ridge is None:
            raise ValueError("ridge model is required for handcrafted-feature predictions")
        return predict_handcrafted_ridge(
            _feature_matrix(states, group.data.actions[indices]), ridge
        )
    raise ValueError(f"unknown handcrafted baseline: {method}")


def _baseline_triplets(
    method: str,
    group: PreparedGroup,
    ridge: Mapping[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = group.intervention_indices
    actions = np.tile(np.asarray(ACTION_VALUES, dtype=np.float32), len(indices))
    states = np.repeat(group.handcrafted_states[indices], len(ACTION_VALUES), axis=0)
    if method == METHOD_HANDCRAFTED_EXACT:
        predictions = mountaincar_step(states, actions)
        epsilon = 1e-3
        plus = mountaincar_step(states[::3], np.full(len(indices), epsilon))
        minus = mountaincar_step(states[::3], np.full(len(indices), -epsilon))
        derivatives = ((plus - minus) / (2.0 * epsilon)).astype(np.float32)
        return predictions.reshape(len(indices), 3, 2), derivatives
    if ridge is None:
        raise ValueError("ridge model is required for handcrafted-feature predictions")
    features = _feature_matrix(states, actions)
    predictions = predict_handcrafted_ridge(features, ridge)
    coefficient = np.asarray(ridge["coefficient"], dtype=np.float64)
    feature_scale = np.asarray(ridge["feature_scale"], dtype=np.float64)
    derivatives = np.tile(coefficient[2] / feature_scale[2], (len(indices), 1))
    return predictions.reshape(len(indices), 3, 2), derivatives.astype(np.float32)


def _linear_probe_record(
    method: str,
    model_seed: int,
    group: PreparedGroup,
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
) -> dict[str, object]:
    train_target = group.data.states[group.train_indices]
    validation_target = group.data.states[group.validation_indices]
    probe = fit_stable_probe(
        train_embeddings,
        train_target,
        validation_embeddings,
        validation_target,
    )
    prediction = apply_probe_numpy(validation_embeddings, probe)
    metrics = probe_metrics(prediction, validation_target)
    return {
        "protocol_version": V3_VERSION,
        "method": method,
        "model_seed": model_seed,
        "data_seed": group.data_seed,
        "probe_alpha": float(probe["alpha"]),
        "probe_validation_mse": float(probe["validation_mse"]),
        "probe_gram_condition": float(probe["gram_condition"]),
        "probe_standardized_coef_l2": float(probe["standardized_coef_l2"]),
        "probe_standardized_coef_max_abs": float(probe["standardized_coef_max_abs"]),
        "probe_raw_coef_l2": float(probe["raw_coef_l2"]),
        "probe_raw_coef_max_abs": float(probe["raw_coef_max_abs"]),
        "probe_position_r2": metrics["position_r2"],
        "probe_velocity_r2": metrics["velocity_r2"],
        "probe_position_mae": metrics["position_mae"],
        "probe_velocity_mae": metrics["velocity_mae"],
    }


def _make_not_run_row(method: str, model_seed: int, data_seed: int, reason: str) -> dict[str, object]:
    return {
        "protocol_version": V3_VERSION,
        "stage": "validation",
        "method": method,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "status": "not_run",
        "finite": False,
        "validation_pass": False,
        "error": reason,
    }


def _exact_dynamics_run_row(group: PreparedGroup) -> dict[str, object]:
    """Record the handcrafted-state exact-dynamics arm without fitting a model."""
    indices = group.validation_indices
    prediction = mountaincar_step(
        group.handcrafted_states[indices], group.data.actions[indices]
    )
    finite = bool(np.isfinite(prediction).all())
    row: dict[str, object] = {
        "protocol_version": V3_VERSION,
        "stage": "validation",
        "method": METHOD_HANDCRAFTED_EXACT,
        "objective": "exact_dynamics_from_visual_state",
        "model_seed": 0,
        "data_seed": group.data_seed,
        "status": "complete" if finite else "failed",
        "epochs": 0,
        "train_rows": int(len(group.train_indices)),
        "validation_rows": int(len(indices)),
        "finite": finite,
        **_coordinate_metrics(prediction, group.data.next_states[indices]),
    }
    if not finite:
        row["error"] = "exact dynamics produced non-finite validation predictions"
    return row


def _data_path_failure_model_runs(
    groups: Sequence[PreparedGroup], reason: str
) -> list[dict[str, object]]:
    """Retain every planned arm when a data-path gate stops the study."""
    rows = [_exact_dynamics_run_row(group) for group in groups]
    rows.extend(
        _make_not_run_row(METHOD_RIDGE, 0, group.data_seed, reason)
        for group in groups
    )
    rows.extend(
        _make_not_run_row(method, model_seed, group.data_seed, reason)
        for group, model_seed in zip(groups, MODEL_SEEDS, strict=True)
        for method in NEURAL_METHODS
    )
    validate_method_run_coverage(rows)
    return rows


def _method_gate_record(
    run: Mapping[str, object],
    action: Mapping[str, object],
    probe: Mapping[str, object],
) -> dict[str, object]:
    record = {
        **run,
        "position_median_normalized_contrast_error": action.get(
            "position_median_normalized_contrast_error"
        ),
        "velocity_median_normalized_contrast_error": action.get(
            "velocity_median_normalized_contrast_error"
        ),
        "position_order_accuracy": action.get("position_order_accuracy"),
        "velocity_order_accuracy": action.get("velocity_order_accuracy"),
        "position_valid_starts": action.get("position_valid_starts"),
        "velocity_valid_starts": action.get("velocity_valid_starts"),
        "unbounded_prediction_fraction": action.get("unbounded_prediction_fraction"),
        "position_clipped_fraction": action.get("position_clipped_fraction"),
        "velocity_clipped_fraction": action.get("velocity_clipped_fraction"),
        "low_variance_dims": run.get("low_variance_dims"),
        "effective_rank": run.get("effective_rank"),
        "position_probe_r2": probe.get("probe_position_r2"),
        "velocity_probe_r2": probe.get("probe_velocity_r2"),
    }
    gate = dynamics_seed_gate(record)
    record.update({f"gate_{key}": value for key, value in gate.items()})
    record["validation_pass"] = bool(
        gate["validation_pass"] and str(run.get("status")) == "complete"
    )
    record["action_fidelity_pass"] = bool(gate["action_fidelity_pass"])
    record["mean_normalized_contrast_error"] = action.get(
        "mean_normalized_contrast_error"
    )
    position_mae = run.get("position_mae")
    velocity_mae = run.get("velocity_mae")
    if position_mae is not None and velocity_mae is not None:
        record["normalized_state_error"] = float(
            (float(position_mae) / 0.01 + float(velocity_mae) / 0.002) / 2.0
        )
    else:
        record["normalized_state_error"] = None
    return record


def _write_manifest(evidence_dir: Path) -> dict[str, object]:
    files = {
        str(path.relative_to(evidence_dir)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(evidence_dir.rglob("*"))
        if path.is_file() and path.name != "sha256-manifest.json"
    }
    manifest = {"protocol_version": V3_VERSION, "files": files}
    write_json(evidence_dir / "sha256-manifest.json", manifest)
    return manifest


def _prepare_baselines(
    group: PreparedGroup,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Fit the fixed ridge arm and evaluate the two non-neural references."""
    ridge = fit_handcrafted_ridge(
        _feature_matrix(
            group.handcrafted_states[group.train_indices],
            group.data.actions[group.train_indices],
        ),
        group.data.next_states[group.train_indices],
    )
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    triplets: list[dict[str, object]] = []
    action_summaries: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for method in (METHOD_HANDCRAFTED_EXACT, METHOD_RIDGE):
        if method == METHOD_HANDCRAFTED_EXACT:
            prediction = mountaincar_step(
                group.handcrafted_states[group.validation_indices],
                group.data.actions[group.validation_indices],
            )
        else:
            prediction = _baseline_action_predictions(
                method, group, group.validation_indices, ridge
            )
        target = group.data.next_states[group.validation_indices]
        metrics = _coordinate_metrics(prediction, target)
        rows.append(
            {
                "protocol_version": V3_VERSION,
                "stage": "validation",
                "method": method,
                "objective": "exact_dynamics_from_visual_state" if method == METHOD_HANDCRAFTED_EXACT else "ridge_alpha_1_train_standardized_features",
                "model_seed": 0,
                "data_seed": group.data_seed,
                "status": "complete",
                "epochs": 0,
                "train_rows": int(len(group.train_indices)),
                "validation_rows": int(len(group.validation_indices)),
                "finite": bool(np.isfinite(prediction).all()),
                **metrics,
            }
        )
        for row_index, source_index in enumerate(group.validation_indices):
            pred = prediction[row_index]
            actual = target[row_index]
            predictions.append(
                {
                    "protocol_version": V3_VERSION,
                    "stage": "validation",
                    "method": method,
                    "model_seed": 0,
                    "data_seed": group.data_seed,
                    "source_row": int(source_index),
                    "prediction_target": "next_state",
                    "actual_position": float(actual[0]),
                    "actual_velocity": float(actual[1]),
                    "predicted_position_raw": float(pred[0]),
                    "predicted_velocity_raw": float(pred[1]),
                    "position_error_raw": float(pred[0] - actual[0]),
                    "velocity_error_raw": float(pred[1] - actual[1]),
                }
            )
        action_prediction, derivative = _baseline_triplets(method, group, ridge)
        raw_rows, summary_rows, action_summary = _build_action_records(
            method=method,
            objective=str(rows[-1]["objective"]),
            model_seed=0,
            group=group,
            predictions=action_prediction,
            derivatives=derivative,
        )
        triplets.extend(raw_rows)
        action_summaries.extend(summary_rows)
        gates.append(
            {
                **action_summary,
                "full_validation_position_mae": metrics["position_mae"],
                "full_validation_velocity_mae": metrics["velocity_mae"],
            }
        )
    return ridge, rows, predictions, triplets, action_summaries, gates


def _data_path_pass(group: PreparedGroup, action: Mapping[str, object]) -> bool:
    return bool(
        group.exact_label_max_error <= 1e-6
        and group.handcrafted_state_metrics["position_mae"] <= 0.005
        and group.handcrafted_state_metrics["velocity_mae"] <= 0.0002
        and float(action["position_median_normalized_contrast_error"]) <= 0.01
        and float(action["velocity_median_normalized_contrast_error"]) <= 0.01
        and float(action["position_order_accuracy"]) >= 0.95
        and float(action["velocity_order_accuracy"]) >= 0.95
        and int(action["position_valid_starts"]) >= 18
        and int(action["velocity_valid_starts"]) >= 18
        and float(action["unbounded_prediction_fraction"]) <= MAX_CLIPPED_FRACTION
    )


def run_validation(output_dir: Path, *, prefer_cuda: bool = True) -> dict[str, object]:
    """Run the frozen validation stage. This function never creates test data."""
    verify_protocol_lock()
    output_dir = require_e_drive(output_dir)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVIDENCE_DIR / "run-summary.json"
    existing_evidence = [
        path
        for path in EVIDENCE_DIR.iterdir()
        if path.name != LOCK_PATH.name
    ]
    if existing_evidence or summary_path.exists():
        raise FileExistsError(
            f"v3 validation evidence already exists under {EVIDENCE_DIR}; do not overwrite it"
        )
    if output_dir.exists() and any(path.is_file() for path in output_dir.rglob("*")):
        raise FileExistsError(
            f"transient v3 output already exists under {output_dir}; do not overwrite it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    check_storage_budget(output_dir)
    stage_started = time.perf_counter()
    groups = [_prepare_group(seed) for seed in DATA_SEEDS]
    target_scales = [row for group in groups for row in group.target_scale_rows]
    partition_rows: list[dict[str, object]] = []
    for group in groups:
        train_index_set = set(map(int, group.train_indices))
        partition_rows.extend(
            {
                "protocol_version": V3_VERSION,
                "data_seed": group.data_seed,
                "source_row": row_index,
                "episode_id": int(group.data.episode_ids[row_index]),
                "partition": "train" if row_index in train_index_set else "validation",
            }
            for row_index in range(len(group.data.episode_ids))
        )
    data_path_rows: list[dict[str, object]] = []
    data_path_action_summaries: list[dict[str, object]] = []
    data_triplets: list[dict[str, object]] = []
    for group in groups:
        exact_triplets, exact_derivatives = _baseline_triplets(
            METHOD_HANDCRAFTED_EXACT, group, None
        )
        raw_rows, summary_rows, action = _build_action_records(
            method=METHOD_HANDCRAFTED_EXACT,
            objective="exact_dynamics_from_visual_state",
            model_seed=0,
            group=group,
            predictions=exact_triplets,
            derivatives=exact_derivatives,
        )
        data_triplets.extend(raw_rows)
        data_path_action_summaries.extend(summary_rows)
        data_path_rows.append(
            {
                "protocol_version": V3_VERSION,
                "data_seed": group.data_seed,
                "train_episodes": int(len(np.unique(group.data.episode_ids[group.train_indices]))),
                "validation_episodes": int(len(np.unique(group.data.episode_ids[group.validation_indices]))),
                "train_rows": int(len(group.train_indices)),
                "validation_rows": int(len(group.validation_indices)),
                "validation_contexts": int(len(group.intervention_indices)),
                "frame_height": 48,
                "frame_width": 128,
                "frame_channels_per_timestep": 3,
                "input_frame_order": "previous_then_current",
                "input_pixel_scale": "uint8_to_float32_divide_by_255",
                "frame_resampling": "Pillow_bilinear",
                "exact_label_max_error": group.exact_label_max_error,
                "handcrafted_position_mae": group.handcrafted_state_metrics["position_mae"],
                "handcrafted_velocity_mae": group.handcrafted_state_metrics["velocity_mae"],
                "action_position_contrast_error": action["position_median_normalized_contrast_error"],
                "action_velocity_contrast_error": action["velocity_median_normalized_contrast_error"],
                "action_position_order_accuracy": action["position_order_accuracy"],
                "action_velocity_order_accuracy": action["velocity_order_accuracy"],
                "clipped_fraction": action["unbounded_prediction_fraction"],
                "position_clipped_fraction": action["position_clipped_fraction"],
                "velocity_clipped_fraction": action["velocity_clipped_fraction"],
                "passed": _data_path_pass(group, action),
            }
        )
    data_path_pass = all(bool(row["passed"]) for row in data_path_rows)
    write_csv(EVIDENCE_DIR / "target-scales.csv", target_scales)
    write_csv(EVIDENCE_DIR / "partition-map.csv", partition_rows)
    write_csv(EVIDENCE_DIR / "data-path.csv", data_path_rows)
    write_csv(EVIDENCE_DIR / "data-path-action-summary.csv", data_path_action_summaries)
    write_csv(EVIDENCE_DIR / "handcrafted-action-triplets.csv", data_triplets)

    model_runs: list[dict[str, object]] = []
    validation_predictions: list[dict[str, object]] = []
    action_triplets: list[dict[str, object]] = []
    action_summaries: list[dict[str, object]] = []
    learning_curves: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    ridge_models: dict[str, object] = {}
    validation_gate_rows: list[dict[str, object]] = []
    if not data_path_pass:
        stop_reason = "not run because a frozen data-path positive control failed"
        model_runs = _data_path_failure_model_runs(groups, stop_reason)
        run_summary = {
            "protocol_version": V3_VERSION,
            "stage": "validation",
            "status": "data_path_failed",
            "data_path_pass": False,
            "records_complete": False,
            "heldout_open": False,
            "reason": "one or more frozen data-path positive controls failed",
            "fit_count": 0,
            "expected_fit_count": len(NEURAL_METHODS) * len(DATA_SEEDS),
            "compute_seconds": time.perf_counter() - stage_started,
        }
        write_csv(EVIDENCE_DIR / "model-runs.csv", model_runs)
        write_csv(EVIDENCE_DIR / "validation-predictions.csv", validation_predictions)
        write_json(summary_path, run_summary)
        _write_manifest(EVIDENCE_DIR)
        return run_summary

    device = select_device(prefer_cuda=prefer_cuda)
    baseline_action_rows: list[dict[str, object]] = []
    baseline_prediction_rows: list[dict[str, object]] = []
    for group in groups:
        ridge, rows, predictions, triplets, summaries, _gates = _prepare_baselines(group)
        ridge_models[str(group.data_seed)] = ridge
        model_runs.extend(rows)
        validation_predictions.extend(predictions)
        baseline_action_rows.extend(triplets)
        action_summaries.extend(summaries)
    write_json(EVIDENCE_DIR / "ridge-models.json", ridge_models)
    validation_predictions.extend(baseline_prediction_rows)
    action_triplets.extend(baseline_action_rows)

    schedule = [
        (method, group, model_seed)
        for group, model_seed in zip(groups, MODEL_SEEDS, strict=True)
        for method in NEURAL_METHODS
    ]
    gpu_seconds_used = 0.0
    fit_count = 0
    failed_reason: str | None = None
    for schedule_index, (method, group, model_seed) in enumerate(schedule):
        if failed_reason:
            model_runs.append(
                _make_not_run_row(method, model_seed, group.data_seed, failed_reason)
            )
            continue
        try:
            run, history, gradients, model, target_mean, target_scale, gpu_seconds_used = _train_one(
                method,
                group,
                model_seed,
                device,
                output_dir,
                gpu_seconds_used,
            )
            fit_count += 1
            learning_curves.extend(history)
            gradient_rows.extend(gradients)
            model_runs.append(run)
            if run["status"] != "complete":
                failed_reason = str(run["error"] or "fit did not complete")
                continue

            train_prediction, train_embeddings, _, _ = _evaluate_visual_model(
                model,
                method,
                group,
                group.train_indices,
                device,
                target_mean,
                target_scale,
            )
            prediction, validation_embeddings, _, _ = _evaluate_visual_model(
                model,
                method,
                group,
                group.validation_indices,
                device,
                target_mean,
                target_scale,
            )
            target_state = (
                group.data.states[group.validation_indices]
                if method == METHOD_STATE_ESTIMATOR
                else group.data.next_states[group.validation_indices]
            )
            for row_index, source_index in enumerate(group.validation_indices):
                pred = prediction[row_index]
                actual = target_state[row_index]
                validation_predictions.append(
                    {
                        "protocol_version": V3_VERSION,
                        "stage": "validation",
                        "method": method,
                        "model_seed": model_seed,
                        "data_seed": group.data_seed,
                        "source_row": int(source_index),
                        "prediction_target": "current_state" if method == METHOD_STATE_ESTIMATOR else "next_state",
                        "actual_position": float(actual[0]),
                        "actual_velocity": float(actual[1]),
                        "predicted_position_raw": float(pred[0]),
                        "predicted_velocity_raw": float(pred[1]),
                        "position_error_raw": float(pred[0] - actual[0]),
                        "velocity_error_raw": float(pred[1] - actual[1]),
                    }
                )
            probe_rows.append(
                _linear_probe_record(
                    method,
                    model_seed,
                    group,
                    train_embeddings,
                    validation_embeddings,
                )
            )
            for dimension in range(validation_embeddings.shape[1]):
                values = validation_embeddings[:, dimension].astype(np.float64)
                embedding_rows.append(
                    {
                        "protocol_version": V3_VERSION,
                        "method": method,
                        "model_seed": model_seed,
                        "data_seed": group.data_seed,
                        "dimension": dimension,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "variance": float(values.var(ddof=0)),
                        "low_spread": bool(values.std(ddof=0) < 1e-3),
                        "rows": int(len(values)),
                    }
                )
            if method != METHOD_STATE_ESTIMATOR:
                predictions_by_action, derivatives = _predict_neural_triplets(
                    model,
                    method,
                    group,
                    device,
                    target_mean,
                    target_scale,
                )
                raw_rows, summary_rows, action = _build_action_records(
                    method=method,
                    objective=_model_objective(method),
                    model_seed=model_seed,
                    group=group,
                    predictions=predictions_by_action,
                    derivatives=derivatives,
                )
                action_triplets.extend(raw_rows)
                action_summaries.extend(summary_rows)
                if method in (*DYNAMICS_CANDIDATES, METHOD_ACTION_SHUFFLE):
                    validation_gate_rows.append(
                        _method_gate_record(run, action, probe_rows[-1])
                    )
        except Exception as exc:
            failed_reason = f"{type(exc).__name__}: {exc}"
            existing_row = next(
                (
                    row
                    for row in model_runs
                    if row.get("method") == method
                    and row.get("model_seed") == model_seed
                    and row.get("data_seed") == group.data_seed
                ),
                None,
            )
            if existing_row is None:
                model_runs.append(
                    {
                        **_make_not_run_row(method, model_seed, group.data_seed, failed_reason),
                        "status": "failed",
                        "error": failed_reason,
                    }
                )
            else:
                existing_row.update(
                    {"status": "failed", "finite": False, "validation_pass": False, "error": failed_reason}
                )
            for later_method, later_group, later_seed in schedule[schedule_index + 1 :]:
                model_runs.append(
                    _make_not_run_row(later_method, later_seed, later_group.data_seed, failed_reason)
                )
            break

    # Fit the ridge records above after the data gate, so no learned work starts early.
    try:
        validate_method_run_coverage(model_runs)
        coverage_pass = True
    except (KeyError, ValueError):
        coverage_pass = False
    all_fits_complete = coverage_pass and all(
        row.get("status") == "complete" for row in model_runs
    )
    required_gate_methods = [
        row for row in validation_gate_rows if row.get("method") in (*DYNAMICS_CANDIDATES, METHOD_ACTION_SHUFFLE)
    ]
    gates_complete = len(required_gate_methods) == len((*DYNAMICS_CANDIDATES, METHOD_ACTION_SHUFFLE)) * len(DATA_SEEDS)
    decision = choose_candidate_method(
        required_gate_methods,
        data_path_pass=data_path_pass,
        records_complete=all_fits_complete and gates_complete,
    )
    run_summary = {
        "protocol_version": V3_VERSION,
        "stage": "validation",
        "status": "complete" if all_fits_complete and gates_complete else "incomplete",
        "data_path_pass": data_path_pass,
        "records_complete": bool(all_fits_complete and gates_complete),
        "fit_count": fit_count,
        "expected_fit_count": len(NEURAL_METHODS) * len(DATA_SEEDS),
        "device": device.type,
        "device_name": _device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "compute_seconds": time.perf_counter() - stage_started,
        "training_seconds": gpu_seconds_used,
        "compute_budget_seconds": GPU_BUDGET_SECONDS,
        "cuda_memory_budget_mib": GPU_MEMORY_LIMIT_MIB,
        "selected_method": decision["selected_method"],
        "candidate_families": decision["candidate_families"],
        "action_shuffle_valid": decision["action_shuffle_valid"],
        "action_shuffle_failure_count": decision["action_shuffle_failure_count"],
        "heldout_open": can_open_heldout(decision) if all_fits_complete and gates_complete else False,
        "heldout_reason": decision["reason"] if all_fits_complete and gates_complete else failed_reason or "validation records are incomplete",
    }
    write_csv(EVIDENCE_DIR / "model-runs.csv", model_runs)
    write_csv(EVIDENCE_DIR / "validation-predictions.csv", validation_predictions)
    write_csv(EVIDENCE_DIR / "action-triplets.csv", action_triplets)
    write_csv(EVIDENCE_DIR / "action-summary.csv", action_summaries)
    write_csv(EVIDENCE_DIR / "learning-curves.csv", learning_curves)
    write_csv(EVIDENCE_DIR / "gradient-diagnostics.csv", gradient_rows)
    write_csv(EVIDENCE_DIR / "embedding-diagnostics.csv", embedding_rows)
    write_csv(EVIDENCE_DIR / "linear-probes.csv", probe_rows)
    write_csv(EVIDENCE_DIR / "validation-gates.csv", required_gate_methods)
    write_json(summary_path, run_summary)
    _write_manifest(EVIDENCE_DIR)
    check_storage_budget(output_dir)
    return run_summary


def _verify_record_manifest(evidence_dir: Path) -> None:
    manifest_path = evidence_dir / "sha256-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("validation evidence manifest is missing; held-out stage stays closed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != V3_VERSION:
        raise StudyError("validation evidence manifest has the wrong protocol version")
    for relative_name, expected_hash in manifest.get("files", {}).items():
        path = evidence_dir / relative_name
        expected_sha256 = expected_hash.get("sha256") if isinstance(expected_hash, dict) else expected_hash
        expected_bytes = expected_hash.get("bytes") if isinstance(expected_hash, dict) else None
        if (
            not path.is_file()
            or sha256_file(path) != expected_sha256
            or (expected_bytes is not None and path.stat().st_size != int(expected_bytes))
        ):
            raise StudyError(f"validation evidence integrity check failed: {relative_name}")


def _collect_test_contexts(test_seed: int) -> list[dict[str, object]]:
    """Build one seeded random-action trajectory and retain only locked starts."""
    env = make_env()
    starts: list[dict[str, object]] = []
    rng = np.random.default_rng(
        stable_seed(V3_VERSION, "heldout-start-trajectory", int(test_seed))
    )
    try:
        observation, _ = env.reset(seed=int(test_seed))
        current_frame = resize_frame(render_frame(float(observation[0])))
        previous_frame = current_frame.copy()
        for step in range(max(TEST_STEPS) + 1):
            state = np.asarray(env.unwrapped.state, dtype=np.float32).copy()
            if step in TEST_STEPS:
                starts.append(
                    {
                        "start_id": f"test-{test_seed}-step-{step}",
                        "test_seed": int(test_seed),
                        "step": int(step),
                        "state": state,
                        "previous_frame": previous_frame.copy(),
                        "current_frame": current_frame.copy(),
                        "visual_state": visual_state(previous_frame, current_frame),
                    }
                )
            action = float(rng.uniform(-1.0, 1.0))
            next_observation, _, terminated, truncated, _ = env.step(
                np.asarray([action], dtype=np.float32)
            )
            next_frame = resize_frame(render_frame(float(next_observation[0])))
            previous_frame, current_frame = current_frame, next_frame
            if terminated or truncated:
                break
    finally:
        env.close()
    return starts


def _scheduled_test_start_records(
    test_seed: int, starts: Sequence[Mapping[str, object]]
) -> tuple[dict[int, Mapping[str, object]], list[dict[str, object]]]:
    """Record every frozen start, including starts missing after termination."""
    by_step = {int(start["step"]): start for start in starts}
    rows: list[dict[str, object]] = []
    for step in TEST_STEPS:
        row: dict[str, object] = {
            "protocol_version": V3_VERSION,
            "stage": "heldout",
            "test_seed": int(test_seed),
            "start_id": f"test-{test_seed}-step-{step}",
            "step": int(step),
            "status": "complete" if step in by_step else "missing",
            "reason": "" if step in by_step else "simulator terminated or truncated before scheduled start",
        }
        start = by_step.get(step)
        if start is not None:
            state = np.asarray(start["state"], dtype=np.float32)
            visual_state_value = np.asarray(start["visual_state"], dtype=np.float32)
            row.update(
                {
                    "actual_position": float(state[0]),
                    "actual_velocity": float(state[1]),
                    "handcrafted_position": float(visual_state_value[0]),
                    "handcrafted_velocity": float(visual_state_value[1]),
                }
            )
        rows.append(row)
    return by_step, rows


def _load_dynamics_checkpoint(
    method: str,
    model_seed: int,
    data_seed: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[SupervisedVisualDynamics, np.ndarray, np.ndarray]:
    checkpoint_path = output_dir / "checkpoints" / (
        f"{method}-model-{model_seed}-data-{data_seed}.pt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"selected validation checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("protocol_version") != V3_VERSION:
        raise StudyError("selected checkpoint protocol version does not match v3")
    if checkpoint.get("protocol_lock_sha256") != sha256_file(LOCK_PATH):
        raise StudyError("selected checkpoint does not match the frozen protocol lock")
    model = SupervisedVisualDynamics().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model,
        np.asarray(checkpoint["target_mean"], dtype=np.float32),
        np.asarray(checkpoint["target_scale"], dtype=np.float32),
    )


def _load_state_estimator_checkpoint(
    model_seed: int,
    data_seed: int,
    output_dir: Path,
    device: torch.device,
) -> VisualStateEstimator:
    checkpoint_path = output_dir / "checkpoints" / (
        f"{METHOD_STATE_ESTIMATOR}-model-{model_seed}-data-{data_seed}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("protocol_lock_sha256") != sha256_file(LOCK_PATH):
        raise StudyError("state-estimator checkpoint does not match the frozen protocol lock")
    model = VisualStateEstimator().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _clip_rollout_state(state: np.ndarray) -> tuple[np.ndarray, bool]:
    tensor = torch.as_tensor(state, dtype=torch.float32)
    clipped, mask = unclipped_state_diagnostics(tensor)
    return clipped.cpu().numpy().astype(np.float32), bool(mask.any().item())


def _rollout_prediction(
    method: str,
    *,
    state: np.ndarray,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    action: float,
    ridge: Mapping[str, object],
    model: SupervisedVisualDynamics | None,
    target_mean: np.ndarray | None,
    target_scale: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    if method == "privileged_exact_dynamics":
        return mountaincar_step(state, action)
    estimated = visual_state(previous_frame, current_frame)
    if method == METHOD_HANDCRAFTED_EXACT:
        return mountaincar_step(estimated, action)
    if method == METHOD_RIDGE:
        return predict_handcrafted_ridge(
            _feature_matrix(estimated[None, :], np.asarray([action], dtype=np.float32)), ridge
        )[0]
    if model is None or target_mean is None or target_scale is None:
        raise ValueError("visual dynamics rollout requires a loaded model and target transform")
    context = context_tensor(
        np.stack((previous_frame, current_frame))[None, ...], device=device
    )
    with torch.no_grad():
        raw = model(context, torch.tensor([action], dtype=torch.float32, device=device))
        decoded = _decode_model_output(
            method,
            raw,
            torch.as_tensor(estimated[None, :], dtype=torch.float32, device=device),
            torch.as_tensor(target_mean, dtype=torch.float32, device=device),
            torch.as_tensor(target_scale, dtype=torch.float32, device=device),
        )
    return decoded[0].cpu().numpy().astype(np.float32)


def _run_rollout(
    *,
    method: str,
    model_seed: int,
    data_seed: int,
    start: Mapping[str, object],
    actions: np.ndarray,
    ridge: Mapping[str, object],
    model: SupervisedVisualDynamics | None,
    target_mean: np.ndarray | None,
    target_scale: np.ndarray | None,
    device: torch.device,
    train_min: np.ndarray,
    train_max: np.ndarray,
) -> list[dict[str, object]]:
    actual_state = np.asarray(start["state"], dtype=np.float32).copy()
    predicted_state = actual_state.copy()
    previous_frame = np.asarray(start["previous_frame"], dtype=np.uint8).copy()
    current_frame = np.asarray(start["current_frame"], dtype=np.uint8).copy()
    rows: list[dict[str, object]] = []
    stopped_reason = ""
    for index, action in enumerate(actions, start=1):
        if stopped_reason:
            rows.append(
                {
                    "protocol_version": V3_VERSION,
                    "stage": "heldout_rollout",
                    "method": method,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "test_seed": start["test_seed"],
                    "start_id": start["start_id"],
                    "horizon": index,
                    "action": float(action),
                    "actual_position": None,
                    "actual_velocity": None,
                    "predicted_position": None,
                    "predicted_velocity": None,
                    "status": "failed",
                    "error": stopped_reason,
                }
            )
            continue
        try:
            predicted_raw = _rollout_prediction(
                method,
                state=predicted_state,
                previous_frame=previous_frame,
                current_frame=current_frame,
                action=float(action),
                ridge=ridge,
                model=model,
                target_mean=target_mean,
                target_scale=target_scale,
                device=device,
            )
            actual_next = mountaincar_step(actual_state, float(action))
            if not np.isfinite(predicted_raw).all():
                raise FloatingPointError("non-finite predicted state")
            clipped_state, clipped = _clip_rollout_state(predicted_raw)
            ood = bool(np.any(predicted_raw < train_min) or np.any(predicted_raw > train_max))
            rows.append(
                {
                    "protocol_version": V3_VERSION,
                    "stage": "heldout_rollout",
                    "method": method,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "test_seed": start["test_seed"],
                    "start_id": start["start_id"],
                    "horizon": index,
                    "action": float(action),
                    "actual_position": float(actual_next[0]),
                    "actual_velocity": float(actual_next[1]),
                    "predicted_position": float(predicted_raw[0]),
                    "predicted_velocity": float(predicted_raw[1]),
                    "position_error": float(predicted_raw[0] - actual_next[0]),
                    "velocity_error": float(predicted_raw[1] - actual_next[1]),
                    "clipped": clipped,
                    "outside_training_range": ood,
                    "status": "complete",
                    "error": "",
                }
            )
            actual_state = actual_next
            predicted_state = clipped_state
            next_frame = resize_frame(render_frame(float(clipped_state[0])))
            previous_frame, current_frame = current_frame, next_frame
            if float(actual_next[0]) >= 0.45:
                stopped_reason = "actual simulator episode terminated at the goal"
        except Exception as exc:
            stopped_reason = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "protocol_version": V3_VERSION,
                    "stage": "heldout_rollout",
                    "method": method,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "test_seed": start["test_seed"],
                    "start_id": start["start_id"],
                    "horizon": index,
                    "action": float(action),
                    "actual_position": None,
                    "actual_velocity": None,
                    "predicted_position": None,
                    "predicted_velocity": None,
                    "status": "failed",
                    "error": stopped_reason,
                }
            )
    return rows


def run_heldout(output_dir: Path, *, prefer_cuda: bool = True) -> dict[str, object]:
    """Run only after the saved validation gate and evidence integrity both pass."""
    verify_protocol_lock()
    output_dir = require_e_drive(output_dir)
    run_summary_path = EVIDENCE_DIR / "run-summary.json"
    if not run_summary_path.is_file():
        raise FileNotFoundError("validation summary is missing; held-out stage stays closed")
    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if not can_open_heldout(summary):
        raise StudyError(
            "validation gate is closed; no independent test trajectory was created"
        )
    _verify_record_manifest(EVIDENCE_DIR)
    heldout_summary_path = EVIDENCE_DIR / "heldout-summary.json"
    if heldout_summary_path.exists():
        raise FileExistsError("held-out evidence already exists; do not overwrite it")

    selected_method = str(summary["selected_method"])
    device = select_device(prefer_cuda=prefer_cuda)
    ridge_models = json.loads((EVIDENCE_DIR / "ridge-models.json").read_text(encoding="utf-8"))
    model_pairs = list(zip(DATA_SEEDS, MODEL_SEEDS, strict=True))
    # The validation gate is checked before the first call that creates test-seed data.
    test_contexts = {
        int(test_seed): _collect_test_contexts(int(test_seed)) for test_seed in TEST_SEEDS
    }
    expected_starts = len(TEST_SEEDS) * len(TEST_STEPS)
    observed_starts = sum(map(len, test_contexts.values()))
    start_rows: list[dict[str, object]] = []
    state_estimate_rows: list[dict[str, object]] = []
    rollouts: list[dict[str, object]] = []
    training_states = []
    for data_seed in DATA_SEEDS:
        train_group = _prepare_group(data_seed)
        training_states.append(train_group.data.states[train_group.train_indices].copy())
        del train_group
    train_min = np.min(np.concatenate(training_states), axis=0)
    train_max = np.max(np.concatenate(training_states), axis=0)
    del training_states

    for test_seed, starts in test_contexts.items():
        by_step, scheduled_rows = _scheduled_test_start_records(test_seed, starts)
        start_rows.extend(scheduled_rows)
        for start in starts:
            if int(start["step"]) not in by_step:
                raise StudyError("held-out start map lost a collected scheduled start")
            for data_seed, model_seed in model_pairs:
                state_model = _load_state_estimator_checkpoint(
                    model_seed, data_seed, output_dir, device
                )
                frame_context = context_tensor(
                    np.stack((start["previous_frame"], start["current_frame"]))[None, ...],
                    device=device,
                )
                with torch.no_grad():
                    estimated = state_model(frame_context)[0].cpu().numpy()
                state_estimate_rows.append(
                    {
                        "protocol_version": V3_VERSION,
                        "test_seed": test_seed,
                        "start_id": start["start_id"],
                        "data_seed": data_seed,
                        "model_seed": model_seed,
                        "actual_position": float(start["state"][0]),
                        "actual_velocity": float(start["state"][1]),
                        "predicted_position": float(estimated[0]),
                        "predicted_velocity": float(estimated[1]),
                        "position_error": float(estimated[0] - start["state"][0]),
                        "velocity_error": float(estimated[1] - start["state"][1]),
                    }
                )
                selected_model, target_mean, target_scale = _load_dynamics_checkpoint(
                    selected_method, model_seed, data_seed, output_dir, device
                )
                action_seed = stable_seed(
                    V3_VERSION,
                    int(test_seed),
                    str(start["start_id"]),
                    "heldout-rollout-actions",
                )
                actions = np.random.default_rng(action_seed).uniform(
                    -1.0, 1.0, size=ROLLOUT_HORIZON
                ).astype(np.float32)
                start_methods = (
                    ("privileged_exact_dynamics", None, None, None),
                    (METHOD_HANDCRAFTED_EXACT, None, None, None),
                    (METHOD_RIDGE, None, None, None),
                    (selected_method, selected_model, target_mean, target_scale),
                )
                ridge = ridge_models[str(data_seed)]
                for method, model, mean, scale in start_methods:
                    rollouts.extend(
                        _run_rollout(
                            method=method,
                            model_seed=model_seed,
                            data_seed=data_seed,
                            start=start,
                            actions=actions,
                            ridge=ridge,
                            model=model,
                            target_mean=mean,
                            target_scale=scale,
                            device=device,
                            train_min=train_min,
                            train_max=train_max,
                        )
                    )
    if observed_starts != expected_starts:
        heldout_status = "incomplete_missing_start_contexts"
    else:
        heldout_status = "complete"
    write_csv(EVIDENCE_DIR / "heldout-starts.csv", start_rows)
    write_csv(EVIDENCE_DIR / "heldout-state-estimates.csv", state_estimate_rows)
    write_csv(EVIDENCE_DIR / "rollout-trajectories.csv", rollouts)
    rollout_index = {
        (
            str(row["method"]),
            int(row["model_seed"]),
            int(row["data_seed"]),
            str(row["start_id"]),
            int(row["horizon"]),
        ): row
        for row in rollouts
    }
    horizon_rows = [
        rollout_index[key]
        for key in sorted(rollout_index)
        if key[4] in (1, 5, 10, 20, 60)
    ]
    write_csv(EVIDENCE_DIR / "rollout-horizons.csv", horizon_rows)
    summary_record = {
        "protocol_version": V3_VERSION,
        "stage": "heldout_rollout",
        "status": heldout_status,
        "selected_method": selected_method,
        "expected_start_contexts": expected_starts,
        "observed_start_contexts": observed_starts,
        "test_seeds": list(TEST_SEEDS),
        "model_data_pairs": [
            {"data_seed": data_seed, "model_seed": model_seed}
            for data_seed, model_seed in model_pairs
        ],
        "device": device.type,
        "device_name": _device_name(device),
        "rollout_horizon": ROLLOUT_HORIZON,
        "rollout_rows": len(rollouts),
    }
    write_json(heldout_summary_path, summary_record)
    _write_manifest(EVIDENCE_DIR)
    return summary_record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("lock", "validation", "heldout"),
        help="freeze code hashes, run validation, or run the gated held-out stage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="E: drive directory for transient data and model checkpoints",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU for validation or held-out smoke runs",
    )
    args = parser.parse_args(argv)
    try:
        if args.stage == "lock":
            result = freeze_protocol_lock()
        elif args.stage == "validation":
            result = run_validation(args.output_dir, prefer_cuda=not args.cpu)
        else:
            result = run_heldout(args.output_dir, prefer_cuda=not args.cpu)
    except (StudyError, OSError, ValueError, FloatingPointError) as exc:
        parser.exit(status=2, message=f"v3 study stopped: {exc}\n")
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
