"""Run the preregistered visual JEPA action-fidelity v2 study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_tutorial.action_conditioned_visual import (  # noqa: E402
    ActionConditionedJEPA,
    StateDynamics,
    VisualEncoder,
)
from jepa_tutorial.action_fidelity_v2 import (  # noqa: E402
    ACTION_VALUES,
    V2_VERSION,
    action_effect_metrics,
    apply_probe_numpy,
    effective_rank_diagnostics,
    fit_stable_probe,
    gradient_cosine,
    gradient_vector,
    probe_bootstrap_rmse,
    probe_metrics,
    sha256_file,
    split_episode_indices,
    stable_seed,
    tensor_norm,
    unclipped_state_diagnostics,
    validate_experiment_completeness,
    validate_rollout_records,
)
from jepa_tutorial.decision_study import (  # noqa: E402
    MAX_POSITION,
    MAX_SPEED,
    MIN_POSITION,
    mountaincar_step,
)
from jepa_tutorial.lejepa_loss import lejepa_total_loss  # noqa: E402
from jepa_tutorial.paper_losses import epps_pulley_sigreg_loss  # noqa: E402
from jepa_tutorial.sigreg import sigreg_loss  # noqa: E402
from jepa_tutorial.visreg_style import visreg_style_regularization_loss  # noqa: E402
from jepa_tutorial.visual_mountaincar import (  # noqa: E402
    TransitionData,
    VisualStart,
    collect_intervention_starts,
    collect_random_transitions,
    render_frame,
    resize_frame,
    visual_state,
)

PROTOCOL_PATH = ROOT / "docs/reproductions/protocols/visual-jepa-action-fidelity-v2.md"
PROTOCOL_AMENDMENT_PATHS = (
    ROOT / "docs/reproductions/protocols/visual-jepa-action-fidelity-v2-amendment-1.md",
    ROOT / "docs/reproductions/protocols/visual-jepa-action-fidelity-v2-amendment-2.md",
)
LOCK_PATH = ROOT / "docs/evidence/visual-jepa-action-fidelity-v2-protocol-lock.json"
EVIDENCE_DIR = ROOT / "docs/evidence"
DEFAULT_OUTPUT_DIR = Path("/mnt/e/codex-research/jepa-world-models-tutorial/visual-jepa-action-fidelity-v2")
TRAIN_DATA_SEEDS = (9301, 9302, 9303)
MODEL_SEEDS = (9401, 9402, 9403)
TEST_DATA_SEEDS = (9501, 9502, 9503, 9504, 9505)
PLANNING_SEEDS = tuple(range(8700, 8720))
OBJECTIVES = ("sigreg", "visreg_style", "variance_cov")
TRAIN_METHODS = tuple(("jepa", objective) for objective in OBJECTIVES) + (
    ("supervised_visual", "supervised_state_mse"),
    ("state_model", "privileged_state_mse"),
)
CODE_FILES = (
    "src/jepa_tutorial/action_conditioned_visual.py",
    "src/jepa_tutorial/action_fidelity_v2.py",
    "src/jepa_tutorial/decision_study.py",
    "src/jepa_tutorial/lejepa_loss.py",
    "src/jepa_tutorial/paper_losses.py",
    "src/jepa_tutorial/sigreg.py",
    "src/jepa_tutorial/visreg_style.py",
    "src/jepa_tutorial/visual_mountaincar.py",
    "scripts/run_visual_jepa_action_fidelity_v2.py",
    "scripts/validate_visual_jepa_action_fidelity_v2.py",
    "scripts/analyze_visual_jepa_action_fidelity_v2.py",
    "scripts/run_visual_jepa_decision_study.py",
)
CHECKPOINT_DIRNAME = "checkpoints"


class SupervisedVisualDynamics(nn.Module):
    """Predict simulator position and velocity from the same two rendered frames."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = VisualEncoder(embedding_dim=32)
        self.predictor = nn.Sequential(
            nn.Linear(33, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if actions.shape != (observations.shape[0], 1):
            raise ValueError("actions must have shape [batch] or [batch, 1]")
        embedding = self.encoder(observations)
        return self.predictor(torch.cat((embedding, actions.to(embedding.dtype)), dim=-1))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = tuple(dict.fromkeys(key for row in rows for key in row.keys())) or ("status",)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    clean[key] = json.dumps(_json_safe(value), sort_keys=True, allow_nan=False)
                elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    clean[key] = ""
                else:
                    clean[key] = _json_safe(value)
            writer.writerow(clean)
    temporary.replace(path)


def _protocol_code_hash() -> tuple[str, list[dict[str, str]]]:
    files = []
    for relative in CODE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"locked source file is missing: {relative}")
        files.append({"path": relative, "sha256": sha256_file(str(path))})
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), files


def create_protocol_lock() -> dict[str, object]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"frozen preregistration is missing: {PROTOCOL_PATH}")
    code_hash, files = _protocol_code_hash()
    protocol_files = [PROTOCOL_PATH, *PROTOCOL_AMENDMENT_PATHS]
    protocol_hashes = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(str(path))}
        for path in protocol_files
    ]
    combined_protocol_hash = hashlib.sha256(
        json.dumps(protocol_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    lock = {
        "schema_version": 1,
        "experiment_version": V2_VERSION,
        "status": "locked",
        "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocol_hash": combined_protocol_hash,
        "protocol_files": protocol_hashes,
        "code_hash": code_hash,
        "code_files": files,
        "training_data_seeds": TRAIN_DATA_SEEDS,
        "model_seeds": MODEL_SEEDS,
        "test_data_seeds": TEST_DATA_SEEDS,
        "planning_seeds": PLANNING_SEEDS,
        "compute_budget_gpu_minutes": 90,
        "memory_budget_mib": 2048,
        "storage_budget_bytes": 5 * 1024**3,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if LOCK_PATH.exists():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") == lock["protocol_hash"] and existing.get("code_hash") == lock["code_hash"]:
            return existing
        raise FileExistsError(f"protocol lock already exists and differs: {LOCK_PATH}")
    write_json(LOCK_PATH, lock)
    return lock


def verify_protocol_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(f"run protocol-lock first: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    code_hash, code_files = _protocol_code_hash()
    protocol_hashes = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(str(path))}
        for path in [PROTOCOL_PATH, *PROTOCOL_AMENDMENT_PATHS]
    ]
    combined_protocol_hash = hashlib.sha256(
        json.dumps(protocol_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        lock.get("schema_version") != 1
        or lock.get("experiment_version") != V2_VERSION
        or lock.get("status") != "locked"
        or lock.get("protocol_hash") != combined_protocol_hash
        or lock.get("protocol_files") != protocol_hashes
        or lock.get("code_hash") != code_hash
        or lock.get("code_files") != code_files
        or lock.get("training_data_seeds") != list(TRAIN_DATA_SEEDS)
        or lock.get("model_seeds") != list(MODEL_SEEDS)
        or lock.get("test_data_seeds") != list(TEST_DATA_SEEDS)
        or lock.get("planning_seeds") != list(PLANNING_SEEDS)
        or lock.get("compute_budget_gpu_minutes") != 90
        or lock.get("memory_budget_mib") != 2048
        or lock.get("storage_budget_bytes") != 5 * 1024**3
    ):
        raise ValueError("protocol or implementation hash does not match the frozen lock")
    return lock


def require_drive_output(path: Path, *, allow_local: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not allow_local and not (str(resolved).startswith("/mnt/e/") or str(resolved).startswith("/mnt/d/")):
        raise ValueError("research outputs must be stored on E: or D:")
    return resolved


def select_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        probe = torch.ones((2, 2), device=device)
        torch.cuda.synchronize(device)
        if float((probe @ probe).sum().item()) != 8.0:
            raise RuntimeError("CUDA verification operation failed")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return device


def context_tensor(data: TransitionData, indices: np.ndarray, device: torch.device, *, target: bool = False) -> torch.Tensor:
    offset = 1 if target else 0
    frames = data.frames[indices]
    pair = np.concatenate((frames[:, offset], frames[:, offset + 1]), axis=-1).transpose(0, 3, 1, 2).copy()
    return torch.from_numpy(pair).to(device=device, dtype=torch.float32).div_(255.0)


def training_batch(data: TransitionData, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        context_tensor(data, indices, device),
        context_tensor(data, indices, device, target=True),
        torch.as_tensor(data.actions[indices], device=device, dtype=torch.float32),
        torch.as_tensor(data.states[indices], device=device, dtype=torch.float32),
        torch.as_tensor(data.next_states[indices], device=device, dtype=torch.float32),
    )


def build_model(method: str) -> nn.Module:
    if method == "jepa":
        return ActionConditionedJEPA(embedding_dim=32)
    if method == "supervised_visual":
        return SupervisedVisualDynamics()
    if method == "state_model":
        return StateDynamics(hidden_dim=64)
    raise ValueError(f"unknown model method: {method}")


def variance_covariance_components(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    variance = centered.var(dim=0, unbiased=False)
    variance_loss = (variance - 1.0).square().mean()
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diagonal.square().mean()
    return variance_loss, covariance_loss


def compute_loss(
    model: nn.Module,
    method: str,
    objective: str,
    batch: tuple[torch.Tensor, ...],
    regularizer_generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context, target, actions, states, next_states = batch
    if method == "state_model":
        prediction = model(states, actions)
        state_loss = F.mse_loss(prediction, next_states)
        return state_loss, {"state_loss": state_loss, "prediction_loss": state_loss}
    if method == "supervised_visual":
        prediction = model(context, actions)
        state_loss = F.mse_loss(prediction, next_states)
        return state_loss, {"state_loss": state_loss, "prediction_loss": state_loss}

    context_embedding = model.encode(context)
    predicted_embedding = model.predictor(context_embedding, actions)
    target_embedding = model.encode(target)
    prediction_loss = F.mse_loss(predicted_embedding, target_embedding)
    paired = torch.cat((context_embedding, target_embedding), dim=0)
    if objective == "sigreg":
        regularizer = epps_pulley_sigreg_loss(
            paired,
            num_projections=128,
            num_points=17,
            generator=regularizer_generator,
        )
        total = lejepa_total_loss(predicted_embedding, target_embedding, regularizer, lambda_sigreg=0.1)
        return total, {"prediction_loss": prediction_loss, "regularizer_loss": regularizer, "sigreg": regularizer}
    if objective == "visreg_style":
        regularizer, components = visreg_style_regularization_loss(
            paired,
            num_projections=128,
            scale_weight=1.0,
            shape_weight=1.0,
            center_weight=1.0,
            epsilon=1e-4,
            generator=regularizer_generator,
        )
        total = 0.4 * prediction_loss + 0.6 * regularizer
        return total, {"prediction_loss": prediction_loss, "regularizer_loss": regularizer, **components}
    if objective == "variance_cov":
        regularizer = sigreg_loss(paired, target_var=1.0, lambda_var=1.0, lambda_cov=0.05)
        variance_loss, covariance_loss = variance_covariance_components(paired)
        total = prediction_loss + 0.1 * regularizer
        return total, {
            "prediction_loss": prediction_loss,
            "regularizer_loss": regularizer,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
        }
    raise ValueError(f"unknown JEPA objective: {objective}")


def component_gradient_diagnostics(
    model: nn.Module,
    method: str,
    objective: str,
    batch: tuple[torch.Tensor, ...],
    generator: torch.Generator,
    *,
    epoch: int,
    model_seed: int,
    data_seed: int,
) -> dict[str, object]:
    model.zero_grad(set_to_none=True)
    loss, components = compute_loss(model, method, objective, batch, generator)
    prediction = components["prediction_loss"]
    regularizer = components.get("regularizer_loss")
    encoder_parameters = list(model.encoder.parameters()) if hasattr(model, "encoder") else []
    predictor_parameters = (
        list(model.predictor.parameters())
        if hasattr(model, "predictor")
        else list(model.network.parameters())
    )
    watched = encoder_parameters + predictor_parameters
    prediction_gradients = torch.autograd.grad(
        prediction, watched, retain_graph=True, allow_unused=True
    )
    if regularizer is None:
        regularizer_gradients: tuple[torch.Tensor | None, ...] = tuple(None for _ in watched)
    else:
        regularizer_gradients = torch.autograd.grad(
            regularizer, watched, retain_graph=True, allow_unused=True
        )
    total_gradients = torch.autograd.grad(loss, watched, allow_unused=True)
    encoder_count = len(encoder_parameters)
    prediction_encoder = gradient_vector(prediction_gradients[:encoder_count])
    regularizer_encoder = gradient_vector(regularizer_gradients[:encoder_count])
    prediction_predictor = gradient_vector(prediction_gradients[encoder_count:])
    regularizer_predictor = gradient_vector(regularizer_gradients[encoder_count:])
    total_encoder = gradient_vector(total_gradients[:encoder_count])
    total_predictor = gradient_vector(total_gradients[encoder_count:])

    def norm(vector: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(vector).cpu().item()) if vector.numel() else 0.0

    return {
        "method": method,
        "objective": objective,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "epoch": epoch,
        "total_loss": float(loss.detach().cpu().item()),
        "prediction_loss": float(prediction.detach().cpu().item()),
        "regularizer_loss": float(regularizer.detach().cpu().item()) if regularizer is not None else 0.0,
        "encoder_grad_prediction": norm(prediction_encoder),
        "encoder_grad_regularizer": norm(regularizer_encoder),
        "encoder_grad_total": norm(total_encoder),
        "predictor_grad_prediction": norm(prediction_predictor),
        "predictor_grad_regularizer": norm(regularizer_predictor),
        "predictor_grad_total": norm(total_predictor),
        "encoder_grad_cosine": gradient_cosine(prediction_encoder, regularizer_encoder)
        if prediction_encoder.numel() == regularizer_encoder.numel() and prediction_encoder.numel()
        else 0.0,
    }


def _model_run_batch_metrics(
    model: nn.Module,
    method: str,
    objective: str,
    data: TransitionData,
    indices: np.ndarray,
    device: torch.device,
    generator: torch.Generator,
    *,
    batch_size: int,
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            batch = training_batch(data, indices[offset : offset + batch_size], device)
            loss, components = compute_loss(model, method, objective, batch, generator)
            values["total_loss"].append(float(loss.detach().cpu().item()))
            for name, value in components.items():
                values[name].append(float(value.detach().cpu().item()))
    return {name: float(np.mean(items)) for name, items in values.items()}


def collect_embeddings(
    model: nn.Module,
    data: TransitionData,
    indices: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
    target: bool = False,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            selected = indices[offset : offset + batch_size]
            embedding = model.encoder(context_tensor(data, selected, device, target=target))
            outputs.append(embedding.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_one(
    method: str,
    objective: str,
    data: TransitionData,
    model_seed: int,
    data_seed: int,
    device: torch.device,
    output_dir: Path,
    *,
    epochs: int = 35,
    batch_size: int = 256,
    train_episode_split: int = 16,
    stage: str = "validation",
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(method).to(device)
    train_indices, validation_indices = split_episode_indices(data.episode_ids, train_episode_split=train_episode_split)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    order_rng = np.random.default_rng(model_seed)
    regularizer_generator = torch.Generator(device=device).manual_seed(model_seed + 1)
    diagnostic_generator = torch.Generator(device=device).manual_seed(model_seed + 10_000)
    diagnostic_indices = train_indices[: min(batch_size, len(train_indices))]
    histories: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        permutation = order_rng.permutation(train_indices)
        train_components: dict[str, list[float]] = defaultdict(list)
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            batch = training_batch(data, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = compute_loss(model, method, objective, batch, regularizer_generator)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite {method}/{objective} loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            train_components["total_loss"].append(float(loss.detach().cpu().item()))
            for name, value in components.items():
                train_components[name].append(float(value.detach().cpu().item()))
        if epoch == 0 or (epoch + 1) % 5 == 0:
            diagnostic_batch = training_batch(data, diagnostic_indices, device)
            gradient_rows.append(
                component_gradient_diagnostics(
                    model,
                    method,
                    objective,
                    diagnostic_batch,
                    diagnostic_generator,
                    epoch=epoch + 1,
                    model_seed=model_seed,
                    data_seed=data_seed,
                )
            )
        validation_generator = torch.Generator(device=device).manual_seed(model_seed + 2 + epoch)
        validation_metrics = _model_run_batch_metrics(
            model,
            method,
            objective,
            data,
            validation_indices,
            device,
            validation_generator,
            batch_size=batch_size,
        )
        history = {
            "method": method,
            "objective": objective,
            "model_seed": model_seed,
            "data_seed": data_seed,
            "epoch": epoch + 1,
            "train_total_loss": float(np.mean(train_components["total_loss"])),
            "train_prediction_loss": float(np.mean(train_components["prediction_loss"])),
            "train_regularizer_loss": float(np.mean(train_components.get("regularizer_loss", [0.0]))),
            "validation_total_loss": validation_metrics.get("total_loss", 0.0),
            "validation_prediction_loss": validation_metrics.get("prediction_loss", 0.0),
            "validation_regularizer_loss": validation_metrics.get("regularizer_loss", 0.0),
        }
        for component in ("scale_loss", "shape_loss", "center_loss", "variance_loss", "covariance_loss"):
            history[f"train_{component}"] = float(np.mean(train_components[component])) if component in train_components else ""
            history[f"validation_{component}"] = validation_metrics.get(component, "")
        histories.append(history)
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"  {method}/{objective}: seed {model_seed}, epoch {epoch + 1}/{epochs}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    train_x = collect_embeddings(model, data, train_indices, device) if method != "state_model" else np.empty((0, 0))
    validation_x = collect_embeddings(model, data, validation_indices, device) if method != "state_model" else np.empty((0, 0))
    probe: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    if method != "state_model":
        probe = fit_stable_probe(
            train_x,
            data.states[train_indices],
            validation_x,
            data.states[validation_indices],
        )
        validation_probe = apply_probe_numpy(validation_x, probe)
        diagnostics.update(probe_metrics(validation_probe, data.states[validation_indices]))
        diagnostics.update(
            probe_bootstrap_rmse(
                train_x,
                data.states[train_indices],
                data.episode_ids[train_indices],
                validation_x,
                alpha=float(probe["alpha"]),
                seed=stable_seed(V2_VERSION, "probe-bootstrap", data_seed, model_seed, method, objective),
                replicates=20,
            )
        )
        diagnostics.update(effective_rank_diagnostics(validation_x))
        target_diagnostics = effective_rank_diagnostics(
            collect_embeddings(model, data, validation_indices, device, target=True)
        )
        diagnostics.update({f"target_{key}": value for key, value in target_diagnostics.items()})
        diagnostics.update({f"probe_{key}": value for key, value in probe.items() if key not in {"feature_mean", "feature_scale", "target_mean", "coef"}})
    diagnostics["finite"] = bool(
        all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters())
        and all(math.isfinite(float(item)) for history in histories for item in (history["train_total_loss"], history["validation_total_loss"]))
    )
    record: dict[str, object] = {
        "protocol_version": V2_VERSION,
        "stage": stage,
        "method": method,
        "objective": objective,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "status": "complete",
        "epochs": epochs,
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "train_seconds": train_seconds,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "precision": "float32",
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        **diagnostics,
    }
    if method == "jepa":
        record["representation_pass"] = (
            diagnostics.get("finite") is True
            and int(diagnostics.get("low_variance_dims", 32)) <= 16
            and float(diagnostics.get("effective_rank", 0.0)) >= 8.0
            and diagnostics.get("target_finite") is True
            and int(diagnostics.get("target_low_variance_dims", 32)) <= 16
            and float(diagnostics.get("target_effective_rank", 0.0)) >= 8.0
        )
    checkpoint_dir = output_dir / CHECKPOINT_DIRNAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"{method}-{objective}-seed-{model_seed}.pt"
    torch.save({"state_dict": model.state_dict(), "probe": probe, "meta": record}, checkpoint_dir / checkpoint_name)
    return record, {"model": model, "probe": probe, "record": record, "checkpoint": checkpoint_dir / checkpoint_name}, histories, gradient_rows


def fit_seed_pair(
    method: str,
    objective: str,
    data: TransitionData,
    model_seed: int,
    data_seed: int,
    device: torch.device,
    output_dir: Path,
    **kwargs: Any,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Keep model and data seed arguments explicit at every training call site."""
    return train_one(
        method,
        objective,
        data,
        model_seed,
        data_seed,
        device,
        output_dir,
        **kwargs,
    )


def _checkpoint_path(output_dir: Path, method: str, objective: str, model_seed: int) -> Path:
    return output_dir / CHECKPOINT_DIRNAME / f"{method}-{objective}-seed-{model_seed}.pt"


def load_fitted_model(
    output_dir: Path,
    method: str,
    objective: str,
    model_seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(
        _checkpoint_path(output_dir, method, objective, model_seed),
        map_location=device,
        weights_only=False,
    )
    model = build_model(method).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, dict(checkpoint.get("probe", {}))


def sample_validation_rows(data: TransitionData, data_seed: int, count: int = 20) -> np.ndarray:
    _train, validation = split_episode_indices(data.episode_ids)
    if len(validation) < count:
        raise RuntimeError(f"data seed {data_seed} has only {len(validation)} validation rows")
    rng = np.random.default_rng(stable_seed(V2_VERSION, "validation-intervention", data_seed))
    return np.sort(rng.choice(validation, size=count, replace=False))


def _predict_one_step(
    method: str,
    model: nn.Module | None,
    probe: Mapping[str, object] | None,
    data: TransitionData,
    row_index: int,
    actions: Sequence[float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return raw states, latent predictions, and norm diagnostics for a fixed context."""
    action_values = torch.as_tensor(actions, dtype=torch.float32, device=device)
    state = torch.as_tensor(data.states[row_index : row_index + 1], dtype=torch.float32, device=device)
    exact = mountaincar_step(data.states[row_index], np.asarray(actions, dtype=np.float32))
    latent_predictions = np.empty((len(actions), 0), dtype=np.float32)
    encoder_norm = math.nan
    predictor_norm = math.nan

    if method in {"exact", "handcrafted"}:
        source_state = data.states[row_index]
        if method == "handcrafted":
            source_state = visual_state(data.frames[row_index, 0], data.frames[row_index, 1])
        predicted = mountaincar_step(source_state, np.asarray(actions, dtype=np.float32))
        return predicted, latent_predictions, exact, encoder_norm, predictor_norm

    if model is None:
        raise ValueError(f"{method} requires a loaded model")
    if method == "state_model":
        with torch.no_grad():
            raw = model(state.expand(len(actions), -1), action_values)
        return raw.detach().cpu().numpy(), latent_predictions, exact, encoder_norm, tensor_norm(raw)

    observation = context_tensor(data, np.asarray([row_index]), device).expand(len(actions), -1, -1, -1)
    with torch.no_grad():
        embedding = model.encoder(observation[:1])
        encoder_norm = tensor_norm(embedding)
        if method == "supervised_visual":
            raw = model(observation, action_values)
            predictor_norm = tensor_norm(raw)
        elif method == "jepa":
            latent = model.predictor(embedding.expand(len(actions), -1), action_values)
            latent_predictions = latent.detach().float().cpu().numpy()
            predictor_norm = tensor_norm(latent)
            if probe is None:
                raise ValueError("JEPA action readout requires a fitted probe")
            raw = torch.as_tensor(
                apply_probe_numpy(latent_predictions, probe), dtype=torch.float32, device=device
            )
        else:
            raise ValueError(f"unsupported method for action prediction: {method}")
    return raw.detach().cpu().numpy(), latent_predictions, exact, encoder_norm, predictor_norm


def evaluate_action_triplets(
    method: str,
    objective: str,
    model_seed: int,
    data_seed: int,
    data: TransitionData,
    indices: np.ndarray,
    device: torch.device,
    *,
    model: nn.Module | None = None,
    probe: Mapping[str, object] | None = None,
    stage: str = "validation",
    start_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    if start_ids is not None and len(start_ids) != len(indices):
        raise ValueError("start_ids must match the action-triplet indices")
    records: list[dict[str, object]] = []
    for local_index, row_index in enumerate(indices):
        rows = _predict_one_step(method, model, probe, data, int(row_index), ACTION_VALUES, device)
        raw_states, latents, actual_states, encoder_norm, predictor_norm = rows
        clipped, clip_mask = unclipped_state_diagnostics(torch.as_tensor(raw_states, dtype=torch.float32))
        clipped_np = clipped.cpu().numpy()
        mask_np = clip_mask.cpu().numpy()
        context_id = str(start_ids[local_index]) if start_ids is not None else f"{stage}-{data_seed}-{local_index:03d}"
        baseline_latent = latents[1] if latents.size else None
        for action_index, action in enumerate(ACTION_VALUES):
            latent = latents[action_index] if latents.size else np.empty(0, dtype=np.float32)
            latent_delta = float(np.linalg.norm(latent - baseline_latent)) if baseline_latent is not None else math.nan
            records.append(
                {
                    "protocol_version": V2_VERSION,
                    "stage": stage,
                    "method": method,
                    "objective": objective,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "start_id": context_id,
                    "source_row": int(row_index),
                    "episode_id": int(data.episode_ids[row_index]),
                    "action": action,
                    "actual_position": float(actual_states[action_index, 0]),
                    "actual_velocity": float(actual_states[action_index, 1]),
                    "predicted_position_raw": float(raw_states[action_index, 0]),
                    "predicted_velocity_raw": float(raw_states[action_index, 1]),
                    "predicted_position": float(clipped_np[action_index, 0]),
                    "predicted_velocity": float(clipped_np[action_index, 1]),
                    "position_clipped": bool(mask_np[action_index, 0]),
                    "velocity_clipped": bool(mask_np[action_index, 1]),
                    "encoder_output_norm": encoder_norm,
                    "predictor_output_norm": predictor_norm,
                    "predicted_latent_norm": float(np.linalg.norm(latent)) if latent.size else math.nan,
                    "latent_displacement_from_action_zero": latent_delta,
                    "action_sensitivity_position": float(raw_states[action_index, 0] - raw_states[1, 0]),
                    "action_sensitivity_velocity": float(raw_states[action_index, 1] - raw_states[1, 1]),
                    "actual_action_delta_position": float(actual_states[action_index, 0] - actual_states[1, 0]),
                    "actual_action_delta_velocity": float(actual_states[action_index, 1] - actual_states[1, 1]),
                }
            )
    return records


def _truthful_diagnostic_fields(record: Mapping[str, object]) -> bool:
    return all(
        isinstance(record.get(name), (bool, np.bool_)) and bool(record.get(name))
        for name in ("finite",)
    )


def _fit_validation_gates(
    record: Mapping[str, object], action_summary: Mapping[str, object], *, method: str
) -> dict[str, object]:
    if method == "state_model":
        return {"state_finite": _truthful_diagnostic_fields(record), "validation_pass": _truthful_diagnostic_fields(record)}
    finite = _truthful_diagnostic_fields(record)
    representation_ok = (
        finite
        and int(record.get("low_variance_dims", 32)) <= 16
        and float(record.get("effective_rank", 0.0)) >= 8.0
        and record.get("target_finite") is True
        and int(record.get("target_low_variance_dims", 32)) <= 16
        and float(record.get("target_effective_rank", 0.0)) >= 8.0
    )
    probe_ok = (
        float(record.get("probe_gram_condition", math.inf)) <= 1e6
        and float(record.get("position_rmse", math.inf)) <= 0.01
        and float(record.get("velocity_rmse", math.inf)) <= 0.005
        and float(record.get("position_r2", -math.inf)) >= 0.80
        and float(record.get("velocity_r2", -math.inf)) >= 0.50
        and float(record.get("position_mae", math.inf)) <= 0.02
        and float(record.get("velocity_mae", math.inf)) <= 0.01
    )
    action_ok = (
        float(action_summary.get("position_median_normalized_contrast_error", math.inf)) <= 0.25
        and float(action_summary.get("velocity_median_normalized_contrast_error", math.inf)) <= 0.25
        and float(action_summary.get("position_order_accuracy", 0.0)) >= 0.80
        and float(action_summary.get("velocity_order_accuracy", 0.0)) >= 0.80
        and int(action_summary.get("position_valid_starts", 0)) >= 12
        and int(action_summary.get("velocity_valid_starts", 0)) >= 12
    )
    clip_count = int(action_summary.get("unbounded_prediction_count", 0))
    prediction_count = max(1, int(action_summary.get("action_prediction_count", 0)))
    clipping_ok = clip_count / prediction_count <= 0.05
    if method == "supervised_visual":
        direct_state_ok = (
            float(action_summary.get("one_step_position_mae", math.inf)) <= 0.02
            and float(action_summary.get("one_step_velocity_mae", math.inf)) <= 0.01
        )
        gates = {
            "finite": finite,
            "one_step_state_pass": direct_state_ok,
            "action_fidelity_pass": action_ok,
            "clipping_pass": clipping_ok,
        }
        gates["validation_pass"] = all(bool(value) for value in gates.values())
        return gates
    one_step_state_ok = (
        float(action_summary.get("one_step_position_mae", math.inf)) <= 0.02
        and float(action_summary.get("one_step_velocity_mae", math.inf)) <= 0.01
    )
    gates = {
        "finite": finite,
        "representation_pass": representation_ok,
        "probe_pass": probe_ok,
        "one_step_state_pass": one_step_state_ok,
        "action_fidelity_pass": action_ok,
        "clipping_pass": clipping_ok,
    }
    gates["validation_pass"] = all(bool(value) for key, value in gates.items() if key != "finite") and finite
    return gates


def summarize_actions(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int, int], dict[str, object]]:
    groups: dict[tuple[str, str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), str(row["objective"]), int(row["model_seed"]), int(row["data_seed"]))
        groups[key].append(row)
    result = {}
    for key, group in groups.items():
        metrics = action_effect_metrics(group)
        # The action metric helper groups by start and method. Preserve the three-act rows.
        metrics["method"] = key[0]
        metrics["objective"] = key[1]
        metrics["model_seed"] = key[2]
        metrics["data_seed"] = key[3]
        metrics["one_step_position_mae"] = float(np.mean([
            abs(float(row["predicted_position_raw"]) - float(row["actual_position"]))
            for row in group
        ]))
        metrics["one_step_velocity_mae"] = float(np.mean([
            abs(float(row["predicted_velocity_raw"]) - float(row["actual_velocity"]))
            for row in group
        ]))
        result[key] = metrics
    return result


def verify_simulator_match(data: TransitionData, sample_count: int = 500) -> dict[str, object]:
    count = min(sample_count, len(data.actions))
    indices = np.linspace(0, len(data.actions) - 1, count, dtype=np.int64)
    expected = mountaincar_step(data.states[indices], data.actions[indices])
    errors = np.abs(expected - data.next_states[indices])
    maximum = float(errors.max(initial=0.0))
    return {"rows_checked": count, "max_abs_error": maximum, "exact_match": maximum <= 2e-6}


def _model_instance_and_probe(
    output_dir: Path,
    method: str,
    objective: str,
    model_seed: int,
    device: torch.device,
) -> tuple[nn.Module | None, dict[str, object] | None]:
    if method in {"exact", "handcrafted"}:
        return None, None
    return load_fitted_model(output_dir, method, objective, model_seed, device)


def evaluate_group_actions(
    output_dir: Path,
    data: TransitionData,
    data_seed: int,
    model_seeds: Sequence[int],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[tuple[str, str, int, int], dict[str, object]]]:
    indices = sample_validation_rows(data, data_seed)
    all_rows: list[dict[str, object]] = []
    for model_seed in model_seeds:
        for method, objective in TRAIN_METHODS:
            if not _checkpoint_path(output_dir, method, objective, model_seed).is_file():
                continue
            model, probe = _model_instance_and_probe(output_dir, method, objective, model_seed, device)
            try:
                all_rows.extend(
                    evaluate_action_triplets(
                        method, objective, model_seed, data_seed, data, indices, device,
                        model=model, probe=probe,
                    )
                )
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    all_rows.extend(evaluate_action_triplets("exact", "simulator_oracle", 0, data_seed, data, indices, device))
    all_rows.extend(evaluate_action_triplets("handcrafted", "visual_state_plus_oracle", 0, data_seed, data, indices, device))
    return all_rows, summarize_actions(all_rows)


def write_validation_stage(output_dir: Path, device_name: str, *, epochs: int, batch_size: int) -> dict[str, object]:
    output_dir = require_drive_output(output_dir)
    if (output_dir / "validation-summary.json").exists():
        raise FileExistsError(f"validation output already exists; use a new run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    device = select_device(device_name)
    model_records: list[dict[str, object]] = []
    gradient_records: list[dict[str, object]] = []
    history_records: list[dict[str, object]] = []
    action_records: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    per_pair_actions: dict[tuple[str, str, int, int], dict[str, object]] = {}
    control_exact_checks: list[dict[str, object]] = []
    for data_seed, model_seed in zip(TRAIN_DATA_SEEDS, MODEL_SEEDS, strict=True):
        print(f"Collect matched training/validation group data={data_seed}, model={model_seed}", flush=True)
        data = collect_random_transitions(data_seed, episodes=20)
        exact_check = verify_simulator_match(data)
        control_exact_checks.append({"data_seed": data_seed, **exact_check})
        if not exact_check["exact_match"]:
            raise RuntimeError(f"simulator dynamics did not match Gymnasium data for seed {data_seed}: {exact_check}")
        train_indices, validation_indices = split_episode_indices(data.episode_ids)
        split_records.append(
            {
                "data_seed": data_seed,
                "train_episodes": sorted(int(value) for value in np.unique(data.episode_ids[train_indices])),
                "validation_episodes": sorted(int(value) for value in np.unique(data.episode_ids[validation_indices])),
                "train_rows": int(len(train_indices)),
                "validation_rows": int(len(validation_indices)),
                "validation_intervention_rows": sample_validation_rows(data, data_seed).tolist(),
            }
        )
        for method, objective in TRAIN_METHODS:
            fit_started = time.perf_counter()
            print(f"  Fit {method}/{objective}, model seed {model_seed}", flush=True)
            try:
                record, _fitted, histories, gradients = fit_seed_pair(
                    method,
                    objective,
                    data,
                    model_seed,
                    data_seed,
                    device,
                    output_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                )
                record["fit_elapsed_seconds"] = time.perf_counter() - fit_started
                model_records.append(record)
                history_records.extend(histories)
                gradient_records.extend(gradients)
                del _fitted
            except Exception as exc:
                model_records.append(
                    {
                        "protocol_version": V2_VERSION,
                        "stage": "validation",
                        "method": method,
                        "objective": objective,
                        "model_seed": model_seed,
                        "data_seed": data_seed,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "fit_elapsed_seconds": time.perf_counter() - fit_started,
                    }
                )
                print(f"  FAILED {method}/{objective}: {type(exc).__name__}: {exc}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                allocated_mib = torch.cuda.max_memory_allocated(device) / 1024**2
                if allocated_mib > 2048:
                    raise RuntimeError(f"GPU memory budget exceeded: {allocated_mib:.1f} MiB")
            disk_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
            if disk_bytes > 5 * 1024**3:
                raise RuntimeError(f"temporary output storage budget exceeded: {disk_bytes} bytes")
            training_seconds = sum(
                float(row.get("train_seconds", 0.0))
                for row in model_records
                if row.get("status") == "complete"
            )
            if device.type == "cuda" and training_seconds > 90 * 60:
                raise RuntimeError("90 GPU-minute maximum validation budget reached")
        group_actions, _group_summary = evaluate_group_actions(output_dir, data, data_seed, (model_seed,), device)
        action_records.extend(group_actions)
        per_pair_actions.update(summarize_actions(group_actions))
        del data

    action_summaries = []
    record_lookup = {
        (str(row.get("method")), str(row.get("objective")), int(row.get("model_seed", -1)), int(row.get("data_seed", -1))): row
        for row in model_records
    }
    for key, metrics in sorted(per_pair_actions.items()):
        model_record = record_lookup.get(key, {})
        action_summary = {
            key_name: value
            for key_name, value in metrics.items()
            if key_name != "per_start"
        }
        related_rows = [
            row for row in action_records
            if row["method"] == key[0]
            and row["objective"] == key[1]
            and row["model_seed"] == key[2]
            and row["data_seed"] == key[3]
        ]
        unbounded_count = sum(
            bool(row["position_clipped"] or row["velocity_clipped"])
            for row in related_rows
        )
        action_prediction_count = len(related_rows)
        clipping_fraction = unbounded_count / max(1, action_prediction_count)
        action_summary.update(
            {
                "unbounded_prediction_count": unbounded_count,
                "action_prediction_count": action_prediction_count,
                "unbounded_prediction_fraction": clipping_fraction,
                "clipping_pass": clipping_fraction <= 0.05,
            }
        )
        if key[0] in {"jepa", "supervised_visual"}:
            action_summary.update(_fit_validation_gates(model_record, action_summary, method=key[0]))
            action_summary["clipping_pass"] = clipping_fraction <= 0.05
            action_summary["validation_pass"] = bool(action_summary.get("validation_pass")) and bool(action_summary["clipping_pass"])
        elif key[0] == "exact":
            action_summary["control_pass"] = bool(action_summary["one_step_position_mae"] <= 2e-6 and action_summary["one_step_velocity_mae"] <= 2e-6)
        elif key[0] == "state_model":
            action_summary["control_pass"] = bool(_truthful_diagnostic_fields(model_record))
        else:
            action_summary["control_pass"] = True
        action_summaries.append(action_summary)
        if model_record:
            model_record.update(action_summary)

    visual_rows = [row for row in action_summaries if row["method"] == "supervised_visual"]
    visual_pass = len(visual_rows) == 3 and all(bool(row.get("validation_pass")) for row in visual_rows)
    objective_status = []
    eligible = []
    for objective in OBJECTIVES:
        rows = [
            row for row in action_summaries
            if row["method"] == "jepa" and row["objective"] == objective
        ]
        passed = sum(bool(row.get("validation_pass")) for row in rows)
        family = {
            "objective": objective,
            "seed_rows": len(rows),
            "passing_seeds": passed,
            "family_valid_starts_position": sum(int(row.get("position_valid_starts", 0)) for row in rows),
            "family_valid_starts_velocity": sum(int(row.get("velocity_valid_starts", 0)) for row in rows),
            "eligible": passed >= 2 and all(
                int(row.get("position_valid_starts", 0)) >= 12
                and int(row.get("velocity_valid_starts", 0)) >= 12
                for row in rows if bool(row.get("validation_pass"))
            ),
        }
        # A complete family must have at least 40 non-tied contexts per coordinate.
        family["eligible"] = bool(family["eligible"]) and int(family["family_valid_starts_position"]) >= 40 and int(family["family_valid_starts_velocity"]) >= 40
        objective_status.append(family)
        if family["eligible"]:
            eligible.append(objective)
    selected = None
    selection_records: list[dict[str, object]] = []
    if visual_pass and eligible:
        selection_rows = []
        for objective in eligible:
            objective_rows = [row for row in action_summaries if row["method"] == "jepa" and row["objective"] == objective and row.get("validation_pass")]
            primary = float(np.median([
                (float(row["position_median_normalized_contrast_error"]) + float(row["velocity_median_normalized_contrast_error"])) / 2
                for row in objective_rows
            ]))
            complete_rows = [
                row for row in model_records
                if row.get("method") == "jepa"
                and row.get("objective") == objective
                and row.get("status") == "complete"
            ]
            five_mae = _validation_rollout_tie_break(output_dir, objective, complete_rows, device)
            tie_order = OBJECTIVES.index(objective)
            selection_rows.append((primary, five_mae, tie_order, objective))
            selection_records.append(
                {
                    "objective": objective,
                    "median_action_contrast_error": primary,
                    "mean_five_step_state_mae": five_mae,
                    "fixed_tie_order": tie_order,
                    "eligible": True,
                }
            )
        selected = min(selection_rows)[-1]
    run_seconds = time.perf_counter() - run_started
    summary = {
        "protocol_version": V2_VERSION,
        "protocol_lock": str(LOCK_PATH.relative_to(ROOT)),
        "stage": "validation",
        "status": "complete",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "model_fit_count": len(model_records),
        "successful_fits": sum(row.get("status") == "complete" for row in model_records),
        "failed_fits": sum(row.get("status") != "complete" for row in model_records),
        "training_seconds": sum(float(row.get("train_seconds", 0.0)) for row in model_records),
        "total_wall_seconds": run_seconds,
        "max_peak_allocated_mib": max((float(row.get("peak_allocated_mib", 0.0)) for row in model_records), default=0.0),
        "simulator_checks": control_exact_checks,
        "visual_positive_control_pass": visual_pass,
        "objective_families": objective_status,
        "objective_selection": selection_records,
        "selected_objective": selected,
        "heldout_open": bool(visual_pass and selected),
        "planning_open": False,
        "test_seed_groups_opened": False,
        "failure_reason": "" if visual_pass and selected else (
            "visual positive control failed" if not visual_pass else "no JEPA objective family passed the frozen validation gates"
        ),
    }
    validate_experiment_completeness(model_records, action_records)
    write_csv(output_dir / "validation-model-runs.csv", model_records)
    write_csv(output_dir / "validation-training-history.csv", history_records)
    write_csv(output_dir / "validation-gradient-diagnostics.csv", gradient_records)
    write_csv(output_dir / "validation-action-triplets.csv", action_records)
    write_csv(output_dir / "validation-action-summary.csv", action_summaries)
    write_json(output_dir / "validation-data-splits.json", {"groups": split_records, "exact_checks": control_exact_checks})
    write_json(output_dir / "validation-summary.json", summary)
    return summary


def _validation_rollout_tie_break(
    output_dir: Path,
    objective: str,
    passed_rows: Sequence[Mapping[str, object]],
    device: torch.device,
) -> float:
    """Return mean five-step MAE on the frozen 12 validation starts."""
    errors = []
    for record in passed_rows:
        data_seed = int(record["data_seed"])
        model_seed = int(record["model_seed"])
        data = collect_random_transitions(data_seed, episodes=20)
        _train, validation = split_episode_indices(data.episode_ids)
        model, probe = load_fitted_model(output_dir, "jepa", objective, model_seed, device)
        try:
            for episode_id in range(16, 20):
                episode_rows = validation[data.episode_ids[validation] == episode_id]
                if len(episode_rows) < 5:
                    continue
                start_index = int(episode_rows[0])
                latent = model.encode(context_tensor(data, np.asarray([start_index]), device))
                with torch.no_grad():
                    for offset, index in enumerate(episode_rows[:5]):
                        action = torch.as_tensor([data.actions[int(index)]], device=device)
                        latent = model.predictor(latent, action)
                    prediction = apply_probe_numpy(latent.detach().cpu().numpy(), probe)[0]
                target = data.next_states[int(episode_rows[4])]
                errors.append(float(np.abs(prediction - target).mean()))
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del data
    if len(errors) != 12:
        return math.inf
    return float(np.mean(errors))


def _context_embeddings(
    model: nn.Module,
    data: TransitionData,
    indices: np.ndarray,
    device: torch.device,
    *,
    target: bool = False,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(indices), 256):
            selected = indices[offset : offset + 256]
            embedding = model.encoder(context_tensor(data, selected, device, target=target))
            outputs.append(embedding.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _historical_probe_bootstrap(
    train_x: np.ndarray,
    train_y: np.ndarray,
    episode_ids: np.ndarray,
    validation_x: np.ndarray,
    *,
    alpha: float,
    seed: int,
    replicates: int = 20,
) -> dict[str, float]:
    """Refit the original v1 standardized ridge procedure by episode blocks."""
    rng = np.random.default_rng(seed)
    predictions = []
    groups = np.unique(episode_ids)
    for _ in range(replicates):
        selected_groups = rng.choice(groups, size=len(groups), replace=True)
        selected = np.concatenate([np.flatnonzero(episode_ids == group) for group in selected_groups])
        x = train_x[selected].astype(np.float64)
        y = train_y[selected].astype(np.float64)
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-4)
        normalized = (x - mean) / scale
        target_mean = y.mean(axis=0)
        gram = normalized.T @ normalized + float(alpha) * np.eye(normalized.shape[1])
        coefficient = np.linalg.lstsq(gram, normalized.T @ (y - target_mean), rcond=None)[0]
        predictions.append(((validation_x - mean) / scale) @ coefficient + target_mean)
    spread = np.stack(predictions).std(axis=0, ddof=0)
    return {
        "v1_bootstrap_position_prediction_sd_rmse": float(np.sqrt(np.mean(spread[:, 0] ** 2))),
        "v1_bootstrap_velocity_prediction_sd_rmse": float(np.sqrt(np.mean(spread[:, 1] ** 2))),
        "v1_bootstrap_replicates": replicates,
    }


def audit_v1_raw_records() -> dict[str, object]:
    """Recompute the recorded v1 claims directly from retained CSV rows."""
    evidence = EVIDENCE_DIR
    model_path = evidence / "visual-jepa-decision-study-v1-model-runs.csv"
    probe_path = evidence / "visual-jepa-decision-study-v1-probe-test.csv"
    action_path = evidence / "visual-jepa-decision-study-v1-interventions.csv"
    episode_path = evidence / "visual-jepa-decision-study-v1-episodes.csv"
    if not all(path.is_file() for path in (model_path, probe_path, action_path, episode_path)):
        raise FileNotFoundError("one or more retained Notebook 23 raw evidence CSVs are missing")
    with model_path.open(newline="", encoding="utf-8") as handle:
        models = list(csv.DictReader(handle))
    with probe_path.open(newline="", encoding="utf-8") as handle:
        probes = list(csv.DictReader(handle))
    with action_path.open(newline="", encoding="utf-8") as handle:
        interventions = list(csv.DictReader(handle))
    with episode_path.open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    jepa_models = [row for row in models if row["method"] == "jepa" and row["status"] == "complete"]
    jepa_probes = [row for row in probes if row["family"] == "jepa"]
    jepa_actions = [row for row in interventions if row["method"] == "jepa" and row["status"] == "complete"]
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in jepa_actions:
        groups[(row["model_seed"], row["start_id"])].append(row)
    spans: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"position": [], "velocity": []})
    for (model_seed, _start_id), group in groups.items():
        for coordinate in ("position", "velocity"):
            predictions = [float(row[f"predicted_{coordinate}"]) for row in group]
            spans[model_seed][coordinate].append(max(predictions) - min(predictions))
    episode_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in episodes:
        if row["status"] == "complete":
            episode_groups[(row["method"], row["initialization"])].append(row)
    episode_summary = {
        f"{method}/{initialization}": {
            "successes": sum(int(row["success"]) for row in rows),
            "episodes": len(rows),
        }
        for (method, initialization), rows in sorted(episode_groups.items())
    }
    return {
        "source_files": {
            "model_runs_sha256": sha256_file(str(model_path)),
            "probe_test_sha256": sha256_file(str(probe_path)),
            "interventions_sha256": sha256_file(str(action_path)),
            "episodes_sha256": sha256_file(str(episode_path)),
        },
        "jepa_model_rows": len(jepa_models),
        "all_five_jepa_models_complete": len(jepa_models) == 5,
        "all_five_recorded_low_spread_all_dimensions": len(jepa_models) == 5 and all(
            int(row["embedding_low_variance_dims"]) == int(row["embedding_dimension"])
            and row["embedding_collapsed"].lower() == "true"
            for row in jepa_models
        ),
        "recorded_position_probe_r2_min": min(float(row["position_r2"]) for row in jepa_probes),
        "recorded_position_probe_r2_max": max(float(row["position_r2"]) for row in jepa_probes),
        "recorded_velocity_probe_r2_min": min(float(row["velocity_r2"]) for row in jepa_probes),
        "recorded_velocity_probe_r2_max": max(float(row["velocity_r2"]) for row in jepa_probes),
        "jepa_intervention_rows": len(jepa_actions),
        "jepa_action_triplet_starts": len(groups),
        "post_clamp_action_spans_by_model_seed": {
            seed: {
                coordinate: {
                    "minimum": min(values),
                    "median": float(np.median(values)),
                    "maximum": max(values),
                    "zero_start_count": sum(value <= 1e-12 for value in values),
                    "full_position_range_start_count": sum(abs(value - 1.8) <= 1e-6 for value in values)
                    if coordinate == "position" else 0,
                }
                for coordinate, values in coordinate_rows.items()
            }
            for seed, coordinate_rows in sorted(spans.items())
        },
        "episode_successes": episode_summary,
    }


def run_v1_audit(
    output_dir: Path,
    checkpoint_dir: Path,
    device_name: str,
) -> dict[str, object]:
    """Replay diagnostic measurements from the retained v1 checkpoints only."""
    output_dir = require_drive_output(output_dir)
    if (output_dir / "v1-audit-summary.json").exists():
        raise FileExistsError(f"v1 audit output already exists; use a new output directory: {output_dir}")
    device = select_device(device_name)
    data_seeds = (8300, 8301, 8302, 8303, 8304)
    model_seeds = (7401, 7402, 7403, 7404, 7405)
    test_seeds = (8400, 8401, 8402, 8403, 8404)
    starts = collect_intervention_starts((8600, 8601, 8602, 8603, 8604))
    audit_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for index, (data_seed, model_seed, test_seed) in enumerate(zip(data_seeds, model_seeds, test_seeds, strict=True)):
        checkpoint_path = checkpoint_dir / f"jepa-seed-{model_seed}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"retained v1 checkpoint is missing: {checkpoint_path}")
        data = collect_random_transitions(data_seed, episodes=20)
        train_indices, validation_indices = split_episode_indices(data.episode_ids)
        model = ActionConditionedJEPA(embedding_dim=32).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        probe = dict(checkpoint["probe"])
        train_context = _context_embeddings(model, data, train_indices, device)
        validation_context = _context_embeddings(model, data, validation_indices, device)
        train_target = _context_embeddings(model, data, train_indices, device, target=True)
        validation_target = _context_embeddings(model, data, validation_indices, device, target=True)
        test_data = collect_random_transitions(test_seed, episodes=20)
        test_indices = np.arange(len(test_data.actions), dtype=np.int64)
        test_target = _context_embeddings(model, test_data, test_indices, device, target=True)
        reproduced_test = probe_metrics(
            apply_probe_numpy(test_target, probe), test_data.next_states[test_indices]
        )
        feature_mean = np.asarray(probe["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(probe["feature_scale"], dtype=np.float64)
        coefficient = np.asarray(probe["coef"], dtype=np.float64)
        normalized_train = (train_target - feature_mean) / feature_scale
        gram = normalized_train.T @ normalized_train + float(probe["alpha"]) * np.eye(normalized_train.shape[1])
        train_effective = effective_rank_diagnostics(train_target)
        validation_effective = effective_rank_diagnostics(validation_target)
        context_validation_effective = effective_rank_diagnostics(validation_context)
        record = {
            "protocol_version": V2_VERSION,
            "historical_protocol": "visual-jepa-decision-study-v1",
            "method": "jepa",
            "objective": "v1_sigreg",
            "model_seed": model_seed,
            "data_seed": data_seed,
            "test_seed": test_seed,
            "status": "complete",
            "checkpoint_sha256": sha256_file(str(checkpoint_path)),
            "v1_metadata_embedding_low_variance_dims": checkpoint.get("meta", {}).get("embedding_low_variance_dims", ""),
            "v1_metadata_embedding_collapsed": checkpoint.get("meta", {}).get("embedding_collapsed", ""),
            "context_validation_mean_std": context_validation_effective["mean_std"],
            "context_validation_low_variance_dims": context_validation_effective["low_variance_dims"],
            "context_validation_effective_rank": context_validation_effective["effective_rank"],
            "context_validation_mean_norm": effective_rank_diagnostics(validation_context)["mean_norm"],
            "context_train_mean_std": effective_rank_diagnostics(train_context)["mean_std"],
            "context_train_effective_rank": effective_rank_diagnostics(train_context)["effective_rank"],
            "target_train_mean_std": train_effective["mean_std"],
            "target_train_effective_rank": train_effective["effective_rank"],
            "target_validation_mean_std": validation_effective["mean_std"],
            "target_validation_low_variance_dims": validation_effective["low_variance_dims"],
            "target_validation_effective_rank": validation_effective["effective_rank"],
            "target_validation_mean_norm": validation_effective["mean_norm"],
            "target_validation_rms_norm": validation_effective["rms_norm"],
            "target_validation_max_norm": validation_effective["max_norm"],
            "v1_probe_alpha": float(probe["alpha"]),
            "v1_probe_min_feature_scale": float(feature_scale.min()),
            "v1_probe_max_raw_coefficient_abs": float(np.abs(coefficient / feature_scale[:, None]).max()),
            "v1_probe_raw_coefficient_l2": float(np.linalg.norm(coefficient / feature_scale[:, None])),
            "v1_probe_standardized_coefficient_l2": float(np.linalg.norm(coefficient)),
            "v1_probe_gram_condition": float(np.linalg.cond(gram)),
            "reproduced_test_position_r2": reproduced_test["position_r2"],
            "reproduced_test_velocity_r2": reproduced_test["velocity_r2"],
            "reproduced_test_position_mae": reproduced_test["position_mae"],
            "reproduced_test_velocity_mae": reproduced_test["velocity_mae"],
            **_historical_probe_bootstrap(
                train_target,
                data.next_states[train_indices],
                data.episode_ids[train_indices],
                validation_target,
                alpha=float(probe["alpha"]),
                seed=stable_seed(V2_VERSION, "v1-probe-bootstrap", data_seed, model_seed),
            ),
        }
        gradient_rows.append(
            component_gradient_diagnostics(
                model,
                "jepa",
                "sigreg",
                training_batch(data, train_indices[:256], device),
                torch.Generator(device=device).manual_seed(model_seed + 1),
                epoch=35,
                model_seed=model_seed,
                data_seed=data_seed,
            )
        )
        seed_starts = [start for start in starts]
        visual_data = TransitionData(
            frames=np.stack([
                np.stack((start.previous_frame, start.current_frame, start.current_frame))
                for start in seed_starts
            ]),
            actions=np.zeros(len(seed_starts), dtype=np.float32),
            states=np.stack([start.state for start in seed_starts]),
            next_states=np.stack([mountaincar_step(start.state, 0.0) for start in seed_starts]),
            episode_ids=np.zeros(len(seed_starts), dtype=np.int16),
        )
        seed_action_rows = evaluate_action_triplets(
            "jepa", "v1_sigreg", model_seed, data_seed, visual_data,
            np.arange(len(seed_starts)), device, model=model, probe=probe,
            stage="historical_v1",
            start_ids=[start.start_id for start in seed_starts],
        )
        action_rows.extend(seed_action_rows)
        summary_metrics = action_effect_metrics(seed_action_rows)
        record.update({
            "historical_action_starts": summary_metrics["starts"],
            "historical_position_median_normalized_contrast_error": summary_metrics["position_median_normalized_contrast_error"],
            "historical_velocity_median_normalized_contrast_error": summary_metrics["velocity_median_normalized_contrast_error"],
            "historical_position_action_order_accuracy": summary_metrics["position_order_accuracy"],
            "historical_velocity_action_order_accuracy": summary_metrics["velocity_order_accuracy"],
        })
        audit_rows.append(record)
        del test_data, data, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Audited retained v1 model seed {model_seed}", flush=True)
    raw_intervention_path = EVIDENCE_DIR / "visual-jepa-decision-study-v1-interventions.csv"
    raw_probe_path = EVIDENCE_DIR / "visual-jepa-decision-study-v1-probe-test.csv"
    with raw_intervention_path.open(newline="", encoding="utf-8") as handle:
        raw_interventions = list(csv.DictReader(handle))
    with raw_probe_path.open(newline="", encoding="utf-8") as handle:
        raw_probe_rows = list(csv.DictReader(handle))
    for record in audit_rows:
        model_seed = int(record["model_seed"])
        replayed = {
            (str(row["start_id"]), float(row["action"])): row
            for row in action_rows
            if int(row["model_seed"]) == model_seed
        }
        originals = {
            (str(row["start_id"]), float(row["action"])): row
            for row in raw_interventions
            if row["method"] == "jepa"
            and row["status"] == "complete"
            and int(row["model_seed"]) == model_seed
        }
        common = sorted(set(replayed).intersection(originals))
        record["original_intervention_rows_matched"] = len(common)
        record["original_intervention_max_abs_position_replay_difference"] = max(
            (abs(float(replayed[key]["predicted_position"]) - float(originals[key]["predicted_position"])) for key in common),
            default=math.nan,
        )
        record["original_intervention_max_abs_velocity_replay_difference"] = max(
            (abs(float(replayed[key]["predicted_velocity"]) - float(originals[key]["predicted_velocity"])) for key in common),
            default=math.nan,
        )
        original_probe = next(
            row for row in raw_probe_rows
            if row["family"] == "jepa" and int(row["model_seed"]) == model_seed
        )
        record["original_probe_position_r2"] = float(original_probe["position_r2"])
        record["original_probe_velocity_r2"] = float(original_probe["velocity_r2"])
        record["reproduced_probe_position_r2_abs_difference"] = abs(float(record["reproduced_test_position_r2"]) - float(original_probe["position_r2"]))
        record["reproduced_probe_velocity_r2_abs_difference"] = abs(float(record["reproduced_test_velocity_r2"]) - float(original_probe["velocity_r2"]))
    raw_claims = audit_v1_raw_records()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "v1-checkpoint-audit.csv", audit_rows)
    write_csv(output_dir / "v1-gradient-diagnostics.csv", gradient_rows)
    write_csv(output_dir / "v1-action-triplets-replayed.csv", action_rows)
    write_json(output_dir / "v1-raw-evidence-audit.json", raw_claims)
    summary = {
        "protocol_version": V2_VERSION,
        "source_protocol": "visual-jepa-decision-study-v1",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_count": len(audit_rows),
        "intervention_seed_count": 5,
        "intervention_starts": len(starts),
        "status": "complete",
        "original_evidence_modified": False,
        "raw_records": raw_claims,
    }
    write_json(output_dir / "v1-audit-summary.json", summary)
    return summary


def run_cpu_smoke(output_dir: Path) -> dict[str, object]:
    """Run a small CPU-only pipeline smoke that is never used for selection."""
    output_dir = require_drive_output(output_dir)
    if (output_dir / "smoke-summary.json").exists():
        raise FileExistsError(f"smoke output already exists: {output_dir}")
    device = select_device("cpu")
    rows = []
    histories = []
    gradients = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_seed, model_seed in ((9601, 9601), (9602, 9602)):
        data = collect_random_transitions(data_seed, episodes=6)
        for method, objective in TRAIN_METHODS:
            record, _fitted, history, gradient = fit_seed_pair(
                method,
                objective,
                data,
                model_seed,
                data_seed,
                device,
                output_dir,
                epochs=1,
                batch_size=128,
                train_episode_split=4,
                stage="smoke",
            )
            rows.append(record)
            histories.extend(history)
            gradients.extend(gradient)
            del _fitted
        validation = np.flatnonzero(data.episode_ids >= 4)
        chosen = validation[:12]
        for method, objective in TRAIN_METHODS:
            model, probe = load_fitted_model(output_dir, method, objective, model_seed, device)
            rows.extend(evaluate_action_triplets(method, objective, model_seed, data_seed, data, chosen, device, model=model, probe=probe, stage="smoke"))
            del model
        del data
    write_csv(output_dir / "smoke-training.csv", [row for row in rows if isinstance(row, Mapping) and row.get("stage") == "smoke" and row.get("status") == "complete"])
    write_csv(output_dir / "smoke-action-triplets.csv", [row for row in rows if isinstance(row, Mapping) and row.get("start_id", "").startswith("smoke-")])
    write_csv(output_dir / "smoke-training-history.csv", histories)
    write_csv(output_dir / "smoke-gradient-diagnostics.csv", gradients)
    summary = {
        "protocol_version": V2_VERSION,
        "stage": "smoke",
        "status": "complete",
        "device": "cpu",
        "data_seeds": (9601, 9602),
        "model_seeds": (9601, 9602),
        "fit_count": len(TRAIN_METHODS) * 2,
        "epochs": 1,
        "validation_contexts_per_fit": int(len(chosen)),
        "selection_eligible": False,
        "heldout_open": False,
        "planning_open": False,
    }
    write_json(output_dir / "smoke-summary.json", summary)
    return summary


def starts_as_transition_data(starts: Sequence[VisualStart]) -> TransitionData:
    if not starts:
        raise ValueError("at least one visual start is required")
    return TransitionData(
        frames=np.stack([
            np.stack((start.previous_frame, start.current_frame, start.current_frame))
            for start in starts
        ]),
        actions=np.zeros(len(starts), dtype=np.float32),
        states=np.stack([start.state for start in starts]).astype(np.float32),
        next_states=np.stack([mountaincar_step(start.state, 0.0) for start in starts]),
        episode_ids=np.zeros(len(starts), dtype=np.int16),
    )


def pair_context_tensor(previous: np.ndarray, current: np.ndarray, device: torch.device) -> torch.Tensor:
    pair = np.concatenate((previous, current), axis=-1).transpose(2, 0, 1).copy()[None]
    return torch.from_numpy(pair).to(device=device, dtype=torch.float32).div_(255.0)


def fit_latent_ood_reference(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("training embeddings must be a finite matrix with at least two rows")
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance) + np.eye(values.shape[1]) * 1e-6
    inverse = np.linalg.pinv(covariance, hermitian=True)
    delta = values - mean
    distances = np.einsum("bi,ij,bj->b", delta, inverse, delta)
    threshold = float(np.sqrt(np.quantile(distances, 0.99)))
    return mean.astype(np.float64), inverse.astype(np.float64), threshold


def _mahalanobis(value: np.ndarray, reference: tuple[np.ndarray, np.ndarray, float]) -> float:
    mean, inverse, _threshold = reference
    delta = np.asarray(value, dtype=np.float64) - mean
    return float(np.sqrt(max(0.0, delta @ inverse @ delta)))


def evaluate_rollout(
    method: str,
    objective: str,
    model_seed: int,
    data_seed: int,
    test_seed: int,
    start: VisualStart,
    actions: np.ndarray,
    device: torch.device,
    *,
    model: nn.Module | None = None,
    probe: Mapping[str, object] | None = None,
    latent_reference: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> list[dict[str, object]]:
    horizons = (1, 5, 10, 20, 60)
    if actions.shape != (60,) or not np.isfinite(actions).all():
        raise ValueError("rollout actions must be a finite 60-step vector")
    if method not in {"exact", "handcrafted"} and model is None:
        raise ValueError(f"{method} rollout requires a loaded model")
    if method == "jepa" and (probe is None or latent_reference is None):
        raise ValueError("JEPA rollout requires its state probe and training latent reference")
    actual_state = np.asarray(start.state, dtype=np.float32).copy()
    predicted_state = actual_state.copy()
    if method == "handcrafted":
        predicted_state = visual_state(start.previous_frame, start.current_frame)
    actual_current = start.current_frame.copy()
    predicted_previous = start.previous_frame.copy()
    predicted_current = start.current_frame.copy()
    predicted_latent: torch.Tensor | None = None
    if method == "jepa":
        with torch.no_grad():
            predicted_latent = model.encoder(pair_context_tensor(predicted_previous, predicted_current, device))
    prior_position_error = 0.0
    prior_velocity_error = 0.0
    clipped_count = 0
    saturated_count = 0
    ood_count = 0
    records: list[dict[str, object]] = []
    completed_steps = 0
    termination_step: int | None = None
    prior_latent_error = 0.0

    for step_index, action in enumerate(actions, start=1):
        actual_next = mountaincar_step(actual_state, float(action))
        latent_prediction = math.nan
        latent_ood = math.nan
        raw_prediction = predicted_state.copy()
        if method == "exact":
            raw_prediction = actual_next.copy()
        elif method == "handcrafted":
            raw_prediction = mountaincar_step(predicted_state, float(action))
        elif method == "state_model":
            action_tensor = torch.as_tensor([action], dtype=torch.float32, device=device)
            with torch.no_grad():
                raw_tensor = model(torch.as_tensor(predicted_state[None], dtype=torch.float32, device=device), action_tensor)
            raw_prediction = raw_tensor[0].detach().cpu().numpy()
        elif method == "supervised_visual":
            action_tensor = torch.as_tensor([action], dtype=torch.float32, device=device)
            visual_input = pair_context_tensor(predicted_previous, predicted_current, device)
            with torch.no_grad():
                raw_tensor = model(visual_input, action_tensor)
            raw_prediction = raw_tensor[0].detach().cpu().numpy()
        elif method == "jepa":
            action_tensor = torch.as_tensor([action], dtype=torch.float32, device=device)
            with torch.no_grad():
                predicted_latent = model.predictor(predicted_latent, action_tensor)
                raw_prediction = apply_probe_numpy(predicted_latent.detach().cpu().numpy(), probe)[0]
                true_next_frame = resize_frame(render_frame(float(actual_next[0])))
                true_latent = model.encoder(pair_context_tensor(actual_current, true_next_frame, device))
            latent_array = predicted_latent[0].detach().cpu().numpy()
            actual_latent = true_latent[0].detach().cpu().numpy()
            latent_prediction = float(np.linalg.norm(latent_array - actual_latent))
            latent_ood = float(_mahalanobis(latent_array, latent_reference))

        if not np.isfinite(raw_prediction).all():
            raise FloatingPointError(f"{method} produced non-finite state at step {step_index}")
        clipped, clip_mask = unclipped_state_diagnostics(torch.as_tensor(raw_prediction[None], dtype=torch.float32))
        next_predicted = clipped[0].cpu().numpy()
        clipped_count += int(bool(clip_mask.any().item()))
        saturated_count += int(bool(
            next_predicted[0] <= MIN_POSITION
            or next_predicted[0] >= MAX_POSITION
            or abs(next_predicted[1]) >= MAX_SPEED - 1e-7
        ))
        if method == "jepa" and latent_reference is not None:
            ood_count += int(latent_ood > latent_reference[2])
        completed_steps += 1
        actual_state = actual_next
        predicted_state = next_predicted
        true_next_frame = resize_frame(render_frame(float(actual_next[0])))
        actual_current = true_next_frame
        if method == "supervised_visual":
            predicted_frame = resize_frame(render_frame(float(predicted_state[0])))
            predicted_previous, predicted_current = predicted_current, predicted_frame
        position_error = abs(float(predicted_state[0]) - float(actual_state[0]))
        velocity_error = abs(float(predicted_state[1]) - float(actual_state[1]))
        if step_index in horizons:
            records.append(
                {
                    "protocol_version": V2_VERSION,
                    "stage": "heldout",
                    "method": method,
                    "objective": objective,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "test_seed": test_seed,
                    "start_id": start.start_id,
                    "horizon": step_index,
                    "status": "complete",
                    "actual_position": float(actual_state[0]),
                    "actual_velocity": float(actual_state[1]),
                    "predicted_position": float(predicted_state[0]),
                    "predicted_velocity": float(predicted_state[1]),
                    "predicted_position_raw": float(raw_prediction[0]),
                    "predicted_velocity_raw": float(raw_prediction[1]),
                    "position_abs_error": position_error,
                    "velocity_abs_error": velocity_error,
                    "position_error_growth_from_prior_horizon": position_error - prior_position_error,
                    "velocity_error_growth_from_prior_horizon": velocity_error - prior_velocity_error,
                    "latent_prediction_error": latent_prediction,
                    "latent_error_growth_from_prior_horizon": latent_prediction - prior_latent_error
                    if math.isfinite(latent_prediction) else math.nan,
                    "clipped_steps_fraction": clipped_count / completed_steps,
                    "saturated_steps_fraction": saturated_count / completed_steps,
                    "latent_ood_distance": latent_ood,
                    "latent_ood_fraction": ood_count / completed_steps if method == "jepa" else math.nan,
                    "position_clipped": bool(clip_mask[0, 0].item()),
                    "velocity_clipped": bool(clip_mask[0, 1].item()),
                    "predicted_state_saturated": bool(saturated_count > 0 and (
                        next_predicted[0] <= MIN_POSITION
                        or next_predicted[0] >= MAX_POSITION
                        or abs(next_predicted[1]) >= MAX_SPEED - 1e-7
                    )),
                    "action_sequence_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                }
            )
            prior_position_error = position_error
            prior_velocity_error = velocity_error
            if math.isfinite(latent_prediction):
                prior_latent_error = latent_prediction
        if actual_state[0] >= 0.45:
            termination_step = step_index
            break

    if termination_step is not None:
        present = {int(row["horizon"]) for row in records}
        for horizon in horizons:
            if horizon in present:
                continue
            records.append(
                {
                    "protocol_version": V2_VERSION,
                    "stage": "heldout",
                    "method": method,
                    "objective": objective,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "test_seed": test_seed,
                    "start_id": start.start_id,
                    "horizon": horizon,
                    "status": "unsupported",
                    "error": f"environment terminated at step {termination_step}",
                    "actual_position": math.nan,
                    "actual_velocity": math.nan,
                    "predicted_position": math.nan,
                    "predicted_velocity": math.nan,
                }
            )
    return sorted(records, key=lambda row: int(row["horizon"]))


def _truthy(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_heldout_stage(
    validation_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, object]:
    validation_dir = require_drive_output(validation_dir)
    output_dir = require_drive_output(output_dir)
    if (output_dir / "heldout-summary.json").exists():
        raise FileExistsError(f"heldout output already exists; use a new output directory: {output_dir}")
    validation_summary_path = validation_dir / "validation-summary.json"
    if not validation_summary_path.is_file():
        raise FileNotFoundError(f"validation summary is missing: {validation_summary_path}")
    validation_summary = json.loads(validation_summary_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not validation_summary.get("heldout_open") or not validation_summary.get("selected_objective"):
        summary = {
            "protocol_version": V2_VERSION,
            "stage": "heldout",
            "status": "closed_by_validation",
            "test_seed_groups_opened": False,
            "planning_open": False,
            "reason": validation_summary.get("failure_reason", "validation gate did not open held-out evaluation"),
        }
        write_json(output_dir / "heldout-summary.json", summary)
        return summary

    verify_protocol_lock()
    device = select_device(device_name)
    objective = str(validation_summary["selected_objective"])
    validation_rows = _read_csv(validation_dir / "validation-action-summary.csv")
    selected_pairs = [
        (int(row["model_seed"]), int(row["data_seed"]))
        for row in validation_rows
        if row.get("method") == "jepa"
        and row.get("objective") == objective
        and _truthy(row.get("validation_pass"))
    ]
    if len(selected_pairs) < 2:
        raise RuntimeError("selected JEPA objective has fewer than two passing validation model/data pairs")
    starts = collect_intervention_starts(TEST_DATA_SEEDS, steps=(40, 90, 140, 190))
    if len(starts) != 20:
        raise RuntimeError(f"expected 20 held-out intervention starts, got {len(starts)}")
    test_start_data = {
        seed: starts_as_transition_data([start for start in starts if start.seed == seed])
        for seed in TEST_DATA_SEEDS
    }
    action_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    pair_summaries: list[dict[str, object]] = []
    rollout_references = []

    # The exact and handcrafted controls are evaluated once on the shared test starts.
    for test_seed in TEST_DATA_SEEDS:
        seed_starts = [start for start in starts if start.seed == test_seed]
        data = test_start_data[test_seed]
        action_rows.extend(evaluate_action_triplets(
            "exact", "simulator_oracle", 0, test_seed, data,
            np.arange(len(seed_starts)), device, stage="heldout",
            start_ids=[start.start_id for start in seed_starts],
        ))
        action_rows.extend(evaluate_action_triplets(
            "handcrafted", "visual_state_plus_oracle", 0, test_seed, data,
            np.arange(len(seed_starts)), device, stage="heldout",
            start_ids=[start.start_id for start in seed_starts],
        ))
        for start in seed_starts:
            sequence_seed = stable_seed(V2_VERSION, "test-actions", test_seed, start.start_id)
            actions = np.random.default_rng(sequence_seed).uniform(-1.0, 1.0, size=60).astype(np.float32)
            for method in ("exact", "handcrafted"):
                rollout_rows.extend(evaluate_rollout(
                    method, "simulator_oracle" if method == "exact" else "visual_state_plus_oracle",
                    0, 0, test_seed, start, actions, device,
                ))

    # Evaluate each validation-passing paired model on the same independent test contexts.
    for model_seed, data_seed in selected_pairs:
        trained_data = collect_random_transitions(data_seed, episodes=20)
        train_indices, _validation_indices = split_episode_indices(trained_data.episode_ids)
        jepa, probe = load_fitted_model(validation_dir, "jepa", objective, model_seed, device)
        supervised, _supervised_probe = load_fitted_model(validation_dir, "supervised_visual", "supervised_state_mse", model_seed, device)
        state_model, _state_probe = load_fitted_model(validation_dir, "state_model", "privileged_state_mse", model_seed, device)
        train_embeddings = collect_embeddings(jepa, trained_data, train_indices, device)
        latent_reference = fit_latent_ood_reference(train_embeddings)
        rollout_references.append({
            "model_seed": model_seed,
            "data_seed": data_seed,
            "latent_training_rows": len(train_embeddings),
            "latent_mahalanobis_99th_percentile": latent_reference[2],
        })
        try:
            pair_action_rows = []
            for test_seed in TEST_DATA_SEEDS:
                seed_starts = [start for start in starts if start.seed == test_seed]
                data = test_start_data[test_seed]
                row_range = np.arange(len(seed_starts))
                for method, model, fitted_probe, method_objective in (
                    ("jepa", jepa, probe, objective),
                    ("supervised_visual", supervised, _supervised_probe, "supervised_state_mse"),
                    ("state_model", state_model, None, "privileged_state_mse"),
                ):
                    new_rows = evaluate_action_triplets(
                        method, method_objective, model_seed, test_seed, data,
                        row_range, device, model=model, probe=fitted_probe,
                        stage="heldout", start_ids=[start.start_id for start in seed_starts],
                    )
                    for row in new_rows:
                        row["test_seed"] = test_seed
                    action_rows.extend(new_rows)
                    pair_action_rows.extend(new_rows)
                for start in seed_starts:
                    sequence_seed = stable_seed(V2_VERSION, "test-actions", test_seed, start.start_id)
                    actions = np.random.default_rng(sequence_seed).uniform(-1.0, 1.0, size=60).astype(np.float32)
                    for method, model, fitted_probe, method_objective in (
                        ("jepa", jepa, probe, objective),
                        ("supervised_visual", supervised, _supervised_probe, "supervised_state_mse"),
                        ("state_model", state_model, None, "privileged_state_mse"),
                    ):
                        rollout_rows.extend(evaluate_rollout(
                            method, method_objective, model_seed, data_seed, test_seed,
                            start, actions, device, model=model, probe=fitted_probe,
                            latent_reference=latent_reference if method == "jepa" else None,
                        ))
            # Combine all five test seeds before applying the seed-level held-out gate.
            combined = action_effect_metrics([row for row in pair_action_rows if row["method"] == "jepa"])
            jepa_state_mae = {
                coordinate: float(np.mean([
                    abs(float(row[f"predicted_{coordinate}_raw"]) - float(row[f"actual_{coordinate}"]))
                    for row in pair_action_rows if row["method"] == "jepa"
                ]))
                for coordinate in ("position", "velocity")
            }
            action_gate = (
                float(combined["position_median_normalized_contrast_error"]) <= 0.25
                and float(combined["velocity_median_normalized_contrast_error"]) <= 0.25
                and float(combined["position_order_accuracy"]) >= 0.80
                and float(combined["velocity_order_accuracy"]) >= 0.80
                and int(combined["position_valid_starts"]) >= 12
                and int(combined["velocity_valid_starts"]) >= 12
            )
            state_gate = jepa_state_mae["position"] <= 0.02 and jepa_state_mae["velocity"] <= 0.01
            jepa_rollouts = [
                row for row in rollout_rows
                if row["method"] == "jepa" and int(row["model_seed"]) == model_seed
            ]
            rollout_gates = []
            for horizon, position_limit, velocity_limit in (
                (1, .02, .01), (5, .05, .02), (10, .10, .03), (20, .20, .05), (60, .40, .07)
            ):
                supported = [row for row in jepa_rollouts if int(row["horizon"]) == horizon and row["status"] == "complete"]
                position_mae = float(np.mean([float(row["position_abs_error"]) for row in supported])) if supported else math.inf
                velocity_mae = float(np.mean([float(row["velocity_abs_error"]) for row in supported])) if supported else math.inf
                rollout_gates.append({
                    "horizon": horizon,
                    "supported_starts": len(supported),
                    "position_mae": position_mae,
                    "velocity_mae": velocity_mae,
                    "pass": position_mae <= position_limit and velocity_mae <= velocity_limit,
                })
            rollout_pass = all(bool(row["pass"]) for row in rollout_gates)
            pair_summaries.append({
                "protocol_version": V2_VERSION,
                "stage": "heldout",
                "method": "jepa",
                "objective": objective,
                "model_seed": model_seed,
                "data_seed": data_seed,
                "status": "complete",
                "one_step_position_mae": jepa_state_mae["position"],
                "one_step_velocity_mae": jepa_state_mae["velocity"],
                "position_median_normalized_contrast_error": combined["position_median_normalized_contrast_error"],
                "velocity_median_normalized_contrast_error": combined["velocity_median_normalized_contrast_error"],
                "position_order_accuracy": combined["position_order_accuracy"],
                "velocity_order_accuracy": combined["velocity_order_accuracy"],
                "one_step_state_pass": state_gate,
                "action_fidelity_pass": action_gate,
                "rollout_gates": rollout_gates,
                "rollout_pass": rollout_pass,
                "heldout_pass": bool(state_gate and action_gate and rollout_pass),
                "test_contexts": sum(row["method"] == "jepa" for row in pair_action_rows) // 3,
            })
        finally:
            del jepa, supervised, state_model, trained_data
            if device.type == "cuda":
                torch.cuda.empty_cache()
    validate_rollout_records(rollout_rows)
    planning_open = sum(bool(row["heldout_pass"]) for row in pair_summaries) >= 2
    summary = {
        "protocol_version": V2_VERSION,
        "stage": "heldout",
        "status": "complete",
        "selected_objective": objective,
        "validation_passed_model_data_pairs": selected_pairs,
        "test_seed_groups_opened": True,
        "test_seeds": TEST_DATA_SEEDS,
        "test_intervention_starts": len(starts),
        "test_action_sequences": len(starts),
        "model_pair_results": pair_summaries,
        "latent_ood_references": rollout_references,
        "planning_open": planning_open,
        "planning_closed_reason": "fewer than two model/data pairs passed held-out one-step and rollout gates" if not planning_open else "gates passed",
    }
    write_csv(output_dir / "heldout-action-triplets.csv", action_rows)
    write_csv(output_dir / "heldout-rollouts.csv", rollout_rows)
    write_csv(output_dir / "heldout-pair-gates.csv", pair_summaries)
    write_json(output_dir / "heldout-latent-references.json", {"references": rollout_references})
    write_json(output_dir / "heldout-summary.json", summary)
    return summary


def _render_state_frames(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float32)
    previous = np.stack([
        resize_frame(render_frame(float(np.clip(state[0] - state[1], MIN_POSITION, MAX_POSITION))))
        for state in values
    ])
    current = np.stack([
        resize_frame(render_frame(float(np.clip(state[0], MIN_POSITION, MAX_POSITION))))
        for state in values
    ])
    return previous, current


def _render_positions(positions: np.ndarray) -> np.ndarray:
    return np.stack([
        resize_frame(render_frame(float(np.clip(position, MIN_POSITION, MAX_POSITION))))
        for position in np.asarray(positions).reshape(-1)
    ])


def _visual_start_state(start: VisualStart) -> np.ndarray:
    """Estimate the initial planner state from the rendered context only."""
    estimate = np.asarray(visual_state(start.previous_frame, start.current_frame), dtype=np.float32)
    if estimate.shape != (2,) or not np.isfinite(estimate).all():
        raise ValueError("rendered start frames must produce one finite visual state estimate")
    return estimate


def _supervised_visual_state_step(
    model: nn.Module,
    states: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    *,
    previous_frames: np.ndarray | None = None,
    current_frames: np.ndarray | None = None,
) -> np.ndarray:
    if previous_frames is None or current_frames is None:
        previous_frames, current_frames = _render_state_frames(states)
    observations = np.concatenate((previous_frames, current_frames), axis=-1).transpose(0, 3, 1, 2).copy()
    observation_tensor = torch.from_numpy(observations).to(device=device, dtype=torch.float32).div_(255.0)
    action_tensor = torch.as_tensor(actions, device=device, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(observation_tensor, action_tensor).detach().float()
    if not bool(torch.isfinite(prediction).all().item()):
        raise FloatingPointError("supervised visual dynamics produced a non-finite candidate state")
    clipped, _mask = unclipped_state_diagnostics(prediction)
    return clipped.cpu().numpy()


def _score_supervised_visual_candidates(
    model: nn.Module,
    start: VisualStart,
    candidates: np.ndarray,
    config: Any,
    device: torch.device,
) -> np.ndarray:
    count = len(candidates)
    estimated_state = _visual_start_state(start)
    states = np.repeat(estimated_state[None, :], count, axis=0)
    previous_frames = np.repeat(start.previous_frame[None], count, axis=0)
    current_frames = np.repeat(start.current_frame[None], count, axis=0)
    active = np.ones(count, dtype=bool)
    returns = np.zeros(count, dtype=np.float64)
    for step in range(config.horizon_steps):
        action = candidates[:, min(step // config.replan_interval, candidates.shape[1] - 1)].astype(np.float32)
        next_states = _supervised_visual_state_step(
            model, states, action, device,
            previous_frames=previous_frames,
            current_frames=current_frames,
        )
        if config.score_name == "cumulative_return":
            returns += ((next_states[:, 0] >= 0.45) * 100.0 - 0.1 * np.square(action)) * active
        states = np.where(active[:, None], next_states, states)
        active &= states[:, 0] < 0.45
        next_current = _render_positions(states[:, 0])
        previous_frames = np.where(active[:, None, None, None], current_frames, previous_frames)
        current_frames = np.where(active[:, None, None, None], next_current, current_frames)
    if config.score_name == "terminal_potential":
        return -((states[:, 0] - 0.45) ** 2 + 0.2 * states[:, 1] ** 2)
    return returns


def _supervised_visual_prior_mean(
    model: nn.Module,
    start: VisualStart,
    config: Any,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    from jepa_tutorial.decision_study import momentum_action

    states = _visual_start_state(start)[None, :]
    previous = start.previous_frame[None]
    current = start.current_frame[None]
    actions = np.empty(config.action_blocks, dtype=np.float32)
    transition_calls = 0
    for block in range(config.action_blocks):
        action = float(momentum_action(states)[0])
        actions[block] = action
        block_steps = min(
            config.replan_interval,
            config.horizon_steps - block * config.replan_interval,
        )
        for _ in range(block_steps):
            states = _supervised_visual_state_step(
                model,
                states,
                np.asarray([action], dtype=np.float32),
                device,
                previous_frames=previous,
                current_frames=current,
            )
            next_current = _render_positions(states[:, 0])
            previous, current = current, next_current
            transition_calls += 1
    return np.clip(actions, -1.0, 1.0), transition_calls


def _planner_action_supervised_visual(
    model: nn.Module,
    start: VisualStart,
    config: Any,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[float, int, int]:
    from jepa_tutorial.decision_study import cem_proposal

    prior = None
    prior_calls = 0
    if config.initialization == "momentum_prior":
        prior, prior_calls = _supervised_visual_prior_mean(model, start, config, device)
    proposal = cem_proposal(
        config,
        lambda sequences: _score_supervised_visual_candidates(model, start, sequences, config, device),
        rng,
        prior_mean=prior,
    )
    return float(proposal.action_blocks[0]), proposal.transition_calls + prior_calls, proposal.forward_batches


def _run_supervised_visual_episode(
    model: nn.Module,
    model_seed: int,
    data_seed: int,
    eval_seed: int,
    config: Any,
    device: torch.device,
) -> dict[str, object]:
    import run_visual_jepa_decision_study as v1_study

    env = v1_study.make_env()
    observation, _ = env.reset(seed=eval_seed)
    previous = resize_frame(render_frame(float(observation[0])))
    current = previous.copy()
    rng = np.random.default_rng(stable_seed(V2_VERSION, "supervised-visual-cem", model_seed, eval_seed, config.initialization))
    total_return = 0.0
    calls = batches = steps = 0
    planning_seconds = 0.0
    success = False
    action = 0.0
    started = time.perf_counter()
    try:
        for step in range(300):
            start = VisualStart(
                start_id=f"planning-{eval_seed}",
                seed=eval_seed,
                step=step,
                state=np.asarray(env.unwrapped.state, dtype=np.float32).copy(),
                previous_frame=previous,
                current_frame=current,
            )
            if step % config.replan_interval == 0:
                started_plan = time.perf_counter()
                action, case_calls, case_batches = _planner_action_supervised_visual(
                    model, start, config, rng, device
                )
                planning_seconds += time.perf_counter() - started_plan
                calls += case_calls
                batches += case_batches
            observation, reward, terminated, truncated, _ = env.step(np.asarray([action], dtype=np.float32))
            total_return += float(reward)
            steps = step + 1
            next_frame = resize_frame(render_frame(float(observation[0])))
            previous, current = current, next_frame
            if terminated:
                success = True
                break
            if truncated:
                break
    finally:
        env.close()
    return {
        "method": "supervised_visual_cem",
        "model_seed": model_seed,
        "data_seed": data_seed,
        "eval_seed": eval_seed,
        "initialization": config.initialization,
        "privileged_state": False,
        "status": "complete",
        "success": int(success),
        "return": total_return,
        "steps": steps,
        "transition_calls": calls,
        "forward_batches": batches,
        "planning_seconds": planning_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        **config.as_dict(),
    }


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count < 1:
        return math.nan, math.nan
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_planning_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row.get("model_seed", "")), str(row["initialization"]))].append(row)
    summaries = []
    for (method, model_seed, initialization), group in sorted(grouped.items()):
        successes = sum(int(row["success"]) for row in group)
        low, high = _wilson_interval(successes, len(group))
        returns = np.asarray([float(row["return"]) for row in group], dtype=np.float64)
        rng = np.random.default_rng(stable_seed(V2_VERSION, "planning-return-bootstrap", method, model_seed, initialization))
        bootstrap_means = np.mean(rng.choice(returns, size=(2000, len(returns)), replace=True), axis=1)
        summaries.append({
            "method": method,
            "model_seed": model_seed,
            "initialization": initialization,
            "starts": len(group),
            "successes": successes,
            "success_rate": successes / len(group),
            "success_wilson_95_low": low,
            "success_wilson_95_high": high,
            "mean_return": float(returns.mean()),
            "return_sd": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
            "return_bootstrap_95_low": float(np.quantile(bootstrap_means, 0.025)),
            "return_bootstrap_95_high": float(np.quantile(bootstrap_means, 0.975)),
            "mean_steps": float(np.mean([float(row["steps"]) for row in group])),
            "mean_transition_calls": float(np.mean([float(row["transition_calls"]) for row in group])),
            "mean_planning_seconds": float(np.mean([float(row["planning_seconds"]) for row in group])),
            "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in group])),
            "devices": sorted({str(row["device"]) for row in group}),
            "hardware_costs_are_comparable": False,
        })
    return summaries


def run_planning_stage(
    validation_dir: Path,
    heldout_dir: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, object]:
    import run_visual_jepa_decision_study as v1_study

    from jepa_tutorial.decision_study import FINAL_TRANSITION_CALL_LIMIT, PlannerConfig

    validation_dir = require_drive_output(validation_dir)
    heldout_dir = require_drive_output(heldout_dir)
    output_dir = require_drive_output(output_dir)
    if (output_dir / "planning-summary.json").exists():
        raise FileExistsError(f"planning output already exists; use a new output directory: {output_dir}")
    heldout_summary = json.loads((heldout_dir / "heldout-summary.json").read_text(encoding="utf-8"))
    if not heldout_summary.get("planning_open"):
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "protocol_version": V2_VERSION,
            "stage": "planning",
            "status": "closed_by_heldout_gate",
            "planning_open": False,
            "reason": heldout_summary.get("planning_closed_reason", "held-out rollout gates did not pass"),
        }
        write_json(output_dir / "planning-summary.json", summary)
        return summary
    verify_protocol_lock()
    objective = str(heldout_summary["selected_objective"])
    model_pairs = [
        (int(row["model_seed"]), int(row["data_seed"]))
        for row in heldout_summary["model_pair_results"]
        if row.get("heldout_pass")
    ]
    if len(model_pairs) < 2:
        raise RuntimeError("planning gate opened with fewer than two passing model/data pairs")
    device = select_device(device_name)
    configurations = [
        PlannerConfig(60, 48, 8, 4, "cumulative_return", initialization, replan_interval=5)
        for initialization in ("uniform", "momentum_prior")
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    transition_calls = 0
    status = "complete"
    stop_reason = ""
    # Exact planner and direct visual-momentum control are shared across paired model seeds.
    for eval_seed in PLANNING_SEEDS:
        for config in configurations:
            result = v1_study.run_episode_exact(eval_seed, config, method="v2_exact_cem")
            result["method"] = "exact_cem"
            result["model_seed"] = ""
            result["data_seed"] = ""
            result["role"] = "privileged exact dynamics"
            if transition_calls + int(result["transition_calls"]) > FINAL_TRANSITION_CALL_LIMIT:
                status, stop_reason = "budget_stopped", "750 million transition-call ceiling reached"
                break
            rows.append(result)
            transition_calls += int(result["transition_calls"])
        if status != "complete":
            break
        direct = v1_study.run_direct_episode(eval_seed)
        direct.update({
            "method": "direct_visual_momentum",
            "initialization": "direct",
            "data_seed": "",
            "role": "direct visual controller and prior-only action arm",
            "prior_only_equivalent": True,
        })
        rows.append(direct)
    if status == "complete":
        for model_seed, data_seed in model_pairs:
            state_model, _ = load_fitted_model(validation_dir, "state_model", "privileged_state_mse", model_seed, device)
            supervised, _ = load_fitted_model(validation_dir, "supervised_visual", "supervised_state_mse", model_seed, device)
            jepa, probe = load_fitted_model(validation_dir, "jepa", objective, model_seed, device)
            try:
                for eval_seed in PLANNING_SEEDS:
                    for config in configurations:
                        for family, model, model_probe, output_name, role in (
                            ("state_model", state_model, None, "state_model_cem", "privileged learned state dynamics"),
                            ("visual", jepa, probe, "jepa_cem", "validated action-conditioned JEPA"),
                        ):
                            result = v1_study.run_episode_model(
                                family, model, model_probe, model_seed, eval_seed, config, device
                            )
                            result.update({"method": output_name, "data_seed": data_seed, "role": role})
                            if transition_calls + int(result["transition_calls"]) > FINAL_TRANSITION_CALL_LIMIT:
                                status, stop_reason = "budget_stopped", "750 million transition-call ceiling reached"
                                break
                            rows.append(result)
                            transition_calls += int(result["transition_calls"])
                        if status != "complete":
                            break
                        result = _run_supervised_visual_episode(
                            supervised, model_seed, data_seed, eval_seed, config, device
                        )
                        result["role"] = "compact supervised visual dynamics control"
                        if transition_calls + int(result["transition_calls"]) > FINAL_TRANSITION_CALL_LIMIT:
                            status, stop_reason = "budget_stopped", "750 million transition-call ceiling reached"
                            break
                        rows.append(result)
                        transition_calls += int(result["transition_calls"])
                    if status != "complete":
                        break
            finally:
                del state_model, supervised, jepa
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if status != "complete":
                break
    summaries = summarize_planning_rows(rows) if rows else []
    write_csv(output_dir / "planning-episodes.csv", rows)
    write_csv(output_dir / "planning-arm-summary.csv", summaries)
    summary = {
        "protocol_version": V2_VERSION,
        "stage": "planning",
        "status": status,
        "planning_open": True,
        "objective": objective,
        "model_data_pairs": model_pairs,
        "evaluation_start_seeds": PLANNING_SEEDS,
        "rows": len(rows),
        "transition_calls": transition_calls,
        "transition_call_limit": FINAL_TRANSITION_CALL_LIMIT,
        "stop_reason": stop_reason,
        "direct_policy_baseline": "not trained; outside the 15-fit preregistered training budget",
        "prior_only_arm": "same fixed direct visual-momentum controller; not an independent planner",
        "hardware_costs_comparable": False,
        "hardware_limit": "per-row device, elapsed time, and planning time retained",
    }
    write_json(output_dir / "planning-summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock", help="write protocol and implementation hashes before fitting")
    lock.add_argument("--confirm-pretraining", action="store_true", help="explicitly confirm no v2 model fit has started")
    subparsers.add_parser("verify-lock", help="verify frozen protocol and implementation hashes")
    audit = subparsers.add_parser("audit-v1", help="replay diagnostics from retained v1 checkpoints")
    audit.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "v1-audit")
    audit.add_argument("--checkpoint-dir", type=Path, default=Path("/mnt/e/codex-research/jepa-world-models-tutorial/visual-jepa-decision-study-v1/study-run-final-6/checkpoints"))
    audit.add_argument("--device", default="cuda")
    smoke = subparsers.add_parser("smoke", help="run a CPU-only non-evaluation smoke")
    smoke.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "smoke")
    heldout = subparsers.add_parser("heldout", help="open held-out evaluation only after its validation gate")
    heldout.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validation")
    heldout.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validation" / "heldout")
    heldout.add_argument("--device", default="cuda")
    planning = subparsers.add_parser("planning", help="run planning only after held-out state and rollout gates pass")
    planning.add_argument("--validation-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validation")
    planning.add_argument("--heldout-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validation" / "heldout")
    planning.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validation" / "planning")
    planning.add_argument("--device", default="cuda")
    validation = subparsers.add_parser("validate", help="run the preregistered v2 validation fits")
    validation.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "validation")
    validation.add_argument("--device", default="cuda")
    validation.add_argument("--epochs", type=int, default=35)
    validation.add_argument("--batch-size", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "lock":
        if not args.confirm_pretraining:
            raise SystemExit("refusing to create protocol lock without --confirm-pretraining")
        print(json.dumps(create_protocol_lock(), indent=2, sort_keys=True))
        return 0
    if args.command == "verify-lock":
        print(json.dumps(verify_protocol_lock(), indent=2, sort_keys=True))
        return 0
    verify_protocol_lock()
    if args.command == "audit-v1":
        result = run_v1_audit(args.output_dir, args.checkpoint_dir, args.device)
    elif args.command == "smoke":
        result = run_cpu_smoke(args.output_dir)
    elif args.command == "heldout":
        result = run_heldout_stage(args.validation_dir, args.output_dir, args.device)
    elif args.command == "planning":
        result = run_planning_stage(args.validation_dir, args.heldout_dir, args.output_dir, args.device)
    else:
        if not 1 <= args.epochs <= 35 or not 1 <= args.batch_size <= 256:
            raise SystemExit("validation epochs and batch size must stay within the frozen budget")
        result = write_validation_stage(args.output_dir, args.device, epochs=args.epochs, batch_size=args.batch_size)
        heldout_result = run_heldout_stage(args.output_dir, args.output_dir / "heldout", args.device)
        result["heldout_status"] = heldout_result.get("status")
        result["planning_open"] = heldout_result.get("planning_open", False)
        if heldout_result.get("planning_open"):
            result["planning"] = run_planning_stage(
                args.output_dir,
                args.output_dir / "heldout",
                args.output_dir / "planning",
                args.device,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
