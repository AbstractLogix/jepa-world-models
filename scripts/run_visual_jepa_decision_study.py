#!/usr/bin/env python3
"""Run the gated, versioned visual-JEPA decision study.

Heavy outputs belong in JEPA_RUN_OUTPUT_DIR (prefer an E: or D: volume). The
script copies only small, named evidence files into docs/evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from jepa_tutorial.action_conditioned_visual import (  # noqa: E402
    ActionConditionedJEPA,
    ImageReconstructionDynamics,
    StateDynamics,
    embedding_collapse_diagnostics,
    select_device,
)
from jepa_tutorial.decision_study import (  # noqa: E402
    ACTION_COST,
    ACTION_MAX,
    ACTION_MIN,
    DEV_CONFIRMATION_MIN_SUCCESS,
    DEV_MIN_SUCCESS,
    DEV_TRANSITION_CALL_LIMIT,
    FINAL_ORACLE_MIN_SUCCESS,
    FINAL_TRANSITION_CALL_LIMIT,
    GOAL_POSITION,
    MAX_EPISODE_STEPS,
    MAX_POSITION,
    MAX_SPEED,
    MIN_POSITION,
    REPLAN_INTERVAL,
    STUDY_VERSION,
    PlannerConfig,
    cem_proposal,
    gate_passes,
    make_momentum_prior,
    momentum_action,
    mountaincar_reward,
    mountaincar_step,
    oracle_gate_is_valid,
    protocol_lock_is_valid,
    sha256_file,
    stable_seed,
    terminal_potential,
    validate_episode_records,
    validate_intervention_records,
)
from jepa_tutorial.lejepa_loss import lejepa_total_loss  # noqa: E402
from jepa_tutorial.paper_losses import epps_pulley_sigreg_loss  # noqa: E402
from jepa_tutorial.visual_mountaincar import (  # noqa: E402
    TransitionData,
    VisualStart,
    collect_intervention_starts,
    collect_random_transitions,
    context_array,
    make_env,
    render_frame,
    resize_frame,
    visual_state,
)


def scalar_momentum_action(state: np.ndarray) -> float:
    """Apply the shared controller to one state with a stable batch shape."""
    state_batch = np.asarray(state, dtype=np.float32).reshape(1, 2)
    return float(momentum_action(state_batch)[0])


PROTOCOL_PATH = ROOT / "docs/reproductions/protocols/visual-jepa-decision-study-v1.md"
EVIDENCE_DIR = ROOT / "docs/evidence"
DEV_SEEDS = tuple(range(8200, 8210))
CONFIRMATION_SEEDS = tuple(range(8210, 8220))
DATA_SEEDS = (8300, 8301, 8302, 8303, 8304)
MODEL_SEEDS = (7401, 7402, 7403, 7404, 7405)
TEST_SEEDS = (8400, 8401, 8402, 8403, 8404)
INTERVENTION_SEEDS = (8600, 8601, 8602, 8603, 8604)
FINAL_SEEDS = tuple(range(8700, 8720))
SMOKE_SEEDS = (8900, 8901)
TRAIN_EPISODES = 20
TRAIN_EPISODE_SPLIT = 16
EPOCHS = 35
BATCH_SIZE = 256
SIGREG_WEIGHT = 0.1
SIGREG_PROJECTIONS = 128
PROBE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TUNING_HORIZONS = (10, 60, 120, 180)
TUNING_CANDIDATES = 24
TUNING_ELITES = 4
TUNING_UPDATES = 2
MODEL_FAMILIES = ("state_model", "image_reconstruction", "jepa")
METHOD_FIELDS = (
    "method",
    "model_seed",
    "eval_seed",
    "initialization",
    "privileged_state",
    "status",
    "success",
    "return",
    "steps",
    "transition_calls",
    "forward_batches",
    "planning_seconds",
    "elapsed_seconds",
    "device",
    "error",
    "horizon_steps",
    "candidates",
    "elites",
    "updates",
    "score_name",
    "replan_interval",
)
INTERVENTION_FIELDS = (
    "method",
    "model_seed",
    "start_id",
    "seed",
    "step",
    "action",
    "actual_position",
    "actual_velocity",
    "predicted_position",
    "predicted_velocity",
    "position_abs_error",
    "velocity_abs_error",
    "observed_return",
    "predicted_return",
    "status",
    "error",
)


def default_output_dir() -> Path:
    configured = os.environ.get("JEPA_RUN_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    external = Path("/mnt/e/codex-research/jepa-world-models-tutorial/visual-jepa-decision-study-v1")
    return external if external.parent.exists() else Path("/tmp/visual-jepa-decision-study-v1")


def code_hash() -> str:
    paths = (
        Path(__file__),
        ROOT / "src/jepa_tutorial/decision_study.py",
        ROOT / "src/jepa_tutorial/visual_mountaincar.py",
        ROOT / "src/jepa_tutorial/action_conditioned_visual.py",
        ROOT / "src/jepa_tutorial/lejepa_loss.py",
        ROOT / "src/jepa_tutorial/paper_losses.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def retain_evidence(source: Path, name: str) -> Path:
    destination = EVIDENCE_DIR / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def locked_config(lock: Mapping[str, object]) -> PlannerConfig:
    raw = lock.get("planner_config")
    if not isinstance(raw, Mapping):
        raise ValueError("protocol lock has no planner_config object")
    return PlannerConfig(
        horizon_steps=int(raw["horizon_steps"]),
        candidates=int(raw["candidates"]),
        elites=int(raw["elites"]),
        updates=int(raw["updates"]),
        score_name=str(raw["score_name"]),
        initialization=str(raw["initialization"]),
        replan_interval=int(raw["replan_interval"]),
    )


def make_config(
    horizon: int,
    candidates: int,
    elites: int,
    updates: int,
    score_name: str,
    initialization: str,
) -> PlannerConfig:
    return PlannerConfig(
        horizon_steps=horizon,
        candidates=candidates,
        elites=elites,
        updates=updates,
        score_name=score_name,
        initialization=initialization,
        replan_interval=REPLAN_INTERVAL,
    )


def exact_rollout_scores(
    candidates: np.ndarray,
    state: np.ndarray,
    config: PlannerConfig,
) -> np.ndarray:
    states = np.repeat(np.asarray(state, dtype=np.float32)[None, :], len(candidates), axis=0)
    returns = np.zeros(len(candidates), dtype=np.float64)
    active = np.ones(len(candidates), dtype=bool)
    for step in range(config.horizon_steps):
        action = candidates[:, min(step // config.replan_interval, candidates.shape[1] - 1)]
        next_states = mountaincar_step(states, action)
        if config.score_name == "cumulative_return":
            rewards = mountaincar_reward(next_states, action)
            returns += rewards * active
        states = np.where(active[:, None], next_states, states)
        active &= states[:, 0] < GOAL_POSITION
    if config.score_name == "terminal_potential":
        return terminal_potential(states)
    return returns


def exact_momentum_prior(state: np.ndarray, config: PlannerConfig) -> tuple[np.ndarray, int]:
    return make_momentum_prior(state, config, mountaincar_step)


def planner_action_exact(
    state: np.ndarray,
    config: PlannerConfig,
    rng: np.random.Generator,
) -> tuple[float, int, int]:
    prior: np.ndarray | None = None
    prior_calls = 0
    if config.initialization == "momentum_prior":
        prior, prior_calls = exact_momentum_prior(state, config)
    proposal = cem_proposal(
        config,
        lambda sequence: exact_rollout_scores(sequence, state, config),
        rng,
        prior_mean=prior,
    )
    return float(proposal.action_blocks[0]), proposal.transition_calls + prior_calls, proposal.forward_batches


def run_episode_exact(
    eval_seed: int,
    config: PlannerConfig,
    *,
    method: str,
    model_seed: int | str = "",
) -> dict[str, object]:
    env = make_env()
    env.reset(seed=eval_seed)
    rng = np.random.default_rng(stable_seed(STUDY_VERSION, method, model_seed, eval_seed))
    total_return = 0.0
    calls = batches = steps = 0
    planning_seconds = 0.0
    success = False
    action = 0.0
    status, error = "complete", ""
    started = time.perf_counter()
    try:
        for step in range(MAX_EPISODE_STEPS):
            if step % config.replan_interval == 0:
                planning_started = time.perf_counter()
                state = np.asarray(env.unwrapped.state, dtype=np.float32).copy()
                action, case_calls, case_batches = planner_action_exact(state, config, rng)
                planning_seconds += time.perf_counter() - planning_started
                calls += case_calls
                batches += case_batches
            _next_observation, reward, terminated, truncated, _ = env.step(
                np.asarray([action], dtype=np.float32)
            )
            total_return += float(reward)
            steps = step + 1
            if terminated:
                success = True
                break
            if truncated:
                break
    except Exception as exc:  # Preserve a failed case as evidence.
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    row = episode_row(
        method=method,
        model_seed=model_seed,
        eval_seed=eval_seed,
        initialization=config.initialization,
        privileged_state=True,
        status=status,
        success=int(success),
        episode_return=total_return,
        steps=steps,
        calls=calls,
        batches=batches,
        planning_seconds=planning_seconds,
        elapsed_seconds=time.perf_counter() - started,
        device="cpu",
        error=error,
    )
    row.update(config.as_dict())
    return row


def episode_row(
    *,
    method: str,
    model_seed: int | str,
    eval_seed: int,
    initialization: str,
    privileged_state: bool,
    status: str,
    success: int,
    episode_return: float | str,
    steps: int,
    calls: int,
    batches: int,
    planning_seconds: float | str,
    elapsed_seconds: float | str,
    device: str,
    error: str,
) -> dict[str, object]:
    return {
        "method": method,
        "model_seed": str(model_seed),
        "eval_seed": int(eval_seed),
        "initialization": initialization,
        "privileged_state": bool(privileged_state),
        "status": status,
        "success": int(success),
        "return": episode_return,
        "steps": int(steps),
        "transition_calls": int(calls),
        "forward_batches": int(batches),
        "planning_seconds": planning_seconds,
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "error": error,
    }


def development_search(output: Path, device: torch.device) -> None:
    if device.type != "cpu":
        print("Exact-dynamics development search uses vectorized CPU arithmetic.")
    starts = DEV_SEEDS
    rows: list[dict[str, object]] = []
    attempts = 0
    stage_a: list[dict[str, object]] = []
    for horizon in TUNING_HORIZONS:
        for initialization in ("uniform", "momentum_prior"):
            for score_name in ("terminal_potential", "cumulative_return"):
                config = make_config(
                    horizon,
                    TUNING_CANDIDATES,
                    TUNING_ELITES,
                    TUNING_UPDATES,
                    score_name,
                    initialization,
                )
                case_rows = [
                    run_episode_exact(seed, config, method="dev_stage_a") for seed in starts
                ]
                attempts += sum(int(row["transition_calls"]) for row in case_rows)
                successes = sum(int(row["success"]) for row in case_rows)
                mean_return = float(np.mean([float(row["return"]) for row in case_rows]))
                stage = {
                    **config.as_dict(),
                    "stage": "A",
                    "seeds": list(starts),
                    "successes": successes,
                    "total": len(case_rows),
                    "mean_return": mean_return,
                    "transition_calls": sum(int(row["transition_calls"]) for row in case_rows),
                    "status": "complete",
                }
                stage_a.append(stage)
                rows.extend(case_rows)
                atomic_csv(output / "development.csv", rows, METHOD_FIELDS)
                print(
                    f"Stage A H={horizon} init={initialization} score={score_name}: "
                    f"{successes}/10; estimated calls={attempts:,}"
                )
                if attempts > DEV_TRANSITION_CALL_LIMIT:
                    break
            if attempts > DEV_TRANSITION_CALL_LIMIT:
                break
        if attempts > DEV_TRANSITION_CALL_LIMIT:
            break
    dev_record: dict[str, object] = {
        "schema_version": 1,
        "experiment_version": STUDY_VERSION,
        "status": "failed",
        "reason": "development transition-call ceiling exceeded",
        "transition_calls": attempts,
        "transition_call_limit": DEV_TRANSITION_CALL_LIMIT,
        "stage_a": stage_a,
        "stage_b": [],
    }
    if attempts > DEV_TRANSITION_CALL_LIMIT or not stage_a:
        atomic_json(output / "development-gate.json", dev_record)
        retain_evidence(output / "development.csv", "visual-jepa-decision-study-v1-development.csv")
        retain_evidence(output / "development-gate.json", "visual-jepa-decision-study-v1-development-gate.json")
        return

    best_a = min(stage_a, key=development_key)
    stage_b: list[dict[str, object]] = []
    for candidates in (24, 48):
        elites = math.ceil(candidates / 6)
        for updates in (2, 4):
            config = make_config(
                int(best_a["horizon_steps"]),
                candidates,
                elites,
                updates,
                str(best_a["score_name"]),
                str(best_a["initialization"]),
            )
            case_rows = [
                run_episode_exact(seed, config, method="dev_stage_b") for seed in starts[:5]
            ]
            attempts += sum(int(row["transition_calls"]) for row in case_rows)
            stage = {
                **config.as_dict(),
                "stage": "B",
                "seeds": list(starts[:5]),
                "successes": sum(int(row["success"]) for row in case_rows),
                "total": len(case_rows),
                "mean_return": float(np.mean([float(row["return"]) for row in case_rows])),
                "transition_calls": sum(int(row["transition_calls"]) for row in case_rows),
                "status": "complete",
            }
            stage_b.append(stage)
            rows.extend(case_rows)
            atomic_csv(output / "development.csv", rows, METHOD_FIELDS)
            print(f"Stage B C={candidates} updates={updates}: {stage['successes']}/5")
            if attempts > DEV_TRANSITION_CALL_LIMIT:
                break
        if attempts > DEV_TRANSITION_CALL_LIMIT:
            break
    if attempts > DEV_TRANSITION_CALL_LIMIT:
        dev_record.update(
            status="failed",
            reason="development transition-call ceiling exceeded during stage B",
            transition_calls=attempts,
            stage_b=stage_b,
        )
        atomic_json(output / "development-gate.json", dev_record)
        retain_evidence(output / "development.csv", "visual-jepa-decision-study-v1-development.csv")
        retain_evidence(output / "development-gate.json", "visual-jepa-decision-study-v1-development-gate.json")
        return

    best_b = min(stage_b, key=development_key)
    locked = make_config(
        int(best_b["horizon_steps"]),
        int(best_b["candidates"]),
        int(best_b["elites"]),
        int(best_b["updates"]),
        str(best_b["score_name"]),
        str(best_b["initialization"]),
    )
    tuning_rows = [
        run_episode_exact(seed, locked, method="dev_locked_tuning") for seed in starts
    ]
    attempts += sum(int(row["transition_calls"]) for row in tuning_rows)
    rows.extend(tuning_rows)
    atomic_csv(output / "development.csv", rows, METHOD_FIELDS)
    confirmation_rows = [
        run_episode_exact(seed, locked, method="dev_confirmation") for seed in CONFIRMATION_SEEDS
    ]
    attempts += sum(int(row["transition_calls"]) for row in confirmation_rows)
    rows.extend(confirmation_rows)
    atomic_csv(output / "development.csv", rows, METHOD_FIELDS)
    tuning_successes = sum(int(row["success"]) for row in tuning_rows)
    confirmation_successes = sum(int(row["success"]) for row in confirmation_rows)
    passed = (
        attempts <= DEV_TRANSITION_CALL_LIMIT
        and tuning_successes + confirmation_successes >= DEV_MIN_SUCCESS
        and confirmation_successes >= DEV_CONFIRMATION_MIN_SUCCESS
    )
    dev_record = {
        "schema_version": 1,
        "experiment_version": STUDY_VERSION,
        "status": "passed" if passed else "failed",
        "reason": "" if passed else "locked development rule not met",
        "transition_calls": attempts,
        "transition_call_limit": DEV_TRANSITION_CALL_LIMIT,
        "tuning_successes": tuning_successes,
        "confirmation_successes": confirmation_successes,
        "confirmation_total": len(confirmation_rows),
        "overall_successes": tuning_successes + confirmation_successes,
        "overall_total": len(tuning_rows) + len(confirmation_rows),
        "planner_config": locked.as_dict(),
        "stage_a": stage_a,
        "stage_b": stage_b,
    }
    atomic_json(output / "development-gate.json", dev_record)
    retain_evidence(output / "development.csv", "visual-jepa-decision-study-v1-development.csv")
    retain_evidence(output / "development-gate.json", "visual-jepa-decision-study-v1-development-gate.json")
    if passed:
        lock = {
            "schema_version": 1,
            "experiment_version": STUDY_VERSION,
            "status": "locked",
            "preregistration_hash": sha256_file(str(PROTOCOL_PATH)),
            "code_hash": code_hash(),
            "development_gate_hash": sha256_file(str(output / "development-gate.json")),
            "planner_config": locked.as_dict(),
            "tuning_successes": tuning_successes,
            "confirmation_successes": confirmation_successes,
            "overall_successes": tuning_successes + confirmation_successes,
            "development_transition_calls": attempts,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(output / "protocol-lock.json", lock)
        retain_evidence(output / "protocol-lock.json", "visual-jepa-decision-study-v1-protocol-lock.json")
    print(f"Development gate: {'PASS' if passed else 'CLOSED'} ({tuning_successes + confirmation_successes}/20)")


def development_key(row: Mapping[str, object]) -> tuple[float, float, float, float, int, int]:
    return (
        -float(row["successes"]),
        -float(row["mean_return"]),
        float(row["transition_calls"]),
        float(row["horizon_steps"]),
        0 if row["initialization"] == "uniform" else 1,
        0 if row["score_name"] == "cumulative_return" else 1,
    )


def verify_lock(output: Path) -> tuple[dict[str, Any], PlannerConfig, str]:
    lock_path = output / "protocol-lock.json"
    if not lock_path.exists():
        raise ValueError("protocol-lock.json is missing; run dev-search first")
    lock = load_json(lock_path)
    current_protocol_hash = sha256_file(str(PROTOCOL_PATH))
    current_code_hash = code_hash()
    if not protocol_lock_is_valid(
        lock,
        preregistration_hash=current_protocol_hash,
        code_hash=current_code_hash,
    ):
        raise ValueError("protocol lock does not match the current preregistration and code")
    lock_hash = sha256_file(str(lock_path))
    return lock, locked_config(lock), lock_hash


def oracle_gate(output: Path) -> None:
    lock, config, lock_hash = verify_lock(output)
    rows = [
        run_episode_exact(seed, config, method="oracle_cem", model_seed="")
        for seed in FINAL_SEEDS
    ]
    validate_episode_records(rows)
    atomic_csv(output / "oracle-gate.csv", rows, METHOD_FIELDS)
    successes = sum(int(row["success"]) for row in rows)
    gate = {
        "schema_version": 1,
        "experiment_version": STUDY_VERSION,
        "status": "passed" if gate_passes(successes, len(rows), FINAL_ORACLE_MIN_SUCCESS) else "failed",
        "total": len(rows),
        "successes": successes,
        "threshold": FINAL_ORACLE_MIN_SUCCESS,
        "protocol_lock_hash": lock_hash,
        "preregistration_hash": lock["preregistration_hash"],
        "code_hash": lock["code_hash"],
        "transition_calls": sum(int(row["transition_calls"]) for row in rows),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(output / "oracle-gate.json", gate)
    retain_evidence(output / "oracle-gate.csv", "visual-jepa-decision-study-v1-oracle-gate.csv")
    retain_evidence(output / "oracle-gate.json", "visual-jepa-decision-study-v1-oracle-gate.json")
    print(f"Held-out exact-dynamics gate: {gate['status'].upper()} ({successes}/20)")


def to_context_tensor(frames: np.ndarray, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = frames[indices]
    previous, current, following = selected[:, 0], selected[:, 1], selected[:, 2]
    context = np.concatenate((previous, current), axis=-1).transpose(0, 3, 1, 2).copy()
    target = np.concatenate((current, following), axis=-1).transpose(0, 3, 1, 2).copy()
    following_rgb = following.transpose(0, 3, 1, 2).copy()
    context_t = torch.from_numpy(context).to(device=device, dtype=torch.float32).div_(255.0)
    target_t = torch.from_numpy(target).to(device=device, dtype=torch.float32).div_(255.0)
    rgb_t = torch.from_numpy(following_rgb).to(device=device, dtype=torch.float32).div_(255.0)
    return context_t, target_t, rgb_t


def tensors_for_batch(data: TransitionData, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
    context, target, following_rgb = to_context_tensor(data.frames, indices, device)
    actions = torch.from_numpy(data.actions[indices]).to(device=device, dtype=torch.float32)
    states = torch.from_numpy(data.states[indices]).to(device=device, dtype=torch.float32)
    next_states = torch.from_numpy(data.next_states[indices]).to(device=device, dtype=torch.float32)
    return context, target, following_rgb, actions, states, next_states


def model_loss(
    model: torch.nn.Module,
    method: str,
    batch: tuple[torch.Tensor, ...],
    regularizer_generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context, target, next_rgb, actions, states, next_states = batch
    if method == "state_model":
        return F.mse_loss(model(states, actions), next_states), {}
    if method == "image_reconstruction":
        predicted, image = model(context, actions)
        target_embedding = model.encoder(target)
        prediction_loss = F.mse_loss(predicted, target_embedding)
        pixel_loss = F.l1_loss(image, next_rgb)
        return prediction_loss + 0.25 * pixel_loss, {
            "prediction_loss": prediction_loss,
            "pixel_loss": pixel_loss,
        }
    predicted, target_embedding = model(context, actions, target)
    prediction_loss = F.mse_loss(predicted, target_embedding)
    paired = torch.cat((model.encoder(context), target_embedding), dim=0)
    sigreg = epps_pulley_sigreg_loss(
        paired,
        num_projections=SIGREG_PROJECTIONS,
        num_points=17,
        generator=regularizer_generator,
    )
    total = lejepa_total_loss(predicted, target_embedding, sigreg, lambda_sigreg=SIGREG_WEIGHT)
    return total, {"prediction_loss": prediction_loss, "sigreg": sigreg}


def embeddings_for(
    model: torch.nn.Module,
    data: TransitionData,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            _context, target, _rgb = to_context_tensor(data.frames, batch_indices, device)
            outputs.append(model.encode(target).detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0), data.next_states[indices].astype(np.float64)


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
) -> dict[str, object]:
    feature_mean = train_x.mean(axis=0)
    feature_scale = np.maximum(train_x.std(axis=0), 1e-4)
    x_train = (train_x - feature_mean) / feature_scale
    x_validation = (validation_x - feature_mean) / feature_scale
    target_mean = train_y.mean(axis=0)
    y_train = train_y - target_mean
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in PROBE_ALPHAS:
        gram = x_train.T @ x_train + float(alpha) * np.eye(x_train.shape[1])
        coef = np.linalg.lstsq(gram, x_train.T @ y_train, rcond=None)[0]
        prediction = x_validation @ coef + target_mean
        mse = float(np.mean(np.square(prediction - validation_y)))
        candidate = (mse, float(alpha), coef)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "coef": best[2].astype(np.float32),
        "alpha": best[1],
        "validation_mse": best[0],
    }


def apply_probe(embedding: torch.Tensor, probe: Mapping[str, object]) -> torch.Tensor:
    feature_mean = torch.as_tensor(probe["feature_mean"], device=embedding.device)
    feature_scale = torch.as_tensor(probe["feature_scale"], device=embedding.device)
    target_mean = torch.as_tensor(probe["target_mean"], device=embedding.device)
    coef = torch.as_tensor(probe["coef"], device=embedding.device)
    return ((embedding - feature_mean) / feature_scale) @ coef + target_mean


def clamp_task_state(state: torch.Tensor) -> torch.Tensor:
    values = state.clamp(
        min=torch.tensor([MIN_POSITION, -MAX_SPEED], device=state.device),
        max=torch.tensor([MAX_POSITION, MAX_SPEED], device=state.device),
    )
    velocity = torch.where(values[..., 0] <= MIN_POSITION, 0.0, values[..., 1])
    return torch.stack((values[..., 0], velocity), dim=-1)


def probe_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = target - prediction
    mae = np.mean(np.abs(residual), axis=0)
    denominator = np.sum(np.square(target - target.mean(axis=0)), axis=0)
    numerator = np.sum(np.square(residual), axis=0)
    r2 = np.where(denominator > 0, 1.0 - numerator / denominator, 0.0)
    return {
        "position_mae": float(mae[0]),
        "velocity_mae": float(mae[1]),
        "position_r2": float(r2[0]),
        "velocity_r2": float(r2[1]),
    }


def train_one_model(
    data: TransitionData,
    method: str,
    model_seed: int,
    device: torch.device,
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    train_episode_split: int = TRAIN_EPISODE_SPLIT,
) -> tuple[torch.nn.Module, dict[str, object], dict[str, object]]:
    torch.manual_seed(model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(model_seed)
        torch.cuda.reset_peak_memory_stats(device)
    model_class: type[torch.nn.Module] = {
        "state_model": StateDynamics,
        "image_reconstruction": ImageReconstructionDynamics,
        "jepa": ActionConditionedJEPA,
    }[method]
    model = model_class().to(device)
    train_indices = np.flatnonzero(data.episode_ids < train_episode_split)
    validation_indices = np.flatnonzero(data.episode_ids >= train_episode_split)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("episode-level split is empty")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(model_seed)
    regularizer_generator = torch.Generator(device=device).manual_seed(model_seed + 1)
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(train_indices)
        for offset in range(0, len(order), batch_size):
            batch_indices = order[offset : offset + batch_size]
            batch = tensors_for_batch(data, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model_loss(model, method, batch, regularizer_generator)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"non-finite {method} loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"    {method} seed {model_seed}: epoch {epoch + 1}/{epochs}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    validation_loss: list[float] = []
    model.eval()
    with torch.no_grad():
        validation_generator = torch.Generator(device=device).manual_seed(model_seed + 2)
        for offset in range(0, len(validation_indices), batch_size):
            batch = tensors_for_batch(
                data,
                validation_indices[offset : offset + batch_size],
                device,
            )
            loss, _ = model_loss(
                model,
                method,
                batch,
                validation_generator,
            )
            validation_loss.append(float(loss.item()))
    record: dict[str, object] = {
        "method": method,
        "model_seed": model_seed,
        "data_seed": "paired_by_index",
        "status": "complete",
        "epochs": epochs,
        "objective": "one_step_state_mse" if method == "state_model" else (
            "latent_prediction_plus_0.25_pixel_l1" if method == "image_reconstruction" else "lejepa_style_predictive_plus_0.1_sigreg"
        ),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "validation_loss": float(np.mean(validation_loss)),
        "train_seconds": train_seconds,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "precision": "float32",
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "error": "",
    }
    probe: dict[str, object] = {}
    if method != "state_model":
        train_x, train_y = embeddings_for(model, data, train_indices, device)
        val_x, val_y = embeddings_for(model, data, validation_indices, device)
        probe = fit_probe(train_x, train_y, val_x, val_y)
        record["probe_alpha"] = probe["alpha"]
        record["probe_validation_mse"] = probe["validation_mse"]
        if method == "jepa":
            with torch.no_grad():
                sample_x, _ = embeddings_for(model, data, validation_indices[: min(512, len(validation_indices))], device)
                diagnostics = embedding_collapse_diagnostics(torch.from_numpy(sample_x))
            record.update({f"embedding_{key}": value for key, value in diagnostics.items()})
    return model, probe, record


def load_test_data(test_seed: int) -> TransitionData:
    return collect_random_transitions(test_seed, episodes=20)


def evaluate_probe(
    model: torch.nn.Module,
    probe: Mapping[str, object],
    test_data: TransitionData,
    device: torch.device,
) -> dict[str, float]:
    indices = np.arange(len(test_data.actions))
    features, target = embeddings_for(model, test_data, indices, device)
    standardized = (features - np.asarray(probe["feature_mean"])) / np.asarray(probe["feature_scale"])
    prediction = standardized @ np.asarray(probe["coef"]) + np.asarray(probe["target_mean"])
    return probe_metrics(prediction, target)


def state_to_probe_tensor(embedding: torch.Tensor, probe: Mapping[str, object]) -> torch.Tensor:
    return clamp_task_state(apply_probe(embedding, probe))


def visual_context_tensor(start: VisualStart, device: torch.device) -> torch.Tensor:
    frames = context_array(start.previous_frame, start.current_frame)
    chw = frames.transpose(2, 0, 1).copy()
    return torch.from_numpy(chw).to(device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)


def visual_prior_mean(
    model: torch.nn.Module,
    probe: Mapping[str, object],
    start: VisualStart,
    config: PlannerConfig,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    with torch.no_grad():
        latent = model.encode(visual_context_tensor(start, device))
        actions = np.empty(config.action_blocks, dtype=np.float32)
        for block in range(config.action_blocks):
            estimated = state_to_probe_tensor(latent, probe)
            action = scalar_momentum_action(estimated.detach().cpu().numpy())
            actions[block] = action
            block_steps = min(
                config.replan_interval,
                config.horizon_steps - block * config.replan_interval,
            )
            for _ in range(block_steps):
                action_t = torch.tensor([action], dtype=torch.float32, device=device)
                latent = model.predictor(latent, action_t)
    return np.clip(actions, ACTION_MIN, ACTION_MAX), config.horizon_steps


def score_visual_candidates(
    model: torch.nn.Module,
    probe: Mapping[str, object],
    start: VisualStart,
    candidates: np.ndarray,
    config: PlannerConfig,
    device: torch.device,
) -> np.ndarray:
    count = len(candidates)
    with torch.no_grad():
        context = visual_context_tensor(start, device)
        latent = model.encode(context).expand(count, -1).clone()
        alive = torch.ones(count, dtype=torch.bool, device=device)
        returns = torch.zeros(count, dtype=torch.float32, device=device)
        states = state_to_probe_tensor(latent, probe)
        for step in range(config.horizon_steps):
            action_values = candidates[:, min(step // config.replan_interval, candidates.shape[1] - 1)]
            action_t = torch.as_tensor(action_values, dtype=torch.float32, device=device)
            next_latent = model.predictor(latent, action_t)
            next_states = state_to_probe_tensor(next_latent, probe)
            if config.score_name == "cumulative_return":
                reward = (next_states[:, 0] >= GOAL_POSITION).float() * 100.0 - ACTION_COST * action_t.square()
                returns += reward * alive.float()
            latent = torch.where(alive[:, None], next_latent, latent)
            states = torch.where(alive[:, None], next_states, states)
            alive &= states[:, 0] < GOAL_POSITION
        scores = (
            -((states[:, 0] - GOAL_POSITION).square() + 0.2 * states[:, 1].square())
            if config.score_name == "terminal_potential"
            else returns
        )
    return scores.detach().cpu().numpy().astype(np.float64)


def predict_model_batch(
    family: str,
    model: torch.nn.Module | None,
    probe: Mapping[str, object] | None,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    if family == "oracle":
        return torch.from_numpy(
            mountaincar_step(states.detach().cpu().numpy(), actions.detach().cpu().numpy())
        ).to(device=states.device, dtype=torch.float32)
    if family == "state_model":
        assert model is not None
        return clamp_task_state(model(states, actions))
    raise ValueError("visual model prediction requires latent state")


def score_state_candidates(
    family: str,
    model: torch.nn.Module | None,
    start_state: np.ndarray,
    candidates: np.ndarray,
    config: PlannerConfig,
    device: torch.device,
) -> np.ndarray:
    count = len(candidates)
    state = torch.as_tensor(start_state, dtype=torch.float32, device=device).expand(count, -1).clone()
    alive = torch.ones(count, dtype=torch.bool, device=device)
    returns = torch.zeros(count, dtype=torch.float32, device=device)
    with torch.no_grad():
        for step in range(config.horizon_steps):
            action_values = candidates[:, min(step // config.replan_interval, candidates.shape[1] - 1)]
            action = torch.as_tensor(action_values, dtype=torch.float32, device=device)
            if family == "oracle":
                next_state = torch.from_numpy(mountaincar_step(state.cpu().numpy(), action.cpu().numpy())).to(device)
            else:
                assert model is not None
                next_state = predict_model_batch(family, model, None, state, action)
            if config.score_name == "cumulative_return":
                reward = (next_state[:, 0] >= GOAL_POSITION).float() * 100.0 - ACTION_COST * action.square()
                returns += reward * alive.float()
            state = torch.where(alive[:, None], next_state, state)
            alive &= state[:, 0] < GOAL_POSITION
        score = (
            -((state[:, 0] - GOAL_POSITION).square() + 0.2 * state[:, 1].square())
            if config.score_name == "terminal_potential"
            else returns
        )
    return score.detach().cpu().numpy().astype(np.float64)


def planner_action_model(
    family: str,
    model: torch.nn.Module | None,
    probe: Mapping[str, object] | None,
    state: np.ndarray,
    start: VisualStart,
    config: PlannerConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[float, int, int]:
    if family == "oracle":
        return planner_action_exact(state, config, rng)
    if family in {"oracle", "state_model"}:
        prior = None
        prior_calls = 0
        if config.initialization == "momentum_prior":
            def predictor(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
                if family == "oracle":
                    return mountaincar_step(states, actions)
                return _state_step(model, states, actions)

            prior, prior_calls = make_momentum_prior(state, config, predictor)

        def scorer(sequences: np.ndarray) -> np.ndarray:
            return score_state_candidates(family, model, state, sequences, config, device)
    else:
        assert model is not None and probe is not None
        prior = None
        prior_calls = 0
        if config.initialization == "momentum_prior":
            prior, prior_calls = visual_prior_mean(model, probe, start, config, device)

        def scorer(sequences: np.ndarray) -> np.ndarray:
            return score_visual_candidates(model, probe, start, sequences, config, device)
    proposal = cem_proposal(config, scorer, rng, prior_mean=prior)
    return float(proposal.action_blocks[0]), proposal.transition_calls + prior_calls, proposal.forward_batches


def _state_step(model: torch.nn.Module | None, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if model is None:
        raise ValueError("state dynamics model is missing")
    device = next(model.parameters()).device
    with torch.no_grad():
        states_t = torch.as_tensor(states, device=device, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
        return clamp_task_state(model(states_t, actions_t)).cpu().numpy()


def run_episode_model(
    family: str,
    model: torch.nn.Module | None,
    probe: Mapping[str, object] | None,
    model_seed: int | str,
    eval_seed: int,
    config: PlannerConfig,
    device: torch.device,
) -> dict[str, object]:
    env = make_env()
    observation, _ = env.reset(seed=eval_seed)
    previous = resize_frame(render_frame(float(observation[0])))
    current = previous.copy()
    rng = np.random.default_rng(stable_seed(STUDY_VERSION, family, model_seed, eval_seed, config.initialization))
    total_return = 0.0
    calls = batches = steps = 0
    planning_seconds = 0.0
    success = False
    status, error = "complete", ""
    action = 0.0
    started = time.perf_counter()
    try:
        for step in range(MAX_EPISODE_STEPS):
            visual_start = VisualStart(
                start_id=f"eval-{eval_seed}",
                seed=eval_seed,
                step=step,
                state=np.asarray(env.unwrapped.state, dtype=np.float32).copy(),
                previous_frame=previous,
                current_frame=current,
            )
            if step % config.replan_interval == 0:
                planning_started = time.perf_counter()
                if family == "state_model":
                    state = visual_start.state
                elif family == "oracle":
                    state = visual_start.state
                else:
                    assert model is not None and probe is not None
                    with torch.no_grad():
                        embedding = model.encode(visual_context_tensor(visual_start, device))
                        state = state_to_probe_tensor(embedding, probe)[0].cpu().numpy()
                action, case_calls, case_batches = planner_action_model(
                    family,
                    model,
                    probe,
                    state,
                    visual_start,
                    config,
                    rng,
                    device,
                )
                planning_seconds += time.perf_counter() - planning_started
                calls += case_calls
                batches += case_batches
            next_observation, reward, terminated, truncated, _ = env.step(
                np.asarray([action], dtype=np.float32)
            )
            total_return += float(reward)
            steps = step + 1
            next_frame = resize_frame(render_frame(float(next_observation[0])))
            previous, current = current, next_frame
            if terminated:
                success = True
                break
            if truncated:
                break
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    row = episode_row(
        method=f"{family}_cem",
        model_seed=model_seed,
        eval_seed=eval_seed,
        initialization=config.initialization,
        privileged_state=family in {"oracle", "state_model"},
        status=status,
        success=int(success),
        episode_return=total_return,
        steps=steps,
        calls=calls,
        batches=batches,
        planning_seconds=planning_seconds,
        elapsed_seconds=time.perf_counter() - started,
        device=str(device),
        error=error,
    )
    row.update(config.as_dict())
    return row


def run_direct_episode(eval_seed: int) -> dict[str, object]:
    env = make_env()
    observation, _ = env.reset(seed=eval_seed)
    previous = resize_frame(render_frame(float(observation[0])))
    current = previous.copy()
    total_return = 0.0
    success = False
    steps = 0
    started = time.perf_counter()
    try:
        for step in range(MAX_EPISODE_STEPS):
            estimated = visual_state(previous, current)
            action = scalar_momentum_action(estimated)
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
    return episode_row(
        method="direct_visual_momentum",
        model_seed="",
        eval_seed=eval_seed,
        initialization="direct",
        privileged_state=False,
        status="complete",
        success=int(success),
        episode_return=total_return,
        steps=steps,
        calls=0,
        batches=0,
        planning_seconds=0.0,
        elapsed_seconds=time.perf_counter() - started,
        device="cpu",
        error="",
    )


def model_checkpoint_path(output: Path, family: str, seed: int) -> Path:
    return output / "checkpoints" / f"{family}-seed-{seed}.pt"


def save_checkpoint(path: Path, model: torch.nn.Module, probe: Mapping[str, object], meta: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "probe": dict(probe),
            "meta": dict(meta),
        },
        path,
    )


def load_checkpoint(
    output: Path,
    family: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    model_class: type[torch.nn.Module] = {
        "state_model": StateDynamics,
        "image_reconstruction": ImageReconstructionDynamics,
        "jepa": ActionConditionedJEPA,
    }[family]
    model = model_class().to(device)
    checkpoint = torch.load(
        model_checkpoint_path(output, family, seed),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, dict(checkpoint.get("probe", {}))


def train_study_models(output: Path, device: torch.device) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_records: list[dict[str, object]] = []
    probe_records: list[dict[str, object]] = []
    for index, (data_seed, model_seed, test_seed) in enumerate(zip(DATA_SEEDS, MODEL_SEEDS, TEST_SEEDS, strict=True)):
        print(f"Collect shared training group {index + 1}/5 (seed {data_seed})")
        data = collect_random_transitions(data_seed, episodes=TRAIN_EPISODES)
        test_data = load_test_data(test_seed)
        for family in MODEL_FAMILIES:
            try:
                model, probe, record = train_one_model(data, family, model_seed, device)
                record["data_seed"] = data_seed
                record["test_seed"] = test_seed
                if family != "state_model":
                    metrics = evaluate_probe(model, probe, test_data, device)
                    record.update({f"test_{key}": value for key, value in metrics.items()})
                    probe_records.append(
                        {
                            "family": family,
                            "model_seed": model_seed,
                            "data_seed": data_seed,
                            "test_seed": test_seed,
                            "alpha": probe["alpha"],
                            **metrics,
                        }
                    )
                save_checkpoint(model_checkpoint_path(output, family, model_seed), model, probe, record)
                model_records.append(record)
                del model
            except Exception as exc:
                failure = {
                    "method": family,
                    "model_seed": model_seed,
                    "data_seed": data_seed,
                    "test_seed": test_seed,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                model_records.append(failure)
                print(f"    FAILED {family} seed {model_seed}: {failure['error']}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del data, test_data
        atomic_csv(output / "model-runs.csv", model_records, tuple(dict.fromkeys(k for r in model_records for k in r)))
        atomic_csv(output / "probe-test.csv", probe_records, tuple(dict.fromkeys(k for r in probe_records for k in r)))
    if model_records:
        retain_evidence(output / "model-runs.csv", "visual-jepa-decision-study-v1-model-runs.csv")
    if probe_records:
        retain_evidence(output / "probe-test.csv", "visual-jepa-decision-study-v1-probe-test.csv")
    return model_records, probe_records


def clipped_state(state: np.ndarray) -> np.ndarray:
    result = np.asarray(state, dtype=np.float32).copy()
    result[0] = np.clip(result[0], MIN_POSITION, MAX_POSITION)
    result[1] = np.clip(result[1], -MAX_SPEED, MAX_SPEED)
    if result[0] <= MIN_POSITION:
        result[1] = 0.0
    return result


def actual_intervention_return(start: VisualStart, first_action: float) -> tuple[np.ndarray, float]:
    env = make_env()
    env.reset(seed=start.seed + 90_000)
    env.unwrapped.state = np.asarray(start.state, dtype=np.float32).copy()
    total = 0.0
    first_next = np.asarray(start.state, dtype=np.float32).copy()
    try:
        for step in range(MAX_EPISODE_STEPS):
            state = np.asarray(env.unwrapped.state, dtype=np.float32)
            action = first_action if step == 0 else scalar_momentum_action(state)
            _observation, reward, terminated, truncated, _ = env.step(np.asarray([action], dtype=np.float32))
            total += float(reward)
            state = np.asarray(env.unwrapped.state, dtype=np.float32).copy()
            if step == 0:
                first_next = state
            if terminated or truncated:
                break
    finally:
        env.close()
    return first_next, total


def predicted_intervention(
    family: str,
    model: torch.nn.Module | None,
    probe: Mapping[str, object] | None,
    start: VisualStart,
    first_action: float,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if family == "oracle":
        state = np.asarray(start.state, dtype=np.float32)
        total = 0.0
        predicted_first = state.copy()
        alive = True
        for step in range(MAX_EPISODE_STEPS):
            action = first_action if step == 0 else scalar_momentum_action(state)
            state = mountaincar_step(state, np.asarray(action))
            if step == 0:
                predicted_first = state.copy()
            reward = 100.0 * float(state[0] >= GOAL_POSITION) - ACTION_COST * action**2
            total += reward * alive
            alive = alive and state[0] < GOAL_POSITION
        return predicted_first, total
    if family == "state_model":
        assert model is not None
        with torch.no_grad():
            state_t = torch.as_tensor(start.state, dtype=torch.float32, device=device).unsqueeze(0)
            alive = torch.ones(1, dtype=torch.bool, device=device)
            total = torch.zeros(1, dtype=torch.float32, device=device)
            predicted_first = state_t
            for step in range(MAX_EPISODE_STEPS):
                action_t = (
                    torch.tensor([first_action], dtype=torch.float32, device=device)
                    if step == 0
                    else torch.where(state_t[:, 1] >= 0.0, 1.0, -1.0)
                )
                state_t = clamp_task_state(model(state_t, action_t))
                if step == 0:
                    predicted_first = state_t.clone()
                reward = (state_t[:, 0] >= GOAL_POSITION).float() * 100.0 - ACTION_COST * action_t.square()
                total += reward * alive.float()
                alive &= state_t[:, 0] < GOAL_POSITION
        return predicted_first[0].cpu().numpy(), float(total.item())
    assert model is not None and probe is not None
    with torch.no_grad():
        latent = model.encode(visual_context_tensor(start, device))
        alive = torch.ones(1, dtype=torch.bool, device=device)
        total = torch.zeros(1, dtype=torch.float32, device=device)
        predicted_first = None
        state_t = state_to_probe_tensor(latent, probe)
        for step in range(MAX_EPISODE_STEPS):
            action_t = (
                torch.tensor([first_action], device=device, dtype=torch.float32)
                if step == 0
                else torch.where(state_t[:, 1] >= 0.0, 1.0, -1.0)
            )
            latent = model.predictor(latent, action_t)
            state_t = state_to_probe_tensor(latent, probe)
            if step == 0:
                predicted_first = state_t.clone()
            reward = (state_t[:, 0] >= GOAL_POSITION).float() * 100.0 - ACTION_COST * action_t.square()
            total += reward * alive.float()
            alive &= state_t[:, 0] < GOAL_POSITION
    assert predicted_first is not None
    return predicted_first[0].cpu().numpy().astype(np.float32), float(total.item())


def run_interventions(output: Path, device: torch.device) -> list[dict[str, object]]:
    starts = collect_intervention_starts(INTERVENTION_SEEDS)
    rows: list[dict[str, object]] = []
    for start in starts:
        families: list[tuple[str, int | str, torch.nn.Module | None, Mapping[str, object] | None]] = [
            ("oracle", "", None, None)
        ]
        for family in MODEL_FAMILIES:
            for model_seed in MODEL_SEEDS:
                checkpoint = model_checkpoint_path(output, family, model_seed)
                if checkpoint.exists():
                    model, probe = load_checkpoint(output, family, model_seed, device)
                    families.append((family, model_seed, model, probe))
                else:
                    families.append((family, model_seed, None, None))
        for action in (-1.0, 0.0, 1.0):
            actual_state, observed_return = actual_intervention_return(start, action)
            for family, model_seed, model, probe in families:
                try:
                    if model is None and family != "oracle":
                        rows.append({
                            "method": family,
                            "model_seed": str(model_seed),
                            "start_id": start.start_id,
                            "seed": start.seed,
                            "step": start.step,
                            "action": action,
                            "actual_position": float(actual_state[0]),
                            "actual_velocity": float(actual_state[1]),
                            "predicted_position": "",
                            "predicted_velocity": "",
                            "position_abs_error": "",
                            "velocity_abs_error": "",
                            "observed_return": observed_return,
                            "predicted_return": "",
                            "status": "missing_model",
                            "error": "checkpoint missing",
                        })
                        continue
                    predicted_state, predicted_return = predicted_intervention(
                        family, model, probe, start, action, device
                    )
                    rows.append({
                        "method": family,
                        "model_seed": str(model_seed),
                        "start_id": start.start_id,
                        "seed": start.seed,
                        "step": start.step,
                        "action": action,
                        "actual_position": float(actual_state[0]),
                        "actual_velocity": float(actual_state[1]),
                        "predicted_position": float(predicted_state[0]),
                        "predicted_velocity": float(predicted_state[1]),
                        "position_abs_error": abs(float(predicted_state[0]) - float(actual_state[0])),
                        "velocity_abs_error": abs(float(predicted_state[1]) - float(actual_state[1])),
                        "observed_return": observed_return,
                        "predicted_return": float(predicted_return),
                        "status": "complete",
                        "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "method": family,
                        "model_seed": str(model_seed),
                        "start_id": start.start_id,
                        "seed": start.seed,
                        "step": start.step,
                        "action": action,
                        "actual_position": float(actual_state[0]),
                        "actual_velocity": float(actual_state[1]),
                        "predicted_position": "",
                        "predicted_velocity": "",
                        "position_abs_error": "",
                        "velocity_abs_error": "",
                        "observed_return": observed_return,
                        "predicted_return": "",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        for _family, _model_seed, model, _probe in families:
            if model is not None:
                del model
    validate_intervention_records(rows)
    expected_groups = {
        (method, seed, start.start_id)
        for start in starts
        for method, seeds in (
            ("oracle", ("",)),
            *((family, tuple(str(seed) for seed in MODEL_SEEDS)) for family in MODEL_FAMILIES),
        )
        for seed in seeds
    }
    observed_groups = {
        (str(row["method"]), str(row["model_seed"]), str(row["start_id"]))
        for row in rows
    }
    if observed_groups != expected_groups:
        raise ValueError(
            f"intervention coverage mismatch: expected {len(expected_groups)} groups, "
            f"observed {len(observed_groups)}"
        )
    expected_rows = len(expected_groups) * 3
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} intervention rows, got {len(rows)}")
    atomic_csv(output / "interventions.csv", rows, INTERVENTION_FIELDS)
    retain_evidence(output / "interventions.csv", "visual-jepa-decision-study-v1-interventions.csv")
    return rows


def final_budget_estimate(config: PlannerConfig) -> int:
    decisions = math.ceil(MAX_EPISODE_STEPS / config.replan_interval)
    per_decision = config.candidates * config.horizon_steps * config.updates
    prior_per_decision = config.horizon_steps if config.initialization == "momentum_prior" else 0
    # Gate cases already cover the locked oracle initialization.
    episode_cases = 20 * (1 + len(MODEL_FAMILIES) * len(MODEL_SEEDS) * 2)
    oracle_extra = 20
    all_locked = episode_cases * decisions * (per_decision + prior_per_decision)
    opposite_oracle = oracle_extra * decisions * (
        per_decision + (config.horizon_steps if config.initialization == "uniform" else 0)
    )
    already_used = 20 * decisions * (per_decision + prior_per_decision)
    return all_locked + opposite_oracle - already_used


def run_final_evaluation(
    output: Path,
    config: PlannerConfig,
    device: torch.device,
    gate_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = list(gate_rows)
    opposite_initialization = "momentum_prior" if config.initialization == "uniform" else "uniform"
    opposite = PlannerConfig(
        horizon_steps=config.horizon_steps,
        candidates=config.candidates,
        elites=config.elites,
        updates=config.updates,
        score_name=config.score_name,
        initialization=opposite_initialization,
        replan_interval=config.replan_interval,
    )
    for seed in FINAL_SEEDS:
        rows.append(run_episode_exact(seed, opposite, method="oracle_cem"))
    for seed in FINAL_SEEDS:
        rows.append(run_direct_episode(seed))
    for family in MODEL_FAMILIES:
        for model_seed in MODEL_SEEDS:
            checkpoint = model_checkpoint_path(output, family, model_seed)
            if not checkpoint.exists():
                for initialization in ("uniform", "momentum_prior"):
                    for eval_seed in FINAL_SEEDS:
                        rows.append(episode_row(
                            method=f"{family}_cem",
                            model_seed=model_seed,
                            eval_seed=eval_seed,
                            initialization=initialization,
                            privileged_state=family == "state_model",
                            status="missing_model",
                            success=0,
                            episode_return="",
                            steps=0,
                            calls=0,
                            batches=0,
                            planning_seconds="",
                            elapsed_seconds="",
                            device=str(device),
                            error="trained checkpoint missing",
                        ))
                continue
            model, probe = load_checkpoint(output, family, model_seed, device)
            for initialization in ("uniform", "momentum_prior"):
                method_config = PlannerConfig(
                    horizon_steps=config.horizon_steps,
                    candidates=config.candidates,
                    elites=config.elites,
                    updates=config.updates,
                    score_name=config.score_name,
                    initialization=initialization,
                    replan_interval=config.replan_interval,
                )
                for eval_seed in FINAL_SEEDS:
                    rows.append(run_episode_model(
                        family,
                        model,
                        probe,
                        model_seed,
                        eval_seed,
                        method_config,
                        device,
                    ))
                atomic_csv(output / "episodes.csv", rows, METHOD_FIELDS)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    validate_episode_records(rows)
    atomic_csv(output / "episodes.csv", rows, METHOD_FIELDS)
    retain_evidence(output / "episodes.csv", "visual-jepa-decision-study-v1-episodes.csv")
    return rows


def rank_accuracy(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        groups.setdefault((str(row["method"]), str(row["model_seed"]), str(row["start_id"])), []).append(row)
    result: dict[str, object] = {}
    by_method: dict[str, list[int]] = {}
    by_seed: dict[tuple[str, str], list[int]] = {}
    for key, group in groups.items():
        if len(group) != 3:
            continue
        correct = total = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                observed_delta = float(group[i]["observed_return"]) - float(group[j]["observed_return"])
                predicted_delta = float(group[i]["predicted_return"]) - float(group[j]["predicted_return"])
                if abs(observed_delta) < 1e-8:
                    continue
                total += 1
                correct += int(observed_delta * predicted_delta > 0)
        result["|".join(key)] = {"correct": correct, "total": total}
        by_method.setdefault(key[0], [0, 0])
        by_method[key[0]][0] += correct
        by_method[key[0]][1] += total
        by_seed.setdefault((key[0], key[1]), [0, 0])
        by_seed[(key[0], key[1])][0] += correct
        by_seed[(key[0], key[1])][1] += total
    return {
        "per_start": result,
        "by_method": {
            method: {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1] if value[1] else None}
            for method, value in by_method.items()
        },
        "by_model_seed": {
            f"{method}|{seed}": {"correct": value[0], "total": value[1], "accuracy": value[0] / value[1] if value[1] else None}
            for (method, seed), value in by_seed.items()
        },
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def summarize_study(
    output: Path,
    lock: Mapping[str, object],
    gate: Mapping[str, object],
    episodes: Sequence[Mapping[str, object]],
    models: Sequence[Mapping[str, object]],
    probes: Sequence[Mapping[str, object]],
    interventions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    completed = [row for row in episodes if row.get("status") == "complete"]
    methods: dict[str, object] = {}
    for method in sorted({str(row["method"]) for row in completed}):
        method_rows = [row for row in completed if row["method"] == method]
        summary: dict[str, object] = {
            "n": len(method_rows),
            "successes": sum(int(row["success"]) for row in method_rows),
            "success_rate": float(np.mean([int(row["success"]) for row in method_rows])),
            "mean_return": float(np.mean([float(row["return"]) for row in method_rows])),
            "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in method_rows])),
            "mean_planning_seconds": float(np.mean([float(row["planning_seconds"]) for row in method_rows])),
            "mean_transition_calls": float(np.mean([int(row["transition_calls"]) for row in method_rows])),
        }
        if method in {"oracle_cem", "direct_visual_momentum"}:
            summary["fixed_start_wilson_95_percent_interval"] = wilson_interval(
                int(summary["successes"]), int(summary["n"])
            )
        if method in {"state_model_cem", "image_reconstruction_cem", "jepa_cem"}:
            summary["by_model_seed"] = {}
            for model_seed in sorted({str(row["model_seed"]) for row in method_rows}):
                seed_rows = [row for row in method_rows if str(row["model_seed"]) == model_seed]
                summary["by_model_seed"][model_seed] = {
                    "n": len(seed_rows),
                    "successes": sum(int(row["success"]) for row in seed_rows),
                    "success_rate": float(np.mean([int(row["success"]) for row in seed_rows])),
                    "mean_return": float(np.mean([float(row["return"]) for row in seed_rows])),
                    "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in seed_rows])),
                    "mean_planning_seconds": float(np.mean([float(row["planning_seconds"]) for row in seed_rows])),
                    "mean_transition_calls": float(np.mean([int(row["transition_calls"]) for row in seed_rows])),
                }
        methods[method] = summary
    primary: dict[str, object] = {"status": "not_estimable"}
    direct = {int(row["eval_seed"]): float(row["return"]) for row in completed if row["method"] == "direct_visual_momentum"}
    jepa_rows = [row for row in completed if row["method"] == "jepa_cem" and row["initialization"] == "momentum_prior"]
    if direct and jepa_rows:
        seed_deltas: dict[str, list[float]] = {}
        for row in jepa_rows:
            eval_seed = int(row["eval_seed"])
            seed_deltas.setdefault(str(row["model_seed"]), []).append(float(row["return"]) - direct[eval_seed])
        means = [float(np.mean(values)) for values in seed_deltas.values()]
        complete_coverage = len(seed_deltas) == len(MODEL_SEEDS) and all(
            len(values) == len(FINAL_SEEDS) for values in seed_deltas.values()
        ) and len(direct) == len(FINAL_SEEDS)
        primary = {
            "status": "descriptive" if complete_coverage else "not_estimable_incomplete_seed_coverage",
            "per_model_seed_mean_return_delta": seed_deltas,
            "mean_seed_delta": float(np.mean(means)) if complete_coverage and means else None,
            "hierarchical_bootstrap_95_percent_interval": hierarchical_bootstrap_interval(seed_deltas, direct) if complete_coverage else None,
            "complete_seed_coverage": complete_coverage,
            "interpretation": "descriptive local synthetic-task estimate; no significance claim" if complete_coverage else "fixed five-seed primary comparison is incomplete",
        }
    models_complete = bool(models) and all(row.get("status") == "complete" for row in models)
    interventions_complete = bool(interventions) and all(row.get("status") == "complete" for row in interventions)
    value = {
        "schema_version": 1,
        "experiment_version": STUDY_VERSION,
        "status": "complete" if len(completed) == len(episodes) and models_complete and interventions_complete else "partial",
        "preregistration_hash": lock["preregistration_hash"],
        "protocol_lock_hash": sha256_file(str(output / "protocol-lock.json")),
        "oracle_gate": gate,
        "planner_config": lock["planner_config"],
        "development_transition_calls": lock["development_transition_calls"],
        "final_transition_calls": sum(int(row.get("transition_calls", 0)) for row in completed),
        "final_transition_call_limit": FINAL_TRANSITION_CALL_LIMIT,
        "model_runs": list(models),
        "probe_test": list(probes),
        "method_summary": methods,
        "primary_comparison": primary,
        "intervention_action_ranking": rank_accuracy(interventions),
        "limitations": [
            "One local synthetic MountainCar task with generated images.",
            "The visual momentum policy is handcrafted; state and oracle dynamics have privileged access.",
            "The learned visual models are compact teaching analogues, not published-model reproductions.",
            "The linear state readout is an auxiliary evaluator and contributes to visual-model planning.",
            "Intervals and seed comparisons are descriptive; they do not support general JEPA claims.",
        ],
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda or "",
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gymnasium": gym.__version__,
        },
    }
    atomic_json(output / "report.json", value)
    retain_evidence(output / "report.json", "visual-jepa-decision-study-v1-report.json")
    return value


def hierarchical_bootstrap_interval(
    seed_deltas: Mapping[str, Sequence[float]],
    direct: Mapping[int, float],
    *,
    replicates: int = 10_000,
) -> list[float] | None:
    if len(seed_deltas) < 2 or not direct:
        return None
    rng = np.random.default_rng(97531)
    seeds = list(seed_deltas)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        per_seed = []
        for seed in selected_seeds:
            deltas = np.asarray(seed_deltas[str(seed)], dtype=np.float64)
            per_seed.append(float(np.mean(rng.choice(deltas, size=len(deltas), replace=True))))
        samples[replicate] = float(np.mean(per_seed))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return [float(lower), float(upper)]


def validate_outputs(output: Path) -> None:
    lock, _config, lock_hash = verify_lock(output)
    gate = load_json(output / "oracle-gate.json")
    if not oracle_gate_is_valid(gate, protocol_lock_hash=lock_hash):
        raise ValueError("oracle gate is missing, stale, or failed")
    if (output / "episodes.csv").exists():
        with (output / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        validate_episode_records(records)
    if (output / "interventions.csv").exists():
        with (output / "interventions.csv").open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        validate_intervention_records(records)
    print(f"Validated lock, oracle gate, and available evidence for {STUDY_VERSION}.")


def cpu_smoke(output: Path) -> None:
    device = torch.device("cpu")
    data = collect_random_transitions(SMOKE_SEEDS[0], episodes=6)
    trained: dict[str, tuple[torch.nn.Module, dict[str, object]]] = {}
    model_records: list[dict[str, object]] = []
    for family in MODEL_FAMILIES:
        model, probe, record = train_one_model(
            data,
            family,
            stable_seed("cpu-smoke", family),
            device,
            epochs=2,
            batch_size=64,
            train_episode_split=4,
        )
        if record["status"] != "complete" or not np.isfinite(float(record["validation_loss"])):
            raise RuntimeError(f"CPU smoke training failed for {family}")
        trained[family] = (model, probe)
        model_records.append(record)
    config = make_config(10, 8, 2, 1, "terminal_potential", "uniform")
    rows = [run_episode_exact(SMOKE_SEEDS[0], config, method="smoke_oracle")]
    for family, (model, probe) in trained.items():
        rows.append(
            run_episode_model(
                family,
                model,
                probe,
                f"smoke-{family}",
                SMOKE_SEEDS[1],
                config,
                device,
            )
        )
    validate_episode_records(rows)
    atomic_csv(output / "smoke-episodes.csv", rows, METHOD_FIELDS)
    atomic_csv(
        output / "smoke-model-runs.csv",
        model_records,
        tuple(dict.fromkeys(key for row in model_records for key in row)),
    )
    print(f"CPU smoke passed: generated {len(data.actions)} transitions, trained all three model families, and ran {len(rows)} planner episodes.")
    del trained, data, device


def run_full(output: Path, device: torch.device) -> None:
    lock, config, lock_hash = verify_lock(output)
    gate = load_json(output / "oracle-gate.json")
    if not oracle_gate_is_valid(gate, protocol_lock_hash=lock_hash):
        raise ValueError("learned-model evaluation requires a passing held-out oracle gate")
    gate_rows_path = output / "oracle-gate.csv"
    if not gate_rows_path.exists():
        raise ValueError("oracle-gate.csv is missing")
    with gate_rows_path.open(newline="", encoding="utf-8") as handle:
        gate_rows = list(csv.DictReader(handle))
    estimated = final_budget_estimate(config)
    gate_used = sum(int(row["transition_calls"]) for row in gate_rows)
    if estimated + gate_used > FINAL_TRANSITION_CALL_LIMIT:
        summary = {
            "schema_version": 1,
            "experiment_version": STUDY_VERSION,
            "status": "gated_compute_budget",
            "planner_config": config.as_dict(),
            "oracle_gate_calls": gate_used,
            "estimated_additional_calls": estimated,
            "call_limit": FINAL_TRANSITION_CALL_LIMIT,
            "message": "The preregistered complete comparison exceeds the final call ceiling. No learned-model training or final cases were opened.",
        }
        atomic_json(output / "study-status.json", summary)
        retain_evidence(output / "study-status.json", "visual-jepa-decision-study-v1-status.json")
        print(summary["message"])
        return
    model_records, probe_records = train_study_models(output, device)
    intervention_rows = run_interventions(output, device)
    episode_rows = run_final_evaluation(output, config, device, gate_rows)
    summary = summarize_study(output, lock, gate, episode_rows, model_records, probe_records, intervention_rows)
    print(f"Full study status: {summary['status']}; final cases={len(episode_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "dev-search", "oracle-gate", "full", "validate"))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu" or args.mode == "smoke":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device("cuda")
    else:
        device = select_device()
    print(f"Study={STUDY_VERSION}; output={output}; device={device}")
    if args.mode == "smoke":
        cpu_smoke(output)
    elif args.mode == "dev-search":
        development_search(output, device)
    elif args.mode == "oracle-gate":
        oracle_gate(output)
    elif args.mode == "full":
        run_full(output, device)
    else:
        validate_outputs(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
