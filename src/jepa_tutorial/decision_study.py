"""Small, auditable primitives for the versioned visual-JEPA decision study."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

STUDY_VERSION = "visual-jepa-decision-study-v1"
GOAL_POSITION = 0.45
MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
MAX_EPISODE_STEPS = 300
ACTION_MIN = -1.0
ACTION_MAX = 1.0
ACTION_COST = 0.1
GOAL_REWARD = 100.0
REPLAN_INTERVAL = 5
FINAL_ORACLE_MIN_SUCCESS = 18
DEV_MIN_SUCCESS = 18
DEV_CONFIRMATION_MIN_SUCCESS = 9
DEV_TRANSITION_CALL_LIMIT = 500_000_000
FINAL_TRANSITION_CALL_LIMIT = 750_000_000


@dataclass(frozen=True)
class PlannerConfig:
    """Locked CEM settings shared by planner methods."""

    horizon_steps: int
    candidates: int
    elites: int
    updates: int
    score_name: str
    initialization: str
    replan_interval: int = REPLAN_INTERVAL

    def __post_init__(self) -> None:
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be positive")
        if self.candidates < 2:
            raise ValueError("candidates must be at least 2")
        if not 1 <= self.elites < self.candidates:
            raise ValueError("elites must be between 1 and candidates - 1")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.replan_interval < 1:
            raise ValueError("replan_interval must be positive")
        if self.score_name not in {"terminal_potential", "cumulative_return"}:
            raise ValueError("unknown score_name")
        if self.initialization not in {"uniform", "momentum_prior"}:
            raise ValueError("unknown initialization")

    @property
    def action_blocks(self) -> int:
        return math.ceil(self.horizon_steps / self.replan_interval)

    @property
    def candidate_transition_budget(self) -> int:
        """Worst-case candidate transitions for one planning decision."""
        return self.candidates * self.horizon_steps * self.updates

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class CEMProposal:
    """A bounded action sequence and its exact candidate-evaluation count."""

    action_blocks: np.ndarray
    transition_calls: int
    forward_batches: int


def stable_seed(*parts: object) -> int:
    """Return a stable 32-bit seed without relying on Python's randomized hash."""
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mountaincar_step(states: np.ndarray, actions: np.ndarray | float) -> np.ndarray:
    """Apply Gymnasium MountainCarContinuous-v0 dynamics to one or many states."""
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.shape[-1:] != (2,):
        raise ValueError("states must have final dimension 2 (position, velocity)")
    action_array = np.asarray(actions, dtype=np.float64)
    if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
        raise ValueError("states and actions must be finite")
    action_array = np.clip(action_array, ACTION_MIN, ACTION_MAX)
    if state_array.ndim > 1 and action_array.ndim > 0:
        if action_array.shape not in {state_array.shape[:-1], state_array.shape[:-1] + (1,)}:
            raise ValueError("batched actions must match the state batch shape")
        action_array = np.reshape(action_array, state_array.shape[:-1])
    position = state_array[..., 0]
    velocity = state_array[..., 1]
    next_velocity = np.clip(
        velocity + action_array * 0.0015 - 0.0025 * np.cos(3.0 * position),
        -MAX_SPEED,
        MAX_SPEED,
    )
    next_position = np.clip(position + next_velocity, MIN_POSITION, MAX_POSITION)
    next_velocity = np.where(next_position <= MIN_POSITION, 0.0, next_velocity)
    return np.stack((next_position, next_velocity), axis=-1).astype(np.float32)


def mountaincar_reward(next_states: np.ndarray, actions: np.ndarray | float) -> np.ndarray:
    """Return the environment reward and a termination mask for predicted states."""
    states = np.asarray(next_states)
    action_array = np.asarray(actions, dtype=np.float64)
    action_array = np.clip(action_array, ACTION_MIN, ACTION_MAX)
    reached = states[..., 0] >= GOAL_POSITION
    return (GOAL_REWARD * reached - ACTION_COST * np.square(action_array)).astype(np.float64)


def terminal_potential(states: np.ndarray) -> np.ndarray:
    """Notebook 22's endpoint potential, retained as a separate planner score."""
    values = np.asarray(states, dtype=np.float64)
    distance = values[..., 0] - GOAL_POSITION
    return -(np.square(distance) + 0.2 * np.square(values[..., 1]))


def momentum_action(states: np.ndarray) -> np.ndarray:
    """The fixed direct-control rule: push with the sign of estimated velocity."""
    values = np.asarray(states)
    if values.shape[-1:] != (2,):
        raise ValueError("states must have final dimension 2")
    return np.where(values[..., 1] >= 0.0, 1.0, -1.0).astype(np.float32)


def make_momentum_prior(
    start_state: np.ndarray,
    config: PlannerConfig,
    predict_step: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Roll the common momentum rule through one model to seed the CEM mean."""
    state = np.asarray(start_state, dtype=np.float32).reshape(1, 2)
    actions = np.empty(config.action_blocks, dtype=np.float32)
    transition_calls = 0
    for block in range(config.action_blocks):
        action = float(momentum_action(state)[0])
        actions[block] = action
        block_steps = min(
            config.replan_interval,
            config.horizon_steps - block * config.replan_interval,
        )
        for _ in range(block_steps):
            state = np.asarray(predict_step(state, np.asarray([action])), dtype=np.float32)
            if state.shape != (1, 2) or not np.isfinite(state).all():
                raise ValueError("predict_step must return one finite [position, velocity] state")
            transition_calls += 1
    return np.clip(actions, ACTION_MIN, ACTION_MAX), transition_calls


def cem_proposal(
    config: PlannerConfig,
    score_candidates: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    *,
    prior_mean: np.ndarray | None = None,
) -> CEMProposal:
    """Run diagonal-Gaussian CEM and return the optimized sequence mean."""
    if config.initialization == "momentum_prior":
        if prior_mean is None:
            raise ValueError("momentum_prior initialization requires prior_mean")
        mean = np.asarray(prior_mean, dtype=np.float64).reshape(-1).copy()
        if mean.shape != (config.action_blocks,):
            raise ValueError("prior_mean length must match the action-block count")
        if not np.isfinite(mean).all():
            raise ValueError("prior_mean must be finite")
        mean = np.clip(mean, ACTION_MIN, ACTION_MAX)
        std = np.full(config.action_blocks, 0.5, dtype=np.float64)
    else:
        mean = np.zeros(config.action_blocks, dtype=np.float64)
        std = np.ones(config.action_blocks, dtype=np.float64)

    batches = 0
    for _ in range(config.updates):
        noise = rng.standard_normal((config.candidates, config.action_blocks))
        candidates = np.clip(mean[None, :] + std[None, :] * noise, ACTION_MIN, ACTION_MAX)
        scores = np.asarray(score_candidates(candidates), dtype=np.float64)
        if scores.shape != (config.candidates,):
            raise ValueError("score_candidates must return one score per candidate")
        if not np.isfinite(scores).all():
            raise ValueError("score_candidates returned a non-finite score")
        elite_indices = np.argpartition(scores, -config.elites)[-config.elites :]
        elites = candidates[elite_indices]
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), 0.05)
        batches += 1

    return CEMProposal(
        action_blocks=np.clip(mean, ACTION_MIN, ACTION_MAX).astype(np.float32),
        transition_calls=config.candidate_transition_budget,
        forward_batches=batches * config.horizon_steps,
    )


def select_dev_configuration(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Apply the preregistered lexicographic setting-selection rule."""
    valid = [row for row in rows if row.get("status") == "complete"]
    if not valid:
        raise ValueError("no complete planner development rows")

    def key(row: Mapping[str, object]) -> tuple[float, float, float, float, int, int]:
        return (
            -float(row["successes"]),
            -float(row["mean_return"]),
            float(row["transition_calls"]),
            float(row["horizon_steps"]),
            0 if row["initialization"] == "uniform" else 1,
            0 if row["score_name"] == "cumulative_return" else 1,
        )

    return min(valid, key=key)


def gate_passes(successes: int, total: int, threshold: int) -> bool:
    if total < 1 or not 0 <= successes <= total or not 0 <= threshold <= total:
        raise ValueError("invalid gate counts")
    return successes >= threshold


def validate_episode_records(records: Iterable[Mapping[str, object]]) -> None:
    """Check required episode fields while retaining failed and missing rows."""
    required = {
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
    }
    seen: set[tuple[str, str, str, str]] = set()
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"episode row {index} is missing fields: {sorted(missing)}")
        status = str(record["status"])
        if status == "complete":
            for field in ("return", "steps", "transition_calls", "forward_batches", "planning_seconds", "elapsed_seconds"):
                value = float(record[field])
                if not math.isfinite(value):
                    raise ValueError(f"episode row {index} has non-finite {field}")
            if int(record["success"]) not in {0, 1}:
                raise ValueError(f"episode row {index} has invalid success value")
            if record["error"] not in {"", None}:
                raise ValueError(f"complete episode row {index} has an error reason")
        elif not str(record["error"]).strip():
            raise ValueError(f"failed episode row {index} needs an error reason")
        key = (
            str(record["method"]),
            str(record["model_seed"]),
            str(record["eval_seed"]),
            str(record["initialization"]),
        )
        if key in seen:
            raise ValueError(f"duplicate episode case: {key}")
        seen.add(key)


def validate_intervention_records(records: Iterable[Mapping[str, object]]) -> None:
    """Check paired action cases and finite common-outcome metrics."""
    groups: dict[tuple[str, str, str], list[float]] = {}
    for index, record in enumerate(records):
        for field in ("method", "model_seed", "start_id", "action", "status", "error"):
            if field not in record:
                raise ValueError(f"intervention row {index} is missing {field}")
        status = str(record["status"])
        if status == "complete":
            for field in ("actual_position", "actual_velocity", "predicted_position", "predicted_velocity", "observed_return", "predicted_return"):
                value = float(record[field])
                if not math.isfinite(value):
                    raise ValueError(f"intervention row {index} has non-finite {field}")
        elif not str(record["error"]).strip():
            raise ValueError(f"failed intervention row {index} needs an error reason")
        key = (str(record["method"]), str(record["model_seed"]), str(record["start_id"]))
        groups.setdefault(key, []).append(float(record["action"]))
    expected = [-1.0, 0.0, 1.0]
    for key, actions in groups.items():
        if sorted(actions) != expected:
            raise ValueError(f"intervention group {key} must contain actions {expected}")


def protocol_lock_is_valid(
    lock: Mapping[str, object], *, preregistration_hash: str, code_hash: str
) -> bool:
    """Fail closed unless the lock names the current preregistration and code."""
    return (
        lock.get("schema_version") == 1
        and lock.get("experiment_version") == STUDY_VERSION
        and lock.get("status") == "locked"
        and lock.get("preregistration_hash") == preregistration_hash
        and lock.get("code_hash") == code_hash
        and bool(lock.get("planner_config"))
    )


def oracle_gate_is_valid(
    gate: Mapping[str, object], *, protocol_lock_hash: str
) -> bool:
    """Fail closed unless the locked exact planner passed the final oracle gate."""
    try:
        total = int(gate.get("total", 0))
        successes = int(gate.get("successes", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        gate.get("schema_version") == 1
        and gate.get("experiment_version") == STUDY_VERSION
        and gate.get("status") == "passed"
        and gate.get("protocol_lock_hash") == protocol_lock_hash
        and total == 20
        and gate_passes(successes, total, FINAL_ORACLE_MIN_SUCCESS)
    )
