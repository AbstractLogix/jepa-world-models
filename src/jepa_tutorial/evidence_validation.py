"""Validation helpers for tutorial experiment evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

_METHODS = {"oracle", "state_model", "image_reconstruction", "jepa"}
_STATUSES = {"complete", "error", "missing_model"}
_ACTIONS = (-1.0, 0.0, 1.0)
_PREDICTION_SPACES = {
    "oracle": "simulator_state",
    "state_model": "simulator_state",
    "image_reconstruction": "pixels_and_latent",
    "jepa": "latent",
}
_SCORING_SPACES = {
    "oracle": "simulator_state_goal_distance",
    "state_model": "simulator_state_goal_distance",
    "image_reconstruction": "pixel_goal_l1",
    "jepa": "latent_goal_mse",
}
_METRIC_FIELDS = (
    "actual_position",
    "predicted_position",
    "position_abs_error",
    "velocity_abs_error",
    "pixel_l1_error",
    "latent_mse",
    "observed_score",
    "predicted_score",
)
_REQUIRED_COMPLETE_METRICS = {
    "oracle": ("predicted_position", "position_abs_error", "velocity_abs_error", "predicted_score"),
    "state_model": ("predicted_position", "position_abs_error", "velocity_abs_error", "predicted_score"),
    "image_reconstruction": ("pixel_l1_error", "latent_mse", "predicted_score"),
    "jepa": ("latent_mse", "predicted_score"),
}
_REQUIRED_FIELDS = (
    "method",
    "prediction_space",
    "scoring_space",
    "device",
    "model_seed",
    "start_id",
    "action",
    "actual_position",
    "observed_score",
    "status",
)


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def validate_intervention_records(records: Iterable[Mapping[str, object]]) -> None:
    """Raise ValueError when intervention rows violate the evidence contract.

    An empty iterable is valid because pilot mode writes a header-only CSV.
    Incomplete rows remain valid when their status explains the missing output.
    """
    errors: list[str] = []
    seen_cases: set[tuple[str, str, str, float]] = set()
    actions_by_case: dict[tuple[str, str, str], set[float]] = {}

    for row_number, row in enumerate(records, start=2):
        prefix = f"row {row_number}"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}: expected a mapping")
            continue

        for field in _REQUIRED_FIELDS:
            if field not in row:
                errors.append(f"{prefix}: missing required field {field!r}")

        method = _text(row.get("method"))
        if method not in _METHODS:
            errors.append(f"{prefix}: invalid method {method!r}")
        elif _text(row.get("prediction_space")) != _PREDICTION_SPACES[method]:
            errors.append(
                f"{prefix}: {method} prediction_space must be {_PREDICTION_SPACES[method]!r}"
            )
        if method in _METHODS and _text(row.get("scoring_space")) != _SCORING_SPACES[method]:
            errors.append(
                f"{prefix}: {method} scoring_space must be {_SCORING_SPACES[method]!r}"
            )
        device = _text(row.get("device")).lower()
        if device not in {"cpu", "cuda"}:
            errors.append(f"{prefix}: device must be 'cpu' or 'cuda', got {device!r}")

        raw_seed = _text(row.get("model_seed"))
        normalized_seed = ""
        if method == "oracle":
            if raw_seed:
                errors.append(f"{prefix}: oracle model_seed must be empty")
        elif method in _METHODS:
            if not raw_seed:
                errors.append(f"{prefix}: {method} model_seed must be an integer")
            else:
                try:
                    normalized_seed = str(int(raw_seed))
                except (TypeError, ValueError):
                    errors.append(f"{prefix}: model_seed must be an integer, got {raw_seed!r}")

        start_id = _text(row.get("start_id"))
        if not start_id:
            errors.append(f"{prefix}: start_id must be non-empty")

        action: float | None = None
        raw_action = _text(row.get("action"))
        try:
            numeric_action = float(raw_action)
            action = next(
                (candidate for candidate in _ACTIONS if math.isclose(numeric_action, candidate, rel_tol=0.0, abs_tol=1e-8)),
                None,
            )
        except (TypeError, ValueError):
            pass
        if action is None:
            errors.append(f"{prefix}: action must be one of -1, 0, or 1, got {raw_action!r}")

        status = _text(row.get("status"))
        if status not in _STATUSES:
            errors.append(f"{prefix}: invalid status {status!r}")
        elif status == "error" and not _text(row.get("error")):
            errors.append(f"{prefix}: error status requires a non-empty error description")

        numeric_values: dict[str, float] = {}
        for field in _METRIC_FIELDS:
            raw_value = _text(row.get(field))
            if not raw_value:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                errors.append(f"{prefix}: {field} must be numeric, got {raw_value!r}")
                continue
            if not math.isfinite(value):
                errors.append(f"{prefix}: {field} must be finite, got {raw_value!r}")
                continue
            numeric_values[field] = value

        for field in ("actual_position", "observed_score"):
            if field not in numeric_values:
                errors.append(f"{prefix}: {field} must contain a finite value")

        if status == "complete" and method in _REQUIRED_COMPLETE_METRICS:
            for field in _REQUIRED_COMPLETE_METRICS[method]:
                if field not in numeric_values:
                    errors.append(f"{prefix}: complete {method} record requires finite {field}")

        if method in _METHODS and start_id and (method == "oracle" or normalized_seed) and action is not None:
            case_key = (method, normalized_seed, start_id, action)
            if case_key in seen_cases:
                errors.append(f"{prefix}: duplicate case {case_key!r}")
            seen_cases.add(case_key)
            actions_by_case.setdefault(case_key[:3], set()).add(action)

    expected_actions = set(_ACTIONS)
    for case_key, actions in actions_by_case.items():
        if actions != expected_actions:
            missing = sorted(expected_actions - actions)
            extra = sorted(actions - expected_actions)
            errors.append(
                f"case {case_key!r}: expected actions -1, 0, and 1 exactly once; "
                f"missing={missing}, extra={extra}"
            )

    if errors:
        raise ValueError("Invalid intervention records:\n- " + "\n- ".join(errors))
