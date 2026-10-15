"""Validate retained episode rows from Notebooks 20 and 21."""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

POLICY_NOISE_COLUMNS = (
    "method",
    "evaluation_seed",
    "start_id",
    "start",
    "success",
    "steps",
    "model_calls",
    "final_position",
    "mean_selected_action_error",
    "off_support_fraction",
)
POLICY_NOISE_METHODS = {"raw_action", "policy_noise", "direct_control"}
POLICY_NOISE_SEEDS = {101, 202, 303, 404, 505}

SAMPLING_COLUMNS = (
    "method",
    "group",
    "group_seed",
    "paired_start",
    "start_x",
    "start_y",
    "success",
    "status",
    "steps",
    "transition_calls",
    "runtime_seconds",
    "final_x",
    "final_y",
)
SAMPLING_METHODS = {"RRT", "KPIECE", "CEM"}
SAMPLING_STATUSES = {"success", "episode_limit", "invalid_planned_action"}
SAMPLING_GROUP_SEEDS = {1: 2101, 2: 2102, 3: 2103, 4: 2104, 5: 2105}


def _read_rows(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read a CSV with an exact header and no blank or malformed records."""
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"{path}: file is empty") from exc

            if tuple(header) != expected_columns:
                raise ValueError(
                    f"{path}: columns do not match the expected schema; "
                    f"expected {list(expected_columns)}, found {header}"
                )

            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if not values or all(not value.strip() for value in values):
                    raise ValueError(f"{path}: blank record at line {line_number}")
                if len(values) != len(expected_columns):
                    raise ValueError(
                        f"{path}: line {line_number} has {len(values)} fields; "
                        f"expected {len(expected_columns)}"
                    )
                rows.append(dict(zip(expected_columns, values, strict=True)))
    except (OSError, csv.Error, UnicodeError) as exc:
        raise ValueError(f"{path}: cannot read CSV: {exc}") from exc

    return rows


def _integer(value: str, path: Path, line: int, field: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{path}: line {line} has invalid integer in {field}: {value!r}") from exc


def _number(value: str, path: Path, line: int, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}: line {line} has invalid number in {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: line {line} has non-finite number in {field}: {value!r}")
    return result


def _boolean(value: str, path: Path, line: int, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{path}: line {line} has invalid boolean in {field}: {value!r}")


def _check_count(rows: list[dict[str, str]], path: Path) -> None:
    if len(rows) != 300:
        raise ValueError(f"{path}: expected 300 records, found {len(rows)}")


def validate_policy_noise(path: Path) -> str:
    """Validate the saved Notebook 20 policy-noise episode records."""
    rows = _read_rows(path, POLICY_NOISE_COLUMNS)
    _check_count(rows, path)

    methods_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    starts_by_seed: dict[int, set[int]] = defaultdict(set)
    start_value_by_pair: dict[tuple[int, int], float] = {}
    keys: set[tuple[int, int, str]] = set()

    for line, row in enumerate(rows, start=2):
        method = row["method"]
        if method not in POLICY_NOISE_METHODS:
            raise ValueError(f"{path}: line {line} has unknown method: {method!r}")

        seed = _integer(row["evaluation_seed"], path, line, "evaluation_seed")
        start_id = _integer(row["start_id"], path, line, "start_id")
        if seed not in POLICY_NOISE_SEEDS:
            raise ValueError(f"{path}: line {line} has unexpected evaluation_seed: {seed}")
        if not 0 <= start_id < 20:
            raise ValueError(f"{path}: line {line} has start_id outside 0–19: {start_id}")

        key = (seed, start_id, method)
        if key in keys:
            raise ValueError(f"{path}: line {line} duplicates method/start key {key}")
        keys.add(key)
        methods_by_pair[(seed, start_id)].add(method)
        starts_by_seed[seed].add(start_id)

        start_value = _number(row["start"], path, line, "start")
        pair = (seed, start_id)
        if pair in start_value_by_pair and start_value_by_pair[pair] != start_value:
            raise ValueError(f"{path}: line {line} has a different start value for paired key {pair}")
        start_value_by_pair[pair] = start_value
        _boolean(row["success"], path, line, "success")
        if _integer(row["steps"], path, line, "steps") < 0:
            raise ValueError(f"{path}: line {line} has negative steps")
        if _integer(row["model_calls"], path, line, "model_calls") < 0:
            raise ValueError(f"{path}: line {line} has negative model_calls")
        _number(row["final_position"], path, line, "final_position")
        planner_metrics = (
            row["mean_selected_action_error"],
            row["off_support_fraction"],
        )
        if method == "direct_control":
            if any(value.strip() for value in planner_metrics):
                raise ValueError(
                    f"{path}: line {line} has planner-only metrics for direct_control"
                )
        else:
            if any(not value.strip() for value in planner_metrics):
                raise ValueError(f"{path}: line {line} is missing a planner-only metric")
            if _number(row["mean_selected_action_error"], path, line, "mean_selected_action_error") < 0:
                raise ValueError(f"{path}: line {line} has negative mean_selected_action_error")
            off_support = _number(row["off_support_fraction"], path, line, "off_support_fraction")
            if not 0 <= off_support <= 1:
                raise ValueError(f"{path}: line {line} has off_support_fraction outside [0, 1]")

    expected_pairs = {(seed, start_id) for seed in POLICY_NOISE_SEEDS for start_id in range(20)}
    if set(methods_by_pair) != expected_pairs:
        raise ValueError(f"{path}: expected 100 paired seed/start combinations")
    if any(methods != POLICY_NOISE_METHODS for methods in methods_by_pair.values()):
        raise ValueError(f"{path}: every paired seed/start must contain all three methods")
    if any(starts != set(range(20)) for starts in starts_by_seed.values()):
        raise ValueError(f"{path}: every evaluation seed must contain start IDs 0–19")

    return "300 records, 5 evaluation seeds, 100 paired starts, 3 methods per start"


def validate_sampling_based(path: Path) -> str:
    """Validate the saved Notebook 21 sampling-based planning records."""
    rows = _read_rows(path, SAMPLING_COLUMNS)
    _check_count(rows, path)

    methods_by_pair: dict[tuple[int, str], set[str]] = defaultdict(set)
    starts_by_group: dict[int, set[str]] = defaultdict(set)
    seeds_by_group: dict[int, set[int]] = defaultdict(set)
    start_state_by_pair: dict[tuple[int, str], tuple[float, float]] = {}
    keys: set[tuple[int, str, str]] = set()

    for line, row in enumerate(rows, start=2):
        method = row["method"]
        if method not in SAMPLING_METHODS:
            raise ValueError(f"{path}: line {line} has unknown method: {method!r}")

        group = _integer(row["group"], path, line, "group")
        group_seed = _integer(row["group_seed"], path, line, "group_seed")
        if group not in SAMPLING_GROUP_SEEDS:
            raise ValueError(f"{path}: line {line} has unexpected group: {group}")
        if group_seed != SAMPLING_GROUP_SEEDS[group]:
            raise ValueError(
                f"{path}: line {line} has group_seed {group_seed} for group {group}; "
                f"expected {SAMPLING_GROUP_SEEDS[group]}"
            )

        paired_start = row["paired_start"]
        key = (group, paired_start, method)
        if key in keys:
            raise ValueError(f"{path}: line {line} duplicates method/start key {key}")
        keys.add(key)
        methods_by_pair[(group, paired_start)].add(method)
        starts_by_group[group].add(paired_start)
        seeds_by_group[group].add(group_seed)

        start_state = (
            _number(row["start_x"], path, line, "start_x"),
            _number(row["start_y"], path, line, "start_y"),
        )
        pair = (group, paired_start)
        if pair in start_state_by_pair and start_state_by_pair[pair] != start_state:
            raise ValueError(f"{path}: line {line} has a different start state for paired key {pair}")
        start_state_by_pair[pair] = start_state
        success = _boolean(row["success"], path, line, "success")
        status = row["status"]
        if status not in SAMPLING_STATUSES:
            raise ValueError(f"{path}: line {line} has unknown status: {status!r}")
        if success != (status == "success"):
            raise ValueError(f"{path}: line {line} has success/status mismatch")
        if _integer(row["steps"], path, line, "steps") < 0:
            raise ValueError(f"{path}: line {line} has negative steps")
        if _integer(row["transition_calls"], path, line, "transition_calls") < 0:
            raise ValueError(f"{path}: line {line} has negative transition_calls")
        if _number(row["runtime_seconds"], path, line, "runtime_seconds") < 0:
            raise ValueError(f"{path}: line {line} has negative runtime_seconds")
        _number(row["final_x"], path, line, "final_x")
        _number(row["final_y"], path, line, "final_y")

    if set(starts_by_group) != set(SAMPLING_GROUP_SEEDS):
        raise ValueError(f"{path}: expected groups 1–5")
    if any(len(starts) != 20 for starts in starts_by_group.values()):
        raise ValueError(f"{path}: each group must contain 20 paired starts")
    if any(seeds != {SAMPLING_GROUP_SEEDS[group]} for group, seeds in seeds_by_group.items()):
        raise ValueError(f"{path}: each group must map to its declared group seed")
    if len({start for starts in starts_by_group.values() for start in starts}) != 100:
        raise ValueError(f"{path}: paired_start identities must cover 100 distinct starts")
    if any(methods != SAMPLING_METHODS for methods in methods_by_pair.values()):
        raise ValueError(f"{path}: every paired group/start must contain all three methods")
    if len(methods_by_pair) != 100:
        raise ValueError(f"{path}: expected 100 paired group/start combinations")

    return "300 records, 5 groups, 100 paired starts, 3 methods per start"


def main() -> int:
    """Validate both retained evidence files without modifying them."""
    repository_root = Path(__file__).resolve().parents[1]
    checks = (
        (repository_root / "docs/evidence/policy-noise-episodes.csv", validate_policy_noise),
        (repository_root / "docs/evidence/sampling-based-episodes.csv", validate_sampling_based),
    )
    failed = False
    for path, validator in checks:
        try:
            result = validator(path)
        except ValueError as error:
            print(f"FAIL {path.relative_to(repository_root)}: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"PASS {path.relative_to(repository_root)}: {result}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
