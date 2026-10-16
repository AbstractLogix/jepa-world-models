#!/usr/bin/env python3
"""Validate the complete, retained visual-JEPA decision-study evidence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_tutorial.decision_study import (  # noqa: E402
    DEV_MIN_SUCCESS,
    FINAL_ORACLE_MIN_SUCCESS,
    FINAL_TRANSITION_CALL_LIMIT,
    STUDY_VERSION,
    oracle_gate_is_valid,
    protocol_lock_is_valid,
    sha256_file,
    validate_episode_records,
    validate_intervention_records,
)
from scripts.run_visual_jepa_decision_study import (  # noqa: E402
    CONFIRMATION_SEEDS,
    DATA_SEEDS,
    DEV_SEEDS,
    FINAL_SEEDS,
    INTERVENTION_FIELDS,
    INTERVENTION_SEEDS,
    METHOD_FIELDS,
    MODEL_FAMILIES,
    MODEL_SEEDS,
    PROTOCOL_PATH,
    TEST_SEEDS,
    code_hash,
)

PREFIX = "visual-jepa-decision-study-v1"
PLANNER_SETTING_FIELDS = (
    "horizon_steps",
    "candidates",
    "elites",
    "updates",
    "score_name",
    "replan_interval",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_fields(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No data rows in {path}")
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")


def validate_development(output: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    gate = read_json(output / "development-gate.json")
    rows = read_csv(output / "development.csv")
    require_fields(rows, set(METHOD_FIELDS), output / "development.csv")
    if gate.get("experiment_version") != STUDY_VERSION or gate.get("status") != "passed":
        raise ValueError("Development gate did not pass")
    if int(gate.get("overall_successes", -1)) < DEV_MIN_SUCCESS:
        raise ValueError("Development success threshold was not met")
    if int(gate.get("confirmation_successes", -1)) < 9:
        raise ValueError("Development confirmation threshold was not met")
    expected_methods = {
        "dev_stage_a": 16 * len(DEV_SEEDS),
        "dev_stage_b": 4 * 5,
        "dev_locked_tuning": len(DEV_SEEDS),
        "dev_confirmation": len(CONFIRMATION_SEEDS),
    }
    methods = Counter(row["method"] for row in rows)
    if methods != Counter(expected_methods):
        raise ValueError(f"Development case counts mismatch: {methods}")
    calls = sum(int(row["transition_calls"]) for row in rows)
    if calls != int(gate["transition_calls"]):
        raise ValueError("Development transition-call total does not match its CSV")
    if calls > int(gate["transition_call_limit"]):
        raise ValueError("Development transition-call ceiling was exceeded")
    if len(gate.get("stage_a", [])) != 16 or len(gate.get("stage_b", [])) != 4:
        raise ValueError("Development search grid is incomplete")
    return gate, rows


def validate_model_runs(output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    model_rows = read_csv(output / "model-runs.csv")
    expected = {(family, str(seed)) for family in MODEL_FAMILIES for seed in MODEL_SEEDS}
    observed = {(row["method"], row["model_seed"]) for row in model_rows}
    if observed != expected or len(model_rows) != len(expected):
        raise ValueError(f"Model-run seed coverage mismatch: expected {len(expected)}, observed {len(model_rows)}")
    for row in model_rows:
        if row.get("status") not in {"complete", "failed"}:
            raise ValueError(f"Invalid model status for {row.get('method')} seed {row.get('model_seed')}")
        if row["status"] == "failed" and not row.get("error", "").strip():
            raise ValueError("Failed model rows need an error reason")
    paired_seeds = dict(zip((str(seed) for seed in MODEL_SEEDS), (str(seed) for seed in DATA_SEEDS), strict=True))
    paired_tests = dict(zip((str(seed) for seed in MODEL_SEEDS), (str(seed) for seed in TEST_SEEDS), strict=True))
    for row in model_rows:
        if row.get("data_seed") != paired_seeds[row["model_seed"]]:
            raise ValueError(f"Model seed {row['model_seed']} uses the wrong data seed")
        if row.get("test_seed") != paired_tests[row["model_seed"]]:
            raise ValueError(f"Model seed {row['model_seed']} uses the wrong representation-test seed")
    probe_rows = read_csv(output / "probe-test.csv")
    probe_expected = {
        (family, str(seed))
        for family in ("image_reconstruction", "jepa")
        for seed in MODEL_SEEDS
        if next(r for r in model_rows if r["method"] == family and r["model_seed"] == str(seed))["status"] == "complete"
    }
    probe_observed = {(row["family"], row["model_seed"]) for row in probe_rows}
    if probe_observed != probe_expected:
        raise ValueError(f"Probe-test coverage mismatch: expected {len(probe_expected)}, observed {len(probe_observed)}")
    for row in probe_rows:
        if row.get("test_seed") not in {str(seed) for seed in TEST_SEEDS}:
            raise ValueError("Probe result uses an undeclared test seed")
        if row.get("data_seed") not in {str(seed) for seed in DATA_SEEDS}:
            raise ValueError("Probe result uses an undeclared training-data seed")
    return model_rows, probe_rows


def validate_interventions(output: Path) -> list[dict[str, str]]:
    path = output / "interventions.csv"
    rows = read_csv(path)
    require_fields(rows, set(INTERVENTION_FIELDS), path)
    validate_intervention_records(rows)
    starts = {row["start_id"] for row in rows}
    expected_starts = {f"{seed}-{step}" for seed in INTERVENTION_SEEDS for step in (40, 90, 140, 190)}
    if starts != expected_starts:
        raise ValueError("Intervention start set does not match the preregistered seeds and steps")
    expected_all = {("oracle", "", start_id) for start_id in expected_starts}
    expected_all |= {
        (family, str(seed), start_id)
        for family in MODEL_FAMILIES
        for seed in MODEL_SEEDS
        for start_id in expected_starts
    }
    groups: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for row in rows:
        groups[(row["method"], row["model_seed"], row["start_id"])].add(float(row["action"]))
    if set(groups) != expected_all:
        raise ValueError(f"Intervention seed/start coverage mismatch: expected {len(expected_all)}, observed {len(groups)}")
    if any(actions != {-1.0, 0.0, 1.0} for actions in groups.values()):
        raise ValueError("An intervention group is missing one or more paired actions")
    if len(rows) != 20 * 3 * (1 + len(MODEL_FAMILIES) * len(MODEL_SEEDS)):
        raise ValueError(f"Unexpected intervention row count: {len(rows)}")
    return rows


def validate_planner_settings(
    rows: list[dict[str, str]], planner_config: dict[str, Any]
) -> None:
    for row in rows:
        if not row["method"].endswith("_cem"):
            continue
        for field in PLANNER_SETTING_FIELDS:
            if row[field] != str(planner_config[field]):
                raise ValueError(f"Planner episode {field} does not match the locked setting")


def validate_oracle_gate_rows(
    rows: list[dict[str, str]],
    gate: dict[str, Any],
    planner_config: dict[str, Any],
) -> None:
    """Check that the retained oracle cases support the recorded gate."""
    expected_seeds = {str(seed) for seed in FINAL_SEEDS}
    if len(rows) != len(expected_seeds):
        raise ValueError("Oracle gate must have one row per final start")
    validate_episode_records(rows)
    validate_planner_settings(rows, planner_config)
    observed_seeds = {row["eval_seed"] for row in rows}
    if observed_seeds != expected_seeds:
        raise ValueError("Oracle gate rows do not match the preregistered final starts")
    if any(
        row["method"] != "oracle_cem"
        or row["initialization"] != "momentum_prior"
        or row["privileged_state"].lower() != "true"
        or row["status"] != "complete"
        for row in rows
    ):
        raise ValueError("Oracle gate must contain complete exact-dynamics rows")
    successes = sum(int(row["success"]) for row in rows)
    transition_calls = sum(int(row["transition_calls"]) for row in rows)
    if successes != int(gate["successes"]):
        raise ValueError("Oracle gate success total does not match its episode rows")
    if transition_calls != int(gate["transition_calls"]):
        raise ValueError("Oracle gate transition-call total does not match its episode rows")
    if int(gate["threshold"]) != FINAL_ORACLE_MIN_SUCCESS:
        raise ValueError("Oracle gate threshold does not match the preregistered threshold")


def validate_episodes(output: Path, lock_hash: str) -> list[dict[str, str]]:
    lock = read_json(output / "protocol-lock.json")
    planner_config = lock["planner_config"]
    gate_record = read_json(output / "oracle-gate.json")
    if not oracle_gate_is_valid(gate_record, protocol_lock_hash=lock_hash):
        raise ValueError("Final oracle gate is missing, stale, or failed")
    if int(gate_record.get("successes", 0)) < FINAL_ORACLE_MIN_SUCCESS:
        raise ValueError("Final oracle gate success threshold was not met")
    gate_rows = read_csv(output / "oracle-gate.csv")
    validate_oracle_gate_rows(gate_rows, gate_record, planner_config)
    path = output / "episodes.csv"
    rows = read_csv(path)
    require_fields(rows, set(METHOD_FIELDS), path)
    validate_episode_records(rows)
    validate_planner_settings(rows, planner_config)
    expected = {("oracle_cem", "", str(seed), init) for seed in FINAL_SEEDS for init in ("uniform", "momentum_prior")}
    expected |= {("direct_visual_momentum", "", str(seed), "direct") for seed in FINAL_SEEDS}
    expected |= {
        (f"{family}_cem", str(model_seed), str(eval_seed), init)
        for family in MODEL_FAMILIES
        for model_seed in MODEL_SEEDS
        for eval_seed in FINAL_SEEDS
        for init in ("uniform", "momentum_prior")
    }
    observed = {(row["method"], row["model_seed"], row["eval_seed"], row["initialization"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"Episode case coverage mismatch: expected {len(expected)}, observed {len(rows)}")
    total_calls = sum(int(row["transition_calls"]) for row in rows if row.get("status") == "complete")
    if total_calls > FINAL_TRANSITION_CALL_LIMIT:
        raise ValueError("Final transition-call ceiling was exceeded")
    return rows


def validate(output: Path) -> dict[str, int]:
    lock = read_json(output / "protocol-lock.json")
    if not protocol_lock_is_valid(
        lock,
        preregistration_hash=sha256_file(str(PROTOCOL_PATH)),
        code_hash=code_hash(),
    ):
        raise ValueError("Protocol lock does not match this preregistration and source")
    lock_hash = sha256_file(str(output / "protocol-lock.json"))
    _development, dev_rows = validate_development(output)
    model_rows, probe_rows = validate_model_runs(output)
    intervention_rows = validate_interventions(output)
    episodes = validate_episodes(output, lock_hash)
    report = read_json(output / "report.json")
    if report.get("protocol_lock_hash") != lock_hash:
        raise ValueError("Final report protocol-lock hash mismatch")
    if int(report.get("final_transition_calls", -1)) > FINAL_TRANSITION_CALL_LIMIT:
        raise ValueError("Final report exceeds the planner-call ceiling")
    expected_calls = sum(
        int(row["transition_calls"])
        for row in episodes
        if row.get("status") == "complete"
    )
    if int(report.get("final_transition_calls", -1)) != expected_calls:
        raise ValueError("Final report transition-call total does not match episode rows")
    return {
        "development_rows": len(dev_rows),
        "model_runs": len(model_rows),
        "probe_tests": len(probe_rows),
        "intervention_rows": len(intervention_rows),
        "episode_rows": len(episodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = validate(args.output_dir.expanduser().resolve())
    print(f"Evidence validated for {STUDY_VERSION}:")
    for key, count in counts.items():
        print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
