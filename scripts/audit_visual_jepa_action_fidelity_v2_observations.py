"""Audit whether rendered frame pairs contain position and velocity information."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_tutorial.action_fidelity_v2 import split_episode_indices  # noqa: E402
from jepa_tutorial.decision_study import mountaincar_step  # noqa: E402
from jepa_tutorial.visual_mountaincar import collect_random_transitions, visual_state  # noqa: E402

DATA_SEEDS = (9301, 9302, 9303)
PROTOCOL_LOCK = ROOT / "docs/evidence/visual-jepa-action-fidelity-v2-protocol-lock.json"
CSV_NAME = "visual-jepa-action-fidelity-v2-observation-control.csv"
SUMMARY_NAME = "visual-jepa-action-fidelity-v2-observation-control-summary.json"
MANIFEST_NAME = "visual-jepa-action-fidelity-v2-observation-control-manifest.json"
SOURCE_FILES = (
    Path("scripts/audit_visual_jepa_action_fidelity_v2_observations.py"),
    Path("src/jepa_tutorial/action_fidelity_v2.py"),
    Path("src/jepa_tutorial/decision_study.py"),
    Path("src/jepa_tutorial/visual_mountaincar.py"),
    Path("docs/evidence/visual-jepa-action-fidelity-v2-protocol-lock.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _errors(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float, float, float]:
    errors = np.abs(np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64))
    return (
        float(errors[:, 0].mean()),
        float(errors[:, 1].mean()),
        float(np.quantile(errors[:, 0], 0.95)),
        float(np.quantile(errors[:, 1], 0.95)),
    )


def audit_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    simulator_max_error = 0.0
    total_rows = {"train": 0, "validation": 0}
    for data_seed in DATA_SEEDS:
        data = collect_random_transitions(data_seed, episodes=20)
        train_indices, validation_indices = split_episode_indices(data.episode_ids)
        for split, indices in (("train", train_indices), ("validation", validation_indices)):
            current_estimates = np.stack(
                [visual_state(data.frames[index, 0], data.frames[index, 1]) for index in indices]
            )
            next_estimates = np.stack(
                [visual_state(data.frames[index, 1], data.frames[index, 2]) for index in indices]
            )
            current_errors = _errors(current_estimates, data.states[indices])
            next_errors = _errors(next_estimates, data.next_states[indices])
            exact_next = mountaincar_step(data.states[indices], data.actions[indices])
            transition_error = np.abs(exact_next - data.next_states[indices])
            max_transition_error = float(transition_error.max())
            simulator_max_error = max(simulator_max_error, max_transition_error)
            episode_count = len(np.unique(data.episode_ids[indices]))
            total_rows[split] += len(indices)
            records.append(
                {
                    "protocol_version": "visual-jepa-action-fidelity-v2",
                    "data_seed": data_seed,
                    "split": split,
                    "episode_count": episode_count,
                    "transition_count": int(len(indices)),
                    "current_position_mae": current_errors[0],
                    "current_velocity_mae": current_errors[1],
                    "current_position_error_p95": current_errors[2],
                    "current_velocity_error_p95": current_errors[3],
                    "next_position_mae": next_errors[0],
                    "next_velocity_mae": next_errors[1],
                    "next_position_error_p95": next_errors[2],
                    "next_velocity_error_p95": next_errors[3],
                    "simulator_next_state_max_abs_error": max_transition_error,
                    "access": "rendered frame pair for estimate; simulator state only for audit target",
                }
            )

    summary: dict[str, object] = {
        "protocol_version": "visual-jepa-action-fidelity-v2",
        "status": "complete",
        "purpose": "observation and label audit; no model fitting",
        "data_seeds": list(DATA_SEEDS),
        "episode_partition": "episodes 0-15 train; episodes 16-19 validation",
        "train_transition_count": total_rows["train"],
        "validation_transition_count": total_rows["validation"],
        "simulator_max_abs_error": simulator_max_error,
        "test_data_opened": False,
        "interpretation": (
            "The deterministic two-frame visual estimator recovers current and next state on both "
            "training and validation episodes. This shows the rendered observations contain the "
            "state information needed by this task; it does not establish that a learned visual "
            "model can recover it under the frozen training recipe."
        ),
    }
    return records, summary


def _write_or_check(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different retained evidence: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _csv_bytes(records: list[dict[str, object]]) -> bytes:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_evidence(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    records, summary = audit_rows()
    csv_path = output_dir / CSV_NAME
    summary_path = output_dir / SUMMARY_NAME
    _write_or_check(csv_path, _csv_bytes(records))
    _write_or_check(summary_path, _json_bytes(summary))
    sources = {
        path.as_posix(): sha256(ROOT / path)
        for path in SOURCE_FILES
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": "visual-jepa-action-fidelity-v2",
        "sources": sources,
        "files": {
            CSV_NAME: sha256(csv_path),
            SUMMARY_NAME: sha256(summary_path),
        },
    }
    manifest_path = output_dir / MANIFEST_NAME
    _write_or_check(manifest_path, _json_bytes(manifest))
    verify_integrity(output_dir)
    return {
        "status": "pass",
        "rows": len(records),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
    }


def verify_integrity(output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        path = output_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"evidence file is missing or changed: {path}")
    for name, expected in manifest.get("sources", {}).items():
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source file is missing or changed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/evidence")
    parser.add_argument("--check-integrity", action="store_true")
    args = parser.parse_args()
    if args.check_integrity:
        verify_integrity(args.output_dir)
        result: dict[str, object] = {"integrity": "pass", "manifest": str(args.output_dir / MANIFEST_NAME)}
    else:
        result = write_evidence(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
