"""Validate v3 evidence hashes, run coverage, and figure integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from run_visual_jepa_supervised_positive_control_v3 import (
    EVIDENCE_DIR,
    METHOD_HANDCRAFTED_EXACT,
    TEST_SEEDS,
    TEST_STEPS,
    V3_VERSION,
    StudyError,
    can_open_heldout,
    verify_protocol_lock,
)

from jepa_tutorial.action_fidelity_v2 import (
    validate_action_triplet_records,
    validate_rollout_records,
)
from jepa_tutorial.supervised_visual_v3 import validate_method_run_coverage

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs/portfolio/figures/visual-jepa-supervised-positive-control-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(evidence_dir: Path = EVIDENCE_DIR) -> int:
    manifest_path = evidence_dir / "sha256-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"evidence manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != V3_VERSION:
        raise StudyError("evidence manifest protocol version is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise StudyError("evidence manifest has no file hashes")
    checked = 0
    for relative_name, expected_hash in files.items():
        path = evidence_dir / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {relative_name}")
        expected_sha256 = expected_hash.get("sha256") if isinstance(expected_hash, dict) else expected_hash
        expected_bytes = expected_hash.get("bytes") if isinstance(expected_hash, dict) else None
        if _sha256(path) != expected_sha256:
            raise StudyError(f"evidence hash mismatch: {relative_name}")
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            raise StudyError(f"evidence size mismatch: {relative_name}")
        checked += 1
    return checked


def validate_figures(
    figure_dir: Path = FIGURE_DIR, evidence_dir: Path | None = None
) -> int:
    manifest_path = figure_dir / "figure-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"figure manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != V3_VERSION:
        raise StudyError("figure manifest protocol version is invalid")
    figures = manifest.get("figures")
    if not isinstance(figures, list) or not figures:
        raise StudyError("figure manifest contains no figures")
    if evidence_dir is not None:
        for name, record in manifest.get("source_evidence", {}).items():
            source = evidence_dir / name
            if not source.is_file() or _sha256(source) != record.get("sha256"):
                raise StudyError(f"figure source evidence mismatch: {name}")
            if source.stat().st_size != int(record.get("bytes", -1)):
                raise StudyError(f"figure source evidence size mismatch: {name}")
    checked = 0
    seen_ids: set[str] = set()
    for item in figures:
        figure_id = str(item.get("id", ""))
        if not figure_id or figure_id in seen_ids:
            raise StudyError("figure IDs must be present and unique")
        seen_ids.add(figure_id)
        if not item.get("caption") or not item.get("alt_text") or not item.get("limitations"):
            raise StudyError(f"figure {figure_id} lacks caption, alt text, or limitations")
        hashes = item.get("sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise StudyError(f"figure {figure_id} has no file hashes")
        for filename, expected_hash in hashes.items():
            path = figure_dir / filename
            expected_sha256 = expected_hash.get("sha256") if isinstance(expected_hash, dict) else expected_hash
            expected_bytes = expected_hash.get("bytes") if isinstance(expected_hash, dict) else None
            if not path.is_file() or _sha256(path) != expected_sha256:
                raise StudyError(f"figure hash mismatch: {filename}")
            if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
                raise StudyError(f"figure size mismatch: {filename}")
            checked += 1
    return checked


def validate_study(evidence_dir: Path = EVIDENCE_DIR) -> dict[str, object]:
    verify_protocol_lock()
    checked_files = validate_manifest(evidence_dir)
    summary_path = evidence_dir / "run-summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("v3 run summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_version") != V3_VERSION:
        raise StudyError("v3 run summary protocol version is invalid")
    if summary.get("status") == "data_path_failed":
        if (
            summary.get("fit_count") != 0
            or summary.get("heldout_open")
            or summary.get("records_complete")
        ):
            raise StudyError("failed data path must stop before fitting and held-out access")
        data_path_path = evidence_dir / "data-path.csv"
        if not data_path_path.is_file():
            raise FileNotFoundError("data-path failure has no data-path records")
        data_path_rows = _read_csv(data_path_path)
        if not data_path_rows or all(row.get("passed") == "True" for row in data_path_rows):
            raise StudyError("data-path failure summary has no recorded failed positive control")
        model_rows = _read_csv(evidence_dir / "model-runs.csv")
        validate_method_run_coverage(model_rows)
        if len(model_rows) != 21:
            raise StudyError("data-path failure must retain all 21 planned arm records")
        for row in model_rows:
            if row["method"] == METHOD_HANDCRAFTED_EXACT:
                if row["status"] not in ("complete", "failed"):
                    raise StudyError("exact-dynamics data-path control has no outcome")
            elif row["status"] != "not_run" or not row.get("error"):
                raise StudyError("every later arm must be marked not_run with a stop reason")
        for filename in ("validation-predictions.csv", "action-triplets.csv"):
            path = evidence_dir / filename
            if path.is_file() and path.stat().st_size > 0:
                rows = _read_csv(path)
                if rows:
                    raise StudyError("data-path failure contains learned validation predictions")
        return {
            "valid": True,
            "status": "data_path_failed",
            "checked_evidence_files": checked_files,
            "heldout_open": False,
        }

    required = (
        "model-runs.csv",
        "validation-predictions.csv",
        "action-triplets.csv",
        "learning-curves.csv",
        "gradient-diagnostics.csv",
        "embedding-diagnostics.csv",
        "linear-probes.csv",
        "validation-gates.csv",
    )
    missing = [name for name in required if not (evidence_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required v3 evidence is missing: {missing}")
    model_rows = _read_csv(evidence_dir / "model-runs.csv")
    if len(model_rows) != 21:
        raise StudyError(f"expected 21 baseline and neural run rows, found {len(model_rows)}")
    validate_method_run_coverage(model_rows)
    embedding_rows = _read_csv(evidence_dir / "embedding-diagnostics.csv")
    probe_rows = _read_csv(evidence_dir / "linear-probes.csv")
    action_rows = _read_csv(evidence_dir / "action-triplets.csv")
    validate_action_triplet_records(action_rows)
    if summary.get("status") == "complete":
        neural_methods = (
            "visual_state_estimator",
            "supervised_visual_absolute_raw",
            "supervised_visual_absolute_standardized",
            "supervised_visual_residual",
            "supervised_visual_action_shuffle",
        )
        expected_neural_groups = {
            (method, str(model_seed), str(data_seed))
            for method in neural_methods
            for model_seed, data_seed in zip((9801, 9802, 9803), (9701, 9702, 9703), strict=True)
        }
        embedding_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in embedding_rows:
            key = (row["method"], row["model_seed"], row["data_seed"])
            embedding_groups.setdefault(key, []).append(row)
            if int(row["rows"]) <= 0 or not all(
                math.isfinite(float(row[field])) for field in ("mean", "std", "variance")
            ):
                raise StudyError("embedding diagnostic contains invalid spread, mean, variance, or row count")
        if set(embedding_groups) != expected_neural_groups:
            raise StudyError("complete validation evidence has missing or extra embedding diagnostic groups")
        for rows in embedding_groups.values():
            dimensions = {int(row["dimension"]) for row in rows}
            if len(rows) != 32 or dimensions != set(range(32)):
                raise StudyError("each learned representation must contain one record for every embedding dimension")
        probe_groups = {
            (row["method"], row["model_seed"], row["data_seed"])
            for row in probe_rows
        }
        if probe_groups != expected_neural_groups or len(probe_rows) != len(expected_neural_groups):
            raise StudyError("complete validation evidence has missing or duplicate state-probe records")
        if any(
            not math.isfinite(float(row[field]))
            for row in probe_rows
            for field in ("probe_position_r2", "probe_velocity_r2", "probe_gram_condition")
        ):
            raise StudyError("state-probe evidence contains non-finite diagnostics")
        action_methods = (
            "handcrafted_state_exact_dynamics",
            "handcrafted_feature_ridge",
            "supervised_visual_absolute_raw",
            "supervised_visual_absolute_standardized",
            "supervised_visual_residual",
            "supervised_visual_action_shuffle",
        )
        expected_action_groups = set()
        for data_seed, model_seed in zip((9701, 9702, 9703), (9801, 9802, 9803), strict=True):
            expected_action_groups.update(
                (method, "0", str(data_seed))
                for method in action_methods[:2]
            )
            expected_action_groups.update(
                (method, str(model_seed), str(data_seed))
                for method in action_methods[2:]
            )
        action_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in action_rows:
            key = (row["method"], row["model_seed"], row["data_seed"])
            action_groups.setdefault(key, []).append(row)
        if set(action_groups) != expected_action_groups:
            raise StudyError("complete validation evidence has missing or extra method/seed action groups")
        if any(
            len(rows) != 60 or len({row["start_id"] for row in rows}) != 20
            for rows in action_groups.values()
        ):
            raise StudyError("each validation action group must contain 20 complete triplets")

        partition_path = evidence_dir / "partition-map.csv"
        prediction_path = evidence_dir / "validation-predictions.csv"
        if not partition_path.is_file():
            raise FileNotFoundError("complete validation partition map is missing")
        partition_rows = _read_csv(partition_path)
        validation_sources: dict[str, set[str]] = {}
        for row in partition_rows:
            if row["partition"] == "validation":
                validation_sources.setdefault(row["data_seed"], set()).add(row["source_row"])
        observed_predictions = _read_csv(prediction_path)
        observed_prediction_keys = {
            (row["method"], row["model_seed"], row["data_seed"], row["source_row"])
            for row in observed_predictions
        }
        if len(observed_prediction_keys) != len(observed_predictions):
            raise StudyError("duplicate row-level validation prediction")
        expected_prediction_keys = {
            (row["method"], row["model_seed"], row["data_seed"], source_row)
            for row in model_rows
            for source_row in validation_sources.get(row["data_seed"], set())
        }
        if observed_prediction_keys != expected_prediction_keys:
            raise StudyError("row-level prediction records do not match validation partitions and model runs")
    if summary.get("heldout_open") and not can_open_heldout(summary):
        raise StudyError("run summary claims held-out access without every validation gate")
    if summary.get("heldout_open"):
        heldout_required = (
            "heldout-starts.csv",
            "heldout-state-estimates.csv",
            "rollout-trajectories.csv",
            "rollout-horizons.csv",
            "heldout-summary.json",
        )
        missing_heldout = [name for name in heldout_required if not (evidence_dir / name).is_file()]
        if missing_heldout:
            raise FileNotFoundError(f"gated held-out evidence is incomplete: {missing_heldout}")
        heldout_summary = json.loads(
            (evidence_dir / "heldout-summary.json").read_text(encoding="utf-8")
        )
        start_rows = _read_csv(evidence_dir / "heldout-starts.csv")
        expected_starts = {
            (str(test_seed), str(step))
            for test_seed in TEST_SEEDS
            for step in TEST_STEPS
        }
        observed_starts = {
            (row["test_seed"], row["step"]): row for row in start_rows
        }
        if set(observed_starts) != expected_starts or len(start_rows) != len(expected_starts):
            raise StudyError("held-out start records must include every scheduled seed and step")
        missing_starts = [
            row for row in start_rows if row.get("status") == "missing"
        ]
        if any(not row.get("reason") for row in missing_starts):
            raise StudyError("each missing held-out start must include a reason")
        completed_count = sum(row.get("status") == "complete" for row in start_rows)
        if (
            heldout_summary.get("expected_start_contexts") != len(expected_starts)
            or heldout_summary.get("observed_start_contexts") != completed_count
            or (heldout_summary.get("status") == "complete") != (not missing_starts)
        ):
            raise StudyError("held-out start coverage summary does not match per-start records")
        validate_rollout_records(_read_csv(evidence_dir / "rollout-trajectories.csv"))
    else:
        for name in (
            "heldout-starts.csv",
            "heldout-state-estimates.csv",
            "rollout-trajectories.csv",
            "rollout-horizons.csv",
            "heldout-summary.json",
        ):
            if (evidence_dir / name).exists():
                raise StudyError("held-out evidence exists while the validation gate is closed")
    return {
        "valid": True,
        "status": summary.get("status"),
        "checked_evidence_files": checked_files,
        "model_run_rows": len(model_rows),
        "action_triplet_rows": len(action_rows),
        "heldout_open": bool(summary.get("heldout_open")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--check-integrity", action="store_true")
    parser.add_argument("--check-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.check_integrity:
        checked = validate_manifest(args.evidence_dir)
        verify_protocol_lock()
        result: dict[str, object] = {"valid": True, "checked_evidence_files": checked}
    else:
        result = validate_study(args.evidence_dir)
    if args.check_figures:
        result["checked_figure_files"] = validate_figures(
            args.figure_dir, evidence_dir=args.evidence_dir
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
