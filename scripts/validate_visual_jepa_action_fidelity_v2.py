"""Validate, retain, and hash the public-safe evidence for action-fidelity v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_visual_jepa_action_fidelity_v2 as runner  # noqa: E402

from jepa_tutorial.action_fidelity_v2 import (  # noqa: E402
    validate_experiment_completeness,
    validate_file_manifest,
    validate_rollout_records,
)

EVIDENCE_DIR = ROOT / "docs/evidence"
MANIFEST_PATH = EVIDENCE_DIR / "visual-jepa-action-fidelity-v2-evidence-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_run(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    summary_path = run_dir / "validation-summary.json"
    model_path = run_dir / "validation-model-runs.csv"
    action_path = run_dir / "validation-action-triplets.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"validation summary is missing: {summary_path}")
    if not model_path.is_file() or not action_path.is_file():
        raise FileNotFoundError("validation fit and action-triplet records are required")
    runner.verify_protocol_lock()
    summary = read_json(summary_path)
    model_rows = read_csv(model_path)
    action_rows = read_csv(action_path)
    validate_experiment_completeness(model_rows, action_rows)
    if len(model_rows) != 15:
        raise ValueError(f"expected 15 candidate fits, found {len(model_rows)}")
    heldout_path = run_dir / "heldout" / "heldout-summary.json"
    heldout_summary: dict[str, Any] | None = None
    if summary.get("heldout_open"):
        if not heldout_path.is_file():
            raise FileNotFoundError("validation gates opened held-out evaluation, but held-out records are missing")
        heldout_summary = read_json(heldout_path)
        if heldout_summary.get("status") != "complete" or not heldout_summary.get("test_seed_groups_opened"):
            raise ValueError("held-out evaluation did not complete after the validation gate opened")
        heldout_actions_path = run_dir / "heldout" / "heldout-action-triplets.csv"
        rollouts_path = run_dir / "heldout" / "heldout-rollouts.csv"
        if not heldout_actions_path.is_file() or not rollouts_path.is_file():
            raise FileNotFoundError("held-out action and rollout records are incomplete")
        validate_rollout_records(read_csv(rollouts_path))
        if heldout_summary.get("planning_open"):
            planning_path = run_dir / "planning" / "planning-summary.json"
            if not planning_path.is_file():
                raise FileNotFoundError("held-out gates opened planning, but planning results are missing")
            planning = read_json(planning_path)
            if planning.get("status") not in {"complete", "budget_stopped"} or not planning.get("planning_open"):
                raise ValueError("planning did not complete or stop at its declared budget")
            episode_path = run_dir / "planning" / "planning-episodes.csv"
            arm_summary_path = run_dir / "planning" / "planning-arm-summary.csv"
            if not episode_path.is_file() or not arm_summary_path.is_file():
                raise FileNotFoundError("planning episode and arm-summary records are required")
            if not read_csv(episode_path):
                raise ValueError("planning stage has no episode records")
    elif heldout_path.is_file():
        heldout_summary = read_json(heldout_path)
        if heldout_summary.get("test_seed_groups_opened"):
            raise ValueError("held-out test data were opened despite a closed validation gate")
    planning_authorized = bool(heldout_summary and heldout_summary.get("planning_open"))
    if not planning_authorized and (run_dir / "planning" / "planning-summary.json").exists():
        planning = read_json(run_dir / "planning" / "planning-summary.json")
        if planning.get("status") != "closed_by_heldout_gate" or planning.get("planning_open"):
            raise ValueError("planning results exist although the held-out gate remained closed")
    return {
        "validation_status": summary.get("status"),
        "fit_count": len(model_rows),
        "successful_fits": sum(row.get("status") == "complete" for row in model_rows),
        "failed_fits": sum(row.get("status") != "complete" for row in model_rows),
        "visual_positive_control_pass": summary.get("visual_positive_control_pass"),
        "selected_objective": summary.get("selected_objective"),
        "heldout_open": bool(summary.get("heldout_open")),
        "heldout_status": heldout_summary.get("status") if heldout_summary else "not opened",
        "planning_open": bool(heldout_summary and heldout_summary.get("planning_open")),
        "protocol_lock_verified": True,
    }


def _copy_record(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        destination_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        if source_hash != destination_hash:
            raise FileExistsError(f"refusing to replace retained evidence: {destination}")
    else:
        shutil.copy2(source, destination)
    return {
        "path": str(destination.relative_to(ROOT)),
        "size_bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def retain_public_evidence(run_dir: Path, v1_audit_dir: Path | None = None) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    v1_audit_dir = v1_audit_dir.expanduser().resolve() if v1_audit_dir else None
    validate_run(run_dir)
    sources: list[tuple[Path, str]] = []
    for name in (
        "validation-model-runs.csv",
        "validation-training-history.csv",
        "validation-gradient-diagnostics.csv",
        "validation-action-triplets.csv",
        "validation-action-summary.csv",
        "validation-data-splits.json",
        "validation-summary.json",
    ):
        path = run_dir / name
        if path.is_file():
            sources.append((path, f"visual-jepa-action-fidelity-v2-{name}"))
    heldout_dir = run_dir / "heldout"
    for name in (
        "heldout-action-triplets.csv",
        "heldout-rollouts.csv",
        "heldout-pair-gates.csv",
        "heldout-latent-references.json",
        "heldout-summary.json",
    ):
        path = heldout_dir / name
        if path.is_file():
            sources.append((path, f"visual-jepa-action-fidelity-v2-{name}"))
    planning_dir = run_dir / "planning"
    if (planning_dir / "planning-summary.json").is_file():
        for path in sorted(planning_dir.glob("planning-*.csv")):
            sources.append((path, f"visual-jepa-action-fidelity-v2-{path.name}"))
        sources.append((planning_dir / "planning-summary.json", "visual-jepa-action-fidelity-v2-planning-summary.json"))
    if v1_audit_dir is not None:
        for name in (
            "v1-checkpoint-audit.csv",
            "v1-gradient-diagnostics.csv",
            "v1-action-triplets-replayed.csv",
            "v1-raw-evidence-audit.json",
        ):
            path = v1_audit_dir / name
            if path.is_file():
                sources.append((path, f"visual-jepa-action-fidelity-v2-{name}"))
    records = [_copy_record(source, EVIDENCE_DIR / name) for source, name in sources]
    lock_path = runner.LOCK_PATH
    records.append(
        {
            "path": str(lock_path.relative_to(ROOT)),
            "size_bytes": lock_path.stat().st_size,
            "sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        }
    )
    manifest = {
        "schema_version": 1,
        "protocol_version": "visual-jepa-action-fidelity-v2",
        "retained_from_run_dir": "external research drive; exact path omitted",
        "files": records,
    }
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if MANIFEST_PATH.exists():
        if MANIFEST_PATH.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to replace a different retained evidence manifest: {MANIFEST_PATH}")
    else:
        MANIFEST_PATH.write_text(serialized, encoding="utf-8")
    validate_file_manifest(MANIFEST_PATH, ROOT)
    return {"manifest": str(MANIFEST_PATH.relative_to(ROOT)), "retained_file_count": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("/mnt/e/codex-research/jepa-world-models-tutorial/visual-jepa-action-fidelity-v2/validation"))
    parser.add_argument("--v1-audit-dir", type=Path, default=Path("/mnt/e/codex-research/jepa-world-models-tutorial/visual-jepa-action-fidelity-v2/v1-audit"))
    parser.add_argument("--retain", action="store_true", help="copy public-safe records into docs/evidence and hash them")
    parser.add_argument("--check-integrity", action="store_true", help="verify the retained evidence manifest")
    args = parser.parse_args()
    if args.check_integrity:
        validate_file_manifest(MANIFEST_PATH, ROOT)
        result = {"integrity": "pass", "manifest": str(MANIFEST_PATH.relative_to(ROOT))}
    else:
        result = validate_run(args.run_dir)
        if args.retain:
            result["retention"] = retain_public_evidence(args.run_dir, args.v1_audit_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
