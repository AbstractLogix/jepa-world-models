#!/usr/bin/env python3
"""Create a claim-aware summary from retained v1 decision-study evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "visual-jepa-decision-study-v1"
DEFAULT_EVIDENCE = ROOT / "docs/evidence"
DEFAULT_OUTPUT = DEFAULT_EVIDENCE / f"{PREFIX}-analysis.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_seed_start_bootstrap_interval(
    seed_deltas: Mapping[str, Mapping[str, float]],
    *,
    replicates: int = 10_000,
    random_seed: int = 97_531,
) -> list[float]:
    """Resample model/data seed pairs and shared evaluation starts hierarchically."""
    if len(seed_deltas) < 2 or replicates < 1:
        raise ValueError("at least two seed pairs and one bootstrap replicate are required")
    start_sets = {tuple(sorted(values)) for values in seed_deltas.values()}
    if len(start_sets) != 1 or not next(iter(start_sets)):
        raise ValueError("every seed pair must contain the same non-empty evaluation starts")
    if any(not np.isfinite(value) for values in seed_deltas.values() for value in values.values()):
        raise ValueError("seed and start deltas must be finite")

    seeds = sorted(seed_deltas)
    starts = list(next(iter(start_sets)))
    rng = np.random.default_rng(random_seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        # These start indices are shared by all selected seed pairs in this replicate.
        selected_starts = rng.choice(starts, size=len(starts), replace=True)
        seed_means = [
            float(np.mean([seed_deltas[str(seed)][str(start)] for start in selected_starts]))
            for seed in selected_seeds
        ]
        samples[replicate] = float(np.mean(seed_means))
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return [float(lower), float(upper)]


def primary_comparison(
    episode_rows: Sequence[Mapping[str, str]],
    *,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Summarize paired prior-guided JEPA minus direct-control return."""
    direct: dict[str, float] = {}
    jepa: dict[str, dict[str, float]] = defaultdict(dict)
    for row in episode_rows:
        if row.get("status") != "complete":
            continue
        method = row["method"]
        start = row["eval_seed"]
        if method == "direct_visual_momentum":
            if start in direct:
                raise ValueError(f"duplicate direct episode for evaluation seed {start}")
            direct[start] = float(row["return"])
        elif method == "jepa_cem" and row.get("initialization") == "momentum_prior":
            model_seed = row["model_seed"]
            if start in jepa[model_seed]:
                raise ValueError(f"duplicate JEPA episode for seed {model_seed}, start {start}")
            jepa[model_seed][start] = float(row["return"])

    if not direct or not jepa:
        raise ValueError("primary comparison needs direct and prior-guided JEPA episodes")
    if any(set(values) != set(direct) for values in jepa.values()):
        raise ValueError("JEPA episodes must pair with the same complete direct-control starts")
    deltas = {
        model_seed: {start: value - direct[start] for start, value in values.items()}
        for model_seed, values in jepa.items()
    }
    seed_means = {seed: float(np.mean(list(values.values()))) for seed, values in deltas.items()}
    return {
        "status": "descriptive",
        "contrast": "jepa_cem_momentum_prior_minus_direct_visual_momentum",
        "training_data_model_seed_pairs": len(deltas),
        "shared_evaluation_starts": len(direct),
        "per_seed_pair_mean_return_delta": seed_means,
        "mean_across_seed_pair_means": float(np.mean(list(seed_means.values()))),
        "hierarchical_bootstrap_95_percent_interval": paired_seed_start_bootstrap_interval(
            deltas, replicates=replicates
        ),
        "bootstrap": {
            "method": "resample paired training-data/model seeds, then resample shared evaluation starts once per replicate",
            "replicates": replicates,
            "random_seed": 97_531,
        },
        "claim_limit": "descriptive estimate for one local synthetic task; no significance claim",
    }


def method_summaries(episode_rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in episode_rows:
        if row.get("status") == "complete":
            groups[(row["method"], row.get("initialization", ""))].append(row)
    result: dict[str, Any] = {}
    for (method, initialization), rows in sorted(groups.items()):
        key = f"{method}|{initialization}"
        by_seed: dict[str, list[Mapping[str, str]]] = defaultdict(list)
        for row in rows:
            by_seed[row.get("model_seed") or "fixed"].append(row)
        result[key] = {
            "n_episode_rows": len(rows),
            "successes": sum(int(row["success"]) for row in rows),
            "success_rate": float(np.mean([int(row["success"]) for row in rows])),
            "mean_return": float(np.mean([float(row["return"]) for row in rows])),
            "mean_steps": float(np.mean([int(row["steps"]) for row in rows])),
            "mean_transition_calls": float(np.mean([int(row["transition_calls"]) for row in rows])),
            "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in rows])),
            "by_seed_pair": {
                seed: {
                    "n_episode_rows": len(seed_rows),
                    "successes": sum(int(row["success"]) for row in seed_rows),
                    "success_rate": float(np.mean([int(row["success"]) for row in seed_rows])),
                    "mean_return": float(np.mean([float(row["return"]) for row in seed_rows])),
                    "mean_transition_calls": float(np.mean([int(row["transition_calls"]) for row in seed_rows])),
                    "mean_elapsed_seconds": float(np.mean([float(row["elapsed_seconds"]) for row in seed_rows])),
                }
                for seed, seed_rows in sorted(by_seed.items())
            },
        }
    return result


def action_effect_diagnostics(
    intervention_rows: Sequence[Mapping[str, str]], family: str = "jepa"
) -> list[dict[str, Any]]:
    """Compare predicted and observed next-state ranges across the three actions."""
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in intervention_rows:
        if row.get("status") == "complete" and row.get("method") == family:
            groups[(row["model_seed"], row["start_id"])].append(row)
    by_seed: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "predicted_position_span": [],
            "observed_position_span": [],
            "predicted_velocity_span": [],
            "observed_velocity_span": [],
        }
    )
    for (seed, _start), rows in groups.items():
        if len(rows) != 3 or {float(row["action"]) for row in rows} != {-1.0, 0.0, 1.0}:
            raise ValueError(f"intervention group for {family} seed {seed} is incomplete")
        values = {
            name: [float(row[field]) for row in rows]
            for name, field in (
                ("predicted_position_span", "predicted_position"),
                ("observed_position_span", "actual_position"),
                ("predicted_velocity_span", "predicted_velocity"),
                ("observed_velocity_span", "actual_velocity"),
            )
        }
        for name, group_values in values.items():
            by_seed[seed][name].append(float(np.ptp(group_values)))
    return [
        {
            "model_seed": seed,
            "held_out_starts": len(metrics["predicted_position_span"]),
            **{
                key: value
                for name, values in metrics.items()
                for key, value in (
                    (name, float(np.mean(values))),
                    (f"{name}_sd", float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                )
            },
        }
        for seed, metrics in sorted(by_seed.items())
    ]


def analyze(evidence_dir: Path, *, replicates: int = 10_000) -> dict[str, Any]:
    episode_path = evidence_dir / f"{PREFIX}-episodes.csv"
    model_path = evidence_dir / f"{PREFIX}-model-runs.csv"
    probe_path = evidence_dir / f"{PREFIX}-probe-test.csv"
    intervention_path = evidence_dir / f"{PREFIX}-interventions.csv"
    episodes = read_csv(episode_path)
    models = read_csv(model_path)
    probes = read_csv(probe_path)
    interventions = read_csv(intervention_path)
    expected_episode_counts = {
        "oracle_cem": 40,
        "direct_visual_momentum": 20,
        "state_model_cem": 200,
        "image_reconstruction_cem": 200,
        "jepa_cem": 200,
    }
    observed_episode_counts: dict[str, int] = defaultdict(int)
    for row in episodes:
        if row.get("status") != "complete":
            raise ValueError("episode evidence contains missing or failed cases")
        observed_episode_counts[row["method"]] += 1
    if dict(observed_episode_counts) != expected_episode_counts:
        raise ValueError(f"expected the complete 660-case matrix, got {dict(observed_episode_counts)}")
    if len(models) != 15 or any(row.get("status") != "complete" for row in models):
        raise ValueError("analysis requires 15 complete model fits")
    if len(probes) != 10:
        raise ValueError("analysis requires ten held-out representation-probe records")
    if len(interventions) != 960 or any(row.get("status") != "complete" for row in interventions):
        raise ValueError("analysis requires 960 complete intervention records")
    by_seed = {row["model_seed"]: row for row in models if row["method"] == "jepa"}
    probe_by_seed = {row["model_seed"]: row for row in probes if row["family"] == "jepa"}
    if len(by_seed) != 5 or set(probe_by_seed) != set(by_seed):
        raise ValueError("analysis requires five matched JEPA model and test-probe seed pairs")
    representation = []
    for seed, model in sorted(by_seed.items()):
        probe = probe_by_seed.get(seed, {})
        representation.append(
            {
                "model_seed": seed,
                "data_seed": model["data_seed"],
                "embedding_mean_std": float(model["embedding_mean_std"]),
                "embedding_low_variance_dims": int(model["embedding_low_variance_dims"]),
                "embedding_dimension": int(model["embedding_dimension"]),
                "embedding_collapsed": model["embedding_collapsed"].lower() == "true",
                "position_probe_r2": float(probe["position_r2"]) if probe else None,
                "velocity_probe_r2": float(probe["velocity_r2"]) if probe else None,
            }
        )
    action_diagnostics = action_effect_diagnostics(interventions)
    if len(action_diagnostics) != 5 or any(
        row["held_out_starts"] != 20 for row in action_diagnostics
    ):
        raise ValueError("analysis requires five JEPA seed pairs on all 20 intervention starts")
    warnings = [row for row in representation if row["embedding_collapsed"]]
    primary = primary_comparison(episodes, replicates=replicates)
    if primary["training_data_model_seed_pairs"] != 5 or primary["shared_evaluation_starts"] != 20:
        raise ValueError("analysis requires five seed pairs on all 20 shared evaluation starts")
    return {
        "schema_version": 1,
        "experiment_version": PREFIX,
        "source_files": {
            path.name: sha256(path)
            for path in (episode_path, model_path, probe_path, intervention_path)
        },
        "episode_rows": len(episodes),
        "completed_episode_rows": sum(row.get("status") == "complete" for row in episodes),
        "method_summary": method_summaries(episodes),
        "primary_comparison": primary,
        "jepa_representation_diagnostics": representation,
        "jepa_action_effect_diagnostics": action_diagnostics,
        "claim_status": "partial" if warnings else "descriptive_local_comparison",
        "claim_status_reason": (
            "At least one JEPA seed triggered the predeclared embedding low-spread diagnostic. "
            "Keep its planner outcomes exploratory."
            if warnings
            else "No JEPA seed triggered the recorded low-spread diagnostic."
        ),
        "methodology_limits": [
            "Each model seed is paired to one training-data seed, so initialization and data variation are not separated.",
            "A prior-only CEM arm was not included; direct control and prior-initialized learned-model planning do not replace it.",
            "The direct controller has one outcome per shared evaluation start and must not be counted once per model seed.",
            "The fixed momentum continuation is an evaluator-side simulator diagnostic, not a deployable learned reward model.",
            "This study does not estimate calibrated model uncertainty.",
            "Timing is measured on the recorded devices; model-transition calls do not equal hardware compute.",
        ],
        "uncertainty": "The interval is descriptive. No statistical significance or generalization claim is made.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(args.evidence_dir, replicates=args.bootstrap_replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
