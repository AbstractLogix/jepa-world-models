"""Regenerate evidence-driven figures for the visual JEPA action-fidelity study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "visual-jepa-action-fidelity-v2"
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
DEFAULT_OUTPUT = ROOT / "docs/figures/research/jepa-action-fidelity-v2"
V1_ACTIONS = EVIDENCE / "visual-jepa-action-fidelity-v2-v1-action-triplets-replayed.csv"
V2_ACTIONS = EVIDENCE / "visual-jepa-action-fidelity-v2-validation-action-triplets.csv"
V1_AUDIT = EVIDENCE / "visual-jepa-action-fidelity-v2-v1-checkpoint-audit.csv"
V2_MODELS = EVIDENCE / "visual-jepa-action-fidelity-v2-validation-model-runs.csv"
V2_SUMMARY = EVIDENCE / "visual-jepa-action-fidelity-v2-validation-summary.json"
HELDOUT_ROLLOUTS = EVIDENCE / "visual-jepa-action-fidelity-v2-heldout-rollouts.csv"
HELDOUT_SUMMARY = EVIDENCE / "visual-jepa-action-fidelity-v2-heldout-summary.json"
PLANNING_EPISODES = EVIDENCE / "visual-jepa-action-fidelity-v2-planning-episodes.csv"
PLANNING_SUMMARY = EVIDENCE / "visual-jepa-action-fidelity-v2-planning-summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def _number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return math.nan
    return float(value)


def _is_complete(row: Mapping[str, object]) -> bool:
    return str(row.get("status", "complete")) == "complete"


def _label(row: Mapping[str, object]) -> str:
    method = str(row.get("method", "unknown"))
    objective = str(row.get("objective", ""))
    if method == "jepa":
        return f"JEPA / {objective.replace('_', ' ')}"
    return method.replace("_", " ")


def _write_figure(fig: plt.Figure, output: Path, stem: str, title: str, alt: str, caption: str) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    svg_path = output / f"{stem}.svg"
    png_path = output / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight", metadata={"Title": title, "Description": alt, "Date": None})
    fig.savefig(png_path, format="png", dpi=260, bbox_inches="tight", metadata={"Title": title, "Description": alt})
    plt.close(fig)
    svg = svg_path.read_text(encoding="utf-8")
    svg = svg.replace("<svg ", '<svg role="img" aria-labelledby="svg-title svg-desc" ', 1)
    root_end = svg.find(">", svg.find("<svg "))
    if root_end >= 0:
        title_xml = html.escape(title)
        alt_xml = html.escape(alt)
        svg = svg[: root_end + 1] + f'<title id="svg-title">{title_xml}</title><desc id="svg-desc">{alt_xml}</desc>' + svg[root_end + 1 :]
    svg_path.write_text(svg, encoding="utf-8")
    return {
        "stem": stem,
        "title": title,
        "alt_text": alt,
        "caption": caption,
        "files": [
            {"path": svg_path.name, "sha256": _sha(svg_path), "size_bytes": svg_path.stat().st_size},
            {"path": png_path.name, "sha256": _sha(png_path), "size_bytes": png_path.stat().st_size},
        ],
    }


def _action_deltas(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, str, str], dict[float, dict[str, float]]]:
    grouped: dict[tuple[str, str, str, str], dict[float, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if not _is_complete(row):
            continue
        key = (
            _label(row),
            str(row.get("model_seed", "")),
            str(row.get("data_seed", "")),
            str(row["start_id"]),
        )
        grouped[key][float(row["action"])] = row
    result = {}
    for key, actions in grouped.items():
        if set(actions) != {-1.0, 0.0, 1.0}:
            continue
        baseline = actions[0.0]
        result[key] = {
            action: {
                "actual_position": _number(actions[action], "actual_position") - _number(baseline, "actual_position"),
                "predicted_position": _number(actions[action], "predicted_position_raw"),
                "position_baseline": _number(baseline, "predicted_position_raw"),
                "actual_velocity": _number(actions[action], "actual_velocity") - _number(baseline, "actual_velocity"),
                "predicted_velocity": _number(actions[action], "predicted_velocity_raw"),
                "velocity_baseline": _number(baseline, "predicted_velocity_raw"),
            }
            for action in (-1.0, 0.0, 1.0)
        }
    return result


def _draw_action_figure(rows: Sequence[Mapping[str, object]], output: Path, stem: str, title: str, source_label: str) -> dict[str, object]:
    deltas = _action_deltas(rows)
    if not deltas:
        raise ValueError(f"no complete action-triplet data for {title}")
    labels = sorted({key[0] for key in deltas})
    contexts = len({(key[0], key[1], key[2], key[3]) for key in deltas})
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, len(labels), figsize=(max(16, 3.1 * len(labels)), 8), sharex=True)
    axes = np.asarray(axes).reshape(2, len(labels))
    for column, label in enumerate(labels):
        seed_keys = sorted({key[1] for key in deltas if key[0] == label})
        for row_index, coordinate in enumerate(("position", "velocity")):
            axis = axes[row_index, column]
            for seed_index, seed in enumerate(seed_keys):
                seed_rows = [value for key, value in deltas.items() if key[0] == label and key[1] == seed]
                if not seed_rows:
                    continue
                actual_values = []
                predicted_values = []
                for action in (-1.0, 0.0, 1.0):
                    actual_values.append(float(np.mean([row[action][f"actual_{coordinate}"] for row in seed_rows])))
                    predicted_key = f"predicted_{coordinate}"
                    baseline_key = f"{coordinate}_baseline"
                    predicted_values.append(float(np.mean([row[action][predicted_key] - row[action][baseline_key] for row in seed_rows])))
                color = colors(seed_index % 10)
                axis.plot((-1, 0, 1), actual_values, linestyle="--", color="#222222", alpha=.38, linewidth=1.0)
                axis.plot((-1, 0, 1), predicted_values, marker="o", color=color, alpha=.88, linewidth=1.5, label=f"seed {seed}")
            axis.axhline(0.0, color="#888888", linewidth=.6)
            axis.set_xticks((-1, 0, 1), ("−1", "0", "+1"))
            axis.grid(True, alpha=.18)
            if row_index == 0:
                axis.set_title(label, fontsize=9, wrap=True)
            if column == 0:
                axis.set_ylabel("Position change" if coordinate == "position" else "Velocity change")
            if row_index == 1:
                axis.set_xlabel("Action")
            if column == len(labels) - 1 and row_index == 0:
                axis.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"{title}\nDashed: simulator; colored: raw model prediction. {source_label}. N={contexts} seed/context groups.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    alt = (
        f"Action-effect fidelity for {source_label}. Two rows compare position and velocity changes from action zero. "
        "Each panel shows actual simulator changes as dashed lines and model predictions as colored lines for actions minus one, zero, and plus one. "
        f"The plot uses {contexts} model-seed and context groups, with every seed shown separately."
    )
    caption = (
        f"{source_label}. Each point is a mean over matched fixed contexts for one model seed. Changes are measured relative to action zero. "
        "Predictions use pre-clip values; simulator trajectories follow MountainCarContinuous-v0. This synthetic environment does not test transfer to physical systems."
    )
    return _write_figure(fig, output, stem, title, alt, caption) | {"sample_groups": contexts}


def _draw_representation_figure(
    rows: Sequence[Mapping[str, object]], output: Path, stem: str, title: str, historical: bool
) -> dict[str, object]:
    all_rows = list(rows)
    if not historical:
        all_rows = [row for row in all_rows if row.get("method") != "state_model"]
    failed_rows = [row for row in all_rows if row.get("status", "complete") != "complete"]
    rows = [row for row in all_rows if row.get("status", "complete") == "complete"]
    if not all_rows:
        raise ValueError(f"no records for {title}")
    if historical:
        rows = [
            {
                **row,
                "plot_context_mean_std": _number(row, "context_validation_mean_std"),
                "plot_target_mean_std": _number(row, "target_validation_mean_std"),
                "plot_low_variance_dims": max(
                    _number(row, "context_validation_low_variance_dims"),
                    _number(row, "target_validation_low_variance_dims"),
                ),
                "plot_effective_rank": min(
                    _number(row, "context_validation_effective_rank"),
                    _number(row, "target_validation_effective_rank"),
                ),
                "plot_position_r2": _number(row, "reproduced_test_position_r2"),
                "plot_velocity_r2": _number(row, "reproduced_test_velocity_r2"),
            }
            for row in rows
        ]
    else:
        rows = [
            {
                **row,
                "plot_context_mean_std": _number(row, "mean_std"),
                "plot_target_mean_std": _number(row, "target_mean_std"),
                "plot_low_variance_dims": max(
                    _number(row, "low_variance_dims"), _number(row, "target_low_variance_dims")
                ),
                "plot_effective_rank": min(
                    _number(row, "effective_rank"), _number(row, "target_effective_rank")
                ),
                "plot_position_r2": _number(row, "position_r2"),
                "plot_velocity_r2": _number(row, "velocity_r2"),
            }
            for row in rows
        ]
    measures = (
        ("plot_context_mean_std", "Context embedding mean SD"),
        ("plot_target_mean_std", "Target embedding mean SD"),
        ("plot_low_variance_dims", "Worst low-spread dimension count"),
        ("plot_effective_rank", "Worst effective rank"),
        ("plot_position_r2", "Position probe R²"),
        ("plot_velocity_r2", "Velocity probe R²"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    labels = [_label(row) for row in all_rows]
    categories = list(dict.fromkeys(labels))
    colors = plt.get_cmap("tab10")
    for axis, (metric, label) in zip(axes.flat, measures, strict=False):
        for category_index, category in enumerate(categories):
            selected = [row for row in rows if _label(row) == category]
            x = np.full(len(selected), category_index, dtype=float)
            values = [_number(row, metric) for row in selected]
            seed_offsets = np.linspace(-.16, .16, len(selected)) if len(selected) > 1 else np.array([0.0])
            axis.scatter(x + seed_offsets, values, color=colors(category_index % 10), s=38, alpha=.9, zorder=3)
            for offset, row, value in zip(seed_offsets, selected, values, strict=True):
                axis.annotate(str(row.get("model_seed", "")), (category_index + offset, value), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set_title(label)
        axis.set_xticks(range(len(categories)), categories, rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=.2)
        for category_index, category in enumerate(categories):
            failures = [
                str(row.get("model_seed", ""))
                for row in failed_rows if _label(row) == category
            ]
            if failures:
                axis.text(
                    category_index,
                    .02,
                    "failed: " + ", ".join(failures),
                    transform=axis.get_xaxis_transform(),
                    color="#b22222",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )
    fig.suptitle(title + f"\nEach point is one completed seed; {len(failed_rows)} failed fit rows are marked. N={len(all_rows)} records.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .93))
    alt = (
        "Six panels show context and target embedding mean standard deviation, the worse low-spread dimension count, the worse effective rank, and position and velocity probe R-squared. "
        "Every completed model seed appears as a labeled point, and failed fits are listed below their method. "
        + ("These are historical v1 checkpoint replays and v1 test records." if historical else "These are v2 validation-only records; no held-out data are shown.")
    )
    caption = (
        "Representation health is reported per seed. The low-spread count and effective rank show the worse of the context and target embeddings; low spread uses the per-dimension 1e-3 rule. Effective rank is computed from the validation covariance spectrum. "
        + ("Historical v1 values are derived from retained v1 checkpoints and the original v1 test split." if historical else "V2 uses the frozen train/validation split and the 20-refit episode-block probe diagnostic.")
    )
    return _write_figure(fig, output, stem, title, alt, caption) | {"sample_size": len(rows), "failed_records": len(failed_rows)}


def _cluster_bootstrap_interval(rows: Sequence[Mapping[str, object]], metric: str, seed: int) -> tuple[float, float, float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "complete" and row.get(metric) not in (None, ""):
            clusters[str(row.get("test_seed", ""))].append(float(row[metric]))
    groups = [values for _key, values in sorted(clusters.items()) if values]
    if not groups:
        return math.nan, math.nan, math.nan
    center = float(np.mean([value for group in groups for value in group]))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(2000):
        chosen = rng.integers(0, len(groups), size=len(groups))
        selected = [value for group_index in chosen for value in groups[int(group_index)]]
        samples.append(float(np.mean(selected)))
    return center, float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def _draw_rollout_figure(rows: Sequence[Mapping[str, object]], output: Path) -> dict[str, object]:
    if not rows:
        raise ValueError("held-out rollout rows are missing")
    horizons = (1, 5, 10, 20, 60)
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        raise ValueError("held-out rollout stage has no supported complete trajectories")
    methods = sorted({str(row["method"]) for row in complete})
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for metric_index, (metric, label) in enumerate((("position_abs_error", "Position MAE"), ("velocity_abs_error", "Velocity MAE"))):
        axis = axes[metric_index]
        for method_index, method in enumerate(methods):
            method_rows = [row for row in complete if row["method"] == method]
            model_seeds = sorted({str(row["model_seed"]) for row in method_rows})
            for seed_index, model_seed in enumerate(model_seeds):
                selected = [row for row in method_rows if str(row["model_seed"]) == model_seed]
                color = colors(method_index % 10)
                # Thin lines show every available start trajectory.
                starts = sorted({str(row["start_id"]) for row in selected})
                for start_id in starts:
                    trajectory = [row for row in selected if str(row["start_id"]) == start_id]
                    trajectory.sort(key=lambda row: int(row["horizon"]))
                    axis.plot([int(row["horizon"]) for row in trajectory], [float(row[metric]) for row in trajectory], color=color, alpha=.10, linewidth=.7)
                means, low, high = [], [], []
                for horizon in horizons:
                    at_horizon = [row for row in selected if int(row["horizon"]) == horizon]
                    center, lower, upper = _cluster_bootstrap_interval(
                        at_horizon, metric,
                        seed=_stable_seed("heldout-figure", method, model_seed, horizon, metric),
                    )
                    means.append(center)
                    low.append(lower)
                    high.append(upper)
                label_text = f"{method}, model seed {model_seed}"
                axis.plot(horizons, means, marker="o", linewidth=2.0, color=color, linestyle=("-", "--", ":")[seed_index % 3], label=label_text)
                axis.fill_between(horizons, low, high, color=color, alpha=.08)
        axis.set_xscale("log")
        axis.set_xticks(horizons, [str(value) for value in horizons])
        axis.set_xlabel("Rollout horizon (simulator steps)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=.2)
    axes[1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Held-out multi-step rollout error\nThin lines: individual starts; ribbons: 95% test-seed-cluster bootstrap interval. N={len(complete)} supported horizon rows.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    alt = (
        "Two panels show held-out position and velocity absolute errors at horizons one, five, ten, twenty, and sixty steps. "
        "Thin lines show each supported evaluation start. Thick lines show model-seed means. Shaded ribbons use a 2,000-replicate bootstrap over the five test-seed clusters."
    )
    caption = (
        "All models use the same deterministic continuous action sequence for each start. Unsupported post-termination horizons are omitted, not imputed. "
        "The uncertainty interval resamples test seeds as clusters so the four starts within a test seed stay together."
    )
    return _write_figure(fig, output, "figure-c-heldout-rollout-stability", "Held-out multi-step rollout stability", alt, caption) | {
        "sample_rows": len(complete), "supported_trajectory_count": len({(row["method"], row["model_seed"], row["start_id"]) for row in complete})
    }


def _draw_objective_figure(rows: Sequence[Mapping[str, object]], output: Path) -> dict[str, object]:
    candidate_rows = [row for row in rows if row.get("method") == "jepa"]
    control_rows = [row for row in rows if row.get("method") == "supervised_visual"]
    if not candidate_rows:
        raise ValueError("validation objective records are missing")
    summary = read_json(V2_SUMMARY)
    selected_objective = summary.get("selected_objective")
    objectives = ("sigreg", "visreg_style", "variance_cov")
    measures = (
        ("effective_rank", "Effective rank"),
        ("velocity_r2", "Velocity probe R²"),
        ("position_median_normalized_contrast_error", "Position action-contrast error"),
        ("velocity_median_normalized_contrast_error", "Velocity action-contrast error"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for axis, (metric, label) in zip(axes.flat, measures, strict=True):
        for objective_index, objective in enumerate(objectives):
            objective_values = [
                _number(row, metric)
                for row in candidate_rows if row.get("objective") == objective and _number(row, metric) == _number(row, metric)
            ]
            x = objective_index + np.linspace(-.15, .15, len(objective_values)) if objective_values else []
            axis.scatter(x, objective_values, color=plt.get_cmap("tab10")(objective_index), label=objective.replace("_", " "), s=42)
            if objective == selected_objective:
                axis.axvline(objective_index, color="#111111", alpha=.25, linewidth=1.5)
        if control_rows:
            control_values = [_number(row, metric) for row in control_rows if _number(row, metric) == _number(row, metric)]
            if control_values:
                control_x = np.full(len(control_values), len(objectives), dtype=float) + np.linspace(-.15, .15, len(control_values))
                axis.scatter(control_x, control_values, marker="D", color="#555555", s=42, label="supervised visual")
        for objective_index, objective in enumerate(objectives):
            failed_seeds = [
                str(row.get("model_seed", ""))
                for row in candidate_rows
                if row.get("objective") == objective and row.get("status") != "complete"
            ]
            if failed_seeds:
                axis.text(
                    objective_index,
                    .02,
                    "failed: " + ",".join(failed_seeds),
                    transform=axis.get_xaxis_transform(),
                    color="#b22222",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                )
        labels = [*objectives, "supervised visual"]
        axis.set_xticks(range(len(labels)), [value.replace("_", " ") for value in labels], rotation=25, ha="right")
        axis.set_title(label)
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    failed = [row for row in candidate_rows if row.get("status") != "complete"]
    fig.suptitle(
        f"V2 validation objective comparison\nN={len(candidate_rows)} JEPA rows ({len(failed)} failed); {len(control_rows)} supervised visual rows. Each point is a seed.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, .93))
    alt = (
        "Four panels compare effective rank, velocity-probe R-squared, position action-contrast error, and velocity action-contrast error for each v2 objective and the supervised visual control. "
        "Every completed seed is shown. The selected objective is marked only when validation gates select one."
    )
    caption = (
        "Validation-only data are used. Error metrics are lower-is-better; rank and probe R-squared are higher-is-better. "
        "A negative or failed seed remains visible. No held-out test data are used for model selection."
    )
    return _write_figure(fig, output, "figure-d-objective-comparison", "V2 validation objective comparison", alt, caption) | {
        "candidate_seed_rows": len(candidate_rows), "supervised_visual_seed_rows": len(control_rows), "selected_objective": selected_objective
    }


def _draw_planning_figure(rows: Sequence[Mapping[str, object]], output: Path) -> dict[str, object]:
    if not rows:
        raise ValueError("planning episode rows are missing")
    methods = sorted({str(row["method"]) for row in rows})
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for index, method in enumerate(methods):
        arms = [row for row in rows if row["method"] == method]
        if not arms:
            continue
        labels = sorted({f"{row['initialization']}\nseed {row.get('model_seed', '')}" for row in arms})
        x = np.arange(len(labels))
        rates: list[float] = []
        rate_errors: list[tuple[float, float]] = []
        returns: list[float] = []
        return_errors: list[tuple[float, float]] = []
        for label in labels:
            selected = [row for row in arms if f"{row['initialization']}\nseed {row.get('model_seed', '')}" == label]
            successes = sum(int(row["success"]) for row in selected)
            count = len(selected)
            rate = successes / max(1, count)
            z = 1.959963984540054
            denominator = 1 + z * z / max(1, count)
            center = (rate + z * z / (2 * max(1, count))) / denominator
            radius = z * math.sqrt(rate * (1 - rate) / max(1, count) + z * z / (4 * max(1, count) ** 2)) / denominator
            rates.append(rate)
            rate_errors.append((rate - max(0.0, center - radius), min(1.0, center + radius) - rate))
            values = np.asarray([float(row["return"]) for row in selected], dtype=np.float64)
            returns.append(float(values.mean()))
            rng = np.random.default_rng(_stable_seed("planning-figure-return", method, label))
            bootstrap = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
            return_errors.append((
                float(values.mean() - np.quantile(bootstrap, .025)),
                float(np.quantile(bootstrap, .975) - values.mean()),
            ))
        display_labels = [f"{method}\n{label}" for label in labels]
        x = np.arange(len(display_labels))
        axes[0].errorbar(x, rates, yerr=np.asarray(rate_errors).T, fmt="o", color=colors(index % 10), label=method, capsize=2)
        axes[1].errorbar(x, returns, yerr=np.asarray(return_errors).T, fmt="o", color=colors(index % 10), label=method, capsize=2)
        axes[0].set_xticks(x, display_labels)
        axes[1].set_xticks(x, display_labels)
    axes[0].set_ylabel("Held-out success fraction")
    axes[0].set_ylim(-.05, 1.05)
    axes[1].set_ylabel("Mean return")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=70, labelsize=7)
        axis.grid(axis="y", alpha=.2)
    compute_groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        compute_groups[(str(row["method"]), str(row.get("model_seed", "")), str(row["initialization"]))].append(row)
    for index, ((method, model_seed, initialization), group) in enumerate(sorted(compute_groups.items())):
        calls = float(np.mean([float(row["transition_calls"]) for row in group]))
        seconds = float(np.mean([float(row["planning_seconds"]) for row in group]))
        axes[2].scatter(calls, seconds, color=colors(index % 10), s=42)
        axes[2].annotate(f"{method}\n{initialization}, seed {model_seed}", (calls, seconds), xytext=(3, 3), textcoords="offset points", fontsize=6)
    axes[2].set_xlabel("Mean model transition calls per start")
    axes[2].set_ylabel("Mean planning time (seconds)")
    axes[2].set_title("Compute accounting")
    axes[2].grid(True, alpha=.2)
    fig.suptitle(f"Conditional planning results\nN={len(rows)} episode rows. Point groups retain model seed, initialization, and 20 start seeds; hardware paths differ.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .9))
    alt = (
        "Three panels show success fraction with Wilson 95 percent intervals, mean return with start-bootstrap 95 percent intervals, and transition calls versus planning time for each planner arm, model seed, and initialization. "
        "Raw per-start rows also report steps, elapsed time, and device."
    )
    caption = (
        "Planning is shown only if held-out validation gates opened this stage. Each point covers the fixed twenty planning starts for one model seed and initialization. "
        "Compute costs are reported per execution path and are not claimed to be equal across CPU and GPU arms."
    )
    return _write_figure(fig, output, "figure-e-decision-usefulness", "Conditional planning decision usefulness", alt, caption) | {"episode_rows": len(rows)}


def _draw_architecture(output: Path) -> dict[str, object]:
    fig, axis = plt.subplots(figsize=(15, 4.2))
    axis.set_xlim(0, 15)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = [
        (0.3, "Rendered\nframe pair", "#e7f0fa"),
        (2.5, "Visual\nencoder", "#d9e8f5"),
        (4.7, "Action-conditioned\npredictor", "#c8def0"),
        (7.0, "Predicted\nlatent", "#e5e1f5"),
        (9.1, "Frozen state\nreadout", "#f1e2bc"),
        (11.2, "Planner", "#f6d8cc"),
        (13.1, "Environment\nfeedback", "#dcebd9"),
    ]
    for x, label, color in boxes:
        axis.add_patch(plt.Rectangle((x, 1.35), 1.5, 1.25, facecolor=color, edgecolor="#334155", linewidth=1.2, zorder=2))
        axis.text(x + .75, 1.97, label, ha="center", va="center", fontsize=10)
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        axis.annotate("", xy=(right[0] - .08, 1.98), xytext=(left[0] + 1.58, 1.98), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"})
    axis.text(3.25, 3.05, "Two RGB frames", ha="center", fontsize=9, color="#475569")
    axis.text(5.45, 3.05, "Action aₜ", ha="center", fontsize=9, color="#475569")
    axis.annotate("", xy=(4.85, 2.58), xytext=(5.35, 2.85), arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#b45309"})
    axis.annotate("environment step", xy=(.95, 1.23), xytext=(13.8, .65), ha="center", fontsize=9, color="#475569", arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-.28", "lw": 1.2, "color": "#64748b"})
    axis.text(7.5, .1, "The readout is trained on training embeddings and then held fixed during prediction and planning.", ha="center", fontsize=9, color="#475569")
    fig.suptitle("Action-conditioned visual JEPA evaluation path", fontsize=14)
    fig.tight_layout()
    alt = (
        "A left-to-right diagram shows rendered frame pairs entering a visual encoder, then an action-conditioned predictor, a predicted latent, a frozen state readout, a planner, and environment feedback. "
        "The environment sends new rendered frames back to the encoder. The action enters the predictor."
    )
    caption = "Architecture diagram for the local synthetic MountainCar study. The state readout is a diagnostic linear probe, not a learned pixel decoder."
    return _write_figure(fig, output, "figure-f-action-conditioned-architecture", "Action-conditioned visual JEPA architecture", alt, caption)


def generate_figures(output: Path) -> dict[str, object]:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, object]] = []
    v1_action_rows = read_csv(V1_ACTIONS)
    v2_action_rows = read_csv(V2_ACTIONS)
    v1_model_rows = read_csv(V1_AUDIT)
    v2_model_rows = read_csv(V2_MODELS)
    if v1_action_rows:
        figures.append(_draw_action_figure(v1_action_rows, output, "figure-a1-historical-v1-action-fidelity", "Historical v1 action-effect fidelity", "Retained Notebook 23 checkpoint replay; pre-clamp probe outputs."))
    if v2_action_rows:
        failed = [row for row in v2_model_rows if row.get("status") != "complete"]
        failed_labels = ", ".join(f"{row.get('method')}/{row.get('objective')} seed {row.get('model_seed')}" for row in failed)
        failed_note = f"Failed fits retained in records: {failed_labels}." if failed else "All recorded fits completed."
        figures.append(_draw_action_figure(v2_action_rows, output, "figure-a2-v2-validation-action-fidelity", "V2 validation action-effect fidelity", f"New validation-only v2 records; {failed_note} Held-out evaluation remains separate."))
    if v1_model_rows:
        figures.append(_draw_representation_figure(v1_model_rows, output, "figure-b1-historical-v1-representation-health", "Historical v1 representation health", historical=True))
    if v2_model_rows:
        figures.append(_draw_representation_figure(v2_model_rows, output, "figure-b2-v2-validation-representation-health", "V2 validation representation health", historical=False))
    if v2_model_rows:
        figures.append(_draw_objective_figure(v2_model_rows, output))
    heldout = read_json(HELDOUT_SUMMARY)
    heldout_rows = read_csv(HELDOUT_ROLLOUTS)
    if heldout.get("status") == "complete" and heldout.get("test_seed_groups_opened") and heldout_rows:
        figures.append(_draw_rollout_figure(heldout_rows, output))
    planning = read_json(PLANNING_SUMMARY)
    planning_rows = read_csv(PLANNING_EPISODES)
    if planning.get("status") in {"complete", "budget_stopped"} and planning_rows:
        figures.append(_draw_planning_figure(planning_rows, output))
    figures.append(_draw_architecture(output))
    sources = [path for path in (V1_ACTIONS, V2_ACTIONS, V1_AUDIT, V2_MODELS, V2_SUMMARY, HELDOUT_ROLLOUTS, HELDOUT_SUMMARY, PLANNING_EPISODES, PLANNING_SUMMARY) if path.is_file()]
    metadata = {
        "protocol_version": "visual-jepa-action-fidelity-v2",
        "figure_count": len(figures),
        "figure_generator": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha(Path(__file__).resolve()),
        },
        "measurement_sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)} for path in sources],
        "figures": figures,
        "not_generated": {
            "figure_c": "held-out stage did not complete or contain supported rollouts" if not any(item["stem"].startswith("figure-c") for item in figures) else "",
            "figure_e": "planning gate stayed closed or no planning rows were recorded" if not any(item["stem"].startswith("figure-e") for item in figures) else "",
        },
    }
    (output / "figure-manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    captions = ["# Action-fidelity study figures", "", "All measurement figures use retained CSV records. SVG files include title and description text; PNG files are rendered at 260 dpi.", ""]
    for figure in figures:
        captions.extend((f"## {figure['title']}", "", f"**Alt text:** {figure['alt_text']}", "", str(figure["caption"]), ""))
    for key, reason in metadata["not_generated"].items():
        if reason:
            captions.extend((f"## {key.upper()} not generated", "", str(reason), ""))
    captions.extend(
        (
            "## Data and regeneration",
            "",
            f"The figure generator is `{metadata['figure_generator']['path']}` with SHA-256 `{metadata['figure_generator']['sha256']}`.",
            "Measurement inputs and their SHA-256 values are listed in `figure-manifest.json`.",
            "",
        )
    )
    (output / "README.md").write_text("\n".join(captions), encoding="utf-8")
    return metadata


def verify_regeneration(output_a: Path, output_b: Path) -> dict[str, object]:
    first = generate_figures(output_a)
    second = generate_figures(output_b)
    hashes_a = {
        entry["path"]: entry["sha256"]
        for figure in first["figures"] for entry in figure["files"]
    }
    hashes_b = {
        entry["path"]: entry["sha256"]
        for figure in second["figures"] for entry in figure["files"]
    }
    if hashes_a != hashes_b:
        raise ValueError("figure regeneration is not byte-deterministic")
    return {"status": "pass", "figure_count": first["figure_count"], "figure_hashes": hashes_a}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args()
    if args.verify_determinism:
        base = args.output_dir.expanduser().resolve()
        result = verify_regeneration(base / "verify-a", base / "verify-b")
    else:
        result = generate_figures(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
