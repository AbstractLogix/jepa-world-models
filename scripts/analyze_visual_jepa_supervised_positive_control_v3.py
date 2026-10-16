"""Create deterministic figures from retained supervised visual v3 records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs/evidence/visual-jepa-supervised-positive-control-v3"
FIGURE_DIR = ROOT / "docs/portfolio/figures/visual-jepa-supervised-positive-control-v3"
METHODS = (
    "handcrafted_state_exact_dynamics",
    "handcrafted_feature_ridge",
    "supervised_visual_absolute_raw",
    "supervised_visual_absolute_standardized",
    "supervised_visual_residual",
    "supervised_visual_action_shuffle",
)
NEURAL_METHODS = (
    "visual_state_estimator",
    "supervised_visual_absolute_raw",
    "supervised_visual_absolute_standardized",
    "supervised_visual_residual",
    "supervised_visual_action_shuffle",
)
LABELS = {
    "privileged_exact_dynamics": "Privileged exact dynamics",
    "handcrafted_state_exact_dynamics": "Visual state + exact",
    "handcrafted_feature_ridge": "Handcrafted ridge",
    "visual_state_estimator": "Visual state estimator*",
    "supervised_visual_absolute_raw": "Raw-target visual",
    "supervised_visual_absolute_standardized": "Scaled-target visual",
    "supervised_visual_residual": "Hybrid residual",
    "supervised_visual_action_shuffle": "Shuffled-action control",
}
COORDS = ("position", "velocity")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(fig, output: Path, stem: str) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{stem}.svg", output / f"{stem}.png"]
    metadata = {"Creator": "JEPA World Models Tutorial v3", "Date": None}
    fig.savefig(paths[0], format="svg", metadata=metadata)
    fig.savefig(paths[1], format="png", dpi=220, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return [path.name for path in paths]


def _base_style() -> None:
    plt.rcParams.update(
        {
            "svg.hashsalt": "jepa-supervised-positive-control-v3",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
        }
    )


def _action_summary(rows: Sequence[Mapping[str, str]], coordinate: str):
    by_context: dict[tuple[str, str, str, str], dict[float, Mapping[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("stage") != "validation" or row.get("method") not in METHODS:
            continue
        key = (row["method"], row["model_seed"], row["data_seed"], row["start_id"])
        by_context[key][float(row["action"])] = row
    out: dict[tuple[str, str, str], dict[float, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"actual": [], "predicted": []})
    )
    for (method, model_seed, data_seed, _), values in by_context.items():
        if set(values) != {-1.0, 0.0, 1.0}:
            continue
        key = (method, model_seed, data_seed)
        actual_zero = float(values[0.0][f"actual_{coordinate}"])
        predicted_zero = float(values[0.0][f"predicted_{coordinate}_raw"])
        for action in (-1.0, 0.0, 1.0):
            out[key][action]["actual"].append(float(values[action][f"actual_{coordinate}"]) - actual_zero)
            out[key][action]["predicted"].append(float(values[action][f"predicted_{coordinate}_raw"]) - predicted_zero)
    return out


def _plot_action(evidence: Path, output: Path, coordinate: str) -> dict[str, object]:
    rows = _read(evidence / "action-triplets.csv")
    grouped = _action_summary(rows, coordinate)
    _base_style()
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"9701": "#176B87", "9702": "#C75B12", "9703": "#694A8E"}
    for ax, method in zip(axes.ravel(), METHODS, strict=True):
        ax.axhline(0, color="#444444", linewidth=0.6)
        for (group_method, model_seed, data_seed), values in sorted(grouped.items()):
            if group_method != method:
                continue
            x = (-1, 0, 1)
            color = colors[data_seed]
            predicted = [float(np.mean(values[action]["predicted"])) for action in x]
            actual = [float(np.mean(values[action]["actual"])) for action in x]
            ax.plot(x, predicted, marker="o", color=color, linewidth=1, alpha=0.65)
            ax.plot(x, actual, linestyle="--", color=color, linewidth=1, alpha=0.8)
        ax.set_title(LABELS[method], weight="bold")
        ax.set_xticks((-1, 0, 1), ("−1", "0", "+1"))
        ax.set_xlabel("Action")
        ax.set_ylabel(f"Mean Δ{coordinate} from action 0")
    fig.suptitle(
        f"Action-effect fidelity: {coordinate}\nSynthetic MountainCarContinuous-v0 · validation only · 20 contexts per seed pair",
        fontsize=13,
        weight="bold",
    )
    fig.legend(
        [Line2D([0], [0], color=colors[key], linewidth=2) for key in colors],
        [f"Data seed {key}" for key in colors],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.01,
        0.005,
        "Synthetic task only; these action effects do not measure planning or control success.",
        fontsize=7,
    )
    stem = f"figure-a-action-fidelity-{coordinate}"
    files = _save(fig, output, stem)
    return {
        "id": stem,
        "files": files,
        "caption": f"Predicted and simulator {coordinate} changes for actions −1, 0, and +1. Solid lines show predictions; dashed lines show the simulator. Color identifies each independent data/model pair.",
        "alt_text": f"Six panels compare {coordinate} action responses for the handcrafted exact control, handcrafted ridge, raw-target visual model, target-scaled visual model, hybrid residual model, and shuffled-action control.",
        "sample_size": "20 shared action contexts per pair; 3 independent pairs",
        "limitations": "Synthetic MountainCar only. Means show response scale and direction, not planning success.",
    }


def _plot_validation(evidence: Path, output: Path) -> dict[str, object]:
    runs = [r for r in _read(evidence / "model-runs.csv") if r.get("status") == "complete"]
    gates = _read(evidence / "validation-gates.csv")
    action = _read(evidence / "action-summary.csv")
    gate_by_key = {(r["method"], r["model_seed"], r["data_seed"]): r for r in gates}
    action_by_key: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in action:
        if row.get("stage") == "validation":
            action_by_key[(row["method"], row["model_seed"], row["data_seed"])].append(row)
    methods = (
        "handcrafted_state_exact_dynamics",
        "handcrafted_feature_ridge",
        "visual_state_estimator",
        "supervised_visual_absolute_raw",
        "supervised_visual_absolute_standardized",
        "supervised_visual_residual",
        "supervised_visual_action_shuffle",
    )
    _base_style()
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for row_index, coordinate in enumerate(COORDS):
        metrics = (
            (f"{coordinate}_mae", 0.01 if coordinate == "position" else 0.002, "Raw one-step MAE"),
            (f"{coordinate}_median_normalized_contrast_error", 0.25, "Action contrast error"),
            (f"{coordinate}_order_accuracy", 0.80, "Action-order accuracy"),
            (f"{coordinate}_clipped_fraction", 0.05, "Coordinate outside state bounds"),
        )
        for col_index, (field, threshold, title) in enumerate(metrics):
            ax = axes[row_index, col_index]
            ax.axhline(threshold, color="#A61B1B", linestyle="--", linewidth=1, label="frozen threshold")
            for method_index, method in enumerate(methods):
                for run in runs:
                    if run["method"] != method:
                        continue
                    model_seed, data_seed = run["model_seed"], run["data_seed"]
                    value = None
                    if col_index == 0:
                        if run.get(field, "") not in ("", "None"):
                            value = float(run[field])
                    else:
                        gate = gate_by_key.get((method, model_seed, data_seed))
                        if gate and gate.get(field, "") not in ("", "None"):
                            value = float(gate[field])
                        else:
                            rows = action_by_key.get((method, model_seed, data_seed), [])
                            if rows:
                                context_field = (
                                    f"{coordinate}_normalized_contrast_error"
                                    if "median_normalized_contrast_error" in field
                                    else field
                                )
                                row_values = [
                                    float(row[context_field])
                                    for row in rows
                                    if row.get(context_field, "") not in ("", "None")
                                ]
                                if "median_normalized_contrast_error" in field and row_values:
                                    value = float(np.median(row_values))
                                elif "order_accuracy" in field:
                                    correct_key = f"{coordinate}_order_correct"
                                    total_key = f"{coordinate}_order_total"
                                    correct = sum(int(row.get(correct_key, "0") or 0) for row in rows)
                                    total = sum(int(row.get(total_key, "0") or 0) for row in rows)
                                    value = correct / total if total else None
                                elif row_values:
                                    value = float(np.mean(row_values))
                    if value is None or not np.isfinite(value):
                        continue
                    offset = (int(data_seed[-1]) - 2) * 0.025
                    ax.scatter(method_index + offset, value, s=20, color=f"C{int(data_seed[-1])}")
            ax.set_title(title)
            ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=33, ha="right")
            if "contrast" in field:
                ax.set_yscale("symlog", linthresh=0.01)
            if "order" in field:
                ax.set_ylim(-0.03, 1.03)
            if "clipped_fraction" in field:
                ax.set_ylim(bottom=-0.01)
            if col_index == 0 and row_index == 0:
                ax.text(0.01, 0.98, "* State estimator predicts current state", transform=ax.transAxes, va="top", fontsize=7)
    fig.suptitle(
        "Per-coordinate validation diagnostics\nSynthetic MountainCarContinuous-v0 · full validation episodes 16–19; 20 shared action contexts per pair",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "Three independent seed pairs are shown. The state estimator predicts current state; other methods predict next state.",
        fontsize=7,
    )
    fig.legend(
        [Line2D([0], [0], marker="o", color="w", markerfacecolor=f"C{i}", markersize=6) for i in (1, 2, 3)],
        ["Data seed 9701", "Data seed 9702", "Data seed 9703"],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    stem = "figure-b-per-coordinate-validation"
    files = _save(fig, output, stem)
    return {
        "id": stem,
        "files": files,
        "caption": "Each point is one seed pair. State-estimator MAE measures current-state recovery; the other MAE values measure one-step next-state prediction. Dashed lines show frozen thresholds.",
        "alt_text": "Eight panels show position and velocity MAE, action-contrast error, action order, and clipping for all methods and seed pairs.",
        "sample_size": "Full validation partition for one-step errors; 20 shared action contexts per pair for action diagnostics",
        "limitations": "Seed pairs are the independent units. Action contexts are not independent replicates.",
    }


def _plot_representation_health(evidence: Path, output: Path) -> dict[str, object]:
    embeddings = _read(evidence / "embedding-diagnostics.csv")
    runs = _read(evidence / "model-runs.csv")
    probes = _read(evidence / "linear-probes.csv")
    by_group: dict[tuple[str, str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in embeddings:
        by_group[(row["method"], row["model_seed"], row["data_seed"])].append(row)
    run_by_group = {
        (row["method"], row["model_seed"], row["data_seed"]): row
        for row in runs
        if row.get("status") == "complete"
    }
    probe_by_group = {
        (row["method"], row["model_seed"], row["data_seed"]): row
        for row in probes
    }
    methods = tuple(method for method in NEURAL_METHODS if any(key[0] == method for key in by_group))
    _base_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    metrics = (
        ("low_spread", "Low-spread dimensions", 30.0, "max"),
        ("effective_rank", "Effective rank", 2.0, "min"),
        ("probe_position_r2", "Position probe R²", 0.80, "min"),
        ("probe_velocity_r2", "Velocity probe R²", 0.80, "min"),
    )
    colors = {"9701": "#176B87", "9702": "#C75B12", "9703": "#694A8E"}
    for axis, (metric, title, threshold, direction) in zip(axes.ravel(), metrics, strict=True):
        axis.axhline(threshold, color="#A61B1B", linestyle="--", linewidth=1)
        for method_index, method in enumerate(methods):
            for data_seed in ("9701", "9702", "9703"):
                for key in sorted(key for key in by_group if key[0] == method and key[2] == data_seed):
                    if metric == "low_spread":
                        value = sum(row.get("low_spread", "False").lower() == "true" for row in by_group[key])
                    elif metric == "effective_rank":
                        run = run_by_group.get(key)
                        value = float(run[metric]) if run and run.get(metric) not in ("", "None") else None
                    else:
                        probe = probe_by_group.get(key)
                        field = "probe_position_r2" if metric == "probe_position_r2" else "probe_velocity_r2"
                        value = float(probe[field]) if probe and probe.get(field) not in ("", "None") else None
                    if value is not None and np.isfinite(value):
                        jitter = (int(data_seed[-1]) - 2) * 0.055
                        axis.scatter(method_index + jitter, value, s=27, color=colors[data_seed], edgecolor="white", linewidth=0.4)
        axis.set_title(title, weight="bold")
        axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods], rotation=30, ha="right")
        axis.set_xlim(-0.5, max(0.5, len(methods) - 0.5))
        if direction == "min":
            axis.set_ylim(bottom=min(-0.05, axis.get_ylim()[0]))
        elif metric == "low_spread":
            axis.set_ylim(-1, 33)
    fig.suptitle("Representation health · validation only · synthetic MountainCar", fontsize=13, weight="bold")
    fig.legend(
        [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[key], markersize=6) for key in colors],
        [f"Data seed {key}" for key in colors],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.01,
        0.005,
        "Frozen gates: ≤30 low-spread dimensions; effective rank ≥2; both probe R² ≥0.80. Each point is one seed pair.",
        fontsize=7,
    )
    stem = "figure-c-representation-health"
    files = _save(fig, output, stem)
    return {
        "id": stem,
        "files": files,
        "caption": "Validation embedding spread and train-fitted linear state probes by model and seed pair. Dashed lines mark the frozen validity thresholds.",
        "alt_text": "Four panels show low-spread dimension counts, effective rank, position probe R squared, and velocity probe R squared for each neural method and independent seed pair.",
        "sample_size": "32 embedding dimensions and full validation rows per model; three paired model/data seeds",
        "limitations": "Probe scores use validation rows after fitting on training rows. Seed pairs are the independent units.",
    }


def _plot_learning_and_gradients(evidence: Path, output: Path) -> list[dict[str, object]]:
    learning = _read(evidence / "learning-curves.csv")
    gradients = _read(evidence / "gradient-diagnostics.csv")
    results: list[dict[str, object]] = []
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, coordinate in zip(axes, COORDS, strict=True):
        for method_index, method in enumerate(NEURAL_METHODS):
            for data_seed in ("9701", "9702", "9703"):
                selected = sorted(
                    (r for r in learning if r["method"] == method and r["data_seed"] == data_seed),
                    key=lambda r: int(r["epoch"]),
                )
                if not selected:
                    continue
                epochs = [int(r["epoch"]) for r in selected]
                color = f"C{method_index}"
                for key, linestyle, alpha in (
                    (f"train_{coordinate}_mse_raw", "-", 0.3),
                    (f"validation_{coordinate}_mse_raw", "--", 0.65),
                ):
                    values = [max(float(r[key]), 1e-12) for r in selected]
                    axis.plot(epochs, values, color=color, linestyle=linestyle, alpha=alpha, linewidth=0.9)
        axis.set_yscale("log")
        axis.set_title(f"{coordinate.capitalize()} raw MSE")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean squared error")
    fig.suptitle(
        "Learning curves by method · synthetic MountainCar · validation only · 3 pairs × 35 epochs",
        fontsize=13,
        weight="bold",
    )
    axes[0].legend(
        [Line2D([0], [0], color=f"C{i}") for i in range(len(NEURAL_METHODS))],
        [LABELS[method] for method in NEURAL_METHODS],
        frameon=False,
        fontsize=7,
        loc="best",
    )
    fig.text(
        0.01,
        0.005,
        "Raw MSE uses different position and velocity scales; the standardized objective is in the CSV record.",
        fontsize=7,
    )
    stem = "figure-d-learning-curves"
    files = _save(fig, output, stem)
    results.append({
        "id": stem,
        "files": files,
        "caption": "Solid traces show raw training MSE; dashed traces show raw validation MSE. Color identifies the method, and each seed pair remains visible.",
        "alt_text": "Two panels show position and velocity training and validation MSE over 35 epochs for the state estimator and four supervised visual dynamics arms.",
        "sample_size": "Three paired model/data seeds per neural method; 35 epochs per complete fit",
        "limitations": "Position and velocity scales differ. The target-scaled objective is retained in the raw CSV.",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, coordinate in zip(axes, COORDS, strict=True):
        for method_index, method in enumerate(NEURAL_METHODS):
            for module, linestyle in (("encoder", "-"), ("prediction_head", "--")):
                for data_seed in ("9701", "9702", "9703"):
                    selected = sorted(
                        (
                            r for r in gradients
                            if r["method"] == method
                            and r["coordinate"] == coordinate
                            and r["data_seed"] == data_seed
                        ),
                        key=lambda r: int(r["epoch"]),
                    )
                    if not selected:
                        continue
                    axis.plot(
                        [int(r["epoch"]) for r in selected],
                        [max(float(r[f"{module}_gradient_norm"]), 1e-12) for r in selected],
                        color=f"C{method_index}",
                        linestyle=linestyle,
                        alpha=0.3 + 0.1 * int(data_seed[-1]),
                        linewidth=0.9,
                    )
        axis.set_yscale("log")
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30, 35))
        axis.set_title(f"{coordinate.capitalize()} loss gradients")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Global L2 gradient norm")
    fig.suptitle(
        "Per-coordinate gradient magnitudes · fixed training batch · 3 pairs × 8 snapshots",
        fontsize=13,
        weight="bold",
    )
    axes[0].legend(
        [Line2D([0], [0], color="#444444", linestyle="-"), Line2D([0], [0], color="#444444", linestyle="--")]
        + [Line2D([0], [0], color=f"C{i}") for i in range(len(NEURAL_METHODS))],
        ["Encoder", "Prediction head"] + [LABELS[method] for method in NEURAL_METHODS],
        frameon=False,
        fontsize=7,
        loc="best",
        ncol=2,
    )
    fig.text(
        0.01,
        0.005,
        "Global norms depend on parameter count; compare within the same module and method.",
        fontsize=7,
    )
    stem = "figure-e-gradient-norms"
    files = _save(fig, output, stem)
    results.append({
        "id": stem,
        "files": files,
        "caption": "Solid traces show encoder gradients; dashed traces show prediction-head gradients at eight fixed epochs.",
        "alt_text": "Two panels show position-loss and velocity-loss encoder and prediction-head gradient norms for all methods and seed pairs.",
        "sample_size": "Eight snapshots per complete fit; three paired model/data seeds per method",
        "limitations": "Global norms depend on parameter count. Compare coordinates within the same module and method.",
    })
    return results


def _plot_heldout_rollouts(evidence: Path, output: Path) -> dict[str, object] | None:
    path = evidence / "rollout-trajectories.csv"
    if not path.is_file():
        return None
    rows = [row for row in _read(path) if row.get("status") == "complete"]
    if not rows:
        return None
    _base_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, coordinate in zip(axes, COORDS, strict=True):
        for method in sorted({row["method"] for row in rows}):
            horizons = sorted({int(row["horizon"]) for row in rows if row["method"] == method})
            means = []
            for horizon in horizons:
                errors = [
                    abs(float(row[f"{coordinate}_error"]))
                    for row in rows
                    if row["method"] == method and int(row["horizon"]) == horizon
                ]
                means.append(float(np.mean(errors)) if errors else np.nan)
            axis.plot(horizons, means, label=LABELS.get(method, method))
        axis.set_title(coordinate.capitalize())
        axis.set_xlabel("Rollout step")
        axis.set_ylabel(f"Mean absolute {coordinate} error")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Held-out multi-step prediction · no planning", fontsize=13, weight="bold")
    stem = "figure-f-heldout-rollout-errors"
    files = _save(fig, output, stem)
    return {
        "id": stem,
        "files": files,
        "caption": "Mean absolute prediction error by step for completed held-out rollouts. Failed and terminated trajectories remain in the raw record and are counted in completion diagnostics.",
        "alt_text": "Two panels show position and velocity errors over 60 steps for privileged exact dynamics, handcrafted exact dynamics, handcrafted ridge, and the validation-selected visual model.",
        "sample_size": "Five test seeds, four starts per seed, and three paired model/data fits; exact completed counts are in rollout-trajectories.csv.",
        "limitations": "Held-out prediction only. This does not measure a policy, planning outcome, or equal compute cost.",
    }


def generate_figures(*, evidence_dir: Path = EVIDENCE_DIR, output_dir: Path = FIGURE_DIR) -> dict[str, object]:
    required = (
        "action-triplets.csv",
        "action-summary.csv",
        "model-runs.csv",
        "validation-gates.csv",
        "learning-curves.csv",
        "gradient-diagnostics.csv",
        "embedding-diagnostics.csv",
        "linear-probes.csv",
    )
    missing = [name for name in required if not (evidence_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"v3 figure inputs are missing: {missing}")
    figures = [_plot_action(evidence_dir, output_dir, coordinate) for coordinate in COORDS]
    figures.append(_plot_validation(evidence_dir, output_dir))
    figures.append(_plot_representation_health(evidence_dir, output_dir))
    figures.extend(_plot_learning_and_gradients(evidence_dir, output_dir))
    heldout = _plot_heldout_rollouts(evidence_dir, output_dir)
    if heldout is not None:
        figures.append(heldout)
    source_names = list(required)
    if (evidence_dir / "rollout-trajectories.csv").is_file():
        source_names.append("rollout-trajectories.csv")
    manifest = {
        "protocol_version": "visual-jepa-supervised-positive-control-v3",
        "evidence_stage": "heldout_rollout" if heldout is not None else "validation_only",
        "source_evidence": {
            name: {
                "bytes": (evidence_dir / name).stat().st_size,
                "sha256": _hash(evidence_dir / name),
            }
            for name in source_names
        },
        "figures": [
            {
                **item,
                "sha256": {
                    name: {
                        "bytes": (output_dir / name).stat().st_size,
                        "sha256": _hash(output_dir / name),
                    }
                    for name in item["files"]
                },
            }
            for item in figures
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    content = ["# Supervised Visual Positive-Control v3 Figures", ""]
    for item in figures:
        content.extend(
            [
                f"## {item['id']}",
                "",
                f"![{item['alt_text']}]({item['files'][0]})",
                "",
                f"**Caption:** {item['caption']}",
                "",
                f"**Sample size:** {item['sample_size']}",
                "",
                f"**Limitations:** {item['limitations']}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(content), encoding="utf-8")
    return manifest


def verify_determinism(*, evidence_dir: Path = EVIDENCE_DIR) -> dict[str, object]:
    e_root = Path("/mnt/e/codex-research/jepa-world-models-tutorial")
    e_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jepa-v3-fig-a-", dir=e_root) as first_dir:
        with tempfile.TemporaryDirectory(prefix="jepa-v3-fig-b-", dir=e_root) as second_dir:
            first = generate_figures(evidence_dir=evidence_dir, output_dir=Path(first_dir))
            second = generate_figures(evidence_dir=evidence_dir, output_dir=Path(second_dir))
            left = [item["sha256"] for item in first["figures"]]
            right = [item["sha256"] for item in second["figures"]]
            if left != right:
                raise RuntimeError("v3 figure regeneration is not deterministic")
    return {"deterministic": True, "figure_count": len(first["figures"])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--verify-determinism", action="store_true")
    args = parser.parse_args(argv)
    result = (
        verify_determinism(evidence_dir=args.evidence_dir)
        if args.verify_determinism
        else generate_figures(evidence_dir=args.evidence_dir, output_dir=args.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
