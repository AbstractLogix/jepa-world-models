"""Plot only the completed controls from the incomplete v3 validation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
OUTPUT_DIR = ROOT / "docs/portfolio/figures/visual-jepa-supervised-positive-control-v3/partial-run"
SEEDS = ("9701", "9702", "9703")
METHODS = ("handcrafted_state_exact_dynamics", "handcrafted_feature_ridge")
METHOD_LABELS = {
    "handcrafted_state_exact_dynamics": "Handcrafted state + exact dynamics",
    "handcrafted_feature_ridge": "Handcrafted feature ridge",
}
COLORS = {"9701": "#176B87", "9702": "#C75B12", "9703": "#694A8E"}
COORDINATES = ("position", "velocity")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_observed_evidence(evidence_dir: Path) -> dict[str, object]:
    manifest_path = evidence_dir / "sha256-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = evidence_dir / name
        digest = expected["sha256"] if isinstance(expected, dict) else expected
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"evidence hash mismatch: {name}")
        if isinstance(expected, dict) and path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"evidence size mismatch: {name}")

    summary = json.loads((evidence_dir / "run-summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "incomplete"
        or summary.get("data_path_pass") is not True
        or summary.get("fit_count") != 0
        or summary.get("heldout_open") is not False
    ):
        raise ValueError("this renderer requires the retained incomplete, validation-only v3 run")
    if any(
        (evidence_dir / name).exists()
        for name in ("heldout-starts.csv", "rollout-trajectories.csv", "heldout-summary.json")
    ):
        raise ValueError("held-out records exist; use the held-out evidence renderer")

    return {
        "summary": summary,
        "protocol_lock": json.loads((evidence_dir / "protocol-lock.json").read_text(encoding="utf-8")),
        "data_path": _read_csv(evidence_dir / "data-path.csv"),
        "runs": _read_csv(evidence_dir / "model-runs.csv"),
        "actions": _read_csv(evidence_dir / "action-triplets.csv"),
        "scales": _read_csv(evidence_dir / "target-scales.csv"),
    }


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "svg.hashsalt": "jepa-v3-observed-controls",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
        }
    )


def _save_figure(fig, output: Path, stem: str) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    svg_path = output / f"{stem}.svg"
    png_path = output / f"{stem}.png"
    fig.savefig(
        svg_path,
        format="svg",
        metadata={"Creator": "JEPA World Models Tutorial", "Date": None},
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(png_path, format="png", dpi=220, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return [svg_path.name, png_path.name]


def _action_contexts(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[str, str], dict[str, dict[float, dict[str, float]]]]:
    contexts: dict[tuple[str, str], dict[str, dict[float, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row.get("stage") != "validation" or row.get("method") not in METHODS:
            continue
        key = (row["method"], row["data_seed"])
        contexts[key][row["start_id"]][float(row["action"])] = {
            "actual_position": float(row["actual_position"]),
            "actual_velocity": float(row["actual_velocity"]),
            "predicted_position": float(row["predicted_position_raw"]),
            "predicted_velocity": float(row["predicted_velocity_raw"]),
        }
    return contexts


def _plot_action_fidelity(data: Mapping[str, object], output: Path) -> dict[str, object]:
    actions = data["actions"]
    contexts = _action_contexts(actions)
    _configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    labels = {"actual": "Simulator", **{method: METHOD_LABELS[method] for method in METHODS}}
    colors = {"actual": "#333333", METHODS[0]: "#007C91", METHODS[1]: "#D55E00"}
    styles = {"actual": "--", METHODS[0]: "-", METHODS[1]: "-"}

    for column, seed in enumerate(SEEDS):
        for row_index, coordinate in enumerate(COORDINATES):
            axis = axes[row_index, column]
            for series in ("actual", *METHODS):
                samples: dict[float, list[float]] = {-1.0: [], 0.0: [], 1.0: []}
                source_method = METHODS[0] if series == "actual" else series
                for context in contexts.get((source_method, seed), {}).values():
                    if set(context) != {-1.0, 0.0, 1.0}:
                        continue
                    value_key = (
                        f"actual_{coordinate}" if series == "actual" else f"predicted_{coordinate}"
                    )
                    zero = context[0.0][value_key]
                    for action in (-1.0, 0.0, 1.0):
                        samples[action].append(context[action][value_key] - zero)
                if not samples[0.0]:
                    continue
                means = [float(np.mean(samples[action])) for action in (-1.0, 0.0, 1.0)]
                lower = [means[i] - min(samples[action]) for i, action in enumerate((-1.0, 0.0, 1.0))]
                upper = [max(samples[action]) - means[i] for i, action in enumerate((-1.0, 0.0, 1.0))]
                axis.errorbar(
                    (-1, 0, 1),
                    means,
                    yerr=np.asarray([lower, upper]),
                    color=colors[series],
                    linestyle=styles[series],
                    marker="o",
                    markersize=3,
                    capsize=2,
                    linewidth=1.2,
                    label=labels[series],
                )
            axis.axhline(0.0, color="#777777", linewidth=0.6)
            axis.set_title(f"Data seed {seed}")
            axis.set_xticks((-1, 0, 1), ("−1", "0", "+1"))
            axis.set_xlabel("Action")
            axis.set_ylabel(f"Mean Δ{coordinate} from action 0")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Action response of completed visual controls\n"
        "Synthetic MountainCarContinuous-v0 · validation only · 20 shared contexts per seed",
        fontsize=13,
        weight="bold",
    )
    stem = "figure-a-observed-action-fidelity"
    files = _save_figure(fig, output, stem)
    return {
        "id": stem,
        "files": files,
        "caption": "Simulator, handcrafted state plus exact dynamics, and handcrafted ridge action effects. The ridge uses the same handcrafted state features but a fixed linear map.",
        "alt_text": "Six panels show position and velocity changes across actions minus one, zero, and plus one for each of three validation data seeds. Solid lines show exact and ridge predictions. Dashed lines show simulator effects. Whiskers show the min–max range across 20 contexts.",
        "sample_size": "20 shared validation action contexts for each of 3 independent data seeds",
        "limitations": "No visual neural fit completed. This figure measures one-step prediction only and does not show planning performance.",
    }


def _plot_positive_controls(data: Mapping[str, object], output: Path) -> dict[str, object]:
    run_rows = [
        row
        for row in data["runs"]
        if row.get("method") in METHODS and row.get("status") == "complete"
    ]
    scale_rows = [
        row
        for row in data["scales"]
        if row.get("split") == "train" and row.get("target") == "next_state"
    ]
    variance_ratios: dict[str, float] = {}
    for seed in SEEDS:
        values = {
            row["coordinate"]: float(row["variance"])
            for row in scale_rows
            if row["data_seed"] == seed
        }
        variance_ratios[seed] = values["position"] / values["velocity"]

    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for axis, coordinate in zip(axes[:2], COORDINATES, strict=True):
        axis.set_title(f"{coordinate.capitalize()} one-step MAE")
        for method_index, method in enumerate(METHODS):
            selected = [row for row in run_rows if row["method"] == method]
            x = method_index
            for row in selected:
                seed = row["data_seed"]
                offset = (SEEDS.index(seed) - 1) * 0.08
                axis.scatter(
                    x + offset,
                    float(row[f"{coordinate}_mae"]),
                    s=36,
                    color=COLORS[seed],
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"Seed {seed}" if method_index == 0 else None,
                )
        axis.set_xticks(range(len(METHODS)), ("Visual + exact", "Feature ridge"))
        axis.set_ylabel("Mean absolute error")
        if coordinate == "position":
            axis.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", color="w", markerfacecolor=COLORS[seed],
                        label=f"Seed {seed}",
                    )
                    for seed in SEEDS
                ],
                frameon=False,
                fontsize=7,
                loc="upper left",
            )

    axis = axes[2]
    axis.set_title("Train target variance ratio")
    for index, seed in enumerate(SEEDS):
        axis.scatter(index, variance_ratios[seed], s=44, color=COLORS[seed], zorder=3)
        axis.annotate(
            f"{variance_ratios[seed]:.1f}×",
            (index, variance_ratios[seed]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    axis.set_xticks(range(len(SEEDS)), [f"{seed}" for seed in SEEDS])
    axis.set_ylabel("Position variance / velocity variance")
    axis.set_ylim(0, max(variance_ratios.values()) * 1.15)

    fig.suptitle(
        "Completed positive controls and run status\n"
        "Validation only · 3 independent data seeds · 0 of 15 neural fits completed",
        fontsize=13,
        weight="bold",
    )
    stem = "figure-b-observed-positive-controls"
    files = _save_figure(fig, output, stem)
    return {
        "id": stem,
        "files": files,
        "caption": "Raw validation MAE for handcrafted state plus exact dynamics and the fixed handcrafted-feature ridge model, alongside the training next-state position-to-velocity variance ratios.",
        "alt_text": "The first two panels show position and velocity one-step MAE for exact dynamics and handcrafted ridge across three seeds. The third panel shows next-state training variance ratios from 138.1 to 142.0. The title states that no neural fit completed.",
        "sample_size": "Full validation partition per seed for MAE; 20 training episodes per seed for target scales",
        "limitations": "The ridge model is a fixed linear baseline and cannot establish what a learned visual encoder can achieve. The neural study stopped before training.",
    }


def generate_figures(
    *, evidence_dir: Path = EVIDENCE_DIR, output_dir: Path = OUTPUT_DIR
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite figure output: {output_dir}")
    data = _load_observed_evidence(evidence_dir)
    figures = [
        _plot_action_fidelity(data, output_dir),
        _plot_positive_controls(data, output_dir),
    ]
    source_names = (
        "action-triplets.csv",
        "data-path.csv",
        "model-runs.csv",
        "run-summary.json",
        "target-scales.csv",
    )
    manifest = {
        "protocol_version": "visual-jepa-supervised-positive-control-v3",
        "evidence_status": data["summary"]["status"],
        "heldout_open": data["summary"]["heldout_open"],
        "protocol_sha256": data["protocol_lock"]["protocol_sha256"],
        "source_evidence": {
            name: {"bytes": (evidence_dir / name).stat().st_size, "sha256": _sha256(evidence_dir / name)}
            for name in source_names
        },
        "figures": [
            {
                **figure,
                "sha256": {
                    filename: {
                        "bytes": (output_dir / filename).stat().st_size,
                        "sha256": _sha256(output_dir / filename),
                    }
                    for filename in figure["files"]
                },
            }
            for figure in figures
        ],
    }
    (output_dir / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    readme = ["# v3 Observed Positive-Control Figures", ""]
    for figure in figures:
        readme.extend(
            [
                f"## {figure['id']}",
                "",
                f"![{figure['alt_text']}]({figure['files'][0]})",
                "",
                f"**Caption:** {figure['caption']}",
                "",
                f"**Sample size:** {figure['sample_size']}",
                "",
                f"**Limitations:** {figure['limitations']}",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return manifest


def verify_determinism(*, evidence_dir: Path = EVIDENCE_DIR) -> dict[str, object]:
    first_output = OUTPUT_DIR.parent / "determinism-check-a"
    second_output = OUTPUT_DIR.parent / "determinism-check-b"
    for path in (first_output, second_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite determinism output: {path}")
    try:
        first = generate_figures(evidence_dir=evidence_dir, output_dir=first_output)
        second = generate_figures(evidence_dir=evidence_dir, output_dir=second_output)
        left = [item["sha256"] for item in first["figures"]]
        right = [item["sha256"] for item in second["figures"]]
        if left != right:
            raise RuntimeError("observed v3 figure regeneration is not deterministic")
        return {"deterministic": True, "figure_count": len(first["figures"])}
    finally:
        for path in (first_output, second_output):
            if path.exists():
                for item in path.iterdir():
                    item.unlink()
                path.rmdir()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
