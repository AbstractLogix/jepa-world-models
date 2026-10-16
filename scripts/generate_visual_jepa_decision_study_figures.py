#!/usr/bin/env python3
"""Build the v1 evidence dashboard and method diagram from retained records."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

if __package__:
    from .analyze_visual_jepa_decision_study import action_effect_diagnostics, sha256
else:
    from analyze_visual_jepa_decision_study import action_effect_diagnostics, sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
FIGURES = ROOT / "docs/portfolio/figures"
PREFIX = "visual-jepa-decision-study-v1"
HISTORY_PREFIX = "visual-jepa"
COLORS = {
    "oracle": "#222222",
    "state_model": "#0072B2",
    "image_reconstruction": "#E69F00",
    "jepa": "#009E73",
    "direct_visual_momentum": "#CC79A7",
    "history": "#6B7280",
    "not_run": "#8A8A8A",
}
MARKERS = {
    "oracle": "X",
    "state_model": "o",
    "image_reconstruction": "s",
    "jepa": "^",
    "direct_visual_momentum": "D",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_complete_evidence(evidence: Path) -> dict[str, list[dict[str, str]]]:
    names = {
        "episodes": f"{PREFIX}-episodes.csv",
        "models": f"{PREFIX}-model-runs.csv",
        "probes": f"{PREFIX}-probe-test.csv",
        "interventions": f"{PREFIX}-interventions.csv",
        "historical_episodes": f"{HISTORY_PREFIX}-episodes.csv",
    }
    rows = {key: read_rows(evidence / name) for key, name in names.items()}
    expected_episode_counts = {
        "oracle_cem": 40,
        "direct_visual_momentum": 20,
        "state_model_cem": 200,
        "image_reconstruction_cem": 200,
        "jepa_cem": 200,
    }
    observed = defaultdict(int)
    for row in rows["episodes"]:
        observed[row["method"]] += 1
        if row.get("status") != "complete":
            raise ValueError("v1 episode evidence has missing or failed cases")
    if dict(observed) != expected_episode_counts:
        raise ValueError(f"Expected 660 complete v1 episodes; observed {dict(observed)}")
    if len(rows["models"]) != 15 or any(row.get("status") != "complete" for row in rows["models"]):
        raise ValueError("v1 evidence must contain 15 complete model fits")
    if Counter(row["method"] for row in rows["models"]) != Counter(
        {"state_model": 5, "image_reconstruction": 5, "jepa": 5}
    ):
        raise ValueError("v1 evidence must contain five model/data seed pairs per family")
    if len(rows["probes"]) != 10 or len(rows["interventions"]) != 960:
        raise ValueError("v1 probe or intervention evidence has incomplete case coverage")
    if Counter(row["family"] for row in rows["probes"]) != Counter(
        {"image_reconstruction": 5, "jepa": 5}
    ):
        raise ValueError("v1 evidence must contain five held-out probe rows per visual family")
    if any(row.get("status") != "complete" for row in rows["interventions"]):
        raise ValueError("v1 intervention evidence has missing or failed cases")
    intervention_groups: dict[tuple[str, str, str], set[float]] = defaultdict(set)
    for row in rows["interventions"]:
        intervention_groups[(row["method"], row["model_seed"], row["start_id"])].add(float(row["action"]))
    if len(intervention_groups) != 320 or any(
        actions != {-1.0, 0.0, 1.0} for actions in intervention_groups.values()
    ):
        raise ValueError("v1 intervention rows must preserve all 320 paired three-action groups")
    if len(rows["historical_episodes"]) != 440 or any(
        row.get("status") != "complete" for row in rows["historical_episodes"]
    ):
        raise ValueError("Notebook 22 evidence must contain 440 complete historical episodes")
    return rows


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total < 1:
        raise ValueError("Wilson interval needs at least one case")
    z = 1.959963984540054
    rate = successes / total
    denom = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denom
    half = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize_fixed(rows: list[dict[str, str]]) -> tuple[float, int, int, tuple[float, float]]:
    successes = sum(int(row["success"]) for row in rows)
    total = len(rows)
    return successes / total, successes, total, wilson(successes, total)


def seed_successes(rows: list[dict[str, str]]) -> list[tuple[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model_seed"]].append(row)
    return [
        (seed, float(np.mean([int(row["success"]) for row in cases])))
        for seed, cases in sorted(grouped.items())
    ]


def planner_ablation(axis: plt.Axes, historical: list[dict[str, str]], current: list[dict[str, str]]) -> None:
    labels = [
        "Direct visual-momentum controller",
        "Prior-only CEM",
        "JEPA CEM · Gaussian start",
        "JEPA CEM · momentum prior",
        "Oracle CEM · Gaussian start",
        "Oracle CEM · momentum prior",
        "Notebook 22 oracle CEM · init unspecified",
    ]
    y_by_label = {label: len(labels) - i - 1 for i, label in enumerate(labels)}
    axis.set_yticks(list(y_by_label.values()), labels, fontsize=7.7)
    axis.set_xlim(-0.05, 1.28)
    axis.set_ylim(-0.8, len(labels) - 0.2)
    axis.set_xlabel("Episode success rate")
    axis.set_title("A  Planner ablation: historical and new protocols", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.22)

    history_map = {
        "direct_visual_momentum": "Direct visual-momentum controller",
        "jepa_uniform_cem": "JEPA CEM · Gaussian start",
        "jepa_momentum_prior_cem": "JEPA CEM · momentum prior",
        "oracle_state_cem": "Notebook 22 oracle CEM · init unspecified",
    }
    current_map = {
        ("direct_visual_momentum", "direct"): "Direct visual-momentum controller",
        ("jepa_cem", "uniform"): "JEPA CEM · Gaussian start",
        ("jepa_cem", "momentum_prior"): "JEPA CEM · momentum prior",
        ("oracle_cem", "uniform"): "Oracle CEM · Gaussian start",
        ("oracle_cem", "momentum_prior"): "Oracle CEM · momentum prior",
    }
    history_jitter: dict[str, float] = {
        "direct_visual_momentum": -0.16,
        "jepa_uniform_cem": -0.11,
        "jepa_momentum_prior_cem": -0.11,
        "oracle_state_cem": -0.16,
    }
    for method, label in history_map.items():
        cases = [row for row in historical if row["method"] == method]
        if not cases:
            continue
        y = y_by_label[label] + history_jitter[method]
        color = COLORS["history"]
        if method in {"jepa_uniform_cem", "jepa_momentum_prior_cem"}:
            points = seed_successes(cases)
            values = [value for _, value in points]
            jitter = np.linspace(-0.075, 0.075, len(values))
            axis.scatter(values, y + jitter, marker="s", s=25, color=color, alpha=0.85)
            axis.scatter(np.mean(values), y, marker="D", s=38, color="#111111", zorder=4)
            axis.text(1.02, y, "5 seeds × 20", va="center", fontsize=7, color=color)
        else:
            rate, successes, total, interval = summarize_fixed(cases)
            axis.errorbar(
                rate,
                y,
                xerr=[[rate - interval[0]], [interval[1] - rate]],
                fmt="s",
                color=color,
                capsize=2,
                markersize=5,
            )
            axis.text(1.02, y, f"{successes}/{total}", va="center", fontsize=7, color=color)

    current_jitter: dict[str, float] = {
        "direct_visual_momentum": 0.16,
        "jepa_cem": 0.11,
        "oracle_cem": 0.16,
    }
    for (method, initialization), label in current_map.items():
        cases = [
            row
            for row in current
            if row["method"] == method and row["initialization"] == initialization
        ]
        if not cases:
            continue
        y = y_by_label[label] + current_jitter[method]
        color = COLORS["jepa"] if method == "jepa_cem" else (
            COLORS["direct_visual_momentum"] if method == "direct_visual_momentum" else COLORS["oracle"]
        )
        if method == "jepa_cem":
            points = seed_successes(cases)
            values = [value for _, value in points]
            jitter = np.linspace(-0.075, 0.075, len(values))
            axis.scatter(values, y + jitter, marker="o", s=25, color=color, alpha=0.85)
            axis.scatter(np.mean(values), y, marker="D", s=38, color="#111111", zorder=4)
            axis.text(1.02, y, "5 seeds × 20", va="center", fontsize=7, color=color)
        else:
            rate, successes, total, interval = summarize_fixed(cases)
            axis.errorbar(
                rate,
                y,
                xerr=[[rate - interval[0]], [interval[1] - rate]],
                fmt="o",
                color=color,
                capsize=2,
                markersize=5,
            )
            axis.text(1.02, y, f"{successes}/{total}", va="center", fontsize=7, color=color)

    axis.text(
        0.02,
        y_by_label["Prior-only CEM"],
        "Not run in either study",
        va="center",
        color=COLORS["not_run"],
        fontsize=8,
        style="italic",
    )


def intervention_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "complete":
            grouped[(row["method"], row["model_seed"], row["start_id"])].append(row)
    sums: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0.0, "total": 0.0, "position_error": 0.0, "position_n": 0.0})
    )
    for (method, seed, _start), group in grouped.items():
        sums[method][seed]["position_error"] += float(
            np.mean([float(row["position_abs_error"]) for row in group])
        )
        sums[method][seed]["position_n"] += 1
        if len(group) != 3:
            raise ValueError(f"intervention group {method}/{seed} must contain three actions")
        for first in range(3):
            for second in range(first + 1, 3):
                actual = float(group[first]["observed_return"]) - float(group[second]["observed_return"])
                predicted = float(group[first]["predicted_return"]) - float(group[second]["predicted_return"])
                if abs(actual) > 1e-8:
                    sums[method][seed]["total"] += 1
                    sums[method][seed]["correct"] += int(actual * predicted > 0.0)
    return {
        method: {
            seed: {
                "position_mae": values["position_error"] / values["position_n"],
                "ranking_accuracy": values["correct"] / values["total"] if values["total"] else np.nan,
                "ranking_pairs": values["total"],
            }
            for seed, values in by_seed.items()
        }
        for method, by_seed in sums.items()
    }


def action_ranking(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    metrics = intervention_metrics(rows)
    families = ("state_model", "image_reconstruction", "jepa")
    x = np.arange(len(families))
    for index, family in enumerate(families):
        per_seed = metrics[family]
        values = [per_seed[seed]["ranking_accuracy"] for seed in sorted(per_seed)]
        axis.scatter(np.full(len(values), index), values, s=30, color=COLORS[family], marker=MARKERS[family], zorder=3)
        axis.scatter(index, np.mean(values), s=45, color="#111111", marker="D", zorder=4)
    axis.set_xlim(-0.5, 2.55)
    axis.set_xticks(x, ["State model", "Image reconstruction", "JEPA"], fontsize=8)
    axis.set_ylim(-0.03, 1.02)
    axis.set_ylabel("Action-pair ranking accuracy")
    axis.set_title("B  Common task-return ranking", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=0.22)


def draw_representation_diagnostics(
    axes: list[plt.Axes],
    models: list[dict[str, str]],
    probes: list[dict[str, str]],
    interventions: list[dict[str, str]],
) -> None:
    if len(axes) != 4:
        raise ValueError("representation diagnostics need four axes")
    jepa_models = sorted((row for row in models if row["method"] == "jepa"), key=lambda row: row["model_seed"])
    if len(jepa_models) != 5:
        raise ValueError("representation plot needs five JEPA seed pairs")
    seeds = [row["model_seed"] for row in jepa_models]
    spread = np.asarray([max(float(row["embedding_mean_std"]), 1e-9) for row in jepa_models])
    low_dims = [int(row["embedding_low_variance_dims"]) for row in jepa_models]
    axes[0].scatter(np.arange(5), spread, color=COLORS["jepa"], marker="^", s=34)
    axes[0].axhline(1e-3, color="#555555", linestyle="--", linewidth=1, label="1e-3 screen")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(5), [seed[-2:] for seed in seeds])
    axes[0].set_ylabel("Mean per-dimension embedding SD")
    axes[0].set_title("C1 · Embedding spread and low-spread count", fontsize=8)
    for index, count in enumerate(low_dims):
        axes[0].annotate(f"{count}/32", (index, spread[index]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.5)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    axes[0].grid(axis="y", alpha=0.2)

    probe_map = {row["model_seed"]: row for row in probes if row["family"] == "jepa"}
    probe_x = np.arange(5)
    for field, label, offset, color, marker in (
        ("position_r2", "Position", -0.1, "#0072B2", "o"),
        ("velocity_r2", "Velocity", 0.1, "#D55E00", "s"),
    ):
        values = [float(probe_map[seed][field]) for seed in seeds]
        axes[1].scatter(probe_x + offset, values, color=color, marker=marker, s=28, label=label)
    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].set_xticks(probe_x, [seed[-2:] for seed in seeds])
    axes[1].set_ylim(-0.15, 1.05)
    axes[1].set_ylabel("Held-out linear-probe R²")
    axes[1].set_title("C2 · Test probes (diagnostic only)", fontsize=8)
    axes[1].legend(frameon=False, fontsize=6)
    axes[1].grid(axis="y", alpha=0.2)

    action = {row["model_seed"]: row for row in action_effect_diagnostics(interventions)}
    for axis, dimension, title, color in (
        (axes[2], "position", "Next-position action span", "#0072B2"),
        (axes[3], "velocity", "Next-velocity action span", "#D55E00"),
    ):
        observed = np.asarray([action[seed][f"observed_{dimension}_span"] for seed in seeds])
        predicted = np.asarray([action[seed][f"predicted_{dimension}_span"] for seed in seeds])
        observed_sd = np.asarray([action[seed][f"observed_{dimension}_span_sd"] for seed in seeds])
        predicted_sd = np.asarray([action[seed][f"predicted_{dimension}_span_sd"] for seed in seeds])
        axis.errorbar(observed, predicted, xerr=observed_sd, yerr=predicted_sd, fmt="none", ecolor="#999999", alpha=0.7, linewidth=0.7)
        axis.scatter(observed, predicted, color=color, marker="^", s=28)
        limit = max(float(np.max(observed + observed_sd)), float(np.max(predicted + predicted_sd)), 1e-6)
        axis.plot((0, limit), (0, limit), color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xlim(0, limit * 1.08)
        axis.set_ylim(0, limit * 1.08)
        axis.set_xlabel("Observed span")
        axis.set_ylabel("Predicted span")
        prefix = "C3 · " if dimension == "position" else "C4 · "
        axis.set_title(prefix + title, fontsize=8)
        axis.grid(alpha=0.2)


def decision_compute(axis_calls: plt.Axes, axis_time: plt.Axes, rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row.get("model_seed") or "fixed", row["initialization"])].append(row)
    for (method, seed, initialization), cases in sorted(grouped.items()):
        family = "oracle" if method == "oracle_cem" else method.removesuffix("_cem")
        color = COLORS.get(family, "#555555")
        marker = MARKERS.get(family, "o")
        filled = initialization not in {"uniform", "direct"}
        face = color if filled else "white"
        success_rate = float(np.mean([int(row["success"]) for row in cases]))
        calls = float(np.mean([int(row["transition_calls"]) for row in cases])) / 1e6
        elapsed = float(np.mean([float(row["elapsed_seconds"]) for row in cases]))
        for axis, x_value in ((axis_calls, calls), (axis_time, elapsed)):
            axis.scatter(x_value, success_rate, color=color, marker=marker, facecolors=face,
                         edgecolors=color, s=28 if seed == "fixed" else 22,
                         alpha=0.85)
    for axis in (axis_calls, axis_time):
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("Success rate")
        axis.grid(alpha=0.2)
    axis_calls.set_xlabel("Mean transition calls / episode (millions)")
    axis_time.set_xlabel("Mean elapsed seconds / episode")
    axis_calls.set_title("D1 · Declared model work", fontsize=8)
    axis_time.set_title("D2 · Measured wall time", fontsize=8)


def overview(evidence: Path, output: Path) -> Path:
    data = require_complete_evidence(evidence)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig = plt.figure(figsize=(16, 11.5), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.35), hspace=0.28, wspace=0.35)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    planner_ablation(ax_a, data["historical_episodes"], data["episodes"])
    action_ranking(ax_b, data["interventions"])
    diagnostics = GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[1, 0], hspace=0.62, wspace=0.48)
    diagnostic_axes = [
        fig.add_subplot(diagnostics[row, column]) for row in range(2) for column in range(2)
    ]
    draw_representation_diagnostics(
        diagnostic_axes, data["models"], data["probes"], data["interventions"]
    )
    compute = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1, 1], wspace=0.48)
    call_axis, time_axis = (fig.add_subplot(compute[0, index]) for index in range(2))
    decision_compute(call_axis, time_axis, data["episodes"])
    legend_handles = [
        Line2D([], [], color=COLORS[family], marker=MARKERS[family], linestyle="none", label=label)
        for family, label in (
            ("state_model", "State model"),
            ("image_reconstruction", "Image model"),
            ("jepa", "JEPA"),
            ("oracle", "Oracle"),
            ("direct_visual_momentum", "Direct control"),
        )
    ] + [
        Line2D([], [], color="#333333", marker="o", markerfacecolor="white", linestyle="none", label="Gaussian start"),
        Line2D([], [], color="#333333", marker="o", markerfacecolor="#333333", linestyle="none", label="Momentum prior"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=False,
        fontsize=7.2,
    )
    fig.suptitle(
        "Matched visual JEPA decision study v1 · synthetic MountainCarContinuous",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.035,
        0.008,
        "Notebook 22 squares and Notebook 23 circles are separate protocols. Trained-method points show five paired model/data seeds. "
        "Fixed-method intervals use Wilson 95% intervals over starts. In D, color marks method; open/filled marks Gaussian/momentum-prior initialization.\n"
        "All JEPA seeds triggered the low-spread warning; their planning outcomes are exploratory. The final starts also served as the oracle gate.\n"
        "Prior-only CEM was not run. Oracle timing used CPU and learned models used GPU; calls and elapsed time do not show compute parity.",
        ha="left",
        fontsize=7.2,
        va="bottom",
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{PREFIX}-overview.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def architecture_svg(output: Path) -> Path:
    """Write a static, accessible SVG of the frozen v1 method and control groups."""
    boxes = [
        (28, "Rendered frame pair", "Generated MountainCar image"),
        (225, "Visual encoder", "Current latent zₜ"),
        (422, "Action predictor", "Predict ẑₜ₊₁ from zₜ, aₜ"),
        (619, "Linear state readout", "Fit on train; select on validation"),
        (816, "CEM planner", "Gaussian or momentum-prior start"),
        (1013, "Environment", "Execute first action; render again"),
    ]
    box_markup = []
    for x, title, subtitle in boxes:
        box_markup.append(
            f'<rect x="{x}" y="160" width="170" height="112" rx="12" fill="#F4F7FB" stroke="#315B7D" stroke-width="2"/>'
            f'<text x="{x + 85}" y="205" class="title" text-anchor="middle">{html.escape(title)}</text>'
            f'<text x="{x + 85}" y="237" class="body" text-anchor="middle">{html.escape(subtitle)}</text>'
        )
    arrows = []
    for left in (198, 395, 592, 789, 986):
        arrows.append(f'<line x1="{left}" y1="216" x2="{left + 27}" y2="216" class="flow" marker-end="url(#arrow)"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="620" viewBox="0 0 1240 620" role="img" aria-labelledby="title desc">
<title id="title">Matched visual JEPA decision study architecture</title>
<desc id="desc">A generated frame pair is encoded. An action-conditioned predictor predicts the next latent. A train-fitted, validation-selected linear readout maps predictions to position and velocity for the shared task return. CEM searches action sequences with either Gaussian or handcrafted momentum-prior initialization. The environment executes the first action and returns a new rendered frame. Separate privileged controls use state inputs.</desc>
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#315B7D"/></marker><style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#172B3A}} .title{{font-size:16px;font-weight:700}} .body{{font-size:12px}} .small{{font-size:13px}} .flow{{stroke:#315B7D;stroke-width:3;fill:none}} .visual{{fill:#E8F5EF;stroke:#23835B;stroke-width:2}} .priv{{fill:#FFF4E5;stroke:#B56600;stroke-width:2;stroke-dasharray:7 5}} .policy{{fill:#F7EDF4;stroke:#9A4E7B;stroke-width:2}}
</style></defs>
<rect width="1240" height="620" fill="#FFFFFF"/>
<text x="30" y="45" font-size="22" font-weight="700">Matched visual JEPA decision study v1</text>
<text x="30" y="76" class="small">Visual-model planning loop · synthetic task · no pretrained checkpoint or external dataset</text>
{''.join(box_markup)}
{''.join(arrows)}
<path d="M901 272 C901 332 507 332 507 272" class="flow" marker-end="url(#arrow)"/>
<text x="700" y="326" class="small" text-anchor="middle">Candidate action sequences are scored through predicted state and the shared Gymnasium return</text>
<path d="M1100 273 C1100 386 113 386 113 274" fill="none" stroke="#23835B" stroke-width="2" marker-end="url(#arrow)"/>
<text x="603" y="411" class="small" text-anchor="middle" fill="#23835B">Executed action produces the next rendered observation</text>
<rect x="55" y="460" width="330" height="90" rx="10" class="policy"/>
<text x="220" y="495" class="title" text-anchor="middle">Direct visual-momentum controller</text>
<text x="220" y="523" class="body" text-anchor="middle">Handcrafted policy; one result per shared start</text>
<rect x="440" y="460" width="350" height="90" rx="10" class="priv"/>
<text x="615" y="495" class="title" text-anchor="middle">Learned state-space dynamics</text>
<text x="615" y="523" class="body" text-anchor="middle">Privileged simulator state; not a visual JEPA</text>
<rect x="845" y="460" width="350" height="90" rx="10" class="priv"/>
<text x="1020" y="495" class="title" text-anchor="middle">Exact-dynamics oracle</text>
<text x="1020" y="523" class="body" text-anchor="middle">Privileged state; validates planner settings</text>
<text x="55" y="590" class="small">The direct controller supplies the momentum prior. A prior-only CEM arm is not in this protocol. Evaluator-side intervention continuation uses simulator state.</text>
</svg>'''
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{PREFIX}-architecture.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def figure_metadata(evidence: Path, output: Path, data: dict[str, list[dict[str, str]]]) -> Path:
    source_names = (
        f"{PREFIX}-episodes.csv",
        f"{PREFIX}-model-runs.csv",
        f"{PREFIX}-probe-test.csv",
        f"{PREFIX}-interventions.csv",
        f"{HISTORY_PREFIX}-episodes.csv",
    )
    collapsed = any(
        row.get("method") == "jepa" and row.get("embedding_collapsed", "").lower() == "true"
        for row in data["models"]
    )
    figures = [
        {
            "path": f"{PREFIX}-overview.png",
            "panels": ["A planner ablation", "B common-return action ranking", "C representation and action-effect diagnostics", "D decision value versus compute"],
            "samples": {
                "historical_episode_rows": len(data["historical_episodes"]),
                "new_episode_rows": len(data["episodes"]),
                "model_data_seed_pairs_per_trained_family": 5,
                "shared_final_starts": 20,
                "intervention_starts": 20,
            },
            "aggregation": "trained methods show one point per model/data seed pair; fixed methods use episode rates",
            "intervals": "Wilson 95% intervals for fixed-start direct/oracle outcomes; no interval across the five trained pairs",
            "limits": [
                "Notebook 22 and v1 use separate protocols and are not pooled.",
                "The final starts also served as the oracle gate; model outcomes are conditional on its pass.",
                "Prior-only CEM was not run and is shown as not run.",
                "All five v1 JEPA seeds triggered the low-spread warning; JEPA planning outcomes are exploratory.",
                "Oracle and learned-method timings use different devices.",
            ],
        },
        {
            "path": f"{PREFIX}-architecture.svg",
            "panels": ["E experiment architecture"],
            "samples": {},
            "aggregation": "static diagram generated from the versioned v1 method",
            "intervals": "not applicable",
            "limits": ["This diagram describes the local synthetic study and does not show a paper implementation."],
        },
    ]
    value = {
        "schema_version": 1,
        "experiment_version": PREFIX,
        "evidence_status": "partial" if collapsed else "descriptive_local_comparison",
        "source_hashes_sha256": {name: sha256(evidence / name) for name in source_names},
        "generator_sha256": sha256(Path(__file__).resolve()),
        "figures": figures,
    }
    path = output / f"{PREFIX}-figures.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def generate(evidence: Path, output: Path) -> list[Path]:
    data = require_complete_evidence(evidence)
    paths = [overview(evidence, output), architecture_svg(output)]
    figure_metadata(evidence, output, data)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for path in generate(args.evidence_dir, args.output_dir):
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
