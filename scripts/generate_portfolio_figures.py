"""Build portfolio figures from retained experiment evidence."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
T_CRITICAL_95_DF4 = 2.7764451051977987
METHOD_LABELS = {
    "learned_model_mpc": "Learned-model MPC",
    "direct_momentum": "Direct momentum rule",
    "uniform_cem": "Neutral-start CEM",
    "momentum_prior_cem": "Momentum-prior CEM",
}
COLORS = {
    "learned_model_mpc": "#0072B2",
    "direct_momentum": "#666666",
    "uniform_cem": "#0072B2",
    "momentum_prior_cem": "#009E73",
}


class _HtmlTables(HTMLParser):
    """Read table rows from the saved notebook HTML output."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell is not None and self.row is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes and total must define a non-empty binomial sample")
    rate = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (rate + z2 / (2 * total)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / total + z2 / (4 * total * total)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _notebook_episode_rows(path: Path) -> list[dict[str, str]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if not html:
                continue
            parser = _HtmlTables()
            parser.feed("".join(html) if isinstance(html, list) else html)
            for table in parser.tables:
                if not table:
                    continue
                header = table[0]
                if {"Method", "Seed", "Status", "Success"}.issubset(header):
                    return [dict(zip(header, values)) for values in table[1:] if len(values) == len(header)]
    raise ValueError(f"saved episode table not found in {path.relative_to(ROOT)}")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _summarize_episodes(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("Status", row.get("status")) != "complete":
            raise ValueError("all episode evidence rows must be complete")
        method = row.get("Method", row.get("method"))
        if not method:
            raise ValueError("episode evidence row has no method")
        grouped[method].append(row)

    result: dict[str, dict[str, Any]] = {}
    for method, method_rows in grouped.items():
        successes = sum(
            row.get("Success", row.get("success", "")).strip().lower() == "true"
            for row in method_rows
        )
        total = len(method_rows)
        low, high = wilson_interval(successes, total)
        result[method] = {
            "successes": successes,
            "total": total,
            "rate": successes / total,
            "wilson_low": low,
            "wilson_high": high,
        }
    return result


def summarize_decision_usefulness() -> dict[str, dict[str, Any]]:
    return _summarize_episodes(_notebook_episode_rows(ROOT / "notebooks/14_decision_usefulness.ipynb"))


def summarize_action_video() -> dict[str, dict[str, Any]]:
    return _summarize_episodes(_csv_rows(ROOT / "docs/evidence/action-video-planning-episodes.csv"))


def summarize_calibration() -> dict[tuple[str, str], dict[str, float]]:
    run_rows = _csv_rows(ROOT / "docs/evidence/model-uncertainty-runs.csv")
    complete_seeds = {
        row["model_seed"]
        for row in run_rows
        if row["status"] == "complete" and row["calibration_status"] == "complete"
    }
    if len(complete_seeds) != 5:
        raise ValueError("expected five complete independent model runs")

    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    with (ROOT / "docs/evidence/model-uncertainty-calibration.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            if row["model_seed"] not in complete_seeds:
                raise ValueError("calibration evidence contains a seed without a complete run")
            if row["split"] not in {"in_distribution_test", "policy_shift_test"}:
                continue
            key = (row["split"], row["component"], row["model_seed"])
            counts[key][1] += 1
            counts[key][0] += row["covered"].strip().lower() == "true"

    per_run: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (split, component, seed), (covered, total) in counts.items():
        if total == 0 or seed not in complete_seeds:
            raise ValueError("empty or invalid calibration run group")
        per_run[(split, component)].append(covered / total)

    summaries: dict[tuple[str, str], dict[str, float]] = {}
    for key, values in per_run.items():
        if len(values) != 5:
            raise ValueError(f"expected five model-run values for {key}")
        average = mean(values)
        half_width = T_CRITICAL_95_DF4 * stdev(values) / math.sqrt(len(values))
        summaries[key] = {
            "mean": average,
            "lower_95": average - half_width,
            "upper_95": average + half_width,
            "runs": len(values),
        }
    if len(summaries) != 4:
        raise ValueError("expected two test splits for each of two state components")
    return summaries


def summarize_visual_jepa_rankings() -> dict[str, list[dict[str, Any]]]:
    rows = _csv_rows(ROOT / "docs/evidence/visual-jepa-interventions.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "complete" or row["error"]:
            raise ValueError("all visual JEPA intervention rows must be complete")
        grouped[(row["method"], row["model_seed"], row["start_id"])].append(row)

    per_seed: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    starts_per_seed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (method, model_seed, _), case_rows in grouped.items():
        starts_per_seed[(method, model_seed)].add(case_rows[0]["start_id"])
        if len(case_rows) != 3 or {row["action"] for row in case_rows} != {"-1.0", "0.0", "1.0"}:
            raise ValueError("each intervention start must include all three actions")
        for index, left in enumerate(case_rows):
            for right in case_rows[index + 1 :]:
                actual_delta = float(left["observed_score"]) - float(right["observed_score"])
                if abs(actual_delta) < 1e-8:
                    continue
                predicted_delta = float(left["predicted_score"]) - float(right["predicted_score"])
                per_seed[(method, model_seed)][1] += 1
                per_seed[(method, model_seed)][0] += actual_delta * predicted_delta > 0

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (method, seed), (correct, total) in sorted(per_seed.items()):
        if total == 0:
            raise ValueError("each model seed must have non-tied action pairs")
        result[method].append({
            "seed": seed,
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        })
    expected = {"oracle": 1, "state_model": 5, "image_reconstruction": 5, "jepa": 5}
    if {method: len(values) for method, values in result.items()} != expected:
        raise ValueError("visual JEPA action-ranking evidence has unexpected seed coverage")
    if any(len(starts) != 20 for starts in starts_per_seed.values()):
        raise ValueError("each intervention seed group must contain 20 starts")
    return dict(result)


def summarize_visual_jepa_episodes() -> dict[str, Any]:
    rows = _csv_rows(ROOT / "docs/evidence/visual-jepa-episodes.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "complete" or row["error"]:
            raise ValueError("all visual JEPA episode rows must be complete")
        grouped[(row["method"], row["model_seed"])].append(row)

    summaries: dict[str, Any] = {"trained": defaultdict(list), "baselines": {}}
    trained_methods = {
        "state_model_cem",
        "image_reconstruction_cem",
        "jepa_uniform_cem",
        "jepa_momentum_prior_cem",
    }
    for (method, seed), method_rows in sorted(grouped.items()):
        successes = sum(int(row["success"]) for row in method_rows)
        total = len(method_rows)
        if method in trained_methods:
            if total != 20:
                raise ValueError(f"expected 20 paired episodes for {method} seed {seed}")
            summaries["trained"][method].append({
                "seed": seed,
                "successes": successes,
                "total": total,
                "rate": successes / total,
            })
        else:
            if method in summaries["baselines"] or total != 20:
                raise ValueError(f"expected one 20-start baseline group for {method}")
            low, high = wilson_interval(successes, total)
            summaries["baselines"][method] = {
                "successes": successes,
                "total": total,
                "rate": successes / total,
                "wilson_low": low,
                "wilson_high": high,
            }
    if {method: len(values) for method, values in summaries["trained"].items()} != {
        "state_model_cem": 5,
        "image_reconstruction_cem": 5,
        "jepa_uniform_cem": 5,
        "jepa_momentum_prior_cem": 5,
    }:
        raise ValueError("visual JEPA episode evidence has unexpected model-seed coverage")
    if set(summaries["baselines"]) != {"oracle_state_cem", "direct_visual_momentum"}:
        raise ValueError("visual JEPA episode evidence has unexpected baselines")
    summaries["trained"] = dict(summaries["trained"])
    return summaries


def _plot_success_panel(ax: Any, title: str, note: str, summary: dict[str, dict[str, Any]], order: list[str]) -> None:
    y_values = list(reversed(range(len(order))))
    for y, method in zip(y_values, order):
        row = summary[method]
        rate = row["rate"]
        ax.errorbar(
            rate,
            y,
            xerr=[[rate - row["wilson_low"]], [row["wilson_high"] - rate]],
            fmt="o",
            markersize=7,
            capsize=4,
            elinewidth=1.6,
            color=COLORS[method],
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        label_offset = -0.2 if rate < 0.5 or y > 0 else 0.2
        ax.text(rate, y + label_offset, f"{row['successes']}/{row['total']}", fontsize=9, ha="center", va="center")
    ax.set_yticks(y_values, [METHOD_LABELS[method] for method in order])
    ax.set_xlim(-0.04, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("Closed-loop success rate")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.text(0, -0.27, note, transform=ax.transAxes, fontsize=8.5, va="top", wrap=True)
    ax.grid(axis="x", color="#d8dee4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)


def plot_decision_usefulness(output_dir: Path) -> Path:
    notebook14 = summarize_decision_usefulness()
    notebook16 = summarize_action_video()
    expected14 = {"learned_model_mpc": (0, 20), "direct_momentum": (20, 20)}
    expected16 = {"uniform_cem": (0, 20), "momentum_prior_cem": (20, 20), "direct_momentum": (20, 20)}
    for observed, expected in ((notebook14, expected14), (notebook16, expected16)):
        actual = {method: (row["successes"], row["total"]) for method, row in observed.items()}
        if actual != expected:
            raise ValueError(f"episode success summary changed: {actual}")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), sharex=True)
    _plot_success_panel(
        axes[0],
        "Notebook 14 · state model",
        "60/60 intervention pairs ranked correctly; one deterministic fitted model; goal 0.25.",
        notebook14,
        ["learned_model_mpc", "direct_momentum"],
    )
    _plot_success_panel(
        axes[1],
        "Notebook 16 · rendered features",
        "60/60 intervention pairs ranked correctly; fixed color-moment features; goal 0.45.",
        notebook16,
        ["uniform_cem", "momentum_prior_cem", "direct_momentum"],
    )
    fig.suptitle(
        "Accurate one-step action rankings do not ensure useful planning",
        x=0.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.015,
        "MountainCarContinuous. Each panel is a separate 20-start evaluation. Bars are 95% Wilson intervals across starts, not training runs. Notebook 14 uses a state model; Notebook 16 uses handcrafted visual features. Neither is a JEPA result.",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout(rect=(0.02, 0.12, 0.99, 0.88), w_pad=2.7)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "decision-usefulness.png"
    fig.savefig(target, dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return target


def plot_uncertainty_calibration(output_dir: Path) -> Path:
    summary = summarize_calibration()
    splits = ["in_distribution_test", "policy_shift_test"]
    labels = ["Random-action test", "Planner-shift test"]
    colors = {"position": "#0072B2", "velocity": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, component, title in zip(axes, ["position", "velocity"], ["Position", "Velocity"]):
        means = [summary[(split, component)]["mean"] for split in splits]
        lows = [summary[(split, component)]["lower_95"] for split in splits]
        highs = [summary[(split, component)]["upper_95"] for split in splits]
        ax.errorbar(
            [0, 1],
            means,
            yerr=[[m - low for m, low in zip(means, lows)], [high - m for m, high in zip(means, highs)]],
            fmt="o-",
            color=colors[component],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            capsize=4,
            linewidth=1.5,
        )
        for x, value in enumerate(means):
            ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        ax.axhline(0.90, color="#555555", linestyle="--", linewidth=1, label="90% target")
        ax.set_xticks([0, 1], labels)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylim(0.75, 1.0)
        ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical coverage of 90% prediction intervals")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Prediction intervals under random and planner actions", x=0.03, ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.03,
        0.015,
        "Notebook 17 MountainCar state-space Ridge ensembles. Points and 95% Student-t intervals summarize five independent model fits. Planner-shift intervals are wide; this does not establish calibration on other tasks or JEPA models.",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout(rect=(0.03, 0.12, 0.99, 0.88), w_pad=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "uncertainty-calibration.png"
    fig.savefig(target, dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return target


def plot_visual_jepa_action_ranking(output_dir: Path) -> Path:
    summary = summarize_visual_jepa_rankings()
    methods = ["state_model", "image_reconstruction", "jepa"]
    labels = ["Privileged state model", "Image reconstruction", "Action-conditioned JEPA"]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for y, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = summary[method]
        accuracies = [row["accuracy"] for row in values]
        jitter = [((index - 2) * 0.055) for index in range(len(values))]
        ax.scatter(
            accuracies,
            [y + offset for offset in jitter],
            color=color,
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            s=55,
            zorder=3,
            label=None,
        )
        average = mean(accuracies)
        ax.scatter(
            [average],
            [y],
            marker="D",
            s=82,
            color="#202938",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        total_correct = sum(row["correct"] for row in values)
        total_pairs = sum(row["total"] for row in values)
        ax.text(1.01, y, f"{total_correct}/{total_pairs}", va="center", fontsize=9)

    ax.axvline(0.5, color="#687386", linestyle="--", linewidth=1, label="50% reference")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0.0, 1.12)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"])
    ax.set_xlabel("Correct ranking of non-tied action pairs")
    ax.set_title("Held-out action ranking varies across model seeds", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#d8dee4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    fig.text(0.02, 0.13, "Each colored dot is one trained seed; the diamond is the mean across five seeds.", fontsize=9)
    fig.text(0.02, 0.07, "Scores use different spaces (state, pixel, latent); compare within a method, not across methods.", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.23, 0.99, 0.98))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "visual-jepa-intervention-ranking.png"
    fig.savefig(target, dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return target


def plot_visual_jepa_success(output_dir: Path) -> Path:
    summary = summarize_visual_jepa_episodes()
    trained = summary["trained"]
    baseline = summary["baselines"]
    methods = [
        ("state_model_cem", "Learned state CEM", "#0072B2"),
        ("image_reconstruction_cem", "Image reconstruction CEM", "#D55E00"),
        ("jepa_uniform_cem", "JEPA uniform-start CEM", "#009E73"),
        ("jepa_momentum_prior_cem", "JEPA momentum-prior CEM", "#785EF0"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharex=True)
    for y, (method, label, color) in enumerate(methods):
        values = trained[method]
        jitter = [((index - 2) * 0.055) for index in range(len(values))]
        rates = [row["rate"] for row in values]
        axes[0].scatter(
            rates,
            [y + offset for offset in jitter],
            color=color,
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            s=52,
            zorder=3,
        )
        axes[0].scatter([mean(rates)], [y], marker="D", color="#202938", edgecolor="white", s=78, zorder=4)
    axes[0].set_yticks(range(len(methods)), [entry[1] for entry in methods])
    axes[0].set_title("Trained methods", loc="left", fontweight="bold")

    baselines = [
        ("oracle_state_cem", "Exact-dynamics CEM", "#0072B2"),
        ("direct_visual_momentum", "Direct visual momentum", "#666666"),
    ]
    for y, (method, label, color) in enumerate(baselines):
        row = baseline[method]
        axes[1].errorbar(
            row["rate"],
            y,
            xerr=[[row["rate"] - row["wilson_low"]], [row["wilson_high"] - row["rate"]]],
            fmt="o",
            markersize=8,
            capsize=4,
            elinewidth=1.6,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        axes[1].text(row["rate"], y + (0.16 if row["rate"] < 0.5 else -0.16), f"{row['successes']}/{row['total']}", ha="center", va="center", fontsize=9)
    axes[1].set_yticks(range(len(baselines)), [entry[1] for entry in baselines])
    axes[1].set_title("Fixed-start baselines", loc="left", fontweight="bold")

    for ax in axes:
        ax.set_xlim(-0.04, 1.08)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Goal success rate")
        ax.grid(axis="x", color="#d8dee4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
    fig.suptitle("Policy guidance changed the local planning outcome", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.17, "Trained methods: each dot is one model seed; each seed uses the same 20 starts.", fontsize=8.5)
    fig.text(0.52, 0.17, "Baselines: one controller per method; 95% Wilson intervals over 20 starts.", fontsize=8.5)
    fig.text(
        0.02,
        0.055,
        "Synthetic MountainCar. The direct visual-momentum controller uses the same hand-written prior as JEPA momentum-prior CEM. The result does not isolate added value from JEPA.",
        fontsize=8.5,
        wrap=True,
    )
    fig.tight_layout(rect=(0.02, 0.25, 0.99, 0.9), w_pad=2.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "visual-jepa-success.png"
    fig.savefig(target, dpi=160, facecolor="white", metadata={"Software": "Matplotlib"})
    plt.close(fig)
    return target


def main() -> None:
    output_dir = ROOT / "docs/portfolio/figures"
    for path in (
        plot_decision_usefulness(output_dir),
        plot_uncertainty_calibration(output_dir),
        plot_visual_jepa_action_ranking(output_dir),
        plot_visual_jepa_success(output_dir),
    ):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
