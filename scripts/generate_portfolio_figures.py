"""Build portfolio figures from retained Notebook 14, 16, and 17 evidence."""

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


def main() -> None:
    output_dir = ROOT / "docs/portfolio/figures"
    for path in (plot_decision_usefulness(output_dir), plot_uncertainty_calibration(output_dir)):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
