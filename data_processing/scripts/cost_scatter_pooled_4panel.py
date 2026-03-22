#!/usr/bin/env python3
"""
4-panel pooled scatterplot for a selected cost metric across experiments 1-4.

Each panel shows one model, with all matched levels pooled across experiments
and points colored by experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from correlation_4panel_style import apply_reference_style, bootstrap_r_ci
from grouped_barplot_style import EXPERIMENTS, EXPERIMENT_COLORS, EXPERIMENT_LABELS


MODEL_PANELS = [
    ("Rational Mentalizing\n(Full Model)", "full_model"),
    ("Social Mentalizing", "social_mentalizing"),
    ("Rational Non-Mentalizing", "rational_non_mentalizing"),
    ("Naive Observer", "naive_observer"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 4-panel pooled scatterplot for a cost metric across experiments 1-4."
    )
    parser.add_argument(
        "--metric",
        choices=["planning_cost", "total_cost", "observe_cost", "move_cost", "interaction_cost"],
        default="observe_cost",
        help="Cost metric to plot.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional output path.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_costs_path(repo_root: Path, exp: str) -> Path:
    if exp == "exp4":
        override = os.environ.get("EXP4_HUMAN_COSTS_FILE", "").strip()
        if override:
            return Path(override)
    return repo_root / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json"


def normalize_exp1_model_key(model_key: str) -> str:
    if model_key.startswith("mod_") and model_key.endswith("_ascii"):
        return model_key.removeprefix("mod_").removesuffix("_ascii")
    return model_key.split("_")[0]


def exp1_human_candidates(model_key: str) -> list[str]:
    normalized = normalize_exp1_model_key(model_key)
    return [f"mod_{normalized}_ascii"]


def default_human_candidates(model_key: str) -> list[str]:
    return [model_key]


def human_candidates(exp: str, model_key: str) -> list[str]:
    if exp == "exp1":
        return exp1_human_candidates(model_key)
    return default_human_candidates(model_key)


def collect_pairs(repo_root: Path, exp: str, model_name: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs"
    if not model_dir.exists():
        model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"

    model_per_case = load_json(model_dir / f"{exp}_{model_name}.json")["per_case"]
    human_per_case = load_json(human_costs_path(repo_root, exp))["per_case"]
    human_mean_key = f"{metric}_mean"

    model_vals = []
    human_vals = []

    for model_key, model_data in model_per_case.items():
        human_key = next((k for k in human_candidates(exp, model_key) if k in human_per_case), None)
        if human_key is None:
            continue
        if metric not in model_data or human_mean_key not in human_per_case[human_key]:
            continue
        model_vals.append(float(model_data[metric]))
        human_vals.append(float(human_per_case[human_key][human_mean_key]))

    return np.array(model_vals, dtype=float), np.array(human_vals, dtype=float)


def pretty_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def annotate_stats(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 3:
        return
    r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
    ax.text(
        0.05,
        0.90,
        f"r = {r:.2f}\nCI = [{ci_low:.2f}, {ci_high:.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        color="#1a1a1a",
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root / "data_processing" / "outputs" / "plots" / f"cost_scatter_pooled_{args.metric}_4panel.png"
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(f"{pretty_metric(args.metric)} Pooled Across Experiments", fontsize=30, color="#1a1a1a", y=0.99)

    panel_data = []
    all_values = []

    for _label, model_name in MODEL_PANELS:
        by_exp = []
        for exp in EXPERIMENTS:
            x, y = collect_pairs(repo_root, exp, model_name, args.metric)
            by_exp.append((exp, x, y))
            if len(x):
                all_values.extend(x.tolist())
                all_values.extend(y.tolist())
        panel_data.append(by_exp)

    if not all_values:
        raise SystemExit(f"No pooled {args.metric} data found.")

    data_min = float(min(all_values))
    data_max = float(max(all_values))
    pad = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    lo = data_min - pad
    hi = data_max + pad

    for idx, (ax, (label, _model_name), by_exp) in enumerate(zip(axes, MODEL_PANELS, panel_data)):
        apply_reference_style(ax)

        pooled_x = []
        pooled_y = []
        for exp in EXPERIMENTS:
            exp_key, x, y = by_exp[EXPERIMENTS.index(exp)]
            if len(x) == 0:
                continue
            pooled_x.append(x)
            pooled_y.append(y)
            ax.scatter(
                x,
                y,
                s=36,
                alpha=0.85,
                color=EXPERIMENT_COLORS[exp_key],
                edgecolors="none",
                label=EXPERIMENT_LABELS[EXPERIMENTS.index(exp_key)],
                zorder=3,
            )

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#666666", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        if pooled_x:
            annotate_stats(ax, np.concatenate(pooled_x), np.concatenate(pooled_y))

        ax.set_xlabel(label, fontsize=22, color="#1a1a1a")
        if idx == 0:
            ax.set_ylabel(f"Human {pretty_metric(args.metric)}", fontsize=24, color="#1a1a1a")
        else:
            ax.set_ylabel("")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, frameon=False, fontsize=11, loc="lower right")

    plt.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.97))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
