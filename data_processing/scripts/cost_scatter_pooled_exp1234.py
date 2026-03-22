#!/usr/bin/env python3
"""
Pooled scatterplot for model-vs-human cost across experiments 1-4.

Defaults to the full model on observe cost, with points colored by experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from correlation_4panel_style import apply_reference_style, bootstrap_r_ci
from grouped_barplot_style import EXPERIMENTS, EXPERIMENT_COLORS, EXPERIMENT_LABELS, LEGEND_FONTSIZE


MODEL_LABELS = {
    "full_model": "Rational Mentalizing (Full Model)",
    "social_mentalizing": "Social Mentalizing",
    "rational_non_mentalizing": "Rational Non-Mentalizing",
    "naive_observer": "Naive Observer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pooled model-vs-human cost scatterplot across experiments 1-4."
    )
    parser.add_argument(
        "--metric",
        choices=["planning_cost", "total_cost", "observe_cost", "move_cost", "interaction_cost"],
        default="observe_cost",
        help="Cost metric to plot.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_LABELS),
        default="full_model",
        help="Model series to plot.",
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
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))
    ax.text(
        0.04,
        0.96,
        f"r = {r:.2f}\nCI = [{ci_low:.2f}, {ci_high:.2f}]\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="#1a1a1a",
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root / "data_processing" / "outputs" / "plots" / f"cost_scatter_pooled_{args.model}_{args.metric}.png"
    )

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    apply_reference_style(ax)

    all_x = []
    all_y = []

    for exp, exp_label in zip(EXPERIMENTS, EXPERIMENT_LABELS):
        x, y = collect_pairs(repo_root, exp, args.model, args.metric)
        if len(x) == 0:
            continue
        all_x.append(x)
        all_y.append(y)
        ax.scatter(
            x,
            y,
            s=40,
            alpha=0.85,
            color=EXPERIMENT_COLORS[exp],
            edgecolors="none",
            label=exp_label,
            zorder=3,
        )

    if not all_x:
        raise SystemExit("No matched data found for the requested model/metric.")

    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)

    data_min = float(min(np.min(x_all), np.min(y_all)))
    data_max = float(max(np.max(x_all), np.max(y_all)))
    pad = 0.05 * (data_max - data_min) if data_max > data_min else 1.0
    lo = data_min - pad
    hi = data_max + pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#666666", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    pretty = pretty_metric(args.metric)
    ax.set_xlabel(f"{MODEL_LABELS[args.model]} {pretty}", fontsize=16, color="#1a1a1a")
    ax.set_ylabel(f"Human {pretty}", fontsize=16, color="#1a1a1a")
    ax.set_title(f"Pooled Across Experiments: {pretty}", fontsize=17, color="#1a1a1a")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower right")
    annotate_stats(ax, x_all, y_all)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
