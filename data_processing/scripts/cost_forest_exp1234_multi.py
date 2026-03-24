#!/usr/bin/env python3
"""
Simple forest plots for paired model-minus-human costs across experiments 1-4.

Each point shows the mean paired difference across matched levels
(`model - human`) for one model and experiment, with a bootstrap 95%
confidence interval and a horizontal zero reference line.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grouped_barplot_style import (
    EXPERIMENTS,
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    FIGSIZE_MULTI_PANEL,
    LEGEND_FONTSIZE,
    SERIES,
    SERIES_TICK_LABELS,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    YLABEL_FONTSIZE,
)


MODEL_SERIES = SERIES[1:]
MODEL_TICK_LABELS = SERIES_TICK_LABELS[1:]
OFFSET_STEP = 0.18
MARKER_SIZE = 7
CAPSIZE = 3
ELINEWIDTH = 1.8
BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_SEED = 20260320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create simple forest plots for reconstructed model and human costs."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["total_cost"],
        choices=["planning_cost", "total_cost", "observe_cost", "move_cost", "interaction_cost"],
        help="Metrics to plot.",
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


def exp1_human_candidates(_label: str, model_key: str) -> list[str]:
    normalized = normalize_exp1_model_key(model_key)
    return [f"mod_{normalized}_ascii"]


def default_human_candidates(_label: str, model_key: str) -> list[str]:
    return [model_key]


def human_candidates(exp: str, label: str, model_key: str) -> list[str]:
    if exp == "exp1":
        return exp1_human_candidates(label, model_key)
    return default_human_candidates(label, model_key)


def common_matched_keys(repo_root: Path, exp: str) -> tuple[dict[str, dict[str, float]], dict]:
    model_dir = repo_root / "model_outputs" / "reconstructed_costs"
    if not model_dir.exists():
        model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"
    human_per_case = load_json(human_costs_path(repo_root, exp))["per_case"]

    series_data = {}
    matched_human_keys = []

    for _series_label, key in MODEL_SERIES:
        model_per_case = load_json(model_dir / f"{exp}_{key}.json")["per_case"]
        aligned = {}
        for model_key, model_value in model_per_case.items():
            human_key = next(
                (candidate for candidate in human_candidates(exp, key, model_key) if candidate in human_per_case),
                None,
            )
            if human_key is not None:
                aligned[human_key] = model_value
        series_data[key] = aligned
        matched_human_keys.append(set(aligned))

    common_keys = set(human_per_case)
    for key_set in matched_human_keys:
        common_keys &= key_set

    common_keys = sorted(common_keys)
    filtered_series = {
        key: {human_key: value for human_key, value in aligned.items() if human_key in common_keys}
        for key, aligned in series_data.items()
    }
    return filtered_series, {key: human_per_case[key] for key in common_keys}


def paired_differences(model_per_case: dict, human_per_case: dict, metric: str) -> np.ndarray:
    human_mean_key = f"{metric}_mean"
    diffs = []
    for case_key in human_per_case:
        model_data = model_per_case.get(case_key)
        human_data = human_per_case.get(case_key)
        if model_data is None or human_data is None:
            continue
        if metric not in model_data or human_mean_key not in human_data:
            continue
        diffs.append(float(model_data[metric]) - float(human_data[human_mean_key]))
    return np.array(diffs, dtype=float)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(values))
    if values.size == 1:
        return mean_val, mean_val, mean_val

    samples = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
    means = np.mean(samples, axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return mean_val, float(ci_low), float(ci_high)


def pretty_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def apply_axis_style(ax: plt.Axes, metric: str) -> None:
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.1, zorder=0)
    ax.set_xticks(np.arange(len(MODEL_SERIES)))
    ax.set_xticklabels(MODEL_TICK_LABELS, fontsize=TICK_FONTSIZE)
    ax.set_ylabel(f"Model - Human {pretty_metric(metric)}", fontsize=YLABEL_FONTSIZE)
    ax.set_title(pretty_metric(metric), fontsize=TITLE_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def plot_metric(ax: plt.Axes, matched_by_experiment: dict[str, tuple[dict[str, dict[str, float]], dict]], metric: str) -> None:
    x = np.arange(len(MODEL_SERIES))
    offsets = (np.arange(len(EXPERIMENTS)) - (len(EXPERIMENTS) - 1) / 2) * OFFSET_STEP
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for exp_idx, exp in enumerate(EXPERIMENTS):
        color = EXPERIMENT_COLORS[exp]
        exp_label = EXPERIMENT_LABELS[exp_idx]
        series_data, human_common = matched_by_experiment[exp]

        xs = []
        means = []
        lower_err = []
        upper_err = []

        for model_idx, (_series_label, series_key) in enumerate(MODEL_SERIES):
            diffs = paired_differences(series_data[series_key], human_common, metric)
            mean_val, ci_low, ci_high = bootstrap_mean_ci(diffs, rng)
            xs.append(x[model_idx] + offsets[exp_idx])
            means.append(mean_val)
            lower_err.append(mean_val - ci_low)
            upper_err.append(ci_high - mean_val)

        ax.errorbar(
            xs,
            means,
            yerr=np.vstack([lower_err, upper_err]),
            fmt="o",
            markersize=MARKER_SIZE,
            color=color,
            ecolor=color,
            elinewidth=ELINEWIDTH,
            capsize=CAPSIZE,
            linewidth=0,
            label=exp_label,
            zorder=3,
        )

    apply_axis_style(ax, metric)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root / "data_processing" / "outputs" / "plots" / "cost_forest_exp1234_multi.png"
    )

    matched_by_experiment = {exp: common_matched_keys(repo_root, exp) for exp in EXPERIMENTS}

    metrics = args.metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(FIGSIZE_MULTI_PANEL[0] * len(metrics), FIGSIZE_MULTI_PANEL[1]))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        plot_metric(ax, matched_by_experiment, metric)

    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE, title="Experiment")
    plt.tight_layout(w_pad=2.5)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
