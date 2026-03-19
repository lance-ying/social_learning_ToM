#!/usr/bin/env python3
"""
Grouped cost barplots across experiments 1-4.

Model categories are on the x-axis, with experiments grouped within each
category. Default output includes one panel for `total_cost`.
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
    SERIES,
    FIGSIZE_MULTI_PANEL,
    LEGEND_FONTSIZE,
    apply_axis_style,
    plot_grouped_bars,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create grouped barplots for reconstructed model and human costs."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["total_cost"],
        choices=["planning_cost", "total_cost", "observe_cost"],
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


def exp1_human_candidates(label: str, model_key: str) -> list[str]:
    normalized = normalize_exp1_model_key(model_key)
    return [f"mod_{normalized}_ascii"]


def default_human_candidates(_label: str, model_key: str) -> list[str]:
    return [model_key]


def human_candidates(exp: str, label: str, model_key: str) -> list[str]:
    if exp == "exp1":
        return exp1_human_candidates(label, model_key)
    return default_human_candidates(label, model_key)


def common_matched_keys(repo_root: Path, exp: str) -> tuple[dict[str, dict[str, float]], dict]:
    model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs"
    if not model_dir.exists():
        model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"
    human_path = human_costs_path(repo_root, exp)
    human_per_case = load_json(human_path)["per_case"]

    series_data = {}
    matched_human_keys = []

    for _series_label, key in SERIES[1:]:
        model_path = model_dir / f"{exp}_{key}.json"
        model_per_case = load_json(model_path)["per_case"]
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


def mean_sd_from_model_per_case(per_case: dict, metric: str) -> tuple[float, float]:
    values = [float(v[metric]) for v in per_case.values() if metric in v]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def mean_sd_from_human_per_case(per_case: dict, metric: str) -> tuple[float, float]:
    mean_key = f"{metric}_mean"
    values = [float(v[mean_key]) for v in per_case.values() if mean_key in v]
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def pretty_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root / "data_processing" / "outputs" / "plots" / "cost_barplot_exp1234_multi.png"
    )
    matched_by_experiment = {exp: common_matched_keys(repo_root, exp) for exp in EXPERIMENTS}

    metrics = args.metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(FIGSIZE_MULTI_PANEL[0] * len(metrics), FIGSIZE_MULTI_PANEL[1]))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        means_by_series = []
        sds_by_series = []
        for _series_label, series_key in SERIES:
            means = []
            sds = []
            for exp in EXPERIMENTS:
                series_data, human_common = matched_by_experiment[exp]
                if series_key == "human":
                    mean_val, sd_val = mean_sd_from_human_per_case(human_common, metric)
                else:
                    mean_val, sd_val = mean_sd_from_model_per_case(series_data[series_key], metric)
                means.append(mean_val)
                sds.append(sd_val)
            means_by_series.append(means)
            sds_by_series.append(sds)

        plot_grouped_bars(ax, means_by_series, sds_by_series)
        apply_axis_style(ax, f"Mean {pretty_metric(metric)}", pretty_metric(metric))

    axes[0].legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout(w_pad=2.5)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
