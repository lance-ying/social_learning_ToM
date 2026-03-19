#!/usr/bin/env python3
"""
Mega scatterplot for model-vs-human cost by level across experiments 1-4.

Rows:
1. Experiment 1
2. Experiment 2
3. Experiment 3
4. Experiment 4

Columns:
1. Rational Mentalizing (Full Model)
2. Social Mentalizing
3. Rational Non-Mentalizing
4. Naive Observer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from correlation_4panel_style import (
    annotate_r_ci,
    apply_reference_style,
    bootstrap_r_ci,
    plot_points_errorbars_and_fit,
)


LABELS = [
    "Rational Mentalizing\n(Full Model)",
    "Social Mentalizing",
    "Rational Non-Mentalizing",
    "Naive Observer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-level model-vs-human cost scatterplots for experiments 1-4."
    )
    parser.add_argument(
        "--metric",
        choices=["planning_cost", "total_cost", "observe_cost", "move_cost", "interaction_cost"],
        default="total_cost",
        help="Cost metric to plot.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional output path. Defaults to data_processing/outputs/plots/cost_scatter_<metric>_exp1234_mega.png",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_exp1_model_key(model_key: str) -> str:
    if model_key.startswith("mod_") and model_key.endswith("_ascii"):
        return model_key.removeprefix("mod_").removesuffix("_ascii")
    return model_key.split("_")[0]


def exp1_full_model_human_keys(model_key: str) -> list[str]:
    normalized = normalize_exp1_model_key(model_key)
    return [f"mod_{normalized}_ascii"]


def exp1_baseline_human_keys(model_key: str) -> list[str]:
    normalized = normalize_exp1_model_key(model_key)
    return [f"mod_{normalized}_ascii"]


def default_human_keys(model_key: str) -> list[str]:
    return [model_key]


def collect_pairs(
    model_per_case: dict,
    human_per_case: dict,
    metric: str,
    human_key_candidates,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model_vals = []
    human_means = []
    human_sds = []
    matched_keys = []

    human_mean_key = f"{metric}_mean"
    human_sd_key = f"{metric}_sd"

    for model_key, model_data in model_per_case.items():
        human_key = next((k for k in human_key_candidates(model_key) if k in human_per_case), None)
        if human_key is None:
            continue

        if metric not in model_data:
            continue

        human_data = human_per_case[human_key]
        if human_mean_key not in human_data:
            continue

        model_vals.append(float(model_data[metric]))
        human_means.append(float(human_data[human_mean_key]))
        human_sds.append(float(human_data.get(human_sd_key, 0.0)))
        matched_keys.append(human_key)

    return (
        np.array(model_vals),
        np.array(human_means),
        np.array(human_sds),
        matched_keys,
    )


def build_row_pairs(repo_root: Path, exp: str, metric: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    model_dir = repo_root / "scripts" / "experiments" / "experiment_outputs" / "reconstructed_costs_mega_plot"
    human_path = repo_root / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json"
    human_per_case = load_json(human_path)["per_case"]

    model_paths = {
        LABELS[0]: model_dir / f"{exp}_full_model.json",
        LABELS[1]: model_dir / f"{exp}_social_mentalizing.json",
        LABELS[2]: model_dir / f"{exp}_rational_non_mentalizing.json",
        LABELS[3]: model_dir / f"{exp}_naive_observer.json",
    }

    pairs = {}
    for label, model_path in model_paths.items():
        model_per_case = load_json(model_path)["per_case"]
        if exp == "exp1" and label == LABELS[0]:
            key_mapper = exp1_full_model_human_keys
        elif exp == "exp1":
            key_mapper = exp1_baseline_human_keys
        else:
            key_mapper = default_human_keys

        pairs[label] = collect_pairs(model_per_case, human_per_case, metric, key_mapper)

    return pairs


def plot_metric(metric: str, output_file: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    row_pairs = [
        ("Experiment 1", build_row_pairs(repo_root, "exp1", metric)),
        ("Experiment 2", build_row_pairs(repo_root, "exp2", metric)),
        ("Experiment 3", build_row_pairs(repo_root, "exp3", metric)),
        ("Experiment 4", build_row_pairs(repo_root, "exp4", metric)),
    ]

    pretty_metric = metric.replace("_", " ")
    fig, axes = plt.subplots(4, 4, figsize=(20, 19))

    for row_idx, (row_label, pairs) in enumerate(row_pairs):
        for col_idx, (ax, label) in enumerate(zip(axes[row_idx], LABELS)):
            x, y, sd, _keys = pairs[label]
            apply_reference_style(ax)

            if len(x) < 3:
                ax.text(
                    0.05,
                    0.90,
                    "Insufficient data",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=18,
                    color="#1a1a1a",
                )
            else:
                plot_points_errorbars_and_fit(ax, x, y, sd)
                r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
                annotate_r_ci(ax, r, ci_low, ci_high)

            if row_idx == len(row_pairs) - 1:
                ax.set_xlabel(f"{label}\nModel {pretty_metric}", fontsize=24, color="#1a1a1a")
            else:
                ax.set_xlabel("")

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nHuman {pretty_metric}", fontsize=22, color="#1a1a1a")
            else:
                ax.set_ylabel("")

    plt.tight_layout(w_pad=2.5, h_pad=3.5)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_file = (
        Path(args.output_file)
        if args.output_file
        else repo_root
        / "data_processing"
        / "outputs"
        / "plots"
        / f"cost_scatter_{args.metric}_exp1234_mega.png"
    )
    plot_metric(args.metric, output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
