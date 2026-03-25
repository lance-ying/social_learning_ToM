#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    MODEL_PANELS,
    common_total_step_keys,
    make_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the mean total-steps grouped bar plot.")
    parser.add_argument("--output-file", help="Optional output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = make_output_path("mean_total_steps_bar_plot_exp1234.png")
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)

    matched_by_exp = {exp: common_total_step_keys(exp) for exp in EXPERIMENTS}
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    x = np.arange(len(MODEL_PANELS) + 1)

    tick_labels = [
        "Human",
        "Rational\nMentalizing",
        "Social\nMentalizing",
        "Rational Non-\nMentalizing",
        "Naive\nPlanner",
        "Naive\nObserver",
    ]
    bar_colors = ["#888888", "#4c78a8", "#72b7b2", "#e39c37", "#6aa84f", "#c95f5f"]

    for ax, exp in zip(axes, EXPERIMENTS):
        model_predictions, human_stats = matched_by_exp[exp]
        human_values = [human_stats[level]["mean"] for level in human_stats]
        human_mean = float(np.mean(human_values)) if human_values else 0.0
        human_sd = float(np.std(human_values, ddof=1)) if len(human_values) > 1 else 0.0

        means = [human_mean]
        sds = [human_sd]
        for _label, model_name in MODEL_PANELS:
            values = [model_predictions[model_name][level]["total_steps"] for level in model_predictions[model_name]]
            means.append(float(np.mean(values)) if values else 0.0)
            sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

        ax.bar(
            x,
            means,
            width=0.72,
            color=bar_colors,
            alpha=0.9,
            edgecolor="none",
            yerr=sds,
            capsize=4,
            ecolor="#333333",
        )
        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=11)

    axes[0].set_ylabel("Mean Total Steps", fontsize=14)

    fig.suptitle("Mean Total Steps by Experiment", fontsize=20, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
