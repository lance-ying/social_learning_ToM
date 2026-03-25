#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_COLORS,
    EXPERIMENTS,
    MODEL_PANELS,
    TOTAL_STEPS_BAR_SERIES,
    TOTAL_STEPS_BAR_TICKS,
    common_total_step_keys,
    make_output_path,
)


BAR_WIDTH = 0.15


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

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(TOTAL_STEPS_BAR_SERIES))

    means_by_series = []
    sds_by_series = []

    matched_by_exp = {exp: common_total_step_keys(exp) for exp in EXPERIMENTS}

    for _series_label, series_key in TOTAL_STEPS_BAR_SERIES:
        series_means = []
        series_sds = []
        for exp in EXPERIMENTS:
            model_predictions, human_stats = matched_by_exp[exp]
            if series_key == "human":
                values = [human_stats[level]["mean"] for level in human_stats]
            else:
                values = [model_predictions[series_key][level]["total_steps"] for level in model_predictions[series_key]]
            series_means.append(float(np.mean(values)) if values else 0.0)
            series_sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        means_by_series.append(series_means)
        sds_by_series.append(series_sds)

    for exp_idx, exp in enumerate(EXPERIMENTS):
        offset = (exp_idx - (len(EXPERIMENTS) - 1) / 2) * BAR_WIDTH
        means = [series_means[exp_idx] for series_means in means_by_series]
        sds = [series_sds[exp_idx] for series_sds in sds_by_series]
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            label=exp.replace("exp", "Experiment "),
            color=EXPERIMENT_COLORS[exp],
            alpha=0.9,
            edgecolor="none",
            yerr=sds,
            capsize=4,
            ecolor="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TOTAL_STEPS_BAR_TICKS, fontsize=12)
    ax.set_ylabel("Mean Total Steps", fontsize=14)
    ax.set_title("Mean Total Steps Across Experiments", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
