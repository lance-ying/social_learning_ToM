#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import EXPERIMENT_LABELS, EXPERIMENTS, common_total_step_keys, make_output_path, preferred_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the mean total-steps bar plot with human bootstrap CI.")
    parser.add_argument("--output-file", help="Optional output path.")
    return parser.parse_args()


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_resamples: int = 10000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    bootstrap_means = np.mean(arr[sample_idx], axis=1)
    alpha = (100.0 - ci) / 2.0
    ci_low, ci_high = np.percentile(bootstrap_means, [alpha, 100.0 - alpha])
    return mean_val, float(ci_low), float(ci_high)


def main() -> int:
    args = parse_args()
    output_file = make_output_path("mean_total_steps_bar_plot_exp1234.png")
    if args.output_file:
        output_file = Path(args.output_file)

    matched_by_exp = {exp: common_total_step_keys(exp) for exp in EXPERIMENTS}
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)

    bar_series = [
        ("Human", "human"),
        ("Full\nModel", "full_model"),
        ("Mental\nOnly", "social_mentalizing"),
        ("Rational\nOnly", "rational_non_mentalizing"),
        ("Naive", "naive_observer"),
    ]

    x = np.arange(len(bar_series))

    color_by_model = {
        "human": "#888888",
        "full_model": "#4c78a8",
        "social_mentalizing": "#72b7b2",
        "rational_non_mentalizing": "#e39c37",
        "naive_observer": "#c95f5f",
    }
    fill_colors = [
        color_by_model[model_name if model_name != "human" else "human"]
        for _label, model_name in bar_series
    ]

    for ax, exp in zip(axes, EXPERIMENTS):
        model_predictions, human_stats = matched_by_exp[exp]
        levels = list(human_stats)

        means: list[float] = []

        human_values = [float(human_stats[level]["median"]) for level in levels]
        human_mean, human_ci_low, human_ci_high = bootstrap_mean_ci(human_values)
        means.append(human_mean)

        for _label, model_name in bar_series[1:]:
            model_dict = model_predictions.get(preferred_model_name(exp, model_name), {})
            values = [float(model_dict[level]["total_steps"]) for level in levels]
            means.append(float(np.mean(values)) if values else 0.0)

        ax.bar(
            x,
            means,
            width=0.82,
            color=fill_colors,
            edgecolor="#333333",
            linewidth=0.9,
        )
        ax.errorbar(
            [x[0]],
            [human_mean],
            yerr=[[human_mean - human_ci_low], [human_ci_high - human_mean]],
            fmt="none",
            ecolor="#111111",
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )

        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _model_name in bar_series], fontsize=16, rotation=0, ha="center")
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=16)

    axes[0].set_ylabel("Mean Total Steps", fontsize=20)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
