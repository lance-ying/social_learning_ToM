#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import EXPERIMENT_LABELS, EXPERIMENTS, common_total_step_keys, make_output_path, preferred_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the mean total-steps violin plot.")
    parser.add_argument("--output-file", help="Optional output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = make_output_path("mean_total_steps_violin_plot_exp1234.png")
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)

    matched_by_exp = {exp: common_total_step_keys(exp) for exp in EXPERIMENTS}
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)

    violin_series = [
        ("Human", "human"),
        ("Rat. Ment.", "full_model"),
        ("Soc. Ment.", "social_mentalizing"),
        ("Rat. Non-M.", "rational_non_mentalizing"),
        ("RNM Expert\nOnly", "rational_non_mentalizing_expert_only_until_expert_wizard"),
        ("RNM Novice\nFull + Exp.", "rational_non_mentalizing_novice_full_expert_until_expert_wizard"),
        ("Naive", "naive_observer"),
        ("Naive Expert\nOnly", "naive_observer_expert_only_until_expert_wizard"),
        ("Naive Novice\nFull + Exp.", "naive_observer_novice_full_expert_until_expert_wizard"),
        ("Non-Obs. Plan.", "agent1_naive_planner"),
    ]
    x = np.arange(len(violin_series))

    fill_colors = [
        "#888888",
        "#4c78a8",
        "#72b7b2",
        "#e39c37",
        "#f0b870",
        "#d8891e",
        "#c95f5f",
        "#e6a5a5",
        "#b64747",
        "#6aa84f",
    ]

    for ax, exp in zip(axes, EXPERIMENTS):
        model_predictions, human_stats = matched_by_exp[exp]
        human_values = [human_stats[level]["median"] for level in human_stats]

        series_values = [human_values]
        for _label, model_name in violin_series[1:]:
            model_dict = model_predictions.get(preferred_model_name(exp, model_name), {})
            values = [model_dict[level]["total_steps"] for level in model_dict]
            if not values:
                values = [np.nan, np.nan]
            series_values.append(values)

        violin = ax.violinplot(
            series_values,
            positions=x,
            widths=0.82,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(violin["bodies"], fill_colors):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.8)

        violin["cmedians"].set_color("#222222")
        violin["cmedians"].set_linewidth(2.2)

        means = [float(np.mean(values)) if values else 0.0 for values in series_values]
        ax.scatter(x, means, color="#111111", s=30, zorder=3)

        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _model_name in violin_series], fontsize=16, rotation=25, ha="right")
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
