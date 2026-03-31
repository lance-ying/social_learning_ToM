#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import EXPERIMENT_LABELS, EXPERIMENTS, common_total_step_keys, make_output_path, preferred_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the mean total-steps bar plot with SE.")
    parser.add_argument("--output-file", help="Optional output path.")
    parser.add_argument(
        "--omit-variations",
        action="store_true",
        help="Omit expert-only / novice-full variant models from the bar plot.",
    )
    return parser.parse_args()


def standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def main() -> int:
    args = parse_args()
    output_file = make_output_path("mean_total_steps_bar_plot_exp1234.png")
    if args.output_file:
        output_file = Path(args.output_file)

    matched_by_exp = {exp: common_total_step_keys(exp) for exp in EXPERIMENTS}
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), sharey=True)

    core_series = [
        ("Human", "human"),
        ("Rat. Ment.", "full_model"),
        ("Soc. Ment.", "social_mentalizing"),
        ("Rat. Non-M.", "rational_non_mentalizing"),
        ("Naive", "naive_observer"),
    ]
    variation_series = [
        ("RNM Expert\nOnly", "rational_non_mentalizing_expert_only_until_expert_wizard"),
        ("RNM Novice\nFull + Exp.", "rational_non_mentalizing_novice_full_expert_until_expert_wizard"),
        ("Naive Expert\nOnly", "naive_observer_expert_only_until_expert_wizard"),
        ("Naive Novice\nFull + Exp.", "naive_observer_novice_full_expert_until_expert_wizard"),
    ]
    bar_series = list(core_series)
    if not args.omit_variations:
        bar_series = (
            core_series[:4]
            + variation_series[:2]
            + [core_series[4]]
            + variation_series[2:]
            + [("Non-Obs. Plan.", "agent1_naive_planner")]
        )

    x = np.arange(len(bar_series))

    color_by_model = {
        "human": "#888888",
        "full_model": "#4c78a8",
        "social_mentalizing": "#72b7b2",
        "rational_non_mentalizing": "#e39c37",
        "rational_non_mentalizing_expert_only_until_expert_wizard": "#f0b870",
        "rational_non_mentalizing_novice_full_expert_until_expert_wizard": "#d8891e",
        "naive_observer": "#c95f5f",
        "naive_observer_expert_only_until_expert_wizard": "#e6a5a5",
        "naive_observer_novice_full_expert_until_expert_wizard": "#b64747",
        "agent1_naive_planner": "#6aa84f",
    }
    fill_colors = [
        color_by_model[model_name if model_name != "human" else "human"]
        for _label, model_name in bar_series
    ]

    for ax, exp in zip(axes, EXPERIMENTS):
        model_predictions, human_stats = matched_by_exp[exp]
        levels = list(human_stats)

        means: list[float] = []
        ses: list[float] = []

        human_values = [float(human_stats[level]["median"]) for level in levels]
        means.append(float(np.mean(human_values)) if human_values else 0.0)
        ses.append(standard_error(human_values))

        for _label, model_name in bar_series[1:]:
            model_dict = model_predictions.get(preferred_model_name(exp, model_name), {})
            values = [float(model_dict[level]["total_steps"]) for level in levels]
            means.append(float(np.mean(values)) if values else 0.0)
            ses.append(standard_error(values))

        ax.bar(
            x,
            means,
            width=0.82,
            color=fill_colors,
            edgecolor="#333333",
            linewidth=0.9,
        )
        ax.errorbar(
            x,
            means,
            yerr=ses,
            fmt="none",
            ecolor="#111111",
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )

        ax.set_title(EXPERIMENT_LABELS[exp], fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _model_name in bar_series], fontsize=16, rotation=25, ha="right")
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
