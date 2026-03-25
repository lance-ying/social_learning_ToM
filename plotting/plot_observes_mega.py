#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from common import (
    MODEL_PANELS,
    annotate_stats,
    apply_reference_style,
    collect_observe_pairs,
    make_output_path,
    plot_points_errorbars_and_fit,
)


ROW_CONFIGS = [
    ("exp1", "combined", "Experiment 1"),
    ("exp2", "combined", "Experiment 2"),
    ("exp3", "combined", "Experiment 3\ncombined"),
    ("exp4", "agent2", "Experiment 4\nagent 2"),
    ("exp4", "agent3", "Experiment 4\nagent 3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the observation-step mega plot across experiments 1-4."
    )
    parser.add_argument("--output-file", help="Optional output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = make_output_path("observes_mega_plot_exp1234.png")
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)

    fig, axes = plt.subplots(len(ROW_CONFIGS), len(MODEL_PANELS), figsize=(20, 24))

    for row_idx, (exp, observe_metric, row_label) in enumerate(ROW_CONFIGS):
        for col_idx, (ax, (panel_label, model_name)) in enumerate(zip(axes[row_idx], MODEL_PANELS)):
            x, y, sd, _keys = collect_observe_pairs(exp, model_name, observe_metric=observe_metric)
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
                annotate_stats(ax, x, y, fontsize=16)

            if row_idx == len(ROW_CONFIGS) - 1:
                ax.set_xlabel(f"{panel_label}\nModel Observation Steps", fontsize=20, color="#1a1a1a")
            else:
                ax.set_xlabel("")

            if col_idx == 0:
                ylabel = "Human Observation Steps"
                if observe_metric == "agent2":
                    ylabel = "Human Agent 2 Observes"
                elif observe_metric == "agent3":
                    ylabel = "Human Agent 3 Observes"
                ax.set_ylabel(f"{row_label}\n{ylabel}", fontsize=18, color="#1a1a1a")
            else:
                ax.set_ylabel("")

    plt.tight_layout(w_pad=2.5, h_pad=3.5)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
