#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from common import (
    annotate_stats,
    apply_reference_style,
    collect_total_steps_pairs,
    make_output_path,
    plot_points_errorbars_and_fit,
    resolve_model_panels,
)


ROW_CONFIGS = [
    ("exp1", "Experiment 1"),
    ("exp2", "Experiment 2"),
    ("exp3", "Experiment 3"),
    ("exp4", "Experiment 4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the total-steps mega plot across experiments 1-4."
    )
    parser.add_argument("--output-file", help="Optional output path.")
    parser.add_argument(
        "--models",
        help="Optional comma-separated model list. Defaults to the standard total-step panels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = make_output_path("total_steps_mega_plot_exp1234.png")
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)

    model_names = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else None
    model_panels = resolve_model_panels(model_names)

    fig, axes = plt.subplots(len(ROW_CONFIGS), len(model_panels), figsize=(5 * len(model_panels), 19), squeeze=False)

    for row_idx, (exp, row_label) in enumerate(ROW_CONFIGS):
        for col_idx, (ax, (panel_label, model_name)) in enumerate(zip(axes[row_idx], model_panels)):
            x, y, sd, _keys = collect_total_steps_pairs(exp, model_name)
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
                ax.set_xlabel(f"{panel_label}\nModel Total Steps", fontsize=20, color="#1a1a1a")
            else:
                ax.set_xlabel("")

            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nHuman Total Steps", fontsize=18, color="#1a1a1a")
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
