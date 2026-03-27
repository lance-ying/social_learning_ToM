#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from common import (
    apply_reference_style,
    bootstrap_ccc_ci,
    bootstrap_r_ci,
    collect_total_steps_pairs,
    make_output_path,
    plot_points_errorbars_and_fit,
    resolve_model_panels,
)


ROW_CONFIGS = [
    {"exp": "exp1", "group_label": "Experiment 1\nHuman Total Steps", "row_label": ""},
    {"exp": "exp2", "group_label": "Experiment 2\nHuman Total Steps", "row_label": ""},
    {"exp": "exp3", "group_label": "Experiment 3\nHuman Total Steps", "row_label": ""},
    {"exp": "exp4", "group_label": "Experiment 4\nHuman Total Steps", "row_label": ""},
]


def annotate_total_stats(ax, x, y, fontsize: int = 16) -> None:
    if len(x) < 3:
        return
    r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
    ccc, _ccc_low, _ccc_high = bootstrap_ccc_ci(x, y, n_resamples=1000)
    ax.text(
        0.05,
        0.90,
        f"r = {r:.2f} [{ci_low:.2f}, {ci_high:.2f}]\nCCC = {ccc:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#1a1a1a",
    )


def add_group_labels(fig: plt.Figure, axes) -> None:
    for row_idx, config in enumerate(ROW_CONFIGS):
        top = axes[row_idx, 0].get_position().y1
        bottom = axes[row_idx, 0].get_position().y0
        y = (top + bottom) / 2
        fig.text(
            0.048,
            y,
            config["group_label"],
            rotation=90,
            va="center",
            ha="center",
            fontsize=18,
            color="#1a1a1a",
        )


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
    if model_names is None:
        model_panels = [panel for panel in model_panels if panel[1] != "agent1_naive_planner"]

    fig, axes = plt.subplots(
        len(ROW_CONFIGS),
        len(model_panels),
        figsize=(5 * len(model_panels), 4.6 * len(ROW_CONFIGS)),
        squeeze=False,
    )

    for row_idx, config in enumerate(ROW_CONFIGS):
        exp = config["exp"]
        row_label = config["row_label"]
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
                annotate_total_stats(ax, x, y, fontsize=16)

            if row_idx == len(ROW_CONFIGS) - 1:
                ax.set_xlabel(panel_label, fontsize=20, color="#1a1a1a")
            else:
                ax.set_xlabel("")

            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=16, color="#1a1a1a", labelpad=2)
            else:
                ax.set_ylabel("")

    plt.tight_layout(rect=(0.082, 0.03, 1.0, 1.0), w_pad=2.5, h_pad=1.6)
    add_group_labels(fig, axes)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
