#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    annotate_stats,
    apply_reference_style,
    collect_total_steps_pairs,
    make_output_path,
    pooled_limits,
    resolve_model_panels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pooled model-vs-human total-step scatterplots for experiments 1-4."
    )
    parser.add_argument("--output-file", help="Optional output path.")
    parser.add_argument(
        "--models",
        help="Optional comma-separated model list. Defaults to the standard total-step panels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = make_output_path("total_steps_pooled_exp1234.png")
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)

    model_names = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else None
    total_step_panels = resolve_model_panels(model_names)
    if model_names is None:
        total_step_panels = list(total_step_panels)
        social_idx = next((i for i, (_label, model_name) in enumerate(total_step_panels) if model_name == "social_mentalizing"), None)
        insert_at = social_idx + 1 if social_idx is not None else len(total_step_panels)
        total_step_panels[insert_at:insert_at] = [
            ("Social Mentalizing\nUntil One Converges", "social_mentalizing_until_one_converges"),
            ("RNM Expert Only\nUntil Expert Wizard", "rational_non_mentalizing_expert_only_until_expert_wizard"),
            (
                "RNM Novice Full + Expert\nUntil Expert Wizard",
                "rational_non_mentalizing_novice_full_expert_until_expert_wizard",
            ),
            ("Naive Expert Only\nUntil Expert Wizard", "naive_observer_expert_only_until_expert_wizard"),
            (
                "Naive Novice Full + Expert\nUntil Expert Wizard",
                "naive_observer_novice_full_expert_until_expert_wizard",
            ),
        ]

    fig, axes = plt.subplots(1, len(total_step_panels), figsize=(5 * len(total_step_panels), 6), squeeze=False)
    axes = axes[0]

    panel_data = []
    for _label, model_name in total_step_panels:
        by_exp = []
        for exp in EXPERIMENTS:
            x, y, _sd, _keys = collect_total_steps_pairs(exp, model_name)
            by_exp.append((exp, x, y))
        panel_data.append(by_exp)

    lo, hi = pooled_limits(panel_data)

    for idx, (ax, (label, _model_name), by_exp) in enumerate(zip(axes, total_step_panels, panel_data)):
        apply_reference_style(ax)

        pooled_x = []
        pooled_y = []
        for exp, x, y in by_exp:
            if len(x) == 0:
                continue
            pooled_x.append(x)
            pooled_y.append(y)
            ax.scatter(
                x,
                y,
                s=36,
                alpha=0.85,
                color=EXPERIMENT_COLORS[exp],
                edgecolors="none",
                label=EXPERIMENT_LABELS[exp],
                zorder=3,
            )

        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#666666", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        if pooled_x:
            pooled_x_arr = np.concatenate(pooled_x)
            pooled_y_arr = np.concatenate(pooled_y)
            annotate_stats(ax, pooled_x_arr, pooled_y_arr)
            if len(pooled_x_arr) >= 2:
                slope, intercept = np.polyfit(pooled_x_arr, pooled_y_arr, 1)
                fit_x = np.array([lo, hi], dtype=float)
                fit_y = slope * fit_x + intercept
                ax.plot(fit_x, fit_y, color="#e15759", linewidth=1.8, zorder=2)

        ax.set_xlabel(label, fontsize=20, color="#1a1a1a")
        if idx == 0:
            ax.set_ylabel("Human Total Steps", fontsize=22, color="#1a1a1a")
        else:
            ax.set_ylabel("")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=16,
            markerscale=1.8,
            loc="lower center",
            ncol=len(EXPERIMENTS),
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout(w_pad=2.5, rect=(0, 0.08, 1, 1))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
