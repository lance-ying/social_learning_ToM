#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    MODEL_PANELS,
    annotate_stats,
    apply_reference_style,
    collect_total_steps_pairs,
    make_output_path,
    pooled_limits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pooled model-vs-human total-step scatterplots for experiments 1-4."
    )
    parser.add_argument("--output-file", help="Optional output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file = make_output_path("total_steps_pooled_exp1234.png")
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle("Total Steps Pooled Across Experiments", fontsize=28, color="#1a1a1a", y=0.99)

    panel_data = []
    for _label, model_name in MODEL_PANELS:
        by_exp = []
        for exp in EXPERIMENTS:
            x, y, _sd, _keys = collect_total_steps_pairs(exp, model_name)
            by_exp.append((exp, x, y))
        panel_data.append(by_exp)

    lo, hi = pooled_limits(panel_data)

    for idx, (ax, (label, _model_name), by_exp) in enumerate(zip(axes, MODEL_PANELS, panel_data)):
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
            annotate_stats(ax, np.concatenate(pooled_x), np.concatenate(pooled_y))

        ax.set_xlabel(f"{label}\nModel Total Steps", fontsize=20, color="#1a1a1a")
        if idx == 0:
            ax.set_ylabel("Human Total Steps", fontsize=22, color="#1a1a1a")
        else:
            ax.set_ylabel("")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, frameon=False, fontsize=11, loc="lower right")

    plt.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.96))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
