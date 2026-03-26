#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    annotate_stats,
    apply_reference_style,
    collect_observe_pairs,
    collect_total_cost_pairs,
    collect_total_steps_pairs,
    make_output_path,
    plot_points_errorbars_and_fit,
)


EXP4_MODELS = [
    ("RNM Baseline", "rational_non_mentalizing"),
    ("RNM Expert Only", "rational_non_mentalizing_expert_only_until_expert_wizard"),
    (
        "RNM Novice Full + Expert",
        "rational_non_mentalizing_novice_full_expert_until_expert_wizard",
    ),
    ("Naive Baseline", "naive_observer"),
    ("Naive Expert Only", "naive_observer_expert_only_until_expert_wizard"),
    (
        "Naive Novice Full + Expert",
        "naive_observer_novice_full_expert_until_expert_wizard",
    ),
]

EXP3_MODELS = [
    ("Social Mentalizing", "social_mentalizing"),
    ("Social Mentalizing\nUntil One Converges", "social_mentalizing_until_one_converges"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create focused scatterplots for Exp4 baseline variants and Exp3 social-mentalizing U1C."
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to plotting/outputs/requested_baseline_scatters.",
    )
    return parser.parse_args()


def metric_spec(metric: str):
    if metric == "observe":
        return collect_observe_pairs, "Human Observation Steps", "Model Observation Steps", "exp4_variant_observes.png", "exp3_social_mentalizing_u1c_observes.png"
    if metric == "total_steps":
        return collect_total_steps_pairs, "Human Total Steps", "Model Total Steps", "exp4_variant_total_steps.png", "exp3_social_mentalizing_u1c_total_steps.png"
    if metric == "total_cost":
        return collect_total_cost_pairs, "Human Total Cost", "Model Total Cost", "exp4_variant_total_cost.png", "exp3_social_mentalizing_u1c_total_cost.png"
    raise ValueError(metric)


def plot_single_experiment(exp: str, panels: list[tuple[str, str]], metric: str, output_path: Path) -> None:
    collector, y_label, x_suffix, _exp4_name, _exp3_name = metric_spec(metric)
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6), squeeze=False)
    axes = axes[0]

    for idx, (ax, (panel_label, model_name)) in enumerate(zip(axes, panels)):
        if metric == "observe":
            x, y, sd, _keys = collector(exp, model_name, observe_metric="combined")
        else:
            x, y, sd, _keys = collector(exp, model_name)

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

        ax.set_xlabel(f"{panel_label}\n{x_suffix}", fontsize=20, color="#1a1a1a")
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=22, color="#1a1a1a")
        else:
            ax.set_ylabel("")

    plt.tight_layout(w_pad=2.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.close()


def main() -> int:
    args = parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else make_output_path("requested_baseline_scatters_anchor.txt").parent / "requested_baseline_scatters"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("observe", "total_steps", "total_cost"):
        _collector, _y_label, _x_suffix, exp4_filename, exp3_filename = metric_spec(metric)
        plot_single_experiment("exp4", EXP4_MODELS, metric, output_dir / exp4_filename)
        plot_single_experiment("exp3", EXP3_MODELS, metric, output_dir / exp3_filename)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
