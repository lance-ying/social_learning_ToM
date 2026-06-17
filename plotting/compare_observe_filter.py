#!/usr/bin/env python3
"""Sensitivity of the 4-panel observation-steps figure to the IQR participant filter.

Figure 3 (observation steps) is rendered with the IQR-filtered participant set. This
script rebuilds the same pooled scatter using ALL participants (no IQR filter) and
reports, per model panel, how much r / CI / CCC move relative to the filtered
("original") figure. It also saves the unfiltered figure for visual comparison.
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_COLORS,
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    PRIMARY_POOLED_OBSERVE_PANELS,
    apply_reference_style,
    bootstrap_ccc_ci,
    bootstrap_r_ci,
    filtered_participant_paths,
    human_level_to_model,
    load_model_observe_predictions,
    make_output_path,
    observe_model_scalar,
    participant_quality_scores,
    pooled_limits,
    preferred_model_name,
    sample_sd,
    should_skip_observe_level,
)


def all_participant_paths(exp: str):
    return tuple(sorted(participant_quality_scores(exp).keys()))


def aggregate_observes_over(exp: str, paths) -> dict[str, dict[str, float]]:
    """Replicates aggregate_human_observes (combined metric) over an arbitrary path set."""
    from common import parse_participant_csv

    combined: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        for level, counts in parse_participant_csv(path).items():
            if should_skip_observe_level(exp, level):
                continue
            model_level = human_level_to_model(exp, level)
            activation = float(counts.get("activation_count", 0.0))
            if activation <= 0:
                continue
            combined[model_level].append(float(counts["observe_count"]) / activation)
    return {
        level: {"combined_mean": float(np.mean(vals)), "combined_sd": sample_sd(vals)}
        for level, vals in combined.items()
    }


def collect_pairs(exp: str, model_name: str, human_stats: dict[str, dict[str, float]]):
    model_predictions = load_model_observe_predictions(exp, model_name)
    xs, ys = [], []
    for level, model_value in model_predictions.items():
        if level not in human_stats:
            continue
        xs.append(observe_model_scalar(exp, model_value, "combined"))
        ys.append(float(human_stats[level]["combined_mean"]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def gather(panels, source: str):
    """source in {'filtered', 'all'}. Returns per-panel by_exp lists of (exp, x, y)."""
    panel_data = []
    for _label, model_name in panels:
        by_exp = []
        for exp in EXPERIMENTS:
            paths = filtered_participant_paths(exp) if source == "filtered" else all_participant_paths(exp)
            human_stats = aggregate_observes_over(exp, paths)
            x, y = collect_pairs(exp, preferred_model_name(exp, model_name), human_stats)
            by_exp.append((exp, x, y))
        panel_data.append(by_exp)
    return panel_data


def pooled_stats(by_exp):
    xs = [x for _exp, x, _y in by_exp if len(x)]
    ys = [y for _exp, _x, y in by_exp if len(y)]
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    r, ci_lo, ci_hi = bootstrap_r_ci(x, y, n_resamples=1000)
    ccc, _lo, _hi = bootstrap_ccc_ci(x, y, n_resamples=1000)
    return {"n": len(x), "r": r, "ci": (ci_lo, ci_hi), "ccc": ccc}


def render_unfiltered(panels, panel_data, output_file):
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6), squeeze=False)
    axes = axes[0]
    lo, hi = pooled_limits(panel_data)

    for idx, (ax, (label, _model_name), by_exp) in enumerate(zip(axes, panels, panel_data)):
        apply_reference_style(ax)
        pooled_x, pooled_y = [], []
        for exp, x, y in by_exp:
            if len(x) == 0:
                continue
            pooled_x.append(x)
            pooled_y.append(y)
            ax.scatter(x, y, s=36, alpha=0.85, color=EXPERIMENT_COLORS[exp],
                       edgecolors="none", label=EXPERIMENT_LABELS[exp], zorder=3)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color="#666666", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        if pooled_x:
            px = np.concatenate(pooled_x)
            py = np.concatenate(pooled_y)
            r, ci_lo, ci_hi = bootstrap_r_ci(px, py, n_resamples=1000)
            ccc, _l, _h = bootstrap_ccc_ci(px, py, n_resamples=1000)
            ax.text(0.05, 0.90, f"r = {r:.2f}\nCI = [{ci_lo:.2f}, {ci_hi:.2f}]\nCCC = {ccc:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=18, color="#1a1a1a")
            if len(px) >= 2:
                slope, intercept = np.polyfit(px, py, 1)
                fit_x = np.array([lo, hi], dtype=float)
                ax.plot(fit_x, slope * fit_x + intercept, color="#e15759", linewidth=1.8, zorder=2)
        ax.set_xlabel(label, fontsize=24, color="#1a1a1a")
        ax.set_ylabel("Human Observation Steps (unfiltered)" if idx == 0 else "", fontsize=24, color="#1a1a1a")

    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=20, markerscale=2.8,
                   loc="lower center", ncol=len(EXPERIMENTS), bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout(w_pad=2.5, rect=(0, 0.12, 1, 1))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    plt.close()


def main() -> int:
    panels = list(PRIMARY_POOLED_OBSERVE_PANELS)
    filt = gather(panels, "filtered")
    allp = gather(panels, "all")

    out = make_output_path("observation_steps_model_vs_human_pooled_exp1234_unfiltered.png")
    render_unfiltered(panels, allp, out)

    print("\nDeviation of UNFILTERED vs ORIGINAL (IQR-filtered) observation-steps figure:\n")
    header = f"{'panel':50s} {'n_filt':>6} {'n_all':>6} {'r_filt':>7} {'r_all':>7} {'dr':>6} {'CCC_filt':>9} {'CCC_all':>8} {'dCCC':>6}"
    print(header)
    print("-" * len(header))
    for (label, _name), bf, ba in zip(panels, filt, allp):
        sf = pooled_stats(bf)
        sa = pooled_stats(ba)
        if sf is None or sa is None:
            continue
        name = label.replace("\n", " ")
        print(f"{name:50s} {sf['n']:6d} {sa['n']:6d} {sf['r']:7.3f} {sa['r']:7.3f} "
              f"{sa['r'] - sf['r']:+6.3f} {sf['ccc']:9.3f} {sa['ccc']:8.3f} {sa['ccc'] - sf['ccc']:+6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
