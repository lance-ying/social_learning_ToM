#!/usr/bin/env python3
"""Lift over the Naive observer for predicting human total steps.

The pooled total-step scatterplots look similar across models because total steps
is dominated by a task-structure floor (how far the agent must travel) that even
the Naive observer reproduces. The sharper question is: how much variance in human
total steps does each model explain *beyond* that task-structure baseline?

For each model we fit two nested OLS regressions of human total steps:
    base : human ~ naive_prediction
    full : human ~ naive_prediction + model_prediction
The incremental R^2 (full minus base) is the variance the model explains on top of
the Naive baseline -- i.e. the contribution of mentalizing/utility once task
difficulty is already accounted for. Bootstrap CIs come from resampling levels.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENTS,
    PRIMARY_POOLED_TOTAL_STEP_PANELS,
    aggregate_human_total_steps,
    apply_reference_style,
    load_model_total_steps_predictions,
    make_output_path,
    preferred_model_name,
)

NAIVE_MODEL = "naive_observer"
BAR_COLOR = "#4878a8"
N_BOOTSTRAP = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bar chart of incremental R^2 over the Naive observer for human total steps."
    )
    parser.add_argument("--output-file", help="Optional output path.")
    return parser.parse_args()


def pooled_table(panels: list[tuple[str, str]]):
    """Return (human, {model_name: predictions}) as pooled arrays over matched levels."""
    model_names = [name for _label, name in panels]
    human_all: list[float] = []
    preds_all: dict[str, list[float]] = {name: [] for name in model_names}

    for exp in EXPERIMENTS:
        human = aggregate_human_total_steps(exp)
        preds = {
            name: load_model_total_steps_predictions(exp, preferred_model_name(exp, name))
            for name in model_names
        }
        # Levels present for the human data and every model (with a total_steps field).
        common = set(human)
        for name in model_names:
            common &= {lvl for lvl, val in preds[name].items() if "total_steps" in val}
        for lvl in sorted(common):
            human_all.append(float(human[lvl]["mean"]))
            for name in model_names:
                preds_all[name].append(float(preds[name][lvl]["total_steps"]))

    human_arr = np.asarray(human_all, dtype=float)
    preds_arr = {name: np.asarray(vals, dtype=float) for name, vals in preds_all.items()}
    return human_arr, preds_arr


def r_squared(y: np.ndarray, X: np.ndarray) -> float:
    """OLS R^2 of y on design X (intercept added internally)."""
    design = np.column_stack([np.ones(len(y)), X])
    coef, _res, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def incremental_r2(y: np.ndarray, naive: np.ndarray, model: np.ndarray) -> float:
    base = r_squared(y, naive[:, None])
    full = r_squared(y, np.column_stack([naive, model]))
    return full - base


def bootstrap_increment(y, naive, model):
    n = len(y)
    point = incremental_r2(y, naive, model)
    samples = np.empty(N_BOOTSTRAP, dtype=float)
    rng_idx = np.arange(n)
    for b in range(N_BOOTSTRAP):
        # Deterministic-ish resample: rotate + jitter via index arithmetic to avoid
        # forbidden RNG calls; use np.random through a seeded generator instead.
        idx = _resample_indices(rng_idx, b)
        samples[b] = incremental_r2(y[idx], naive[idx], model[idx])
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return point, lo, hi


def _resample_indices(base_idx: np.ndarray, seed: int) -> np.ndarray:
    gen = np.random.default_rng(seed)
    return gen.integers(0, len(base_idx), size=len(base_idx))


def render_plot(panels: list[tuple[str, str]], output_file) -> None:
    human, preds = pooled_table(panels)
    naive = preds[NAIVE_MODEL]
    base_r2 = r_squared(human, naive[:, None])

    # Everything except the Naive baseline gets an incremental-R^2 bar.
    bars = [(label, name) for label, name in panels if name != NAIVE_MODEL]

    labels, points, errs_low, errs_high = [], [], [], []
    for label, name in bars:
        point, lo, hi = bootstrap_increment(human, naive, preds[name])
        labels.append(label)
        points.append(point)
        errs_low.append(point - lo)
        errs_high.append(hi - point)

    fig, ax = plt.subplots(figsize=(2.6 * len(bars) + 2, 6.5))
    apply_reference_style(ax)

    x = np.arange(len(bars))
    ax.bar(
        x,
        points,
        width=0.62,
        color=BAR_COLOR,
        edgecolor="none",
        zorder=2,
    )
    ax.errorbar(
        x,
        points,
        yerr=[errs_low, errs_high],
        fmt="none",
        ecolor="#1a1a1a",
        elinewidth=1.6,
        capsize=6,
        capthick=1.6,
        zorder=3,
    )
    ax.axhline(0.0, color="#666666", linewidth=1.2, zorder=1)

    for xi, point in zip(x, points):
        offset = 0.012 if point >= 0 else -0.012
        ax.text(
            xi,
            point + offset,
            f"{point:.2f}",
            ha="center",
            va="bottom" if point >= 0 else "top",
            fontsize=18,
            color="#1a1a1a",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18, color="#1a1a1a")
    ax.set_ylabel("Incremental $R^2$ over Naive Observer", fontsize=22, color="#1a1a1a")
    ax.set_title(
        f"Variance in human total steps explained beyond task structure\n"
        f"(Naive baseline $R^2$ = {base_r2:.2f}, n = {len(human)} levels)",
        fontsize=18,
        color="#1a1a1a",
    )

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved -> {output_file}")
    print(f"Naive baseline R^2 = {base_r2:.4f} (n = {len(human)} levels)")
    for label, name in bars:
        point, lo, hi = bootstrap_increment(human, naive, preds[name])
        print(f"  {label.replace(chr(10), ' '):55s} incremental R^2 = {point:.4f}  CI [{lo:.4f}, {hi:.4f}]")
    plt.close()


def main() -> int:
    args = parse_args()
    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)
    else:
        output_file = make_output_path("total_steps_lift_over_naive_exp1234.png")
    render_plot(list(PRIMARY_POOLED_TOTAL_STEP_PANELS), output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
