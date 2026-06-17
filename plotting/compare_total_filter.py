#!/usr/bin/env python3
"""How much does the IQR participant filter move the total-steps 4-panel numbers?"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from common import (
    EXPERIMENTS,
    NON_TASK_LEVELS,
    PRIMARY_POOLED_TOTAL_STEP_PANELS,
    bootstrap_ccc_ci,
    bootstrap_r_ci,
    filtered_participant_paths,
    human_total_level_to_model,
    load_model_total_steps_predictions,
    parse_participant_csv,
    participant_quality_scores,
    preferred_model_name,
)


def all_paths(exp):
    return tuple(sorted(participant_quality_scores(exp).keys()))


def human_total_over(exp, paths):
    vals = defaultdict(list)
    for path in paths:
        for level, counts in parse_participant_csv(path).items():
            if level in NON_TASK_LEVELS:
                continue
            vals[human_total_level_to_model(exp, level)].append(float(counts["total_steps"]))
    return {lvl: float(np.mean(v)) for lvl, v in vals.items()}


def pairs(exp, model_name, human):
    preds = load_model_total_steps_predictions(exp, model_name)
    xs, ys = [], []
    for lvl, val in preds.items():
        if lvl in human and "total_steps" in val:
            xs.append(float(val["total_steps"]))
            ys.append(human[lvl])
    return np.asarray(xs), np.asarray(ys)


def stats(panels, source):
    out = []
    for _label, name in panels:
        xs, ys = [], []
        for exp in EXPERIMENTS:
            paths = filtered_participant_paths(exp) if source == "filtered" else all_paths(exp)
            x, y = pairs(exp, preferred_model_name(exp, name), human_total_over(exp, paths))
            xs.append(x)
            ys.append(y)
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        r, _lo, _hi = bootstrap_r_ci(x, y, n_resamples=1000)
        ccc, _l, _h = bootstrap_ccc_ci(x, y, n_resamples=1000)
        out.append((len(x), r, ccc))
    return out


def main():
    panels = list(PRIMARY_POOLED_TOTAL_STEP_PANELS)
    filt = stats(panels, "filtered")
    allp = stats(panels, "all")
    print("Deviation of UNFILTERED vs ORIGINAL (IQR-filtered) total-steps figure:\n")
    header = f"{'panel':50s} {'n':>4} {'r_filt':>7} {'r_all':>7} {'dr':>6} {'CCC_filt':>9} {'CCC_all':>8} {'dCCC':>6}"
    print(header)
    print("-" * len(header))
    for (label, _name), f, a in zip(panels, filt, allp):
        print(f"{label.replace(chr(10), ' '):50s} {f[0]:4d} {f[1]:7.3f} {a[1]:7.3f} {a[1]-f[1]:+6.3f} "
              f"{f[2]:9.3f} {a[2]:8.3f} {a[2]-f[2]:+6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
