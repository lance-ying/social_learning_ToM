#!/usr/bin/env python3
"""Observation-vs-execution trade-off panels.

`total_steps = observe_steps + execution_steps` (execution = move + interact).
The execution component dominates total steps and is largely reproduced by every
observer, which is why the pooled total-step scatterplots look similar across
models. This view instead asks a sharper question: does each model reproduce the
*relationship* between how long an agent observes and how many steps it then needs
to execute? Per panel we overlay the human observe->execute cloud (grey) against a
single model's predicted cloud (red), pooled across experiments, with a linear fit
and slope for each.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENTS,
    NON_TASK_LEVELS,
    PRIMARY_POOLED_TOTAL_STEP_PANELS,
    REPO_ROOT,
    apply_reference_style,
    filtered_participant_paths,
    human_total_level_to_model,
    load_json,
    load_model_total_steps_predictions,
    make_output_path,
    parse_participant_csv,
    preferred_model_name,
    resolve_model_panels,
)

PALETTES = {
    "default": ("#7a7a7a", "#e15759"),       # grey / red
    "blue_orange": ("#4C72B0", "#DD8452"),   # colorblind-safe classic
    "teal_coral": ("#2A9D8F", "#E76F51"),    # warm / modern
    "slate_amber": ("#5D6D7E", "#E1A140"),   # muted / elegant
}
DEFAULT_PALETTE = "teal_coral"
HUMAN_COLOR, MODEL_COLOR = PALETTES[DEFAULT_PALETTE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the observation-vs-execution-steps trade-off, human vs each model, pooled over exps 1-4."
    )
    parser.add_argument("--output-file", help="Optional output path.")
    parser.add_argument("--models", help="Optional comma-separated model list.")
    parser.add_argument(
        "--human-source",
        choices=["filtered", "json"],
        default="filtered",
        help=(
            "filtered (default): recompute human observe/execution from the IQR-cleaned "
            "participant set used by figures 3/4. json: read the unfiltered human_costs JSON."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["steps", "cost"],
        default="steps",
        help="steps (default): observe/execution step counts. cost: 1/3/5-weighted observe/execution cost.",
    )
    parser.add_argument("--palette", choices=list(PALETTES), default=DEFAULT_PALETTE,
                        help="Human/Model colour pair.")
    return parser.parse_args()


def aggregate_human_components_json(exp: str) -> dict[str, tuple[float, float]]:
    """Unfiltered human components straight from the human_costs JSON (all participants)."""
    raw = load_json(REPO_ROOT / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs.json")
    per_case = raw.get("per_case", {})
    out: dict[str, tuple[float, float]] = {}
    for level, stats in per_case.items():
        observe = float(stats["observe_steps_mean"])
        execution = float(stats["planning_steps_mean"])  # move + interact
        out[human_total_level_to_model(exp, level)] = (observe, execution)
    return out


def aggregate_human_components_filtered(exp: str) -> dict[str, tuple[float, float]]:
    """Human components from the IQR-filtered participant set used by figures 3/4.

    Per participant per level: observe = observe_count, execution = total_steps - observe_count
    (= move + interact), matching the model's planning_steps definition.
    """
    obs_vals: dict[str, list[float]] = defaultdict(list)
    exec_vals: dict[str, list[float]] = defaultdict(list)
    for path in filtered_participant_paths(exp):
        for level, counts in parse_participant_csv(path).items():
            if level in NON_TASK_LEVELS:
                continue
            model_level = human_total_level_to_model(exp, level)
            observe = float(counts["observe_count"])
            execution = float(counts["total_steps"]) - observe
            obs_vals[model_level].append(observe)
            exec_vals[model_level].append(execution)
    return {
        level: (float(np.mean(obs_vals[level])), float(np.mean(exec_vals[level])))
        for level in obs_vals
    }


def aggregate_human_components(exp: str, source: str = "filtered") -> dict[str, tuple[float, float]]:
    if source == "json":
        return aggregate_human_components_json(exp)
    return aggregate_human_components_filtered(exp)


def collect_tradeoff_pairs(exp: str, model_name: str, human_source: str = "filtered"):
    """Matched per-level (observe, execution) points for human and model."""
    human = aggregate_human_components(exp, human_source)
    model_preds = load_model_total_steps_predictions(exp, model_name)

    h_obs, h_exec, m_obs, m_exec = [], [], [], []
    for level, value in model_preds.items():
        if level not in human:
            continue
        if "observe_steps" not in value or "planning_steps" not in value:
            continue
        ho, he = human[level]
        h_obs.append(ho)
        h_exec.append(he)
        m_obs.append(float(value["observe_steps"]))
        m_exec.append(float(value["planning_steps"]))
    return (
        np.asarray(h_obs, dtype=float),
        np.asarray(h_exec, dtype=float),
        np.asarray(m_obs, dtype=float),
        np.asarray(m_exec, dtype=float),
    )


def collect_cost_pairs(exp: str, model_name: str, human_source: str = "filtered"):
    """Matched per-level (observe_cost, execution_cost) points for human and model.

    Observe cost = 1*observe_steps; execution cost = 3*move + 5*interact. Human values
    come from the IQR-filtered human_costs JSON so the participant set matches figures 3/4.
    """
    suffix = "_iqr_filtered" if human_source == "filtered" else ""
    raw = load_json(REPO_ROOT / "data_processing" / "outputs" / "human_costs" / f"{exp}_human_costs{suffix}.json")

    def norm(lvl):
        key = human_total_level_to_model(exp, lvl)
        return key[:-6] if key.endswith("_ascii") else key  # exp1 JSON keys carry an _ascii suffix

    human = {
        norm(lvl): (float(s["observe_cost_mean"]), float(s["planning_cost_mean"]))
        for lvl, s in raw.get("per_case", {}).items()
    }
    model_preds = load_model_total_steps_predictions(exp, model_name)

    h_obs, h_exec, m_obs, m_exec = [], [], [], []
    for level, value in model_preds.items():
        if level not in human or "observe_cost" not in value or "planning_cost" not in value:
            continue
        ho, he = human[level]
        h_obs.append(ho)
        h_exec.append(he)
        m_obs.append(float(value["observe_cost"]))
        m_exec.append(float(value["planning_cost"]))
    return (
        np.asarray(h_obs, dtype=float),
        np.asarray(h_exec, dtype=float),
        np.asarray(m_obs, dtype=float),
        np.asarray(m_exec, dtype=float),
    )


def draw_fit(ax, x, y, color, lo, hi, seed, n_resamples=1000):
    """Draw the best-fit line plus a bootstrap 95% CI band; return (slope, ci_lo, ci_hi).

    The slope CI and the shaded band both come from resampling (x, y) pairs with
    replacement (n_resamples), matching the bootstrap approach used for figures 3/4.
    """
    if len(x) < 2:
        return None, None, None
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.array([lo, hi], dtype=float)

    slope_lo = slope_hi = None
    if len(x) >= 3:
        rng = np.random.default_rng(seed)
        n = len(x)
        grid = np.linspace(lo, hi, 60)
        slopes = np.empty(n_resamples)
        preds = np.empty((n_resamples, grid.size))
        for i in range(n_resamples):
            idx = rng.integers(0, n, n)
            mi, bi = np.polyfit(x[idx], y[idx], 1)
            slopes[i] = mi
            preds[i] = mi * grid + bi
        slope_lo, slope_hi = (float(v) for v in np.percentile(slopes, [2.5, 97.5]))
        band_lo = np.percentile(preds, 2.5, axis=0)
        band_hi = np.percentile(preds, 97.5, axis=0)
        ax.fill_between(grid, band_lo, band_hi, color=color, alpha=0.15, linewidth=0, zorder=2)

    ax.plot(fit_x, slope * fit_x + intercept, color=color, linewidth=2.0, zorder=4)
    return float(slope), slope_lo, slope_hi


def render_plot(panels: list[tuple[str, str]], output_file, human_source: str = "filtered",
                metric: str = "steps") -> None:
    collector = collect_cost_pairs if metric == "cost" else collect_tradeoff_pairs
    y_title = "Execution Cost\n(3*move + 5*interact)" if metric == "cost" else "Execution Steps\n(move + interact)"
    x_title = "Observation Cost" if metric == "cost" else "Observation Steps"

    # Width 5 in/panel matches figure 3 so square panels end up ~the same physical size
    # (and tick numbers therefore render at the same apparent size); extra height covers
    # the on-top title.
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 6.6), squeeze=False)
    axes = axes[0]

    # Pool data first so every panel shares identical axis limits.
    panel_data = []
    all_obs, all_exec = [], []
    for _label, model_name in panels:
        h_obs, h_exec, m_obs, m_exec = [], [], [], []
        for exp in EXPERIMENTS:
            ho, he, mo, me = collector(exp, preferred_model_name(exp, model_name), human_source)
            h_obs.append(ho)
            h_exec.append(he)
            m_obs.append(mo)
            m_exec.append(me)
        h_obs = np.concatenate(h_obs) if h_obs else np.array([])
        h_exec = np.concatenate(h_exec) if h_exec else np.array([])
        m_obs = np.concatenate(m_obs) if m_obs else np.array([])
        m_exec = np.concatenate(m_exec) if m_exec else np.array([])
        panel_data.append((h_obs, h_exec, m_obs, m_exec))
        all_obs.extend(h_obs.tolist() + m_obs.tolist())
        all_exec.extend(h_exec.tolist() + m_exec.tolist())

    obs_lo, obs_hi = (min(all_obs), max(all_obs)) if all_obs else (0.0, 1.0)
    exec_lo, exec_hi = (min(all_exec), max(all_exec)) if all_exec else (0.0, 1.0)
    obs_pad = 0.05 * (obs_hi - obs_lo or 1.0)
    exec_pad = 0.05 * (exec_hi - exec_lo or 1.0)
    obs_lo, obs_hi = obs_lo - obs_pad, obs_hi + obs_pad
    exec_lo, exec_hi = exec_lo - exec_pad, exec_hi + exec_pad

    for idx, (ax, (label, _model_name), (h_obs, h_exec, m_obs, m_exec)) in enumerate(zip(axes, panels, panel_data)):
        apply_reference_style(ax)

        ax.scatter(h_obs, h_exec, s=18, alpha=0.5, color=HUMAN_COLOR, edgecolors="none", zorder=3, label="Human")
        ax.scatter(m_obs, m_exec, s=18, alpha=0.6, color=MODEL_COLOR, edgecolors="none", zorder=3, label="Model")

        # The two fitted trend lines carry the message (does the red model line track or
        # invert the grey human line). Only the model slope -- the part that varies across
        # panels -- is labelled, minor text in the top-right; the human slope is in the caption.
        h_slope, h_lo, h_hi = draw_fit(ax, h_obs, h_exec, HUMAN_COLOR, obs_lo, obs_hi, seed=1)
        m_slope, m_lo, m_hi = draw_fit(ax, m_obs, m_exec, MODEL_COLOR, obs_lo, obs_hi, seed=2)
        print(f"  {label.replace(chr(10), ' '):50s} human slope = {h_slope:+.3f}  model slope = {m_slope:+.3f}")
        slope_lines = []
        if h_slope is not None:
            h_txt = rf"$m_{{\mathrm{{human}}}} = {h_slope:.2f}$"
            if h_lo is not None:
                h_txt += rf" [{h_lo:.2f}, {h_hi:.2f}]"
            slope_lines.append(h_txt)
        if m_slope is not None:
            m_txt = rf"$m_{{\mathrm{{model}}}} = {m_slope:.2f}$"
            if m_lo is not None:
                m_txt += rf" [{m_lo:.2f}, {m_hi:.2f}]"
            slope_lines.append(m_txt)
        if slope_lines:
            ax.text(0.05, 0.95, "\n".join(slope_lines), transform=ax.transAxes,
                    ha="left", va="top", ma="left", fontsize=14, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.65),
                    zorder=5)

        ax.set_xlim(obs_lo, obs_hi)
        ax.set_ylim(exec_lo, exec_hi)
        ax.set_box_aspect(1)  # square panel boxes, matching figures 3/4
        # Model name is a TITLE (not an x-label) so the bottom axis reads as the shared
        # "Observation Steps" quantity; the legend (red = Model) carries the colour coding.
        # Single-line titles (e.g. "Naive Observer") are dropped ~half a line so they sit
        # centered against the two-line titles instead of aligning to the top line.
        title_pad = 10 if "\n" in label else -1
        ax.set_title(label, fontsize=16, color="#1a1a1a", pad=title_pad)
        ax.set_xlabel(x_title, fontsize=20, color="#1a1a1a")
        if idx == 0:
            ax.set_ylabel(y_title, fontsize=20, color="#1a1a1a", labelpad=8)
        else:
            ax.set_ylabel("")

    # Reserve a generous bottom band for: per-panel model labels, the shared
    # axis title, and the Human/Model legend -- stacked with clear gaps so the
    # shared x-axis title reads as the axis label, not another panel label.
    left, right = 0.075, 0.99
    fig.subplots_adjust(left=left, right=right, top=0.90, bottom=0.20, wspace=0.16)

    # Human/Model legend as a centered row below the panels (like the experiment legend
    # in the observation/total-steps figures); each panel carries its own x-axis label.
    panel_center = (left + right) / 2
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=20, markerscale=2.4,
                   loc="lower center", ncol=2, bbox_to_anchor=(panel_center, 0.0))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved -> {output_file}")
    plt.close()


def main() -> int:
    args = parse_args()
    model_names = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else None

    global HUMAN_COLOR, MODEL_COLOR
    HUMAN_COLOR, MODEL_COLOR = PALETTES[args.palette]

    if args.output_file:
        from pathlib import Path

        output_file = Path(args.output_file)
    else:
        parts = "" if args.human_source == "filtered" else f"_{args.human_source}"
        parts += "_cost" if args.metric == "cost" else ""
        parts += "" if args.palette == DEFAULT_PALETTE else f"_{args.palette}"
        output_file = make_output_path(f"observe_execute_tradeoff_exp1234{parts}.png")

    panels = resolve_model_panels(model_names) if model_names is not None else list(PRIMARY_POOLED_TOTAL_STEP_PANELS)
    print(f"human_source = {args.human_source}  metric = {args.metric}")
    render_plot(panels, output_file, args.human_source, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
