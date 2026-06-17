#!/usr/bin/env python3
"""Per-experiment view of the observation-vs-execution trade-off.

(1) Prints per-experiment human/model slopes for each model, to check that the
    pooled trade-off is not a between-experiment pooling artifact (Simpson's paradox).
(2) Renders a 4x4 facet: rows = experiments 1-4, columns = the four observer models,
    each cell showing that experiment's human (grey) vs model (red) clouds + fits.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (
    EXPERIMENT_LABELS,
    EXPERIMENTS,
    PRIMARY_POOLED_TOTAL_STEP_PANELS,
    REPO_ROOT,
    apply_reference_style,
    human_total_level_to_model,
    load_json,
    load_model_total_steps_predictions,
    make_output_path,
    preferred_model_name,
)
from plot_observe_execute_tradeoff import HUMAN_COLOR, MODEL_COLOR, collect_tradeoff_pairs, draw_fit


def collect_cost_pairs(exp: str, model_name: str, human_source: str = "filtered"):
    """Matched per-level (observe_cost, execution_cost) points for human and model.

    Observe cost = 1*observe_steps; execution cost = 3*move + 5*interact. Uses the
    IQR-filtered human_costs JSON so the participant set matches figures 3/4.
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
    for lvl, val in model_preds.items():
        if lvl not in human or "observe_cost" not in val or "planning_cost" not in val:
            continue
        ho, he = human[lvl]
        h_obs.append(ho)
        h_exec.append(he)
        m_obs.append(float(val["observe_cost"]))
        m_exec.append(float(val["planning_cost"]))
    return (np.asarray(h_obs), np.asarray(h_exec), np.asarray(m_obs), np.asarray(m_exec))


def slope(x, y):
    if len(x) < 2:
        return None
    return float(np.polyfit(x, y, 1)[0])


def gather(panels, human_source, metric="steps"):
    """data[(exp, model_name)] = (h_obs, h_exec, m_obs, m_exec)."""
    collector = collect_cost_pairs if metric == "cost" else collect_tradeoff_pairs
    data = {}
    obs_vals, exec_vals = [], []
    for exp in EXPERIMENTS:
        for _label, name in panels:
            ho, he, mo, me = collector(exp, preferred_model_name(exp, name), human_source)
            data[(exp, name)] = (ho, he, mo, me)
            obs_vals += ho.tolist() + mo.tolist()
            exec_vals += he.tolist() + me.tolist()
    return data, obs_vals, exec_vals


def print_step_vs_cost(panels, data_steps, data_cost):
    """Direct comparison: does cost-weighting change the model slopes vs step counts?"""
    print("\nStep-slope vs Cost-slope (model clouds) -- tests whether interaction-heavy")
    print("cells diverge once interactions are weighted 5x:\n")
    header = f"{'model':46s} {'exp':5s} {'M_steps':>8} {'M_cost':>8} {'H_steps':>8} {'H_cost':>8}"
    print(header)
    print("-" * len(header))
    for label, name in panels:
        clean = label.replace("\n", " ")
        for exp in EXPERIMENTS:
            hs, he, ms, me = data_steps[(exp, name)]
            hc, hec, mc, mec = data_cost[(exp, name)]
            def s(a, b):
                v = slope(a, b)
                return f"{v:+.2f}" if v is not None else "  n/a"
            print(f"{clean:46s} {exp:5s} {s(ms, me):>8} {s(mc, mec):>8} {s(hs, he):>8} {s(hc, hec):>8}")
        print()


def print_slope_check(panels, data):
    print("\nPer-experiment slope check (execution ~ observation):\n")
    header = f"{'model':46s} {'exp':5s} {'n':>4} {'human':>7} {'model':>7}"
    print(header)
    print("-" * len(header))
    for label, name in panels:
        clean = label.replace("\n", " ")
        for exp in EXPERIMENTS:
            ho, he, mo, me = data[(exp, name)]
            hs = slope(ho, he)
            ms = slope(mo, me)
            hs_s = f"{hs:+.2f}" if hs is not None else "  n/a"
            ms_s = f"{ms:+.2f}" if ms is not None else "  n/a"
            print(f"{clean:46s} {exp:5s} {len(ho):4d} {hs_s:>7} {ms_s:>7}")
        print()


def render_facet(panels, data, obs_vals, exec_vals, output_file, metric="steps"):
    y_title = "Execution Cost (3*move + 5*interact)" if metric == "cost" else "Execution Steps (move + interact)"
    x_title = "Observation Cost" if metric == "cost" else "Observation Steps"
    n_exp = len(EXPERIMENTS)
    n_model = len(panels)
    # 5 in/panel matches figure 3 and the pooled trade-off, so tick numbers render at the
    # same apparent size when the figures are placed at a common width.
    fig, axes = plt.subplots(n_exp, n_model, figsize=(5 * n_model, 5 * n_exp), squeeze=False)

    obs_lo, obs_hi = (min(obs_vals), max(obs_vals)) if obs_vals else (0.0, 1.0)
    exec_lo, exec_hi = (min(exec_vals), max(exec_vals)) if exec_vals else (0.0, 1.0)
    obs_pad = 0.05 * (obs_hi - obs_lo or 1.0)
    exec_pad = 0.05 * (exec_hi - exec_lo or 1.0)
    obs_lo, obs_hi = obs_lo - obs_pad, obs_hi + obs_pad
    exec_lo, exec_hi = exec_lo - exec_pad, exec_hi + exec_pad

    for r, exp in enumerate(EXPERIMENTS):
        for c, (label, name) in enumerate(panels):
            ax = axes[r][c]
            apply_reference_style(ax)
            ho, he, mo, me = data[(exp, name)]
            ax.scatter(ho, he, s=18, alpha=0.5, color=HUMAN_COLOR, edgecolors="none", zorder=3, label="Human")
            ax.scatter(mo, me, s=18, alpha=0.6, color=MODEL_COLOR, edgecolors="none", zorder=3, label="Model")
            # Same fit + bootstrap 95% CI band styling as the pooled figures.
            hs, h_lo, h_hi = draw_fit(ax, ho, he, HUMAN_COLOR, obs_lo, obs_hi, seed=1)
            ms, m_lo, m_hi = draw_fit(ax, mo, me, MODEL_COLOR, obs_lo, obs_hi, seed=2)

            slope_lines = []
            if hs is not None:
                t = rf"$m_{{\mathrm{{human}}}} = {hs:.2f}$"
                if h_lo is not None:
                    t += rf" [{h_lo:.2f}, {h_hi:.2f}]"
                slope_lines.append(t)
            if ms is not None:
                t = rf"$m_{{\mathrm{{model}}}} = {ms:.2f}$"
                if m_lo is not None:
                    t += rf" [{m_lo:.2f}, {m_hi:.2f}]"
                slope_lines.append(t)
            if slope_lines:
                ax.text(0.05, 0.96, "\n".join(slope_lines), transform=ax.transAxes,
                        ha="left", va="top", ma="left", fontsize=12, color="#333333",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.65),
                        zorder=5)

            ax.set_xlim(obs_lo, obs_hi)
            ax.set_ylim(exec_lo, exec_hi)
            ax.set_box_aspect(1)  # square panel boxes, matching the pooled figures
            ax.tick_params(labelsize=20)  # match the tick-number font used across the other figures
            if r == 0:
                ax.set_title(label, fontsize=17, color="#1a1a1a")
            if c == 0:
                ax.set_ylabel(EXPERIMENT_LABELS[exp], fontsize=18, color="#1a1a1a", labelpad=8)

    fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.18)
    # Shared axis titles, placed clear of the per-row / per-column labels.
    fig.text(0.035, 0.5, y_title, rotation=90, ha="center", va="center",
             fontsize=20, color="#1a1a1a")
    fig.text(0.56, 0.045, x_title, ha="center", va="center", fontsize=20, color="#1a1a1a")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=18, markerscale=2.2,
                   loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.0))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"Saved -> {output_file}")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-experiment trade-off facet + slope check.")
    parser.add_argument("--human-source", choices=["filtered", "json"], default="filtered")
    parser.add_argument("--metric", choices=["steps", "cost"], default="steps")
    parser.add_argument("--output-file")
    args = parser.parse_args()

    panels = list(PRIMARY_POOLED_TOTAL_STEP_PANELS)
    data, obs_vals, exec_vals = gather(panels, args.human_source, args.metric)

    print(f"human_source = {args.human_source}  metric = {args.metric}")
    print_slope_check(panels, data)

    if args.metric == "cost":
        # Show cost slopes side by side with step slopes to test the interaction hypothesis.
        data_steps, _o, _e = gather(panels, args.human_source, "steps")
        print_step_vs_cost(panels, data_steps, data)

    from pathlib import Path
    suffix = "_cost" if args.metric == "cost" else ""
    out = Path(args.output_file) if args.output_file else make_output_path(f"observe_execute_tradeoff_facet_exp1234{suffix}.png")
    render_facet(panels, data, obs_vals, exec_vals, out, args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
