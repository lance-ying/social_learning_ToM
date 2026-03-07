"""
Combined 2×4 scatter plot: Exp1 (top row) and Exp2 (bottom row).
Each column is one model; row labels show experiment; column labels on bottom row only.
"""
import json, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from correlation_4panel_style import (
    annotate_r_ci,
    apply_reference_style,
    bootstrap_r_ci,
    plot_points_errorbars_and_fit,
)

script_dir = Path(__file__).parent
data_processing_dir = script_dir.parent
workspace_root = data_processing_dir.parent

# ── Load Exp1 ──────────────────────────────────────────────────────────────
with open(workspace_root / 'steps.dict.json') as f:
    model_exp1 = json.load(f)
with open(workspace_root / 'scripts/baselines/exp1/step_dict_naive_exp1.json') as f:
    naive_exp1 = json.load(f)
with open(workspace_root / 'scripts/baselines/exp1/step_dict_nonmentalize_exp1.json') as f:
    nonmentalize_exp1 = json.load(f)
with open(workspace_root / 'scripts/baselines/exp1/step_dict_mentalize_exp1.json') as f:
    mentalize_exp1 = json.load(f)

results_dict = {}
exec((data_processing_dir / 'results/current/results_dict_exp1_50.py').read_text())
human_exp1 = results_dict.copy()

# ── Load Exp2 ──────────────────────────────────────────────────────────────
with open(workspace_root / 'steps_exp2.json') as f:
    model_exp2 = json.load(f)
with open(workspace_root / 'scripts/baselines/exp2/step_dict_naive_exp2.json') as f:
    naive_exp2 = json.load(f)
with open(workspace_root / 'scripts/baselines/exp2/step_dict_nonmentalize_exp2.json') as f:
    nonmentalize_exp2 = json.load(f)
with open(workspace_root / 'scripts/baselines/exp2/step_dict_mentalize_exp2.json') as f:
    mentalize_exp2 = json.load(f)

results_dict = {}
exec((data_processing_dir / 'results/current/results_dict_merged_exp2.py').read_text())
human_exp2 = results_dict.copy()

# ── helpers ────────────────────────────────────────────────────────────────
def collect_pairs(model_dict, human_dict, key_transform):
    m_vals, h_means, h_sds, keys = [], [], [], []
    for mk in model_dict:
        hk = key_transform(mk)
        if hk in human_dict:
            m_vals.append(model_dict[mk])
            h_means.append(human_dict[hk]['mean_observe_per_activation'])
            h_sds.append(human_dict[hk]['bootstrap_sd'])
            keys.append(hk)
    return np.array(m_vals), np.array(h_means), np.array(h_sds), keys

main_transform_exp1      = lambda k: k.replace('_ascii', '_1')
baseline_transform_exp1  = lambda k: f'mod_{k}_1'
identity_transform       = lambda k: k

LABELS = [
    'Rational Mentalizing\n(Full Model)',
    'Social Mentalizing',
    'Rational Non-Mentalizing',
    'Naive Observer',
]

pairs_exp1 = {
    LABELS[0]: collect_pairs(model_exp1,       human_exp1, main_transform_exp1),
    LABELS[1]: collect_pairs(mentalize_exp1,   human_exp1, baseline_transform_exp1),
    LABELS[2]: collect_pairs(nonmentalize_exp1,human_exp1, baseline_transform_exp1),
    LABELS[3]: collect_pairs(naive_exp1,       human_exp1, baseline_transform_exp1),
}

pairs_exp2 = {
    LABELS[0]: collect_pairs(model_exp2,       human_exp2, identity_transform),
    LABELS[1]: collect_pairs(mentalize_exp2,   human_exp2, identity_transform),
    LABELS[2]: collect_pairs(nonmentalize_exp2,human_exp2, identity_transform),
    LABELS[3]: collect_pairs(naive_exp2,       human_exp2, identity_transform),
}

# ── plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Experiments 1 & 2", fontsize=30, color="#1a1a1a", y=1.01)

for row, (exp_label, pairs) in enumerate([
    ("Experiment 1", pairs_exp1),
    ("Experiment 2", pairs_exp2),
]):
    for idx, (ax, (label, (x, y, sd, _))) in enumerate(zip(axes[row], pairs.items())):
        apply_reference_style(ax)

        if len(x) < 3:
            ax.text(0.05, 0.90, "Insufficient data", transform=ax.transAxes,
                    ha="left", va="top", fontsize=18, color="#1a1a1a")
            continue

        plot_points_errorbars_and_fit(ax, x, y, sd)
        r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
        annotate_r_ci(ax, r, ci_low, ci_high)

        # Column labels only on bottom row
        if row == 1:
            ax.set_xlabel(label, fontsize=22, color="#1a1a1a")
        else:
            ax.set_xlabel('')

        # Row label on leftmost column only
        if idx == 0:
            ax.set_ylabel(f'{exp_label}\nHuman', fontsize=20, color="#1a1a1a")
        else:
            ax.set_ylabel('')

plt.tight_layout(w_pad=2.5, h_pad=3.5)

output_dir = data_processing_dir / 'outputs/plots'
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / 'correlation_exp12_combined.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()
