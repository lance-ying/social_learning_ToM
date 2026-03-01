"""
4x1 scatter subplot: Exp2 human observation data vs 4 model/baseline predictions.
Each panel shows model predictions (x) vs human mean observe per activation (y)
with bootstrap SD error bars on the y-axis and a line of best fit.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from correlation_4panel_style import (
    annotate_r_ci,
    apply_reference_style,
    bootstrap_r_ci,
    plot_points_errorbars_and_fit,
)

# ── paths ──────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent            # data_processing/scripts/
data_processing_dir = script_dir.parent       # data_processing/
workspace_root = data_processing_dir.parent   # repo root

# ── load model / baseline JSONs ───────────────────────────────────────────
with open(workspace_root / 'steps_exp2.json') as f:
    steps_dict = json.load(f)

with open(workspace_root / 'scripts/baselines/exp2/step_dict_naive_exp2.json') as f:
    baseline_naive = json.load(f)

with open(workspace_root / 'scripts/baselines/exp2/step_dict_nonmentalize_exp2.json') as f:
    baseline_nonmentalize = json.load(f)

with open(workspace_root / 'scripts/baselines/exp2/step_dict_mentalize_exp2.json') as f:
    baseline_mentalize = json.load(f)

# ── load human data ───────────────────────────────────────────────────────
results_dict = {}
exec((data_processing_dir / 'results/current/results_dict_merged_exp2.py').read_text())
human = results_dict.copy()

# extract participant count from comment
_content = (data_processing_dir / 'results/current/results_dict_merged_exp2.py').read_text()
n_participants = None
for line in _content.split('\n'):
    if 'Unique files' in line:
        m = re.search(r'(\d+)', line)
        if m:
            n_participants = int(m.group(1))
            break

# ── helper: collect matched arrays ────────────────────────────────────────
def collect_pairs(model_dict, human_dict, key_transform):
    """Return arrays: model_vals, human_means, human_sds, matched_keys."""
    m_vals, h_means, h_sds, keys = [], [], [], []
    for mk in model_dict:
        hk = key_transform(mk)
        if hk in human_dict:
            m_vals.append(model_dict[mk])
            h_means.append(human_dict[hk]['mean_observe_per_activation'])
            h_sds.append(human_dict[hk]['bootstrap_sd'])
            keys.append(hk)
    return np.array(m_vals), np.array(h_means), np.array(h_sds), keys

# exp2 keys are already in the same format (s211_1, etc.) for both model and human
identity_transform = lambda k: k

pairs = {
    'Rational Mentalizing\n(Full Model)': collect_pairs(steps_dict,            human, identity_transform),
    'Social Mentalizing':                 collect_pairs(baseline_mentalize,     human, identity_transform),
    'Rational Non-Mentalizing':           collect_pairs(baseline_nonmentalize,  human, identity_transform),
    'Naive Observer':                     collect_pairs(baseline_naive,         human, identity_transform),
}

# ── plot (4 columns x 1 row) ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Experiment 2", fontsize=30, color="#1a1a1a", y=0.99)

for idx, (ax, (label, (x, y, sd, _keys))) in enumerate(zip(axes, pairs.items())):
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
        continue

    plot_points_errorbars_and_fit(ax, x, y, sd)
    r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
    annotate_r_ci(ax, r, ci_low, ci_high)

    ax.set_xlabel(label, fontsize=22, color="#1a1a1a")
    if idx == 0:
        ax.set_ylabel('Human', fontsize=24, color="#1a1a1a")
    else:
        ax.set_ylabel('')

plt.tight_layout(w_pad=2.5, rect=(0, 0, 1, 0.97))

output_dir = data_processing_dir / 'outputs/plots'
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / 'correlation_exp2_4panel.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()
