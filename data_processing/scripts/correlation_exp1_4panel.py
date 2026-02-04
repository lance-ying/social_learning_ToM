"""
4x1 scatter subplot: Exp1 human observation data vs 4 model/baseline predictions.
Each panel shows model predictions (x) vs human mean observe per activation (y)
with bootstrap SD error bars on the y-axis and a line of best fit.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent            # data_processing/scripts/
data_processing_dir = script_dir.parent       # data_processing/
workspace_root = data_processing_dir.parent   # repo root

# ── load model / baseline JSONs ───────────────────────────────────────────
with open(workspace_root / 'steps.dict.json') as f:
    steps_dict = json.load(f)

with open(workspace_root / 'scripts/baselines/exp1/step_dict_naive_exp1.json') as f:
    baseline_naive = json.load(f)

with open(workspace_root / 'scripts/baselines/exp1/step_dict_nonmentalize_exp1.json') as f:
    baseline_nonmentalize = json.load(f)

with open(workspace_root / 'scripts/baselines/exp1/step_dict_mentalize_exp1.json') as f:
    baseline_mentalize = json.load(f)

# ── load human data ───────────────────────────────────────────────────────
results_dict = {}
exec((data_processing_dir / 'results/current/results_dict_exp1_50.py').read_text())
human = results_dict.copy()

# extract participant count from comment
import re
_content = (data_processing_dir / 'results/current/results_dict_exp1_50.py').read_text()
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

# key transforms
main_transform     = lambda k: k.replace('_ascii', '_1')       # mod_s544_ascii -> mod_s544_1
baseline_transform  = lambda k: f'mod_{k}_1'                   # s351 -> mod_s351_1

pairs = {
    'ToM Model':      collect_pairs(steps_dict,            human, main_transform),
    'Naive':           collect_pairs(baseline_naive,        human, baseline_transform),
    'Non-mentalize':   collect_pairs(baseline_nonmentalize, human, baseline_transform),
    'Mentalize':       collect_pairs(baseline_mentalize,    human, baseline_transform),
}

# ── plot (4 columns x 1 row) ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for ax, (label, (x, y, sd, _keys)) in zip(axes, pairs.items()):
    # scatter with error bars
    ax.errorbar(x, y, yerr=sd, fmt='o', alpha=0.7, capsize=3, markersize=6,
                elinewidth=1, color='#1f77b4', ecolor='gray')

    # correlation
    r, p = stats.pearsonr(x, y)
    n = len(x)

    # line of best fit
    if n > 1:
        fit = np.polyfit(x, y, 1)
        fit_fn = np.poly1d(fit)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, fit_fn(x_line), color='red', linewidth=2)

    ax.set_xlabel('Model Predicted Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    n_label = f', N={n_participants}' if n_participants else ''
    ax.set_title(f'{label}{n_label}\nr = {r:.3f}, p = {p:.3f}, n = {n}',
                 fontsize=20)
    ax.grid(False)
    ax.tick_params(labelsize=14)

plt.tight_layout()

output_dir = data_processing_dir / 'outputs/plots'
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / 'correlation_exp1_4panel.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.show()
