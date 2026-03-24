"""
Combined 4×4 scatter plot for Exp3 and Exp4, both agents in one figure.
Row order: Exp3 agent2, Exp3 agent3, Exp4 agent2, Exp4 agent3.
"""
import json, re, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from correlation_4panel_style import (
    annotate_r_ci,
    apply_reference_style,
    bootstrap_r_ci,
    plot_points_errorbars_and_fit,
)

script_dir = Path(__file__).parent
data_processing_dir = script_dir.parent
workspace_root = data_processing_dir.parent


def baseline_step_path(exp: str, model: str) -> Path:
    return workspace_root / "scripts" / "baselines" / "outputs" / exp / f"step_dict_{model}.json"

# ── Load Exp3 model/baseline JSONs ─────────────────────────────────────────
with open(workspace_root / 'scripts/experiments/outputs/exp3/steps_dict.json') as f:
    model_exp3 = json.load(f)
with open(baseline_step_path('exp3', 'naive_observer')) as f:
    naive_exp3 = json.load(f)
with open(baseline_step_path('exp3', 'rational_non_mentalizing')) as f:
    nonmentalize_exp3 = json.load(f)
with open(baseline_step_path('exp3', 'social_mentalizing')) as f:
    mentalize_exp3 = json.load(f)

# ── Load Exp4 model/baseline JSONs ─────────────────────────────────────────
with open(workspace_root / 'scripts/experiments/outputs/exp4/steps_dict.json') as f:
    model_exp4 = json.load(f)
with open(baseline_step_path('exp4', 'naive_observer')) as f:
    naive_exp4 = json.load(f)
with open(baseline_step_path('exp4', 'rational_non_mentalizing')) as f:
    nonmentalize_exp4 = json.load(f)
with open(baseline_step_path('exp4', 'social_mentalizing')) as f:
    mentalize_exp4 = json.load(f)

# ── CSV parsing ────────────────────────────────────────────────────────────
LEVEL_RE = re.compile(r'^Level:\s*(.+?)\s*$', re.IGNORECASE)
SKIP_LEVELS = {'comprehension_check', 'experiment', 's111_1'}
SKIP_PREFIXES = ('sm111_', 'sm112_')


def parse_csv(path):
    level_data = defaultdict(lambda: {'agent2_count': 0, 'agent3_count': 0, 'activation_count': 0})
    current_level = None
    in_table = False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue
            m = LEVEL_RE.match(line)
            if m:
                current_level = m.group(1)
                in_table = False
                continue
            if current_level and line.lower().startswith('timestamp,type'):
                in_table = True
                level_data[current_level]['activation_count'] += 1
                continue
            if current_level and in_table:
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = line.split(',')
                if len(row) >= 6:
                    type_field = (row[1] or '').strip().upper()
                    if 'OBSERVE' in type_field:
                        agent_id = (row[5] or '').strip()
                        if agent_id in ('2.0', '2'):
                            level_data[current_level]['agent2_count'] += 1
                        elif agent_id in ('3.0', '3'):
                            level_data[current_level]['agent3_count'] += 1
    return dict(level_data)


def map_level_name(human_level):
    m = re.match(r'^(sm\d+)_(\d+)$', human_level)
    if m:
        return f"{m.group(1)}_scenario{m.group(2)}"
    m = re.match(r'^(sm\d+)_true$', human_level)
    if m:
        return f"{m.group(1)}_scenario1"
    return human_level


def bootstrap_sd(values, n_boot=1000):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    rng = np.random.default_rng(42)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    return float(samples.mean(axis=1).std(ddof=1))


def build_human_stats(csv_dir):
    csv_paths = sorted(Path(csv_dir).glob('*.csv'))
    print(f"  Parsing {len(csv_paths)} CSV files from {Path(csv_dir).name}...")
    per_a2 = defaultdict(list)
    per_a3 = defaultdict(list)
    for p in csv_paths:
        file_data = parse_csv(p)
        for level, counts in file_data.items():
            if level in SKIP_LEVELS or any(level.startswith(pfx) for pfx in SKIP_PREFIXES):
                continue
            model_level = map_level_name(level)
            act = counts['activation_count']
            if act > 0:
                per_a2[model_level].append(counts['agent2_count'] / act)
                per_a3[model_level].append(counts['agent3_count'] / act)
    human_stats = {}
    for level in set(per_a2) | set(per_a3):
        a2 = per_a2.get(level, [])
        a3 = per_a3.get(level, [])
        human_stats[level] = {
            'agent2_mean': float(np.mean(a2)) if a2 else 0.0,
            'agent2_sd':   bootstrap_sd(a2) if a2 else 0.0,
            'agent3_mean': float(np.mean(a3)) if a3 else 0.0,
            'agent3_sd':   bootstrap_sd(a3) if a3 else 0.0,
        }
    return human_stats


human_stats_exp3 = build_human_stats(data_processing_dir / 'data_processed/exp3')
human_stats_exp4 = build_human_stats(data_processing_dir / 'data_processed/exp4')
print(f"  Exp3: {len(human_stats_exp3)} levels, Exp4: {len(human_stats_exp4)} levels")


def collect_pairs(pred_dict, human_stats, agent):
    m_vals, h_means, h_sds, keys = [], [], [], []
    seen = set()
    for model_level in pred_dict:
        if model_level in seen or model_level not in human_stats:
            continue
        m_vals.append(pred_dict[model_level][f'{agent}_count'])
        h_means.append(human_stats[model_level][f'{agent}_mean'])
        h_sds.append(human_stats[model_level][f'{agent}_sd'])
        keys.append(model_level)
        seen.add(model_level)
    return np.array(m_vals), np.array(h_means), np.array(h_sds), keys


# ── plot ───────────────────────────────────────────────────────────────────
output_dir = data_processing_dir / 'outputs/plots'
output_dir.mkdir(parents=True, exist_ok=True)

# Row order: Exp3 agent2, Exp3 agent3, Exp4 agent2, Exp4 agent3
row_configs = [
    ("Exp 3\nagent 2", model_exp3, mentalize_exp3, nonmentalize_exp3, naive_exp3, human_stats_exp3, 'agent2'),
    ("Exp 3\nagent 3", model_exp3, mentalize_exp3, nonmentalize_exp3, naive_exp3, human_stats_exp3, 'agent3'),
    ("Exp 4\nagent 2", model_exp4, mentalize_exp4, nonmentalize_exp4, naive_exp4, human_stats_exp4, 'agent2'),
    ("Exp 4\nagent 3", model_exp4, mentalize_exp4, nonmentalize_exp4, naive_exp4, human_stats_exp4, 'agent3'),
]

fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle("Experiments 3 & 4", fontsize=30, color="#1a1a1a", y=1.01)

for row, (row_label, model, mentalize, nonmentalize, naive, human_stats, agent) in enumerate(row_configs):
    pairs = {
        'Rational Mentalizing\n(Full Model)': collect_pairs(model,        human_stats, agent),
        'Social Mentalizing':                 collect_pairs(mentalize,     human_stats, agent),
        'Rational Non-Mentalizing':           collect_pairs(nonmentalize,  human_stats, agent),
        'Naive Observer':                     collect_pairs(naive,         human_stats, agent),
    }

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
        if row == 3:
            ax.set_xlabel(label, fontsize=22, color="#1a1a1a")
        else:
            ax.set_xlabel('')

        # Row label on leftmost column only
        if idx == 0:
            ax.set_ylabel(f'{row_label}\nHuman', fontsize=20, color="#1a1a1a")
        else:
            ax.set_ylabel('')

plt.tight_layout(w_pad=2.5, h_pad=3.5)
out_path = output_dir / 'correlation_exp34_combined.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")
plt.close()
