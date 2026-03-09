"""
Two 4x1 scatter subplots for Exp3: one for agent2, one for agent3.
Each figure has 4 panels: Rational-Mentalizing, Naive, Non-mentalize, Mentalize.
Human data is parsed from individual CSVs to get per-participant SD for error bars.
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

# ── paths ──────────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent
data_processing_dir = script_dir.parent
workspace_root = data_processing_dir.parent

# ── load model / baseline JSONs ───────────────────────────────────────────
with open(workspace_root / 'steps_dict_exp3.json') as f:
    model_dict = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_naive_exp3.json') as f:
    baseline_naive = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_nonmentalize_exp3_v2.json') as f:
    baseline_nonmentalize = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_mentalize_exp3.json') as f:
    baseline_mentalize = json.load(f)

# ── parse per-participant agent2/agent3 counts from CSVs ──────────────────
LEVEL_RE = re.compile(r'^Level:\s*(.+?)\s*$', re.IGNORECASE)

def parse_exp3_csv(path):
    """Parse one CSV, return dict: level -> {agent2_count, agent3_count, activation_count}"""
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
    """sm111_1 -> sm111_scenario1"""
    m = re.match(r'^(sm\d+)_(\d+)$', human_level)
    if m:
        return f"{m.group(1)}_scenario{m.group(2)}"
    m = re.match(r'^(sm\d+)_true$', human_level)
    if m:
        return f"{m.group(1)}_scenario1"
    return human_level


# Parse all CSVs
csv_dir = data_processing_dir / 'data_processed/exp3'
csv_paths = sorted(csv_dir.glob('*.csv'))
n_participants = len(csv_paths)
print(f"Parsing {n_participants} CSV files...")

# per_participant[level_scenario] = list of (agent2_mean, agent3_mean) per participant
per_participant_agent2 = defaultdict(list)
per_participant_agent3 = defaultdict(list)

for p in csv_paths:
    file_data = parse_exp3_csv(p)
    for level, counts in file_data.items():
        # Skip non-game levels
        if level in ('comprehension_check', 'experiment', 's111_1'):
            continue
        if level.startswith('sm111_') or level.startswith('sm112_'):
            continue

        model_level = map_level_name(level)
        act = counts['activation_count']
        if act > 0:
            per_participant_agent2[model_level].append(counts['agent2_count'] / act)
            per_participant_agent3[model_level].append(counts['agent3_count'] / act)

# Compute mean and bootstrap SD per level
def bootstrap_sd(values, n_boot=1000):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    rng = np.random.default_rng(42)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    boot_means = samples.mean(axis=1)
    return float(boot_means.std(ddof=1))

human_stats = {}  # level -> {agent2_mean, agent2_sd, agent3_mean, agent3_sd}
for level in set(per_participant_agent2.keys()) | set(per_participant_agent3.keys()):
    a2 = per_participant_agent2.get(level, [])
    a3 = per_participant_agent3.get(level, [])
    human_stats[level] = {
        'agent2_mean': np.mean(a2) if a2 else 0.0,
        'agent2_sd': bootstrap_sd(a2) if a2 else 0.0,
        'agent3_mean': np.mean(a3) if a3 else 0.0,
        'agent3_sd': bootstrap_sd(a3) if a3 else 0.0,
    }

print(f"Computed stats for {len(human_stats)} levels")

# ── helper: collect matched arrays for a given agent ──────────────────────
def collect_pairs(pred_dict, human_stats, agent):
    """Return model_vals, human_means, human_sds, keys for one agent."""
    m_vals, h_means, h_sds, keys = [], [], [], []
    seen = set()
    for model_level in pred_dict:
        if model_level in seen:
            continue
        if model_level in human_stats:
            count_key = f'{agent}_count'
            m_vals.append(pred_dict[model_level][count_key])
            h_means.append(human_stats[model_level][f'{agent}_mean'])
            h_sds.append(human_stats[model_level][f'{agent}_sd'])
            keys.append(model_level)
            seen.add(model_level)
    return np.array(m_vals), np.array(h_means), np.array(h_sds), keys


# ── build pairs for each agent ────────────────────────────────────────────
output_dir = data_processing_dir / 'outputs/plots'
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("Experiment 3", fontsize=30, color="#1a1a1a", y=1.01)

for row, agent in enumerate(('agent2', 'agent3')):
    agent_label = "agent 2" if agent == "agent2" else "agent 3"
    pairs = {
        'Rational Mentalizing\n(Full Model)': collect_pairs(model_dict,            human_stats, agent),
        'Social Mentalizing':                 collect_pairs(baseline_mentalize,     human_stats, agent),
        'Rational Non-Mentalizing':           collect_pairs(baseline_nonmentalize,  human_stats, agent),
        'Naive Observer':                     collect_pairs(baseline_naive,         human_stats, agent),
    }

    for idx, (ax, (label, (x, y, sd, _keys))) in enumerate(zip(axes[row], pairs.items())):
        apply_reference_style(ax)

        if len(x) < 3:
            ax.text(0.05, 0.90, "Insufficient data", transform=ax.transAxes,
                    ha="left", va="top", fontsize=18, color="#1a1a1a")
            continue

        plot_points_errorbars_and_fit(ax, x, y, sd)
        r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
        annotate_r_ci(ax, r, ci_low, ci_high)

        if row == 1:
            ax.set_xlabel(label, fontsize=22, color="#1a1a1a")
        else:
            ax.set_xlabel('')

        if idx == 0:
            ax.set_ylabel(f'{agent_label}\nHuman', fontsize=20, color="#1a1a1a")
        else:
            ax.set_ylabel('')

plt.tight_layout(w_pad=2.5, h_pad=3.5)
out_path = output_dir / 'correlation_exp3_4panel.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved -> {out_path}")
plt.show()
