"""
Mega correlation figure for Experiments 1-4.

Rows:
1. Experiment 1
2. Experiment 2
3. Experiment 3 combined across agent2 and agent3
4. Experiment 4 agent2
5. Experiment 4 agent3
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from correlation_4panel_style import (
    annotate_r_ci,
    apply_reference_style,
    bootstrap_r_ci,
    plot_points_errorbars_and_fit,
)
from model_name_resolution import baseline_step_path


script_dir = Path(__file__).parent
data_processing_dir = script_dir.parent
workspace_root = data_processing_dir.parent


LABELS = [
    "Rational Mentalizing\n(Full Model)",
    "Social Mentalizing",
    "Rational Non-Mentalizing",
    "Naive Observer",
]

LEVEL_RE = re.compile(r"^Level:\s*(.+?)\s*$", re.IGNORECASE)
SKIP_LEVELS = {"comprehension_check", "experiment", "s111_1"}
SKIP_PREFIXES = ("sm111_", "sm112_")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
def bootstrap_sd(values, n_boot=1000):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    rng = np.random.default_rng(42)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    return float(samples.mean(axis=1).std(ddof=1))


def collect_pairs_scalar(model_dict, human_dict, key_transform):
    model_vals, human_means, human_sds, keys = [], [], [], []
    for model_key in model_dict:
        human_key = key_transform(model_key)
        if human_key in human_dict:
            model_vals.append(model_dict[model_key])
            human_means.append(human_dict[human_key]["mean_observe_per_activation"])
            human_sds.append(human_dict[human_key]["bootstrap_sd"])
            keys.append(human_key)
    return np.array(model_vals), np.array(human_means), np.array(human_sds), keys


def collect_pairs_agent(pred_dict, human_stats, stat_key):
    model_vals, human_means, human_sds, keys = [], [], [], []
    seen = set()
    for model_level in pred_dict:
        if model_level in seen or model_level not in human_stats:
            continue
        model_vals.append(pred_dict[model_level][f"{stat_key}_count"])
        human_means.append(human_stats[model_level][f"{stat_key}_mean"])
        human_sds.append(human_stats[model_level][f"{stat_key}_sd"])
        keys.append(model_level)
        seen.add(model_level)
    return np.array(model_vals), np.array(human_means), np.array(human_sds), keys


def collect_pairs_exp3_combined(pred_dict, human_stats):
    model_vals, human_means, human_sds, keys = [], [], [], []
    seen = set()
    for model_level in pred_dict:
        if model_level in seen or model_level not in human_stats:
            continue
        pred = pred_dict[model_level]
        model_vals.append(pred.get("agent2_count", 0) + pred.get("agent3_count", 0))
        human_means.append(human_stats[model_level]["combined_mean"])
        human_sds.append(human_stats[model_level]["combined_sd"])
        keys.append(model_level)
        seen.add(model_level)
    return np.array(model_vals), np.array(human_means), np.array(human_sds), keys


def map_level_name(human_level):
    match = re.match(r"^(sm\d+)_(\d+)$", human_level)
    if match:
        return f"{match.group(1)}_scenario{match.group(2)}"
    match = re.match(r"^(sm\d+)_true$", human_level)
    if match:
        return f"{match.group(1)}_scenario1"
    return human_level


def parse_multi_agent_csv(path):
    level_data = defaultdict(
        lambda: {"agent2_count": 0, "agent3_count": 0, "activation_count": 0}
    )
    current_level = None
    in_table = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                in_table = False
                continue

            match = LEVEL_RE.match(line)
            if match:
                current_level = match.group(1)
                in_table = False
                continue

            if current_level and line.lower().startswith("timestamp,type"):
                in_table = True
                level_data[current_level]["activation_count"] += 1
                continue

            if current_level and in_table:
                try:
                    row = next(csv.reader([line]))
                except Exception:
                    row = line.split(",")

                if len(row) >= 6:
                    type_field = (row[1] or "").strip().upper()
                    if "OBSERVE" in type_field:
                        agent_id = (row[5] or "").strip()
                        if agent_id in ("2.0", "2"):
                            level_data[current_level]["agent2_count"] += 1
                        elif agent_id in ("3.0", "3"):
                            level_data[current_level]["agent3_count"] += 1

    return dict(level_data)


def build_multi_agent_human_stats(csv_dir):
    csv_paths = sorted(Path(csv_dir).glob("*.csv"))
    per_agent2 = defaultdict(list)
    per_agent3 = defaultdict(list)
    per_combined = defaultdict(list)

    for path in csv_paths:
        file_data = parse_multi_agent_csv(path)
        for level, counts in file_data.items():
            if level in SKIP_LEVELS or any(level.startswith(prefix) for prefix in SKIP_PREFIXES):
                continue
            model_level = map_level_name(level)
            activations = counts["activation_count"]
            if activations <= 0:
                continue

            agent2_mean = counts["agent2_count"] / activations
            agent3_mean = counts["agent3_count"] / activations
            combined_mean = (counts["agent2_count"] + counts["agent3_count"]) / activations

            per_agent2[model_level].append(agent2_mean)
            per_agent3[model_level].append(agent3_mean)
            per_combined[model_level].append(combined_mean)

    human_stats = {}
    for level in set(per_agent2) | set(per_agent3) | set(per_combined):
        agent2_vals = per_agent2.get(level, [])
        agent3_vals = per_agent3.get(level, [])
        combined_vals = per_combined.get(level, [])
        human_stats[level] = {
            "agent2_mean": float(np.mean(agent2_vals)) if agent2_vals else 0.0,
            "agent2_sd": bootstrap_sd(agent2_vals) if agent2_vals else 0.0,
            "agent3_mean": float(np.mean(agent3_vals)) if agent3_vals else 0.0,
            "agent3_sd": bootstrap_sd(agent3_vals) if agent3_vals else 0.0,
            "combined_mean": float(np.mean(combined_vals)) if combined_vals else 0.0,
            "combined_sd": bootstrap_sd(combined_vals) if combined_vals else 0.0,
        }
    return human_stats


def plot_row(ax_row, row_index, row_label, pairs, bottom_row_index):
    for col_index, (ax, (label, (x, y, sd, _keys))) in enumerate(zip(ax_row, pairs.items())):
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
        else:
            plot_points_errorbars_and_fit(ax, x, y, sd)
            r, ci_low, ci_high = bootstrap_r_ci(x, y, n_resamples=1000)
            annotate_r_ci(ax, r, ci_low, ci_high)

        if row_index == bottom_row_index:
            ax.set_xlabel(label, fontsize=28, color="#1a1a1a")
        else:
            ax.set_xlabel("")

        ax.set_ylabel("")


# Exp1
model_exp1 = load_json(workspace_root / "model_outputs/experiments/exp1/steps_dict.json")
naive_exp1 = load_json(baseline_step_path(workspace_root, "exp1", "naive_observer"))
nonmentalize_exp1 = load_json(baseline_step_path(workspace_root, "exp1", "rational_non_mentalizing"))
mentalize_exp1 = load_json(baseline_step_path(workspace_root, "exp1", "social_mentalizing"))
results_dict = {}
exec((data_processing_dir / "results/current/results_dict_exp1_50.py").read_text())
human_exp1 = results_dict.copy()

main_transform_exp1 = lambda key: key.replace("_ascii", "_1")
baseline_transform_exp1 = lambda key: f"mod_{key}_1"

pairs_exp1 = {
    LABELS[0]: collect_pairs_scalar(model_exp1, human_exp1, main_transform_exp1),
    LABELS[1]: collect_pairs_scalar(mentalize_exp1, human_exp1, baseline_transform_exp1),
    LABELS[2]: collect_pairs_scalar(nonmentalize_exp1, human_exp1, baseline_transform_exp1),
    LABELS[3]: collect_pairs_scalar(naive_exp1, human_exp1, baseline_transform_exp1),
}

# Exp2
model_exp2 = load_json(workspace_root / "model_outputs/experiments/exp2/steps_dict.json")
naive_exp2 = load_json(baseline_step_path(workspace_root, "exp2", "naive_observer"))
nonmentalize_exp2 = load_json(baseline_step_path(workspace_root, "exp2", "rational_non_mentalizing"))
mentalize_exp2 = load_json(baseline_step_path(workspace_root, "exp2", "social_mentalizing"))
results_dict = {}
exec((data_processing_dir / "results/current/results_dict_merged_exp2.py").read_text())
human_exp2 = results_dict.copy()

pairs_exp2 = {
    LABELS[0]: collect_pairs_scalar(model_exp2, human_exp2, lambda key: key),
    LABELS[1]: collect_pairs_scalar(mentalize_exp2, human_exp2, lambda key: key),
    LABELS[2]: collect_pairs_scalar(nonmentalize_exp2, human_exp2, lambda key: key),
    LABELS[3]: collect_pairs_scalar(naive_exp2, human_exp2, lambda key: key),
}

# Exp3
model_exp3 = load_json(workspace_root / "model_outputs/experiments/exp3/steps_dict.json")
naive_exp3 = load_json(baseline_step_path(workspace_root, "exp3", "naive_observer"))
nonmentalize_exp3 = load_json(baseline_step_path(workspace_root, "exp3", "rational_non_mentalizing"))
mentalize_exp3 = load_json(baseline_step_path(workspace_root, "exp3", "social_mentalizing"))
human_stats_exp3 = build_multi_agent_human_stats(data_processing_dir / "data_processed/exp3")

pairs_exp3_combined = {
    LABELS[0]: collect_pairs_exp3_combined(model_exp3, human_stats_exp3),
    LABELS[1]: collect_pairs_exp3_combined(mentalize_exp3, human_stats_exp3),
    LABELS[2]: collect_pairs_exp3_combined(nonmentalize_exp3, human_stats_exp3),
    LABELS[3]: collect_pairs_exp3_combined(naive_exp3, human_stats_exp3),
}

# Exp4
model_exp4 = load_json(workspace_root / "model_outputs/experiments/exp4/steps_dict.json")
naive_exp4 = load_json(baseline_step_path(workspace_root, "exp4", "naive_observer"))
nonmentalize_exp4 = load_json(baseline_step_path(workspace_root, "exp4", "rational_non_mentalizing"))
mentalize_exp4 = load_json(baseline_step_path(workspace_root, "exp4", "social_mentalizing"))
human_stats_exp4 = build_multi_agent_human_stats(data_processing_dir / "data_processed/exp4")

pairs_exp4_agent2 = {
    LABELS[0]: collect_pairs_agent(model_exp4, human_stats_exp4, "agent2"),
    LABELS[1]: collect_pairs_agent(mentalize_exp4, human_stats_exp4, "agent2"),
    LABELS[2]: collect_pairs_agent(nonmentalize_exp4, human_stats_exp4, "agent2"),
    LABELS[3]: collect_pairs_agent(naive_exp4, human_stats_exp4, "agent2"),
}

pairs_exp4_agent3 = {
    LABELS[0]: collect_pairs_agent(model_exp4, human_stats_exp4, "agent3"),
    LABELS[1]: collect_pairs_agent(mentalize_exp4, human_stats_exp4, "agent3"),
    LABELS[2]: collect_pairs_agent(nonmentalize_exp4, human_stats_exp4, "agent3"),
    LABELS[3]: collect_pairs_agent(naive_exp4, human_stats_exp4, "agent3"),
}


fig, axes = plt.subplots(5, 4, figsize=(20, 24))

row_configs = [
    ("Experiment 1", pairs_exp1),
    ("Experiment 2", pairs_exp2),
    ("Experiment 3\ncombined", pairs_exp3_combined),
    ("Experiment 4\nagent 2", pairs_exp4_agent2),
    ("Experiment 4\nagent 3", pairs_exp4_agent3),
]

for row_index, (row_label, pairs) in enumerate(row_configs):
    plot_row(axes[row_index], row_index, row_label, pairs, bottom_row_index=len(row_configs) - 1)

plt.tight_layout(w_pad=2.5, h_pad=3.5)

output_dir = data_processing_dir / "outputs/plots"
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / "correlation_exp1234_mega.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved -> {out_path}")
plt.show()
