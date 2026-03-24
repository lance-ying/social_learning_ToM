import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get script directory and workspace root
script_dir = Path(__file__).parent  # data_processing/scripts/
data_processing_dir = script_dir.parent  # data_processing/
workspace_root = script_dir.parent.parent  # workspace root


def baseline_step_path(exp: str, model: str) -> Path:
    return workspace_root / "scripts" / "baselines" / "outputs" / exp / f"step_dict_{model}.json"

# Load model predictions
with open(workspace_root / 'model_outputs/experiments/exp1/steps_dict.json', 'r') as f:
    steps_dict = json.load(f)

with open(workspace_root / 'model_outputs/experiments/exp2/steps_dict.json', 'r') as f:
    steps_dict_exp2 = json.load(f)

# Load baseline predictions for exp1
with open(baseline_step_path('exp1', 'naive_observer'), 'r') as f:
    baseline_naive_exp1 = json.load(f)

with open(baseline_step_path('exp1', 'social_mentalizing'), 'r') as f:
    baseline_mentalize_exp1 = json.load(f)

with open(baseline_step_path('exp1', 'rational_non_mentalizing'), 'r') as f:
    baseline_nonmentalize_exp1 = json.load(f)

# Load baseline predictions for exp2
with open(baseline_step_path('exp2', 'naive_observer'), 'r') as f:
    baseline_naive_exp2 = json.load(f)

with open(baseline_step_path('exp2', 'social_mentalizing'), 'r') as f:
    baseline_mentalize_exp2 = json.load(f)

with open(baseline_step_path('exp2', 'rational_non_mentalizing'), 'r') as f:
    baseline_nonmentalize_exp2 = json.load(f)

# Load human data from results/current
results_dict = {}  # Initialize for linter
exp1_file_content = (data_processing_dir / 'results/current/results_dict_exp1_50.py').read_text()
exec(exp1_file_content)
exp1_50_dict = results_dict.copy()

# Extract CSV file count from comments
exp1_csv_count = None
import re
for line in exp1_file_content.split('\n'):
    if 'Unique files' in line:
        match = re.search(r'(\d+)', line)
        if match:
            exp1_csv_count = int(match.group(1))
            break
if exp1_csv_count is None:
    # Fallback: count files in exp1_combined directory
    exp1_csv_count = len(list((data_processing_dir / 'data_processed/exp1_combined').glob('*.csv'))) if (data_processing_dir / 'data_processed/exp1_combined').exists() else 0
print("✓ Loaded exp1_50 from results/current/")

results_dict = {}  # Reset for next file
exp2_file_content = (data_processing_dir / 'results/current/results_dict_merged_exp2.py').read_text()
exec(exp2_file_content)
exp2_dict = results_dict.copy()

# Extract CSV file count from comments
exp2_csv_count = None
for line in exp2_file_content.split('\n'):
    if 'Unique files' in line:
        match = re.search(r'(\d+)', line)
        if match:
            exp2_csv_count = int(match.group(1))
            break
if exp2_csv_count is None:
    # Fallback: count files in exp2_combined directory
    exp2_csv_count = len(list((data_processing_dir / 'data_processed/exp2_combined').glob('*.csv'))) if (data_processing_dir / 'data_processed/exp2_combined').exists() else 0
print("✓ Loaded exp2 from results/current/")
print()

# Match and collect data for exp1
arr_model_50 = []
arr_human_50 = []

print("=" * 80)
print("EXP1: steps.dict.json vs exp1_50")
print("=" * 80)

for key in steps_dict:
    human_key = key.replace("_ascii", "_1")
    if human_key in exp1_50_dict:
        model_val = steps_dict[key]
        human_val = exp1_50_dict[human_key]['mean_observe_per_activation']
        arr_model_50.append(model_val)
        arr_human_50.append(human_val)
        print(f"{key}: model={model_val}, human={human_val:.2f}")

# Match and collect data for exp2
arr_model_exp2 = []
arr_human_exp2 = []

print("\n" + "=" * 80)
print("EXP2: steps_exp2.json vs exp2")
print("=" * 80)

for key in steps_dict_exp2:
    if key in exp2_dict:
        model_val = steps_dict_exp2[key]
        human_val = exp2_dict[key]['mean_observe_per_activation']
        arr_model_exp2.append(model_val)
        arr_human_exp2.append(human_val)
        print(f"{key}: model={model_val}, human={human_val:.2f}")

# Convert to numpy arrays
arr_model_50 = np.array(arr_model_50)
arr_human_50 = np.array(arr_human_50)
arr_model_exp2 = np.array(arr_model_exp2)
arr_human_exp2 = np.array(arr_human_exp2)

# Calculate correlations
corr_50 = np.corrcoef(arr_model_50, arr_human_50)[0, 1]
corr_exp2 = np.corrcoef(arr_model_exp2, arr_human_exp2)[0, 1]

# Match and collect data for exp1 baselines
arr_naive_exp1 = []
arr_human_naive_exp1 = []
arr_mentalize_exp1 = []
arr_human_mentalize_exp1 = []
arr_nonmentalize_exp1 = []
arr_human_nonmentalize_exp1 = []

print("\n" + "=" * 80)
print("EXP1 BASELINES vs exp1_50")
print("=" * 80)

# Naive baseline
print("\nNaive baseline:")
for key in baseline_naive_exp1:
    human_key = f"mod_{key}_1"
    if human_key in exp1_50_dict:
        baseline_val = baseline_naive_exp1[key]
        human_val = exp1_50_dict[human_key]['mean_observe_per_activation']
        arr_naive_exp1.append(baseline_val)
        arr_human_naive_exp1.append(human_val)
        print(f"{key}: baseline={baseline_val}, human={human_val:.2f}")

# Mentalize baseline
print("\nMentalize baseline:")
for key in baseline_mentalize_exp1:
    human_key = f"mod_{key}_1"
    if human_key in exp1_50_dict:
        baseline_val = baseline_mentalize_exp1[key]
        human_val = exp1_50_dict[human_key]['mean_observe_per_activation']
        arr_mentalize_exp1.append(baseline_val)
        arr_human_mentalize_exp1.append(human_val)
        print(f"{key}: baseline={baseline_val}, human={human_val:.2f}")

# Non-mentalize baseline
print("\nNon-mentalize baseline:")
for key in baseline_nonmentalize_exp1:
    human_key = f"mod_{key}_1"
    if human_key in exp1_50_dict:
        baseline_val = baseline_nonmentalize_exp1[key]
        human_val = exp1_50_dict[human_key]['mean_observe_per_activation']
        arr_nonmentalize_exp1.append(baseline_val)
        arr_human_nonmentalize_exp1.append(human_val)
        print(f"{key}: baseline={baseline_val}, human={human_val:.2f}")

# Match and collect data for exp2 baselines
arr_naive_exp2 = []
arr_human_naive_exp2 = []
arr_mentalize_exp2 = []
arr_human_mentalize_exp2 = []
arr_nonmentalize_exp2 = []
arr_human_nonmentalize_exp2 = []

print("\n" + "=" * 80)
print("EXP2 BASELINES vs exp2")
print("=" * 80)

# Naive baseline
print("\nNaive baseline:")
for key in baseline_naive_exp2:
    if key in exp2_dict:
        baseline_val = baseline_naive_exp2[key]
        human_val = exp2_dict[key]['mean_observe_per_activation']
        arr_naive_exp2.append(baseline_val)
        arr_human_naive_exp2.append(human_val)
        print(f"{key}: baseline={baseline_val}, human={human_val:.2f}")

# Mentalize baseline
print("\nMentalize baseline:")
for key in baseline_mentalize_exp2:
    if key in exp2_dict:
        baseline_val = baseline_mentalize_exp2[key]
        human_val = exp2_dict[key]['mean_observe_per_activation']
        arr_mentalize_exp2.append(baseline_val)
        arr_human_mentalize_exp2.append(human_val)
        print(f"{key}: baseline={baseline_val}, human={human_val:.2f}")

# Non-mentalize baseline
print("\nNon-mentalize baseline:")
for key in baseline_nonmentalize_exp2:
    if key in exp2_dict:
        baseline_val = baseline_nonmentalize_exp2[key]
        human_val = exp2_dict[key]['mean_observe_per_activation']
        arr_nonmentalize_exp2.append(baseline_val)
        arr_human_nonmentalize_exp2.append(human_val)
        print(f"{key}: baseline={baseline_val}, human={human_val:.2f}")

# Convert to numpy arrays
arr_naive_exp1 = np.array(arr_naive_exp1)
arr_human_naive_exp1 = np.array(arr_human_naive_exp1)
arr_mentalize_exp1 = np.array(arr_mentalize_exp1)
arr_human_mentalize_exp1 = np.array(arr_human_mentalize_exp1)
arr_nonmentalize_exp1 = np.array(arr_nonmentalize_exp1)
arr_human_nonmentalize_exp1 = np.array(arr_human_nonmentalize_exp1)
arr_naive_exp2 = np.array(arr_naive_exp2)
arr_human_naive_exp2 = np.array(arr_human_naive_exp2)
arr_mentalize_exp2 = np.array(arr_mentalize_exp2)
arr_human_mentalize_exp2 = np.array(arr_human_mentalize_exp2)
arr_nonmentalize_exp2 = np.array(arr_nonmentalize_exp2)
arr_human_nonmentalize_exp2 = np.array(arr_human_nonmentalize_exp2)

# Calculate correlations
corr_naive_exp1 = np.corrcoef(arr_naive_exp1, arr_human_naive_exp1)[0, 1] if len(arr_naive_exp1) > 0 else 0
corr_mentalize_exp1 = np.corrcoef(arr_mentalize_exp1, arr_human_mentalize_exp1)[0, 1] if len(arr_mentalize_exp1) > 0 else 0
corr_nonmentalize_exp1 = np.corrcoef(arr_nonmentalize_exp1, arr_human_nonmentalize_exp1)[0, 1] if len(arr_nonmentalize_exp1) > 0 else 0
corr_naive_exp2 = np.corrcoef(arr_naive_exp2, arr_human_naive_exp2)[0, 1] if len(arr_naive_exp2) > 0 else 0
corr_mentalize_exp2 = np.corrcoef(arr_mentalize_exp2, arr_human_mentalize_exp2)[0, 1] if len(arr_mentalize_exp2) > 0 else 0
corr_nonmentalize_exp2 = np.corrcoef(arr_nonmentalize_exp2, arr_human_nonmentalize_exp2)[0, 1] if len(arr_nonmentalize_exp2) > 0 else 0

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)
print(f"EXP1 Main Model: r = {corr_50:.6f}, N = {len(arr_model_50)}")
print(f"EXP1 Naive: r = {corr_naive_exp1:.6f}, N = {len(arr_naive_exp1)}")
print(f"EXP1 Mentalize: r = {corr_mentalize_exp1:.6f}, N = {len(arr_mentalize_exp1)}")
print(f"EXP1 Non-mentalize: r = {corr_nonmentalize_exp1:.6f}, N = {len(arr_nonmentalize_exp1)}")
print(f"EXP2 Main Model: r = {corr_exp2:.6f}, N = {len(arr_model_exp2)}")
print(f"EXP2 Naive: r = {corr_naive_exp2:.6f}, N = {len(arr_naive_exp2)}")
print(f"EXP2 Mentalize: r = {corr_mentalize_exp2:.6f}, N = {len(arr_mentalize_exp2)}")
print(f"EXP2 Non-mentalize: r = {corr_nonmentalize_exp2:.6f}, N = {len(arr_nonmentalize_exp2)}")

# Create plot for Experiment 1
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.scatter(arr_model_50, arr_human_50, alpha=0.7)
ax1.set_xlabel('Model Observe Steps', fontsize=18)
ax1.set_ylabel('Human Observe Steps', fontsize=18)
ax1.set_title(f'Experiment 1, {exp1_csv_count} CSV files\nr = {corr_50:.3f}', fontsize=20)
ax1.grid(False)

# Add line of best fit
fit_50 = np.polyfit(arr_model_50, arr_human_50, 1)
fit_fn_50 = np.poly1d(fit_50)
x_vals_50 = np.linspace(min(arr_model_50), max(arr_model_50), 100)
ax1.plot(x_vals_50, fit_fn_50(x_vals_50), color='red', linestyle='-', linewidth=2)

plt.tight_layout()
output_dir = data_processing_dir / "outputs/plots"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'correlation_exp1.png', dpi=300, bbox_inches='tight')
print(f"\nExperiment 1 plot saved as '{output_dir / 'correlation_exp1.png'}'")

# Create plot for Experiment 2
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.scatter(arr_model_exp2, arr_human_exp2, alpha=0.7)
ax2.set_xlabel('Model Observe Steps', fontsize=18)
ax2.set_ylabel('Human Observe Steps', fontsize=18)
ax2.set_title(f'Experiment 2, {exp2_csv_count} CSV files\nr = {corr_exp2:.3f}', fontsize=20)
ax2.grid(False)

# Add line of best fit
fit_exp2 = np.polyfit(arr_model_exp2, arr_human_exp2, 1)
fit_fn_exp2 = np.poly1d(fit_exp2)
x_vals_exp2 = np.linspace(min(arr_model_exp2), max(arr_model_exp2), 100)
ax2.plot(x_vals_exp2, fit_fn_exp2(x_vals_exp2), color='red', linestyle='-', linewidth=2)

plt.tight_layout()
plt.savefig(output_dir / 'correlation_exp2.png', dpi=300, bbox_inches='tight')
print(f"Experiment 2 plot saved as '{output_dir / 'correlation_exp2.png'}'")

# Create plots for exp1 baselines
# Naive baseline
if len(arr_naive_exp1) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr_naive_exp1, arr_human_naive_exp1, alpha=0.7)
    ax.set_xlabel('Model Observe Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    ax.set_title(f'Experiment 1 - Naive, {exp1_csv_count} CSV files\nr = {corr_naive_exp1:.3f}', fontsize=20)
    ax.grid(False)
    if len(arr_naive_exp1) > 1:
        fit = np.polyfit(arr_naive_exp1, arr_human_naive_exp1, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(min(arr_naive_exp1), max(arr_naive_exp1), 100)
        ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp1_naive.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 1 Naive plot saved as '{output_dir / 'correlation_exp1_naive.png'}'")

# Mentalize baseline
if len(arr_mentalize_exp1) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr_mentalize_exp1, arr_human_mentalize_exp1, alpha=0.7)
    ax.set_xlabel('Model Observe Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    ax.set_title(f'Experiment 1 - Mentalize, {exp1_csv_count} CSV files\nr = {corr_mentalize_exp1:.3f}', fontsize=20)
    ax.grid(False)
    if len(arr_mentalize_exp1) > 1:
        fit = np.polyfit(arr_mentalize_exp1, arr_human_mentalize_exp1, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(min(arr_mentalize_exp1), max(arr_mentalize_exp1), 100)
        ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp1_mentalize.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 1 Mentalize plot saved as '{output_dir / 'correlation_exp1_mentalize.png'}'")

# Non-mentalize baseline
if len(arr_nonmentalize_exp1) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr_nonmentalize_exp1, arr_human_nonmentalize_exp1, alpha=0.7)
    ax.set_xlabel('Model Observe Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    ax.set_title(f'Experiment 1 - Non-mentalize, {exp1_csv_count} CSV files\nr = {corr_nonmentalize_exp1:.3f}', fontsize=20)
    ax.grid(False)
    if len(arr_nonmentalize_exp1) > 1:
        fit = np.polyfit(arr_nonmentalize_exp1, arr_human_nonmentalize_exp1, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(min(arr_nonmentalize_exp1), max(arr_nonmentalize_exp1), 100)
        ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp1_nonmentalize.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 1 Non-mentalize plot saved as '{output_dir / 'correlation_exp1_nonmentalize.png'}'")

# Create plots for exp2 baselines
# Naive baseline
if len(arr_naive_exp2) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr_naive_exp2, arr_human_naive_exp2, alpha=0.7)
    ax.set_xlabel('Model Observe Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    ax.set_title(f'Experiment 2 - Naive, {exp2_csv_count} CSV files\nr = {corr_naive_exp2:.3f}', fontsize=20)
    ax.grid(False)
    if len(arr_naive_exp2) > 1:
        fit = np.polyfit(arr_naive_exp2, arr_human_naive_exp2, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(min(arr_naive_exp2), max(arr_naive_exp2), 100)
        ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp2_naive.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 2 Naive plot saved as '{output_dir / 'correlation_exp2_naive.png'}'")

# Mentalize baseline
if len(arr_mentalize_exp2) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr_mentalize_exp2, arr_human_mentalize_exp2, alpha=0.7)
    ax.set_xlabel('Model Observe Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    ax.set_title(f'Experiment 2 - Mentalize, {exp2_csv_count} CSV files\nr = {corr_mentalize_exp2:.3f}', fontsize=20)
    ax.grid(False)
    if len(arr_mentalize_exp2) > 1:
        fit = np.polyfit(arr_mentalize_exp2, arr_human_mentalize_exp2, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(min(arr_mentalize_exp2), max(arr_mentalize_exp2), 100)
        ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp2_mentalize.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 2 Mentalize plot saved as '{output_dir / 'correlation_exp2_mentalize.png'}'")

# Non-mentalize baseline
if len(arr_nonmentalize_exp2) > 0:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(arr_nonmentalize_exp2, arr_human_nonmentalize_exp2, alpha=0.7)
    ax.set_xlabel('Model Observe Steps', fontsize=18)
    ax.set_ylabel('Human Observe Steps', fontsize=18)
    ax.set_title(f'Experiment 2 - Non-mentalize, {exp2_csv_count} CSV files\nr = {corr_nonmentalize_exp2:.3f}', fontsize=20)
    ax.grid(False)
    if len(arr_nonmentalize_exp2) > 1:
        fit = np.polyfit(arr_nonmentalize_exp2, arr_human_nonmentalize_exp2, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(min(arr_nonmentalize_exp2), max(arr_nonmentalize_exp2), 100)
        ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp2_nonmentalize.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 2 Non-mentalize plot saved as '{output_dir / 'correlation_exp2_nonmentalize.png'}'")

plt.show()
