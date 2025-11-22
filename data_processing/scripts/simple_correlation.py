import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load model predictions
with open('data_raw/steps.dict.json', 'r') as f:
    steps_dict = json.load(f)

with open('data_raw/steps_exp2.json', 'r') as f:
    steps_dict_exp2 = json.load(f)

# Load human data from results/current
results_dict = {}  # Initialize for linter
exp1_file_content = Path('results/current/results_dict_exp1_50.py').read_text()
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
    exp1_csv_count = len(list(Path('data_processed/exp1_combined').glob('*.csv'))) if Path('data_processed/exp1_combined').exists() else 0
print("✓ Loaded exp1_50 from results/current/")

results_dict = {}  # Reset for next file
exp2_file_content = Path('results/current/results_dict_merged_exp2.py').read_text()
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
    exp2_csv_count = len(list(Path('data_processed/exp2_combined').glob('*.csv'))) if Path('data_processed/exp2_combined').exists() else 0
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

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)
print(f"EXP1: r = {corr_50:.6f}, N = {len(arr_model_50)}")
print(f"EXP2: r = {corr_exp2:.6f}, N = {len(arr_model_exp2)}")

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
output_dir = Path("outputs/plots")
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

plt.show()
