#!/usr/bin/env python3
"""
Correlation analysis for EXP3: Compare model predictions (steps_dict_exp3_optimized_good.json)
with human data from test_output CSV files.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Load model predictions
with open('../results/dictionaries/steps_dict_exp3_optimized_good.json', 'r') as f:
    model_dict = json.load(f)

# Load human data from agent_observes_results
with open('../data_raw/agent_observes_results.json', 'r') as f:
    human_dict = json.load(f)

print("=" * 80)
print("EXP3: Model Predictions vs Human Data")
print("=" * 80)
print(f"Model levels: {len(model_dict)}")
print(f"Human levels: {len(human_dict)}")
print()

# Function to map human level names to model level names
def map_level_name(human_level: str) -> str:
    """
    Map human level names (e.g., "sm111_1", "sm541_true") to model level names (e.g., "sm111_scenario1", "sm541_scenario1")
    """
    # Pattern: smXXX_1 -> smXXX_scenario1, smXXX_2 -> smXXX_scenario2
    match = re.match(r'^(sm\d+)_(\d+)$', human_level)
    if match:
        base = match.group(1)
        scenario_num = match.group(2)
        return f"{base}_scenario{scenario_num}"
    
    # Pattern: smXXX_true -> smXXX_scenario1 (assume "true" means scenario1)
    match = re.match(r'^(sm\d+)_true$', human_level)
    if match:
        base = match.group(1)
        return f"{base}_scenario1"
    
    # If no match, return as-is (might already be in correct format)
    return human_level

# Collect matching data for agent2 and agent3 separately
arr_model_agent2 = []
arr_human_agent2 = []
arr_model_agent3 = []
arr_human_agent3 = []
matched_levels = []
seen_model_levels = set()  # Track to avoid duplicates

print("Matching levels:")
for human_level, human_data in sorted(human_dict.items()):
    # Skip non-game levels
    if human_level in ["comprehension_check", "experiment", "s111_1"]:
        continue
    
    # Skip sm111 and sm112 levels
    if human_level.startswith("sm111_") or human_level.startswith("sm112_"):
        print(f"  ⚠ {human_level} (skipped: sm111/sm112)")
        continue
    
    model_level = map_level_name(human_level)
    
    # Skip if we've already matched this model level (avoid duplicates)
    if model_level in seen_model_levels:
        print(f"  ⚠ {human_level} (duplicate, skipping)")
        continue
    
    if model_level in model_dict:
        model_agent2 = model_dict[model_level]["agent2_count"]
        model_agent3 = model_dict[model_level]["agent3_count"]
        
        # Use mean per activation for human data
        human_total_agent2 = human_data.get("agent2_count", 0)
        human_total_agent3 = human_data.get("agent3_count", 0)
        human_activations = human_data.get("activation_count", 1)
        human_mean_agent2 = human_total_agent2 / human_activations if human_activations > 0 else 0.0
        human_mean_agent3 = human_total_agent3 / human_activations if human_activations > 0 else 0.0
        
        arr_model_agent2.append(model_agent2)
        arr_human_agent2.append(human_mean_agent2)
        arr_model_agent3.append(model_agent3)
        arr_human_agent3.append(human_mean_agent3)
        matched_levels.append((human_level, model_level))
        seen_model_levels.add(model_level)
        print(f"  {human_level}: agent2: model={model_agent2}, human={human_mean_agent2:.2f} | agent3: model={model_agent3}, human={human_mean_agent3:.2f}")
    else:
        print(f"  ⚠ {human_level} (not found in model)")

print(f"\nMatched {len(arr_model_agent2)} levels")

# Convert to numpy arrays
arr_model_agent2 = np.array(arr_model_agent2)
arr_human_agent2 = np.array(arr_human_agent2)
arr_model_agent3 = np.array(arr_model_agent3)
arr_human_agent3 = np.array(arr_human_agent3)

# Calculate correlations
if len(arr_model_agent2) > 0:
    corr_agent2 = np.corrcoef(arr_model_agent2, arr_human_agent2)[0, 1]
    corr_agent3 = np.corrcoef(arr_model_agent3, arr_human_agent3)[0, 1]
    
    print("\n" + "=" * 80)
    print("CORRELATIONS")
    print("=" * 80)
    print(f"Agent 2: r = {corr_agent2:.6f}, N = {len(arr_model_agent2)}")
    print(f"Agent 3: r = {corr_agent3:.6f}, N = {len(arr_model_agent3)}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Agent 2 plot
    ax1.scatter(arr_model_agent2, arr_human_agent2, alpha=0.7, s=100, color='blue')
    ax1.set_xlabel('Model Agent 2 Count', fontsize=14)
    ax1.set_ylabel('Human Mean Agent 2 per Activation', fontsize=14)
    ax1.set_title(f'Agent 2: r = {corr_agent2:.3f}, N = {len(arr_model_agent2)}', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    if len(arr_model_agent2) > 1:
        fit2 = np.polyfit(arr_model_agent2, arr_human_agent2, 1)
        fit_fn2 = np.poly1d(fit2)
        x_vals2 = np.linspace(min(arr_model_agent2), max(arr_model_agent2), 100)
        ax1.plot(x_vals2, fit_fn2(x_vals2), color='red', linestyle='--', linewidth=2, label='Best fit')
        ax1.legend()
    
    # Agent 3 plot
    ax2.scatter(arr_model_agent3, arr_human_agent3, alpha=0.7, s=100, color='green')
    ax2.set_xlabel('Model Agent 3 Count', fontsize=14)
    ax2.set_ylabel('Human Mean Agent 3 per Activation', fontsize=14)
    ax2.set_title(f'Agent 3: r = {corr_agent3:.3f}, N = {len(arr_model_agent3)}', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    if len(arr_model_agent3) > 1:
        fit3 = np.polyfit(arr_model_agent3, arr_human_agent3, 1)
        fit_fn3 = np.poly1d(fit3)
        x_vals3 = np.linspace(min(arr_model_agent3), max(arr_model_agent3), 100)
        ax2.plot(x_vals3, fit_fn3(x_vals3), color='red', linestyle='--', linewidth=2, label='Best fit')
        ax2.legend()
    
    plt.tight_layout()
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'correlation_exp3_agents.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as '{output_dir / 'correlation_exp3_agents.png'}'")
    
    plt.show()
else:
    print("\n⚠ No matching levels found for correlation analysis")

