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

# Get script directory and workspace root
script_dir = Path(__file__).parent  # data_processing/scripts/
data_processing_dir = script_dir.parent  # data_processing/
workspace_root = script_dir.parent.parent  # workspace root

# Load model predictions
with open(data_processing_dir / 'results/dictionaries/steps_dict_exp3_optimized_good.json', 'r') as f:
    model_dict = json.load(f)

# Load human data from agent_observes_results
with open(data_processing_dir / 'data_raw/agent_observes_results.json', 'r') as f:
    human_dict = json.load(f)

# Load baseline predictions for exp3
with open(workspace_root / 'scripts/baselines/exp3/step_dict_naive_exp3.json', 'r') as f:
    baseline_naive_exp3 = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_mentalize_exp3.json', 'r') as f:
    baseline_mentalize_exp3 = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_nonmentalize_exp3.json', 'r') as f:
    baseline_nonmentalize_exp3 = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_naive_distance_heur_exp3.json', 'r') as f:
    baseline_naive_distance_heur_exp3 = json.load(f)

with open(workspace_root / 'scripts/baselines/exp3/step_dict_nonmentalize_distance_heur_exp3.json', 'r') as f:
    baseline_nonmentalize_distance_heur_exp3 = json.load(f)

# Count CSV files for exp3
exp3_csv_count = len(list((data_processing_dir / 'data_processed/exp3').glob('*.csv'))) if (data_processing_dir / 'data_processed/exp3').exists() else 0

print("=" * 80)
print("EXP3: Model Predictions vs Human Data")
print("=" * 80)
print(f"Model levels: {len(model_dict)}")
print(f"Human levels: {len(human_dict)}")
print(f"CSV files: {exp3_csv_count}")
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
    print("CORRELATIONS - Main Model")
    print("=" * 80)
    print(f"Agent 2: r = {corr_agent2:.6f}, N = {len(arr_model_agent2)}")
    print(f"Agent 3: r = {corr_agent3:.6f}, N = {len(arr_model_agent3)}")
    
    # Helper function to match baseline data with human data
    def match_baseline_with_human(baseline_dict, human_dict, map_level_name_func):
        """Match baseline predictions with human data for both agent2 and agent3"""
        arr_baseline_agent2 = []
        arr_human_agent2 = []
        arr_baseline_agent3 = []
        arr_human_agent3 = []
        seen_levels = set()
        
        for human_level, human_data in sorted(human_dict.items()):
            # Skip non-game levels
            if human_level in ["comprehension_check", "experiment", "s111_1"]:
                continue
            
            # Skip sm111 and sm112 levels
            if human_level.startswith("sm111_") or human_level.startswith("sm112_"):
                continue
            
            model_level = map_level_name_func(human_level)
            
            # Skip if we've already matched this model level (avoid duplicates)
            if model_level in seen_levels:
                continue
            
            if model_level in baseline_dict:
                baseline_agent2 = baseline_dict[model_level].get("agent2_count", 0)
                baseline_agent3 = baseline_dict[model_level].get("agent3_count", 0)
                
                # Use mean per activation for human data
                human_total_agent2 = human_data.get("agent2_count", 0)
                human_total_agent3 = human_data.get("agent3_count", 0)
                human_activations = human_data.get("activation_count", 1)
                human_mean_agent2 = human_total_agent2 / human_activations if human_activations > 0 else 0.0
                human_mean_agent3 = human_total_agent3 / human_activations if human_activations > 0 else 0.0
                
                arr_baseline_agent2.append(baseline_agent2)
                arr_human_agent2.append(human_mean_agent2)
                arr_baseline_agent3.append(baseline_agent3)
                arr_human_agent3.append(human_mean_agent3)
                seen_levels.add(model_level)
        
        return (np.array(arr_baseline_agent2), np.array(arr_human_agent2),
                np.array(arr_baseline_agent3), np.array(arr_human_agent3))
    
    # Match and collect data for exp3 baselines
    print("\n" + "=" * 80)
    print("EXP3 BASELINES vs Human Data")
    print("=" * 80)
    
    # Naive baseline
    arr_naive_agent2, arr_human_naive_agent2, arr_naive_agent3, arr_human_naive_agent3 = \
        match_baseline_with_human(baseline_naive_exp3, human_dict, map_level_name)
    
    # Mentalize baseline
    arr_mentalize_agent2, arr_human_mentalize_agent2, arr_mentalize_agent3, arr_human_mentalize_agent3 = \
        match_baseline_with_human(baseline_mentalize_exp3, human_dict, map_level_name)
    
    # Non-mentalize baseline
    arr_nonmentalize_agent2, arr_human_nonmentalize_agent2, arr_nonmentalize_agent3, arr_human_nonmentalize_agent3 = \
        match_baseline_with_human(baseline_nonmentalize_exp3, human_dict, map_level_name)
    
    # Naive distance heur baseline
    arr_naive_dist_agent2, arr_human_naive_dist_agent2, arr_naive_dist_agent3, arr_human_naive_dist_agent3 = \
        match_baseline_with_human(baseline_naive_distance_heur_exp3, human_dict, map_level_name)
    
    # Non-mentalize distance heur baseline
    arr_nonmentalize_dist_agent2, arr_human_nonmentalize_dist_agent2, arr_nonmentalize_dist_agent3, arr_human_nonmentalize_dist_agent3 = \
        match_baseline_with_human(baseline_nonmentalize_distance_heur_exp3, human_dict, map_level_name)
    
    # Calculate correlations for baselines
    corr_naive_agent2 = np.corrcoef(arr_naive_agent2, arr_human_naive_agent2)[0, 1] if len(arr_naive_agent2) > 0 else 0
    corr_naive_agent3 = np.corrcoef(arr_naive_agent3, arr_human_naive_agent3)[0, 1] if len(arr_naive_agent3) > 0 else 0
    corr_mentalize_agent2 = np.corrcoef(arr_mentalize_agent2, arr_human_mentalize_agent2)[0, 1] if len(arr_mentalize_agent2) > 0 else 0
    corr_mentalize_agent3 = np.corrcoef(arr_mentalize_agent3, arr_human_mentalize_agent3)[0, 1] if len(arr_mentalize_agent3) > 0 else 0
    corr_nonmentalize_agent2 = np.corrcoef(arr_nonmentalize_agent2, arr_human_nonmentalize_agent2)[0, 1] if len(arr_nonmentalize_agent2) > 0 else 0
    corr_nonmentalize_agent3 = np.corrcoef(arr_nonmentalize_agent3, arr_human_nonmentalize_agent3)[0, 1] if len(arr_nonmentalize_agent3) > 0 else 0
    corr_naive_dist_agent2 = np.corrcoef(arr_naive_dist_agent2, arr_human_naive_dist_agent2)[0, 1] if len(arr_naive_dist_agent2) > 0 else 0
    corr_naive_dist_agent3 = np.corrcoef(arr_naive_dist_agent3, arr_human_naive_dist_agent3)[0, 1] if len(arr_naive_dist_agent3) > 0 else 0
    corr_nonmentalize_dist_agent2 = np.corrcoef(arr_nonmentalize_dist_agent2, arr_human_nonmentalize_dist_agent2)[0, 1] if len(arr_nonmentalize_dist_agent2) > 0 else 0
    corr_nonmentalize_dist_agent3 = np.corrcoef(arr_nonmentalize_dist_agent3, arr_human_nonmentalize_dist_agent3)[0, 1] if len(arr_nonmentalize_dist_agent3) > 0 else 0
    
    print("\n" + "=" * 80)
    print("CORRELATIONS - Baselines")
    print("=" * 80)
    print(f"Naive - Agent 2: r = {corr_naive_agent2:.6f}, N = {len(arr_naive_agent2)}")
    print(f"Naive - Agent 3: r = {corr_naive_agent3:.6f}, N = {len(arr_naive_agent3)}")
    print(f"Mentalize - Agent 2: r = {corr_mentalize_agent2:.6f}, N = {len(arr_mentalize_agent2)}")
    print(f"Mentalize - Agent 3: r = {corr_mentalize_agent3:.6f}, N = {len(arr_mentalize_agent3)}")
    print(f"Non-mentalize - Agent 2: r = {corr_nonmentalize_agent2:.6f}, N = {len(arr_nonmentalize_agent2)}")
    print(f"Non-mentalize - Agent 3: r = {corr_nonmentalize_agent3:.6f}, N = {len(arr_nonmentalize_agent3)}")
    print(f"Naive Distance Heur - Agent 2: r = {corr_naive_dist_agent2:.6f}, N = {len(arr_naive_dist_agent2)}")
    print(f"Naive Distance Heur - Agent 3: r = {corr_naive_dist_agent3:.6f}, N = {len(arr_naive_dist_agent3)}")
    print(f"Non-mentalize Distance Heur - Agent 2: r = {corr_nonmentalize_dist_agent2:.6f}, N = {len(arr_nonmentalize_dist_agent2)}")
    print(f"Non-mentalize Distance Heur - Agent 3: r = {corr_nonmentalize_dist_agent3:.6f}, N = {len(arr_nonmentalize_dist_agent3)}")
    
    output_dir = data_processing_dir / "outputs/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot for Agent 2 (matching exp1 format)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(arr_model_agent2, arr_human_agent2, alpha=0.7)
    ax1.set_xlabel('Model Agent 2 Count', fontsize=18)
    ax1.set_ylabel('Human Mean Agent 2', fontsize=18)
    ax1.set_title(f'Experiment 3 - Agent 2, {exp3_csv_count} CSV files\nr = {corr_agent2:.3f}', fontsize=20)
    ax1.grid(False)
    
    # Add line of best fit
    if len(arr_model_agent2) > 1:
        fit2 = np.polyfit(arr_model_agent2, arr_human_agent2, 1)
        fit_fn2 = np.poly1d(fit2)
        x_vals2 = np.linspace(min(arr_model_agent2), max(arr_model_agent2), 100)
        ax1.plot(x_vals2, fit_fn2(x_vals2), color='red', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp3_agent2.png', dpi=300, bbox_inches='tight')
    print(f"\nExperiment 3 Agent 2 plot saved as '{output_dir / 'correlation_exp3_agent2.png'}'")
    
    # Create plot for Agent 3 (matching exp1 format)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(arr_model_agent3, arr_human_agent3, alpha=0.7)
    ax2.set_xlabel('Model Agent 3 Count', fontsize=18)
    ax2.set_ylabel('Human Mean Agent 3', fontsize=18)
    ax2.set_title(f'Experiment 3 - Agent 3, {exp3_csv_count} CSV files\nr = {corr_agent3:.3f}', fontsize=20)
    ax2.grid(False)
    
    # Add line of best fit
    if len(arr_model_agent3) > 1:
        fit3 = np.polyfit(arr_model_agent3, arr_human_agent3, 1)
        fit_fn3 = np.poly1d(fit3)
        x_vals3 = np.linspace(min(arr_model_agent3), max(arr_model_agent3), 100)
        ax2.plot(x_vals3, fit_fn3(x_vals3), color='red', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_exp3_agent3.png', dpi=300, bbox_inches='tight')
    print(f"Experiment 3 Agent 3 plot saved as '{output_dir / 'correlation_exp3_agent3.png'}'")
    
    # Create plots for exp3 baselines - Agent 2
    # Naive baseline
    if len(arr_naive_agent2) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_naive_agent2, arr_human_naive_agent2, alpha=0.7)
        ax.set_xlabel('Model Agent 2 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 2', fontsize=18)
        ax.set_title(f'Experiment 3 - Naive (Agent 2), {exp3_csv_count} CSV files\nr = {corr_naive_agent2:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_naive_agent2) > 1:
            fit = np.polyfit(arr_naive_agent2, arr_human_naive_agent2, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_naive_agent2), max(arr_naive_agent2), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_naive_agent2.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Naive (Agent 2) plot saved as '{output_dir / 'correlation_exp3_naive_agent2.png'}'")
    
    # Mentalize baseline
    if len(arr_mentalize_agent2) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_mentalize_agent2, arr_human_mentalize_agent2, alpha=0.7)
        ax.set_xlabel('Model Agent 2 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 2', fontsize=18)
        ax.set_title(f'Experiment 3 - Mentalize (Agent 2), {exp3_csv_count} CSV files\nr = {corr_mentalize_agent2:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_mentalize_agent2) > 1:
            fit = np.polyfit(arr_mentalize_agent2, arr_human_mentalize_agent2, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_mentalize_agent2), max(arr_mentalize_agent2), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_mentalize_agent2.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Mentalize (Agent 2) plot saved as '{output_dir / 'correlation_exp3_mentalize_agent2.png'}'")
    
    # Non-mentalize baseline
    if len(arr_nonmentalize_agent2) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_nonmentalize_agent2, arr_human_nonmentalize_agent2, alpha=0.7)
        ax.set_xlabel('Model Agent 2 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 2', fontsize=18)
        ax.set_title(f'Experiment 3 - Non-mentalize (Agent 2), {exp3_csv_count} CSV files\nr = {corr_nonmentalize_agent2:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_nonmentalize_agent2) > 1:
            fit = np.polyfit(arr_nonmentalize_agent2, arr_human_nonmentalize_agent2, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_nonmentalize_agent2), max(arr_nonmentalize_agent2), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_nonmentalize_agent2.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Non-mentalize (Agent 2) plot saved as '{output_dir / 'correlation_exp3_nonmentalize_agent2.png'}'")
    
    # Naive distance heur baseline
    if len(arr_naive_dist_agent2) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_naive_dist_agent2, arr_human_naive_dist_agent2, alpha=0.7)
        ax.set_xlabel('Model Agent 2 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 2', fontsize=18)
        ax.set_title(f'Experiment 3 - Naive Distance Heur (Agent 2), {exp3_csv_count} CSV files\nr = {corr_naive_dist_agent2:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_naive_dist_agent2) > 1:
            fit = np.polyfit(arr_naive_dist_agent2, arr_human_naive_dist_agent2, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_naive_dist_agent2), max(arr_naive_dist_agent2), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_naive_distance_heur_agent2.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Naive Distance Heur (Agent 2) plot saved as '{output_dir / 'correlation_exp3_naive_distance_heur_agent2.png'}'")
    
    # Non-mentalize distance heur baseline
    if len(arr_nonmentalize_dist_agent2) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_nonmentalize_dist_agent2, arr_human_nonmentalize_dist_agent2, alpha=0.7)
        ax.set_xlabel('Model Agent 2 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 2', fontsize=18)
        ax.set_title(f'Experiment 3 - Non-mentalize Distance Heur (Agent 2), {exp3_csv_count} CSV files\nr = {corr_nonmentalize_dist_agent2:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_nonmentalize_dist_agent2) > 1:
            fit = np.polyfit(arr_nonmentalize_dist_agent2, arr_human_nonmentalize_dist_agent2, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_nonmentalize_dist_agent2), max(arr_nonmentalize_dist_agent2), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_nonmentalize_distance_heur_agent2.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Non-mentalize Distance Heur (Agent 2) plot saved as '{output_dir / 'correlation_exp3_nonmentalize_distance_heur_agent2.png'}'")
    
    # Create plots for exp3 baselines - Agent 3
    # Naive baseline
    if len(arr_naive_agent3) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_naive_agent3, arr_human_naive_agent3, alpha=0.7)
        ax.set_xlabel('Model Agent 3 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 3', fontsize=18)
        ax.set_title(f'Experiment 3 - Naive (Agent 3), {exp3_csv_count} CSV files\nr = {corr_naive_agent3:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_naive_agent3) > 1:
            fit = np.polyfit(arr_naive_agent3, arr_human_naive_agent3, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_naive_agent3), max(arr_naive_agent3), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_naive_agent3.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Naive (Agent 3) plot saved as '{output_dir / 'correlation_exp3_naive_agent3.png'}'")
    
    # Mentalize baseline
    if len(arr_mentalize_agent3) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_mentalize_agent3, arr_human_mentalize_agent3, alpha=0.7)
        ax.set_xlabel('Model Agent 3 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 3', fontsize=18)
        ax.set_title(f'Experiment 3 - Mentalize (Agent 3), {exp3_csv_count} CSV files\nr = {corr_mentalize_agent3:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_mentalize_agent3) > 1:
            fit = np.polyfit(arr_mentalize_agent3, arr_human_mentalize_agent3, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_mentalize_agent3), max(arr_mentalize_agent3), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_mentalize_agent3.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Mentalize (Agent 3) plot saved as '{output_dir / 'correlation_exp3_mentalize_agent3.png'}'")
    
    # Non-mentalize baseline
    if len(arr_nonmentalize_agent3) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_nonmentalize_agent3, arr_human_nonmentalize_agent3, alpha=0.7)
        ax.set_xlabel('Model Agent 3 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 3', fontsize=18)
        ax.set_title(f'Experiment 3 - Non-mentalize (Agent 3), {exp3_csv_count} CSV files\nr = {corr_nonmentalize_agent3:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_nonmentalize_agent3) > 1:
            fit = np.polyfit(arr_nonmentalize_agent3, arr_human_nonmentalize_agent3, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_nonmentalize_agent3), max(arr_nonmentalize_agent3), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_nonmentalize_agent3.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Non-mentalize (Agent 3) plot saved as '{output_dir / 'correlation_exp3_nonmentalize_agent3.png'}'")
    
    # Naive distance heur baseline
    if len(arr_naive_dist_agent3) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_naive_dist_agent3, arr_human_naive_dist_agent3, alpha=0.7)
        ax.set_xlabel('Model Agent 3 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 3', fontsize=18)
        ax.set_title(f'Experiment 3 - Naive Distance Heur (Agent 3), {exp3_csv_count} CSV files\nr = {corr_naive_dist_agent3:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_naive_dist_agent3) > 1:
            fit = np.polyfit(arr_naive_dist_agent3, arr_human_naive_dist_agent3, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_naive_dist_agent3), max(arr_naive_dist_agent3), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_naive_distance_heur_agent3.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Naive Distance Heur (Agent 3) plot saved as '{output_dir / 'correlation_exp3_naive_distance_heur_agent3.png'}'")
    
    # Non-mentalize distance heur baseline
    if len(arr_nonmentalize_dist_agent3) > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(arr_nonmentalize_dist_agent3, arr_human_nonmentalize_dist_agent3, alpha=0.7)
        ax.set_xlabel('Model Agent 3 Count', fontsize=18)
        ax.set_ylabel('Human Mean Agent 3', fontsize=18)
        ax.set_title(f'Experiment 3 - Non-mentalize Distance Heur (Agent 3), {exp3_csv_count} CSV files\nr = {corr_nonmentalize_dist_agent3:.3f}', fontsize=20)
        ax.grid(False)
        if len(arr_nonmentalize_dist_agent3) > 1:
            fit = np.polyfit(arr_nonmentalize_dist_agent3, arr_human_nonmentalize_dist_agent3, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(min(arr_nonmentalize_dist_agent3), max(arr_nonmentalize_dist_agent3), 100)
            ax.plot(x_vals, fit_fn(x_vals), color='red', linestyle='-', linewidth=2)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_exp3_nonmentalize_distance_heur_agent3.png', dpi=300, bbox_inches='tight')
        print(f"Experiment 3 Non-mentalize Distance Heur (Agent 3) plot saved as '{output_dir / 'correlation_exp3_nonmentalize_distance_heur_agent3.png'}'")
    
    plt.show()
else:
    print("\n⚠ No matching levels found for correlation analysis")

