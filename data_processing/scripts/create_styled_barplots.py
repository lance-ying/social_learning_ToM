#!/usr/bin/env python3
"""
Create styled bar plots comparing model vs human data.
Follows specific styling format with hyperparameters.
"""
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# HYPERPARAMETERS - Easy to adjust
FIGURE_WIDTH = 5.75  # Half of 11.5 for side-by-side agents
FIGURE_HEIGHT = 2
TITLE_FONTSIZE = 14
YLABEL_FONTSIZE = 12
XLABEL_FONTSIZE = 9
Y_AXIS_MAX = 15
BAR_WIDTH = 0.8
ERROR_BAR_CAPSIZE = 4
DPI = 300
COLORS = ['crimson', 'orange']
ALPHAS = [0.6, 0.75]


def map_level_name(human_level: str) -> str:
    """Map human level names to model level names."""
    match = re.match(r'^(sm\d+)_(\d+)$', human_level)
    if match:
        base = match.group(1)
        scenario_num = match.group(2)
        return f"{base}_scenario{scenario_num}"
    return human_level


def calculate_sds(level, per_file_df):
    """Calculate standard deviations for agent2 and agent3."""
    level_data = per_file_df[per_file_df['level'] == level]
    agent2_means = []
    agent3_means = []

    for _, row in level_data.iterrows():
        agent2_count = row['agent2_count']
        agent3_count = row['agent3_count']
        activations_file = row['activation_count']

        if activations_file > 0:
            agent2_means.append(agent2_count / activations_file)
            agent3_means.append(agent3_count / activations_file)

    agent2_sd = np.std(agent2_means, ddof=1) if len(agent2_means) > 1 else 0.0
    agent3_sd = np.std(agent3_means, ddof=1) if len(agent3_means) > 1 else 0.0

    return agent2_sd, agent3_sd


def create_barplots(analysis_dir, model_json_path, output_dir=None):
    """
    Create styled bar plots for model vs human comparison.

    Args:
        analysis_dir: Directory containing analysis results (overall_observations.json, per_file_observations.csv)
        model_json_path: Path to model predictions JSON file
        output_dir: Optional output directory for plots (default: analysis_dir/barplots_styled)
    """
    analysis_dir = Path(analysis_dir)
    model_json_path = Path(model_json_path)

    if output_dir is None:
        output_dir = analysis_dir / "barplots_styled"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    human_json = analysis_dir / "overall_observations.json"
    per_file_csv = analysis_dir / "per_file_observations.csv"

    with open(human_json, 'r') as f:
        human_dict = json.load(f)

    with open(model_json_path, 'r') as f:
        model_dict = json.load(f)

    per_file_df = pd.read_csv(per_file_csv)

    print("=" * 80)
    print("CREATING STYLED BAR PLOTS")
    print("=" * 80)
    print(f"Analysis directory: {analysis_dir}")
    print(f"Model predictions: {model_json_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Prepare data for matched stimuli
    matched_data = []
    for human_level, human_data in sorted(human_dict.items()):
        # Skip non-game levels
        if human_level in ["comprehension_check", "experiment", "s111_1"]:
            continue

        # Skip tutorials
        if human_level.startswith("sm111_") or human_level.startswith("sm112_"):
            continue

        model_level = map_level_name(human_level)

        if model_level in model_dict:
            model_data = model_dict[model_level]

            # Get counts
            model_agent2 = model_data.get("agent2_count", 0)
            model_agent3 = model_data.get("agent3_count", 0)

            human_agent2 = human_data.get("agent2_count", 0)
            human_agent3 = human_data.get("agent3_count", 0)
            activations = human_data.get("activation_count", 1)

            human_mean_agent2 = human_agent2 / activations if activations > 0 else 0
            human_mean_agent3 = human_agent3 / activations if activations > 0 else 0

            # Calculate SDs
            agent2_sd, agent3_sd = calculate_sds(human_level, per_file_df)

            matched_data.append({
                'level': human_level,
                'model_level': model_level,
                'model_agent2': model_agent2,
                'model_agent3': model_agent3,
                'human_agent2': human_mean_agent2,
                'human_agent3': human_mean_agent3,
                'agent2_sd': agent2_sd,
                'agent3_sd': agent3_sd,
                'n_participants': activations
            })

    print(f"Found {len(matched_data)} matched stimuli with human data")
    print()

    # Create plots for matched stimuli
    for item in matched_data:
        level = item['level']

        # Create figure
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

        # Prepare data with grouped layout
        # Group 1: Agent 2 (Model and Human close together)
        # Group 2: Agent 3 (Model and Human close together)
        agent_types = ['Model\nAgent 2', 'Human\nAgent 2', 'Model\nAgent 3', 'Human\nAgent 3']
        values = [
            item['model_agent2'],
            item['human_agent2'],
            item['model_agent3'],
            item['human_agent3']
        ]

        # Create grouped bar positions
        # Bars within a group are closer (0.9 apart), groups are further (2.5 apart)
        x_pos = np.array([0, 0.9, 2.5, 3.4])

        ax.bar(x_pos[0], values[0], color=COLORS[0], width=BAR_WIDTH, alpha=ALPHAS[0], edgecolor='none')
        ax.bar(x_pos[1], values[1], color=COLORS[0], width=BAR_WIDTH, alpha=ALPHAS[1], edgecolor='none')
        ax.bar(x_pos[2], values[2], color=COLORS[1], width=BAR_WIDTH, alpha=ALPHAS[0], edgecolor='none')
        ax.bar(x_pos[3], values[3], color=COLORS[1], width=BAR_WIDTH, alpha=ALPHAS[1], edgecolor='none')

        # Add error bars only for human data
        ax.errorbar(x_pos[1], values[1], yerr=item['agent2_sd'], fmt='none',
                    color='black', capsize=ERROR_BAR_CAPSIZE, capthick=1.5, linewidth=1.5)
        ax.errorbar(x_pos[3], values[3], yerr=item['agent3_sd'], fmt='none',
                    color='black', capsize=ERROR_BAR_CAPSIZE, capthick=1.5, linewidth=1.5)

        # Add value labels on top of bars
        for i, (x, val) in enumerate(zip(x_pos, values)):
            # For human bars (with error bars), place text above the error bar
            if i == 1:  # Human Agent 2
                y_offset = val + item['agent2_sd'] + 0.3
            elif i == 3:  # Human Agent 3
                y_offset = val + item['agent3_sd'] + 0.3
            else:  # Model bars
                y_offset = val + 0.3

            ax.text(x, y_offset, f'{val:.1f}', ha='center', va='bottom',
                   fontsize=XLABEL_FONTSIZE, fontweight='bold')

        # Set x-axis with labels centered on each group
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_types, fontsize=XLABEL_FONTSIZE)
        ax.set_xlim(-0.6, 4.0)

        # Set y-axis
        ax.set_ylim(0, Y_AXIS_MAX)
        ax.set_yticks([10])
        ax.set_ylabel('Number of\nObservations', fontsize=YLABEL_FONTSIZE)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        ax.set_title(level, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)

        plt.tight_layout()

        # Save plot
        output_path = output_dir / f"barplot_{level}.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Saved {output_path.name}")

    # Create plots for model-only stimuli
    print()
    print("Creating model-only stimuli plots...")
    print()

    model_only_dir = output_dir / "model_only"
    model_only_dir.mkdir(parents=True, exist_ok=True)

    model_only_count = 0
    for model_level, model_data in sorted(model_dict.items()):
        # Skip if already in matched data
        if any(item['model_level'] == model_level for item in matched_data):
            continue

        model_agent2 = model_data.get("agent2_count", 0)
        model_agent3 = model_data.get("agent3_count", 0)

        # Create figure
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

        # Prepare data
        agent_types = ['Model\nAgent 2', 'Model\nAgent 3']
        values = [model_agent2, model_agent3]

        # Create bars with some spacing
        x_pos = np.array([0, 1.5])
        ax.bar(x_pos[0], values[0], color=COLORS[0], width=BAR_WIDTH, alpha=ALPHAS[0], edgecolor='none')
        ax.bar(x_pos[1], values[1], color=COLORS[1], width=BAR_WIDTH, alpha=ALPHAS[0], edgecolor='none')

        # Add value labels on top of bars
        for x, val in zip(x_pos, values):
            ax.text(x, val + 0.3, f'{val:.1f}', ha='center', va='bottom',
                   fontsize=XLABEL_FONTSIZE, fontweight='bold')

        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agent_types, fontsize=XLABEL_FONTSIZE)
        ax.set_xlim(-0.6, 2.1)

        # Set y-axis
        ax.set_ylim(0, Y_AXIS_MAX)
        ax.set_yticks([10])
        ax.set_ylabel('Number of\nObservations', fontsize=YLABEL_FONTSIZE)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add title
        ax.set_title(f"{model_level}", fontsize=TITLE_FONTSIZE, fontweight='bold', pad=10)

        plt.tight_layout()

        # Save plot
        output_path = model_only_dir / f"barplot_{model_level}.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close()

        model_only_count += 1
        print(f"✓ Saved {output_path.name}")

    print()
    print("=" * 80)
    print("BAR PLOTS COMPLETE")
    print("=" * 80)
    print(f"\nMatched stimuli: {len(matched_data)} plots saved to {output_dir}")
    print(f"Model-only stimuli: {model_only_count} plots saved to {model_only_dir}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_styled_barplots.py <analysis_directory> <model_json_path> [output_dir]")
        print("\nExample:")
        print("  python create_styled_barplots.py analysis_pilot_final_I_think steps_dict_exp4_012626_merged.json")
        print("  python create_styled_barplots.py analysis_pilot_final_I_think steps_dict_exp4_012626_merged.json my_plots")
        sys.exit(1)

    analysis_directory = sys.argv[1]
    model_json = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else None

    create_barplots(analysis_directory, model_json, output_directory)
