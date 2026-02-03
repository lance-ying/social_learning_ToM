#!/usr/bin/env python3
"""
Split-half reliability analysis for participant data.

This script splits participants into two random halves, calculates mean
observations for each half, and computes the correlation between halves.
Includes Spearman-Brown correction for reliability estimation.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt


def load_participant_data(per_file_csv):
    """
    Load per-file observations from per_file_observations.csv.

    Args:
        per_file_csv: Path to per_file_observations.csv

    Returns:
        dict: {participant_id: {level: {'agent2': count, 'agent3': count}}}
    """
    per_file_csv = Path(per_file_csv)
    df = pd.read_csv(per_file_csv)

    print(f"Loaded {len(df)} rows from {per_file_csv.name}")

    participant_data = {}

    for _, row in df.iterrows():
        participant_id = row['file']
        level = row['level']
        agent2_count = row['agent2_count']
        agent3_count = row['agent3_count']

        if participant_id not in participant_data:
            participant_data[participant_id] = {}

        participant_data[participant_id][level] = {
            'agent2': agent2_count,
            'agent3': agent3_count
        }

    n_participants = len(participant_data)
    n_levels = len(df['level'].unique())
    print(f"Found {n_participants} participants across {n_levels} levels")

    return participant_data


def calculate_means_by_half(participant_data, half1_ids, half2_ids):
    """
    Calculate mean observations for each half.

    Returns:
        tuple: (half1_means, half2_means) where each is {level: {agent: mean}}
    """
    # Get all unique levels and agents
    all_levels = set()
    all_agents = set()

    for p_data in participant_data.values():
        for level, agents in p_data.items():
            all_levels.add(level)
            all_agents.update(agents.keys())

    # Initialize means dictionaries
    half1_means = {level: {agent: [] for agent in all_agents} for level in all_levels}
    half2_means = {level: {agent: [] for agent in all_agents} for level in all_levels}

    # Collect observations for half 1
    for p_id in half1_ids:
        p_data = participant_data[p_id]
        for level in all_levels:
            for agent in all_agents:
                obs = p_data.get(level, {}).get(agent, 0)
                half1_means[level][agent].append(obs)

    # Collect observations for half 2
    for p_id in half2_ids:
        p_data = participant_data[p_id]
        for level in all_levels:
            for agent in all_agents:
                obs = p_data.get(level, {}).get(agent, 0)
                half2_means[level][agent].append(obs)

    # Calculate means
    for level in all_levels:
        for agent in all_agents:
            half1_means[level][agent] = np.mean(half1_means[level][agent])
            half2_means[level][agent] = np.mean(half2_means[level][agent])

    return half1_means, half2_means


def calculate_correlation(half1_means, half2_means, agent_filter=None):
    """
    Calculate correlation between two halves for specified agent(s).

    Args:
        half1_means: Mean observations for half 1
        half2_means: Mean observations for half 2
        agent_filter: Specific agent to analyze (e.g., 'agent2', 'agent3'), or None for all

    Returns:
        dict: Correlation statistics
    """
    half1_values = []
    half2_values = []
    levels = []

    for level in half1_means.keys():
        for agent, mean1 in half1_means[level].items():
            # Skip tutorial levels
            if level.startswith('sm11'):
                continue

            # Apply agent filter if specified
            if agent_filter and agent != agent_filter:
                continue

            mean2 = half2_means[level][agent]
            half1_values.append(mean1)
            half2_values.append(mean2)
            levels.append(f"{level}_{agent}")

    # Calculate Pearson correlation
    r, p_value = stats.pearsonr(half1_values, half2_values)

    # Spearman-Brown correction for split-half reliability
    # Formula: reliability = (2 * r) / (1 + r)
    spearman_brown = (2 * r) / (1 + r) if r != -1 else np.nan

    return {
        'correlation': r,
        'p_value': p_value,
        'spearman_brown_reliability': spearman_brown,
        'n_points': len(half1_values),
        'half1_values': half1_values,
        'half2_values': half2_values,
        'levels': levels
    }


def plot_split_half(stats, output_path, title="Split-Half Correlation"):
    """Create a scatter plot of the split-half correlation."""
    plt.figure(figsize=(10, 8))

    half1 = stats['half1_values']
    half2 = stats['half2_values']

    plt.scatter(half1, half2, alpha=0.6, s=100)

    # Add regression line
    z = np.polyfit(half1, half2, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(half1), max(half1), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Add diagonal reference line (perfect correlation)
    max_val = max(max(half1), max(half2))
    plt.plot([0, max_val], [0, max_val], 'k:', alpha=0.3, linewidth=1, label='Perfect correlation')

    plt.xlabel('Half 1 Mean Observations', fontsize=12)
    plt.ylabel('Half 2 Mean Observations', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Add statistics text
    text = f"r = {stats['correlation']:.3f}\n"
    text += f"p = {stats['p_value']:.4f}\n"
    text += f"Spearman-Brown = {stats['spearman_brown_reliability']:.3f}\n"
    text += f"N = {stats['n_points']}"

    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved plot to {output_path}")


def run_split_half_analysis(per_file_csv, output_dir=None, n_iterations=1000, random_seed=42):
    """
    Run split-half correlation analysis with multiple random splits.

    Args:
        per_file_csv: Path to per_file_observations.csv
        output_dir: Output directory for results (default: same directory as per_file_csv)
        n_iterations: Number of random splits to average over
        random_seed: Random seed for reproducibility
    """
    per_file_csv = Path(per_file_csv)

    if output_dir is None:
        output_dir = per_file_csv.parent / "split_half_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SPLIT-HALF RELIABILITY ANALYSIS")
    print("=" * 80)
    print(f"Input file: {per_file_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {random_seed}")
    print(f"Iterations: {n_iterations}")
    print()

    # Load participant data
    print("[Step 1] Loading participant data...")
    participant_data = load_participant_data(per_file_csv)
    participant_ids = list(participant_data.keys())
    n_participants = len(participant_ids)

    print(f"✓ Loaded {n_participants} participants")

    if n_participants < 2:
        print("ERROR: Need at least 2 participants for split-half analysis")
        return

    # Run multiple iterations with different random splits
    print(f"\n[Step 2] Running {n_iterations} random splits...")

    np.random.seed(random_seed)

    correlations_agent2 = []
    correlations_agent3 = []
    correlations_all = []

    for i in range(n_iterations):
        # Random split
        shuffled_ids = np.random.permutation(participant_ids)
        half_point = len(shuffled_ids) // 2
        half1_ids = shuffled_ids[:half_point]
        half2_ids = shuffled_ids[half_point:]

        # Calculate means
        half1_means, half2_means = calculate_means_by_half(participant_data, half1_ids, half2_ids)

        # Calculate correlations
        stats_agent2 = calculate_correlation(half1_means, half2_means, 'agent2')
        stats_agent3 = calculate_correlation(half1_means, half2_means, 'agent3')
        stats_all = calculate_correlation(half1_means, half2_means, None)

        correlations_agent2.append(stats_agent2['correlation'])
        correlations_agent3.append(stats_agent3['correlation'])
        correlations_all.append(stats_all['correlation'])

    # Calculate summary statistics
    print(f"✓ Completed {n_iterations} iterations")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nAgent 2 (across {n_iterations} random splits):")
    print(f"  Mean r = {np.mean(correlations_agent2):.3f}")
    print(f"  Median r = {np.median(correlations_agent2):.3f}")
    print(f"  SD = {np.std(correlations_agent2):.3f}")
    print(f"  Range: [{np.min(correlations_agent2):.3f}, {np.max(correlations_agent2):.3f}]")

    print(f"\nAgent 3 (across {n_iterations} random splits):")
    print(f"  Mean r = {np.mean(correlations_agent3):.3f}")
    print(f"  Median r = {np.median(correlations_agent3):.3f}")
    print(f"  SD = {np.std(correlations_agent3):.3f}")
    print(f"  Range: [{np.min(correlations_agent3):.3f}, {np.max(correlations_agent3):.3f}]")

    print(f"\nAll Agents Combined (across {n_iterations} random splits):")
    print(f"  Mean r = {np.mean(correlations_all):.3f}")
    print(f"  Median r = {np.median(correlations_all):.3f}")
    print(f"  SD = {np.std(correlations_all):.3f}")
    print(f"  Range: [{np.min(correlations_all):.3f}, {np.max(correlations_all):.3f}]")

    # Run one final split for visualization (using seed)
    print("\n[Step 3] Generating visualization...")
    np.random.seed(random_seed)
    shuffled_ids = np.random.permutation(participant_ids)
    half_point = len(shuffled_ids) // 2
    half1_ids = shuffled_ids[:half_point]
    half2_ids = shuffled_ids[half_point:]

    half1_means, half2_means = calculate_means_by_half(participant_data, half1_ids, half2_ids)

    # Generate plots
    stats_agent2 = calculate_correlation(half1_means, half2_means, 'agent2')
    stats_agent3 = calculate_correlation(half1_means, half2_means, 'agent3')
    stats_all = calculate_correlation(half1_means, half2_means, None)

    plot_split_half(stats_agent2, output_dir / "split_half_agent2.png",
                    "Split-Half Correlation: Agent 2")
    plot_split_half(stats_agent3, output_dir / "split_half_agent3.png",
                    "Split-Half Correlation: Agent 3")
    plot_split_half(stats_all, output_dir / "split_half_all_agents.png",
                    "Split-Half Correlation: All Agents")

    # Save results to JSON
    results = {
        'n_participants': n_participants,
        'n_iterations': n_iterations,
        'random_seed': random_seed,
        'agent2': {
            'mean_r': float(np.mean(correlations_agent2)),
            'median_r': float(np.median(correlations_agent2)),
            'sd': float(np.std(correlations_agent2)),
            'min': float(np.min(correlations_agent2)),
            'max': float(np.max(correlations_agent2)),
            'mean_spearman_brown': float((2 * np.mean(correlations_agent2)) / (1 + np.mean(correlations_agent2)))
        },
        'agent3': {
            'mean_r': float(np.mean(correlations_agent3)),
            'median_r': float(np.median(correlations_agent3)),
            'sd': float(np.std(correlations_agent3)),
            'min': float(np.min(correlations_agent3)),
            'max': float(np.max(correlations_agent3)),
            'mean_spearman_brown': float((2 * np.mean(correlations_agent3)) / (1 + np.mean(correlations_agent3)))
        },
        'all_agents': {
            'mean_r': float(np.mean(correlations_all)),
            'median_r': float(np.median(correlations_all)),
            'sd': float(np.std(correlations_all)),
            'min': float(np.min(correlations_all)),
            'max': float(np.max(correlations_all)),
            'mean_spearman_brown': float((2 * np.mean(correlations_all)) / (1 + np.mean(correlations_all)))
        }
    }

    results_path = output_dir / "split_half_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to {results_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_half_correlation.py <per_file_observations.csv> [output_directory] [n_iterations] [random_seed]")
        print("\nExample:")
        print("  python split_half_correlation.py data_processing/exp4_15_merged_analysis/per_file_observations.csv")
        print("  python split_half_correlation.py data_processing/exp4_15_merged_analysis/per_file_observations.csv results/split_half 1000 42")
        print("\nArguments:")
        print("  per_file_observations.csv: Path to per_file_observations.csv from pipeline")
        print("  output_directory: (Optional) Output directory for results")
        print("  n_iterations: (Optional) Number of random splits to average over (default: 1000)")
        print("  random_seed: (Optional) Random seed for reproducibility (default: 42)")
        sys.exit(1)

    per_file_csv = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else None
    n_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    run_split_half_analysis(per_file_csv, output_directory, n_iterations, random_seed)
