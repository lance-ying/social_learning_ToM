#!/usr/bin/env python3
"""
Calculate individual correlations for each participant.
Shows how well each person's observations match the model predictions.
"""
import json
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def map_level_name(human_level: str) -> str:
    """Map human level names to model level names."""
    match = re.match(r'^(sm\d+)_(\d+)$', human_level)
    if match:
        base = match.group(1)
        scenario_num = match.group(2)
        return f"{base}_scenario{scenario_num}"
    return human_level


def calculate_participant_correlations(per_file_csv: Path, model_json_path: Path):
    """
    Calculate correlations for each participant individually.

    Returns:
        DataFrame with per-participant correlation results
    """
    # Load data
    per_file_df = pd.read_csv(per_file_csv)
    with open(model_json_path, 'r') as f:
        model_dict = json.load(f)

    # Get unique participants
    participants = per_file_df['file'].unique()

    results = []

    print("Calculating per-participant correlations...")
    print()

    for participant in sorted(participants):
        participant_data = per_file_df[per_file_df['file'] == participant]

        # Collect data for correlation
        model_agent2 = []
        human_agent2 = []
        model_agent3 = []
        human_agent3 = []
        matched_levels = []

        for _, row in participant_data.iterrows():
            human_level = row['level']

            # Skip non-game and tutorial levels
            if human_level in ["comprehension_check", "experiment", "s111_1"]:
                continue
            if human_level.startswith("sm111_") or human_level.startswith("sm112_"):
                continue

            model_level = map_level_name(human_level)

            if model_level in model_dict:
                model_agent2.append(model_dict[model_level].get("agent2_count", 0))
                model_agent3.append(model_dict[model_level].get("agent3_count", 0))

                # Calculate means from participant's observation counts
                activations = row['activation_count']
                human_agent2.append(row['agent2_count'] / activations if activations > 0 else 0)
                human_agent3.append(row['agent3_count'] / activations if activations > 0 else 0)

                matched_levels.append(human_level)

        # Calculate correlations
        if len(matched_levels) > 1:
            model_agent2 = np.array(model_agent2)
            human_agent2 = np.array(human_agent2)
            model_agent3 = np.array(model_agent3)
            human_agent3 = np.array(human_agent3)

            corr_agent2 = np.corrcoef(model_agent2, human_agent2)[0, 1] if len(model_agent2) > 1 else np.nan
            corr_agent3 = np.corrcoef(model_agent3, human_agent3)[0, 1] if len(model_agent3) > 1 else np.nan


            results.append({
                'participant_id': participant,
                'n_levels': len(matched_levels),
                'agent2_r': corr_agent2,
                'agent3_r': corr_agent3,
                'levels': ', '.join(matched_levels)
            })

    df = pd.DataFrame(results)
    return df


def main(analysis_dir: str):
    """
    Main function for per-participant correlation analysis.

    Args:
        analysis_dir: Directory containing analysis results
    """
    analysis_dir = Path(analysis_dir)

    per_file_csv = analysis_dir / "per_file_observations.csv"

    script_dir = Path(__file__).parent
    data_processing_dir = script_dir.parent
    model_json = data_processing_dir / "results/dictionaries/steps_dict_exp4_point5_updated.json"

    output_dir = analysis_dir / "per_participant_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PER-PARTICIPANT CORRELATION ANALYSIS")
    print("=" * 80)
    print()

    if not per_file_csv.exists():
        print(f"Error: {per_file_csv} not found.")
        return

    # Calculate per-participant correlations
    results_df = calculate_participant_correlations(per_file_csv, model_json)

    # Print results
    print("=" * 80)
    print("PER-PARTICIPANT CORRELATION RESULTS")
    print("=" * 80)
    print()
    print(f"{'Participant':<20} {'N Levels':<10} {'Agent2 r':<12} {'Agent3 r':<12}")
    print("-" * 54)

    for _, row in results_df.sort_values('agent3_r', ascending=False).iterrows():
        a2_r = f"{row['agent2_r']:.3f}" if not np.isnan(row['agent2_r']) else "N/A"
        a3_r = f"{row['agent3_r']:.3f}" if not np.isnan(row['agent3_r']) else "N/A"
        print(f"{row['participant_id']:<20} {row['n_levels']:<10} {a2_r:<12} {a3_r:<12}")

    print()
    print("Summary Statistics:")
    print(f"  Mean Agent2 Correlation: {results_df['agent2_r'].mean():.3f}")
    print(f"  Mean Agent3 Correlation: {results_df['agent3_r'].mean():.3f}")
    print(f"  Std Dev Agent2:          {results_df['agent2_r'].std():.3f}")
    print(f"  Std Dev Agent3:          {results_df['agent3_r'].std():.3f}")

    # Save results
    csv_path = output_dir / "per_participant_correlations.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")

    return results_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python per_participant_correlation.py <analysis_directory>")
        print("\nExample:")
        print("  python per_participant_correlation.py analysis_pilot_exp4")
        sys.exit(1)

    analysis_directory = sys.argv[1]
    main(analysis_directory)
