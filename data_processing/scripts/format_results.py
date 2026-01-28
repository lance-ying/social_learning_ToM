#!/usr/bin/env python3
"""
Format results CSVs for easier analysis:
1. Simplify per_level_comparison.csv to show only model vs human with SD
2. Add bonus payment calculation to participant_scores.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json


def format_per_level_comparison(input_csv: Path, output_csv: Path, per_file_csv: Path):
    """
    Simplify per_level comparison to show:
    - level
    - model_agent2, human_agent2_mean, human_agent2_sd
    - model_agent3, human_agent3_mean, human_agent3_sd
    """
    # Read the comparison data
    df = pd.read_csv(input_csv)

    # Read per-file data to calculate SD
    per_file = pd.read_csv(per_file_csv)

    # Calculate per-level human means and SDs from per_file data
    human_stats = []

    for level in df['level']:
        level_data = per_file[per_file['level'] == level]

        if len(level_data) > 0:
            # Get mean observe per activation for each file in this level
            observations = level_data['mean_observe_per_activation'].values

            # Calculate agent2 and agent3 separately by looking at raw per-file data
            # We need to recalculate from the per_file observations
            agent2_means = []
            agent3_means = []

            for _, row in level_data.iterrows():
                # Mean observations per activation for this file/level
                total_mean = row['mean_observe_per_activation']
                agent2_count = row['agent2_count']
                agent3_count = row['agent3_count']
                activations = row['activation_count']

                if activations > 0:
                    agent2_mean = agent2_count / activations
                    agent3_mean = agent3_count / activations
                    agent2_means.append(agent2_mean)
                    agent3_means.append(agent3_mean)

            agent2_sd = np.std(agent2_means, ddof=1) if len(agent2_means) > 1 else 0.0
            agent3_sd = np.std(agent3_means, ddof=1) if len(agent3_means) > 1 else 0.0
        else:
            agent2_sd = 0.0
            agent3_sd = 0.0

        human_stats.append({
            'level': level,
            'agent2_sd': agent2_sd,
            'agent3_sd': agent3_sd
        })

    stats_df = pd.DataFrame(human_stats)
    df = df.merge(stats_df, on='level', how='left')

    # Create simplified dataframe
    simplified = pd.DataFrame({
        'level': df['level'],
        'model_agent2': df['model_agent2'],
        'human_agent2_mean': df['human_agent2'],
        'human_agent2_sd': df['agent2_sd'],
        'model_agent3': df['model_agent3'],
        'human_agent3_mean': df['human_agent3'],
        'human_agent3_sd': df['agent3_sd'],
        'n_participants': df['n_participants']
    })

    simplified.to_csv(output_csv, index=False)
    print(f"✓ Formatted per-level comparison saved to {output_csv.name}")
    return simplified


def calculate_bonus(score: float) -> float:
    """
    Calculate bonus payment:
    - $0.50 per 25 points
    - Capped at $1.00
    - If score <= 0, bonus = $0.00
    """
    if score <= 0:
        return 0.0

    bonus = (score / 25.0) * 0.50
    return min(bonus, 1.0)


def format_participant_scores(input_csv: Path, output_csv: Path):
    """
    Add prolific_id and bonus_payment columns to participant scores.
    """
    df = pd.read_csv(input_csv)

    # Calculate bonus payment
    df['bonus_payment'] = df['total_steps_remaining'].apply(calculate_bonus)

    # Reorder columns to have prolific_id and bonus_payment at the end
    cols = [c for c in df.columns if c not in ['prolific_id', 'bonus_payment']]
    cols.extend(['prolific_id', 'bonus_payment'])
    df = df[cols]

    df.to_csv(output_csv, index=False)
    print(f"✓ Formatted participant scores saved to {output_csv.name}")

    # Print summary
    print()
    print("Bonus Payment Summary:")
    print(f"  Total bonus to be paid: ${df['bonus_payment'].sum():.2f}")
    print(f"  Mean bonus: ${df['bonus_payment'].mean():.2f}")
    print(f"  Participants receiving max bonus ($1.00): {(df['bonus_payment'] >= 1.0).sum()}")
    print(f"  Participants receiving no bonus ($0.00): {(df['bonus_payment'] == 0.0).sum()}")

    return df


def main(analysis_dir: str):
    """
    Format result CSVs in the analysis directory.
    """
    analysis_dir = Path(analysis_dir)
    per_level_dir = analysis_dir / "per_level_analysis"

    if not per_level_dir.exists():
        print(f"Error: {per_level_dir} not found. Run per_level_analysis.py first.")
        return

    print("=" * 80)
    print("FORMATTING RESULT CSVs")
    print("=" * 80)
    print()

    # Format per-level comparison
    per_level_csv = per_level_dir / "per_level_comparison.csv"
    per_file_csv = analysis_dir / "per_file_observations.csv"
    per_level_formatted = per_level_dir / "per_level_comparison_formatted.csv"

    if per_level_csv.exists() and per_file_csv.exists():
        print("[1/2] Formatting per-level comparison...")
        format_per_level_comparison(per_level_csv, per_level_formatted, per_file_csv)
    else:
        print(f"⚠ Skipping per-level comparison (files not found)")

    print()

    # Format participant scores
    scores_csv = per_level_dir / "participant_scores.csv"
    scores_formatted = per_level_dir / "participant_scores_formatted.csv"

    if scores_csv.exists():
        print("[2/2] Formatting participant scores and calculating bonuses...")
        format_participant_scores(scores_csv, scores_formatted)
    else:
        print(f"⚠ Skipping participant scores (file not found)")

    print()
    print("=" * 80)
    print("FORMATTING COMPLETE")
    print("=" * 80)
    print(f"Formatted files saved in: {per_level_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_results.py <analysis_directory>")
        print("\nExample:")
        print("  python format_results.py analysis_pilot_exp4")
        sys.exit(1)

    analysis_directory = sys.argv[1]
    main(analysis_directory)
