#!/usr/bin/env python3
"""
Per-level correlation analysis for Exp4.
Shows correlation for each level individually and calculates participant scores.
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


def calculate_participant_scores(csv_dir: Path, exclude_patterns=None):
    """
    Calculate scores for each participant from CSV files.
    Scores are based on steps remaining when level was completed.

    Args:
        csv_dir: Directory containing CSV files
        exclude_patterns: List of level name patterns to exclude (e.g., ['sm111_', 'sm112_'])

    Returns:
        DataFrame with participant scores
    """
    if exclude_patterns is None:
        exclude_patterns = []

    participant_data = []

    for csv_file in csv_dir.glob("*.csv"):
        participant_id = csv_file.stem

        # Read the CSV file into a dataframe for easier parsing
        import csv as csv_module

        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Parse session info
        session_id = None
        prolific_id = None
        for i, line in enumerate(lines):
            if line.startswith('Session ID,'):
                session_id = line.split(',', 1)[1].strip()
            if line.startswith('prolificId,'):
                prolific_id = line.split(',', 1)[1].strip()

        # Find all levels and get steps remaining at completion
        current_level = None
        level_scores = {}
        in_level_data = False
        header = None
        steps_remaining_idx = None

        for line in lines:
            line = line.strip()

            # Check for level header
            if line.startswith('Level:'):
                current_level = line.split(':', 1)[1].strip()
                in_level_data = False
                header = None
                steps_remaining_idx = None
                # Skip excluded levels
                skip = any(current_level.startswith(pattern) for pattern in exclude_patterns)
                if not skip and current_level not in ['comprehension_check', 'experiment', 's111_1']:
                    level_scores.setdefault(current_level, None)
                continue

            # Check for table header
            if current_level and current_level in level_scores and 'Timestamp' in line and 'Type' in line:
                in_level_data = True
                # Parse header to find Steps Remaining column
                try:
                    reader = csv_module.reader([line])
                    header = next(reader)
                    header_lower = [h.strip().lower() for h in header]
                    if 'steps remaining' in header_lower:
                        steps_remaining_idx = header_lower.index('steps remaining')
                except:
                    pass
                continue

            # Parse data rows
            if current_level and current_level in level_scores and in_level_data and header:
                if not line or line.startswith('Level:'):
                    in_level_data = False
                    continue

                try:
                    reader = csv_module.reader([line])
                    row = next(reader)

                    # Check if this row is LEVEL_COMPLETE with steps remaining
                    if 'LEVEL_COMPLETE' in line and steps_remaining_idx is not None:
                        # Get steps remaining value
                        if len(row) > steps_remaining_idx:
                            steps_val = row[steps_remaining_idx].strip()
                            if steps_val and steps_val != '' and steps_val != 'Steps Remaining':
                                try:
                                    steps = float(steps_val)
                                    # Always update to the latest value (there are usually 2 LEVEL_COMPLETE events)
                                    level_scores[current_level] = steps
                                except:
                                    pass
                except:
                    pass

        # Calculate total score (sum of steps remaining across all levels)
        valid_scores = {k: v for k, v in level_scores.items() if v is not None}
        total_score = sum(valid_scores.values())

        # Calculate bonus payment
        bonus = calculate_bonus(total_score)

        participant_data.append({
            'participant_id': participant_id,
            'session_id': session_id,
            'total_steps_remaining': total_score,
            'level_scores': valid_scores,
            'prolific_id': prolific_id,
            'bonus_payment': bonus
        })

    return pd.DataFrame(participant_data)


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


def per_level_correlation(human_json_path: Path, model_json_path: Path, output_dir: Path, per_file_csv: Path):
    """
    Calculate correlation for each level individually.
    """
    # Load data
    with open(human_json_path, 'r') as f:
        human_dict = json.load(f)

    with open(model_json_path, 'r') as f:
        model_dict = json.load(f)

    # Load per-file data for SD calculations
    per_file_df = pd.read_csv(per_file_csv)

    # Analyze per level
    level_comparisons = []

    print("=" * 80)
    print("PER-LEVEL CORRELATION ANALYSIS")
    print("=" * 80)
    print()

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
            model_total = model_agent2 + model_agent3

            human_agent2 = human_data.get("agent2_count", 0)
            human_agent3 = human_data.get("agent3_count", 0)
            activations = human_data.get("activation_count", 1)

            human_mean_agent2 = human_agent2 / activations if activations > 0 else 0
            human_mean_agent3 = human_agent3 / activations if activations > 0 else 0
            human_mean_total = human_mean_agent2 + human_mean_agent3

            # Calculate absolute errors
            error_agent2 = abs(model_agent2 - human_mean_agent2)
            error_agent3 = abs(model_agent3 - human_mean_agent3)
            error_total = abs(model_total - human_mean_total)

            # Calculate percent errors (avoid division by zero)
            pct_error_agent2 = (error_agent2 / model_agent2 * 100) if model_agent2 > 0 else (100 if human_mean_agent2 > 0 else 0)
            pct_error_agent3 = (error_agent3 / model_agent3 * 100) if model_agent3 > 0 else (100 if human_mean_agent3 > 0 else 0)
            pct_error_total = (error_total / model_total * 100) if model_total > 0 else (100 if human_mean_total > 0 else 0)

            # Calculate SD from per-file data
            level_data = per_file_df[per_file_df['level'] == human_level]
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

            level_comparisons.append({
                'level': human_level,
                'model_level': model_level,
                'model_agent2': model_agent2,
                'human_agent2_mean': human_mean_agent2,
                'human_agent2_sd': agent2_sd,
                'error_agent2': error_agent2,
                'pct_error_agent2': pct_error_agent2,
                'model_agent3': model_agent3,
                'human_agent3_mean': human_mean_agent3,
                'human_agent3_sd': agent3_sd,
                'error_agent3': error_agent3,
                'pct_error_agent3': pct_error_agent3,
                'model_total': model_total,
                'human_total': human_mean_total,
                'error_total': error_total,
                'pct_error_total': pct_error_total,
                'n_participants': activations
            })

    df = pd.DataFrame(level_comparisons)

    # Print summary
    print("Level-by-Level Comparison:")
    print(f"{'Level':<15} {'N':<4} {'Agent2 (M/H±SD)':<25} {'Agent3 (M/H±SD)':<25} {'Total Error':<12}")
    print("-" * 85)

    for _, row in df.iterrows():
        print(f"{row['level']:<15} {row['n_participants']:<4} "
              f"{row['model_agent2']:>4.0f}/{row['human_agent2_mean']:>5.2f}±{row['human_agent2_sd']:>4.2f} "
              f"{row['model_agent3']:>4.0f}/{row['human_agent3_mean']:>5.2f}±{row['human_agent3_sd']:>4.2f} "
              f"{row['error_total']:>6.2f} ({row['pct_error_total']:>5.1f}%)")

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Mean Absolute Error (Agent 2): {df['error_agent2'].mean():.2f}")
    print(f"Mean Absolute Error (Agent 3): {df['error_agent3'].mean():.2f}")
    print(f"Mean Absolute Error (Total):   {df['error_total'].mean():.2f}")
    print()
    print(f"Mean Percent Error (Agent 2):  {df['pct_error_agent2'].mean():.1f}%")
    print(f"Mean Percent Error (Agent 3):  {df['pct_error_agent3'].mean():.1f}%")
    print(f"Mean Percent Error (Total):    {df['pct_error_total'].mean():.1f}%")

    # Save simplified version as the main CSV
    simplified = pd.DataFrame({
        'level': df['level'],
        'model_agent2': df['model_agent2'],
        'human_agent2_mean': df['human_agent2_mean'],
        'human_agent2_sd': df['human_agent2_sd'],
        'model_agent3': df['model_agent3'],
        'human_agent3_mean': df['human_agent3_mean'],
        'human_agent3_sd': df['human_agent3_sd'],
        'n_participants': df['n_participants']
    })

    csv_path = output_dir / "per_level_comparison.csv"
    simplified.to_csv(csv_path, index=False)
    print(f"\n✓ Saved per-level comparison to {csv_path}")


    return simplified


def main(analysis_dir: str, model_json_path: str = None):
    """
    Main function for per-level analysis.

    Args:
        analysis_dir: Directory containing analysis results
        model_json_path: Optional path to model predictions JSON file
    """
    analysis_dir = Path(analysis_dir)

    human_json = analysis_dir / "overall_observations.json"
    csv_dir = analysis_dir / "csv_files"

    # Use provided model JSON path or default
    if model_json_path:
        model_json = Path(model_json_path)
    else:
        script_dir = Path(__file__).parent
        data_processing_dir = script_dir.parent
        model_json = data_processing_dir / "results/dictionaries/steps_dict_exp4_point5_updated.json"

    output_dir = analysis_dir / "per_level_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXPERIMENT 4 - PER-LEVEL ANALYSIS")
    print("=" * 80)
    print()

    # Per-level correlation
    print("[1/2] Calculating per-level correlations...")
    per_file_csv = analysis_dir / "per_file_observations.csv"
    per_level_df = per_level_correlation(human_json, model_json, output_dir, per_file_csv)

    # Participant scores
    print()
    print("[2/2] Calculating participant scores (excluding sm111/sm112)...")
    print("=" * 80)
    scores_df = calculate_participant_scores(csv_dir, exclude_patterns=['sm111_', 'sm112_'])

    print()
    print("PARTICIPANT SCORES (Steps Remaining)")
    print("=" * 80)
    print(f"{'Participant':<20} {'Prolific ID':<25} {'Total Steps':<12} {'Bonus':<8}")
    print("-" * 70)

    for _, row in scores_df.sort_values('total_steps_remaining', ascending=False).iterrows():
        prolific = row['prolific_id'] if row['prolific_id'] else 'N/A'
        print(f"{row['participant_id']:<20} {prolific:<25} "
              f"{row['total_steps_remaining']:<12.1f} ${row['bonus_payment']:.2f}")

    print()
    print("Summary:")
    print(f"  Mean total steps remaining: {scores_df['total_steps_remaining'].mean():.2f}")
    print(f"  Std dev:                    {scores_df['total_steps_remaining'].std():.2f}")
    print(f"  Total bonus to be paid:     ${scores_df['bonus_payment'].sum():.2f}")
    print(f"  Mean bonus:                 ${scores_df['bonus_payment'].mean():.2f}")
    print(f"  Max bonus recipients:       {(scores_df['bonus_payment'] >= 1.0).sum()}")
    print(f"  No bonus recipients:        {(scores_df['bonus_payment'] == 0.0).sum()}")

    # Show per-level breakdown
    print()
    print("Per-Level Score Breakdown:")
    all_level_scores = {}
    for _, row in scores_df.iterrows():
        for level, score in row['level_scores'].items():
            if level not in all_level_scores:
                all_level_scores[level] = []
            all_level_scores[level].append(score)

    print(f"{'Level':<15} {'Mean Steps':<12} {'Std Dev':<10} {'N':<5}")
    print("-" * 45)
    for level in sorted(all_level_scores.keys()):
        scores = all_level_scores[level]
        print(f"{level:<15} {np.mean(scores):<12.2f} {np.std(scores):<10.2f} {len(scores):<5}")

    # Save scores
    scores_path = output_dir / "participant_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"\n✓ Saved participant scores to {scores_path}")


    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}")

    return per_level_df, scores_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python per_level_analysis.py <analysis_directory> [model_json_path]")
        print("\nExample:")
        print("  python per_level_analysis.py analysis_pilot_exp4")
        print("  python per_level_analysis.py analysis_pilot_exp4 steps_dict_exp4_012626_merged.json")
        sys.exit(1)

    analysis_directory = sys.argv[1]
    model_json_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(analysis_directory, model_json_path)
