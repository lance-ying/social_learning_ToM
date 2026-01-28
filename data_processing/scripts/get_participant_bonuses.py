#!/usr/bin/env python3
"""
Simple script to extract participant scores and bonuses from JSON files.
Reads JSON files directly and outputs participant information with bonuses.
"""

import json
import sys
from pathlib import Path
import pandas as pd


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


def get_steps_remaining_from_events(events):
    """Extract the final steps remaining value from level events."""
    steps_remaining = None

    for event in events:
        event_type = event.get("type", "")
        data = event.get("data", {})

        # Look for LEVEL_COMPLETE events
        if "LEVEL_COMPLETE" in event_type.upper():
            # Try various field names for steps remaining
            steps = data.get("stepsRemaining") or data.get("remainingSteps") or data.get("steps_remaining")
            if steps is not None:
                steps_remaining = float(steps)

    return steps_remaining


def extract_participant_scores(json_path: Path, exclude_patterns=None):
    """
    Extract participant scores directly from JSON file.

    Args:
        json_path: Path to JSON file
        exclude_patterns: List of level name patterns to exclude (e.g., ['sm111_', 'sm112_'])

    Returns:
        List of participant dictionaries
    """
    if exclude_patterns is None:
        exclude_patterns = []

    with open(json_path, 'r') as f:
        data = json.load(f)

    users = data.get("users", {})
    participant_data = []

    for user_id, user_data in users.items():
        if not user_data:
            continue

        # Get demographics
        demographics = user_data.get("demographics", {})
        prolific_id = demographics.get("prolificId", "N/A")

        # Only process users with prolific IDs
        if not prolific_id or prolific_id == "N/A":
            continue

        # Get session info
        session_id = user_data.get("sessionId", user_id)

        # Process levels
        levels = user_data.get("levels", {})
        level_scores = {}

        for level_id, level_data in levels.items():
            # Skip excluded levels
            skip = any(level_id.startswith(pattern) for pattern in exclude_patterns)
            if skip or level_id in ['comprehension_check', 'experiment', 's111_1']:
                continue

            # Get batches for this level
            batches = level_data.get("batches", {})

            # Look for LEVEL_COMPLETE events in all batches
            level_steps = None
            for batch_id, batch_data in batches.items():
                events = batch_data.get("events", [])
                steps = get_steps_remaining_from_events(events)
                if steps is not None:
                    level_steps = steps

            if level_steps is not None:
                level_scores[level_id] = level_steps

        # Calculate total score
        total_score = sum(level_scores.values())
        bonus = calculate_bonus(total_score)

        participant_data.append({
            'file': json_path.name,
            'participant_id': user_id,
            'session_id': session_id,
            'prolific_id': prolific_id,
            'total_steps_remaining': total_score,
            'bonus_payment': bonus,
            'level_scores': level_scores,
            'n_levels_completed': len(level_scores)
        })

    return participant_data


def main(json_files: list, exclude_patterns=None):
    """
    Process multiple JSON files and output participant scores and bonuses.

    Args:
        json_files: List of paths to JSON files
        exclude_patterns: List of level name patterns to exclude
    """
    if exclude_patterns is None:
        exclude_patterns = ['sm111_', 'sm112_']  # Exclude tutorial levels

    all_participants = []

    print("=" * 80)
    print("PARTICIPANT SCORES AND BONUSES")
    print("=" * 80)
    print()

    for json_file in json_files:
        json_path = Path(json_file)
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            continue

        print(f"Processing: {json_path.name}")
        participants = extract_participant_scores(json_path, exclude_patterns)
        all_participants.extend(participants)
        print(f"  Found {len(participants)} participants with Prolific IDs")

    if not all_participants:
        print("\nNo participants found!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_participants)

    # Print detailed results
    print()
    print("=" * 80)
    print("PARTICIPANT DETAILS")
    print("=" * 80)
    print(f"{'File':<25} {'Prolific ID':<25} {'Score':<8} {'Bonus':<8}")
    print("-" * 80)

    for _, row in df.sort_values(['file', 'total_steps_remaining'], ascending=[True, False]).iterrows():
        print(f"{row['file']:<25} {row['prolific_id']:<25} "
              f"{row['total_steps_remaining']:<8.1f} ${row['bonus_payment']:.2f}")

    # Print summary statistics
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for file_name in df['file'].unique():
        file_df = df[df['file'] == file_name]
        print(f"\n{file_name}:")
        print(f"  Total participants:         {len(file_df)}")
        print(f"  Mean score:                 {file_df['total_steps_remaining'].mean():.2f}")
        print(f"  Std dev:                    {file_df['total_steps_remaining'].std():.2f}")
        print(f"  Total bonus to be paid:     ${file_df['bonus_payment'].sum():.2f}")
        print(f"  Mean bonus:                 ${file_df['bonus_payment'].mean():.2f}")
        print(f"  Max bonus recipients ($1):  {(file_df['bonus_payment'] >= 1.0).sum()}")
        print(f"  No bonus recipients ($0):   {(file_df['bonus_payment'] == 0.0).sum()}")

    # Overall summary
    if len(df['file'].unique()) > 1:
        print(f"\nOVERALL (all files):")
        print(f"  Total participants:         {len(df)}")
        print(f"  Mean score:                 {df['total_steps_remaining'].mean():.2f}")
        print(f"  Std dev:                    {df['total_steps_remaining'].std():.2f}")
        print(f"  Total bonus to be paid:     ${df['bonus_payment'].sum():.2f}")
        print(f"  Mean bonus:                 ${df['bonus_payment'].mean():.2f}")
        print(f"  Max bonus recipients ($1):  {(df['bonus_payment'] >= 1.0).sum()}")
        print(f"  No bonus recipients ($0):   {(df['bonus_payment'] == 0.0).sum()}")

    # Save to CSV
    output_file = Path("participant_bonuses_summary.csv")

    # Create simplified output
    output_df = df[['file', 'prolific_id', 'bonus_payment', 'total_steps_remaining', 'n_levels_completed']]
    output_df.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 80)

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_participant_bonuses.py <json_file1> [json_file2] ...")
        print("\nExample:")
        print("  python get_participant_bonuses.py exp4_pilot_2.json pilot_exp4_12.json")
        print("\nOptions:")
        print("  --include-tutorials   Include tutorial levels (sm111_, sm112_) in score calculation")
        sys.exit(1)

    # Parse arguments
    include_tutorials = "--include-tutorials" in sys.argv
    json_files = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    # Set exclusion patterns
    exclude_patterns = [] if include_tutorials else ['sm111_', 'sm112_']

    main(json_files, exclude_patterns)
