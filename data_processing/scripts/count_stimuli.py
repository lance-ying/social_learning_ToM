#!/usr/bin/env python3
"""
Count unique stimuli (levels) in exp1 and exp2 directories.
"""

from pathlib import Path
import re

def extract_stimuli_from_file(csv_file):
    """Extract unique stimuli (level names) from a single CSV file."""
    stimuli = set()
    try:
        with open(csv_file, 'r') as f:
            for line in f:
                # Look for lines like "Level: level_name"
                if line.startswith('Level: '):
                    level_name = line.strip()[7:]  # Remove "Level: " prefix
                    stimuli.add(level_name)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
    return stimuli


def count_stimuli_in_directory(dir_path):
    """Count unique stimuli across all CSV files in a directory."""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"Directory not found: {dir_path}")
        return None

    csv_files = list(dir_path.glob('*.csv'))
    print(f"\nAnalyzing {dir_path.name}...")
    print(f"Found {len(csv_files)} CSV files")

    all_stimuli = set()
    stimuli_per_file = {}

    for csv_file in sorted(csv_files):
        stimuli = extract_stimuli_from_file(csv_file)
        stimuli_per_file[csv_file.name] = stimuli
        all_stimuli.update(stimuli)

    return all_stimuli, stimuli_per_file


def main():
    # Define directories
    exp1_dir = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/data_processing/data_processed/exp1"
    exp2_dir = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/data_processing/data_processed/exp2"

    print("=" * 80)
    print("UNIQUE STIMULI COUNTER")
    print("=" * 80)

    # Count for exp1
    exp1_stimuli, exp1_per_file = count_stimuli_in_directory(exp1_dir)

    # Count for exp2
    exp2_stimuli, exp2_per_file = count_stimuli_in_directory(exp2_dir)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if exp1_stimuli is not None:
        print(f"\nExp1: {len(exp1_stimuli)} unique stimuli")
        for stimulus in sorted(exp1_stimuli):
            print(f"  - {stimulus}")

    if exp2_stimuli is not None:
        print(f"\nExp2: {len(exp2_stimuli)} unique stimuli")
        for stimulus in sorted(exp2_stimuli):
            print(f"  - {stimulus}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if exp1_stimuli is not None:
        print(f"Exp1 unique stimuli: {len(exp1_stimuli)}")
    if exp2_stimuli is not None:
        print(f"Exp2 unique stimuli: {len(exp2_stimuli)}")

    # Compare
    if exp1_stimuli is not None and exp2_stimuli is not None:
        common = exp1_stimuli & exp2_stimuli
        only_exp1 = exp1_stimuli - exp2_stimuli
        only_exp2 = exp2_stimuli - exp1_stimuli

        print(f"\nCommon stimuli: {len(common)}")
        print(f"Only in Exp1: {len(only_exp1)}")
        print(f"Only in Exp2: {len(only_exp2)}")


if __name__ == "__main__":
    main()
