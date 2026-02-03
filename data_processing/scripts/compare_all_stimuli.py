#!/usr/bin/env python3
"""
Compare stimuli across all experiments and JSON files.
Filters CSV to only count mod_sXXX (exp1) or sXXX (exp2), excluding comprehension_check, experiment, etc.
"""

from pathlib import Path
import json
import re


def extract_stimuli_from_csv_files(dir_path, prefix_filter=None):
    """
    Extract filtered stimuli from CSV files.

    Args:
        dir_path: Path to CSV directory
        prefix_filter: 'mod_s' for exp1 or 's' for exp2, or None for all
    """
    dir_path = Path(dir_path)
    all_stimuli = set()

    csv_files = list(dir_path.glob('*.csv'))

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                for line in f:
                    # Look for lines like "Level: level_name"
                    if line.startswith('Level: '):
                        level_name = line.strip()[7:]  # Remove "Level: " prefix

                        # Skip non-game levels
                        if level_name in ['comprehension_check', 'experiment']:
                            continue

                        # Apply prefix filter
                        if prefix_filter == 'mod_s':
                            # Only keep mod_s* (and not mod_s111, mod_s112)
                            if level_name.startswith('mod_s'):
                                if not (level_name.startswith('mod_s111') or level_name.startswith('mod_s112')):
                                    all_stimuli.add(level_name)
                        elif prefix_filter == 's':
                            # Only keep s* (not mod_s*)
                            if level_name.startswith('s') and not level_name.startswith('mod_s'):
                                all_stimuli.add(level_name)
                        else:
                            # No filter
                            all_stimuli.add(level_name)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    return all_stimuli


def count_stimuli_in_json(json_file):
    """Count unique stimuli (keys) in a JSON file."""
    json_file = Path(json_file)

    if not json_file.exists():
        print(f"File not found: {json_file}")
        return None

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            stimuli = set(data.keys())
            return stimuli
        else:
            print(f"Error: {json_file} does not contain a JSON object (dict)")
            return None

    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None


def main():
    # Define paths
    exp1_dir = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/data_processing/data_processed/exp1"
    exp2_dir = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/data_processing/data_processed/exp2"
    steps_exp2 = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/steps_exp2.json"
    steps_dict = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/steps.dict.json"

    print("=" * 80)
    print("COMPLETE STIMULI COMPARISON (ALL EXPERIMENTS)")
    print("=" * 80)

    # Extract filtered stimuli from CSV files
    print(f"\nExtracting stimuli from Exp1 CSV files...")
    print("  Filters: Only mod_s*, exclude mod_s111, mod_s112")
    exp1_csv_stimuli = extract_stimuli_from_csv_files(exp1_dir, prefix_filter='mod_s')

    print(f"Extracting stimuli from Exp2 CSV files...")
    print("  Filters: Only s* (not mod_s*)")
    exp2_csv_stimuli = extract_stimuli_from_csv_files(exp2_dir, prefix_filter='s')

    # Count stimuli in JSON files
    print(f"\nAnalyzing steps_exp2.json...")
    steps_exp2_stimuli = count_stimuli_in_json(steps_exp2)

    print(f"Analyzing steps.dict.json...")
    steps_dict_stimuli = count_stimuli_in_json(steps_dict)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if exp1_csv_stimuli is not None:
        print(f"\nExp1 CSV (filtered mod_s*): {len(exp1_csv_stimuli)} unique stimuli")
        for stimulus in sorted(exp1_csv_stimuli):
            print(f"  - {stimulus}")

    if exp2_csv_stimuli is not None:
        print(f"\nExp2 CSV (filtered s*): {len(exp2_csv_stimuli)} unique stimuli")
        for stimulus in sorted(exp2_csv_stimuli):
            print(f"  - {stimulus}")

    if steps_exp2_stimuli is not None:
        print(f"\nsteps_exp2.json: {len(steps_exp2_stimuli)} unique stimuli")
        for stimulus in sorted(steps_exp2_stimuli):
            print(f"  - {stimulus}")

    if steps_dict_stimuli is not None:
        print(f"\nsteps.dict.json: {len(steps_dict_stimuli)} unique stimuli")
        for stimulus in sorted(steps_dict_stimuli):
            print(f"  - {stimulus}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if exp1_csv_stimuli is not None:
        print(f"Exp1 CSV (filtered): {len(exp1_csv_stimuli)} stimuli")
    if exp2_csv_stimuli is not None:
        print(f"Exp2 CSV (filtered): {len(exp2_csv_stimuli)} stimuli")
    if steps_exp2_stimuli is not None:
        print(f"steps_exp2.json: {len(steps_exp2_stimuli)} stimuli")
    if steps_dict_stimuli is not None:
        print(f"steps.dict.json: {len(steps_dict_stimuli)} stimuli")

    # Compare Exp2 CSV vs steps_exp2.json
    print("\n" + "-" * 80)
    print("COMPARISON: Exp2 CSV vs steps_exp2.json")
    print("-" * 80)

    if exp2_csv_stimuli is not None and steps_exp2_stimuli is not None:
        common = exp2_csv_stimuli & steps_exp2_stimuli
        only_csv = exp2_csv_stimuli - steps_exp2_stimuli
        only_json = steps_exp2_stimuli - exp2_csv_stimuli

        print(f"\nCommon: {len(common)}")
        if common:
            for s in sorted(common):
                print(f"  - {s}")

        print(f"\nOnly in Exp2 CSV: {len(only_csv)}")
        if only_csv:
            for s in sorted(only_csv):
                print(f"  - {s}")

        print(f"\nOnly in steps_exp2.json: {len(only_json)}")
        if only_json:
            for s in sorted(only_json):
                print(f"  - {s}")

    # Compare Exp1 CSV vs steps.dict.json
    print("\n" + "-" * 80)
    print("COMPARISON: Exp1 CSV vs steps.dict.json")
    print("-" * 80)

    if exp1_csv_stimuli is not None and steps_dict_stimuli is not None:
        # Strip "_ascii" and "_1" suffixes for comparison
        dict_base = {s.replace('_ascii', '').replace('mod_', '') for s in steps_dict_stimuli}
        csv_base = {s.replace('mod_', '') for s in exp1_csv_stimuli}

        print(f"\nExp1 CSV base names (mod_ prefix removed): {len(csv_base)}")
        print(f"steps.dict.json base names (_ascii suffix removed): {len(dict_base)}")

        common = csv_base & dict_base
        only_csv = csv_base - dict_base
        only_dict = dict_base - csv_base

        print(f"\nCommon (base names): {len(common)}")
        if common:
            for s in sorted(common):
                print(f"  - {s}")

        print(f"\nOnly in CSV: {len(only_csv)}")
        if only_csv:
            for s in sorted(only_csv):
                print(f"  - {s}")

        print(f"\nOnly in steps.dict.json: {len(only_dict)}")
        if only_dict:
            for s in sorted(only_dict):
                print(f"  - {s}")


if __name__ == "__main__":
    main()
