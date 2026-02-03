#!/usr/bin/env python3
"""
Count unique stimuli in JSON files.
"""

import json
from pathlib import Path


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
    # Define files
    steps_exp2 = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/steps_exp2.json"
    steps_dict = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/steps.dict.json"

    print("=" * 80)
    print("UNIQUE STIMULI COUNTER (JSON FILES)")
    print("=" * 80)

    # Count stimuli in steps_exp2.json
    print(f"\nAnalyzing steps_exp2.json...")
    exp2_stimuli = count_stimuli_in_json(steps_exp2)

    # Count stimuli in steps.dict.json
    print(f"Analyzing steps.dict.json...")
    dict_stimuli = count_stimuli_in_json(steps_dict)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if exp2_stimuli is not None:
        print(f"\nsteps_exp2.json: {len(exp2_stimuli)} unique stimuli")
        for stimulus in sorted(exp2_stimuli):
            print(f"  - {stimulus}")

    if dict_stimuli is not None:
        print(f"\nsteps.dict.json: {len(dict_stimuli)} unique stimuli")
        for stimulus in sorted(dict_stimuli):
            print(f"  - {stimulus}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if exp2_stimuli is not None:
        print(f"steps_exp2.json unique stimuli: {len(exp2_stimuli)}")
    if dict_stimuli is not None:
        print(f"steps.dict.json unique stimuli: {len(dict_stimuli)}")

    # Compare
    if exp2_stimuli is not None and dict_stimuli is not None:
        common = exp2_stimuli & dict_stimuli
        only_exp2 = exp2_stimuli - dict_stimuli
        only_dict = dict_stimuli - exp2_stimuli

        print(f"\nCommon stimuli: {len(common)}")
        if common:
            for s in sorted(common):
                print(f"  - {s}")

        print(f"\nOnly in steps_exp2.json: {len(only_exp2)}")
        if only_exp2:
            for s in sorted(only_exp2):
                print(f"  - {s}")

        print(f"\nOnly in steps.dict.json: {len(only_dict)}")
        if only_dict:
            for s in sorted(only_dict):
                print(f"  - {s}")


if __name__ == "__main__":
    main()
