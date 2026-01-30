#!/usr/bin/env python3
"""
Compare two step dictionaries and calculate correlations.
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd


def main(dict1_path: str, dict2_path: str, output_dir: str = None):
    """
    Compare two dictionaries and calculate correlations.

    Args:
        dict1_path: Path to first dictionary JSON
        dict2_path: Path to second dictionary JSON
        output_dir: Optional output directory for results
    """
    dict1_path = Path(dict1_path)
    dict2_path = Path(dict2_path)

    if output_dir is None:
        output_dir = dict1_path.parent / "comparison_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dictionaries
    print("=" * 80)
    print("DICTIONARY COMPARISON AND CORRELATION")
    print("=" * 80)
    print(f"\nLoading dictionaries...")

    with open(dict1_path, 'r') as f:
        dict1 = json.load(f)

    with open(dict2_path, 'r') as f:
        dict2 = json.load(f)

    print(f"✓ Loaded {dict1_path.name} ({len(dict1)} items)")
    print(f"✓ Loaded {dict2_path.name} ({len(dict2)} items)")

    # Collect matching data
    comparison_data = []
    matched_keys = []

    print(f"\nMatching keys across dictionaries:")

    for key in dict1.keys():
        if key in dict2:
            val1 = dict1[key]
            val2 = dict2[key]

            # Extract agent2 and agent3 counts
            agent2_val1 = val1.get('agent2_count', 0) if isinstance(val1, dict) else 0
            agent2_val2 = val2.get('agent2_count', 0) if isinstance(val2, dict) else 0
            agent3_val1 = val1.get('agent3_count', 0) if isinstance(val1, dict) else 0
            agent3_val2 = val2.get('agent3_count', 0) if isinstance(val2, dict) else 0

            # Also try 't' for total
            total_val1 = val1.get('t', 0) if isinstance(val1, dict) else 0
            total_val2 = val2.get('t', 0) if isinstance(val2, dict) else 0

            comparison_data.append({
                'key': key,
                'dict1_agent2': agent2_val1,
                'dict2_agent2': agent2_val2,
                'dict1_agent3': agent3_val1,
                'dict2_agent3': agent3_val2,
                'dict1_total': total_val1,
                'dict2_total': total_val2,
            })
            matched_keys.append(key)
            print(f"  ✓ {key}: agent2: {agent2_val1} vs {agent2_val2}, agent3: {agent3_val1} vs {agent3_val2}")

    print(f"\nMatched {len(matched_keys)} keys")

    if len(matched_keys) == 0:
        print("\n⚠ No matching keys found!")
        return

    # Create comparison dataframe
    df = pd.DataFrame(comparison_data)

    # Calculate correlations
    print("\n" + "=" * 80)
    print("CORRELATION RESULTS")
    print("=" * 80)

    results = {}

    # Agent2 correlation
    if len(df) > 1:
        valid_agent2 = df.dropna(subset=['dict1_agent2', 'dict2_agent2'])
        if len(valid_agent2) > 1 and valid_agent2['dict1_agent2'].std() > 0 and valid_agent2['dict2_agent2'].std() > 0:
            r_agent2, p_agent2 = stats.pearsonr(valid_agent2['dict1_agent2'], valid_agent2['dict2_agent2'])
            results['agent2'] = {'r': r_agent2, 'p': p_agent2, 'n': len(valid_agent2)}
            print(f"\nAgent2 Correlation:")
            print(f"  r = {r_agent2:.6f}")
            print(f"  p-value = {p_agent2:.6e}")
            print(f"  n = {len(valid_agent2)}")
            print(f"  Interpretation: {'Significant' if p_agent2 < 0.05 else 'Not significant'} at α=0.05")
        else:
            print(f"\nAgent2 Correlation: Could not calculate (constant or insufficient data)")

        # Agent3 correlation
        valid_agent3 = df.dropna(subset=['dict1_agent3', 'dict2_agent3'])
        if len(valid_agent3) > 1 and valid_agent3['dict1_agent3'].std() > 0 and valid_agent3['dict2_agent3'].std() > 0:
            r_agent3, p_agent3 = stats.pearsonr(valid_agent3['dict1_agent3'], valid_agent3['dict2_agent3'])
            results['agent3'] = {'r': r_agent3, 'p': p_agent3, 'n': len(valid_agent3)}
            print(f"\nAgent3 Correlation:")
            print(f"  r = {r_agent3:.6f}")
            print(f"  p-value = {p_agent3:.6e}")
            print(f"  n = {len(valid_agent3)}")
            print(f"  Interpretation: {'Significant' if p_agent3 < 0.05 else 'Not significant'} at α=0.05")
        else:
            print(f"\nAgent3 Correlation: Could not calculate (constant or insufficient data)")

        # Total correlation
        valid_total = df.dropna(subset=['dict1_total', 'dict2_total'])
        if len(valid_total) > 1 and valid_total['dict1_total'].std() > 0 and valid_total['dict2_total'].std() > 0:
            r_total, p_total = stats.pearsonr(valid_total['dict1_total'], valid_total['dict2_total'])
            results['total'] = {'r': r_total, 'p': p_total, 'n': len(valid_total)}
            print(f"\nTotal Correlation:")
            print(f"  r = {r_total:.6f}")
            print(f"  p-value = {p_total:.6e}")
            print(f"  n = {len(valid_total)}")
            print(f"  Interpretation: {'Significant' if p_total < 0.05 else 'Not significant'} at α=0.05")
        else:
            print(f"\nTotal Correlation: Could not calculate (constant or insufficient data)")

    # Save results
    print("\n" + "=" * 80)

    # Save comparison CSV
    comparison_path = output_dir / "comparison.csv"
    df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison to {comparison_path}")

    # Save correlation stats
    stats_path = output_dir / "correlation_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved correlation stats to {stats_path}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

    return df, results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_dicts.py <dict1.json> <dict2.json> [output_dir]")
        print("\nExample:")
        print("  python compare_dicts.py steps_dict_exp4_012926.json exp4_20.json comparison_results")
        sys.exit(1)

    dict1_file = sys.argv[1]
    dict2_file = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else None

    main(dict1_file, dict2_file, output_directory)
