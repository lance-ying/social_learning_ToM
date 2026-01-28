#!/usr/bin/env python3
"""
Automated pipeline to:
1. Convert JSON data to CSV
2. Parse observation counts from CSV
3. Compare to reference dictionary
4. Output correlation analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from scipy import stats

# Import existing modules
from extract_json import build_report_files
from observe_parser_multi_npc import count_observes_and_means


def load_reference_dict(reference_path: Path) -> Dict:
    """Load reference observation dictionary."""
    with open(reference_path, 'r') as f:
        return json.load(f)


def compare_observations(actual_dict: Dict, reference_dict: Dict) -> pd.DataFrame:
    """
    Compare actual observations to reference observations.
    Returns DataFrame with comparison metrics.
    """
    comparison_rows = []

    # Get all unique levels
    all_levels = set(actual_dict.keys()) | set(reference_dict.keys())

    for level in sorted(all_levels):
        actual = actual_dict.get(level, {})
        reference = reference_dict.get(level, {})

        # Extract observation counts
        actual_obs_count = actual.get('observe_count', 0)
        actual_mean = actual.get('mean_observe_per_activation', np.nan)
        actual_t = actual.get('t', 0)
        actual_agent2 = actual.get('agent2_count', 0)
        actual_agent3 = actual.get('agent3_count', 0)

        ref_obs_count = reference.get('t', 0)  # 't' in reference is total observations
        ref_agent2 = reference.get('agent2_count', 0)
        ref_agent3 = reference.get('agent3_count', 0)
        ref_mean = reference.get('mean_observe_per_activation', np.nan)

        # Calculate differences
        obs_count_diff = actual_obs_count - ref_obs_count
        agent2_diff = actual_agent2 - ref_agent2
        agent3_diff = actual_agent3 - ref_agent3
        mean_diff = actual_mean - ref_mean if not np.isnan(actual_mean) and not np.isnan(ref_mean) else np.nan

        # Calculate percent differences (avoid division by zero)
        obs_pct_diff = (obs_count_diff / ref_obs_count * 100) if ref_obs_count > 0 else np.nan
        agent2_pct_diff = (agent2_diff / ref_agent2 * 100) if ref_agent2 > 0 else np.nan
        agent3_pct_diff = (agent3_diff / ref_agent3 * 100) if ref_agent3 > 0 else np.nan

        comparison_rows.append({
            'level': level,
            'actual_obs_count': actual_obs_count,
            'ref_obs_count': ref_obs_count,
            'obs_diff': obs_count_diff,
            'obs_pct_diff': obs_pct_diff,
            'actual_agent2': actual_agent2,
            'ref_agent2': ref_agent2,
            'agent2_diff': agent2_diff,
            'agent2_pct_diff': agent2_pct_diff,
            'actual_agent3': actual_agent3,
            'ref_agent3': ref_agent3,
            'agent3_diff': agent3_diff,
            'agent3_pct_diff': agent3_pct_diff,
            'actual_mean': actual_mean,
            'ref_mean': ref_mean,
            'mean_diff': mean_diff,
        })

    return pd.DataFrame(comparison_rows)


def calculate_correlation(comparison_df: pd.DataFrame) -> Dict:
    """Calculate correlation statistics between actual and reference."""
    # Filter out rows with missing data
    valid_obs = comparison_df.dropna(subset=['actual_obs_count', 'ref_obs_count'])
    valid_agent2 = comparison_df.dropna(subset=['actual_agent2', 'ref_agent2'])
    valid_agent3 = comparison_df.dropna(subset=['actual_agent3', 'ref_agent3'])
    valid_mean = comparison_df.dropna(subset=['actual_mean', 'ref_mean'])

    results = {}

    # Observation count correlation
    if len(valid_obs) > 1:
        r_obs, p_obs = stats.pearsonr(valid_obs['actual_obs_count'], valid_obs['ref_obs_count'])
        results['obs_correlation'] = {'r': r_obs, 'p': p_obs, 'n': len(valid_obs)}

    # Agent2 correlation
    if len(valid_agent2) > 1:
        r_a2, p_a2 = stats.pearsonr(valid_agent2['actual_agent2'], valid_agent2['ref_agent2'])
        results['agent2_correlation'] = {'r': r_a2, 'p': p_a2, 'n': len(valid_agent2)}

    # Agent3 correlation
    if len(valid_agent3) > 1:
        r_a3, p_a3 = stats.pearsonr(valid_agent3['actual_agent3'], valid_agent3['ref_agent3'])
        results['agent3_correlation'] = {'r': r_a3, 'p': p_a3, 'n': len(valid_agent3)}

    # Mean observations per activation correlation
    if len(valid_mean) > 1:
        r_mean, p_mean = stats.pearsonr(valid_mean['actual_mean'], valid_mean['ref_mean'])
        results['mean_correlation'] = {'r': r_mean, 'p': p_mean, 'n': len(valid_mean)}

    return results


def main(json_path: str, reference_path: str, output_dir: str = None):
    """
    Main pipeline function.

    Args:
        json_path: Path to input JSON file
        reference_path: Path to reference dictionary JSON
        output_dir: Optional output directory for results
    """
    json_path = Path(json_path)
    reference_path = Path(reference_path)

    if output_dir is None:
        output_dir = json_path.parent / f"analysis_{json_path.stem}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUTOMATED OBSERVATION ANALYSIS PIPELINE")
    print("=" * 80)

    # Step 1: Convert JSON to CSV
    print("\n[Step 1/4] Converting JSON to CSV...")
    csv_dir = output_dir / "csv_files"
    csv_files = build_report_files(str(json_path), str(csv_dir))
    print(f"✓ Generated {len(csv_files)} CSV files in {csv_dir}")

    # Step 2: Parse observations from CSV
    print("\n[Step 2/4] Parsing observations from CSV files...")
    csv_paths = list(Path(csv_dir).glob("*.csv"))
    per_file, overall, overall_dict, overall_json = count_observes_and_means(csv_paths)
    print(f"✓ Parsed {len(csv_paths)} CSV files")
    print(f"✓ Found {len(overall_dict)} unique levels")

    # Save parsed results
    per_file_path = output_dir / "per_file_observations.csv"
    overall_path = output_dir / "overall_observations.csv"
    overall_json_path = output_dir / "overall_observations.json"

    per_file.to_csv(per_file_path, index=False)
    overall.to_csv(overall_path, index=False)

    with open(overall_json_path, 'w') as f:
        f.write(overall_json)

    print(f"✓ Saved per-file results to {per_file_path.name}")
    print(f"✓ Saved overall results to {overall_path.name}")
    print(f"✓ Saved JSON results to {overall_json_path.name}")

    # Step 3: Load reference dictionary
    print("\n[Step 3/4] Loading reference dictionary...")
    reference_dict = load_reference_dict(reference_path)
    print(f"✓ Loaded reference with {len(reference_dict)} levels")

    # Step 4: Compare and correlate
    print("\n[Step 4/4] Comparing to reference and calculating correlations...")
    comparison_df = compare_observations(overall_dict, reference_dict)
    correlation_stats = calculate_correlation(comparison_df)

    # Save comparison results
    comparison_path = output_dir / "comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison to {comparison_path.name}")

    # Save correlation stats
    correlation_path = output_dir / "correlation_stats.json"
    with open(correlation_path, 'w') as f:
        json.dump(correlation_stats, f, indent=2)
    print(f"✓ Saved correlation stats to {correlation_path.name}")

    # Print summary
    print("\n" + "=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)

    for metric, stats_dict in correlation_stats.items():
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  Pearson r: {stats_dict['r']:.4f}")
        print(f"  p-value: {stats_dict['p']:.4e}")
        print(f"  n: {stats_dict['n']}")
        print(f"  Interpretation: {'Significant' if stats_dict['p'] < 0.05 else 'Not significant'} at α=0.05")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {per_file_path.name} - Per-file observation counts")
    print(f"  - {overall_path.name} - Overall observation statistics")
    print(f"  - {overall_json_path.name} - JSON format of overall stats")
    print(f"  - {comparison_path.name} - Side-by-side comparison")
    print(f"  - {correlation_path.name} - Correlation statistics")
    print(f"  - csv_files/ - Individual CSV files for each user")

    return comparison_df, correlation_stats


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_and_compare.py <json_file> <reference_dict> [output_dir]")
        print("\nExample:")
        print("  python process_and_compare.py data_2026-01-23_21-00-54.json \\")
        print("         results/dictionaries/steps_dict_exp4_point5_updated.json \\")
        print("         analysis_output")
        sys.exit(1)

    json_file = sys.argv[1]
    reference_file = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else None

    main(json_file, reference_file, output_directory)
