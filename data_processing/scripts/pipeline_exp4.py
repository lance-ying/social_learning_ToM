#!/usr/bin/env python3
"""
End-to-end pipeline for Exp4 analysis:
1. Convert JSON to CSV
2. Parse observations (with agent tracking)
3. Calculate correlations with model predictions
"""

import sys
from pathlib import Path

# Import our modules
from extract_json import build_report_files
from observe_parser_exp4 import count_observes_and_means
from correlation_exp4 import main as run_correlation


def pipeline(json_path: str, model_json_path: str, output_dir: str = None):
    """
    Run the full analysis pipeline.

    Args:
        json_path: Path to raw JSON data file
        model_json_path: Path to model predictions JSON
        output_dir: Optional output directory (default: analysis_exp4)
    """
    json_path = Path(json_path)
    model_json_path = Path(model_json_path)

    if output_dir is None:
        output_dir = json_path.parent / "analysis_exp4"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXP4 ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"JSON data: {json_path.name}")
    print(f"Model predictions: {model_json_path.name}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Convert JSON to CSV
    print("[Step 1/3] Converting JSON to CSV...")
    csv_dir = output_dir / "csv_files"
    csv_files = build_report_files(str(json_path), str(csv_dir))
    print(f"✓ Generated {len(csv_files)} CSV files")

    # Step 2: Parse observations
    print("\n[Step 2/3] Parsing observations from CSV files...")
    csv_paths = list(Path(csv_dir).glob("*.csv"))
    per_file, overall, overall_dict, overall_json = count_observes_and_means(csv_paths)
    print(f"✓ Parsed {len(csv_paths)} CSV files")
    print(f"✓ Found {len(overall_dict)} unique levels")

    # Save results
    per_file_path = output_dir / "per_file_observations.csv"
    overall_path = output_dir / "overall_observations.csv"
    overall_json_path = output_dir / "overall_observations.json"

    per_file.to_csv(per_file_path, index=False)
    overall.to_csv(overall_path, index=False)

    with open(overall_json_path, 'w') as f:
        f.write(overall_json)

    print(f"✓ Saved results to {output_dir}")

    # Step 3: Calculate correlations
    print("\n[Step 3/3] Calculating correlations with model predictions...")
    plots_dir = output_dir / "plots"
    correlation_stats = run_correlation(
        str(overall_json_path),
        str(model_json_path),
        str(plots_dir)
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {per_file_path.name}")
    print(f"  - {overall_path.name}")
    print(f"  - {overall_json_path.name}")
    print(f"  - plots/correlation_exp4_agent2.png")
    print(f"  - plots/correlation_exp4_agent3.png")
    print(f"  - plots/correlation_stats_exp4.json")

    return correlation_stats


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline_exp4.py <json_file> <model_json> [output_dir]")
        print("\nExample:")
        print("  python pipeline_exp4.py \\")
        print("    data_2026-01-23_21-00-54.json \\")
        print("    results/dictionaries/steps_dict_exp4_point5_updated.json \\")
        print("    analysis_exp4")
        sys.exit(1)

    json_file = sys.argv[1]
    model_file = sys.argv[2]
    output_directory = sys.argv[3] if len(sys.argv) > 3 else None

    pipeline(json_file, model_file, output_directory)
