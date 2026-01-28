#!/usr/bin/env python3
"""
Run correlation analysis on filtered participants based on performance criteria.
"""
import json
import sys
from pathlib import Path
import pandas as pd

from observe_parser_exp4 import count_observes_and_means
from correlation_exp4 import main as run_correlation


def filter_participants_by_score(scores_csv: Path, min_score: float = 0):
    """
    Filter participants based on their total score.

    Args:
        scores_csv: Path to participant_scores.csv
        min_score: Minimum total_steps_remaining to include

    Returns:
        List of participant IDs that meet the criteria
    """
    df = pd.read_csv(scores_csv)
    filtered = df[df['total_steps_remaining'] > min_score]

    print(f"Filtering participants with score > {min_score}:")
    print(f"  Total participants: {len(df)}")
    print(f"  Filtered participants: {len(filtered)}")
    print(f"  Excluded: {len(df) - len(filtered)}")
    print()

    if len(filtered) > 0:
        print("Included participants:")
        for _, row in filtered.iterrows():
            print(f"  {row['participant_id']}: {row['total_steps_remaining']} steps")
        print()

        print("Excluded participants:")
        excluded = df[df['total_steps_remaining'] <= min_score]
        for _, row in excluded.iterrows():
            print(f"  {row['participant_id']}: {row['total_steps_remaining']} steps")
        print()

    return filtered['participant_id'].tolist()


def main(analysis_dir: str, min_score: float = 0, output_suffix: str = "filtered"):
    """
    Run filtered correlation analysis.

    Args:
        analysis_dir: Directory containing the full analysis
        min_score: Minimum score threshold for inclusion
        output_suffix: Suffix for output directory
    """
    analysis_dir = Path(analysis_dir)

    # Paths
    scores_csv = analysis_dir / "per_level_analysis" / "participant_scores.csv"
    csv_dir = analysis_dir / "csv_files"

    script_dir = Path(__file__).parent
    data_processing_dir = script_dir.parent
    model_json = data_processing_dir / "results/dictionaries/steps_dict_exp4_point5_updated.json"

    output_dir = analysis_dir / f"correlation_{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"FILTERED CORRELATION ANALYSIS (score > {min_score})")
    print("=" * 80)
    print()

    # Get filtered participant IDs
    if not scores_csv.exists():
        print(f"Error: {scores_csv} not found. Run per_level_analysis.py first.")
        return

    filtered_ids = filter_participants_by_score(scores_csv, min_score)

    if len(filtered_ids) == 0:
        print("No participants meet the filtering criteria!")
        return

    # Get corresponding CSV files
    filtered_csv_files = []
    for participant_id in filtered_ids:
        csv_file = csv_dir / f"{participant_id}.csv"
        if csv_file.exists():
            filtered_csv_files.append(csv_file)

    print(f"Found {len(filtered_csv_files)} CSV files for filtered participants")
    print()

    # Parse observations from filtered CSVs
    print("[1/2] Parsing observations from filtered participants...")
    per_file, overall, overall_dict, overall_json = count_observes_and_means(filtered_csv_files)
    print(f"✓ Parsed {len(filtered_csv_files)} CSV files")
    print(f"✓ Found {len(overall_dict)} unique levels")

    # Save filtered results
    per_file_path = output_dir / "per_file_observations.csv"
    overall_path = output_dir / "overall_observations.csv"
    overall_json_path = output_dir / "overall_observations.json"

    per_file.to_csv(per_file_path, index=False)
    overall.to_csv(overall_path, index=False)

    with open(overall_json_path, 'w') as f:
        f.write(overall_json)

    print(f"✓ Saved filtered results")
    print()

    # Run correlation analysis
    print("[2/2] Calculating correlations with model predictions...")
    plots_dir = output_dir / "plots"
    correlation_stats = run_correlation(
        str(overall_json_path),
        str(model_json),
        str(plots_dir)
    )

    # Save filter metadata
    metadata = {
        "filter_criterion": f"total_steps_remaining > {min_score}",
        "min_score": min_score,
        "total_participants": len(filtered_ids),
        "filtered_participant_ids": filtered_ids,
        "correlation_stats": correlation_stats
    }

    metadata_path = output_dir / "filter_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 80)
    print("FILTERED ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Filter: {len(filtered_ids)} participants with score > {min_score}")

    return correlation_stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filtered_correlation.py <analysis_directory> [min_score]")
        print("\nExample:")
        print("  python filtered_correlation.py analysis_pilot_exp4 0")
        print("  python filtered_correlation.py analysis_pilot_exp4 10")
        sys.exit(1)

    analysis_directory = sys.argv[1]
    minimum_score = float(sys.argv[2]) if len(sys.argv) > 2 else 0

    main(analysis_directory, minimum_score)
