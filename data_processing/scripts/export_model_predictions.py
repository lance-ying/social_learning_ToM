#!/usr/bin/env python3
"""
Export model predictions to CSV format.
Simple script to convert a steps_dict JSON to CSV with agent2 and agent3 columns.
"""

import json
import sys
from pathlib import Path
import pandas as pd


def export_predictions_to_csv(json_path: str, output_csv: str = None):
    """
    Export model predictions to CSV.

    Args:
        json_path: Path to steps_dict JSON file
        output_csv: Optional output CSV path (defaults to same name as JSON with .csv)
    """
    json_path = Path(json_path)

    if output_csv is None:
        output_csv = json_path.with_suffix('.csv')
    else:
        output_csv = Path(output_csv)

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loading: {json_path}")
    print(f"Found {len(data)} levels")

    # Extract data
    rows = []
    for level, level_data in sorted(data.items()):
        agent2 = level_data.get('agent2_count', 0)
        agent3 = level_data.get('agent3_count', 0)

        rows.append({
            'level': level,
            'agent2': agent2,
            'agent3': agent3
        })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"✓ Saved {len(df)} levels to {output_csv}")
    print(f"\nPreview:")
    print(df.head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_model_predictions.py <steps_dict.json> [output.csv]")
        print("\nExample:")
        print("  python export_model_predictions.py steps_dict_exp3.json")
        print("  python export_model_predictions.py steps_dict_exp3.json model_predictions.csv")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    export_predictions_to_csv(json_file, output_file)
