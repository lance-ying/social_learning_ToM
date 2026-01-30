#!/usr/bin/env python3
"""
Convert bonus CSV to simple format: prolificid,X.XX
"""

import pandas as pd
import sys
from pathlib import Path


def format_bonuses_simple(input_csv, output_txt=None):
    """
    Convert bonuses CSV to simple format with just prolific_id and bonus.

    Args:
        input_csv: Path to input CSV file
        output_txt: Path to output txt file (defaults to bonuses_simple.txt)
    """
    if output_txt is None:
        output_txt = Path(input_csv).parent / "bonuses_simple.txt"

    # Read input CSV
    df = pd.read_csv(input_csv)

    # Filter out zero bonuses first
    df = df[df['bonus_payment'] > 0]

    # Select and rename columns
    simple_df = df[['prolific_id', 'bonus_payment']].copy()
    simple_df.columns = ['prolificid', 'bonus']

    # Format bonus to 2 decimal places
    simple_df['bonus'] = simple_df['bonus'].apply(lambda x: f"{x:.2f}")

    # Save to txt file
    with open(output_txt, 'w') as f:
        for _, row in simple_df.iterrows():
            f.write(f"{row['prolificid']},{row['bonus']}\n")

    print(f"✓ Saved {len(simple_df)} participants to {output_txt}")
    print(f"\nFile ready to copy from: {output_txt}")
    print(f"\nPreview (first 5 entries):")
    for i, (_, row) in enumerate(simple_df.iterrows()):
        if i >= 5:
            break
        print(f"  {row['prolificid']},{row['bonus']}")

    return output_txt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_file = Path("participant_bonuses_summary.csv")
    else:
        input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)

    output_file = input_file.parent / "bonuses_simple.txt"
    format_bonuses_simple(input_file, output_file)
