#!/usr/bin/env python3
"""
Split smXXX.txt visualization files into per-scenario files with model predictions.

Usage:
    python3 split_viz_scenarios.py <viz_directory>
"""

import sys
import csv
from pathlib import Path


def parse_csv(csv_path):
    """Parse CSV into a dict: { "sm211_scenario1": {"agent2": 1, "agent3": 5}, ... }"""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['level']
            data[key] = {
                'agent2': int(row['agent2']),
                'agent3': int(row['agent3']),
            }
    return data


def strip_suffix(value_str):
    """Strip trailing 'a' or 'n' from a value string like '1a' or '3n'."""
    s = value_str.strip()
    if s and s[-1] in ('a', 'n'):
        return s[:-1]
    return s


def parse_xy_line(line):
    """Parse 'X: 1a, 2a' into ['1a', '2a']."""
    # Format: "X: 1a, 2a"
    parts = line.split(':', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid X/Y line: {line}")
    values_str = parts[1].strip()
    values = [v.strip() for v in values_str.split(',')]
    return values


def process_file(txt_path, csv_data, output_dir):
    """Process a single smXXX.txt file and write two output files."""
    with open(txt_path, 'r') as f:
        content = f.read()

    # Split on blank line
    parts = content.split('\n\n')
    if len(parts) < 2:
        raise ValueError(f"File does not have expected blank line separator: {txt_path}")

    ascii_map = parts[0]
    xy_section = '\n'.join(parts[1:]).strip()

    lines = xy_section.split('\n')
    x_line = lines[0]
    y_line = lines[1] if len(lines) > 1 else ""

    # Parse X and Y values
    x_values = parse_xy_line(x_line)
    y_values = parse_xy_line(y_line)

    if len(x_values) != 2 or len(y_values) != 2:
        raise ValueError(f"Expected 2 values for X and Y in {txt_path}")

    # Get filename without extension
    filename = txt_path.stem

    # Process for both scenarios
    for scenario_idx in range(2):
        scenario_num = scenario_idx + 1  # 1 or 2

        # Get ground truth values (strip a/n suffix)
        x_gt = strip_suffix(x_values[scenario_idx])
        y_gt = strip_suffix(y_values[scenario_idx])

        # Look up model predictions from CSV
        csv_key = f"{filename}_scenario{scenario_num}"
        if csv_key not in csv_data:
            raise KeyError(f"Key not found in CSV: {csv_key}")

        x_pred = csv_data[csv_key]['agent2']
        y_pred = csv_data[csv_key]['agent3']

        # Write output file
        output_filename = f"{filename}_{scenario_num}.txt"
        output_path = output_dir / output_filename

        # Create ASCII table format
        table = (
            "     G  M\n"
            f"X:   {x_gt}  {x_pred}\n"
            f"Y:   {y_gt}  {y_pred}"
        )
        output_content = f"{ascii_map}\n\n{table}\n"

        with open(output_path, 'w') as f:
            f.write(output_content)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 split_viz_scenarios.py <viz_directory>", file=sys.stderr)
        sys.exit(1)

    viz_dir = Path(sys.argv[1])
    if not viz_dir.is_dir():
        print(f"Error: {viz_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find CSV file
    csv_path = viz_dir / "steps_dict_exp4_013026.csv"
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Parse CSV
    csv_data = parse_csv(csv_path)
    print(f"Loaded {len(csv_data)} entries from CSV")

    # Find all smXXX.txt files
    txt_files = sorted(viz_dir.glob("sm*.txt"))
    txt_files = [f for f in txt_files if f.name.startswith("sm") and f.name[2:5].isdigit()]

    print(f"Found {len(txt_files)} smXXX.txt files")

    # Process each file
    for txt_path in txt_files:
        try:
            process_file(txt_path, csv_data, viz_dir)
            print(f"  Processed {txt_path.name}")
        except Exception as e:
            print(f"  ERROR processing {txt_path.name}: {e}", file=sys.stderr)
            sys.exit(1)

    output_count = sum(1 for f in viz_dir.glob("sm*_[12].txt"))
    print(f"Successfully created {output_count} output files")


if __name__ == "__main__":
    main()
