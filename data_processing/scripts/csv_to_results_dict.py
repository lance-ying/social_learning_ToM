#!/usr/bin/env python3
"""
Simple script: Take a directory of CSV files and generate a results_dict.
Handles deduplication by user ID automatically.
"""
import re
from pathlib import Path
from typing import List, Set
import sys

# Import the multi-NPC CSV parser
import importlib.util
spec = importlib.util.spec_from_file_location(
    "observe_parser_multi_npc",
    Path(__file__).parent / "observe_parser_multi_npc.py"
)
observe_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observe_parser)


def deduplicate_csvs_by_user_id(csv_files: List[Path]) -> List[Path]:
    """
    Deduplicate CSV files by extracting user IDs from filenames.
    Returns list of unique files (first occurrence kept).
    """
    seen_user_ids: Set[str] = set()
    unique_files = []
    skipped_files = []
    
    for csv_file in csv_files:
        # Extract user ID from filename (format: uXXXXXXXXX.csv or dir_uXXXXXXXXX.csv)
        filename = csv_file.name
        user_id_match = re.search(r'u\d+', filename)
        
        if user_id_match:
            user_id = user_id_match.group(0)
            if user_id in seen_user_ids:
                skipped_files.append(csv_file)
                continue
            seen_user_ids.add(user_id)
        
        unique_files.append(csv_file)
    
    if skipped_files:
        print(f"  Skipped {len(skipped_files)} duplicate user IDs:")
        for f in skipped_files[:5]:  # Show first 5
            print(f"    - {f.name}")
        if len(skipped_files) > 5:
            print(f"    ... and {len(skipped_files) - 5} more")
    
    return unique_files


def csv_dir_to_results_dict(
    csv_dir: Path,
    output_file: Path,
    description: str = ""
) -> dict:
    """
    Process all CSV files in a directory and generate a results_dict.
    
    Args:
        csv_dir: Directory containing CSV files
        output_file: Path to output Python file with results_dict
        description: Optional description to include in output file
    
    Returns:
        The results dictionary
    """
    print("="*80)
    print(f"PROCESSING CSV DIRECTORY: {csv_dir}")
    print("="*80)
    
    if not csv_dir.exists():
        print(f"❌ Error: Directory {csv_dir} does not exist")
        return {}
    
    # Find all CSV files
    all_csv_files = sorted(csv_dir.glob("*.csv"))
    print(f"\nFound {len(all_csv_files)} CSV files")
    
    if not all_csv_files:
        print("❌ No CSV files found")
        return {}
    
    # Deduplicate by user ID
    print(f"\nDeduplicating by user ID...")
    unique_csv_files = deduplicate_csvs_by_user_id(all_csv_files)
    print(f"  Processing {len(unique_csv_files)} unique files")
    
    # Process with observe_parser
    print(f"\nProcessing files...")
    per_file, overall, dict_result, json_result = observe_parser.count_observes_and_means(unique_csv_files)
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"# Generated observe statistics from CSV directory\n")
        if description:
            f.write(f"# {description}\n")
        f.write(f"# Directory: {csv_dir}\n")
        f.write(f"# Total CSV files: {len(all_csv_files)}\n")
        f.write(f"# Unique files (after deduplication): {len(unique_csv_files)}\n")
        f.write(f"# Only includes statistics from users with prolificId (filtered in CSV generation)\n")
        f.write(f"results_dict = {repr(dict_result)}\n")
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total levels: {len(dict_result)}")
    
    return dict_result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_to_results_dict.py <csv_directory> [output_file]")
        print("\nExample:")
        print("  python csv_to_results_dict.py data_processed/exp2_combined results/current/results_dict_exp2.py")
        sys.exit(1)
    
    csv_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results_dict.py")
    description = sys.argv[3] if len(sys.argv) > 3 else ""
    
    csv_dir_to_results_dict(csv_dir, output_file, description)

