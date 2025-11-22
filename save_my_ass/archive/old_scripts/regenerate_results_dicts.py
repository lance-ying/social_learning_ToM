#!/usr/bin/env python3
"""
Regenerate results dictionaries by:
1. Combining CSV files from different sources
2. Using observe_parser_1007.py to generate statistics
"""
import shutil
from pathlib import Path
from observe_parser_1007 import count_observes_and_means
import json


def find_mod_csvs(base_dir: Path) -> list[Path]:
    """Find all CSV files that contain mod_XXX levels."""
    mod_csvs = []
    exclude_dirs = ["archive", "filtered_out"]
    
    for subdir in base_dir.iterdir():
        if not subdir.is_dir() or subdir.name in exclude_dirs:
            continue
        
        for csv_file in subdir.glob("*.csv"):
            # Check if file has mod_ levels
            try:
                with csv_file.open('r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Level: mod_' in content:
                        mod_csvs.append(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return mod_csvs


def find_s_csvs(base_dir: Path) -> list[Path]:
    """Find all CSV files that contain s_XXX levels (but not mod_XXX)."""
    s_csvs = []
    exclude_dirs = ["archive", "filtered_out"]
    
    for subdir in base_dir.iterdir():
        if not subdir.is_dir() or subdir.name in exclude_dirs:
            continue
        
        for csv_file in subdir.glob("*.csv"):
            # Check if file has s_ levels but not mod_
            try:
                with csv_file.open('r', encoding='utf-8') as f:
                    content = f.read()
                    if 'Level: s' in content and 'Level: mod_' not in content:
                        s_csvs.append(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return s_csvs


def combine_and_process_exp1(pilot_dir: Path, mod_csvs: list[Path], output_dir: Path):
    """Combine pilot_100725_exp1 files with mod_XXX files and process."""
    print("\n" + "="*80)
    print("EXP1: Combining pilot_100725_exp1 + mod_XXX CSV files")
    print("="*80)
    
    # Create combined directory
    combined_dir = output_dir / "exp1_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy pilot files
    pilot_files = list(pilot_dir.glob("*.csv"))
    print(f"\nCopying {len(pilot_files)} files from {pilot_dir.name}/")
    for f in pilot_files:
        shutil.copy2(f, combined_dir / f.name)
    
    # Copy mod_XXX files (deduplicate by user ID to avoid processing same participant twice)
    print(f"Copying {len(mod_csvs)} mod_XXX CSV files")
    seen_user_ids = set()
    copied_count = 0
    skipped_count = 0
    
    for csv_file in mod_csvs:
        # Extract user ID from filename (format: uXXXXXXXXX.csv or dir_uXXXXXXXXX.csv)
        filename = csv_file.name
        # Try to extract user ID pattern
        import re
        user_id_match = re.search(r'u\d+', filename)
        if user_id_match:
            user_id = user_id_match.group(0)
            if user_id in seen_user_ids:
                # Skip duplicate user ID
                skipped_count += 1
                continue
            seen_user_ids.add(user_id)
        
        dest_name = csv_file.name
        dest_path = combined_dir / dest_name
        
        # If filename conflicts, add directory name prefix
        if dest_path.exists():
            dest_name = f"{csv_file.parent.name}_{csv_file.name}"
            dest_path = combined_dir / dest_name
        
        shutil.copy2(csv_file, dest_path)
        copied_count += 1
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} duplicate user IDs")
    print(f"  Actually copied {copied_count} unique files")
    
    total_files = len(pilot_files) + copied_count
    print(f"\nTotal files in combined directory: {total_files}")
    
    # Process with observe_parser
    print("\nProcessing files with observe_parser...")
    csv_files = list(combined_dir.glob("*.csv"))
    per_file, overall, dict_result, json_result = count_observes_and_means(csv_files)
    
    # Save results
    output_file = output_dir / "results_dict_exp1_50.py"
    with open(output_file, "w") as f:
        f.write(f"# Generated observe statistics for EXP1\n")
        f.write(f"# Combined: {len(pilot_files)} pilot_100725_exp1 files + {len(mod_csvs)} mod_XXX files\n")
        f.write(f"results_dict = {repr(dict_result)}\n")
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total levels: {len(dict_result)}")
    
    return dict_result


def combine_and_process_exp2(exp2_dirs: list[Path], output_dir: Path):
    """Process CSV files from specific directories for exp2."""
    print("\n" + "="*80)
    print("EXP2: Processing CSV files from specified directories")
    print("="*80)
    
    # Create combined directory
    combined_dir = output_dir / "exp2_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all CSV files from the specified directories
    all_csvs = []
    for exp2_dir in exp2_dirs:
        if not exp2_dir.exists():
            print(f"⚠ Warning: {exp2_dir} does not exist, skipping")
            continue
        
        csv_files = list(exp2_dir.glob("*.csv"))
        all_csvs.extend(csv_files)
        print(f"  Found {len(csv_files)} CSV files in {exp2_dir.name}/")
    
    # Copy CSV files (deduplicate by user ID to avoid processing same participant twice)
    print(f"\nCopying {len(all_csvs)} CSV files")
    seen_user_ids = set()
    copied_count = 0
    skipped_count = 0
    
    for csv_file in all_csvs:
        # Extract user ID from filename
        filename = csv_file.name
        import re
        user_id_match = re.search(r'u\d+', filename)
        if user_id_match:
            user_id = user_id_match.group(0)
            if user_id in seen_user_ids:
                # Skip duplicate user ID
                skipped_count += 1
                continue
            seen_user_ids.add(user_id)
        
        dest_name = csv_file.name
        dest_path = combined_dir / dest_name
        
        # If filename conflicts, add directory name prefix
        if dest_path.exists():
            dest_name = f"{csv_file.parent.name}_{csv_file.name}"
            dest_path = combined_dir / dest_name
        
        shutil.copy2(csv_file, dest_path)
        copied_count += 1
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} duplicate user IDs")
    print(f"  Actually copied {copied_count} unique files")
    
    print(f"\nTotal files in combined directory: {copied_count}")
    
    # Process with observe_parser
    print("\nProcessing files with observe_parser...")
    csv_files = list(combined_dir.glob("*.csv"))
    per_file, overall, dict_result, json_result = count_observes_and_means(csv_files)
    
    # Save results
    output_file = output_dir / "results_dict_merged_exp2.py"
    with open(output_file, "w") as f:
        f.write(f"# Generated observe statistics for EXP2\n")
        f.write(f"# Combined from directories: {', '.join(d.name for d in exp2_dirs)}\n")
        f.write(f"# Total files: {len(all_csvs)}\n")
        f.write(f"results_dict = {repr(dict_result)}\n")
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total levels: {len(dict_result)}")
    
    return dict_result


if __name__ == "__main__":
    import sys
    
    # Directories
    csv_reports_dir = Path("restrcuture_downloaded_data/csv_reports")
    pilot_dir = Path("raw_data/pilot_100725_exp1")
    output_dir = Path("regenerated_results")
    
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("REGENERATING RESULTS DICTIONARIES")
    print("="*80)
    
    # Find mod_XXX CSV files
    print("\nFinding mod_XXX CSV files...")
    mod_csvs = find_mod_csvs(csv_reports_dir)
    print(f"Found {len(mod_csvs)} mod_XXX CSV files")
    
    # Process EXP1: pilot_100725_exp1 + mod_XXX files
    if pilot_dir.exists():
        exp1_dict = combine_and_process_exp1(pilot_dir, mod_csvs, output_dir)
    else:
        print(f"\n⚠ Warning: {pilot_dir} not found, skipping EXP1")
    
    # Process EXP2: Use specific directories
    exp2_dirs = [
        csv_reports_dir / "data_2025-10-21_02-50-38",
        csv_reports_dir / "data_2025-10-20_16-21-17",
        csv_reports_dir / "data_2025-10-18_23-49-16",
    ]
    
    print(f"\nEXP2 directories:")
    for d in exp2_dirs:
        exists = "✓" if d.exists() else "✗"
        print(f"  {exists} {d}")
    
    if any(d.exists() for d in exp2_dirs):
        exp2_dict = combine_and_process_exp2(exp2_dirs, output_dir)
    else:
        print(f"\n⚠ Warning: None of the EXP2 directories found, skipping EXP2")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}/")
    print(f"  - exp1_combined/ (combined CSV files)")
    print(f"  - exp2_combined/ (combined CSV files)")
    print(f"  - results_dict_exp1_50.py")
    print(f"  - results_dict_merged_exp2.py")

