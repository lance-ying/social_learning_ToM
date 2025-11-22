#!/usr/bin/env python3
"""
Filter CSV files to only keep those that have a prolificId field.
Files without prolificId will be moved to a 'filtered_out' subdirectory.
"""
import csv
from pathlib import Path
from typing import List, Tuple
import shutil


def has_prolific_id(csv_path: Path) -> bool:
    """Check if a CSV file contains a prolificId field in the User Data section."""
    try:
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.reader(f)
            in_user_data = False
            
            for row in reader:
                if not row:  # Empty row
                    continue
                
                # Check if we're entering the User Data section
                if len(row) > 0 and row[0] == "User Data":
                    in_user_data = True
                    continue
                
                # If we hit another section header (non-empty row that's not a key-value pair), exit User Data
                if in_user_data and len(row) > 0:
                    # Check if this is a section header (starts with capital letter and has no comma-separated value)
                    if row[0] and not row[0].startswith(('(', '[')) and len(row) == 1:
                        # Might be a new section, but let's be more careful
                        # Actually, if row[0] is not a key-value format, we've left User Data
                        if row[0] not in ["prolificId", "age", "gender", "feedback"] and not any(c in row[0].lower() for c in ['id', 'age', 'gender']):
                            # Check if it looks like a section header
                            if row[0][0].isupper() and len(row[0]) > 3:
                                in_user_data = False
                                continue
                
                # Check for prolificId in User Data section
                if in_user_data and len(row) >= 1 and row[0] == "prolificId":
                    return True
                
                # If we hit Comprehension Check, we've passed User Data
                if len(row) > 0 and row[0] == "Comprehension Check":
                    break
            
        return False
    except Exception as e:
        print(f"Error reading {csv_path.name}: {e}")
        return False


def filter_csvs_in_directory(csv_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """
    Filter CSV files in a directory, keeping only those with prolificId.
    Returns (kept_count, removed_count)
    """
    csv_files = list(csv_dir.glob("*.csv"))
    kept_count = 0
    removed_count = 0
    
    # Create filtered_out subdirectory if needed
    filtered_out_dir = csv_dir / "filtered_out"
    
    for csv_file in csv_files:
        if has_prolific_id(csv_file):
            kept_count += 1
            if not dry_run:
                print(f"  ✓ Keeping: {csv_file.name}")
        else:
            removed_count += 1
            if not dry_run:
                filtered_out_dir.mkdir(exist_ok=True)
                dest = filtered_out_dir / csv_file.name
                shutil.move(str(csv_file), str(dest))
                print(f"  ✗ Moving: {csv_file.name} -> filtered_out/")
            else:
                print(f"  ✗ Would move: {csv_file.name}")
    
    return kept_count, removed_count


def filter_all_directories(base_dir: Path, exclude_dirs: List[str] = None, dry_run: bool = False) -> None:
    """
    Filter CSV files in all subdirectories of base_dir.
    exclude_dirs: list of directory names to skip
    """
    if exclude_dirs is None:
        exclude_dirs = ["archive", "filtered_out"]
    
    base_dir = Path(base_dir)
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name not in exclude_dirs]
    
    print(f"Found {len(subdirs)} directories to process...\n")
    
    total_kept = 0
    total_removed = 0
    
    for subdir in sorted(subdirs):
        print(f"Processing: {subdir.name}/")
        kept, removed = filter_csvs_in_directory(subdir, dry_run=dry_run)
        total_kept += kept
        total_removed += removed
        print(f"  Kept: {kept}, Removed: {removed}\n")
    
    print(f"Summary:")
    print(f"  Total CSV files kept: {total_kept}")
    print(f"  Total CSV files removed: {total_removed}")
    if dry_run:
        print(f"\n(DRY RUN - no files were actually moved)")


if __name__ == "__main__":
    import sys
    
    base_dir = Path("restrcuture_downloaded_data/csv_reports")
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
    
    if len(sys.argv) > 1 and sys.argv[1] not in ["--dry-run", "-n"]:
        base_dir = Path(sys.argv[1])
    
    if dry_run:
        print("DRY RUN MODE - No files will be moved\n")
    
    filter_all_directories(base_dir, dry_run=dry_run)

