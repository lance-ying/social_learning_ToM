#!/usr/bin/env python3
"""
Remove duplicate CSV files from a directory based on user ID.
Keeps the first occurrence of each user ID and removes subsequent duplicates.
"""
import re
from pathlib import Path
from typing import Set, List
import sys


def find_and_remove_duplicates(csv_dir: Path, dry_run: bool = True) -> None:
    """
    Find and remove duplicate CSV files based on user ID.
    
    Args:
        csv_dir: Directory containing CSV files
        dry_run: If True, only report what would be deleted without actually deleting
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return
    
    print("="*80)
    print(f"SCANNING: {csv_dir}")
    print("="*80)
    print(f"Found {len(csv_files)} CSV files\n")
    
    seen_user_ids: Set[str] = set()
    duplicates: List[Path] = []
    unique_files: List[Path] = []
    
    for csv_file in csv_files:
        # Extract user ID from filename
        filename = csv_file.name
        user_id_match = re.search(r'u\d+', filename)
        
        if user_id_match:
            user_id = user_id_match.group(0)
            if user_id in seen_user_ids:
                duplicates.append(csv_file)
            else:
                seen_user_ids.add(user_id)
                unique_files.append(csv_file)
        else:
            # File doesn't match pattern, keep it
            unique_files.append(csv_file)
    
    print(f"Unique files: {len(unique_files)}")
    print(f"Duplicate files: {len(duplicates)}")
    
    if duplicates:
        print(f"\nDuplicate files to {'remove' if not dry_run else 'be removed'}:")
        for dup in duplicates[:10]:  # Show first 10
            print(f"  - {dup.name}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
        
        if not dry_run:
            print(f"\nRemoving {len(duplicates)} duplicate files...")
            for dup in duplicates:
                dup.unlink()
                print(f"  ✓ Removed: {dup.name}")
            print(f"\n✓ Done! Removed {len(duplicates)} duplicate files")
            print(f"  Remaining: {len(unique_files)} unique files")
        else:
            print(f"\n⚠ DRY RUN - No files were actually deleted")
            print(f"  Run with --execute to actually remove duplicates")
    else:
        print("\n✓ No duplicates found!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_duplicate_csvs.py <csv_directory> [--execute]")
        print("\nExample:")
        print("  python remove_duplicate_csvs.py data_processed/exp1_combined  # Dry run")
        print("  python remove_duplicate_csvs.py data_processed/exp1_combined --execute  # Actually remove")
        sys.exit(1)
    
    csv_dir = Path(sys.argv[1])
    dry_run = "--execute" not in sys.argv
    
    if not csv_dir.exists():
        print(f"❌ Error: Directory {csv_dir} does not exist")
        sys.exit(1)
    
    find_and_remove_duplicates(csv_dir, dry_run=dry_run)

