#!/usr/bin/env python3
"""
Improved pipeline for regenerating results dictionaries:
- Works directly from JSON files (faster, no intermediate CSV step)
- Still supports existing CSV files (backward compatibility)
- Optional CSV generation for inspection/debugging
"""
import shutil
import re
from pathlib import Path
from typing import List, Set, Optional
import json

# Import the new direct JSON parser
try:
    from json_to_statistics import process_json_files_for_statistics
    USE_DIRECT_JSON = True
except ImportError as e:
    print(f"⚠ Error: json_to_statistics module not available: {e}")
    print("⚠ Please ensure json_to_statistics.py is in the scripts directory")
    raise


def find_json_files_with_level_type(base_dir: Path, level_pattern: str) -> List[Path]:
    """
    Find JSON files that contain specific level types by checking the JSON directly.
    Much faster than generating CSVs just to check level types.
    """
    json_files = []
    exclude_dirs = {"archive", "filtered_out", "__pycache__"}
    
    # Search for JSON files directly in base_dir
    for json_file in base_dir.glob("*.json"):
        try:
            content = json_file.read_text(encoding="utf-8")
            if content.startswith("version https://git-lfs.github.com"):
                continue
            
            data = json.loads(content)
            users = data.get("users", {})
            
            # Check if any user has the level type we're looking for
            for user_data in users.values():
                levels = (user_data or {}).get("levels", {})
                for level_id in levels.keys():
                    if level_pattern in level_id:
                        json_files.append(json_file)
                        break
                else:
                    continue
                break
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue
    
    # Also search in subdirectories
    for subdir in base_dir.iterdir():
        if not subdir.is_dir() or subdir.name in exclude_dirs:
            continue
        
        for json_file in subdir.glob("*.json"):
            try:
                content = json_file.read_text(encoding="utf-8")
                if content.startswith("version https://git-lfs.github.com"):
                    continue
                
                data = json.loads(content)
                users = data.get("users", {})
                
                # Check if any user has the level type we're looking for
                for user_data in users.values():
                    levels = (user_data or {}).get("levels", {})
                    for level_id in levels.keys():
                        if level_pattern in level_id:
                            json_files.append(json_file)
                            break
                    else:
                        continue
                    break
            except Exception as e:
                print(f"Error reading {json_file.name}: {e}")
                continue
    
    return json_files


def find_csv_files_with_level_type(base_dir: Path, level_pattern: str) -> List[Path]:
    """Find CSV files that contain specific level types (backward compatibility)."""
    csv_files = []
    exclude_dirs = {"archive", "filtered_out"}
    
    for subdir in base_dir.iterdir():
        if not subdir.is_dir() or subdir.name in exclude_dirs:
            continue
        
        for csv_file in subdir.glob("*.csv"):
            try:
                content = csv_file.read_text(encoding="utf-8")
                if f'Level: {level_pattern}' in content:
                    csv_files.append(csv_file)
            except Exception as e:
                print(f"Error reading {csv_file.name}: {e}")
    
    return csv_files


def deduplicate_by_user_id(files: List[Path], is_json: bool = False) -> List[Path]:
    """
    Deduplicate files by extracting user IDs from filenames or JSON content.
    Returns list of unique files (first occurrence kept).
    """
    seen_user_ids: Set[str] = set()
    unique_files = []
    
    for file_path in files:
        user_id = None
        
        # Try to extract from filename first
        user_id_match = re.search(r'u\d+', file_path.name)
        if user_id_match:
            user_id = user_id_match.group(0)
        elif is_json:
            # For JSON files, try to extract from content
            try:
                content = file_path.read_text(encoding="utf-8")
                if not content.startswith("version https://git-lfs.github.com"):
                    data = json.loads(content)
                    users = data.get("users", {})
                    if users:
                        # Get first user ID
                        user_id = list(users.keys())[0]
                        # Normalize to uXXXXXXXXX format
                        user_id_match = re.search(r'u\d+', str(user_id))
                        if user_id_match:
                            user_id = user_id_match.group(0)
            except Exception:
                pass
        
        if user_id:
            if user_id in seen_user_ids:
                continue
            seen_user_ids.add(user_id)
        
        unique_files.append(file_path)
    
    return unique_files


def combine_and_process_exp1_v2(
    pilot_json_dir: Optional[Path],
    all_json_files: List[Path],
    output_dir: Path,
    generate_csv: bool = False,
    csv_only_prolific: bool = True,
    stats_only_prolific: bool = True
):
    """
    Process EXP1 directly from JSON files.
    
    Args:
        pilot_json_dir: Optional directory for pilot JSONs (if None, all_json_files should contain all files)
        all_json_files: List of all JSON files to process (pilot + mod_XXX)
        output_dir: Output directory for results
    """
    print("\n" + "="*80)
    print("EXP1: Processing JSON files directly (pilot + mod_XXX)")
    print("="*80)
    
    # If pilot_json_dir is provided, add those files
    if pilot_json_dir and pilot_json_dir.exists():
        pilot_jsons = list(pilot_json_dir.glob("*.json"))
        all_json_files = pilot_jsons + all_json_files
        print(f"  Found {len(pilot_jsons)} pilot JSON files")
    
    print(f"  Total: {len(all_json_files)} JSON files")
    
    # Deduplicate by user ID
    print(f"\nDeduplicating {len(all_json_files)} files by user ID...")
    unique_files = deduplicate_by_user_id(all_json_files, is_json=True)
    skipped = len(all_json_files) - len(unique_files)
    if skipped > 0:
        print(f"  Skipped {skipped} duplicate user IDs")
    print(f"  Processing {len(unique_files)} unique files")
    
    # Process with direct JSON parser
    if USE_DIRECT_JSON:
        print("\nProcessing JSON files directly (fast path)...")
        csv_output_dir = output_dir / "exp1_combined_csvs" if generate_csv else None
        per_file, overall, dict_result, json_result = process_json_files_for_statistics(
            unique_files,
            generate_csv=generate_csv,
            csv_output_dir=csv_output_dir,
            csv_only_prolific=csv_only_prolific,
            stats_only_prolific=stats_only_prolific
        )
    else:
        # Fallback: would need to generate CSVs first
        print("⚠ Direct JSON processing not available, falling back to CSV method")
        raise NotImplementedError("CSV fallback not implemented in v2")
    
    # Save results
    output_file = output_dir / "results_dict_exp1_50.py"
    with open(output_file, "w") as f:
        f.write(f"# Generated observe statistics for EXP1\n")
        f.write(f"# Processed {len(unique_files)} unique JSON files directly\n")
        f.write(f"# Only includes statistics from users with prolificId\n")
        f.write(f"results_dict = {repr(dict_result)}\n")
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total levels: {len(dict_result)}")
    
    return dict_result


def combine_and_process_exp2_v2(
    exp2_json_files: List[Path],
    output_dir: Path,
    generate_csv: bool = False,
    csv_only_prolific: bool = True,
    stats_only_prolific: bool = True
):
    """
    Process EXP2 directly from specified JSON files.
    
    Args:
        exp2_json_files: List of JSON file paths to process
        output_dir: Output directory for results
    """
    print("\n" + "="*80)
    print("EXP2: Processing JSON files directly")
    print("="*80)
    
    # Filter to only existing files
    all_json_files = [f for f in exp2_json_files if f.exists()]
    print(f"  Processing {len(all_json_files)} JSON files")
    
    if not all_json_files:
        print("⚠ No JSON files found")
        return {}
    
    # Deduplicate by user ID
    print(f"\nDeduplicating {len(all_json_files)} files by user ID...")
    unique_files = deduplicate_by_user_id(all_json_files, is_json=True)
    skipped = len(all_json_files) - len(unique_files)
    if skipped > 0:
        print(f"  Skipped {skipped} duplicate user IDs")
    print(f"  Processing {len(unique_files)} unique files")
    
    # Process with direct JSON parser
    if USE_DIRECT_JSON:
        print("\nProcessing JSON files directly (fast path)...")
        csv_output_dir = output_dir / "exp2_combined_csvs" if generate_csv else None
        per_file, overall, dict_result, json_result = process_json_files_for_statistics(
            unique_files,
            generate_csv=generate_csv,
            csv_output_dir=csv_output_dir,
            csv_only_prolific=csv_only_prolific,
            stats_only_prolific=stats_only_prolific
        )
    else:
        raise NotImplementedError("CSV fallback not implemented in v2")
    
    # Save results
    output_file = output_dir / "results_dict_merged_exp2.py"
    with open(output_file, "w") as f:
        f.write(f"# Generated observe statistics for EXP2\n")
        f.write(f"# Combined from JSON files: {', '.join(f.name for f in all_json_files)}\n")
        f.write(f"# Processed {len(unique_files)} unique JSON files directly\n")
        f.write(f"# Only includes statistics from users with prolificId\n")
        f.write(f"results_dict = {repr(dict_result)}\n")
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Total levels: {len(dict_result)}")
    
    return dict_result


if __name__ == "__main__":
    import sys
    
    # Updated paths for new structure
    data_raw_dir = Path("data_raw")
    data_processed_dir = Path("data_processed")
    csv_reports_dir = data_processed_dir / "csv_reports"
    pilot_json_dir = data_processed_dir / "pilot_data" / "pilot_100725_exp1"  # Pilot CSVs are here, but we need JSONs
    # Actually, pilot JSONs should be in data_raw, let's check for pilot JSON files
    pilot_json_files = list(data_raw_dir.glob("pilot_*.json"))
    output_dir = Path("results/current")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generate_csv = "--csv" in sys.argv
    
    print("="*80)
    print("REGENERATING RESULTS DICTIONARIES (v2 - Direct JSON Processing)")
    print("="*80)
    
    if USE_DIRECT_JSON:
        print("✓ Using direct JSON processing (fast path)")
    else:
        print("⚠ Using CSV-based processing (slower)")
    
    # Find mod_XXX JSON files in data_raw
    print("\nFinding mod_XXX JSON files...")
    mod_json_files = find_json_files_with_level_type(data_raw_dir, "mod_")
    print(f"Found {len(mod_json_files)} mod_XXX JSON files")
    
    # Process EXP1: Combine pilot JSON files with mod_XXX JSON files
    all_exp1_json_files = pilot_json_files + mod_json_files
    if all_exp1_json_files:
        # Create a temporary list of JSON files for EXP1
        exp1_dict = combine_and_process_exp1_v2(
            None,  # No pilot_dir needed, we're passing files directly
            all_exp1_json_files,
            output_dir,
            generate_csv=generate_csv
        )
    else:
        print(f"\n⚠ Warning: No EXP1 JSON files found (pilot or mod_XXX)")
    
    # Process EXP2: Use specific JSON files that correspond to CSV directories
    exp2_json_files = [
        data_raw_dir / "data_2025-10-21_02-50-38.json",
        data_raw_dir / "data_2025-10-20_16-21-17.json",
        data_raw_dir / "data_2025-10-18_23-49-16.json",
    ]
    
    # Filter to only existing files
    exp2_json_files = [f for f in exp2_json_files if f.exists()]
    
    print(f"\nEXP2 JSON files:")
    for f in exp2_json_files:
        exists = "✓" if f.exists() else "✗"
        print(f"  {exists} {f.name}")
    
    if exp2_json_files:
        exp2_dict = combine_and_process_exp2_v2(
            exp2_json_files,  # Pass list of files, not directories
            output_dir,
            generate_csv=generate_csv
        )
    else:
        print(f"\n⚠ Warning: None of the EXP2 JSON files found, skipping EXP2")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}/")
    if generate_csv:
        print(f"  - exp1_combined_csvs/ (optional CSV files for inspection)")
        print(f"  - exp2_combined_csvs/ (optional CSV files for inspection)")
    print(f"  - results_dict_exp1_50.py")
    print(f"  - results_dict_merged_exp2.py")

