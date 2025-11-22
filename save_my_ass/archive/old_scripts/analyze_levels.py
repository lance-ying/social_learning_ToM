#!/usr/bin/env python3
"""
Analyze CSV files to count how many correspond to different level types.
Counts CSVs with mod_XXX levels vs s_XXX levels.
"""
import csv
from pathlib import Path
from typing import Set, Dict, List
from collections import defaultdict


def extract_levels_from_csv(csv_path: Path) -> Set[str]:
    """Extract all level names from a CSV file."""
    levels = set()
    try:
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 0 and row[0].startswith("Level: "):
                    level_name = row[0].replace("Level: ", "").strip()
                    if level_name:
                        levels.add(level_name)
    except Exception as e:
        print(f"Error reading {csv_path.name}: {e}")
    return levels


def classify_csv(levels: Set[str]) -> Dict[str, bool]:
    """
    Classify a CSV based on its levels.
    Returns dict with keys: has_mod, has_s, has_sm, has_other
    """
    has_mod = any(level.startswith("mod_") for level in levels)
    has_s = any(level.startswith("s") and not level.startswith("mod_") and not level.startswith("sm") for level in levels)
    has_sm = any(level.startswith("sm") for level in levels)
    has_other = any(level not in ["comprehension_check", "experiment"] and 
                   not level.startswith("mod_") and 
                   not level.startswith("s") and 
                   not level.startswith("sm") 
                   for level in levels)
    
    return {
        "has_mod": has_mod,
        "has_s": has_s,
        "has_sm": has_sm,
        "has_other": has_other,
        "levels": levels
    }


def analyze_directory(csv_dir: Path) -> Dict:
    """Analyze all CSV files in a directory."""
    csv_files = [f for f in csv_dir.iterdir() if f.is_file() and f.suffix == ".csv"]
    
    results = {
        "total_csvs": len(csv_files),
        "mod_csvs": 0,
        "s_csvs": 0,
        "sm_csvs": 0,
        "other_csvs": 0,
        "mod_only": 0,
        "s_only": 0,
        "both_mod_and_s": 0,
        "level_counts": defaultdict(int),
        "all_levels": set()
    }
    
    for csv_file in csv_files:
        levels = extract_levels_from_csv(csv_file)
        classification = classify_csv(levels)
        
        results["all_levels"].update(levels)
        for level in levels:
            results["level_counts"][level] += 1
        
        if classification["has_mod"]:
            results["mod_csvs"] += 1
        if classification["has_s"]:
            results["s_csvs"] += 1
        if classification["has_sm"]:
            results["sm_csvs"] += 1
        if classification["has_other"]:
            results["other_csvs"] += 1
        
        # Check exclusivity
        if classification["has_mod"] and not classification["has_s"]:
            results["mod_only"] += 1
        elif classification["has_s"] and not classification["has_mod"]:
            results["s_only"] += 1
        elif classification["has_mod"] and classification["has_s"]:
            results["both_mod_and_s"] += 1
    
    return results


def analyze_all_directories(base_dir: Path, exclude_dirs: List[str] = None) -> None:
    """Analyze CSV files in all subdirectories."""
    if exclude_dirs is None:
        exclude_dirs = ["archive", "filtered_out"]
    
    base_dir = Path(base_dir)
    subdirs = [d for d in base_dir.iterdir() 
               if d.is_dir() and d.name not in exclude_dirs]
    
    print(f"Analyzing {len(subdirs)} directories...\n")
    print("=" * 80)
    
    total_mod = 0
    total_s = 0
    total_csvs = 0
    total_mod_only = 0
    total_s_only = 0
    total_both = 0
    
    for subdir in sorted(subdirs):
        print(f"\nDirectory: {subdir.name}/")
        print("-" * 80)
        
        results = analyze_directory(subdir)
        total_csvs += results["total_csvs"]
        total_mod += results["mod_csvs"]
        total_s += results["s_csvs"]
        total_mod_only += results["mod_only"]
        total_s_only += results["s_only"]
        total_both += results["both_mod_and_s"]
        
        print(f"Total CSV files: {results['total_csvs']}")
        print(f"  - CSVs with mod_XXX levels: {results['mod_csvs']}")
        print(f"  - CSVs with s_XXX levels: {results['s_csvs']}")
        print(f"  - CSVs with sm_XXX levels: {results['sm_csvs']}")
        print(f"  - CSVs with other levels: {results['other_csvs']}")
        print(f"\nExclusivity:")
        print(f"  - mod_XXX only: {results['mod_only']}")
        print(f"  - s_XXX only: {results['s_only']}")
        print(f"  - Both mod_XXX and s_XXX: {results['both_mod_and_s']}")
        
        # Show unique level types found
        mod_levels = sorted([l for l in results["all_levels"] if l.startswith("mod_")])
        s_levels = sorted([l for l in results["all_levels"] 
                          if l.startswith("s") and not l.startswith("mod_") and not l.startswith("sm")])
        
        if mod_levels:
            print(f"\n  mod_XXX levels found: {', '.join(mod_levels[:10])}")
            if len(mod_levels) > 10:
                print(f"    ... and {len(mod_levels) - 10} more")
        
        if s_levels:
            print(f"  s_XXX levels found: {', '.join(s_levels[:10])}")
            if len(s_levels) > 10:
                print(f"    ... and {len(s_levels) - 10} more")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total CSV files across all directories: {total_csvs}")
    print(f"\nBreakdown by level type:")
    print(f"  - CSVs with mod_XXX levels (any): {total_mod}")
    print(f"  - CSVs with s_XXX levels (any): {total_s}")
    print(f"\nExclusive breakdown:")
    print(f"  - CSVs with ONLY mod_XXX (no s_XXX): {total_mod_only}")
    print(f"  - CSVs with ONLY s_XXX (no mod_XXX): {total_s_only}")
    print(f"  - CSVs with BOTH mod_XXX and s_XXX: {total_both}")
    print(f"\nExpected: ~50 for mod_XXX, ~100 for s_XXX")
    print(f"Actual: {total_mod} CSVs have mod_XXX levels, {total_s} CSVs have s_XXX levels")
    if total_mod_only == 0:
        print(f"\nNote: All CSVs with mod_XXX also contain s_XXX levels.")
        print(f"      So: {total_both} CSVs have both, {total_s_only} CSVs have only s_XXX")


if __name__ == "__main__":
    import sys
    
    base_dir = Path("restrcuture_downloaded_data/csv_reports")
    
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    
    analyze_all_directories(base_dir)

