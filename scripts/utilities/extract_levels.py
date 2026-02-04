#!/usr/bin/env python3
"""
Script to extract all unique smXXX_X levels from all CSV files in exp3 directory.
"""

import os
import re
import glob
from pathlib import Path
from collections import defaultdict

def extract_levels_from_csv(csv_path):
    """
    Extract all smXXX_X level patterns from a CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        set: Set of unique level patterns found
    """
    levels = set()
    
    try:
        with open(csv_path, 'r') as f:
            content = f.read()
            
        # Find all "Level: smXXX_X" patterns
        pattern = r'Level: (sm\d{3}_\d{1,2})'
        matches = re.findall(pattern, content)
        levels.update(matches)
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    
    return levels

def analyze_levels(directory):
    """
    Analyze all CSV files in directory to extract unique levels.
    
    Args:
        directory (str): Path to directory containing CSV files
    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    all_levels = set()
    level_counts = defaultdict(int)
    file_level_mapping = {}
    
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        levels = extract_levels_from_csv(csv_file)
        
        if levels:
            file_level_mapping[filename] = sorted(levels)
            for level in levels:
                all_levels.add(level)
                level_counts[level] += 1
    
    # Print results
    print("=" * 60)
    print("UNIQUE LEVELS FOUND:")
    print("=" * 60)
    print(f"Total unique levels: {len(all_levels)}")
    print()
    
    # Group levels by first digit
    grouped_levels = defaultdict(list)
    for level in sorted(all_levels):
        first_digit = level[2]  # Extract first digit after 'sm'
        grouped_levels[first_digit].append(level)
    
    for digit in sorted(grouped_levels.keys()):
        print(f"sm{digit}XX levels ({len(grouped_levels[digit])}):")
        for level in sorted(grouped_levels[digit]):
            count = level_counts[level]
            print(f"  {level}: appears in {count} files")
        print()
    
    # Show which files contain which levels (optional - can be commented out)
    print("=" * 60)
    print("FILE-LEVEL MAPPING:")
    print("=" * 60)
    for filename, levels in sorted(file_level_mapping.items()):
        print(f"{filename}: {', '.join(levels)}")
    
    return all_levels, level_counts

if __name__ == "__main__":
    directory = "/Users/heyodogo/code/lab/social_learning/social_learning_ToM/data_processing/data_processed/exp3"
    all_levels, level_counts = analyze_levels(directory)