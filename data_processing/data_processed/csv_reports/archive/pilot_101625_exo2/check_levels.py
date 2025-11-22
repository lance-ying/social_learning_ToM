#!/usr/bin/env python3
"""
Simple script to check what levels have been hit in the pilot data.
"""

import os
import csv
from collections import defaultdict

def extract_levels_from_csv(filepath):
    """Extract all levels from a single CSV file."""
    levels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all lines that start with "Level: "
    for line in content.split('\n'):
        if line.startswith('Level: '):
            level_name = line.replace('Level: ', '').strip()
            levels.append(level_name)
    
    return levels

def main():
    # Get all CSV files in current directory
    directory = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Track levels across all files
    all_levels = set()
    levels_by_participant = {}
    level_counts = defaultdict(int)
    
    # Process each file
    for csv_file in sorted(csv_files):
        filepath = os.path.join(directory, csv_file)
        levels = extract_levels_from_csv(filepath)
        
        # Store unique levels for this participant
        unique_levels = [l for l in levels if l != 'comprehension_check']
        levels_by_participant[csv_file] = unique_levels
        
        # Add to overall set
        for level in unique_levels:
            all_levels.add(level)
            level_counts[level] += 1
        
        print(f"{csv_file}: {len(unique_levels)} levels")
        print(f"  Levels: {', '.join(unique_levels)}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal unique levels (excluding comprehension_check): {len(all_levels)}")
    print(f"\nAll unique levels:")
    for level in sorted(all_levels):
        print(f"  - {level} (hit {level_counts[level]} times)")
    
    print(f"\nTotal participants: {len(csv_files)}")

if __name__ == "__main__":
    main()

