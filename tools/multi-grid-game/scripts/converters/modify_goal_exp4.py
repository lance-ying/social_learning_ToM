#!/usr/bin/env python3
"""
Script to modify goal types in exp4 level files based on ASCII map treasure analysis.
Analyzes the ASCII map to determine treasure types and updates the goal field accordingly.
"""

import os
import re
import glob

def determine_treasure_type_from_map(ascii_map):
    """
    Determine treasure type based on position of G relative to other treasures.
    
    Logic: Count all treasures (g and G) from left-to-right, top-to-bottom.
    The first treasure encountered is type A, second is type B, third is type C.
    The goal type is whatever type the G (uppercase) treasure is.
    """
    treasures = []
    
    # Find all treasures (both lowercase g and uppercase G)
    for y, line in enumerate(ascii_map.split('\n')):
        for x, char in enumerate(line):
            if char.lower() == 'g':  # Both g and G are treasures
                treasures.append((char, x, y))
    
    # Sort treasures by reading order (top-to-bottom, then left-to-right)
    treasures.sort(key=lambda t: (t[2], t[1]))
    
    # Assign types based on position (1st=A, 2nd=B, 3rd=C)
    treasure_types = ['A', 'B', 'C']
    
    for i, (char, x, y) in enumerate(treasures):
        if i < len(treasure_types):
            treasure_type = treasure_types[i]
            if char == 'G':  # Found the uppercase G treasure
                return treasure_type
    
    # If no G found, default to first treasure type
    return treasure_types[0] if treasures else 'A'

def update_level_goal_type(level_file_path):
    """Update the goal type and description in a level file based on its ASCII map"""
    
    # Read the current level file
    with open(level_file_path, 'r') as f:
        content = f.read()
    
    # Extract ASCII map from the content
    ascii_map_match = re.search(r'asciiMap:\s*`\s*\n(.*?)\n\s*`\.trim\(\)', content, re.MULTILINE | re.DOTALL)
    if not ascii_map_match:
        print(f"  Warning: Could not find ASCII map in {level_file_path}")
        return False
    
    ascii_map = ascii_map_match.group(1)
    
    # Determine the correct treasure type from the map
    correct_type = determine_treasure_type_from_map(ascii_map)
    
    # Extract current goal type
    current_goal_match = re.search(r'goal:\s*\{\s*type:\s*\'([A-Z])\'', content)
    if not current_goal_match:
        print(f"  Warning: Could not find current goal type in {level_file_path}")
        return False
    
    current_type = current_goal_match.group(1)
    
    if current_type == correct_type:
        print(f"  ✓ Goal type already correct: {correct_type}")
        return True
    
    print(f"  Updating goal type: {current_type} → {correct_type}")
    
    # Update the goal type and description
    new_content = re.sub(
        r'(goal:\s*\{\s*type:\s*\')[A-Z](\'\s*,\s*description:\s*\'Find and obtain Treasure\s)[A-Z](\'\s*\})',
        f'\\g<1>{correct_type}\\g<2>{correct_type}\\g<3>',
        content
    )
    
    # Write the updated content back to the file
    with open(level_file_path, 'w') as f:
        f.write(new_content)
    
    return True

def main():
    """Main function to process all exp4 level files"""
    
    # Directory containing exp4 level files
    levels_dir = "../../src/data/levels/exp4"
    
    if not os.path.exists(levels_dir):
        print(f"Error: Directory {levels_dir} not found")
        return
    
    # Find all TypeScript level files
    pattern = os.path.join(levels_dir, "*.ts")
    level_files = sorted(glob.glob(pattern))
    
    if not level_files:
        print(f"No TypeScript files found in {levels_dir}")
        return
    
    print(f"Found {len(level_files)} level files to process...")
    
    updated_count = 0
    skipped_count = 0
    
    for level_file in level_files:
        level_name = os.path.basename(level_file)
        print(f"\nProcessing {level_name}:")
        
        if update_level_goal_type(level_file):
            updated_count += 1
        else:
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  Updated: {updated_count} files")
    print(f"  Skipped: {skipped_count} files")

if __name__ == "__main__":
    main()