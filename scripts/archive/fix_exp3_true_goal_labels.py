#!/usr/bin/env python3
"""
Fix goal labels for exp3_true levels based on capital G position in ASCII maps.
Adapted from fix_exp2_goal_labels.py for 3-agent levels.
"""

import os
import re
import glob

def parse_ascii_file(file_path):
    """Parse an ASCII file to find chest positions and which one is the goal (G)"""
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    # Split by empty lines to separate map from agent goal annotations
    parts = content.split('\n\n')
    map_content = parts[0].strip()
    map_lines = map_content.split('\n')
    
    chests = []  # List of (y, x, is_goal) tuples
    
    for y, line in enumerate(map_lines):
        for x, char in enumerate(line):
            if char in ['g', 'G']:
                is_goal = (char == 'G')
                chests.append((y, x, is_goal))
    
    # Sort chests: first by y (top to bottom), then by x (left to right)
    chests.sort(key=lambda c: (c[0], c[1]))
    
    # Assign labels A, B, C based on sorted order
    labeled_chests = []
    labels = ['A', 'B', 'C', 'D', 'E']  # Support up to 5 chests
    
    for i, (y, x, is_goal) in enumerate(chests):
        label = labels[i] if i < len(labels) else '?'
        labeled_chests.append({
            'position': (y, x),
            'label': label,
            'is_goal': is_goal
        })
    
    # Find which label is the goal (capital G)
    goal_label = None
    for chest in labeled_chests:
        if chest['is_goal']:
            goal_label = chest['label']
            break
    
    return {
        'map_lines': map_lines,
        'chests': labeled_chests,
        'goal_label': goal_label
    }

def get_current_goal_type(ts_file_path):
    """Extract the current goal type from a .ts file"""
    with open(ts_file_path, 'r') as f:
        content = f.read()
    
    match = re.search(r"type:\s*['\"]([A-Z])['\"]", content)
    if match:
        return match.group(1)
    return None

def update_ts_goal_label(ts_file_path, correct_goal_label):
    """Update only the goal label in the .ts file (not the ASCII map)"""
    with open(ts_file_path, 'r') as f:
        content = f.read()
    
    # Update goal type
    content = re.sub(
        r"(goal:\s*\{\s*type:\s*['\"])([A-Z])(['\"])",
        rf"\g<1>{correct_goal_label}\g<3>",
        content
    )
    
    # Update goal description
    content = re.sub(
        r"(description:\s*['\"]Find and obtain Treasure\s+)([A-Z])(['\"])",
        rf"\g<1>{correct_goal_label}\g<3>",
        content
    )
    
    with open(ts_file_path, 'w') as f:
        f.write(content)

def analyze_all_levels(ascii_dir, ts_dir):
    """Analyze all levels and report differences"""
    
    ascii_files = sorted(glob.glob(os.path.join(ascii_dir, "sm*.txt")))
    
    print(f"{'Level':<10} {'ASCII Goal':<12} {'TS Goal':<13} {'Match':<7} {'Action'}")
    print("-" * 70)
    
    results = []
    
    for ascii_path in ascii_files:
        level_name = os.path.basename(ascii_path).replace('.txt', '')
        ts_path = os.path.join(ts_dir, f"{level_name}.ts")
        
        if not os.path.exists(ts_path):
            print(f"{level_name:<10} {'N/A':<12} {'N/A':<13} {'N/A':<7} .ts file not found")
            continue
        
        # Parse ASCII file
        ascii_data = parse_ascii_file(ascii_path)
        ascii_goal = ascii_data['goal_label']
        
        # Get current goal from TS
        current_goal = get_current_goal_type(ts_path)
        
        # Check if they match
        matches = (ascii_goal == current_goal)
        action = "OK" if matches else "NEEDS UPDATE"
        
        print(f"{level_name:<10} {ascii_goal or 'None':<12} {current_goal or 'None':<13} {'✓' if matches else '✗':<7} {action}")
        
        results.append({
            'level_name': level_name,
            'ascii_path': ascii_path,
            'ts_path': ts_path,
            'ascii_goal': ascii_goal,
            'current_goal': current_goal,
            'needs_update': not matches
        })
    
    return results

def update_all_levels(results, backup_levels=True):
    """Update all levels that need fixing"""
    
    updated_count = 0
    
    for result in results:
        if not result['needs_update']:
            continue
        
        try:
            # Backup if requested
            if backup_levels:
                backup_path = result['ts_path'] + '.goal_backup'
                if not os.path.exists(backup_path):
                    with open(result['ts_path'], 'r') as f:
                        with open(backup_path, 'w') as b:
                            b.write(f.read())
            
            # Update the file
            update_ts_goal_label(
                result['ts_path'],
                result['ascii_goal']
            )
            
            print(f"✓ Updated {result['level_name']}: {result['current_goal']} → {result['ascii_goal']}")
            updated_count += 1
            
        except Exception as e:
            print(f"✗ Failed to update {result['level_name']}: {e}")
    
    return updated_count

def main():
    import sys
    
    # Default to exp3_true directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ascii_dir = os.path.join(script_dir, "../../extracted_ascii_maps/exp3_true")
    ts_dir = os.path.join(script_dir, "../../src/data/levels/exp3_true")
    
    print("Analyzing goal labels (uppercase G) in exp3_true ASCII maps vs current .ts files...\n")
    
    results = analyze_all_levels(ascii_dir, ts_dir)
    
    needs_update = [r for r in results if r['needs_update']]
    
    if not needs_update:
        print("\n✓ All levels have correct goal labels!")
        return
    
    print(f"\n{len(needs_update)} levels need updates.")
    
    # Always update by default (no prompt)
    update_files = '--analyze-only' not in sys.argv
    
    if update_files:
        print("\nUpdating files...")
        updated = update_all_levels(results)
        print(f"\n✓ Successfully updated {updated} files")
        print("\nNext step: Run update_m_player_points.py to recalculate stepsRemaining for the corrected goals!")
    else:
        print("\nAnalysis only - no files updated.")

if __name__ == "__main__":
    main()


