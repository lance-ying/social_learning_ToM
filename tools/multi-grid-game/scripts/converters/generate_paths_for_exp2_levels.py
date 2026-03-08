#!/usr/bin/env python3
"""
Generate optimal paths for all levels in the 'exp2' directory.
Generates paths for both expert1 (goal 1) and expert2 (goal 2).
"""

import os
import sys
import re
from typing import List, Dict, Tuple
from pathlib import Path

# Import from pathfinder_for_z_agent
from pathfinder_for_z_agent import ZAgentPathfinder, parse_typescript_level


def parse_typescript_level_with_goals(file_path: str) -> Tuple[str, int, int]:
    """
    Parse TypeScript level file and extract ascii_map, goal1, and goal2.
    Returns: (ascii_map, goal1, goal2)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract ASCII map
        map_match = re.search(r'asciiMap:\s*`\s*\n(.*?)\n\s*`', content, re.DOTALL)
        ascii_map = map_match.group(1) if map_match else ""

        # Extract goal numbers for experienced1 and experienced2
        exp1_goal_match = re.search(r'experienced1:\s*\{\s*path:.*?goal:\s*(\d+)', content, re.DOTALL)
        exp2_goal_match = re.search(r'experienced2:\s*\{\s*path:.*?goal:\s*(\d+)', content, re.DOTALL)

        goal1 = int(exp1_goal_match.group(1)) if exp1_goal_match else 1
        goal2 = int(exp2_goal_match.group(1)) if exp2_goal_match else 2

        return ascii_map, goal1, goal2

    except Exception as e:
        print(f"  ✗ Error parsing file {file_path}: {e}")
        return "", 1, 2


def map_goal_number_to_type(goal_num: int) -> str:
    """Map goal number to treasure type (1=A, 2=B, 3=C)"""
    mapping = {1: 'A', 2: 'B', 3: 'C'}
    return mapping.get(goal_num, 'A')


def update_typescript_paths(file_path: str, exp1_path: List[str], exp2_path: List[str]) -> bool:
    """
    Update both experienced1 and experienced2 paths in the TypeScript file.
    Also updates experienced3 (same as exp1) and experienced4 (same as exp2).
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Format paths
        exp1_path_str = ', '.join(f'"{step}"' for step in exp1_path)
        exp2_path_str = ', '.join(f'"{step}"' for step in exp2_path)

        # Replace experienced1 path
        pattern1 = r'(experienced1:\s*\{\s*path:\s*\[)[^\]]*(\])'
        content = re.sub(pattern1, r'\1' + exp1_path_str + r'\2', content)

        # Replace experienced2 path
        pattern2 = r'(experienced2:\s*\{\s*path:\s*\[)[^\]]*(\])'
        content = re.sub(pattern2, r'\1' + exp2_path_str + r'\2', content)

        # Replace experienced3 path (same as exp1)
        pattern3 = r'(experienced3:\s*\{\s*path:\s*\[)[^\]]*(\])'
        content = re.sub(pattern3, r'\1' + exp1_path_str + r'\2', content)

        # Replace experienced4 path (same as exp2)
        pattern4 = r'(experienced4:\s*\{\s*path:\s*\[)[^\]]*(\])'
        content = re.sub(pattern4, r'\1' + exp2_path_str + r'\2', content)

        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"  ✗ Error updating file {file_path}: {e}")
        return False


def process_new_levels():
    """Process all levels in the 'exp2' directory and generate paths"""

    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels_path = os.path.join(base_path, 'src', 'data', 'levels', 'exp2')

    # Get all .ts files in the exp2 directory
    if not os.path.exists(levels_path):
        print(f"Error: Directory not found: {levels_path}")
        return

    all_files = [f for f in os.listdir(levels_path)
                 if f.endswith('.ts') and not f.startswith('_')]
    level_names = sorted([f[:-3] for f in all_files])

    print("=" * 70)
    print(f"Processing {len(level_names)} levels in 'exp2' directory")
    print("Generating optimal paths for experienced1 (goal1) and experienced2 (goal2)")
    print("=" * 70)

    results = []
    failed = []

    for level_name in level_names:
        file_path = os.path.join(levels_path, f'{level_name}.ts')

        print(f"\n{level_name}:")
        print("-" * 50)

        # Parse level and extract goals
        ascii_map, goal1, goal2 = parse_typescript_level_with_goals(file_path)

        if not ascii_map:
            print(f"  ✗ Could not parse ASCII map")
            failed.append(level_name)
            continue

        goal1_type = map_goal_number_to_type(goal1)
        goal2_type = map_goal_number_to_type(goal2)

        print(f"  Goal 1 (expert1/expert3): {goal1} → Treasure {goal1_type}")
        print(f"  Goal 2 (expert2/expert4): {goal2} → Treasure {goal2_type}")

        # Generate path for experienced1 (goal1)
        pathfinder1 = ZAgentPathfinder(ascii_map, goal1_type)
        exp1_path = pathfinder1.find_efficient_path()

        if not exp1_path:
            print(f"  ✗ No path found for experienced1 (goal {goal1})")
            failed.append(level_name)
            continue

        print(f"  ✓ experienced1 path: {len(exp1_path)} steps")

        # Generate path for experienced2 (goal2)
        pathfinder2 = ZAgentPathfinder(ascii_map, goal2_type)
        exp2_path = pathfinder2.find_efficient_path()

        if not exp2_path:
            print(f"  ✗ No path found for experienced2 (goal {goal2})")
            failed.append(level_name)
            continue

        print(f"  ✓ experienced2 path: {len(exp2_path)} steps")

        # Update the TypeScript file with both paths
        success = update_typescript_paths(file_path, exp1_path, exp2_path)

        if success:
            print(f"  ✓ Updated {level_name}.ts with both paths")
            results.append({
                'name': level_name,
                'exp1_length': len(exp1_path),
                'exp2_length': len(exp2_path),
                'goal1': goal1,
                'goal2': goal2
            })
        else:
            failed.append(level_name)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print(f"\n✓ Successfully processed {len(results)} levels:\n")
        print(f"{'Level':<20} {'Goal1':<8} {'Exp1 Steps':<12} {'Goal2':<8} {'Exp2 Steps'}")
        print("-" * 70)
        for result in results:
            print(f"{result['name']:<20} {result['goal1']:<8} {result['exp1_length']:<12} "
                  f"{result['goal2']:<8} {result['exp2_length']}")

    if failed:
        print(f"\n✗ Failed to process {len(failed)} levels:")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "=" * 70)
    print("✓ Path generation complete!")
    print("All levels have been updated with optimal paths for both expert types.")
    print("=" * 70)


def main():
    process_new_levels()


if __name__ == "__main__":
    main()
