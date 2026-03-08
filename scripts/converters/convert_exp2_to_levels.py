#!/usr/bin/env python3
"""
Convert ASCII maps from exp2 directory to TypeScript level files.
Reads ASCII maps with goal annotations and generates complete level configs.
"""

import os
import sys
import glob
import re
from pathlib import Path

# Add map_generator to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from level_generator import LevelGenerator


def parse_ascii_file_with_goals(file_path):
    """
    Parse ASCII file that contains map and goals.
    Format:
        <ascii map>

        goal1, goal2

    Returns: (ascii_map, goal1, goal2)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Split by empty lines to separate map from goals
    parts = content.split('\n\n')

    if len(parts) < 2:
        print(f"Warning: No goals found in {file_path}, using defaults")
        return content, 1, 2

    ascii_map = parts[0].strip()
    goals_line = parts[1].strip()

    # Parse goals (format: "2, 1" or "1, 3")
    goal_match = re.search(r'(\d+)\s*,\s*(\d+)', goals_line)
    if goal_match:
        goal1 = int(goal_match.group(1))
        goal2 = int(goal_match.group(2))
    else:
        print(f"Warning: Could not parse goals in {file_path}, using defaults")
        goal1, goal2 = 1, 2

    return ascii_map, goal1, goal2


def map_goal_number_to_treasure_type(goal_num):
    """Map goal number to treasure type letter (1=A, 2=B, 3=C)"""
    mapping = {1: 'A', 2: 'B', 3: 'C'}
    return mapping.get(goal_num, 'A')


def count_treasures_in_map(ascii_map):
    """Count treasures in map from left to right, top to bottom"""
    treasures = []
    for y, line in enumerate(ascii_map.split('\n')):
        for x, char in enumerate(line):
            if char.lower() == 'g':  # treasure
                treasures.append((x, y, char))
    return len(treasures)


def generate_typescript_level(level_id, ascii_map, goal1, goal2, steps_remaining=50):
    """Generate TypeScript level file content with proper goal types"""

    # Map goal numbers to treasure types
    goal1_type = map_goal_number_to_treasure_type(goal1)
    goal2_type = map_goal_number_to_treasure_type(goal2)

    # Count treasures to validate
    treasure_count = count_treasures_in_map(ascii_map)
    print(f"  Found {treasure_count} treasures in map")

    # Generate TypeScript content with empty paths (will be filled by pathfinder)
    ts_content = f"""import {{ LevelConfig }} from '../types';

export const {level_id}: LevelConfig = {{
  id: '{level_id}',
  name: '{level_id}',
  asciiMap:`
{ascii_map}
`.trim(),
  agentPaths: {{
    1: {{
      movements: {{
        experienced1: {{
          path: [],
          goal: {goal1},
          type: 'Expert'
        }},
        experienced2: {{
          path: [],
          goal: {goal2},
          type: 'Expert'
        }},
        experienced3: {{
          path: [],
          goal: {goal1},
          type: 'Expert_2'
        }},
        experienced4: {{
          path: [],
          goal: {goal2},
          type: 'Novice_2'
        }},
      }}
    }},
  }},
  stepsRemaining: {steps_remaining},
  goal: {{
    type: '{goal1_type}',
    description: 'Find and obtain Treasure {goal1_type}'
  }}
}};
"""

    return ts_content, goal1_type, goal2_type


def main():
    """Main conversion function"""

    # Define paths
    ascii_dir = "../extracted_ascii_maps/exp2"
    output_dir = "../src/data/levels/exp2"

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all ASCII text files
    pattern = os.path.join(ascii_dir, "*.txt")
    ascii_files = sorted(glob.glob(pattern))

    if not ascii_files:
        print(f"No ASCII files found in {ascii_dir}")
        return

    print(f"Found {len(ascii_files)} ASCII files to process...\n")

    processed = []
    failed = []

    for ascii_file in ascii_files:
        filename = Path(ascii_file).stem
        
        # Skip files that start with "old_" or "mod_"
        if filename.startswith(('old_', 'mod_')):
            print(f"Skipping: {filename} (starts with old_ or mod_)")
            continue
        
        # Level ID is just the filename (e.g., s211, s221)
        level_id = filename

        print(f"Processing: {filename}")

        try:
            # Parse ASCII file with goals
            ascii_map, goal1, goal2 = parse_ascii_file_with_goals(ascii_file)
            print(f"  Goals: expert1={goal1} ({map_goal_number_to_treasure_type(goal1)}), expert2={goal2} ({map_goal_number_to_treasure_type(goal2)})")

            # Generate TypeScript content
            ts_content, goal1_type, goal2_type = generate_typescript_level(
                level_id, ascii_map, goal1, goal2
            )

            # Write TypeScript file
            output_path = os.path.join(output_dir, f"{level_id}.ts")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ts_content)

            print(f"  ✓ Created: {output_path}")
            processed.append({
                'id': level_id,
                'file': output_path,
                'goal1': goal1,
                'goal2': goal2,
                'goal1_type': goal1_type,
                'goal2_type': goal2_type
            })

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(filename)

        print()

    # Summary
    print("=" * 60)
    print(f"Successfully processed: {len(processed)}")
    print(f"Failed: {len(failed)}")
    print(f"Output directory: {output_dir}")

    if failed:
        print(f"\nFailed files: {', '.join(failed)}")

    # Save metadata for next step
    metadata_file = os.path.join(output_dir, "_conversion_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write("Level ID | Goal1 | Goal2 | Goal1_Type | Goal2_Type\n")
        f.write("-" * 60 + "\n")
        for item in processed:
            f.write(f"{item['id']} | {item['goal1']} | {item['goal2']} | {item['goal1_type']} | {item['goal2_type']}\n")

    print(f"\nMetadata saved to: {metadata_file}")
    print("\n✓ Phase 1 complete! TypeScript files created with empty paths.")
    print("Next step: Run pathfinder to generate optimal paths.")


if __name__ == "__main__":
    main()
