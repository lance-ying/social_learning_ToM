#!/usr/bin/env python3
"""
Convert ASCII maps from exp3_true directory to TypeScript level files.
Reads ASCII maps with X and Y agent goal annotations and generates complete level configs.
For 3-agent levels: M (player), X (agent2), Y (agent3)
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
    Parse ASCII file that contains map and goals for both X and Y agents.
    Format:
        <ascii map>

        X: goal1, goal2
        Y: goal1, goal2

    Returns: (ascii_map, x_goal1, x_goal2, y_goal1, y_goal2)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Split by empty lines to separate map from goals
    parts = content.split('\n\n')

    if len(parts) < 2:
        print(f"Warning: No goals found in {file_path}, using defaults")
        return content, 1, 2, 1, 2

    ascii_map = parts[0].strip()
    goals_section = parts[1].strip()

    # Parse X agent goals (format: "X: 1, 3")
    x_match = re.search(r'X:\s*(\d+)\s*,\s*(\d+)', goals_section)
    if x_match:
        x_goal1 = int(x_match.group(1))
        x_goal2 = int(x_match.group(2))
    else:
        print(f"Warning: Could not parse X goals in {file_path}, using defaults")
        x_goal1, x_goal2 = 1, 2

    # Parse Y agent goals (format: "Y: 2, 1")
    y_match = re.search(r'Y:\s*(\d+)\s*,\s*(\d+)', goals_section)
    if y_match:
        y_goal1 = int(y_match.group(1))
        y_goal2 = int(y_match.group(2))
    else:
        print(f"Warning: Could not parse Y goals in {file_path}, using defaults")
        y_goal1, y_goal2 = 1, 2

    return ascii_map, x_goal1, x_goal2, y_goal1, y_goal2


def map_goal_number_to_treasure_type(goal_num):
    """Map goal number to treasure type letter (1=A, 2=B, 3=C)"""
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    return mapping.get(goal_num, 'A')


def count_treasures_in_map(ascii_map):
    """Count treasures in map from left to right, top to bottom"""
    treasures = []
    for y, line in enumerate(ascii_map.split('\n')):
        for x, char in enumerate(line):
            if char.lower() == 'g':  # treasure
                treasures.append((x, y, char))
    return len(treasures)


def generate_typescript_level(level_id, ascii_map, x_goal1, x_goal2, y_goal1, y_goal2, steps_remaining=50):
    """Generate TypeScript level file content with three agents (M, X, Y)"""

    # Keep X and Y as is in the ASCII map (no conversion needed for 3-agent levels)
    ascii_map_converted = ascii_map

    # Map goal numbers to treasure types
    x_goal1_type = map_goal_number_to_treasure_type(x_goal1)
    x_goal2_type = map_goal_number_to_treasure_type(x_goal2)
    y_goal1_type = map_goal_number_to_treasure_type(y_goal1)
    y_goal2_type = map_goal_number_to_treasure_type(y_goal2)

    # Count treasures to validate
    treasure_count = count_treasures_in_map(ascii_map)
    print(f"  Found {treasure_count} treasures in map")

    # Generate TypeScript content with empty paths for three agents
    # Agent 2 (X) uses x_goal1/x_goal2, Agent 3 (Y) uses y_goal1/y_goal2
    ts_content = f"""import {{ LevelConfig }} from '../types';

export const {level_id}: LevelConfig = {{
  id: '{level_id}',
  name: '{level_id}',
  asciiMap: `
{ascii_map_converted}
`.trim(),
  agentPaths: {{
    2: {{
      movements: {{
        experienced1: {{
          path: [],
          goal: {x_goal1},
          type: 'Expert'
        }},
        experienced2: {{
          path: [],
          goal: {x_goal2},
          type: 'Expert'
        }},
        experienced3: {{
          path: [],
          goal: {x_goal1},
          type: 'Expert_2'
        }},
        experienced4: {{
          path: [],
          goal: {x_goal2},
          type: 'Expert_2'
        }}
      }}
    }},
    3: {{
      movements: {{
        experienced1: {{
          path: [],
          goal: {y_goal1},
          type: 'Expert'
        }},
        experienced2: {{
          path: [],
          goal: {y_goal2},
          type: 'Expert'
        }},
        experienced3: {{
          path: [],
          goal: {y_goal1},
          type: 'Expert_2'
        }},
        experienced4: {{
          path: [],
          goal: {y_goal2},
          type: 'Expert_2'
        }}
      }}
    }}
  }},
  stepsRemaining: {steps_remaining},
  goal: {{
    type: '{x_goal1_type}',
    description: 'Find and obtain Treasure {x_goal1_type}'
  }}
}};
"""

    return ts_content, x_goal1_type, x_goal2_type, y_goal1_type, y_goal2_type


def main():
    """Main conversion function"""

    # Define paths (relative to scripts/converters/ directory)
    script_dir = Path(__file__).parent
    ascii_dir = script_dir.parent.parent / "extracted_ascii_maps" / "exp_3_111925"
    output_dir = script_dir.parent.parent / "src" / "data" / "levels" / "exp3_true"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ASCII text files
    pattern = str(ascii_dir / "*.txt")
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

        # Level ID is just the filename (e.g., sm211, sm221)
        level_id = filename

        print(f"Processing: {filename}")

        try:
            # Parse ASCII file with goals for both X and Y agents
            ascii_map, x_goal1, x_goal2, y_goal1, y_goal2 = parse_ascii_file_with_goals(ascii_file)
            print(f"  X Agent (agent2) Goals: goal1={x_goal1} ({map_goal_number_to_treasure_type(x_goal1)}), goal2={x_goal2} ({map_goal_number_to_treasure_type(x_goal2)})")
            print(f"  Y Agent (agent3) Goals: goal1={y_goal1} ({map_goal_number_to_treasure_type(y_goal1)}), goal2={y_goal2} ({map_goal_number_to_treasure_type(y_goal2)})")

            # Generate TypeScript content
            ts_content, x_goal1_type, x_goal2_type, y_goal1_type, y_goal2_type = generate_typescript_level(
                level_id, ascii_map, x_goal1, x_goal2, y_goal1, y_goal2
            )

            # Write TypeScript file
            output_path = output_dir / f"{level_id}.ts"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ts_content)

            print(f"  ✓ Created: {output_path}")
            processed.append({
                'id': level_id,
                'file': str(output_path),
                'x_goal1': x_goal1,
                'x_goal2': x_goal2,
                'y_goal1': y_goal1,
                'y_goal2': y_goal2,
                'x_goal1_type': x_goal1_type,
                'x_goal2_type': x_goal2_type,
                'y_goal1_type': y_goal1_type,
                'y_goal2_type': y_goal2_type
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
    metadata_file = output_dir / "_conversion_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("Level ID | X_Goal1 | X_Goal2 | X_Goal1_Type | X_Goal2_Type | Y_Goal1 | Y_Goal2 | Y_Goal1_Type | Y_Goal2_Type\n")
        f.write("-" * 100 + "\n")
        for item in processed:
            f.write(f"{item['id']} | {item['x_goal1']} | {item['x_goal2']} | {item['x_goal1_type']} | {item['x_goal2_type']} | ")
            f.write(f"{item['y_goal1']} | {item['y_goal2']} | {item['y_goal1_type']} | {item['y_goal2_type']}\n")

    print(f"\nMetadata saved to: {metadata_file}")
    print("\n✓ Phase 1 complete! TypeScript files created with empty paths for three agents (M, X, Y).")
    print("Next step: Run pathfinder to generate optimal paths.")


if __name__ == "__main__":
    main()


