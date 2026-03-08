#!/usr/bin/env python3
"""
Convert ASCII maps from problem_exp4 directory to TypeScript level files.
Reads ASCII maps with X and Y agent goal annotations and generates complete level configs.
"""

import os
import sys
import glob
import re
from pathlib import Path

# Add map_generator to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_ascii_file_with_goals_exp4(file_path):
    """
    Parse ASCII file that contains map and goals for both X and Y agents.
    Format:
        <ascii map>

        X: 3n, 2n, 2n
        Y: 1a, 3a, 1a

    Returns: (ascii_map, x_goals, y_goals, x_expertise, y_expertise)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Split by empty lines to separate map from goals
    parts = content.split('\n\n')

    if len(parts) < 2:
        print(f"Warning: No goals found in {file_path}, using defaults")
        return content, [1, 2, 1], [1, 2, 1], ['n', 'n', 'n'], ['n', 'n', 'n']

    ascii_map = parts[0].strip()
    goals_section = parts[1].strip()

    # Parse X agent goals (format: "X: 3n, 2n, 2n")
    x_match = re.search(r'X:\s*([^\n]+)', goals_section)
    if x_match:
        x_goals_str = x_match.group(1).strip()
        x_goals, x_expertise = parse_goal_list(x_goals_str)
    else:
        print(f"Warning: Could not parse X goals in {file_path}, using defaults")
        x_goals, x_expertise = [1, 2, 1], ['n', 'n', 'n']

    # Parse Y agent goals (format: "Y: 1a, 3a, 1a")
    y_match = re.search(r'Y:\s*([^\n]+)', goals_section)
    if y_match:
        y_goals_str = y_match.group(1).strip()
        y_goals, y_expertise = parse_goal_list(y_goals_str)
    else:
        print(f"Warning: Could not parse Y goals in {file_path}, using defaults")
        y_goals, y_expertise = [1, 2, 1], ['n', 'n', 'n']

    return ascii_map, x_goals, y_goals, x_expertise, y_expertise


def parse_goal_list(goals_str):
    """
    Parse goal list like "3n, 2n, 2n" into goals and expertise arrays.
    Returns: (goals, expertise) where goals=[3,2,2], expertise=['n','n','n']
    """
    goal_parts = [part.strip() for part in goals_str.split(',')]
    goals = []
    expertise = []
    
    for part in goal_parts:
        if len(part) >= 2:
            goal_num = int(part[:-1])  # Everything except last character
            exp_type = part[-1]       # Last character
            goals.append(goal_num)
            expertise.append(exp_type)
        else:
            # Fallback for malformed entries
            goals.append(int(part) if part.isdigit() else 1)
            expertise.append('n')
    
    return goals, expertise


def map_goal_number_to_treasure_type(goal_num):
    """Map goal number to treasure type letter (1=A, 2=B, 3=C)"""
    mapping = {1: 'A', 2: 'B', 3: 'C'}
    return mapping.get(goal_num, 'A')


def map_expertise_to_movement_type(expertise, index):
    """
    Map expertise to movement type.
    All normal expertise -> Novice
    All expert expertise -> Expert
    """
    if expertise == 'a':  # expert
        return 'Expert'
    else:  # normal
        return 'Novice'


def count_treasures_in_map(ascii_map):
    """Count treasures in map from left to right, top to bottom"""
    treasures = []
    for y, line in enumerate(ascii_map.split('\n')):
        for x, char in enumerate(line):
            if char.lower() == 'g':  # treasure
                treasures.append((x, y, char))
    return len(treasures)


def generate_typescript_level_exp4(level_id, ascii_map, x_goals, y_goals, x_expertise, y_expertise, steps_remaining=50):
    """Generate TypeScript level file content with two agents (X→Z and Y→O)"""

    # Convert X to Z and Y to O in the ASCII map for consistency with existing code
    ascii_map_converted = ascii_map.replace('X', 'Z').replace('Y', 'O')

    # Count treasures to validate
    treasure_count = count_treasures_in_map(ascii_map)
    print(f"  Found {treasure_count} treasures in map")

    # Ensure we have exactly 3 goals per agent by padding if necessary
    while len(x_goals) < 3:
        x_goals.append(x_goals[0] if x_goals else 1)
        x_expertise.append(x_expertise[0] if x_expertise else 'n')
    
    while len(y_goals) < 3:
        y_goals.append(y_goals[0] if y_goals else 1)
        y_expertise.append(y_expertise[0] if y_expertise else 'n')

    # Generate movement configurations for Agent 1 (X)
    agent1_movements = {}
    for i, (goal, expertise) in enumerate(zip(x_goals, x_expertise)):
        movement_type = map_expertise_to_movement_type(expertise, i)
        agent1_movements[f'experienced{i+1}'] = {
            'path': [],
            'goal': goal,
            'type': movement_type
        }

    # Generate movement configurations for Agent 2 (Y)
    agent2_movements = {}
    for i, (goal, expertise) in enumerate(zip(y_goals, y_expertise)):
        movement_type = map_expertise_to_movement_type(expertise, i)
        agent2_movements[f'experienced{i+1}'] = {
            'path': [],
            'goal': goal,
            'type': movement_type
        }

    # Use first goal of first agent for overall goal description
    primary_goal = x_goals[0]
    primary_goal_type = map_goal_number_to_treasure_type(primary_goal)

    # Generate TypeScript content
    ts_content = f"""import {{ LevelConfig }} from '../types';

export const {level_id}: LevelConfig = {{
  id: '{level_id}',
  name: '{level_id}',
  asciiMap: `
{ascii_map_converted}
`.trim(),
  agentPaths: {{
    1: {{
      movements: {{
        experienced1: {{
          path: [],
          goal: {agent1_movements['experienced1']['goal']},
          type: '{agent1_movements['experienced1']['type']}'
        }},
        experienced2: {{
          path: [],
          goal: {agent1_movements['experienced2']['goal']},
          type: '{agent1_movements['experienced2']['type']}'
        }},
        experienced3: {{
          path: [],
          goal: {agent1_movements['experienced3']['goal']},
          type: '{agent1_movements['experienced3']['type']}'
        }}
      }}
    }},
    2: {{
      movements: {{
        experienced1: {{
          path: [],
          goal: {agent2_movements['experienced1']['goal']},
          type: '{agent2_movements['experienced1']['type']}'
        }},
        experienced2: {{
          path: [],
          goal: {agent2_movements['experienced2']['goal']},
          type: '{agent2_movements['experienced2']['type']}'
        }},
        experienced3: {{
          path: [],
          goal: {agent2_movements['experienced3']['goal']},
          type: '{agent2_movements['experienced3']['type']}'
        }}
      }}
    }}
  }},
  stepsRemaining: {steps_remaining},
  goal: {{
    type: '{primary_goal_type}',
    description: 'Find and obtain Treasure {primary_goal_type}'
  }}
}};
"""

    return ts_content, primary_goal_type


def main():
    """Main conversion function"""

    # Define paths
    ascii_dir = "../../extracted_ascii_maps/problem_exp4"
    output_dir = "../../src/data/levels/exp4"

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

        # Level ID is just the filename (e.g., sm331, sm543)
        level_id = filename

        print(f"Processing: {filename}")

        try:
            # Parse ASCII file with goals for both X and Y agents
            ascii_map, x_goals, y_goals, x_expertise, y_expertise = parse_ascii_file_with_goals_exp4(ascii_file)
            print(f"  X Agent Goals: {x_goals} with expertise: {x_expertise}")
            print(f"  Y Agent Goals: {y_goals} with expertise: {y_expertise}")

            # Generate TypeScript content
            ts_content, primary_goal_type = generate_typescript_level_exp4(
                level_id, ascii_map, x_goals, y_goals, x_expertise, y_expertise
            )

            # Write TypeScript file
            output_path = os.path.join(output_dir, f"{level_id}.ts")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ts_content)

            print(f"  ✓ Created: {output_path}")
            processed.append({
                'id': level_id,
                'file': output_path,
                'x_goals': x_goals,
                'y_goals': y_goals,
                'x_expertise': x_expertise,
                'y_expertise': y_expertise,
                'primary_goal_type': primary_goal_type
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
        f.write("Level ID | X_Goals | X_Expertise | Y_Goals | Y_Expertise | Primary_Goal_Type\n")
        f.write("-" * 100 + "\n")
        for item in processed:
            f.write(f"{item['id']} | {item['x_goals']} | {item['x_expertise']} | ")
            f.write(f"{item['y_goals']} | {item['y_expertise']} | {item['primary_goal_type']}\n")

    print(f"\nMetadata saved to: {metadata_file}")
    print("\n✓ Phase 1 complete! TypeScript files created with empty paths for both agents.")
    print("Next step: Run pathfinder to generate optimal paths.")


if __name__ == "__main__":
    main()