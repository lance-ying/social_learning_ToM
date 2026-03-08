#!/usr/bin/env python3
"""
Update TypeScript level files in src/data/levels/exp4/ based on ASCII maps from problem_exp4/.
Combines map parsing, path conversion, goal determination, and stepsRemaining calculation.
"""

import os
import sys
import glob
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import heapq

# Point costs for M player calculations
MOVE_COST = 3
WIZARD_INTERACTION_COST = 5
BUFFER_POINTS = 15

@dataclass(frozen=True)
class GameState:
    position: Tuple[int, int]
    inventory: frozenset


def parse_ascii_file_with_goals(file_path):
    """
    Parse ASCII file with goal annotations.
    Format:
        <ascii map>

        X: 3n, 3n
        Y: 1a, 2a

    Returns: (ascii_map, x_goals, y_goals, x_expertise, y_expertise)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Split by empty lines to separate map from goals
    parts = content.split('\n\n')

    if len(parts) < 2:
        print(f"    Warning: No goals found, using defaults")
        return content, [1, 1, 1], [1, 1, 1], ['n', 'n', 'n'], ['n', 'n', 'n']

    ascii_map = parts[0].strip()
    goals_section = parts[1].strip()

    # Parse X agent goals
    x_match = re.search(r'X:\s*([^\n]+)', goals_section)
    if x_match:
        x_goals_str = x_match.group(1).strip()
        x_goals, x_expertise = parse_goal_list(x_goals_str)
    else:
        x_goals, x_expertise = [1, 1, 1], ['n', 'n', 'n']

    # Parse Y agent goals
    y_match = re.search(r'Y:\s*([^\n]+)', goals_section)
    if y_match:
        y_goals_str = y_match.group(1).strip()
        y_goals, y_expertise = parse_goal_list(y_goals_str)
    else:
        y_goals, y_expertise = [1, 1, 1], ['n', 'n', 'n']

    return ascii_map, x_goals, y_goals, x_expertise, y_expertise


def parse_goal_list(goals_str):
    """
    Parse goal list like "3n, 3n" into goals and expertise arrays.
    Returns: (goals, expertise) where goals=[3,3], expertise=['n','n']
    """
    goal_parts = [part.strip() for part in goals_str.split(',')]
    goals = []
    expertise = []

    for part in goal_parts:
        if len(part) >= 2:
            goal_num = int(part[:-1])
            exp_type = part[-1]
            goals.append(goal_num)
            expertise.append(exp_type)
        else:
            goals.append(1)
            expertise.append('n')

    # Pad to 3 elements if needed (3rd is duplicate of 2nd)
    while len(goals) < 3:
        if goals:
            goals.append(goals[-1])
            expertise.append(expertise[-1])
        else:
            goals.append(1)
            expertise.append('n')

    return goals[:3], expertise[:3]


def map_expertise_to_movement_type(expertise_char):
    """Map expertise character to movement type"""
    return 'Expert' if expertise_char == 'a' else 'Novice'


def determine_treasure_type_from_map(ascii_map):
    """
    Determine treasure type based on position of G (uppercase) relative to other treasures.
    Count all treasures from top-to-bottom, left-to-right.
    First=A, Second=B, Third=C.
    Return the type of the uppercase G treasure.
    """
    treasures = []

    for y, line in enumerate(ascii_map.split('\n')):
        for x, char in enumerate(line):
            if char.lower() == 'g':
                treasures.append((char, x, y))

    # Sort by reading order (top-to-bottom, left-to-right)
    treasures.sort(key=lambda t: (t[2], t[1]))

    treasure_types = ['A', 'B', 'C']

    for i, (char, x, y) in enumerate(treasures):
        if i < len(treasure_types):
            treasure_type = treasure_types[i]
            if char == 'G':  # Found uppercase G
                return treasure_type

    # Default to first treasure type
    return treasure_types[0] if treasures else 'A'


class MPlayerMapParser:
    """Parse ASCII map for M player pathfinding"""

    def __init__(self, ascii_map: str):
        self.ascii_map = ascii_map
        self.walls = set()
        self.wizards = {}
        self.barriers = {}
        self.treasures = {}
        self.m_start = None

        self._parse_map()

        rows = self.ascii_map.strip().split('\n')
        self.height = len(rows)
        self.width = max(len(row) for row in rows) if rows else 0

    def _parse_map(self):
        rows = self.ascii_map.strip().split('\n')

        treasure_positions = []
        for y, row in enumerate(rows):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'M':
                    self.m_start = pos
                elif char in ['b', 'r', 'e']:
                    self.wizards[pos] = {
                        'b': 'blueAmulet',
                        'r': 'redAmulet',
                        'e': 'nothing'
                    }.get(char, 'nothing')
                elif char in ['B', 'R']:
                    self.barriers[pos] = {
                        'B': 'blueAmulet',
                        'R': 'redAmulet'
                    }.get(char)
                elif char in ['g', 'G']:
                    treasure_positions.append((y, x, pos))

        # Assign treasure types in order
        treasure_positions.sort()
        for idx, (y, x, pos) in enumerate(treasure_positions):
            treasure_type = (idx % 3) + 1
            self.treasures[pos] = treasure_type


class MPlayerPathfinder:
    """Find optimal paths for M player"""

    def __init__(self, ascii_map: str):
        self.map_parser = MPlayerMapParser(ascii_map)

    def find_optimal_path_to_goal(self, goal_number: int) -> List[str]:
        """Find optimal path from M to specific treasure goal"""
        if not self.map_parser.m_start:
            return []

        target_treasure = None
        for pos, treasure_type in self.map_parser.treasures.items():
            if treasure_type == goal_number:
                target_treasure = pos
                break

        if not target_treasure:
            return []

        start_state = GameState(
            position=self.map_parser.m_start,
            inventory=frozenset()
        )

        return self._astar_search(start_state, target_treasure)

    def _astar_search(self, start_state: GameState, target: Tuple[int, int]) -> List[str]:
        """A* pathfinding"""
        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, start_state, []))
        visited = set()

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state.position == target:
                return path

            state_key = (state.position, state.inventory)
            if state_key in visited:
                continue
            visited.add(state_key)

            for direction, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)),
                                       ('left', (-1, 0)), ('right', (1, 0))]:
                new_x, new_y = state.position[0] + dx, state.position[1] + dy
                new_pos = (new_x, new_y)

                if new_x < 0 or new_x >= self.map_parser.width or new_y < 0 or new_y >= self.map_parser.height:
                    continue

                if new_pos in self.map_parser.walls:
                    continue

                new_inventory = state.inventory

                if new_pos in self.map_parser.wizards:
                    amulet = self.map_parser.wizards[new_pos]
                    if amulet != 'nothing':
                        new_inventory = state.inventory | {amulet}
                    new_pos = state.position

                if new_pos in self.map_parser.barriers:
                    required_amulet = self.map_parser.barriers[new_pos]
                    if required_amulet not in state.inventory:
                        continue

                new_state = GameState(position=new_pos, inventory=new_inventory)
                new_path = path + [direction]
                new_cost = len(new_path) + heuristic(new_pos)

                counter += 1
                heapq.heappush(frontier, (new_cost, counter, new_state, new_path))

        return []


def count_wizard_interactions_in_path(path: List[str], ascii_map: str, start_char: str = 'M') -> int:
    """Count wizard interactions along a path"""
    if not path:
        return 0

    rows = ascii_map.strip().split('\n')
    wizard_positions = set()
    player_start = None

    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char in ['b', 'r', 'e']:
                wizard_positions.add((x, y))
            elif char == start_char:
                player_start = (x, y)

    if not player_start:
        return 0

    current_pos = player_start
    interactions = 0

    for move in path:
        if move == 'up':
            new_pos = (current_pos[0], current_pos[1] - 1)
        elif move == 'down':
            new_pos = (current_pos[0], current_pos[1] + 1)
        elif move == 'left':
            new_pos = (current_pos[0] - 1, current_pos[1])
        elif move == 'right':
            new_pos = (current_pos[0] + 1, current_pos[1])
        else:
            continue

        if new_pos in wizard_positions:
            interactions += 1
        else:
            current_pos = new_pos

    return interactions


def round_up_to_nearest_five(value: int) -> int:
    """Round UP to nearest 5"""
    remainder = value % 5
    if remainder == 0:
        return value
    else:
        return value + (5 - remainder)


def calculate_steps_remaining(ascii_map: str, goal_number: int) -> int:
    """Calculate stepsRemaining based on M player optimal path"""
    try:
        pathfinder = MPlayerPathfinder(ascii_map)
        optimal_path = pathfinder.find_optimal_path_to_goal(goal_number)

        if optimal_path:
            path_length = len(optimal_path)
            wizard_interactions = count_wizard_interactions_in_path(optimal_path, ascii_map, 'M')
            total_cost = (path_length * MOVE_COST) + (wizard_interactions * WIZARD_INTERACTION_COST)
            rounded_cost = round_up_to_nearest_five(total_cost)
            return rounded_cost + BUFFER_POINTS
        else:
            return 50  # Default fallback
    except:
        return 50


def parse_level_entities(ascii_map):
    """Parse ASCII map to extract entity positions"""
    entities = {
        "wizards": {},
        "barriers": {},
        "treasures": {},
        "start_pos": {},
    }

    lines = ascii_map.strip().split("\n")
    wizard_counter = 1
    gem_counter = 1

    b_barriers = []
    r_barriers = []

    for y, line in enumerate(lines):
        if not line.strip() or (len(line) >= 2 and line[1] == ":"):
            break

        for x, char in enumerate(line):
            if char == "e":
                entities["wizards"][f"wizard{wizard_counter}"] = (x, y)
                wizard_counter += 1
            elif char == "b":
                entities["wizards"][f"wizard{wizard_counter}"] = (x, y)
                wizard_counter += 1
            elif char == "r":
                entities["wizards"][f"wizard{wizard_counter}"] = (x, y)
                wizard_counter += 1
            elif char == "B":
                b_barriers.append((x, y))
            elif char == "R":
                r_barriers.append((x, y))
            elif char in ["g", "G"]:
                entities["treasures"][f"gem{gem_counter}"] = (x, y)
                gem_counter += 1
            elif char == "X":
                entities["start_pos"]["agent2"] = (x, y)
            elif char == "Y":
                entities["start_pos"]["agent3"] = (x, y)
            elif char == "M":
                if "agent2" not in entities["start_pos"]:
                    entities["start_pos"]["agent2"] = (x, y)

    door_counter = 1
    for pos in b_barriers:
        entities["barriers"][f"door{door_counter}"] = pos
        door_counter += 1
    for pos in r_barriers:
        entities["barriers"][f"door{door_counter}"] = pos
        door_counter += 1

    return entities


def convert_exp4_plan_to_movement_path(exp4_plan, level_entities):
    """Convert exp4 action plan to movement path"""
    if not exp4_plan:
        return []

    movement_path = []

    if exp4_plan and isinstance(exp4_plan[0], dict):
        current_x = exp4_plan[0].get("x", 1) - 1
        current_y = exp4_plan[0].get("y", 1) - 1
    else:
        current_x, current_y = 0, 0

    for action_entry in exp4_plan:
        if not isinstance(action_entry, dict):
            continue

        action_str = action_entry.get("action", "")
        agent_x = action_entry.get("x", 1) - 1
        agent_y = action_entry.get("y", 1) - 1

        # Extract action type
        if "(" in action_str:
            action_type = action_str.split("(")[0].lower()
            params_str = action_str.split("(")[1].rstrip(")")
        else:
            action_type = action_str.lower()
            params_str = ""

        # Handle different action types
        if action_type in ["up", "down", "left", "right"]:
            movement_path.append(action_type)
            if action_type == "up":
                current_y -= 1
            elif action_type == "down":
                current_y += 1
            elif action_type == "left":
                current_x -= 1
            elif action_type == "right":
                current_x += 1

        elif action_type == "interact":
            wizard_name = params_str.strip()
            if wizard_name in level_entities["wizards"]:
                wizard_x, wizard_y = level_entities["wizards"][wizard_name]
                direction = get_direction(agent_x, agent_y, wizard_x, wizard_y)
                if direction:
                    movement_path.append(direction)

        elif action_type == "pickup":
            parts = [p.strip() for p in params_str.split(",")]
            gem_name = None
            for part in parts:
                if "gem" in part:
                    gem_name = part
                    break

            if gem_name and gem_name in level_entities["treasures"]:
                gem_x, gem_y = level_entities["treasures"][gem_name]
                direction = get_direction(agent_x, agent_y, gem_x, gem_y)
                if direction:
                    movement_path.append(direction)
                    current_x, current_y = gem_x, gem_y

        elif action_type == "final_position":
            final_x, final_y = agent_x, agent_y
            if current_x != final_x or current_y != final_y:
                final_path = compute_path_to_target(current_x, current_y, final_x, final_y)
                movement_path.extend(final_path)
                current_x, current_y = final_x, final_y

    return movement_path


def get_direction(from_x, from_y, to_x, to_y):
    """Compute direction from one position to another"""
    dx = to_x - from_x
    dy = to_y - from_y

    if dx == 1 and dy == 0:
        return "right"
    elif dx == -1 and dy == 0:
        return "left"
    elif dx == 0 and dy == 1:
        return "down"
    elif dx == 0 and dy == -1:
        return "up"
    else:
        return None


def compute_path_to_target(from_x, from_y, to_x, to_y):
    """Compute simple path using horizontal-then-vertical movement"""
    path = []
    current_x, current_y = from_x, from_y

    while current_x != to_x:
        if current_x < to_x:
            path.append("right")
            current_x += 1
        else:
            path.append("left")
            current_x -= 1

    while current_y != to_y:
        if current_y < to_y:
            path.append("down")
            current_y += 1
        else:
            path.append("up")
            current_y -= 1

    return path


def generate_typescript_level(level_id, ascii_map, x_goals, y_goals, x_expertise, y_expertise, steps_remaining, goal_type):
    """Generate TypeScript level file content"""

    # Convert X to Z and Y to O in ASCII map
    ascii_map_converted = ascii_map.replace('X', 'Z').replace('Y', 'O')

    # Generate Agent 2 (X) movements
    agent2_movements = {}
    for i, (goal, expertise) in enumerate(zip(x_goals, x_expertise)):
        movement_type = map_expertise_to_movement_type(expertise)
        agent2_movements[f'experienced{i+1}'] = {
            'path': [],
            'goal': goal,
            'type': movement_type
        }

    # Generate Agent 3 (Y) movements
    agent3_movements = {}
    for i, (goal, expertise) in enumerate(zip(y_goals, y_expertise)):
        movement_type = map_expertise_to_movement_type(expertise)
        agent3_movements[f'experienced{i+1}'] = {
            'path': [],
            'goal': goal,
            'type': movement_type
        }

    # Use goal_type determined from the map (uppercase G position)
    primary_goal_type = goal_type

    # Build TypeScript content
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
    }},
    3: {{
      movements: {{
        experienced1: {{
          path: [],
          goal: {agent3_movements['experienced1']['goal']},
          type: '{agent3_movements['experienced1']['type']}'
        }},
        experienced2: {{
          path: [],
          goal: {agent3_movements['experienced2']['goal']},
          type: '{agent3_movements['experienced2']['type']}'
        }},
        experienced3: {{
          path: [],
          goal: {agent3_movements['experienced3']['goal']},
          type: '{agent3_movements['experienced3']['type']}'
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

    return ts_content


def update_level_with_paths(level_content, agent_num, movement_index, movement_path):
    """Update movement path in level content"""
    agent_section_pattern = rf"{agent_num}:\s*\{{\s*movements:\s*\{{"
    agent_section_match = re.search(agent_section_pattern, level_content, re.MULTILINE | re.DOTALL)

    if not agent_section_match:
        return level_content

    agent_start_pos = agent_section_match.end()

    brace_count = 1
    agent_end_pos = agent_start_pos
    for i in range(agent_start_pos, len(level_content)):
        if level_content[i] == "{":
            brace_count += 1
        elif level_content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                agent_end_pos = i
                break

    agent_section = level_content[agent_start_pos:agent_end_pos]

    pattern = rf"(experienced{movement_index}:\s*\{{\s*path:\s*\[)[^\]]*(\]\s*,\s*goal:\s*\d+\s*,\s*type:\s*\'[^\']*\')"
    match = re.search(pattern, agent_section, re.MULTILINE | re.DOTALL)

    if match:
        path_str = ", ".join([f'"{move}"' for move in movement_path])
        new_section = f"{match.group(1)}{path_str}{match.group(2)}"
        updated_agent_section = agent_section.replace(match.group(0), new_section, 1)
        level_content = (
            level_content[:agent_start_pos]
            + updated_agent_section
            + level_content[agent_end_pos:]
        )

    return level_content


def main():
    """Main conversion function"""

    ascii_dir = "../../extracted_ascii_maps/problem_exp4"
    output_dir = "../../src/data/levels/exp4"
    pathing_file = "../../extracted_ascii_maps/paths_exp4/pathing_exp4.json"

    # Files to skip (not in problem_exp4)
    skip_files = {'sm111', 'sm112', 'sm631'}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load pathing data
    pathing_data = {}
    if os.path.exists(pathing_file):
        with open(pathing_file, 'r') as f:
            pathing_data = json.load(f)
        print(f"✓ Loaded pathing data for {len(pathing_data)} maps\n")

    # Find all ASCII files
    pattern = os.path.join(ascii_dir, "*.txt")
    ascii_files = sorted(glob.glob(pattern))

    if not ascii_files:
        print(f"Error: No ASCII files found in {ascii_dir}")
        return

    print(f"Found {len(ascii_files)} ASCII files to process\n")

    processed = []
    failed = []

    for ascii_file in ascii_files:
        filename = Path(ascii_file).stem

        # Skip files we don't want to process
        if filename in skip_files:
            print(f"Skipping: {filename} (not in processing list)")
            continue

        level_id = filename
        print(f"Processing: {level_id}")

        try:
            # Parse ASCII map with goals
            ascii_map, x_goals, y_goals, x_expertise, y_expertise = parse_ascii_file_with_goals(ascii_file)
            print(f"  X goals: {x_goals} ({x_expertise}), Y goals: {y_goals} ({y_expertise})")

            # Determine goal type from treasure positions
            goal_type = determine_treasure_type_from_map(ascii_map)
            print(f"  Goal type: {goal_type}")

            # Calculate stepsRemaining (use first goal number)
            goal_number = x_goals[0]
            steps_remaining = calculate_steps_remaining(ascii_map, goal_number)
            print(f"  Steps remaining: {steps_remaining}")

            # Generate TypeScript content
            ts_content = generate_typescript_level(
                level_id, ascii_map, x_goals, y_goals, x_expertise, y_expertise, steps_remaining, goal_type
            )

            # Now add paths from pathing data if available
            if level_id in pathing_data:
                print(f"  Found pathing data, adding paths...")
                level_entities = parse_level_entities(ascii_map)

                scenarios = pathing_data[level_id]
                scenario_counter = 1
                for scenario_name, agents in scenarios.items():
                    for exp4_agent_name, agent_data in agents.items():
                        # Map agent names
                        agent_num = 3 if exp4_agent_name == "agent3" else 2

                        # Convert plan to path
                        exp4_plan = agent_data.get("plan", [])
                        movement_path = convert_exp4_plan_to_movement_path(exp4_plan, level_entities)

                        # Map to movement index
                        movement_index = int(scenario_name.replace("scenario", ""))

                        print(f"    Agent {agent_num} experienced{movement_index}: {len(movement_path)} moves")

                        # Update level content
                        ts_content = update_level_with_paths(
                            ts_content, agent_num, movement_index, movement_path
                        )

                    scenario_counter += 1

            # Write TypeScript file
            output_path = os.path.join(output_dir, f"{level_id}.ts")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ts_content)

            print(f"  ✓ Created: {output_path}")
            processed.append(level_id)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            failed.append(filename)

        print()

    # Summary
    print("=" * 60)
    print(f"Successfully processed: {len(processed)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skip_files)}")

    if failed:
        print(f"\nFailed files: {', '.join(failed)}")

    print(f"\nProcessed: {', '.join(processed)}")
    print("\n✓ Update complete!")


if __name__ == "__main__":
    main()
