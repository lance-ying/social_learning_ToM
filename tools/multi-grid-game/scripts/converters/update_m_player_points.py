#!/usr/bin/env python3

import os
import sys
import glob
import re
from typing import List, Tuple, Set, Optional
import heapq
from dataclasses import dataclass

# Point costs
MOVE_COST = 3
WIZARD_INTERACTION_COST = 5
BUFFER_POINTS = 15

@dataclass(frozen=True)
class GameState:
    position: Tuple[int, int]
    inventory: frozenset

def parse_typescript_level(file_path: str) -> str:
    """Extract ASCII map from TypeScript level file"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Try with optional space after colon to support both formats
    match = re.search(r'asciiMap:\s*`\s*(.*?)\s*`\.trim\(\)', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

class MPlayerMapParser:
    def __init__(self, ascii_map: str):
        self.ascii_map = ascii_map
        self.walls = set()
        self.wizards = {}  # Position -> amulet type
        self.barriers = {}  # Position -> required amulet
        self.treasures = {}  # Position -> treasure type (A=1, B=2, C=3)
        self.m_start = None  # M player position

        self._parse_map()

        # Calculate map dimensions
        rows = self.ascii_map.strip().split('\n')
        self.height = len(rows)
        self.width = max(len(row) for row in rows) if rows else 0

    def _parse_map(self):
        rows = self.ascii_map.strip().split('\n')

        # First pass: collect all treasure positions
        treasure_positions = []
        for y, row in enumerate(rows):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.walls.add(pos)
                elif char == 'M':  # M is the player!
                    self.m_start = pos
                elif char == 'b':
                    self.wizards[pos] = 'blueAmulet'
                elif char == 'r':
                    self.wizards[pos] = 'redAmulet'
                elif char == 'e':
                    self.wizards[pos] = 'nothing'
                elif char == 'f':
                    self.barriers[pos] = 'blueAmulet'
                elif char == 'd':
                    self.barriers[pos] = 'redAmulet'
                elif char == 'B':
                    self.barriers[pos] = 'blueAmulet'
                elif char == 'R':
                    self.barriers[pos] = 'redAmulet'
                elif char in ['g', 'G']:
                    # Collect treasure positions for later assignment
                    treasure_positions.append((y, x, pos))  # (y, x, pos) for sorting

        # Assign treasure types in order (top-to-bottom, left-to-right)
        # First treasure = A (type 1), second = B (type 2), third = C (type 3)
        treasure_positions.sort()  # Sort by (y, x)
        for idx, (y, x, pos) in enumerate(treasure_positions):
            treasure_type = (idx % 3) + 1  # 1, 2, 3, 1, 2, 3, ...
            self.treasures[pos] = treasure_type

class MPlayerPathfinder:
    def __init__(self, ascii_map: str):
        self.map_parser = MPlayerMapParser(ascii_map)

    def find_optimal_path_to_goal(self, goal_number: int) -> List[str]:
        """Find optimal path from M to specific treasure goal"""
        if not self.map_parser.m_start:
            return []

        # Find treasure position for this goal
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
        """A* pathfinding from M to target treasure"""
        def heuristic(pos: Tuple[int, int]) -> int:
            return abs(pos[0] - target[0]) + abs(pos[1] - target[1])

        frontier = []
        counter = 0  # To break ties in the priority queue
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

            # Try all four directions
            for direction, (dx, dy) in [('up', (0, -1)), ('down', (0, 1)),
                                       ('left', (-1, 0)), ('right', (1, 0))]:
                new_x, new_y = state.position[0] + dx, state.position[1] + dy
                new_pos = (new_x, new_y)

                # Check if move is within map bounds
                if new_x < 0 or new_x >= self.map_parser.width or new_y < 0 or new_y >= self.map_parser.height:
                    continue

                # Check if move is valid (not a wall)
                if new_pos in self.map_parser.walls:
                    continue

                new_inventory = state.inventory

                # Check if there's a wizard at new position
                if new_pos in self.map_parser.wizards:
                    amulet = self.map_parser.wizards[new_pos]
                    if amulet != 'nothing':
                        new_inventory = state.inventory | {amulet}
                    # Stay in place when hitting wizard
                    new_pos = state.position

                # Check if there's a barrier
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
    """Count wizard interactions along a path from M's position"""
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
            # Stay in place
        else:
            current_pos = new_pos

    return interactions

def round_up_to_nearest_five(value: int) -> int:
    """Round UP to nearest 5 or 0"""
    remainder = value % 5
    if remainder == 0:
        return value
    else:
        return value + (5 - remainder)

def analyze_m_player_optimal_cost(file_path: str):
    """Calculate optimal stepsRemaining based on M player's optimal path"""
    try:
        ascii_map = parse_typescript_level(file_path)
        if not ascii_map:
            return None

        # Read current stepsRemaining
        with open(file_path, 'r') as f:
            content = f.read()

        current_steps_match = re.search(r'stepsRemaining:\s*(\d+)', content)
        current_steps = int(current_steps_match.group(1)) if current_steps_match else 0

        # Get goal type
        goal_match = re.search(r'goal:\s*\{\s*type:\s*[\'"](\w+)[\'"]', content)
        goal_type = goal_match.group(1) if goal_match else 'Unknown'

        # Map goal types to numbers
        goal_map = {'A': 1, 'B': 2, 'C': 3}
        goal_number = goal_map.get(goal_type, 0)

        # Initialize pathfinder for M player
        pathfinder = MPlayerPathfinder(ascii_map)

        # Get optimal path for M player
        optimal_path = pathfinder.find_optimal_path_to_goal(goal_number)

        analysis = {
            'file_name': os.path.basename(file_path),
            'current_steps_remaining': current_steps,
            'goal_type': goal_type,
        }

        if optimal_path:
            path_length = len(optimal_path)
            wizard_interactions = count_wizard_interactions_in_path(optimal_path, ascii_map, 'M')

            # Calculate cost
            total_cost = (path_length * MOVE_COST) + (wizard_interactions * WIZARD_INTERACTION_COST)
            rounded_cost = round_up_to_nearest_five(total_cost)
            recommended_steps = rounded_cost + BUFFER_POINTS

            analysis['optimal_path'] = {
                'path_length': path_length,
                'wizard_interactions': wizard_interactions,
                'raw_cost': total_cost,
                'rounded_cost': rounded_cost,
                'recommended_steps_remaining': recommended_steps
            }
            analysis['recommended_steps_remaining'] = recommended_steps
        else:
            analysis['recommended_steps_remaining'] = current_steps
            analysis['error'] = 'Could not find optimal path for M player'

        return analysis

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_all_levels(directory: str):
    """Analyze all levels based on M player position"""
    # Try both s*.ts and mod_s*.ts patterns
    pattern1 = os.path.join(directory, "s*.ts")
    pattern2 = os.path.join(directory, "mod_s*.ts")
    level_files = []
    level_files.extend(glob.glob(pattern1))
    level_files.extend(glob.glob(pattern2))
    level_files = [f for f in level_files if not f.endswith('.backup') and not f.endswith('.goal_backup') and not f.endswith('types.ts')]

    if not level_files:
        print(f"No s*.ts or mod_s*.ts files found in {directory}")
        return None

    print(f"Analyzing {len(level_files)} level files...")
    print(f"Formula: (M_moves × {MOVE_COST}) + (M_wizards × {WIZARD_INTERACTION_COST}), rounded up to nearest 5, + {BUFFER_POINTS} buffer")
    print()

    results = []

    for file_path in sorted(level_files):
        analysis = analyze_m_player_optimal_cost(file_path)
        if analysis:
            results.append(analysis)

    # Print summary table
    print(f"{'File':<15} {'Goal':<5} {'Current':<8} {'M_Moves':<8} {'M_Wiz':<6} {'Raw':<6} {'Rounded':<8} {'Recommended':<12} {'Change':<8}")
    print("-" * 100)

    for result in results:
        file_name = result['file_name'].replace('.ts', '')
        goal = result['goal_type']
        current = result['current_steps_remaining']
        recommended = result['recommended_steps_remaining']
        change = recommended - current

        if 'optimal_path' in result:
            moves = result['optimal_path']['path_length']
            wizards = result['optimal_path']['wizard_interactions']
            raw_cost = result['optimal_path']['raw_cost']
            rounded = result['optimal_path']['rounded_cost']
        else:
            moves = 'N/A'
            wizards = 'N/A'
            raw_cost = 'N/A'
            rounded = 'N/A'

        print(f"{file_name:<15} {goal:<5} {current:<8} {moves:<8} {wizards:<6} {raw_cost:<6} {rounded:<8} {recommended:<12} {change:+d}")

    return results

def update_steps_remaining(directory: str, backup: bool = True):
    """Update stepsRemaining values based on M player optimal paths"""
    results = analyze_all_levels(directory)

    if not results:
        print("No results to update")
        return

    print(f"\nUpdating stepsRemaining values...")

    updated_count = 0

    for result in results:
        file_path = os.path.join(directory, result['file_name'])

        try:
            # Create backup if requested
            if backup:
                backup_path = file_path + ".m_backup"
                if not os.path.exists(backup_path):
                    with open(file_path, 'r') as original:
                        with open(backup_path, 'w') as backup_file:
                            backup_file.write(original.read())

            # Read and update file
            with open(file_path, 'r') as f:
                content = f.read()

            # Update stepsRemaining
            new_steps = result['recommended_steps_remaining']
            pattern = r'stepsRemaining:\s*\d+'
            replacement = f'stepsRemaining: {new_steps}'

            updated_content = re.sub(pattern, replacement, content)

            # Write back
            with open(file_path, 'w') as f:
                f.write(updated_content)

            print(f"✓ Updated {result['file_name']}: {result['current_steps_remaining']} → {new_steps}")
            updated_count += 1

        except Exception as e:
            print(f"✗ Failed to update {result['file_name']}: {e}")

    print(f"\nSuccessfully updated {updated_count} files")

def main():
    # Default to exp4 directory
    if len(sys.argv) < 2:
        # Get the script directory and construct path to exp4
        script_dir = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.join(script_dir, "../../src/data/levels/exp4")
    else:
        directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return

    # Always update by default (no prompt)
    update_files = '--analyze-only' not in sys.argv

    if update_files:
        print(f"Updating stepsRemaining values based on M player optimal paths in {directory}")
        print()
        update_steps_remaining(directory)
    else:
        print(f"Analyzing only (no updates) for {directory}")
        print()
        analyze_all_levels(directory)

if __name__ == "__main__":
    main()
