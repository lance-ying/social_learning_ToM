#!/usr/bin/env python3

import sys
import os
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from collections import deque
import re

@dataclass
class Position:
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

@dataclass
class GameState:
    position: Position
    inventory: Set[str]  # Set of amulets collected
    
    def __hash__(self):
        return hash((self.position, frozenset(self.inventory)))

class MapParser:
    def __init__(self, ascii_map: str):
        self.rows = ascii_map.strip().split('\n')
        self.height = len(self.rows)
        self.width = max(len(row) for row in self.rows) if self.rows else 0
        
        # Parse map elements
        self.walls = set()
        self.wizards = {}  # Position -> amulet_type
        self.barriers = {}  # Position -> required_amulet
        self.treasures = set()
        self.agent_start = None
        
        self._parse_map()
    
    def _parse_map(self):
        # First pass: collect all treasures to sort them
        treasure_positions = []
        for y, row in enumerate(self.rows):
            for x, char in enumerate(row):
                if char in ['g', 'G']:
                    treasure_positions.append(Position(x, y))

        # Sort treasures: first by y (top to bottom), then by x (left to right)
        treasure_positions.sort(key=lambda p: (p.y, p.x))

        # Map positions to labels (A, B, C)
        self.treasure_labels = {}  # Position -> 'A'/'B'/'C'
        for i, pos in enumerate(treasure_positions):
            if i < 3:  # Only label first 3 treasures
                self.treasure_labels[pos] = chr(ord('A') + i)

        # Second pass: parse all map elements
        for y, row in enumerate(self.rows):
            for x, char in enumerate(row):
                pos = Position(x, y)

                if char == 'W':
                    self.walls.add(pos)
                elif char == 'Z':
                    self.agent_start = pos
                elif char == 'b':
                    self.wizards[pos] = 'blueAmulet'
                elif char == 'r':
                    self.wizards[pos] = 'redAmulet'
                elif char == 'e':
                    self.wizards[pos] = 'nothing'
                elif char == 'B':
                    self.barriers[pos] = 'blueAmulet'
                elif char == 'R':
                    self.barriers[pos] = 'redAmulet'
                elif char in ['g', 'G']:
                    self.treasures.add(pos)
    
    def is_valid_position(self, pos: Position) -> bool:
        return (0 <= pos.x < self.width and 
                0 <= pos.y < self.height and 
                pos not in self.walls)
    
    def get_neighbors(self, pos: Position) -> List[Tuple[Position, str]]:
        directions = [
            (Position(pos.x, pos.y - 1), 'up'),
            (Position(pos.x, pos.y + 1), 'down'),
            (Position(pos.x - 1, pos.y), 'left'),
            (Position(pos.x + 1, pos.y), 'right')
        ]
        
        return [(new_pos, direction) for new_pos, direction in directions 
                if self.is_valid_position(new_pos)]

class ZAgentPathfinder:
    def __init__(self, ascii_map: str, goal_type: str = 'A'):
        self.map_parser = MapParser(ascii_map)
        self.goal_type = goal_type
        self.target_treasure = self._get_target_treasure()

    def _get_target_treasure(self) -> Optional[Position]:
        """Get the position of the target treasure based on goal type"""
        for pos, label in self.map_parser.treasure_labels.items():
            if label == self.goal_type:
                return pos
        return None

    def find_path_to_treasures(self) -> List[str]:
        if not self.map_parser.agent_start:
            return []

        # If no specific target, fail
        if not self.target_treasure:
            return []

        start_state = GameState(
            position=self.map_parser.agent_start,
            inventory=set()
        )

        # BFS to find optimal path considering inventory state
        queue = deque([(start_state, [])])
        visited = set([start_state])

        while queue:
            current_state, path = queue.popleft()

            # Check if we reached the TARGET treasure
            if current_state.position == self.target_treasure:
                return path
            
            # Explore all possible moves
            for next_pos, direction in self.map_parser.get_neighbors(current_state.position):
                new_inventory = current_state.inventory.copy()
                actual_next_pos = next_pos
                
                if next_pos in self.map_parser.wizards:
                    amulet_type = self.map_parser.wizards[next_pos]
                    if amulet_type != 'nothing':
                        new_inventory.add(amulet_type)
                    actual_next_pos = current_state.position
                
                elif next_pos in self.map_parser.barriers:
                    required_amulet = self.map_parser.barriers[next_pos]
                    if required_amulet not in current_state.inventory:
                        continue
                
                new_state = GameState(
                    position=actual_next_pos,
                    inventory=new_inventory
                )
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [direction]))
        
        return []
    
    def find_efficient_path(self, goal_preference: Optional[str] = None) -> List[str]:
        if not self.map_parser.agent_start:
            return []
        
        direct_path = self.find_path_to_treasures()
        if direct_path:
            return direct_path
        
        return self._find_path_with_amulet_collection()
    
    def _find_path_with_amulet_collection(self) -> List[str]:
        if not self.map_parser.agent_start:
            return []

        # If no specific target, fail
        if not self.target_treasure:
            return []

        start_state = GameState(
            position=self.map_parser.agent_start,
            inventory=set()
        )

        queue = deque([(start_state, [], 0)])
        visited = {}

        while queue:
            current_state, path, priority = queue.popleft()

            if current_state in visited and visited[current_state] <= len(path):
                continue
            visited[current_state] = len(path)

            # Check if we reached the TARGET treasure
            if current_state.position == self.target_treasure:
                return path
            for next_pos, direction in self.map_parser.get_neighbors(current_state.position):
                new_inventory = current_state.inventory.copy()
                actual_next_pos = next_pos
                new_priority = priority
                
                if next_pos in self.map_parser.wizards:
                    amulet_type = self.map_parser.wizards[next_pos]
                    if amulet_type != 'nothing':
                        new_inventory.add(amulet_type)
                        new_priority += 10
                    actual_next_pos = current_state.position
                
                elif next_pos in self.map_parser.barriers:
                    required_amulet = self.map_parser.barriers[next_pos]
                    if required_amulet not in current_state.inventory:
                        continue
                    new_priority += 5
                
                new_state = GameState(
                    position=actual_next_pos,
                    inventory=new_inventory
                )
                
                if new_state not in visited or visited[new_state] > len(path) + 1:
                    queue.append((new_state, path + [direction], new_priority))
        
        return []

def parse_typescript_level(file_path: str) -> tuple[str, str]:
    """Parse TypeScript level file and return (ascii_map, goal_type)"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        map_match = re.search(r'asciiMap:\s*`\s*\n(.*?)\n\s*`', content, re.DOTALL)
        ascii_map = map_match.group(1) if map_match else ""

        # Parse goal type (A, B, or C)
        goal_match = re.search(r"goal:\s*\{\s*type:\s*['\"]([ABC])['\"]", content)
        goal_type = goal_match.group(1) if goal_match else "A"

        return ascii_map, goal_type
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return "", "A"

def generate_paths_for_mod_file(file_path: str) -> Dict[str, List[str]]:
    ascii_map, goal_type = parse_typescript_level(file_path)
    if not ascii_map:
        return {}

    pathfinder = ZAgentPathfinder(ascii_map, goal_type)

    paths = {}

    expert_path = pathfinder.find_efficient_path()
    if expert_path:
        paths['experienced1'] = expert_path

    novice_path = pathfinder.find_path_to_treasures()
    if novice_path:
        paths['experienced2'] = novice_path

    if expert_path:
        paths['experienced3'] = expert_path.copy()

    if novice_path:
        paths['experienced4'] = novice_path.copy()

    return paths

def update_typescript_file_with_path(file_path: str, new_path: List[str], dry_run: bool = False) -> bool:
    """Update the experienced1 path in a TypeScript level file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Format the new path
        path_str = ', '.join(f'"{step}"' for step in new_path)
        
        # Replace the experienced1 path
        pattern = r'(experienced1:\s*\{\s*path:\s*\[)[^\]]*(\])'
        replacement = r'\1' + path_str + r'\2'
        
        new_content = re.sub(pattern, replacement, content)
        
        if dry_run:
            print(f"Would update {file_path} with path length {len(new_path)}")
            return True
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Updated {os.path.basename(file_path)} with optimal path (length: {len(new_path)})")
        return True
    except Exception as e:
        print(f"✗ Error updating file {file_path}: {e}")
        return False

def process_old_levels():
    """Process all levels in the old directory"""
    # Define base path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels_path = os.path.join(base_path, 'src', 'data', 'levels', 'old')

    # Get all .ts files in the old directory
    if not os.path.exists(levels_path):
        print(f"Error: Directory not found: {levels_path}")
        return

    all_files = [f for f in os.listdir(levels_path) if f.endswith('.ts')]
    level_names = sorted([f[:-3] for f in all_files])  # Remove .ts extension and sort

    print("="*60)
    print(f"Processing {len(level_names)} levels in 'old' directory for optimal paths")
    print("="*60)

    results = []
    for level_name in level_names:
        file_path = os.path.join(levels_path, f'{level_name}.ts')

        if not os.path.exists(file_path):
            print(f"\n✗ File not found: {file_path}")
            continue

        print(f"\n{level_name}:")
        print("-" * 40)

        # Generate optimal path
        ascii_map, goal_type = parse_typescript_level(file_path)
        if not ascii_map:
            print(f"  ✗ Could not parse ASCII map")
            continue

        print(f"  Target goal: {goal_type}")
        pathfinder = ZAgentPathfinder(ascii_map, goal_type)
        optimal_path = pathfinder.find_efficient_path()

        if not optimal_path:
            print(f"  ✗ No path found")
            continue

        print(f"  Optimal path length: {len(optimal_path)}")
        print(f"  Path: {optimal_path}")

        # Update the file
        success = update_typescript_file_with_path(file_path, optimal_path)
        results.append((level_name, len(optimal_path), success))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for level_name, path_length, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {level_name}: {path_length} steps")

    print("\nDone! All levels have been updated with optimal paths.")
    print("Note: These are BFS-optimal paths guaranteed to be shortest.")

def process_specific_levels():
    """Process the specific mod levels requested"""
    # Define base path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels_path = os.path.join(base_path, 'src', 'data', 'levels', 'mod')

    # Specific levels to process
    level_names = ['mod_s411', 'mod_s431', 'mod_s441', 'mod_s531', 'mod_s532', 'mod_s541', 'mod_s542']

    print("="*60)
    print("Processing specific mod levels for optimal paths")
    print("="*60)

    results = []
    for level_name in level_names:
        file_path = os.path.join(levels_path, f'{level_name}.ts')

        if not os.path.exists(file_path):
            print(f"\n✗ File not found: {file_path}")
            continue

        print(f"\n{level_name}:")
        print("-" * 40)

        # Generate optimal path
        ascii_map, goal_type = parse_typescript_level(file_path)
        if not ascii_map:
            print(f"  ✗ Could not parse ASCII map")
            continue

        print(f"  Target goal: {goal_type}")
        pathfinder = ZAgentPathfinder(ascii_map, goal_type)
        optimal_path = pathfinder.find_efficient_path()

        if not optimal_path:
            print(f"  ✗ No path found")
            continue

        print(f"  Optimal path length: {len(optimal_path)}")
        print(f"  Path: {optimal_path}")

        # Update the file
        success = update_typescript_file_with_path(file_path, optimal_path)
        results.append((level_name, len(optimal_path), success))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for level_name, path_length, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {level_name}: {path_length} steps")

    print("\nDone! All levels have been updated with optimal paths.")
    print("Note: These are BFS-optimal paths guaranteed to be shortest.")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == '--update-specific':
            # Process specific levels
            process_specific_levels()
        elif sys.argv[1] == '--update-old':
            # Process all old levels
            process_old_levels()
        else:
            # Single file mode
            file_path = sys.argv[1]
            print(f"Generating paths for: {file_path}")
            paths = generate_paths_for_mod_file(file_path)

            for path_name, path in paths.items():
                print(f"\n{path_name}:")
                print(f"Path length: {len(path)}")
                print(f"Path: {path}")
    else:
        # Default: process specific levels
        process_specific_levels()

if __name__ == "__main__":
    main()