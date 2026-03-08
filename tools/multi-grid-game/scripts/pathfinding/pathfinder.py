"""
A* Pathfinding for Multi-Grid Game
Generates optimal and suboptimal paths for different agent types
"""
import heapq
import random
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from map_parser import Position, ParsedMap, Agent, Treasure, Wizard, Barrier, is_walkable, get_neighbors


@dataclass
class PathResult:
    path: List[str]  # List of movement directions
    goal_reached: bool
    steps: int


class Pathfinder:
    def __init__(self, parsed_map: ParsedMap):
        self.map = parsed_map
        self.opened_barriers: Set[Position] = set()
        self.collected_items: Set[str] = set()
    
    def reset_state(self):
        """Reset pathfinder state for new path generation"""
        self.opened_barriers = set()
        self.collected_items = set()
    
    def heuristic(self, pos1: Position, pos2: Position) -> int:
        """Manhattan distance heuristic"""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
    
    def find_path_astar(self, start: Position, goal: Position, 
                       avoid_positions: Set[Position] = None) -> Optional[List[Tuple[Position, str]]]:
        """A* pathfinding algorithm"""
        if avoid_positions is None:
            avoid_positions = set()
        
        open_set = [(0, start, [])]  # (f_score, position, path)
        closed_set = set()
        g_scores = {start: 0}
        
        while open_set:
            current_f, current_pos, path = heapq.heappop(open_set)
            
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            
            if current_pos == goal:
                return path
            
            for neighbor_pos, direction in get_neighbors(current_pos):
                if (neighbor_pos in closed_set or 
                    neighbor_pos in avoid_positions or
                    not is_walkable(neighbor_pos, self.map, self.opened_barriers)):
                    continue
                
                tentative_g = g_scores[current_pos] + 1
                
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor_pos, goal)
                    new_path = path + [(current_pos, direction)]
                    heapq.heappush(open_set, (f_score, neighbor_pos, new_path))
        
        return None  # No path found
    
    def generate_expert_path(self, agent: Agent, goal_pos: Position) -> PathResult:
        """Generate optimal path for expert agent, handling wizard interactions"""
        # First check if we can reach goal directly
        direct_path = self.find_path_astar(agent.position, goal_pos)
        if direct_path is not None:
            directions = [direction for _, direction in direct_path]
            return PathResult(directions, True, len(directions))
        
        # Need to collect items from wizards first
        required_wizard_positions = self.find_required_wizards_for_goal(agent, goal_pos)
        if not required_wizard_positions:
            return PathResult([], False, 0)
        
        # Generate path visiting required wizards then goal
        full_path = []
        current_pos = agent.position
        
        for wizard_pos in required_wizard_positions:
            path_to_wizard = self.find_path_astar(current_pos, wizard_pos)
            if path_to_wizard is None:
                return PathResult([], False, 0)
            
            # Add path to wizard
            wizard_directions = [direction for _, direction in path_to_wizard]
            full_path.extend(wizard_directions)
            current_pos = wizard_pos
            
            # Simulate collecting the item and opening barriers
            wizard = next(w for w in self.map.wizards if w.position == wizard_pos)
            if wizard.amulet_type != 'none':
                amulet_item = f"{wizard.amulet_type}Amulet"
                # Open barriers that require this item
                for barrier in self.map.barriers:
                    if amulet_item in barrier.required_items:
                        self.opened_barriers.add(barrier.position)
        
        # Now try to reach the goal
        final_path = self.find_path_astar(current_pos, goal_pos)
        if final_path is None:
            return PathResult([], False, 0)
        
        final_directions = [direction for _, direction in final_path]
        full_path.extend(final_directions)
        
        return PathResult(full_path, True, len(full_path))
    
    def generate_novice_path(self, agent: Agent, goal_pos: Position, 
                           exploration_factor: float = 0.3) -> PathResult:
        """Generate suboptimal path with exploration/mistakes for novice agent"""
        # Reset opened barriers for novice path generation
        original_opened = self.opened_barriers.copy()
        self.opened_barriers = set()
        
        # Check if we need wizards for this goal
        required_wizard_positions = self.find_required_wizards_for_goal(agent, goal_pos)
        
        directions = []
        current_pos = agent.position
        visited_positions = set()
        collected_items = set()
        
        # If we need wizards, visit them first (with some exploration)
        for wizard_pos in required_wizard_positions:
            # Navigate to wizard with exploration
            while current_pos != wizard_pos and len(directions) < 100:
                visited_positions.add(current_pos)
                
                # Sometimes explore
                if random.random() < exploration_factor and len(directions) < 80:
                    exploration_moves = random.randint(1, 2)
                    for _ in range(exploration_moves):
                        neighbors = [(pos, direction) for pos, direction in get_neighbors(current_pos)
                                   if is_walkable(pos, self.map, self.opened_barriers)]
                        if neighbors:
                            next_pos, direction = random.choice(neighbors)
                            directions.append(direction)
                            current_pos = self._move_position(current_pos, direction)
                
                # Move toward wizard
                path_to_wizard = self.find_path_astar(current_pos, wizard_pos, visited_positions)
                if path_to_wizard and len(path_to_wizard) > 0:
                    _, next_direction = path_to_wizard[0]
                    directions.append(next_direction)
                    current_pos = self._move_position(current_pos, next_direction)
                else:
                    break
            
            # Collect item from wizard
            if current_pos == wizard_pos:
                wizard = next(w for w in self.map.wizards if w.position == wizard_pos)
                if wizard.amulet_type != 'none':
                    amulet_item = f"{wizard.amulet_type}Amulet"
                    collected_items.add(amulet_item)
                    # Open barriers
                    for barrier in self.map.barriers:
                        if amulet_item in barrier.required_items:
                            self.opened_barriers.add(barrier.position)
        
        # Now navigate to goal with exploration
        while current_pos != goal_pos and len(directions) < 150:
            visited_positions.add(current_pos)
            
            # Sometimes explore
            if random.random() < exploration_factor and len(directions) < 120:
                exploration_moves = random.randint(1, 3)
                for _ in range(exploration_moves):
                    neighbors = [(pos, direction) for pos, direction in get_neighbors(current_pos)
                               if is_walkable(pos, self.map, self.opened_barriers)]
                    if neighbors:
                        next_pos, direction = random.choice(neighbors)
                        directions.append(direction)
                        current_pos = self._move_position(current_pos, direction)
            
            # Move toward goal
            optimal_path = self.find_path_astar(current_pos, goal_pos, visited_positions)
            if optimal_path and len(optimal_path) > 0:
                _, next_direction = optimal_path[0]
                directions.append(next_direction)
                current_pos = self._move_position(current_pos, next_direction)
            else:
                # Try any valid move
                neighbors = [(pos, direction) for pos, direction in get_neighbors(current_pos)
                           if is_walkable(pos, self.map, self.opened_barriers)]
                if neighbors:
                    next_pos, direction = random.choice(neighbors)
                    directions.append(direction)
                    current_pos = next_pos
                else:
                    break
        
        # Restore original opened barriers
        self.opened_barriers = original_opened
        
        return PathResult(directions, current_pos == goal_pos, len(directions))
    
    def _move_position(self, pos: Position, direction: str) -> Position:
        """Apply movement direction to position"""
        if direction == 'up':
            return Position(pos.x, pos.y - 1)
        elif direction == 'down':
            return Position(pos.x, pos.y + 1)
        elif direction == 'left':
            return Position(pos.x - 1, pos.y)
        elif direction == 'right':
            return Position(pos.x + 1, pos.y)
        return pos
    
    def find_required_wizards_for_goal(self, agent: Agent, goal_pos: Position) -> List[Position]:
        """Find wizard positions needed to reach the goal through barriers"""
        required_wizards = []
        
        # Check if we can reach goal without collecting items
        direct_path = self.find_path_astar(agent.position, goal_pos)
        if direct_path is not None:
            return []  # No wizards needed
        
        # Try progressively opening barriers to find which ones are blocking us
        blocking_barriers = []
        
        # First, identify all barriers that might be in our path
        for barrier in self.map.barriers:
            # Try opening just this barrier
            temp_opened = self.opened_barriers.copy()
            temp_opened.add(barrier.position)
            
            old_opened = self.opened_barriers
            self.opened_barriers = temp_opened
            path_with_barrier_open = self.find_path_astar(agent.position, goal_pos)
            self.opened_barriers = old_opened
            
            # If opening this barrier helps us get closer, it's potentially blocking
            if path_with_barrier_open is not None:
                blocking_barriers.append(barrier)
        
        # If we found blocking barriers, find wizards for their required items
        for barrier in blocking_barriers:
            for required_item in barrier.required_items:
                amulet_type = required_item.replace('Amulet', '').lower()
                for wizard in self.map.wizards:
                    if wizard.amulet_type == amulet_type and wizard.position not in required_wizards:
                        required_wizards.append(wizard.position)
        
        # If we still can't find a path even with all barriers opened, check if we missed something
        if not required_wizards:
            # Try opening ALL barriers to see if goal is reachable at all
            temp_opened = set(barrier.position for barrier in self.map.barriers)
            old_opened = self.opened_barriers
            self.opened_barriers = temp_opened
            path_all_open = self.find_path_astar(agent.position, goal_pos)
            self.opened_barriers = old_opened
            
            if path_all_open is not None:
                # Goal is reachable if all barriers are opened, so find wizards for all barriers
                for barrier in self.map.barriers:
                    for required_item in barrier.required_items:
                        amulet_type = required_item.replace('Amulet', '').lower()
                        for wizard in self.map.wizards:
                            if wizard.amulet_type == amulet_type and wizard.position not in required_wizards:
                                required_wizards.append(wizard.position)
        
        return required_wizards
    
    def generate_paths_for_agent(self, agent: Agent, treasures: List[Treasure]) -> Dict[str, Dict]:
        """Generate all path variants for an agent to all treasures"""
        paths = {}
        
        for treasure in treasures:
            goal_pos = treasure.position
            
            # Reset state before each path generation
            self.reset_state()
            expert1 = self.generate_expert_path(agent, goal_pos)
            
            self.reset_state()
            expert2 = self.generate_expert_path(agent, goal_pos)
            
            # Generate novice paths (2 variants with different exploration)
            self.reset_state()
            random.seed(42)  # For reproducible results
            novice1 = self.generate_novice_path(agent, goal_pos, 0.2)
            
            self.reset_state()
            random.seed(123)
            novice2 = self.generate_novice_path(agent, goal_pos, 0.4)
            
            paths[treasure.treasure_type] = {
                'experienced1': {
                    'path': expert1.path,
                    'goal': 1,
                    'type': 'Expert'
                },
                'experienced2': {
                    'path': novice1.path,
                    'goal': 2,
                    'type': 'Novice'
                },
                'experienced3': {
                    'path': expert2.path,
                    'goal': 1,
                    'type': 'Expert_2'
                },
                'experienced4': {
                    'path': novice2.path,
                    'goal': 2,
                    'type': 'Novice_2'
                }
            }
        
        return paths