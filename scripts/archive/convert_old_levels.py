#!/usr/bin/env python3
"""
Convert mapdata_old.js levels to TypeScript level format

This script converts the old coordinate-based level format to the new
ASCII map format used in the levels/ directory.
"""

import re
import json
import os
from typing import Dict, List, Any, Set, Tuple

class OldLevelConverter:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.levels = {}
        
        # Symbol mappings for ASCII map
        self.symbol_map = {
            'wall': 'W',
            'player': 'M',
            'npc': 'Z',  # First NPC becomes Z (agent 1)
            'treasure': 'g',
            'wizard_empty': 'e',
            'wizard_blue': 'b',
            'wizard_red': 'r',
            'barrier_blue': 'B',
            'barrier_red': 'R',
            'empty': '.'
        }
    
    def parse_js_file(self):
        """Parse the JavaScript file and extract level data"""
        print(f"Reading {self.input_file}...")
        
        with open(self.input_file, 'r') as f:
            content = f.read()
        
        # Find all level definitions using regex
        level_pattern = r'(level\d+):\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}(?=,\s*(?:level\d+:|}\s*;))'
        
        # Split the content to find individual levels
        # First, remove the outer structure
        start_marker = 'const mapData = {'
        end_marker = '};'
        
        start_idx = content.find(start_marker)
        end_idx = content.rfind(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Could not find mapData object boundaries")
        
        levels_content = content[start_idx + len(start_marker):end_idx].strip()
        
        # Now parse individual levels
        self._parse_levels_content(levels_content)
        
        print(f"Found {len(self.levels)} levels")
        return self.levels
    
    def _parse_levels_content(self, content: str):
        """Parse the levels from the content string"""
        # Find level boundaries by looking for level patterns
        level_starts = []
        pattern = r'(level\d+):\s*\{'
        
        for match in re.finditer(pattern, content):
            level_starts.append((match.group(1), match.start()))
        
        # Parse each level
        for i, (level_name, start_pos) in enumerate(level_starts):
            # Find the end of this level (start of next level or end of content)
            if i + 1 < len(level_starts):
                end_pos = level_starts[i + 1][1]
                level_content = content[start_pos:end_pos].rstrip(',\n ')
            else:
                level_content = content[start_pos:].rstrip(',\n ')
            
            # Parse this individual level
            try:
                level_data = self._parse_single_level(level_content)
                self.levels[level_name] = level_data
                print(f"  Parsed {level_name}")
            except Exception as e:
                print(f"  Error parsing {level_name}: {e}")
    
    def _parse_single_level(self, level_content: str) -> Dict[str, Any]:
        """Parse a single level's content"""
        level_data = {}
        
        # Extract player position
        player_match = re.search(r'player:\s*\{\s*x:\s*(\d+),\s*y:\s*(\d+)\s*\}', level_content)
        if player_match:
            level_data['player'] = {'x': int(player_match.group(1)), 'y': int(player_match.group(2))}
        
        # Extract NPC data
        npc_pattern = r'npc:\s*\{\s*x:\s*(\d+),\s*y:\s*(\d+),\s*movements:\s*\{(.*?)\}\s*,\s*currentMovementIndex'
        npc_match = re.search(npc_pattern, level_content, re.DOTALL)
        if npc_match:
            npc_x, npc_y = int(npc_match.group(1)), int(npc_match.group(2))
            movements_content = npc_match.group(3)
            movements = self._parse_movements(movements_content)
            level_data['npc'] = {'x': npc_x, 'y': npc_y, 'movements': movements}
        
        # Extract arrays (barriers, treasurePots, wizards, blocks)
        level_data['barriers'] = self._parse_array_field(level_content, 'barriers')
        level_data['treasurePots'] = self._parse_array_field(level_content, 'treasurePots')
        level_data['wizards'] = self._parse_array_field(level_content, 'wizards')
        level_data['blocks'] = self._parse_array_field(level_content, 'blocks')
        
        # Extract step limit
        step_match = re.search(r'stepLimit:\s*(\d+)', level_content)
        if step_match:
            level_data['stepLimit'] = int(step_match.group(1))
        
        # Extract goal
        goal_match = re.search(r'goal:\s*\{\s*type:\s*[\'"]([^\'"]+)[\'"]\s*,\s*description:\s*[\'"]([^\'"]+)[\'"]\s*\}', level_content)
        if goal_match:
            level_data['goal'] = {'type': goal_match.group(1), 'description': goal_match.group(2)}
        
        return level_data
    
    def _parse_movements(self, movements_content: str) -> Dict[str, Any]:
        """Parse the movements object"""
        movements = {}
        
        # Find each movement type - improved regex to handle multiline arrays
        movement_pattern = r'(\w+):\s*\{\s*path:\s*\[(.*?)\]\s*,\s*goal:\s*(\d+)\s*\}'
        
        for match in re.finditer(movement_pattern, movements_content, re.DOTALL):
            movement_name = match.group(1)
            path_content = match.group(2)
            goal = int(match.group(3))
            
            # Parse the path array
            path = []
            if path_content.strip():
                # Handle both single and double quotes, and multiline arrays
                path_items = re.findall(r'[\'"]([^\'"]+)[\'"]', path_content)
                path = path_items
            
            movements[movement_name] = {'path': path, 'goal': goal}
        
        return movements
    
    def _parse_array_field(self, content: str, field_name: str) -> List[Dict[str, Any]]:
        """Parse an array field like barriers, treasurePots, etc."""
        # Improved regex to handle nested arrays and objects properly
        start_pattern = f'{field_name}:\\s*\\['
        
        # Find the start of the array
        start_match = re.search(start_pattern, content)
        if not start_match:
            return []
        
        # Find the matching closing bracket
        start_pos = start_match.end() - 1  # Position of the opening '['
        bracket_count = 0
        pos = start_pos
        
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    array_content = content[start_pos + 1:i]  # Content between [ and ]
                    break
        else:
            return []
        
        items = []
        
        # Debug output for barriers
        # if field_name == 'barriers':
        #     print(f"  Debug: Found {field_name} array content: {array_content[:200]}...")
        
        # Find individual objects in the array
        if field_name == 'blocks':
            # Blocks have a simple {x:1,y:2} format
            block_pattern = r'\{\s*x:\s*(\d+)\s*,\s*y:\s*(\d+)\s*\}'
            for block_match in re.finditer(block_pattern, array_content):
                items.append({
                    'x': int(block_match.group(1)),
                    'y': int(block_match.group(2))
                })
        else:
            # Other objects have more complex structure - find balanced braces
            brace_count = 0
            start_obj = None
            
            for i, char in enumerate(array_content):
                if char == '{':
                    if brace_count == 0:
                        start_obj = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_obj is not None:
                        obj_content = array_content[start_obj + 1:i]
                        obj = self._parse_object_content(obj_content)
                        if obj:
                            items.append(obj)
                            # Debug output for barriers
                            # if field_name == 'barriers':
                            #     print(f"    Debug: Parsed barrier: {obj}")
        
        return items
    
    def _parse_object_content(self, content: str) -> Dict[str, Any]:
        """Parse the content of an object"""
        obj = {}
        
        # Parse x, y coordinates
        x_match = re.search(r'x:\s*(\d+)', content)
        y_match = re.search(r'y:\s*(\d+)', content)
        if x_match and y_match:
            obj['x'] = int(x_match.group(1))
            obj['y'] = int(y_match.group(1))
        
        # Parse other string fields
        for field in ['type', 'color', 'label', 'content']:
            field_match = re.search(f'{field}:\\s*[\'"]([^\'"]+)[\'"]', content)
            if field_match:
                obj[field] = field_match.group(1)
        
        # Parse requiredItems array
        required_match = re.search(r'requiredItems:\s*\[([^\]]+)\]', content)
        if required_match:
            items_content = required_match.group(1)
            items = re.findall(r'[\'"]([^\'"]+)[\'"]', items_content)
            obj['requiredItems'] = items
        return obj
    
    def determine_grid_size(self, level_data: Dict[str, Any]) -> Tuple[int, int]:
        """Determine the grid size needed for this level"""
        max_x, max_y = 0, 0
        
        # Check all coordinate-containing objects
        for obj_list in [level_data.get('blocks', []), level_data.get('barriers', []),
                        level_data.get('treasurePots', []), level_data.get('wizards', [])]:
            for obj in obj_list:
                max_x = max(max_x, obj.get('x', 0))
                max_y = max(max_y, obj.get('y', 0))
        
        # Check player and NPC positions
        if 'player' in level_data:
            max_x = max(max_x, level_data['player']['x'])
            max_y = max(max_y, level_data['player']['y'])
        
        if 'npc' in level_data:
            max_x = max(max_x, level_data['npc']['x'])
            max_y = max(max_y, level_data['npc']['y'])
        
        return max_x + 1, max_y + 1
    
    def convert_to_ascii_map(self, level_data: Dict[str, Any]) -> str:
        """Convert level data to ASCII map format"""
        width, height = self.determine_grid_size(level_data)
        
        # Initialize grid with empty spaces
        grid = [['.' for _ in range(width)] for _ in range(height)]
        

        
        # Place blocks (walls)
        for block in level_data.get('blocks', []):
            x, y = block['x'], block['y']
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 'W'
        
        # Place player
        if 'player' in level_data:
            x, y = level_data['player']['x'], level_data['player']['y']
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 'M'
        
        # Place NPC (becomes agent Z)
        if 'npc' in level_data:
            x, y = level_data['npc']['x'], level_data['npc']['y']
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 'Z'
        
        # Place treasures
        for treasure in level_data.get('treasurePots', []):
            x, y = treasure['x'], treasure['y']
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 'g'
        
        # Place wizards
        for wizard in level_data.get('wizards', []):
            x, y = wizard['x'], wizard['y']
            if 0 <= x < width and 0 <= y < height:
                content = wizard.get('content', 'nothing')
                if content == 'blueAmulet':
                    grid[y][x] = 'b'
                elif content == 'redAmulet':
                    grid[y][x] = 'r'
                else:
                    grid[y][x] = 'e'
        
        # Place barriers
        for barrier in level_data.get('barriers', []):
            x, y = barrier['x'], barrier['y']
            if 0 <= x < width and 0 <= y < height:
                required_items = barrier.get('requiredItems', [])
                if 'blueAmulet' in required_items:
                    grid[y][x] = 'B'
                elif 'redAmulet' in required_items:
                    grid[y][x] = 'R'
        
        # Convert grid to string
        return '\n'.join(''.join(row) for row in grid)
    
    def convert_movements_to_agent_paths(self, level_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old movement format to new agentPaths format"""
        agent_paths = {}
        
        if 'npc' not in level_data or 'movements' not in level_data['npc']:
            return agent_paths
        
        movements = level_data['npc']['movements']
        
        # Create agent 1 (Z) paths
        agent_paths['1'] = {
            'movements': {
                'experienced1': {
                    'path': movements.get('experienced1', {}).get('path', []),
                    'goal': movements.get('experienced1', {}).get('goal', 1),
                    'type': 'Expert'
                },
                'experienced2': {
                    'path': movements.get('experienced2', {}).get('path', []),
                    'goal': movements.get('experienced2', {}).get('goal', 2),
                    'type': 'Novice'
                }
            }
        }
        
        # If there are novice1/novice2, add them as experienced3/experienced4
        if 'novice1' in movements:
            agent_paths['1']['movements']['experienced3'] = {
                'path': movements.get('novice1', {}).get('path', []),
                'goal': movements.get('novice1', {}).get('goal', 1),
                'type': 'Expert_2'
            }
        
        if 'novice2' in movements:
            agent_paths['1']['movements']['experienced4'] = {
                'path': movements.get('novice2', {}).get('path', []),
                'goal': movements.get('novice2', {}).get('goal', 2),
                'type': 'Novice_2'
            }
        
        return agent_paths
    
    def generate_typescript_file(self, level_name: str, level_data: Dict[str, Any]) -> str:
        """Generate TypeScript file content for a level"""
        # Convert level111 -> old_s111, etc.
        level_id = f"old_s{level_name[5:]}"  # Remove 'level' prefix
        
        ascii_map = self.convert_to_ascii_map(level_data)
        agent_paths = self.convert_movements_to_agent_paths(level_data)
        
        steps_remaining = level_data.get('stepLimit', 50)
        goal = level_data.get('goal', {'type': 'A', 'description': 'Find and obtain Treasure A'})
        
        ts_content = f"""import {{ LevelConfig }} from './types';

export const {level_id}: LevelConfig = {{
  id: '{level_id}',
  name: '{level_id}',
  asciiMap: `
{ascii_map}
`.trim(),
  agentPaths: {{
{self._format_agent_paths(agent_paths)}
  }},
  stepsRemaining: {steps_remaining},
  goal: {{
    type: '{goal['type']}',
    description: '{goal['description']}'
  }}
}};"""
        
        return ts_content
    
    def _format_agent_paths(self, agent_paths: Dict[str, Any]) -> str:
        """Format agent paths with proper TypeScript syntax"""
        if not agent_paths:
            return ""
        
        formatted_paths = []
        
        for agent_id, agent_data in agent_paths.items():
            movements = agent_data['movements']
            formatted_movements = []
            
            for movement_key, movement_data in movements.items():
                path_str = ', '.join(f'"{move}"' for move in movement_data['path'])
                formatted_movement = f"""        {movement_key}: {{ 
          path: [{path_str}],
          goal: {movement_data['goal']},
          type: '{movement_data['type']}'
        }}"""
                formatted_movements.append(formatted_movement)
            
            agent_section = f"""    {agent_id}: {{
      movements: {{
{',\n'.join(formatted_movements)}
      }}
    }}"""
            formatted_paths.append(agent_section)
        
        return ',\n'.join(formatted_paths)
    
    def convert_all_levels(self):
        """Convert all levels and save to TypeScript files"""
        self.parse_js_file()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\nConverting {len(self.levels)} levels to TypeScript...")
        
        converted_files = []
        for level_name, level_data in self.levels.items():
            try:
                ts_content = self.generate_typescript_file(level_name, level_data)
                
                # Generate filename
                level_id = f"old_s{level_name[5:]}"  # Remove 'level' prefix
                filename = f"{level_id}.ts"
                filepath = os.path.join(self.output_dir, filename)
                
                # Write file
                with open(filepath, 'w') as f:
                    f.write(ts_content)
                
                converted_files.append(filename)
                print(f"  Created {filename}")
                
            except Exception as e:
                print(f"  Error converting {level_name}: {e}")
        
        print(f"\nConversion complete! Created {len(converted_files)} files:")
        for filename in sorted(converted_files):
            print(f"  - {filename}")
        
        return converted_files

def main():
    # File paths
    input_file = "src/data/mapdata_old.js"
    output_dir = "src/data/levels"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Create converter and run conversion
    converter = OldLevelConverter(input_file, output_dir)
    converter.convert_all_levels()

if __name__ == "__main__":
    main()