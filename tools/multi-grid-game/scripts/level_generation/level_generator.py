"""
Level Generator - Converts ASCII maps to TypeScript level configurations
"""
import json
from typing import Dict, Any
from map_parser import parse_ascii_map, ParsedMap
from pathfinder import Pathfinder


class LevelGenerator:
    def __init__(self):
        pass
    
    def generate_level_config(self, level_id: str, ascii_map: str, 
                            steps_remaining: int = 50, 
                            goal_type: str = 'A',
                            goal_description: str = None) -> Dict[str, Any]:
        """Generate complete level configuration from ASCII map"""
        
        if goal_description is None:
            goal_description = f'Find and obtain Treasure {goal_type}'
        
        # Parse the map
        parsed_map = parse_ascii_map(ascii_map)
        
        # Create pathfinder
        pathfinder = Pathfinder(parsed_map)
        
        # Generate paths for all agents
        agent_paths = {}
        for agent in parsed_map.agents:
            # Generate paths to all treasures, but we'll use the ones for the goal
            treasure_paths = pathfinder.generate_paths_for_agent(agent, parsed_map.treasures)
            
            # Find the target treasure(s) based on goal_type
            target_movements = None
            if parsed_map.treasures:
                # First try to find treasure with matching type
                if goal_type in treasure_paths and treasure_paths[goal_type]:
                    target_movements = treasure_paths[goal_type]
                else:
                    # Fallback to first available treasure with valid paths
                    for treasure_type, movements in treasure_paths.items():
                        if movements and movements.get('experienced1', {}).get('path'):
                            target_movements = movements
                            break
                
                # Only add agent if we found valid movements
                if target_movements and target_movements.get('experienced1', {}).get('path'):
                    agent_paths[str(agent.id)] = {
                        'movements': target_movements
                    }
                else:
                    print(f"Warning: No valid paths found for agent {agent.id} to any treasure")
        
        # Build the level configuration
        level_config = {
            'id': level_id,
            'name': level_id,
            'asciiMap': ascii_map.strip(),
            'agentPaths': agent_paths,
            'stepsRemaining': steps_remaining,
            'goal': {
                'type': goal_type,
                'description': goal_description
            }
        }
        
        return level_config
    
    def export_typescript_level(self, level_config: Dict[str, Any]) -> str:
        """Export level configuration as TypeScript code"""
        level_id = level_config['id']
        
        ts_code = f"""import {{ LevelConfig }} from './types';

export const {level_id}: LevelConfig = {{
  id: '{level_config['id']}',
  name: '{level_config['name']}',
  asciiMap:`
{self._format_ascii_map(level_config['asciiMap'])}
`.trim(),
  agentPaths: {{
{self._format_agent_paths(level_config['agentPaths'])}
  }},
  stepsRemaining: {level_config['stepsRemaining']},
  goal: {{
    type: '{level_config['goal']['type']}',
    description: '{level_config['goal']['description']}'
  }}
}};"""
        
        return ts_code
    
    def _format_ascii_map(self, ascii_map: str) -> str:
        """Format ASCII map with proper indentation"""
        lines = ascii_map.strip().split('\n')
        return '\n'.join(line for line in lines)
    
    def _format_agent_paths(self, agent_paths: Dict[str, Any]) -> str:
        """Format agent paths with proper TypeScript syntax"""
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
{chr(10).join(formatted_movements)}
      }}
    }}"""
            formatted_paths.append(agent_section)
        
        return ',\n'.join(formatted_paths)
    
    def save_level_file(self, level_config: Dict[str, Any], output_path: str):
        """Save TypeScript level file"""
        ts_code = self.export_typescript_level(level_config)
        with open(output_path, 'w') as f:
            f.write(ts_code)
    
    def print_level_summary(self, level_config: Dict[str, Any]):
        """Print summary of generated level"""
        print(f"Generated Level: {level_config['id']}")
        print(f"Goal: {level_config['goal']['description']}")
        print(f"Steps Remaining: {level_config['stepsRemaining']}")
        print(f"Agents: {len(level_config['agentPaths'])}")
        
        for agent_id, agent_data in level_config['agentPaths'].items():
            print(f"\nAgent {agent_id} paths:")
            for movement_key, movement_data in agent_data['movements'].items():
                path_length = len(movement_data['path'])
                print(f"  {movement_key} ({movement_data['type']}): {path_length} steps")


def main():
    """Example usage"""
    generator = LevelGenerator()
    
    # Example ASCII map (from your s0001.ts)
    test_map = """
WWrWeWbW
WW.W.W.W
gR.W.W.W
WW....M.
gB.WZWWW
""".strip()
    
    # Generate level
    level_config = generator.generate_level_config(
        level_id='s0002',
        ascii_map=test_map,
        steps_remaining=50,
        goal_type='B',
        goal_description='Find and obtain Treasure B'
    )
    
    # Print summary
    generator.print_level_summary(level_config)
    
    # Export TypeScript
    ts_code = generator.export_typescript_level(level_config)
    print("\n" + "="*50)
    print("TypeScript Export:")
    print("="*50)
    print(ts_code)


if __name__ == "__main__":
    main()