#!/usr/bin/env python3
"""
Generate optimal paths for all levels in the 'exp3_true' directory.
Generates paths for agents X (agent2) and Y (agent3) with their respective goals.
3-agent levels: M (player), X (agent2), Y (agent3)
"""

import os
import sys
import re
from typing import List, Dict, Tuple
from pathlib import Path

# Add pathfinding directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pathfinding'))

from pathfinder_for_z_agent import ZAgentPathfinder


def extract_movements_section(content: str, agent_id: str) -> str:
    """
    Extract the movements section for a given agent using balanced brace matching.
    """
    # Find the agent section
    agent_pattern = f'{agent_id}:\\s*{{.*?movements:\\s*{{'
    match = re.search(agent_pattern, content, re.DOTALL)
    if not match:
        return ""

    # Start from the opening brace of movements
    start_pos = match.end() - 1  # Position of the opening brace

    # Count braces to find the matching closing brace
    brace_count = 1
    i = start_pos + 1
    while i < len(content) and brace_count > 0:
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
        i += 1

    # Extract the content between the braces
    return content[start_pos + 1:i - 1]


def parse_typescript_level_with_three_agents(file_path: str) -> Tuple[str, int, int, int, int]:
    """
    Parse TypeScript level file and extract goals for both agents X and Y.
    Returns: (ascii_map, agent2_goal1, agent2_goal2, agent3_goal1, agent3_goal2)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract ASCII map
        map_match = re.search(r'asciiMap:\s*`\s*\n(.*?)\n\s*`', content, re.DOTALL)
        ascii_map = map_match.group(1) if map_match else ""

        # Extract agent 2 (X) movements section
        agent2_content = extract_movements_section(content, '2')
        if agent2_content:
            a2_exp1_match = re.search(r'experienced1:\s*\{\s*path:.*?goal:\s*(\d+)', agent2_content, re.DOTALL)
            a2_exp2_match = re.search(r'experienced2:\s*\{\s*path:.*?goal:\s*(\d+)', agent2_content, re.DOTALL)
            agent2_goal1 = int(a2_exp1_match.group(1)) if a2_exp1_match else 1
            agent2_goal2 = int(a2_exp2_match.group(1)) if a2_exp2_match else 2
        else:
            agent2_goal1, agent2_goal2 = 1, 2

        # Extract agent 3 (Y) movements section
        agent3_content = extract_movements_section(content, '3')
        if agent3_content:
            a3_exp1_match = re.search(r'experienced1:\s*\{\s*path:.*?goal:\s*(\d+)', agent3_content, re.DOTALL)
            a3_exp2_match = re.search(r'experienced2:\s*\{\s*path:.*?goal:\s*(\d+)', agent3_content, re.DOTALL)
            agent3_goal1 = int(a3_exp1_match.group(1)) if a3_exp1_match else 1
            agent3_goal2 = int(a3_exp2_match.group(1)) if a3_exp2_match else 2
        else:
            agent3_goal1, agent3_goal2 = 1, 2

        return ascii_map, agent2_goal1, agent2_goal2, agent3_goal1, agent3_goal2

    except Exception as e:
        print(f"  ✗ Error parsing file {file_path}: {e}")
        return "", 1, 2, 1, 2


def map_goal_number_to_type(goal_num: int) -> str:
    """Map goal number to treasure type (1=A, 2=B, 3=C, 4=D)"""
    mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    return mapping.get(goal_num, 'A')


def update_typescript_paths_for_three_agents(file_path: str,
                                              agent2_exp1_path: List[str], agent2_exp2_path: List[str],
                                              agent3_exp1_path: List[str], agent3_exp2_path: List[str]) -> bool:
    """
    Update paths for agents X (agent2) and Y (agent3) in the TypeScript file.
    Agent 2 (X): experienced1/2/3/4
    Agent 3 (Y): experienced1/2/3/4
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Format paths for agent 2 (X)
        a2_exp1_str = ', '.join(f'"{step}"' for step in agent2_exp1_path)
        a2_exp2_str = ', '.join(f'"{step}"' for step in agent2_exp2_path)

        # Format paths for agent 3 (Y)
        a3_exp1_str = ', '.join(f'"{step}"' for step in agent3_exp1_path)
        a3_exp2_str = ', '.join(f'"{step}"' for step in agent3_exp2_path)

        # Find agent 2 section
        agent2_pattern = r'(2:\s*\{.*?movements:\s*\{)(.*?)(\}\s*\})'
        agent2_match = re.search(agent2_pattern, content, re.DOTALL)

        if agent2_match:
            agent2_movements = agent2_match.group(2)
            # Update agent 2 paths (experienced1 and experienced3 get path1, experienced2 and experienced4 get path2)
            agent2_movements = re.sub(r'(experienced1:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp1_str + r'\2', agent2_movements)
            agent2_movements = re.sub(r'(experienced2:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp2_str + r'\2', agent2_movements)
            agent2_movements = re.sub(r'(experienced3:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp1_str + r'\2', agent2_movements)
            agent2_movements = re.sub(r'(experienced4:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp2_str + r'\2', agent2_movements)
            content = content[:agent2_match.start(2)] + agent2_movements + content[agent2_match.end(2):]

        # Find agent 3 section
        agent3_pattern = r'(3:\s*\{.*?movements:\s*\{)(.*?)(\}\s*\})'
        agent3_match = re.search(agent3_pattern, content, re.DOTALL)

        if agent3_match:
            agent3_movements = agent3_match.group(2)
            # Update agent 3 paths (experienced1 and experienced3 get path1, experienced2 and experienced4 get path2)
            agent3_movements = re.sub(r'(experienced1:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a3_exp1_str + r'\2', agent3_movements)
            agent3_movements = re.sub(r'(experienced2:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a3_exp2_str + r'\2', agent3_movements)
            agent3_movements = re.sub(r'(experienced3:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a3_exp1_str + r'\2', agent3_movements)
            agent3_movements = re.sub(r'(experienced4:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a3_exp2_str + r'\2', agent3_movements)
            content = content[:agent3_match.start(2)] + agent3_movements + content[agent3_match.end(2):]

        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"  ✗ Error updating file {file_path}: {e}")
        return False


def process_exp3_true_levels():
    """Process all levels in the 'exp3_true' directory and generate paths for agents X and Y"""

    # Define paths
    script_dir = Path(__file__).parent
    levels_path = script_dir.parent.parent / 'src' / 'data' / 'levels' / 'exp3_true'

    # Get all .ts files in the exp3_true directory
    if not levels_path.exists():
        print(f"Error: Directory not found: {levels_path}")
        return

    all_files = [f for f in os.listdir(levels_path)
                 if f.endswith('.ts') and not f.startswith('_')]
    level_names = sorted([f[:-3] for f in all_files])

    print("=" * 90)
    print(f"Processing {len(level_names)} levels in 'exp3_true' directory")
    print("Generating optimal paths for Agent X (agent2) and Agent Y (agent3)")
    print("=" * 90)

    results = []
    failed = []

    for level_name in level_names:
        file_path = levels_path / f'{level_name}.ts'

        print(f"\n{level_name}:")
        print("-" * 70)

        # Parse level and extract goals for agents X and Y
        ascii_map, a2_goal1, a2_goal2, a3_goal1, a3_goal2 = parse_typescript_level_with_three_agents(str(file_path))

        if not ascii_map:
            print(f"  ✗ Could not parse ASCII map")
            failed.append(level_name)
            continue

        # Convert goals to types
        a2_goal1_type = map_goal_number_to_type(a2_goal1)
        a2_goal2_type = map_goal_number_to_type(a2_goal2)
        a3_goal1_type = map_goal_number_to_type(a3_goal1)
        a3_goal2_type = map_goal_number_to_type(a3_goal2)

        print(f"  Agent 2 (X): Goal1={a2_goal1}→{a2_goal1_type}, Goal2={a2_goal2}→{a2_goal2_type}")
        print(f"  Agent 3 (Y): Goal1={a3_goal1}→{a3_goal1_type}, Goal2={a3_goal2}→{a3_goal2_type}")

        # Generate paths for Agent 2 (X)
        # Replace Y with . and X with Z to calculate X agent paths as if it were Z
        ascii_map_for_x = ascii_map.replace('Y', '.').replace('X', 'Z')

        pathfinder_a2_g1 = ZAgentPathfinder(ascii_map_for_x, a2_goal1_type)
        a2_exp1_path = pathfinder_a2_g1.find_efficient_path()

        if not a2_exp1_path:
            print(f"  ✗ No path found for Agent 2 (X), Goal 1")
            failed.append(level_name)
            continue

        pathfinder_a2_g2 = ZAgentPathfinder(ascii_map_for_x, a2_goal2_type)
        a2_exp2_path = pathfinder_a2_g2.find_efficient_path()

        if not a2_exp2_path:
            print(f"  ✗ No path found for Agent 2 (X), Goal 2")
            failed.append(level_name)
            continue

        print(f"  ✓ Agent 2 (X) paths: {len(a2_exp1_path)} / {len(a2_exp2_path)} steps")

        # Generate paths for Agent 3 (Y)
        # Replace X with . and Y with Z to calculate Y agent paths as if it were Z
        ascii_map_for_y = ascii_map.replace('X', '.').replace('Y', 'Z')

        pathfinder_a3_g1 = ZAgentPathfinder(ascii_map_for_y, a3_goal1_type)
        a3_exp1_path = pathfinder_a3_g1.find_efficient_path()

        if not a3_exp1_path:
            print(f"  ✗ No path found for Agent 3 (Y), Goal 1")
            failed.append(level_name)
            continue

        pathfinder_a3_g2 = ZAgentPathfinder(ascii_map_for_y, a3_goal2_type)
        a3_exp2_path = pathfinder_a3_g2.find_efficient_path()

        if not a3_exp2_path:
            print(f"  ✗ No path found for Agent 3 (Y), Goal 2")
            failed.append(level_name)
            continue

        print(f"  ✓ Agent 3 (Y) paths: {len(a3_exp1_path)} / {len(a3_exp2_path)} steps")

        # Update the TypeScript file with paths for both agents
        success = update_typescript_paths_for_three_agents(
            str(file_path), a2_exp1_path, a2_exp2_path, a3_exp1_path, a3_exp2_path
        )

        if success:
            print(f"  ✓ Updated {level_name}.ts with paths for agents X and Y")
            results.append({
                'name': level_name,
                'a2_exp1_len': len(a2_exp1_path),
                'a2_exp2_len': len(a2_exp2_path),
                'a3_exp1_len': len(a3_exp1_path),
                'a3_exp2_len': len(a3_exp2_path),
            })
        else:
            failed.append(level_name)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if results:
        print(f"\n✓ Successfully processed {len(results)} levels:\n")
        print(f"{'Level':<15} {'AgentX E1':<12} {'AgentX E2':<12} {'AgentY E1':<12} {'AgentY E2':<12}")
        print("-" * 90)
        for result in results:
            print(f"{result['name']:<15} {result['a2_exp1_len']:<12} {result['a2_exp2_len']:<12} "
                  f"{result['a3_exp1_len']:<12} {result['a3_exp2_len']:<12}")

    if failed:
        print(f"\n✗ Failed to process {len(failed)} levels:")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "=" * 90)
    print("✓ Path generation complete!")
    print("All levels have been updated with optimal paths for agents X (agent2) and Y (agent3).")
    print("=" * 90)


def main():
    process_exp3_true_levels()


if __name__ == "__main__":
    main()


