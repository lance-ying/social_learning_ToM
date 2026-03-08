#!/usr/bin/env python3
"""
Generate optimal paths for all levels in the 'exp3' directory.
Generates paths for both agents (Z and O) with their respective goals.
"""

import os
import sys
import re
from typing import List, Dict, Tuple
from pathlib import Path

# Import from pathfinder_for_z_agent
from pathfinder_for_z_agent import ZAgentPathfinder, parse_typescript_level


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


def parse_typescript_level_with_two_agents(file_path: str) -> Tuple[str, int, int, int, int]:
    """
    Parse TypeScript level file and extract goals for both agents.
    Returns: (ascii_map, agent1_goal1, agent1_goal2, agent2_goal1, agent2_goal2)
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract ASCII map
        map_match = re.search(r'asciiMap:\s*`\s*\n(.*?)\n\s*`', content, re.DOTALL)
        ascii_map = map_match.group(1) if map_match else ""

        # Extract agent 1 (Z) movements section
        agent1_content = extract_movements_section(content, '1')
        if agent1_content:
            a1_exp1_match = re.search(r'experienced1:\s*\{\s*path:.*?goal:\s*(\d+)', agent1_content, re.DOTALL)
            a1_exp2_match = re.search(r'experienced2:\s*\{\s*path:.*?goal:\s*(\d+)', agent1_content, re.DOTALL)
            agent1_goal1 = int(a1_exp1_match.group(1)) if a1_exp1_match else 1
            agent1_goal2 = int(a1_exp2_match.group(1)) if a1_exp2_match else 2
        else:
            agent1_goal1, agent1_goal2 = 1, 2

        # Extract agent 2 (O) movements section
        agent2_content = extract_movements_section(content, '2')
        if agent2_content:
            a2_exp1_match = re.search(r'experienced1:\s*\{\s*path:.*?goal:\s*(\d+)', agent2_content, re.DOTALL)
            a2_exp2_match = re.search(r'experienced2:\s*\{\s*path:.*?goal:\s*(\d+)', agent2_content, re.DOTALL)
            agent2_goal1 = int(a2_exp1_match.group(1)) if a2_exp1_match else 1
            agent2_goal2 = int(a2_exp2_match.group(1)) if a2_exp2_match else 2
        else:
            agent2_goal1, agent2_goal2 = 1, 2

        return ascii_map, agent1_goal1, agent1_goal2, agent2_goal1, agent2_goal2

    except Exception as e:
        print(f"  ✗ Error parsing file {file_path}: {e}")
        return "", 1, 2, 1, 2


def map_goal_number_to_type(goal_num: int) -> str:
    """Map goal number to treasure type (1=A, 2=B, 3=C)"""
    mapping = {1: 'A', 2: 'B', 3: 'C'}
    return mapping.get(goal_num, 'A')


def update_typescript_paths_for_two_agents(file_path: str,
                                           agent1_exp1_path: List[str], agent1_exp2_path: List[str],
                                           agent2_exp1_path: List[str], agent2_exp2_path: List[str]) -> bool:
    """
    Update paths for both agents in the TypeScript file.
    Agent 1: experienced1/2/3/4
    Agent 2: experienced1/2/3/4
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Format paths for agent 1
        a1_exp1_str = ', '.join(f'"{step}"' for step in agent1_exp1_path)
        a1_exp2_str = ', '.join(f'"{step}"' for step in agent1_exp2_path)

        # Format paths for agent 2
        a2_exp1_str = ', '.join(f'"{step}"' for step in agent2_exp1_path)
        a2_exp2_str = ', '.join(f'"{step}"' for step in agent2_exp2_path)

        # Split content into agent 1 and agent 2 sections
        # Find agent 1 section
        agent1_pattern = r'(1:\s*\{.*?movements:\s*\{)(.*?)(\}\s*\})'
        agent1_match = re.search(agent1_pattern, content, re.DOTALL)

        if agent1_match:
            agent1_movements = agent1_match.group(2)
            # Update agent 1 paths
            agent1_movements = re.sub(r'(experienced1:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a1_exp1_str + r'\2', agent1_movements)
            agent1_movements = re.sub(r'(experienced2:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a1_exp2_str + r'\2', agent1_movements)
            agent1_movements = re.sub(r'(experienced3:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a1_exp1_str + r'\2', agent1_movements)
            agent1_movements = re.sub(r'(experienced4:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a1_exp2_str + r'\2', agent1_movements)
            content = content[:agent1_match.start(2)] + agent1_movements + content[agent1_match.end(2):]

        # Find agent 2 section
        agent2_pattern = r'(2:\s*\{.*?movements:\s*\{)(.*?)(\}\s*\})'
        agent2_match = re.search(agent2_pattern, content, re.DOTALL)

        if agent2_match:
            agent2_movements = agent2_match.group(2)
            # Update agent 2 paths
            agent2_movements = re.sub(r'(experienced1:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp1_str + r'\2', agent2_movements)
            agent2_movements = re.sub(r'(experienced2:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp2_str + r'\2', agent2_movements)
            agent2_movements = re.sub(r'(experienced3:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp1_str + r'\2', agent2_movements)
            agent2_movements = re.sub(r'(experienced4:\s*\{\s*path:\s*\[)[^\]]*(\])',
                                     r'\1' + a2_exp2_str + r'\2', agent2_movements)
            content = content[:agent2_match.start(2)] + agent2_movements + content[agent2_match.end(2):]

        # Write updated content
        with open(file_path, 'w') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"  ✗ Error updating file {file_path}: {e}")
        return False


def process_new_levels():
    """Process all levels in the 'exp3' directory and generate paths for two agents"""

    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    levels_path = os.path.join(base_path, 'src', 'data', 'levels', 'exp3')

    # Get all .ts files in the exp3 directory
    if not os.path.exists(levels_path):
        print(f"Error: Directory not found: {levels_path}")
        return

    all_files = [f for f in os.listdir(levels_path)
                 if f.endswith('.ts') and not f.startswith('_')]
    level_names = sorted([f[:-3] for f in all_files])

    print("=" * 90)
    print(f"Processing {len(level_names)} levels in 'exp3' directory")
    print("Generating optimal paths for Agent 1 (Z) and Agent 2 (O)")
    print("=" * 90)

    results = []
    failed = []

    for level_name in level_names:
        file_path = os.path.join(levels_path, f'{level_name}.ts')

        print(f"\n{level_name}:")
        print("-" * 70)

        # Parse level and extract goals for both agents
        ascii_map, a1_goal1, a1_goal2, a2_goal1, a2_goal2 = parse_typescript_level_with_two_agents(file_path)

        if not ascii_map:
            print(f"  ✗ Could not parse ASCII map")
            failed.append(level_name)
            continue

        # Convert goals to types
        a1_goal1_type = map_goal_number_to_type(a1_goal1)
        a1_goal2_type = map_goal_number_to_type(a1_goal2)
        a2_goal1_type = map_goal_number_to_type(a2_goal1)
        a2_goal2_type = map_goal_number_to_type(a2_goal2)

        print(f"  Agent 1 (Z): Goal1={a1_goal1}→{a1_goal1_type}, Goal2={a1_goal2}→{a1_goal2_type}")
        print(f"  Agent 2 (O): Goal1={a2_goal1}→{a2_goal1_type}, Goal2={a2_goal2}→{a2_goal2_type}")

        # Generate paths for Agent 1 (Z)
        # Replace O with Z temporarily to calculate Z agent paths
        ascii_map_for_z = ascii_map.replace('O', '.')

        pathfinder_a1_g1 = ZAgentPathfinder(ascii_map_for_z, a1_goal1_type)
        a1_exp1_path = pathfinder_a1_g1.find_efficient_path()

        if not a1_exp1_path:
            print(f"  ✗ No path found for Agent 1, Goal 1")
            failed.append(level_name)
            continue

        pathfinder_a1_g2 = ZAgentPathfinder(ascii_map_for_z, a1_goal2_type)
        a1_exp2_path = pathfinder_a1_g2.find_efficient_path()

        if not a1_exp2_path:
            print(f"  ✗ No path found for Agent 1, Goal 2")
            failed.append(level_name)
            continue

        print(f"  ✓ Agent 1 paths: {len(a1_exp1_path)} / {len(a1_exp2_path)} steps")

        # Generate paths for Agent 2 (O)
        # Replace Z with . and O with Z to calculate O agent paths as if it were Z
        ascii_map_for_o = ascii_map.replace('Z', '.').replace('O', 'Z')

        pathfinder_a2_g1 = ZAgentPathfinder(ascii_map_for_o, a2_goal1_type)
        a2_exp1_path = pathfinder_a2_g1.find_efficient_path()

        if not a2_exp1_path:
            print(f"  ✗ No path found for Agent 2, Goal 1")
            failed.append(level_name)
            continue

        pathfinder_a2_g2 = ZAgentPathfinder(ascii_map_for_o, a2_goal2_type)
        a2_exp2_path = pathfinder_a2_g2.find_efficient_path()

        if not a2_exp2_path:
            print(f"  ✗ No path found for Agent 2, Goal 2")
            failed.append(level_name)
            continue

        print(f"  ✓ Agent 2 paths: {len(a2_exp1_path)} / {len(a2_exp2_path)} steps")

        # Update the TypeScript file with paths for both agents
        success = update_typescript_paths_for_two_agents(
            file_path, a1_exp1_path, a1_exp2_path, a2_exp1_path, a2_exp2_path
        )

        if success:
            print(f"  ✓ Updated {level_name}.ts with paths for both agents")
            results.append({
                'name': level_name,
                'a1_exp1_len': len(a1_exp1_path),
                'a1_exp2_len': len(a1_exp2_path),
                'a2_exp1_len': len(a2_exp1_path),
                'a2_exp2_len': len(a2_exp2_path),
            })
        else:
            failed.append(level_name)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if results:
        print(f"\n✓ Successfully processed {len(results)} levels:\n")
        print(f"{'Level':<15} {'Agent1 E1':<12} {'Agent1 E2':<12} {'Agent2 E1':<12} {'Agent2 E2':<12}")
        print("-" * 90)
        for result in results:
            print(f"{result['name']:<15} {result['a1_exp1_len']:<12} {result['a1_exp2_len']:<12} "
                  f"{result['a2_exp1_len']:<12} {result['a2_exp2_len']:<12}")

    if failed:
        print(f"\n✗ Failed to process {len(failed)} levels:")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "=" * 90)
    print("✓ Path generation complete!")
    print("All levels have been updated with optimal paths for both agents.")
    print("=" * 90)


def main():
    process_new_levels()


if __name__ == "__main__":
    main()
