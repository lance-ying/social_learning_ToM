#!/usr/bin/env python3
"""
Convert exp4 pathing data from pathing_exp4_new_maps.json to movement paths compatible with multi-grid-game.
Transforms explicit actions (interact, pass, pickup) into movement sequences.
Uses agent2 and agent3 instead of 1 and 2.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add map_generator to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HAS_PATHFINDER = False


def parse_level_entities(ascii_map):
    """
    Parse ASCII map to extract positions of all interactive elements.
    Returns dictionary with entity positions and numbering.

    Note: Barriers are numbered with B barriers first, then R barriers,
    each in reading order (top to bottom, left to right).
    """
    entities = {
        "wizards": {},  # wizard1, wizard2... -> positions
        "barriers": {},  # door1, door2... -> positions (B, R barriers)
        "treasures": {},  # gem1, gem2, gem3 -> positions (g treasures)
        "start_pos": {},  # X -> Agent 2, Y -> Agent 3
    }

    lines = ascii_map.strip().split("\n")
    wizard_counter = 1
    gem_counter = 1

    # Collect barriers separately by type to number B before R
    b_barriers = []  # List of (x, y) positions for B barriers
    r_barriers = []  # List of (x, y) positions for R barriers

    for y, line in enumerate(lines):
        # Stop parsing if we hit an empty line or metadata (lines starting with letters followed by colon)
        if not line.strip() or (len(line) >= 2 and line[1] == ":"):
            break

        for x, char in enumerate(line):
            if char == "e":  # Empty wizard
                entities["wizards"][f"wizard{wizard_counter}"] = (x, y)
                wizard_counter += 1
            elif char == "b":  # Blue wizard
                entities["wizards"][f"wizard{wizard_counter}"] = (x, y)
                wizard_counter += 1
            elif char == "r":  # Red wizard
                entities["wizards"][f"wizard{wizard_counter}"] = (x, y)
                wizard_counter += 1
            elif char == "B":  # Blue barrier
                b_barriers.append((x, y))
            elif char == "R":  # Red barrier
                r_barriers.append((x, y))
            elif char in ["g", "G"]:  # Treasure (lowercase and uppercase)
                entities["treasures"][f"gem{gem_counter}"] = (x, y)
                gem_counter += 1
            elif char == "X":  # Agent 2 start (was Agent 1)
                entities["start_pos"]["agent2"] = (x, y)
            elif char == "Y":  # Agent 3 start (was Agent 2)
                entities["start_pos"]["agent3"] = (x, y)
            elif (
                char == "M"
            ):  # Player position (legacy, use as Agent 2 fallback if X/Y not found)
                if "agent2" not in entities["start_pos"]:
                    entities["start_pos"]["agent2"] = (x, y)

    # Number barriers: B barriers first, then R barriers (each in reading order)
    door_counter = 1
    for pos in b_barriers:
        entities["barriers"][f"door{door_counter}"] = pos
        door_counter += 1
    for pos in r_barriers:
        entities["barriers"][f"door{door_counter}"] = pos
        door_counter += 1

    return entities


def simple_pathfind(start, end, ascii_map):
    """
    Simple A* pathfinding for movement between positions.
    Returns list of movement directions: ['up', 'down', 'left', 'right']
    """
    if start == end:
        return []

    lines = ascii_map.strip().split("\n")

    # Filter out metadata lines (empty lines or lines with X:/Y: metadata)
    map_lines = []
    for line in lines:
        if not line.strip() or (len(line) >= 2 and line[1] == ":"):
            break
        map_lines.append(line)

    lines = map_lines
    height = len(lines)
    width = max(len(line) for line in lines) if height > 0 else 0

    def is_walkable(x, y):
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        if y >= len(lines) or x >= len(lines[y]):
            return False
        char = lines[y][x]
        return char not in ["W"]  # Only walls are blocking

    # A* implementation
    from heapq import heappop, heappush

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set:
        current = heappop(open_set)[1]

        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                prev = came_from[current]
                # Determine movement direction
                dx, dy = current[0] - prev[0], current[1] - prev[1]
                if dx == 1:
                    path.append("right")
                elif dx == -1:
                    path.append("left")
                elif dy == 1:
                    path.append("down")
                elif dy == -1:
                    path.append("up")
                current = prev
            return list(reversed(path))

        # Check neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not is_walkable(neighbor[0], neighbor[1]):
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    print(f"Warning: No path found from {start} to {end}")
    return []


def get_direction(from_x, from_y, to_x, to_y):
    """Compute direction from one position to another."""
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
        # Not adjacent, return None
        return None


def compute_path_to_target(from_x, from_y, to_x, to_y):
    """
    Compute a simple path (list of directions) from one position to another.
    Uses simple horizontal-then-vertical movement.
    """
    path = []
    current_x, current_y = from_x, from_y

    # Move horizontally first
    while current_x != to_x:
        if current_x < to_x:
            path.append("right")
            current_x += 1
        else:
            path.append("left")
            current_x -= 1

    # Then move vertically
    while current_y != to_y:
        if current_y < to_y:
            path.append("down")
            current_y += 1
        else:
            path.append("up")
            current_y -= 1

    return path


def convert_exp4_plan_to_movement_plan(exp4_plan, level_entities):
    """
    Convert exp4 action plan to movement path.
    New format: each action is a dict with 'action', 'x', 'y' keys.

    - Directional moves (up/down/left/right): add the direction, update tracked position
    - Interact: compute direction to wizard, add that direction
    - Pickup: compute direction to gem, add that direction
    - Pass: ignored (agent walks through door with regular movement)
    - FINAL_POSITION: compute path from current position to final position
    """
    if not exp4_plan:
        return []

    movement_path = []

    # Track agent's current position as we process movements
    # Initialize from first action's position
    if exp4_plan and isinstance(exp4_plan[0], dict):
        current_x = exp4_plan[0].get("x", 1) - 1
        current_y = exp4_plan[0].get("y", 1) - 1
    else:
        current_x, current_y = 0, 0

    for action_entry in exp4_plan:
        # Handle new format: action is a dict with 'action', 'x', 'y'
        if isinstance(action_entry, dict):
            action_str = action_entry.get("action", "")
            # Pathing data uses 1-indexed coordinates, convert to 0-indexed
            agent_x = action_entry.get("x", 1) - 1
            agent_y = action_entry.get("y", 1) - 1
        else:
            # Fallback for old format (string) - skip
            continue

        # Extract action type and target from string like "interact(wizard1)"
        if "(" in action_str:
            action_type = action_str.split("(")[0].lower()
            params_str = action_str.split("(")[1].rstrip(")")
        else:
            action_type = action_str.lower()
            params_str = ""

        # Handle different action types
        if action_type in ["up", "down", "left", "right"]:
            movement_path.append(action_type)
            # Update tracked position
            if action_type == "up":
                current_y -= 1
            elif action_type == "down":
                current_y += 1
            elif action_type == "left":
                current_x -= 1
            elif action_type == "right":
                current_x += 1

        elif action_type == "interact":
            # Find wizard and compute direction to it
            # params_str is like "wizard1"
            wizard_name = params_str.strip()
            if wizard_name in level_entities["wizards"]:
                wizard_x, wizard_y = level_entities["wizards"][wizard_name]
                direction = get_direction(agent_x, agent_y, wizard_x, wizard_y)
                if direction:
                    movement_path.append(direction)
                    # Note: position doesn't change for interact (agent stays in place)
            else:
                print(f"    Warning: Wizard {wizard_name} not found in level entities")

        elif action_type == "pickup":
            # Find gem and compute direction to it
            # params_str is like "agent2, gem1" - extract gem name
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
                    # Update position - agent moves onto gem
                    current_x, current_y = gem_x, gem_y
            else:
                print(f"    Warning: Gem {gem_name} not found in level entities")

        elif action_type == "final_position":
            # Compute path from current tracked position to the final position
            final_x, final_y = agent_x, agent_y
            if current_x != final_x or current_y != final_y:
                final_path = compute_path_to_target(current_x, current_y, final_x, final_y)
                movement_path.extend(final_path)
                current_x, current_y = final_x, final_y

        # pass actions are ignored - agent moves through door with regular directional movements

    return movement_path


def map_exp4_agent_to_current_agent(exp4_agent_name):
    """
    Map exp4 agent names to current system agent numbers.
    exp4 agent2 -> current Agent 2 (X character)
    exp4 agent3 -> current Agent 3 (Y character)
    """
    if exp4_agent_name == "agent2":
        return 2  # Agent 2 in current system
    elif exp4_agent_name == "agent3":
        return 3  # Agent 3 in current system
    else:
        return (
            int(exp4_agent_name.replace("agent", ""))
            if exp4_agent_name.startswith("agent")
            else 2
        )


def load_existing_level_file(level_id):
    """Load existing level configuration to update paths"""
    level_file = f"../../src/data/levels/exp4/{level_id}.ts"

    if not os.path.exists(level_file):
        print(f"Warning: Level file {level_file} not found")
        return None

    with open(level_file, "r") as f:
        content = f.read()

    return content


def update_level_paths(
    level_content, agent_num, movement_index, movement_path, goal, path_type
):
    """
    Update movement path in level content for a specific agent.
    movement_index: 1 for experienced1, 2 for experienced2, 3 for experienced3
    """
    # First, find the agent section by locating the agent number in agentPaths
    # Pattern: agentPaths: { ... agent_num: { movements: { ... } } ... }
    # We need to be agent-aware to avoid updating the wrong agent's path

    # Find the start of the agent section
    agent_section_pattern = rf"{agent_num}:\s*\{{\s*movements:\s*\{{"
    agent_section_match = re.search(
        agent_section_pattern, level_content, re.MULTILINE | re.DOTALL
    )

    if not agent_section_match:
        print(f"Warning: Could not find agent {agent_num} section in level content")
        return level_content

    # Get the position where agent section starts
    agent_start_pos = agent_section_match.end()

    # Find the end of this agent's movements section (next closing brace at appropriate level)
    # Look for the pattern that closes the movements object
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

    # Extract just this agent's section
    agent_section = level_content[agent_start_pos:agent_end_pos]

    # Now look for the experienced{movement_index} pattern within THIS agent's section only
    pattern = rf"(experienced{movement_index}:\s*\{{\s*path:\s*\[)[^\]]*(\]\s*,\s*goal:\s*\d+\s*,\s*type:\s*\'[^\']*\')"
    match = re.search(pattern, agent_section, re.MULTILINE | re.DOTALL)

    if match:
        # Replace the path array content while preserving goal and type
        path_str = ", ".join([f'"{move}"' for move in movement_path])
        new_section = f"{match.group(1)}{path_str}{match.group(2)}"

        # Replace in the agent section
        updated_agent_section = agent_section.replace(match.group(0), new_section, 1)

        # Replace the agent section in the full content
        level_content = (
            level_content[:agent_start_pos]
            + updated_agent_section
            + level_content[agent_end_pos:]
        )

        print(
            f"    Agent {agent_num} experienced{movement_index}: Updated with {len(movement_path)} movements"
        )
    else:
        print(
            f"Warning: Could not find experienced{movement_index} pattern for Agent {agent_num}"
        )

    return level_content


def main():
    """Main conversion function"""

    # Load pathing data - using pathing_exp4_new_maps.json
    pathing_file = "../../extracted_ascii_maps/paths_exp4/pathing_exp4_new_maps.json"
    if not os.path.exists(pathing_file):
        print(f"Error: Pathing file {pathing_file} not found")
        return

    with open(pathing_file, "r") as f:
        pathing_data = json.load(f)

    # Load ASCII maps directory - using problem_exp4_true
    ascii_dir = "../../extracted_ascii_maps/problem_exp4_true"

    processed = 0
    failed = 0

    for level_id, scenarios in pathing_data.items():
        print(f"\nProcessing level: {level_id}")

        # Load ASCII map
        ascii_file = os.path.join(ascii_dir, f"{level_id}.txt")
        if not os.path.exists(ascii_file):
            print(f"  Warning: ASCII file {ascii_file} not found")
            failed += 1
            continue

        with open(ascii_file, "r") as f:
            ascii_map = f.read()

        # Parse entities from ASCII map
        level_entities = parse_level_entities(ascii_map)
        print(
            f"  Found {len(level_entities['wizards'])} wizards, {len(level_entities['barriers'])} barriers, {len(level_entities['treasures'])} treasures"
        )

        # Load existing level file
        level_content = load_existing_level_file(level_id)
        if not level_content:
            failed += 1
            continue

        # Process each scenario and agent
        scenario_counter = 1
        for scenario_name, agents in scenarios.items():
            print(f"  Processing {scenario_name}:")

            for exp4_agent_name, agent_data in agents.items():
                # Map exp4 agent to current system agent (now agent2 and agent3)
                current_agent_num = map_exp4_agent_to_current_agent(exp4_agent_name)

                # Convert exp4 plan to movement path
                exp4_plan = agent_data.get("plan", [])
                movement_path = convert_exp4_plan_to_movement_plan(exp4_plan, level_entities)

                print(
                    f"    {exp4_agent_name} -> Agent {current_agent_num}: {len(movement_path)} movements"
                )

                # Map to movement index (experienced1, experienced2, experienced3)
                # Use scenario mapping: scenario1 -> experienced1, scenario2 -> experienced2, scenario3 -> experienced3
                movement_index = int(scenario_name.replace("scenario", ""))

                # Update level content with new path
                goal = agent_data.get("gem", 1)  # Use gem number as goal
                path_type = "actual" if agent_data.get("type") == "actual" else "naive"
                level_content = update_level_paths(
                    level_content,
                    current_agent_num,
                    movement_index,
                    movement_path,
                    goal,
                    path_type,
                )

            scenario_counter += 1

        # Write updated level file
        output_file = f"../../src/data/levels/exp4/{level_id}.ts"
        with open(output_file, "w") as f:
            f.write(level_content)

        print(f"  ✓ Updated: {output_file}")
        processed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    print(f"\n✓ Conversion complete! Pathing data integrated into exp4 level files.")
    print("Note: Uses agent2 and agent3 instead of 1 and 2.")


if __name__ == "__main__":
    main()
