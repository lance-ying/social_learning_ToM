using PDDL
using SymbolicPlanners
using JSON
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))

# Helper function to plan to an adjacent position of a wizard (never on top)
# This matches the exact logic from inference_multi_exp4.jl
function plan_to_wizard_location(
    domain::Domain, state::State, wizard_loc::Tuple{Int,Int},
    agent_name::Symbol, planner
)
    agent_loc = get_obj_loc(state, Const(agent_name))

    # Check if already adjacent (Manhattan distance = 1)
    agent_adjacent = (abs(agent_loc[1] - wizard_loc[1]) + abs(agent_loc[2] - wizard_loc[2]) == 1)

    if agent_adjacent
        return Term[]  # Already adjacent, no movement needed
    end

    # Try to plan to each adjacent position, return on first success (fast)
    for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        adj_pos = (wizard_loc[1] + dx, wizard_loc[2] + dy)
        adj_goal = PDDL.parse_pddl("(and (= (xloc $agent_name) $(adj_pos[1])) (= (yloc $agent_name) $(adj_pos[2])))")
        try
            plan = collect(planner(domain, state, adj_goal))
            if !isempty(plan)
                return plan  # Return immediately on first success
            end
        catch
            continue
        end
    end

    return Term[]
end

function generate_naive_plan_if_needed(
    domain::Domain, state::State, goal::Any, blue_wizards::Vector,
    goal_type::String, agent_name::Symbol
)
    planner_optimal = AStarPlanner(GoalManhattan())

    # Check if optimal path requires blue wizards
    plan_optimal = collect(planner_optimal(domain, state, goal))
    needs_wizards = any(x -> x.name == :interact && x.args[end] in blue_wizards, plan_optimal)

    if goal_type == "naive" && needs_wizards && !isempty(blue_wizards)
        # Find blue key
        blue_keys = [k for k in PDDL.get_objects(state, :key) if state[pddl"(iscolor $k blue)"]]
        if isempty(blue_keys)
            return plan_optimal
        end
        blue_key = blue_keys[1]

        # Visit blue wizards in order of distance until we get the key
        current_state = copy(state)
        full_naive_plan = Term[]
        visited_wizards = Set()

        while !current_state[pddl"(has $agent_name $blue_key)"] && length(visited_wizards) < length(blue_wizards)
            # Find closest unvisited blue wizard
            agent_loc = get_obj_loc(current_state, Const(agent_name))
            closest_wizard = nothing
            closest_wizard_loc = nothing
            min_dist = Inf

            for wizard in blue_wizards
                if wizard in visited_wizards
                    continue
                end
                wizard_loc = get_obj_loc(current_state, wizard)
                dist = sum(abs.(agent_loc .- wizard_loc))
                if dist < min_dist
                    min_dist = dist
                    closest_wizard = wizard
                    closest_wizard_loc = wizard_loc
                end
            end

            if closest_wizard === nothing
                break  # No more wizards to visit
            end

            # Plan to wizard and interact
            plan_to_wizard = plan_to_wizard_location(domain, current_state, closest_wizard_loc, agent_name, planner_optimal)
            append!(full_naive_plan, plan_to_wizard)

            # Execute plan to wizard
            for action in plan_to_wizard
                current_state = PDDL.execute(domain, current_state, action)
            end

            # Interact with wizard
            interact_action = PDDL.parse_pddl("(interact $agent_name $closest_wizard)")
            push!(full_naive_plan, interact_action)
            current_state = PDDL.execute(domain, current_state, interact_action)

            push!(visited_wizards, closest_wizard)
        end

        # If we still don't have the key, fall back to optimal
        if !current_state[pddl"(has $agent_name $blue_key)"]
            return plan_optimal
        end

        # Plan to goal from current location
        plan_to_goal = collect(planner_optimal(domain, current_state, goal))
        append!(full_naive_plan, plan_to_goal)

        return full_naive_plan
    else
        # Use optimal planning
        return plan_optimal
    end
end

function filter_ascii_agents(ascii_content::String, keep_agent::Symbol)
    agent_chars = Dict(:agent1 => 'M', :agent2 => 'X', :agent3 => 'Y')
    filtered = ascii_content
    for (agent_sym, char) in agent_chars
        if agent_sym != keep_agent
            filtered = replace(filtered, char => '.')
        end
    end
    return filtered
end

# Convert action to string representation
function action_to_string(action::Term)
    if action.name == :move
        dir = action.args[2]
        return "move($dir)"
    elseif action.name == :interact
        wizard = action.args[2]
        return "interact($wizard)"
    else
        return string(action)
    end
end

# Simulate plan execution to track coordinates for each action
function get_plan_with_coordinates(domain::Domain, state::State, plan::Vector{Term}, agent_name::Symbol)
    plan_with_coords = []
    current_state = copy(state)
    current_loc = get_obj_loc(current_state, Const(agent_name))

    for action in plan
        # Get action string
        action_str = action_to_string(action)

        # Add current location to action info (location BEFORE action execution)
        action_info = Dict(
            "action" => action_str,
            "x" => current_loc[1],
            "y" => current_loc[2]
        )

        push!(plan_with_coords, action_info)

        # Execute action to update state for next iteration
        current_state = PDDL.execute(domain, current_state, action)

        # Always update agent location after action execution
        # Move actions will change location, interact actions won't
        current_loc = get_obj_loc(current_state, Const(agent_name))
    end

    # Add final position entry showing where agent ends up after completing the plan
    final_entry = Dict(
        "action" => "FINAL_POSITION",
        "x" => current_loc[1],
        "y" => current_loc[2]
    )
    push!(plan_with_coords, final_entry)

    return plan_with_coords
end

# Main script
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_exp4_new")
OUTPUT_DIR = joinpath(@__DIR__, "experiment_outputs")
mkpath(OUTPUT_DIR)

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

pathing_dict = Dict()

domain = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))

for (map_id, agent_goals) in metadata
    println("Processing map: $map_id")
    pathing_dict[map_id] = Dict()

    # Load the ASCII content for filtering
    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    for scenario in 1:2
        scenario_key = "scenario$(scenario)"
        pathing_dict[map_id][scenario_key] = Dict()

        # Get agent goals for this scenario
        agent2_goal_info = agent_goals["agent2"][scenario]
        agent3_goal_info = agent_goals["agent3"][scenario]

        agent2_gem = agent2_goal_info["gem"]
        agent2_type = agent2_goal_info["type"]
        agent3_gem = agent3_goal_info["gem"]
        agent3_type = agent3_goal_info["type"]

        println("  Scenario $scenario: agent2 -> gem$(agent2_gem) ($(agent2_type)), agent3 -> gem$(agent3_gem) ($(agent3_type))")

        # Process agent2
        domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
        temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
        if !isfile(temp_path_agent2)
            filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
            write(temp_path_agent2, filtered_ascii_agent2)
        end
        problem_agent2 = load_ascii_problem(temp_path_agent2)
        state_agent2 = initstate(domain_agent2, problem_agent2)
        domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)

        goal_agent2 = PDDL.parse_pddl("(has agent2 gem$(agent2_gem))")
        blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]

        plan_agent2 = generate_naive_plan_if_needed(
            domain_agent2, state_agent2, goal_agent2, blue_wizards_agent2, agent2_type, :agent2
        )
        plan_agent2_coords = get_plan_with_coordinates(domain_agent2, state_agent2, plan_agent2, :agent2)

        pathing_dict[map_id][scenario_key]["agent2"] = Dict(
            "gem" => agent2_gem,
            "type" => agent2_type,
            "plan_length" => length(plan_agent2),
            "plan" => plan_agent2_coords,
            "needs_wizards" => any(x -> x.name == :interact && x.args[end] in blue_wizards_agent2, plan_agent2)
        )

        # Process agent3
        domain_agent3 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
        temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
        if !isfile(temp_path_agent3)
            filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
            write(temp_path_agent3, filtered_ascii_agent3)
        end
        problem_agent3 = load_ascii_problem(temp_path_agent3)
        state_agent3 = initstate(domain_agent3, problem_agent3)
        domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)

        goal_agent3 = PDDL.parse_pddl("(has agent3 gem$(agent3_gem))")
        blue_wizards_agent3 = [w for w in PDDL.get_objects(state_agent3, :wizard) if state_agent3[pddl"(iscolor $w blue)"]]

        plan_agent3 = generate_naive_plan_if_needed(
            domain_agent3, state_agent3, goal_agent3, blue_wizards_agent3, agent3_type, :agent3
        )
        plan_agent3_coords = get_plan_with_coordinates(domain_agent3, state_agent3, plan_agent3, :agent3)

        pathing_dict[map_id][scenario_key]["agent3"] = Dict(
            "gem" => agent3_gem,
            "type" => agent3_type,
            "plan_length" => length(plan_agent3),
            "plan" => plan_agent3_coords,
            "needs_wizards" => any(x -> x.name == :interact && x.args[end] in blue_wizards_agent3, plan_agent3)
        )
    end
end

# Save to JSON
output_path = joinpath(OUTPUT_DIR, "pathing_exp4_new_maps_again.json")
open(output_path, "w") do io
    JSON.print(io, pathing_dict, 4)
end

println("\n=== Pathing Generation Complete ===")
println("Saved to: $output_path")
