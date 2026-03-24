using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))

# Configuration section (matching wrapper pattern)
experiment_id = "exp4"  # Problem directory: problems_exp4
inference_file = "inference_exp4_020126_1.jld2"  # Configurable inference file (not used in non-mentalizing)

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # ~21 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing non-mentalize baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, _) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")

    # Loop over both scenarios (matching wrapper pattern)
    for scenario in 1:2
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"

        # Clear planner cache for each scenario
        clear_planner_cache!()

        println("  Scenario $scenario")

        # Distance-based heuristic: don't use agent goals, plan to nearest blue wizard instead

        domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))
        problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))

        # Initialize and compile reference state
        state = initstate(domain, problem)
        state_render = copy(state)
        domain, state = PDDL.compiled(domain, problem)

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

        # Load filtered problems for each agent to get agent-specific blue wizards
        include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))
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

        txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
        ascii_content = read(txt_path, String)

        # Load filtered problem for agent2
        domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
        if !isfile(temp_path_agent2)
            filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
            write(temp_path_agent2, filtered_ascii_agent2)
        end
        problem_agent2 = load_ascii_problem(temp_path_agent2)
        state_agent2 = initstate(domain_agent2, problem_agent2)
        domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)
        blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]

        # Load filtered problem for agent3
        domain_agent3 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
        if !isfile(temp_path_agent3)
            filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
            write(temp_path_agent3, filtered_ascii_agent3)
        end
        problem_agent3 = load_ascii_problem(temp_path_agent3)
        state_agent3 = initstate(domain_agent3, problem_agent3)
        domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)
        blue_wizards_agent3 = [w for w in PDDL.get_objects(state_agent3, :wizard) if state_agent3[pddl"(iscolor $w blue)"]]

        new_state = copy(state_render)

        # Compute Q_not_observe (cost without observing)
        Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, blue_wizards, action_cost)

        planner = AStarPlanner(GoalManhattan())

        # Distance-based heuristic: plan paths to nearest blue wizard (not actual goals)
        # Find nearest blue wizard for each agent
        agent2_loc = get_obj_loc(state_agent2, Const(:agent2))
        agent3_loc = get_obj_loc(state_agent3, Const(:agent3))

        # Find nearest blue wizard for agent2
        nearest_wizard_agent2 = blue_wizards_agent2[1]
        min_dist_agent2 = Inf
        for wizard in blue_wizards_agent2
            wizard_loc = get_obj_loc(state_agent2, wizard)
            dist = sum(abs.(agent2_loc .- wizard_loc))
            if dist < min_dist_agent2
                min_dist_agent2 = dist
                nearest_wizard_agent2 = wizard
            end
        end

        # Find nearest blue wizard for agent3
        nearest_wizard_agent3 = blue_wizards_agent3[1]
        min_dist_agent3 = Inf
        for wizard in blue_wizards_agent3
            wizard_loc = get_obj_loc(state_agent3, wizard)
            dist = sum(abs.(agent3_loc .- wizard_loc))
            if dist < min_dist_agent3
                min_dist_agent3 = dist
                nearest_wizard_agent3 = wizard
            end
        end

        # Plan paths to nearest blue wizard (not actual goals)
        wizard2_loc = get_obj_loc(state_agent2, nearest_wizard_agent2)
        wizard3_loc = get_obj_loc(state_agent3, nearest_wizard_agent3)

        goal_agent2 = PDDL.parse_pddl("(and (= (xloc agent2) $(wizard2_loc[1])) (= (yloc agent2) $(wizard2_loc[2])))")
        goal_agent3 = PDDL.parse_pddl("(and (= (xloc agent3) $(wizard3_loc[1])) (= (yloc agent3) $(wizard3_loc[2])))")

        plan_agent2 = collect(planner(domain_agent2, state_agent2, goal_agent2))
        plan_agent3 = collect(planner(domain_agent3, state_agent3, goal_agent3))

        # Compute Q_observe_agent2 using same logic as exp2: plan cost + observation cost
        Q_observe_agent2 = calculate_plan_cost(plan_agent2, action_cost)

        # Find first interaction with blue wizard in agent2's plan
        T_agent2 = -1
        for (idx, action) in enumerate(plan_agent2)
            if action.name == :interact && action.args[end] in blue_wizards_agent2
                T_agent2 = idx
                break
            end
        end

        if T_agent2 == -1
            T_agent2 = length(plan_agent2)
        end

        Q_observe_agent2 = Q_observe_agent2 + action_cost[:observe] * T_agent2

        # Compute Q_observe_agent3 using same logic as exp2: plan cost + observation cost
        Q_observe_agent3 = calculate_plan_cost(plan_agent3, action_cost)

        # Find first interaction with blue wizard in agent3's plan
        T_agent3 = -1
        for (idx, action) in enumerate(plan_agent3)
            if action.name == :interact && action.args[end] in blue_wizards_agent3
                T_agent3 = idx
                break
            end
        end

        if T_agent3 == -1
            T_agent3 = length(plan_agent3)
        end

        Q_observe_agent3 = Q_observe_agent3 + action_cost[:observe] * T_agent3

        # Decide based on cost comparison - choose agent with lowest Q-value
        q_values = [Q_observe_agent2, Q_observe_agent3, Q_not_observe]
        best_action_idx = argmin(q_values)

        if best_action_idx == 3
            # Not observing has the lowest Q-value
            steps_dict[map_key] = Dict(
                "observations" => [],
                "agent2_count" => 0,
                "agent3_count" => 0,
                "t" => 0
            )
        elseif best_action_idx == 1
            # Observe agent2 - it has the lowest Q-value
            observations = ["agent2" for _ in 1:T_agent2]
            steps_dict[map_key] = Dict(
                "observations" => observations,
                "agent2_count" => T_agent2,
                "agent3_count" => 0,
                "t" => T_agent2
            )
        else
            # Observe agent3 - it has the lowest Q-value
            observations = ["agent3" for _ in 1:T_agent3]
            steps_dict[map_key] = Dict(
                "observations" => observations,
                "agent2_count" => 0,
                "agent3_count" => T_agent3,
                "t" => T_agent3
            )
        end

        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        result_t = steps_dict[map_key]["t"]
        println("    Result: t=$result_t")
        println("    Time: $(round(scenario_elapsed, digits=2))s")
        println("    Cache: $(cache_stats.hits) hits, $(cache_stats.misses) misses, $(round(cache_stats.hit_rate * 100, digits=1))% hit rate")

        next!(progress)
    end

    map_elapsed = time() - map_start_time
    map_times[map_id] = map_elapsed
    println("  Map completed in $(round(map_elapsed, digits=2))s")
end

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
println("Average per map: $(round(mean(values(map_times)), digits=2))s")

# Save results
output_filename = "step_dict_nonmentalize_distance_exp4.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
