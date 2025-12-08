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

# Define directory paths
experiment_id = "exp3"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # 25 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing non-mentalize baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, agent_goals) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    # Loop over both scenarios
    for scenario in 1:2
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"
        
        # Clear planner cache for each scenario
        clear_planner_cache!()
        
        println("  Scenario $scenario")
        
        # Get which gems each agent wants in this scenario
        agent2_gem = agent_goals["agent2"][scenario]  # X's goal
        agent3_gem = agent_goals["agent3"][scenario]  # Y's goal

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
        plan_main = collect(planner(domain, state, problem.goal))

        # Compute Q_observe for agent2: same logic as exp2 but using agent2's filtered state
        # Plan agent2's path in filtered state (only agent2 exists)
        # Create goal for agent2 based on scenario metadata
        goal_agent2 = PDDL.parse_pddl("(has agent2 gem$agent2_gem)")
        plan_agent2 = collect(planner(domain_agent2, state_agent2, goal_agent2))
        
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
        
        # Compute Q_observe for agent3: same logic as exp2 but using agent3's filtered state
        # Plan agent3's path in filtered state (only agent3 exists)
        # Create goal for agent3 based on scenario metadata
        goal_agent3 = PDDL.parse_pddl("(has agent3 gem$agent3_gem)")
        plan_agent3 = collect(planner(domain_agent3, state_agent3, goal_agent3))
        
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

        # Decide based on cost comparison (same as exp2)
        if Q_observe_agent2 < Q_not_observe
            # We observe both agent2 and agent3 simultaneously (non-mentalizing doesn't distinguish)
            # Each agent gets T observations - we treat observing as observing both agents
            T = T_agent2 # Use T_agent2 as the number of observations for both
            agent2_count = T
            agent3_count = T
            # Alternate between agent2 and agent3 observations
            observations = [i % 2 == 1 ? "agent2" : "agent3" for i in 1:(2*T)]
            steps_dict[map_key] = Dict(
                "observations" => observations,
                "agent2_count" => agent2_count,
                "agent3_count" => agent3_count,
                "t" => T
            )
        else
            # Not observing has the lower Q-value
            steps_dict[map_key] = Dict(
                "observations" => [],
                "agent2_count" => 0,
                "agent3_count" => 0,
                "t" => 0
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
output_filename = "step_dict_nonmentalize_exp3.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
