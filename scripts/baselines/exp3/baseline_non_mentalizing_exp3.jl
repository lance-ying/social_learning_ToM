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

        domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))
        problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))
        
        # Initialize and compile reference state
        state = initstate(domain, problem)
        state_render = copy(state)
        domain, state = PDDL.compiled(domain, problem)

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

        new_state = copy(state_render)

        # Compute Q_not_observe (cost without observing)
        Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, blue_wizards, action_cost)

        planner = AStarPlanner(GoalManhattan())
        plan_main = collect(planner(domain, state, problem.goal))

        # Compute Q_observe (cost with observing) - using main agent's plan only
        Q_observe = calculate_plan_cost(plan_main, action_cost)

        # Find first interaction with blue wizard in main agent's plan
        T = -1
        for (idx, action) in enumerate(plan_main)
            if action.name == :interact && action.args[end] in blue_wizards
                T = idx
                break
            end
        end

        if T == -1
            T = length(plan_main)
        end

        Q_observe = Q_observe + action_cost[:observe] * T

        # Decide based on cost comparison (same as exp2)
        if Q_observe < Q_not_observe
            # We observe both agent2 and agent3 equally (non-mentalizing doesn't know their goals)
            # Each agent gets T observations - split evenly
            agent2_count = T ÷ 2  # Integer division for first half
            agent3_count = T - agent2_count  # Remainder goes to agent3
            # Alternate between agent2 and agent3 observations
            observations = [i % 2 == 1 ? "agent2" : "agent3" for i in 1:(agent2_count + agent3_count)]
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
