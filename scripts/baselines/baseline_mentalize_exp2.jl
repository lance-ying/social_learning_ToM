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

include(joinpath(@__DIR__, "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "render.jl"))

# Define directory paths
experiment_id = "exp2"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

goal_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "goal")
state_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "state")
possible_worlds = load(joinpath(@__DIR__, "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "worlds")

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 0.5)

# Create progress bar
total_iterations = sum(length(v) for v in values(metadata))
progress = Progress(total_iterations, desc="Processing non-mentalize baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, goal_list) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    for (i, goal_str) in enumerate(goal_list)
        scenario_start_time = time()
        map_key = "$(map_id)_$(i)"
        
        # Clear planner cache for each scenario
        clear_planner_cache!()
        
        println("  Scenario $i (goal=$goal_str)")

        domain = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
        problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
        
        # Initialize and compile reference state
        state = initstate(domain, problem)
        state_render = copy(state)
        
        # Enumerate belief states
        initial_states, belief_probs, state_names = enumerate_beliefs(state)

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

        g_id = goal_str
        s_id = -1

        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
                break
            end
        end

        new_state = copy(state_render)

        # Compute Q_not_observe (cost without observing)
        Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, blue_wizards, action_cost)

        planner = AStarPlanner(GoalManhattan())
        plan_main = collect(planner(domain, state, problem.goal))

        # Compute Q_observe (cost with observing)
        Q_observe = calculate_plan_cost(plan_main, action_cost)

        # Find first interaction with blue wizard
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

        # Decide based on cost comparison
        if Q_observe < Q_not_observe
            steps_dict[map_key] = T
        else
            steps_dict[map_key] = 0
        end
        
        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        println("    Result: t=$(steps_dict[map_key])")
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
output_filename = "step_dict_nonmentalize.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")