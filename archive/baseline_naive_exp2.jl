using PDDL, SymbolicPlanners
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
# beliefs.jl not needed for naive baseline
include(joinpath(@__DIR__, "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "render.jl"))

# Define directory paths
experiment_id = "exp2"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 0.5)

# Create progress bar
total_iterations = sum(length(v) for v in values(metadata))
progress = Progress(total_iterations, desc="Processing naive baseline: ")

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
        
        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

        planner = AStarPlanner(GoalManhattan())
        plan = collect(planner(domain, state, problem.goal))

        # Find first interaction with blue wizard
        T = -1
        for (idx, action) in enumerate(plan)
            if action.name == :interact && action.args[end] in blue_wizards
                T = idx
                break
            end
        end

        if T == -1
            T = length(plan)
        end

        steps_dict[map_key] = T
        
        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        println("    Result: t=$T")
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
output_filename = "step_dict_naive_exp2.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")

