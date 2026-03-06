using PDDL, SymbolicPlanners
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "heuristics.jl"))
# beliefs.jl not needed for naive baseline
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
progress = Progress(total_iterations, desc="Processing naive baseline: ")

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

        # Distance-based heuristic: observe the agent closest to a blue wizard
        agent2_loc = get_obj_loc(state, Const(:agent2))
        agent3_loc = get_obj_loc(state, Const(:agent3))
        
        # Find minimum distance from each agent to any blue wizard
        min_dist_agent2 = Inf
        min_dist_agent3 = Inf
        
        for wizard in blue_wizards
            wizard_loc = get_obj_loc(state, wizard)
            dist_agent2 = sum(abs.(agent2_loc .- wizard_loc))
            dist_agent3 = sum(abs.(agent3_loc .- wizard_loc))
            min_dist_agent2 = min(min_dist_agent2, dist_agent2)
            min_dist_agent3 = min(min_dist_agent3, dist_agent3)
        end
        
        # Observe the agent closer to a blue wizard (or agent2 if tie)
        if min_dist_agent2 <= min_dist_agent3
            observed_agent = "agent2"
            agent2_count = T
            agent3_count = 0
        else
            observed_agent = "agent3"
            agent2_count = 0
            agent3_count = T
        end
        
        observations = [observed_agent for _ in 1:T]
        
        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "t" => T
        )
        
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
output_filename = "step_dict_naive_distance_heur_exp3.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
