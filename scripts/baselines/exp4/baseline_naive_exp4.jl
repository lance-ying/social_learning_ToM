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

# Configuration section (matching wrapper pattern)
experiment_id = "exp4"  # Problem directory: problems_exp4
inference_file = "inference_exp4_020126_1.jld2"  # Configurable inference file (not used in naive)

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_exp4_013026")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # ~21 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing naive baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, agent_goals) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")

    # Loop over both scenarios (matching wrapper pattern)
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

        # Naive baseline observes both agents equally (50/50 split) - simplest possible observation strategy
        # No reasoning about which agent to observe, just split observations evenly
        agent2_count = T ÷ 2  # Integer division for first half
        agent3_count = T - agent2_count  # Remainder goes to agent3

        observations = vcat(
            ["agent2" for _ in 1:agent2_count],
            ["agent3" for _ in 1:agent3_count]
        )

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
output_filename = "step_dict_naive_exp4.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
