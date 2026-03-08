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

include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))

for (map_id, agent_goals) in sort(collect(metadata), by=x->parse(Int, match(r"\d+", x[1]).match))
    map_start_time = time()
    println("\nProcessing map: $map_id")

    # Clear planner cache once per map (both scenarios use same plan)
    clear_planner_cache!()

    # Load, init, and compile once per map (shared across scenarios)
    domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))
    state = initstate(domain, problem)
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

    # Naive expected-value policy: both agents receive T/2 expected observations.
    # We do not instantiate a per-step observation assignment when T is odd.
    expected_count = T / 2
    observations = String[]

    # Both scenarios get the same result (naive doesn't use scenario-specific goals)
    for scenario in 1:2
        map_key = "$(map_id)_scenario$(scenario)"
        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => expected_count,
            "agent3_count" => expected_count,
            "t" => T
        )
        next!(progress)
    end

    map_elapsed = time() - map_start_time
    map_times[map_id] = map_elapsed
    println("  Result: t=$T, completed in $(round(map_elapsed, digits=2))s")
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
