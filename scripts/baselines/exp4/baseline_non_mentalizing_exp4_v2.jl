using PDDL, SymbolicPlanners
# using GenGPT3
# using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "heuristics.jl"))
# include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))

# Configuration section (matching wrapper pattern)
experiment_id = "exp4_013026"  # Problem directory: problems_exp4_013026
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
progress = Progress(total_iterations, desc="Processing non-mentalize baseline v2: ")

# Track timing
map_times = Dict()
total_start_time = time()

# Filter ASCII map to keep agent1 + one other agent (remove the third)
function filter_to_pair(ascii_content::String, remove_agent::Symbol)
    agent_chars = Dict(:agent2 => 'X', :agent3 => 'Y')
    return replace(ascii_content, agent_chars[remove_agent] => '.')
end

"""
Run cost comparison on a sub-problem with agent1 + one observed agent.
The other agent is removed from the map so it doesn't inflate compilation.
Returns (should_observe::Bool, T::Int).
"""
function agent_cost_comparison(domain_render, domain_path, problem_dir, map_id,
                                observe_agent::Symbol, remove_agent::Symbol, action_cost)
    domain_sub = load_domain(domain_path)
    txt_path = joinpath(problem_dir, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    temp_path = joinpath(problem_dir, ".temp_pair_$(observe_agent)_$(map_id).txt")
    if !isfile(temp_path)
        filtered_ascii = filter_to_pair(ascii_content, remove_agent)
        write(temp_path, filtered_ascii)
    end
    problem_sub = load_ascii_problem(temp_path)
    state_sub = initstate(domain_sub, problem_sub)
    state_render_sub = copy(state_sub)
    # Don't compile — planner works fine without compilation (see mentalize baseline v2)

    blue_wizards = [w for w in PDDL.get_objects(state_sub, :wizard) if state_sub[pddl"(iscolor $w blue)"]]

    if isempty(blue_wizards)
        return (false, 0)
    end

    new_state = copy(state_render_sub)

    # Q_not_observe: cost of agent1 self-exploring on this sub-problem
    Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem_sub.goal, blue_wizards, action_cost)

    planner = AStarPlanner(GoalManhattan())
    plan = collect(planner(domain_sub, state_sub, problem_sub.goal))

    # Q_observe: plan cost + observation cost
    Q_observe = calculate_plan_cost(plan, action_cost)

    # T = first interaction with blue wizard
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

    Q_observe = Q_observe + action_cost[:observe] * T

    return (Q_observe < Q_not_observe, T)
end

domain_path = joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl")

for (map_id, agent_goals) in sort(collect(metadata), by=x->parse(Int, match(r"\d+", x[1]).match))
    map_start_time = time()
    println("\nProcessing map: $map_id")

    clear_planner_cache!()

    # Compute cost comparison once per map (results are the same for both scenarios)
    # Sub-problem 1: agent1 + agent2 (remove agent3)
    should_observe_agent2, T_agent2 = agent_cost_comparison(
        domain_render, domain_path, PROBLEM_DIR, map_id, :agent2, :agent3, action_cost)

    clear_planner_cache!()

    # Sub-problem 2: agent1 + agent3 (remove agent2)
    should_observe_agent3, T_agent3 = agent_cost_comparison(
        domain_render, domain_path, PROBLEM_DIR, map_id, :agent3, :agent2, action_cost)

    agent2_count = should_observe_agent2 ? T_agent2 : 0
    agent3_count = should_observe_agent3 ? T_agent3 : 0
    T = agent2_count + agent3_count

    # Build interleaved observations list
    observations = String[]
    a2_remaining = agent2_count
    a3_remaining = agent3_count
    while a2_remaining > 0 || a3_remaining > 0
        if a2_remaining > 0
            push!(observations, "agent2")
            a2_remaining -= 1
        end
        if a3_remaining > 0
            push!(observations, "agent3")
            a3_remaining -= 1
        end
    end

    # Both scenarios get the same result (non-mentalizing doesn't use scenario-specific goals)
    for scenario in 1:2
        map_key = "$(map_id)_scenario$(scenario)"
        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "t" => T
        )
        next!(progress)
    end

    map_elapsed = time() - map_start_time
    println("  Result: agent2=$agent2_count (observe=$should_observe_agent2), agent3=$agent3_count (observe=$should_observe_agent3), total=$T")
    println("  Map completed in $(round(map_elapsed, digits=2))s")
end

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
println("Average per map: $(round(mean(collect(Float64, values(map_times))), digits=2))s")

# Save results
output_filename = "step_dict_nonmentalize_exp4_v2.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
