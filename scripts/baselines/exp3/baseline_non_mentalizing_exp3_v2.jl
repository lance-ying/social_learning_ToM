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

# Define directory paths
experiment_id = "exp3"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()
replay_trace_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # 25 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing non-mentalize baseline v2: ")

# Track timing
map_times = Dict()
total_start_time = time()

# Filter ASCII map to keep only agent1
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

serialize_observation(agent::String, action::Term) = Dict(
    "agent" => agent,
    "action" => write_pddl(action),
)

function materialize_interleaved_observation_trace(map_key::String, observations, plan_agent2, plan_agent3)
    trace = Any[]
    events = Any[]
    agent2_idx = 0
    agent3_idx = 0
    for (obs_idx, observed_agent) in enumerate(observations)
        if observed_agent == "agent2"
            agent2_idx += 1
            agent2_idx <= length(plan_agent2) || error("Observed plan exhausted for $map_key: need $agent2_idx agent2 actions, found $(length(plan_agent2))")
            action = plan_agent2[agent2_idx]
        else
            agent3_idx += 1
            agent3_idx <= length(plan_agent3) || error("Observed plan exhausted for $map_key: need $agent3_idx agent3 actions, found $(length(plan_agent3))")
            action = plan_agent3[agent3_idx]
        end
        push!(trace, serialize_observation(observed_agent, action))
        push!(events, Dict(
            "observation_index" => obs_idx,
            "observed_agent" => observed_agent,
            "action" => write_pddl(action),
        ))
    end
    return trace, events
end

"""
Run cost comparison on an agent1-only planning sub-problem.
Returns (should_observe::Bool, T::Int).
"""
function agent_cost_comparison(domain_render, domain_path, problem_dir, map_id,
                                observe_agent::Symbol, remove_agent::Symbol, action_cost)
    domain_sub = load_domain(domain_path)
    txt_path = joinpath(problem_dir, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    temp_path = joinpath(problem_dir, ".temp_agent1_$(map_id).txt")
    if !isfile(temp_path)
        filtered_ascii = filter_ascii_agents(ascii_content, :agent1)
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

for (map_id, agent_goals) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")

    clear_planner_cache!()

    domain_agent2 = load_domain(domain_path)
    domain_agent3 = load_domain(domain_path)
    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    if !isfile(temp_path_agent2)
        filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
        write(temp_path_agent2, filtered_ascii_agent2)
    end
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2 = initstate(domain_agent2, problem_agent2)
    goals_agent2, _ = initialize_goals(state_agent2, :agent2)

    temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
    if !isfile(temp_path_agent3)
        filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
        write(temp_path_agent3, filtered_ascii_agent3)
    end
    problem_agent3 = load_ascii_problem(temp_path_agent3)
    state_agent3 = initstate(domain_agent3, problem_agent3)
    goals_agent3, _ = initialize_goals(state_agent3, :agent3)

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
        observed_plan_agent2 = collect(AStarPlanner(GoalManhattan())(domain_agent2, state_agent2, goals_agent2[agent_goals["agent2"][scenario]]))
        observed_plan_agent3 = collect(AStarPlanner(GoalManhattan())(domain_agent3, state_agent3, goals_agent3[agent_goals["agent3"][scenario]]))
        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "t" => T
        )
        observation_trace, observation_events = materialize_interleaved_observation_trace(
            map_key, observations, observed_plan_agent2, observed_plan_agent3
        )
        replay_trace_dict[map_key] = Dict(
            "t" => T,
            "observations" => observation_trace,
            "observation_events" => observation_events,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "agent2_should_observe" => should_observe_agent2,
            "agent3_should_observe" => should_observe_agent3,
            "agent2_t_if_observed" => T_agent2,
            "agent3_t_if_observed" => T_agent3,
            "stop_reason" => "cost_comparison",
        )
        next!(progress)
    end

    map_elapsed = time() - map_start_time
    map_times[map_id] = map_elapsed
    println("  Result: agent2=$agent2_count (observe=$should_observe_agent2), agent3=$agent3_count (observe=$should_observe_agent3), total=$T")
    println("  Map completed in $(round(map_elapsed, digits=2))s")
end

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
println("Average per map: $(round(mean(collect(Float64, values(map_times))), digits=2))s")

# Save results
output_filename = "step_dict_nonmentalize_exp3_v2.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

replay_trace_filename = "replay_trace_nonmentalize_exp3_v2.json"
open(replay_trace_filename, "w") do f
    JSON.print(f, replay_trace_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
println("Replay trace saved to: $replay_trace_filename")
