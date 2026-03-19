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
include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "planners.jl"))

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

# Configuration section (matching wrapper pattern)
experiment_id = "exp4"  # Problem directory: problems_exp4
inference_file = "inference_exp4_020126_1.jld2"  # Configurable inference file (not used in naive)

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_exp4_013026")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()
replay_trace_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # ~21 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing naive baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

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

    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)
    domain_agent1 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent1 = joinpath(PROBLEM_DIR, ".temp_agent1_$(map_id).txt")
    if !isfile(temp_path_agent1)
        filtered_ascii_agent1 = filter_ascii_agents(ascii_content, :agent1)
        write(temp_path_agent1, filtered_ascii_agent1)
    end
    problem_agent1 = load_ascii_problem(temp_path_agent1)
    state_agent1 = initstate(domain_agent1, problem_agent1)
    domain_agent1, state_agent1 = PDDL.compiled(domain_agent1, problem_agent1)

    domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    if !isfile(temp_path_agent2)
        filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
        write(temp_path_agent2, filtered_ascii_agent2)
    end
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2 = initstate(domain_agent2, problem_agent2)
    domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)
    goals_agent2, _ = initialize_goals(state_agent2, :agent2)

    domain_agent3 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
    if !isfile(temp_path_agent3)
        filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
        write(temp_path_agent3, filtered_ascii_agent3)
    end
    problem_agent3 = load_ascii_problem(temp_path_agent3)
    state_agent3 = initstate(domain_agent3, problem_agent3)
    domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)
    goals_agent3, _ = initialize_goals(state_agent3, :agent3)

    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

    planner = AStarPlanner(GoalManhattan())
    plan = collect(planner(domain_agent1, state_agent1, problem_agent1.goal))

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

    # Replay requires a concrete observation trace, so materialize an
    # alternating sequence whose counts sum to T.
    agent2_count = cld(T, 2)
    agent3_count = fld(T, 2)
    observations = String[]
    for obs_idx in 1:T
        push!(observations, isodd(obs_idx) ? "agent2" : "agent3")
    end

    # Both scenarios get the same result (naive doesn't use scenario-specific goals)
    for scenario in 1:2
        map_key = "$(map_id)_scenario$(scenario)"
        agent2_goal_info = agent_goals["agent2"][scenario]
        agent3_goal_info = agent_goals["agent3"][scenario]
        agent2_gem = agent2_goal_info["gem"]
        agent3_gem = agent3_goal_info["gem"]
        agent2_type = agent2_goal_info["type"]
        agent3_type = agent3_goal_info["type"]
        blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]
        blue_wizards_agent3 = [w for w in PDDL.get_objects(state_agent3, :wizard) if state_agent3[pddl"(iscolor $w blue)"]]
        observed_plan_agent2 = agent2_type == "naive" ?
            generate_naive_plan(domain_agent2, state_agent2, goals_agent2[agent2_gem], blue_wizards_agent2, :agent2, planner) :
            collect(planner(domain_agent2, state_agent2, goals_agent2[agent2_gem]))
        observed_plan_agent3 = agent3_type == "naive" ?
            generate_naive_plan(domain_agent3, state_agent3, goals_agent3[agent3_gem], blue_wizards_agent3, :agent3, planner) :
            collect(planner(domain_agent3, state_agent3, goals_agent3[agent3_gem]))
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
            "stop_reason" => "first_blue_wizard_interaction",
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

replay_trace_filename = "replay_trace_naive_exp4.json"
open(replay_trace_filename, "w") do f
    JSON.print(f, replay_trace_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
println("Replay trace saved to: $replay_trace_filename")
