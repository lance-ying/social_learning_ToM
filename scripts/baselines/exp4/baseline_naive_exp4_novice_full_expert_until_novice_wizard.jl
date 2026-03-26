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

serialize_observation(agent::String, action::Term, interaction_outcome::String="none") = Dict(
    "agent" => agent,
    "action" => write_pddl(action),
    "interaction_outcome" => interaction_outcome,
)

function agent_has_blue_item(state, agent_sym::Symbol)
    for key in PDDL.get_objects(state, :key)
        if state[pddl"(iscolor $key blue)"] && state[pddl"(has $agent_sym $key)"]
            return true
        end
    end
    return false
end

function interaction_outcome(state_before, state_after, agent_sym::Symbol, action::Term)
    if action.name != :interact
        return "none"
    end
    had_blue_before = agent_has_blue_item(state_before, agent_sym)
    has_blue_after = agent_has_blue_item(state_after, agent_sym)
    return (!had_blue_before && has_blue_after) ? "blue_amulet_present" : "blue_amulet_absent"
end

function observation_stop_horizon(plan)
    for (idx, action) in enumerate(plan)
        if action.name == :interact
            return idx
        end
    end
    return length(plan)
end

realized_observation_horizon(target_horizon::Int, observed_plan) = min(target_horizon, length(observed_plan))

function materialize_interleaved_observation_trace(map_key::String, observations, plan_agent2, plan_agent3, domain_agent2, state_agent2, domain_agent3, state_agent3)
    trace = Any[]
    events = Any[]
    agent2_idx = 0
    agent3_idx = 0
    observed_state_agent2 = copy(state_agent2)
    observed_state_agent3 = copy(state_agent3)
    for (obs_idx, observed_agent) in enumerate(observations)
        if observed_agent == "agent2"
            agent2_idx += 1
            agent2_idx <= length(plan_agent2) || error("Observed plan exhausted for $map_key: need $agent2_idx agent2 actions, found $(length(plan_agent2))")
            action = plan_agent2[agent2_idx]
            state_before_observation = copy(observed_state_agent2)
            observed_state_agent2 = PDDL.execute(domain_agent2, observed_state_agent2, action)
            observed_outcome = interaction_outcome(state_before_observation, observed_state_agent2, :agent2, action)
        else
            agent3_idx += 1
            agent3_idx <= length(plan_agent3) || error("Observed plan exhausted for $map_key: need $agent3_idx agent3 actions, found $(length(plan_agent3))")
            action = plan_agent3[agent3_idx]
            state_before_observation = copy(observed_state_agent3)
            observed_state_agent3 = PDDL.execute(domain_agent3, observed_state_agent3, action)
            observed_outcome = interaction_outcome(state_before_observation, observed_state_agent3, :agent3, action)
        end
        push!(trace, serialize_observation(observed_agent, action, observed_outcome))
        push!(events, Dict(
            "observation_index" => obs_idx,
            "observed_agent" => observed_agent,
            "action" => write_pddl(action),
            "interaction_outcome" => observed_outcome,
        ))
    end
    return trace, events
end

# Configuration section (matching wrapper pattern)
experiment_id = "exp4"  # Problem directory: problems_exp4
inference_file = "inference_exp4_020126_1.jld2"  # Configurable inference file (not used in naive)
output_experiment_id = "exp4"
model_label = "naive_observer_novice_full_expert_until_novice_wizard"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_exp4_013026")
OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "..", "model_outputs", "baselines", output_experiment_id)
mkpath(OUTPUT_DIR)

function write_json_to_paths(paths, payload; indent::Int=4)
    for path in paths
        open(path, "w") do io
            JSON.print(io, payload, indent)
        end
    end
end

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()
replay_trace_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # ~21 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing naive novice-full/expert-until-novice-wizard baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, agent_goals) in sort(collect(metadata), by=x->parse(Int, match(r"\d+", x[1]).match))
    map_start_time = time()
    println("\nProcessing map: $map_id")

    # Clear planner cache once per map (both scenarios use same plan)
    clear_planner_cache!()

    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)
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

    planner = AStarPlanner(GoalManhattan())
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

        # In exp4_013026, agent2 is always the expert (actual) agent and agent3 is the novice.
        novice_stop_horizon = observation_stop_horizon(observed_plan_agent3)
        agent2_count = realized_observation_horizon(novice_stop_horizon, observed_plan_agent2)
        agent3_count = length(observed_plan_agent3)
        T = agent2_count + agent3_count

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

        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "t" => T
        )
        observation_trace, observation_events = materialize_interleaved_observation_trace(
            map_key, observations, observed_plan_agent2, observed_plan_agent3,
            domain_agent2, state_agent2, domain_agent3, state_agent3
        )
        replay_trace_dict[map_key] = Dict(
            "t" => T,
            "observations" => observation_trace,
            "observation_events" => observation_events,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "stop_reason" => "novice_full_expert_until_novice_first_wizard_interaction",
        )
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
output_filename = "step_dict_naive_novice_full_expert_until_novice_wizard_exp4.json"
canonical_output_path = joinpath(OUTPUT_DIR, "step_dict_$(model_label).json")
write_json_to_paths((canonical_output_path,), steps_dict)

replay_trace_filename = "replay_trace_naive_novice_full_expert_until_novice_wizard_exp4.json"
canonical_replay_trace_path = joinpath(OUTPUT_DIR, "replay_trace_$(model_label).json")
write_json_to_paths((canonical_replay_trace_path,), replay_trace_dict)

println("\n=== Experiment Complete ===")
println("Results saved to: $canonical_output_path")
println("Replay trace saved to: $canonical_replay_trace_path")
