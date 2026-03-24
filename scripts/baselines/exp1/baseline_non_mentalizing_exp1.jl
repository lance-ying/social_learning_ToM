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
include(joinpath(@__DIR__, "..", "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))

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

function materialize_single_observation_trace(observed_agent::String, observed_plan, domain, start_state, T::Int, agent_sym::Symbol)
    trace = Any[]
    events = Any[]
    observed_state = copy(start_state)
    for obs_idx in 1:T
        action = observed_plan[obs_idx]
        state_before_observation = copy(observed_state)
        observed_state = PDDL.execute(domain, observed_state, action)
        observed_outcome = interaction_outcome(state_before_observation, observed_state, agent_sym, action)
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

function observation_stop_horizon(plan)
    for (idx, action) in enumerate(plan)
        if action.name == :interact
            return idx
        end
    end
    return length(plan)
end

# Define directory paths
experiment_id = "exp1"
model_label = "rational_non_mentalizing"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")
OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "..", "model_outputs", "baselines", experiment_id)
mkpath(OUTPUT_DIR)

function write_json_to_paths(paths, payload; indent::Int=4)
    for path in paths
        open(path, "w") do io
            JSON.print(io, payload, indent)
        end
    end
end

#--- Initial Setup ---#
steps_dict = Dict()
replay_trace_dict = Dict()

goal_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "inference", "inference_data_$experiment_id.jld2"), "goal")
state_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "inference", "inference_data_$experiment_id.jld2"), "state")
possible_worlds = load(joinpath(@__DIR__, "..", "..", "..", "inference", "inference_data_$experiment_id.jld2"), "worlds")

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 0.5)

# Get all map files
map_files = filter(f -> endswith(f, ".pddl") && !occursin("_plan", f), readdir(PROBLEM_DIR))
map_ids = [replace(f, ".pddl" => "") for f in map_files]

# Create progress bar
total_iterations = length(map_ids)
progress = Progress(total_iterations, desc="Processing non-mentalize baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for map_id in map_ids
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    scenario_start_time = time()
    map_key = map_id
    
    # Clear planner cache
    clear_planner_cache!()

    domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
    
    # Initialize and compile reference state
    state = initstate(domain, problem)
    state_render = copy(state)

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

    domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    if !isfile(temp_path_agent2)
        filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
        write(temp_path_agent2, filtered_ascii_agent2)
    end
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2 = initstate(domain_agent2, problem_agent2)
    domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)
    
    # Get goal_id from problem
    goal_id = parse(Int, string(problem.goal.args[2])[end:end])
    observed_agent_goals, _ = initialize_goals(state_agent2, :agent2)
    
    # Enumerate belief states
    initial_states, belief_probs, state_names = enumerate_beliefs(state)

    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

    g_id = goal_id
        s_id = -1

        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
                break
            end
        end

        new_state = copy(state_agent1)

        # Compute Q_not_observe (cost without observing)
        Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem_agent1.goal, blue_wizards, action_cost)

        planner = AStarPlanner(GoalManhattan())
        plan_main = collect(planner(domain_agent1, state_agent1, problem_agent1.goal))
        observed_plan_agent2 = collect(AStarPlanner(GoalManhattan())(domain_agent2, state_agent2, observed_agent_goals[g_id]))
        observed_T = observation_stop_horizon(observed_plan_agent2)

        # Compute Q_observe (cost with observing) using the realizable observed-agent horizon.
        Q_observe = calculate_plan_cost(plan_main, action_cost) + action_cost[:observe] * observed_T

    # Decide based on cost comparison
    if Q_observe < Q_not_observe
        steps_dict[map_key] = observed_T
    else
        steps_dict[map_key] = 0
    end
    observation_trace, observation_events = materialize_single_observation_trace(
        "agent2", observed_plan_agent2, domain_agent2, state_agent2, steps_dict[map_key], :agent2
    )
    replay_trace_dict[map_key] = Dict(
        "t" => steps_dict[map_key],
        "observations" => observation_trace,
        "observation_events" => observation_events,
        "q_observe" => Q_observe,
        "q_not_observe" => Q_not_observe,
        "stop_reason" => steps_dict[map_key] > 0 ? "q_observe_better" : "q_not_observe_better",
    )
    
    scenario_elapsed = time() - scenario_start_time
    cache_stats = get_cache_stats()
    println("    Result: t=$(steps_dict[map_key])")
    println("    Time: $(round(scenario_elapsed, digits=2))s")
    println("    Cache: $(cache_stats.hits) hits, $(cache_stats.misses) misses, $(round(cache_stats.hit_rate * 100, digits=1))% hit rate")
    
    map_elapsed = time() - map_start_time
    map_times[map_id] = map_elapsed
    println("  Map completed in $(round(map_elapsed, digits=2))s")
    
    next!(progress)
end

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
if length(map_times) > 0
    println("Average per map: $(round(mean(values(map_times)), digits=2))s")
else
    println("No maps processed")
end

# Save results
output_filename = "step_dict_nonmentalize_exp1.json"
canonical_output_path = joinpath(OUTPUT_DIR, "step_dict_$(model_label).json")
write_json_to_paths((canonical_output_path,), steps_dict)

replay_trace_filename = "replay_trace_nonmentalize_exp1.json"
canonical_replay_trace_path = joinpath(OUTPUT_DIR, "replay_trace_$(model_label).json")
write_json_to_paths((canonical_replay_trace_path,), replay_trace_dict)

println("\n=== Experiment Complete ===")
println("Results saved to: $canonical_output_path")
println("Replay trace saved to: $canonical_replay_trace_path")
