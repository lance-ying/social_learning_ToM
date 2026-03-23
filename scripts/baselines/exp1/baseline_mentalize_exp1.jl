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

function realized_observation_horizon(target_horizon::Int, observed_plan)
    return min(target_horizon, length(observed_plan))
end

# Define directory paths
experiment_id = "exp1"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
steps_dict = Dict()
replay_trace_dict = Dict()

goal_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "goal")
state_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "state")
possible_worlds = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "worlds")

# Debug: Print available maps in inference data
println("Maps in inference data: ", sort(collect(keys(goal_probs_conditioned_dict))))

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 1)

# Get all map files
map_files = filter(f -> endswith(f, ".pddl") && !occursin("_plan", f), readdir(PROBLEM_DIR))
map_ids = [replace(f, ".pddl" => "") for f in map_files]

# Create progress bar
total_iterations = length(map_ids)
progress = Progress(total_iterations, desc="Processing mentalizing baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for map_id in map_ids
    # Map map_id to inference data format (mod_XXX_ascii)
    inference_map_id = "mod_$(map_id)_ascii"
    
    # Skip maps that don't exist in inference data
    if !haskey(goal_probs_conditioned_dict, inference_map_id)
        println("\nSkipping map $map_id (not in inference data as $inference_map_id)")
        next!(progress)
        continue
    end
    
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

    goal_probs = goal_probs_conditioned_dict[inference_map_id][g_id][s_id]
    state_probs = state_probs_conditioned_dict[inference_map_id][g_id][s_id]

        new_state = copy(state_agent1)

        planner = AStarPlanner(GoalManhattan())
        plan = planner(domain_agent1, state_agent1, problem_agent1.goal)

        max_t = length(goal_probs[1,:]) - 1
        T = max_t

        # Find when state distributions diverge
        for t in 1:max_t  # Check all timesteps (removed 50 limit)

            curr_state_dist = state_probs[:, t]
            flag = true

            for g in 1:3
                if goal_probs[g, t+1] > 0.1
                    for s in 1:length(initial_states)
                        if state_probs[s, t+1] > 0.1
                            max_t_available = size(state_probs_conditioned_dict[inference_map_id][g][s], 2)
                            for val in t:max_t_available  # Check all future timesteps (removed t+10 lookahead limit)
                                if eval_state_dist(curr_state_dist, state_probs_conditioned_dict[inference_map_id][g][s][:, val])
                                    flag = false
                                    break
                                end
                            end
                        end
                        if !flag
                            break
                        end
                    end
                end
                if !flag
                    break
                end
            end

            if flag
                T = t
                break
            end
        end

        observed_plan_agent2 = collect(planner(domain_agent2, state_agent2, observed_agent_goals[g_id]))
        T = realized_observation_horizon(T, observed_plan_agent2)
        observation_trace, observation_events = materialize_single_observation_trace(
            "agent2", observed_plan_agent2, domain_agent2, state_agent2, T, :agent2
        )

        steps_dict[map_key] = T
        replay_trace_dict[map_key] = Dict(
            "t" => T,
            "observations" => observation_trace,
            "observation_events" => observation_events,
            "stop_reason" => T < max_t ? "state_divergence" : "observed_plan_or_horizon_exhausted",
        )
    
    scenario_elapsed = time() - scenario_start_time
    cache_stats = get_cache_stats()
    println("    Result: t=$T")
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
output_filename = "step_dict_mentalize_exp1.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict)
end

replay_trace_filename = "replay_trace_mentalize_exp1.json"
open(replay_trace_filename, "w") do f
    JSON.print(f, replay_trace_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
println("Replay trace saved to: $replay_trace_filename")
