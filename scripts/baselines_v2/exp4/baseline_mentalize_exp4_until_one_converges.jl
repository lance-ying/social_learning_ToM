using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
# using GenGPT3
using InversePlanning
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
include(joinpath(@__DIR__, "..", "..", "..", "src", "beliefs.jl"))
# include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "planners.jl"))

# Configuration section (matching wrapper pattern)
experiment_id = "exp4_013026"  # Problem directory: problems_exp4_013026
inference_file = "inference_exp4_020126_1.jld2"  # Configurable inference file
output_experiment_id = "exp4"
model_label = "social_mentalizing_until_one_converges"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")
OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "..", "model_outputs", "baselines_v2", output_experiment_id)
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

# Load inference data for both agents (agent2=X, agent3=Y)
data = load(joinpath(@__DIR__, "..", "..", "..", "inference", inference_file))
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]
possible_worlds = data["worlds"]

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # ~21 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing mentalizing baseline until one converges: ")

# Track timing
map_times = Dict()
total_start_time = time()

# Helper function to filter ASCII maps to single agent
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
serialize_wizards(wizards) = sort(string.(wizards))

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

function realized_observation_horizon(target_horizon::Int, observed_plan)
    return min(target_horizon, length(observed_plan))
end

function build_observation_prefix_until_one_converges(agent2_horizon::Int, agent3_horizon::Int)
    observations = String[]
    agent2_count = 0
    agent3_count = 0

    if agent2_horizon <= 0 || agent3_horizon <= 0
        return observations, agent2_count, agent3_count
    end

    while true
        if agent2_count < agent2_horizon
            push!(observations, "agent2")
            agent2_count += 1
            if agent2_count >= agent2_horizon
                break
            end
        end

        if agent3_count < agent3_horizon
            push!(observations, "agent3")
            agent3_count += 1
            if agent3_count >= agent3_horizon
                break
            end
        end

        if agent2_count >= agent2_horizon || agent3_count >= agent3_horizon
            break
        end
    end

    return observations, agent2_count, agent3_count
end

function materialize_interleaved_observation_trace(
    map_key::String,
    observations,
    plan_agent2,
    plan_agent3,
    domain_agent2,
    state_agent2,
    domain_agent3,
    state_agent3,
    blue_wizards,
    state_probs_agent2,
    state_probs_agent3,
)
    trace = Any[]
    events = Any[]
    agent2_idx = 0
    agent3_idx = 0
    observed_state_agent2 = copy(state_agent2)
    observed_state_agent3 = copy(state_agent3)
    wizard_candidates = copy(blue_wizards)
    for (obs_idx, observed_agent) in enumerate(observations)
        candidates_before = serialize_wizards(wizard_candidates)
        if observed_agent == "agent2"
            agent2_idx += 1
            agent2_idx <= length(plan_agent2) || error("Observed plan exhausted for $map_key: need $agent2_idx agent2 actions, found $(length(plan_agent2))")
            action = plan_agent2[agent2_idx]
            state_before_observation = copy(observed_state_agent2)
            observed_state_agent2 = PDDL.execute(domain_agent2, observed_state_agent2, action)
            observed_outcome = interaction_outcome(state_before_observation, observed_state_agent2, :agent2, action)
            wizard_candidates = [
                blue_wizards[j] for j in 1:length(blue_wizards)
                if state_probs_agent2[j, agent2_idx + 1] > 0.1
            ]
        else
            agent3_idx += 1
            agent3_idx <= length(plan_agent3) || error("Observed plan exhausted for $map_key: need $agent3_idx agent3 actions, found $(length(plan_agent3))")
            action = plan_agent3[agent3_idx]
            state_before_observation = copy(observed_state_agent3)
            observed_state_agent3 = PDDL.execute(domain_agent3, observed_state_agent3, action)
            observed_outcome = interaction_outcome(state_before_observation, observed_state_agent3, :agent3, action)
            wizard_candidates = [
                blue_wizards[j] for j in 1:length(blue_wizards)
                if state_probs_agent3[j, agent3_idx + 1] > 0.1
            ]
        end
        push!(trace, serialize_observation(observed_agent, action, observed_outcome))
        push!(events, Dict(
            "observation_index" => obs_idx,
            "observed_agent" => observed_agent,
            "action" => write_pddl(action),
            "interaction_outcome" => observed_outcome,
            "wizard_candidates_before" => candidates_before,
            "wizard_candidates_after" => serialize_wizards(wizard_candidates),
        ))
    end
    return trace, events, serialize_wizards(wizard_candidates)
end

"""
Find when an observed agent converges to a single blue-wizard candidate that
is relevant to agent1's own goal. Returns the first timestep where the
posterior support over blue wizards has size <= 1. If that never happens,
returns the maximum available timestep.
"""
function find_relevant_wizard_convergence_horizon(state_probs)
    max_t = size(state_probs, 2) - 1

    for t in 1:max_t
        candidate_count = count(prob -> prob > 0.1, state_probs[:, t + 1])
        if candidate_count <= 1
            return t
        end
    end

    return max_t
end

for (map_id, agent_goals) in sort(collect(metadata), by=x->x[1])
    map_start_time = time()
    println("\nProcessing map: $map_id")

    clear_planner_cache!()

    #--- Map-level setup (shared across both scenarios) ---#
    println("  [1] Loading domain and problem...")
    flush(stdout)

    # Load main domain and compile it (compilation is required for fast A* planning)
    domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))
    state = initstate(domain, problem)
    println("  [2] Compiling domain...")
    flush(stdout)
    domain, state = PDDL.compiled(domain, problem)
    println("  [3] Domain compiled. Running A* planner...")
    flush(stdout)

    #--- Agent-filtered domains (compile once per map, reuse across scenarios) ---#
    println("  [5] Loading agent-filtered domains...")
    flush(stdout)

    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    # Load and compile filtered problem for agent1
    domain_agent1 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent1 = joinpath(PROBLEM_DIR, ".temp_agent1_$(map_id).txt")
    if !isfile(temp_path_agent1)
        filtered_ascii_agent1 = filter_ascii_agents(ascii_content, :agent1)
        write(temp_path_agent1, filtered_ascii_agent1)
    end
    problem_agent1 = load_ascii_problem(temp_path_agent1)
    state_agent1 = initstate(domain_agent1, problem_agent1)
    println("  [5.5] Compiling agent1 domain...")
    flush(stdout)
    domain_agent1, state_agent1 = PDDL.compiled(domain_agent1, problem_agent1)

    # Check if agent1 even needs blue wizards (same answer for both scenarios)
    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    planner = AStarPlanner(GoalManhattan())
    plan_agent1 = planner(domain_agent1, state_agent1, problem_agent1.goal)
    println("  [4] A* planner done.")
    flush(stdout)
    agent1_needs_wizards = any(x -> x.name == :interact && x.args[end] in blue_wizards, plan_agent1)

    if !agent1_needs_wizards
        println("    Agent1 doesn't need blue wizards -> t=0 for both scenarios")
        for scenario in 1:2
            map_key = "$(map_id)_scenario$(scenario)"
            steps_dict[map_key] = Dict(
                "observations" => [],
                "agent2_count" => 0,
                "agent3_count" => 0,
                "t" => 0
            )
            next!(progress)
        end
        map_elapsed = time() - map_start_time
        map_times[map_id] = map_elapsed
        println("  Map completed in $(round(map_elapsed, digits=2))s")
        continue
    end

    # Load and compile filtered problem for agent2
    domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    if !isfile(temp_path_agent2)
        filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
        write(temp_path_agent2, filtered_ascii_agent2)
    end
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2 = initstate(domain_agent2, problem_agent2)
    println("  [6] Compiling agent2 domain...")
    flush(stdout)
    domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)

    # Load and compile filtered problem for agent3
    domain_agent3 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
    if !isfile(temp_path_agent3)
        filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
        write(temp_path_agent3, filtered_ascii_agent3)
    end
    problem_agent3 = load_ascii_problem(temp_path_agent3)
    state_agent3 = initstate(domain_agent3, problem_agent3)
    println("  [7] Compiling agent3 domain...")
    flush(stdout)
    domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)

    # Goals and beliefs (same across scenarios — only depends on map layout)
    println("  [8] Enumerating beliefs...")
    flush(stdout)
    goals_agent2, goal_names_agent2 = initialize_goals(state_agent2, :agent2)
    goals_agent3, goal_names_agent3 = initialize_goals(state_agent3, :agent3)

    initial_states_agent2, belief_probs_agent2, state_names_agent2 = enumerate_beliefs(state_agent2)
    initial_states_agent3, belief_probs_agent3, state_names_agent3 = enumerate_beliefs(state_agent3)
    println("  [9] Setup complete, starting scenarios...")
    flush(stdout)

    # Find current state ID for agent2 (using FILTERED state)
    s_id_agent2 = -1
    for s in 1:length(initial_states_agent2)
        if check_equal_state(state_agent2, initial_states_agent2[s])
            s_id_agent2 = s
            break
        end
    end

    # Find current state ID for agent3 (using FILTERED state)
    s_id_agent3 = -1
    for s in 1:length(initial_states_agent3)
        if check_equal_state(state_agent3, initial_states_agent3[s])
            s_id_agent3 = s
            break
        end
    end

    # Loop over both scenarios (only goal assignments differ)
    for scenario in 1:2
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"

        println("  Scenario $scenario")

        # Get which gems each agent wants in this scenario (exp4 metadata format)
        # exp4: agent_goals["agent2"][scenario]["gem"] = gem int
        #       agent_goals["agent2"][scenario]["type"] = "naive" or "actual"
        agent2_gem = agent_goals["agent2"][scenario]["gem"]  # X's goal
        agent2_type = agent_goals["agent2"][scenario]["type"]
        agent3_gem = agent_goals["agent3"][scenario]["gem"]  # Y's goal
        agent3_type = agent_goals["agent3"][scenario]["type"]
        planner = AStarPlanner(GoalManhattan())
        blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]
        blue_wizards_agent3 = [w for w in PDDL.get_objects(state_agent3, :wizard) if state_agent3[pddl"(iscolor $w blue)"]]
        observed_plan_agent2 = agent2_type == "naive" ?
            generate_naive_plan(domain_agent2, state_agent2, goals_agent2[agent2_gem], blue_wizards_agent2, :agent2, planner) :
            collect(planner(domain_agent2, state_agent2, goals_agent2[agent2_gem]))
        observed_plan_agent3 = agent3_type == "naive" ?
            generate_naive_plan(domain_agent3, state_agent3, goals_agent3[agent3_gem], blue_wizards_agent3, :agent3, planner) :
            collect(planner(domain_agent3, state_agent3, goals_agent3[agent3_gem]))

        # Load scenario-specific probabilities
        goal_probs_agent2 = goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
        state_probs_agent2 = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]

        goal_probs_agent3 = goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]
        state_probs_agent3 = state_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]

        #--- Find relevant-wizard convergence horizon independently for each agent ---#
        println("    [10] Finding horizon for agent2...")
        flush(stdout)
        T_agent2 = find_relevant_wizard_convergence_horizon(state_probs_agent2)
        println("    [11] Agent2 horizon: $T_agent2. Finding horizon for agent3...")
        flush(stdout)

        T_agent3 = find_relevant_wizard_convergence_horizon(state_probs_agent3)
        println("    [12] Agent3 horizon: $T_agent3")
        flush(stdout)

        agent2_horizon = realized_observation_horizon(T_agent2, observed_plan_agent2)
        agent3_horizon = realized_observation_horizon(T_agent3, observed_plan_agent3)
        observations, agent2_count, agent3_count = build_observation_prefix_until_one_converges(
            agent2_horizon, agent3_horizon
        )
        T = length(observations)

        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "t" => T
        )
        blue_wizards = [w for w in PDDL.get_objects(state_agent1, :wizard) if state_agent1[pddl"(iscolor $w blue)"]]
        observation_trace, observation_events, final_candidates = materialize_interleaved_observation_trace(
            map_key, observations, observed_plan_agent2, observed_plan_agent3,
            domain_agent2, state_agent2, domain_agent3, state_agent3,
            blue_wizards, state_probs_agent2, state_probs_agent3
        )
        replay_trace_dict[map_key] = Dict(
            "t" => T,
            "observations" => observation_trace,
            "observation_events" => observation_events,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "initial_candidates" => serialize_wizards(blue_wizards),
            "final_candidates" => final_candidates,
            "stop_reason" => "first_relevant_wizard_convergence",
        )

        scenario_elapsed = time() - scenario_start_time
        println("    Result: agent2=$agent2_count, agent3=$agent3_count, total=$T")
        println("    Time: $(round(scenario_elapsed, digits=2))s")

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
output_filename = "step_dict_mentalize_exp4_until_one_converges.json"
canonical_output_path = joinpath(OUTPUT_DIR, "step_dict_$(model_label).json")
write_json_to_paths((canonical_output_path,), steps_dict)

replay_trace_filename = "replay_trace_mentalize_exp4_until_one_converges.json"
canonical_replay_trace_path = joinpath(OUTPUT_DIR, "replay_trace_$(model_label).json")
write_json_to_paths((canonical_replay_trace_path,), replay_trace_dict)

println("\n=== Experiment Complete ===")
println("Results saved to: $canonical_output_path")
println("Replay trace saved to: $canonical_replay_trace_path")
