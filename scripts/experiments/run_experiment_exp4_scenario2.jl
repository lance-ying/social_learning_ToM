using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using ProgressMeter
using Base.Threads
using Statistics
# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "planners.jl"))

function wizard_candidate_cache_key(wizards)
    return join(sort!(string.(wizards)), "|")
end

serialize_wizards(wizards) = sort(string.(wizards))
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

# Helper function to filter ASCII map to only include specified agent
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

# Parse command-line arguments or use defaults
if length(ARGS) >= 3
    experiment_id = ARGS[1]
    inference_file = ARGS[2]
    output_prefix = ARGS[3]
else
    # Default values for standalone execution
    experiment_id = "exp4_012626"
    inference_file = "inference_data_exp4_012626.jld2"
    output_prefix = "steps_dict_exp4_012626"
end
selected_maps = length(ARGS) >= 4 && !isempty(ARGS[4]) ? split(ARGS[4], ",") : String[]

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_$experiment_id")
OUTPUT_DIR = joinpath(@__DIR__, "experiment_outputs")
mkpath(OUTPUT_DIR)  # Create output directory if it doesn't exist
CANONICAL_OUTPUT_DIR = joinpath(@__DIR__, "outputs", "exp4", "scenario2")
mkpath(CANONICAL_OUTPUT_DIR)

# Open debug log file
debug_log_path = joinpath(OUTPUT_DIR, "$(output_prefix)_scenario2_debug.txt")
debug_log_file = open(debug_log_path, "w")
function debug_println(args...)
    msg = join(string.(args), " ")
    println(msg)
    println(debug_log_file, msg)
    flush(debug_log_file)
end

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()
replay_trace_dict = Dict()

# Load inference data for both agents (agent2=X, agent3=Y)
data = load(joinpath(@__DIR__, "..", "..", "data", "inference", inference_file))
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]
possible_worlds = data["worlds"]

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * length(possible_worlds) # maps × 3 scenarios
progress = Progress(total_iterations, desc="Processing exp4: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, agent_goals) in metadata

    # if map_id != "sm221" && map_id != "sm311"
    #     continue
    # end
    if !isempty(selected_maps) && !(map_id in selected_maps)
        continue
    end

    map_start_time = time()
    debug_println("\nProcessing map: $map_id")

    # === LOAD AND COMPILE DOMAINS ONCE PER MAP (outside scenario loop) ===
    # This prevents PDDL.compiled() type conflicts between scenarios

    domain = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
    problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))

    # Initialize and compile reference state for the FULL problem
    state_init = initstate(domain, problem)
    state_render = copy(state_init)
    domain, state_init = PDDL.compiled(domain, problem)

    # Load FILTERED problem for agent2
    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    domain_agent1 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
    temp_path_agent1 = joinpath(PROBLEM_DIR, ".temp_agent1_$(map_id).txt")
    if !isfile(temp_path_agent1)
        filtered_ascii_agent1 = filter_ascii_agents(ascii_content, :agent1)
        write(temp_path_agent1, filtered_ascii_agent1)
    end
    problem_agent1 = load_ascii_problem(temp_path_agent1)
    state_agent1_init = initstate(domain_agent1, problem_agent1)
    state_render_agent1 = copy(state_agent1_init)
    domain_agent1, state_agent1_init = PDDL.compiled(domain_agent1, problem_agent1)

    domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    if !isfile(temp_path_agent2)
        filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
        write(temp_path_agent2, filtered_ascii_agent2)
    end
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2_init = initstate(domain_agent2, problem_agent2)
    domain_agent2, state_agent2_init = PDDL.compiled(domain_agent2, problem_agent2)

    # Load FILTERED problem for agent3
    domain_agent3 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
    temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
    if !isfile(temp_path_agent3)
        filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
        write(temp_path_agent3, filtered_ascii_agent3)
    end
    problem_agent3 = load_ascii_problem(temp_path_agent3)
    state_agent3_init = initstate(domain_agent3, problem_agent3)
    domain_agent3, state_agent3_init = PDDL.compiled(domain_agent3, problem_agent3)

    # Specify possible goals for each agent (from FILTERED states)
    goals_agent2, goal_names_agent2 = initialize_goals(state_agent2_init, :agent2)
    goals_agent3, goal_names_agent3 = initialize_goals(state_agent3_init, :agent3)

    # Enumerate over possible initial states (from FILTERED states)
    initial_states_agent2, belief_probs_agent2, state_names_agent2 = enumerate_beliefs(state_agent2_init)
    initial_states_agent3, belief_probs_agent3, state_names_agent3 = enumerate_beliefs(state_agent3_init)

    # === NOW LOOP OVER SCENARIOS (reusing compiled domains) ===
    for scenario in 1:2
        if scenario !== 2
            continue
        end
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"
        debug_println("  Scenario $scenario")

        # Clear planner cache for each scenario to avoid memory issues
        clear_planner_cache!()

        # Get which gems each agent wants in this scenario (with type: naive/actual)
        agent2_goal_info = agent_goals["agent2"][scenario]
        agent3_goal_info = agent_goals["agent3"][scenario]

        agent2_gem = agent2_goal_info["gem"]
        agent2_type = agent2_goal_info["type"]
        agent3_gem = agent3_goal_info["gem"]
        agent3_type = agent3_goal_info["type"]

        debug_println("    agent2 (X) -> gem$(agent2_gem) ($(agent2_type)), agent3 (Y) -> gem$(agent3_gem) ($(agent3_type))")

        # Use fresh copies of the compiled states for this scenario
        state = copy(state_init)
        state_agent1 = copy(state_agent1_init)
        state_agent2 = copy(state_agent2_init)
        state_agent3 = copy(state_agent3_init)

        t = 0


        # Track observations
        observations = []
        agent2_count = 0
        agent3_count = 0
        observation_events = Any[]
        stop_reason = ""

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
        wizard_candicates = blue_wizards

        # Pre-compute blue wizards for filtered states (used in Q computation)
        blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]
        blue_wizards_agent3 = [w for w in PDDL.get_objects(state_agent3, :wizard) if state_agent3[pddl"(iscolor $w blue)"]]

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

        # Load initial probabilities from scenario-specific goals
        debug_println("    Loading probability data: s_id_agent2=$s_id_agent2, s_id_agent3=$s_id_agent3")
        debug_println("    agent2_gem=$agent2_gem, agent3_gem=$agent3_gem")

        goal_probs_agent2 = goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
        state_probs_agent2 = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
        debug_println("    Loaded agent2 probs: goal_probs size=$(size(goal_probs_agent2)), state_probs size=$(size(state_probs_agent2))")

        goal_probs_agent3 = goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]
        state_probs_agent3 = state_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]
        debug_println("    Loaded agent3 probs: goal_probs size=$(size(goal_probs_agent3)), state_probs size=$(size(state_probs_agent3))")

        # Pre-compute state copy and planner (moved outside loop for efficiency)
        new_state = copy(state_render_agent1)
        planner = AStarPlanner(GoalManhattan())
        exploration_cost_cache = Dict{String, Float64}()
        exploration_cost_cache_lock = ReentrantLock()
        function cached_exploration_cost(wizards)
            key = wizard_candidate_cache_key(wizards)
            lock(exploration_cost_cache_lock) do
                if haskey(exploration_cost_cache, key)
                    return exploration_cost_cache[key]
                end
            end
            cost = estimate_self_exploration_cost(domain_render, new_state, problem_agent1.goal, wizards, action_cost)
            lock(exploration_cost_cache_lock) do
                exploration_cost_cache[key] = cost
            end
            return cost
        end

        # Check if agent1's plan requires blue wizards
        debug_println("    Planning for agent1...")
        debug_println("    problem.goal = $(problem_agent1.goal)")
        debug_println("    agent1 location = ($(state[pddl"(xloc agent1)"]), $(state[pddl"(yloc agent1)"]))")
        # Note: AStarPlanner is used here because we need the actual plan (list of actions)
        # to check for wizard interactions. First run in a fresh Julia session may be slow
        # due to JIT compilation, but subsequent runs are fast.
        plan_agent1 = planner(domain_agent1, state_agent1, problem_agent1.goal)
        debug_println("    Agent1 plan computed, length=$(length(collect(plan_agent1)))")
        agent1_needs_wizards = any(x-> x.name == :interact && x.args[end] in blue_wizards, plan_agent1)
        if !agent1_needs_wizards
            print("t=", 0)
            steps_dict[map_key] = Dict(
                "t" => 0,
                "observations" => [],
                "agent2_count" => 0,
                "agent3_count" => 0
            )
            stop_reason = "agent1_no_blue_wizard_needed"
            replay_trace_dict[map_key] = Dict(
                "t" => 0,
                "observations" => Any[],
                "observation_events" => observation_events,
                "agent2_count" => 0,
                "agent3_count" => 0,
                "initial_candidates" => serialize_wizards(blue_wizards),
                "final_candidates" => serialize_wizards(wizard_candicates),
                "stop_reason" => stop_reason,
            )
            next!(progress)
            continue
        end

        observed_plan_agent2 = agent2_type == "naive" ?
            generate_naive_plan(domain_agent2, state_agent2, goals_agent2[agent2_gem], blue_wizards_agent2, :agent2, planner) :
            collect(planner(domain_agent2, state_agent2, goals_agent2[agent2_gem]))
        observed_plan_agent3 = agent3_type == "naive" ?
            generate_naive_plan(domain_agent3, state_agent3, goals_agent3[agent3_gem], blue_wizards_agent3, :agent3, planner) :
            collect(planner(domain_agent3, state_agent3, goals_agent3[agent3_gem]))
        observed_state_agent2 = copy(state_agent2)
        observed_state_agent3 = copy(state_agent3)

        # Note: We don't skip observations here - let Q-values determine if observing
        # is worthwhile. If agents don't need blue wizards, their Q-values will be
        # high and they won't be chosen.

        while !PDDL.satisfy(domain_agent1, state_agent1, problem_agent1.goal)
            q_start_time = time()

            # Per-agent inference tables are indexed by that agent's own observation count.
            max_t_agent2 = size(goal_probs_agent2, 2) - 1
            max_t_agent3 = size(goal_probs_agent3, 2) - 1
            can_observe_agent2 = agent2_count < max_t_agent2 && agent2_count < length(observed_plan_agent2)
            can_observe_agent3 = agent3_count < max_t_agent3 && agent3_count < length(observed_plan_agent3)

            if !can_observe_agent2 && !can_observe_agent3
                steps_dict[map_key] = Dict(
                    "t" => t,
                    "observations" => observations,
                    "agent2_count" => agent2_count,
                    "agent3_count" => agent3_count
                )
                stop_reason = "observation_horizon_exhausted"
                break
            end

            timestep_agent2 = agent2_count + 1
            timestep_agent3 = agent3_count + 1

            # Parallelize Q computation for both agents
            task_agent2 = Threads.@spawn begin
                # Compute Q_observe for agent2 (X)
                Q_observe_agent2 = Inf
                total_probs_agent2 = 0.0
                can_observe_agent2 || return (Q_observe_agent2, total_probs_agent2, Any[])
                Q_observe_agent2 = 0.0

                # Pre-cache the nested dictionary access for agent2 to avoid repeated lookups
                agent2_dict = state_probs_conditioned_dict["agent2"][map_id][scenario]
                agent2_goal_dict = goal_probs_conditioned_dict["agent2"][map_id][scenario]

                # Debug: track T values and wizard candidates for each (g,i) pair
                debug_entries_agent2 = []

                # Loop over ALL possible goals (observer doesn't know which goal agent has)
                for g in 1:length(goals_agent2)
                if goal_probs_agent2[g, timestep_agent2] < 0.1
                    continue
                end

                for i in 1:length(initial_states_agent2)
                    if state_probs_agent2[i, timestep_agent2] < 0.1
                        continue
                    end

                    joint_prob = goal_probs_agent2[g, timestep_agent2] * state_probs_agent2[i, timestep_agent2]

                    # Cache the (g,i) dictionary access
                    state_probs_gi = agent2_dict[g][i]
                    goal_probs_gi = agent2_goal_dict[g][i]

                    # Find T = timestep when state_probs converge
                    T = -1
                    T_from_state = false
                    for val in 1:size(state_probs_gi, 2)
                        if any(x -> x>0.95, state_probs_gi[:,val])
                            T = val
                            T_from_state = true
                            break
                        end
                    end

                    if T == -1
                        for val in 1:size(goal_probs_gi, 2)
                            if any(x -> x<0.1, goal_probs_gi[:,val])
                                T = val
                                break
                            end
                        end
                    end

                    # Validate T is within bounds
                    max_T_state = size(state_probs_gi, 2)

                    if T == -1 || T > max_T_state || !T_from_state
                        # If T is invalid or came from goal_probs (not state_probs),
                        # observing this agent won't help identify the wizard
                        new_wizard_candicates = copy(blue_wizards_agent2)
                    else
                        # Get blue wizards from pre-computed list at convergence time
                        new_wizard_candicates = []
                        for j in 1:length(blue_wizards_agent2)
                            if state_probs_gi[j, T] > 0.1
                                push!(new_wizard_candicates, blue_wizards_agent2[j])
                            end
                        end
                    end

                    # Check if this goal/state combination is consistent with learned wizard_candicates
                    # If wizard_candicates has been filtered (not empty), verify compatibility
                    if !isempty(wizard_candicates)
                        is_compatible = false
                        for wiz in new_wizard_candicates
                            if wiz in wizard_candicates
                                is_compatible = true
                                break
                            end
                        end

                        if !is_compatible
                            continue
                        end
                    end

                    # Use problem.goal (agent1's goal) for cost estimation, matching exp3_debug pattern
                    Q_T = cached_exploration_cost(new_wizard_candicates)
                    # Use REMAINING time to convergence, not total time
                    # This encourages continuing with an agent we've already started observing
                    remaining_T = T_from_state ? max(T - timestep_agent2 + 1, 1) : max(T, 1)
                    obs_cost = action_cost[:observe] * remaining_T
                    total_cost = Q_T + obs_cost
                    contribution = goal_probs_agent2[g, timestep_agent2] * state_probs_agent2[i, timestep_agent2] * total_cost
                    Q_observe_agent2 += contribution
                    total_probs_agent2 += goal_probs_agent2[g, timestep_agent2] * state_probs_agent2[i, timestep_agent2]

                    # Store debug info
                    push!(debug_entries_agent2, (g=g, i=i, T=T, n_wiz=length(new_wizard_candicates), Q_T=Q_T, obs_cost=obs_cost, prob=joint_prob))
                end
                end

                if total_probs_agent2 > 0
                    Q_observe_agent2 /= total_probs_agent2
                else
                    Q_observe_agent2 = Inf  # If no valid probabilities, set to infinity
                    debug_println("    WARNING: total_probs_agent2 = 0 at t=$t")
                end
                (Q_observe_agent2, total_probs_agent2, debug_entries_agent2)
            end

            task_agent3 = Threads.@spawn begin
                # Compute Q_observe for agent3 (Y)
                Q_observe_agent3 = Inf
                total_probs_agent3 = 0.0
                can_observe_agent3 || return (Q_observe_agent3, total_probs_agent3, Any[])
                Q_observe_agent3 = 0.0

                # Pre-cache the nested dictionary access for agent3 to avoid repeated lookups
                agent3_dict = state_probs_conditioned_dict["agent3"][map_id][scenario]
                agent3_goal_dict = goal_probs_conditioned_dict["agent3"][map_id][scenario]

                # Debug: track T values and wizard candidates for each (g,i) pair
                debug_entries_agent3 = []

                # Debug: Check initial probabilities for agent3
                if t == 0 && agent3_count == 0
                    debug_println("    agent3 initial goal_probs: $(round.(goal_probs_agent3[:, 1], digits=3))")
                    debug_println("    agent3 initial state_probs: $(round.(state_probs_agent3[:, 1], digits=3))")
                end

                # Loop over ALL possible goals (observer doesn't know which goal agent has)
                for g in 1:length(goals_agent3)
                if goal_probs_agent3[g, timestep_agent3] < 0.1
                    continue
                end

                for i in 1:length(initial_states_agent3)
                    if state_probs_agent3[i, timestep_agent3] < 0.1
                        continue
                    end

                    joint_prob = goal_probs_agent3[g, timestep_agent3] * state_probs_agent3[i, timestep_agent3]

                    # Cache the (g,i) dictionary access
                    state_probs_gi = agent3_dict[g][i]
                    goal_probs_gi = agent3_goal_dict[g][i]

                    # Find T = timestep when state_probs converge
                    T = -1
                    T_from_state = false
                    for val in 1:size(state_probs_gi, 2)
                        if any(x -> x>0.95, state_probs_gi[:,val])
                            T = val
                            T_from_state = true
                            break
                        end
                    end

                    if T == -1
                        for val in 1:size(goal_probs_gi, 2)
                            if any(x -> x<0.1, goal_probs_gi[:,val])
                                T = val
                                break
                            end
                        end
                    end

                    # Validate T is within bounds
                    max_T_state = size(state_probs_gi, 2)

                    if T == -1 || T > max_T_state || !T_from_state
                        # If T is invalid or came from goal_probs (not state_probs),
                        # observing this agent won't help identify the wizard
                        new_wizard_candicates = copy(blue_wizards_agent3)
                    else
                        # Get blue wizards from pre-computed list at convergence time
                        new_wizard_candicates = []
                        for j in 1:length(blue_wizards_agent3)
                            if state_probs_gi[j, T] > 0.1
                                push!(new_wizard_candicates, blue_wizards_agent3[j])
                            end
                        end
                    end

                    # Check if this goal/state combination is consistent with learned wizard_candicates
                    # If wizard_candicates has been filtered (not empty), verify compatibility
                    if !isempty(wizard_candicates)
                        is_compatible = false
                        for wiz in new_wizard_candicates
                            if wiz in wizard_candicates
                                is_compatible = true
                                break
                            end
                        end

                        if !is_compatible
                            continue
                        end
                    end

                    # Use problem.goal (agent1's goal) for cost estimation, matching exp3_debug pattern
                    Q_T = cached_exploration_cost(new_wizard_candicates)
                    # Use REMAINING time to convergence, not total time
                    # This encourages continuing with an agent we've already started observing
                    remaining_T = T_from_state ? max(T - timestep_agent3 + 1, 1) : max(T, 1)
                    obs_cost = action_cost[:observe] * remaining_T
                    total_cost = Q_T + obs_cost
                    contribution = goal_probs_agent3[g, timestep_agent3] * state_probs_agent3[i, timestep_agent3] * total_cost
                    Q_observe_agent3 += contribution
                    total_probs_agent3 += goal_probs_agent3[g, timestep_agent3] * state_probs_agent3[i, timestep_agent3]

                    # Store debug info
                    push!(debug_entries_agent3, (g=g, i=i, T=T, n_wiz=length(new_wizard_candicates), Q_T=Q_T, obs_cost=obs_cost, prob=joint_prob))
                end
                end

                if total_probs_agent3 > 0
                    Q_observe_agent3 /= total_probs_agent3
                else
                    Q_observe_agent3 = Inf  # If no valid probabilities, set to infinity
                    debug_println("    WARNING: total_probs_agent3 = 0 at t=$t")
                end
                (Q_observe_agent3, total_probs_agent3, debug_entries_agent3)
            end

            # Wait for both parallel tasks to complete
            (Q_observe_agent2, total_probs_agent2, debug_entries_agent2) = fetch(task_agent2)
            (Q_observe_agent3, total_probs_agent3, debug_entries_agent3) = fetch(task_agent3)

            # Compute Q_not_observe
            Q_not_observe = cached_exploration_cost(wizard_candicates)

            # Debug: Print Q-values for understanding decision
            debug_println("    t=$t: Q_observe_agent2=$(round(Q_observe_agent2, digits=2)), Q_observe_agent3=$(round(Q_observe_agent3, digits=2)), Q_not_observe=$(round(Q_not_observe, digits=2))")
            debug_println("    wizard_candidates: $(length(wizard_candicates))")
            debug_println("    agent2_count=$agent2_count, agent3_count=$agent3_count")
            debug_println("    timestep_agent2=$timestep_agent2, timestep_agent3=$timestep_agent3")

            # Detailed debug: show breakdown for each agent
            debug_println("    === agent2 ($(agent2_type)) breakdown ===")
            for entry in debug_entries_agent2
                debug_println("      g=$(entry.g), i=$(entry.i): T=$(entry.T), n_wiz=$(entry.n_wiz), Q_T=$(round(entry.Q_T, digits=1)), obs_cost=$(round(entry.obs_cost, digits=1)), prob=$(round(entry.prob, digits=3))")
            end
            debug_println("    === agent3 ($(agent3_type)) breakdown ===")
            for entry in debug_entries_agent3
                debug_println("      g=$(entry.g), i=$(entry.i): T=$(entry.T), n_wiz=$(entry.n_wiz), Q_T=$(round(entry.Q_T, digits=1)), obs_cost=$(round(entry.obs_cost, digits=1)), prob=$(round(entry.prob, digits=3))")
            end

            # Take argmin to decide which action
            q_values = [Q_observe_agent2, Q_observe_agent3, Q_not_observe]
            best_action_idx = argmin(q_values)

            if best_action_idx == 1
                # Observe agent2 (X) - it has the lowest Q-value
                candidates_before = serialize_wizards(wizard_candicates)
                observed_action = observed_plan_agent2[agent2_count + 1]
                state_before_observation = copy(observed_state_agent2)
                observed_state_agent2 = PDDL.execute(domain_agent2, observed_state_agent2, observed_action)
                observed_outcome = interaction_outcome(state_before_observation, observed_state_agent2, :agent2, observed_action)
                push!(observations, "agent2")
                agent2_count += 1
                t += 1
                wizard_candicates = []

                # Simple wizard update: use state_probs directly (matching exp3_debug approach)
                for j in 1:length(blue_wizards)
                    if state_probs_agent2[j, agent2_count + 1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
                push!(observation_events, Dict(
                    "observation_index" => t,
                    "observed_agent" => "agent2",
                    "action" => write_pddl(observed_action),
                    "interaction_outcome" => observed_outcome,
                    "q_observe_agent2" => Q_observe_agent2,
                    "q_observe_agent3" => Q_observe_agent3,
                    "q_not_observe" => Q_not_observe,
                    "agent2_count_after" => agent2_count,
                    "agent3_count_after" => agent3_count,
                    "wizard_candidates_before" => candidates_before,
                    "wizard_candidates_after" => serialize_wizards(wizard_candicates),
                ))
            elseif best_action_idx == 2
                # Observe agent3 (Y) - it has the lowest Q-value
                candidates_before = serialize_wizards(wizard_candicates)
                observed_action = observed_plan_agent3[agent3_count + 1]
                state_before_observation = copy(observed_state_agent3)
                observed_state_agent3 = PDDL.execute(domain_agent3, observed_state_agent3, observed_action)
                observed_outcome = interaction_outcome(state_before_observation, observed_state_agent3, :agent3, observed_action)
                push!(observations, "agent3")
                agent3_count += 1
                t += 1
                wizard_candicates = []

                # Simple wizard update: use state_probs directly (matching exp3_debug approach)
                for j in 1:length(blue_wizards)
                    if state_probs_agent3[j, agent3_count + 1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
                push!(observation_events, Dict(
                    "observation_index" => t,
                    "observed_agent" => "agent3",
                    "action" => write_pddl(observed_action),
                    "interaction_outcome" => observed_outcome,
                    "q_observe_agent2" => Q_observe_agent2,
                    "q_observe_agent3" => Q_observe_agent3,
                    "q_not_observe" => Q_not_observe,
                    "agent2_count_after" => agent2_count,
                    "agent3_count_after" => agent3_count,
                    "wizard_candidates_before" => candidates_before,
                    "wizard_candidates_after" => serialize_wizards(wizard_candicates),
                ))
            else
                # best_action_idx == 3: Not observing has the lowest Q-value
                steps_dict[map_key] = Dict(
                    "t" => t,
                    "observations" => observations,
                    "agent2_count" => agent2_count,
                    "agent3_count" => agent3_count
                )
                stop_reason = "q_not_observe_better"
                break
            end
        end

        if !haskey(steps_dict, map_key)
            steps_dict[map_key] = Dict(
                "t" => t,
                "observations" => observations,
                "agent2_count" => agent2_count,
                "agent3_count" => agent3_count
            )
            stop_reason = "goal_satisfied"
        end
        replay_trace_dict[map_key] = Dict(
            "t" => steps_dict[map_key]["t"],
            "observations" => [serialize_observation(event["observed_agent"], parse_pddl(event["action"]), get(event, "interaction_outcome", "none")) for event in observation_events],
            "observation_events" => observation_events,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "initial_candidates" => serialize_wizards(blue_wizards),
            "final_candidates" => serialize_wizards(wizard_candicates),
            "stop_reason" => stop_reason,
        )

        # Update progress bar after each scenario
        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        println("    Scenario completed in $(round(scenario_elapsed, digits=2))s")
        println("    Cache stats: $(cache_stats.hits) hits, $(cache_stats.misses) misses, $(round(cache_stats.hit_rate * 100, digits=1))% hit rate")
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
println("Fastest map: $(round(minimum(values(map_times)), digits=2))s")
println("Slowest map: $(round(maximum(values(map_times)), digits=2))s")


output_filename = "$(output_prefix)_scenario2.json"
output_path = joinpath(OUTPUT_DIR, output_filename)
open(output_path, "w") do io
    JSON.print(io, steps_dict, 4)
end
open(joinpath(CANONICAL_OUTPUT_DIR, "steps_dict.json"), "w") do io
    JSON.print(io, steps_dict, 4)
end

replay_trace_filename = "$(output_prefix)_scenario2_replay_trace.json"
replay_trace_path = joinpath(OUTPUT_DIR, replay_trace_filename)
open(replay_trace_path, "w") do io
    JSON.print(io, replay_trace_dict, 4)
end
open(joinpath(CANONICAL_OUTPUT_DIR, "replay_trace.json"), "w") do io
    JSON.print(io, replay_trace_dict, 4)
end

debug_println("\n=== Experiment Complete ===")
debug_println("Results saved to: $output_path")
debug_println("Replay trace saved to: $replay_trace_path")
close(debug_log_file)
println("\nResults saved to: $output_path")
println("Replay trace saved to: $replay_trace_path")
println("Canonical scenario outputs: $CANONICAL_OUTPUT_DIR")
println("Debug output saved to: $debug_log_path")
