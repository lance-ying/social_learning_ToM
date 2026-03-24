using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JLD2, FileIO
using JSON
using ProgressMeter
using Statistics
# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "planners.jl"))

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_exp4_013026")
OUTPUT_DIR = joinpath(@__DIR__, "..", "..", "scripts", "experiments", "experiment_outputs")
mkpath(OUTPUT_DIR)

# Open debug log file (note: temperature variable is defined after ARGS parsing)
debug_log_file = nothing
debug_log_path = ""

function debug_println(args...)
    msg = join(string.(args), " ")
    println(msg)
    if debug_log_file !== nothing
        println(debug_log_file, msg)
        flush(debug_log_file)
    end
end

# Initialize debug log after temperature is parsed
function init_debug_log(temp::Float64)
    global debug_log_path, debug_log_file
    debug_log_path = joinpath(OUTPUT_DIR, "debug_output_inference_hyperparam_temp$(replace(string(temp), "." => "p")).txt")
    debug_log_file = open(debug_log_path, "w")
end

# #--- Initial Setup ---#

goal_probs_conditioned_dict = Dict()
state_probs_conditioned_dict = Dict()
possible_worlds = Dict()
agent_types_dict = Dict()  # NEW: stores agent type per (agent, map, scenario)

# Parse command-line arguments FIRST
# ARGS[1] = comma-separated map IDs (e.g., "sm211,sm221,sm341,sm611,sm421")
# ARGS[2] = output filename (e.g., "inference_exp4_hyperparam_temp0.3.jld2")
# ARGS[3] = temperature for action noise (e.g., "0.3", "0.4", "0.5")
selected_maps = isempty(ARGS) || isempty(ARGS[1]) ? String[] : split(ARGS[1], ",")
output_filename = length(ARGS) >= 2 && !isempty(ARGS[2]) ? ARGS[2] : "inference_exp4_013026.jld2"
temperature = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5

# Initialize debug log now that temperature is defined
init_debug_log(temperature)

problem_files = filter(f -> endswith(f, ".pddl") && !occursin("plan", f), readdir(PROBLEM_DIR))

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

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

# Helper function to plan to an adjacent position of a wizard (never on top)
function plan_to_wizard_location(
    domain::Domain, state::State, wizard_loc::Tuple{Int,Int},
    agent_name::Symbol, planner
)
    agent_loc = get_obj_loc(state, Const(agent_name))

    # Check if already adjacent (Manhattan distance = 1)
    agent_adjacent = (abs(agent_loc[1] - wizard_loc[1]) + abs(agent_loc[2] - wizard_loc[2]) == 1)

    if agent_adjacent
        return Term[]  # Already adjacent, no movement needed
    end

    # Try to plan to each adjacent position, return on first success (fast)
    for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        adj_pos = (wizard_loc[1] + dx, wizard_loc[2] + dy)
        adj_goal = PDDL.parse_pddl("(and (= (xloc $agent_name) $(adj_pos[1])) (= (yloc $agent_name) $(adj_pos[2])))")
        try
            plan = collect(planner(domain, state, adj_goal))
            if !isempty(plan)
                return plan  # Return immediately on first success
            end
        catch
            continue
        end
    end

    return Term[]
end

# Helper function to generate naive plan if needed
# If goal type is "naive" and optimal path requires blue wizards, visit blue wizards
# in order of distance until getting the blue key, then continue to goal
function generate_naive_plan_if_needed(
    domain::Domain, state::State, goal::Any, blue_wizards::Vector,
    goal_type::String, agent_name::Symbol
)
    planner_optimal = AStarPlanner(GoalManhattan())

    # Check if optimal path requires blue wizards
    plan_optimal = collect(planner_optimal(domain, state, goal))
    needs_wizards = any(x -> x.name == :interact && x.args[end] in blue_wizards, plan_optimal)

    if goal_type == "naive" && needs_wizards && !isempty(blue_wizards)
        # Find blue key
        blue_keys = [k for k in PDDL.get_objects(state, :key) if state[pddl"(iscolor $k blue)"]]
        if isempty(blue_keys)
            return plan_optimal
        end
        blue_key = blue_keys[1]

        # Visit blue wizards in order of distance until we get the key
        current_state = copy(state)
        full_naive_plan = Term[]
        visited_wizards = Set()

        while !current_state[pddl"(has $agent_name $blue_key)"] && length(visited_wizards) < length(blue_wizards)
            # Find closest unvisited blue wizard
            agent_loc = get_obj_loc(current_state, Const(agent_name))
            closest_wizard = nothing
            closest_wizard_loc = nothing
            min_dist = Inf

            for wizard in blue_wizards
                if wizard in visited_wizards
                    continue
                end
                wizard_loc = get_obj_loc(current_state, wizard)
                dist = sum(abs.(agent_loc .- wizard_loc))
                if dist < min_dist
                    min_dist = dist
                    closest_wizard = wizard
                    closest_wizard_loc = wizard_loc
                end
            end

            if closest_wizard === nothing
                break  # No more wizards to visit
            end

            # Plan to wizard and interact
            plan_to_wizard = plan_to_wizard_location(domain, current_state, closest_wizard_loc, agent_name, planner_optimal)
            append!(full_naive_plan, plan_to_wizard)

            # Execute plan to wizard
            for action in plan_to_wizard
                current_state = PDDL.execute(domain, current_state, action)
            end

            # Interact with wizard
            interact_action = PDDL.parse_pddl("(interact $agent_name $closest_wizard)")
            push!(full_naive_plan, interact_action)
            current_state = PDDL.execute(domain, current_state, interact_action)

            push!(visited_wizards, closest_wizard)
        end

        # If we still don't have the key, fall back to optimal
        if !current_state[pddl"(has $agent_name $blue_key)"]
            return plan_optimal
        end

        # Plan to goal from current location
        plan_to_goal = collect(planner_optimal(domain, current_state, goal))
        append!(full_naive_plan, plan_to_goal)

        return full_naive_plan
    else
        # Use optimal planning
        return plan_optimal
    end
end

# Loop over both agents (agent2=X, agent3=Y, agent1=M is not inferred)
agents_to_infer = ["agent2", "agent3"]

# Calculate total iterations for progress bar
# 2 agents × num_maps × 2 scenarios × 3 goals × 2 states
n_maps = length(metadata)
n_scenarios = 2
n_goals = 3
n_states = 2
total_iterations = length(agents_to_infer) * n_maps * n_scenarios * n_goals * n_states
progress = Progress(total_iterations, desc="Inference progress (temp=$temperature): ", showspeed=true)

# Track timing
agent_times = Dict()
total_start_time = time()

debug_println("=== Hyperparameter Search: temperature = $temperature ===\n")

for agent_name in agents_to_infer
    agent_sym = Symbol(agent_name)

    goal_probs_conditioned_dict[agent_name] = Dict()
    state_probs_conditioned_dict[agent_name] = Dict()
    possible_worlds[agent_name] = Dict()
    agent_types_dict[agent_name] = Dict()  # NEW: track agent types

    debug_println("\n=== Starting inference for $agent_name ===\n")
    agent_start_time = time()

    for (map_id, agent_goals) in metadata
        # if map_id != "sm221" && map_id != "sm311"
        #     continue
        # end
        if !isempty(selected_maps) && !(map_id in selected_maps)
            continue
        end
        debug_println("Processing map: $map_id for $agent_name")

        # Get goals for this agent: [{"gem": 1, "type": "naive"}, {"gem": 3, "type": "naive"}]
        goal_info_list = agent_goals[agent_name]

        goal_probs_conditioned_dict[agent_name][map_id] = Dict()
        state_probs_conditioned_dict[agent_name][map_id] = Dict()
        agent_types_dict[agent_name][map_id] = Dict()  # NEW: track types per scenario

        # Loop over all three scenarios
        for scenario in 1:2
            # if scenario != 3
            #     continue
            # end
            # Extract gem and type from metadata
            goal_info = goal_info_list[scenario]
            goal_gem_idx = goal_info["gem"]
            goal_type = goal_info["type"]  # "naive" or "actual"

            goal = PDDL.parse_pddl("(has $agent_sym gem$(goal_gem_idx))")
            debug_println("[DEBUG] Processing: agent=$agent_name, map=$map_id, scenario=$scenario, gem=$goal_gem_idx, type=$goal_type")
            debug_println("  Scenario $scenario: $(agent_name) -> gem$(goal_gem_idx) ($(goal_type))")

            goal_probs_conditioned_dict[agent_name][map_id][scenario] = Dict()
            state_probs_conditioned_dict[agent_name][map_id][scenario] = Dict()
            agent_types_dict[agent_name][map_id][scenario] = goal_type  # NEW: store agent type

            # Load domain
            domain = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))

            # Load problem with agent filtering for speed
            problem_path = joinpath(PROBLEM_DIR, "$map_id.pddl")
            txt_path = joinpath(PROBLEM_DIR, "$map_id.txt")

            if isfile(txt_path)
                include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl"))
                ascii_content = read(txt_path, String)
                filtered_ascii = filter_ascii_agents(ascii_content, agent_sym)
                temp_path = joinpath(PROBLEM_DIR, ".temp_$(agent_name)_$(map_id).txt")
                write(temp_path, filtered_ascii)
                problem = load_ascii_problem(temp_path)
            elseif isfile(problem_path)
                problem = load_problem(problem_path)
            else
                error("No problem file found for $map_id")
            end

            state = initstate(domain, problem)

            heuristic = GoalManhattan()
            planner = AStarPlanner(heuristic)

            domain, state = PDDL.compiled(domain, problem)

            #--- Goal Inference Setup ---#

            # Specify possible goals for this agent
            goals, goal_names = initialize_goals(state, agent_sym)

            # Define uniform prior over possible goals
            @gen function goal_prior()
                goal_id ~ uniform_discrete(1, length(goals))
                return Specification(goals[goal_id])
            end

            # Enumerate over possible initial states
            initial_states, belief_probs, state_names = enumerate_beliefs(state)

            # Store possible worlds (only need to do this once per map/agent)
            if scenario == 1
                possible_worlds[agent_name][map_id] = initial_states
            end

            # Define uniform prior over possible initial states
            @gen function state_prior()
                state_id ~ categorical(belief_probs)
                return initial_states[state_id]
            end

            # Construct iterator over initial choicemaps for stratified sampling
            init_state_addr = :init => :env => :state_id
            goal_addr = :init => :agent => :goal => :goal_id
            init_strata = choiceproduct(
                (goal_addr, 1:length(goals)),
                (init_state_addr, 1:length(initial_states))
            )

            # Get blue wizards (needed for NaivePlanner and ground truth plan generation)
            blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

            # Define planning algorithm - TYPE-AWARE
            # If agent is naive, observer uses NaivePlanner (expects naive behavior)
            # If agent is actual, observer uses RTHS (expects optimal behavior)
            heuristic = GoalManhattan()
            if goal_type == "naive"
                # Observer knows agent is naive - use NaivePlanner
                # Pass domain for dynamic action computation during inference
                # Constructor order: blue_wizards, agent_name, fallback_planner, domain
                planner = NaivePlanner(blue_wizards, agent_sym, AStarPlanner(heuristic), domain)
                println("    Using NaivePlanner for naive agent")
            else
                # Observer knows agent is actual/optimal - use RTHS
                planner = RTHS(heuristic, n_iters=1, max_nodes=2^15)
                println("    Using RTHS for actual agent")
            end

            # Define action noise model (use temperature from command-line argument)
            temperatures = temperature

            act_config = BoltzmannActConfig(temperatures)

            # Define agent configuration
            agent_config = AgentConfig(
                domain, planner;
                # Assume fixed goal over time
                goal_config = StaticGoalConfig(goal_prior),
                # Assume the agent refines its policy at every timestep
                replan_args = (
                    plan_at_init = true, # Plan at initial timestep
                    prob_refine = 1.0, # Probability of refining policy at each step
                    prob_replan = 0, # Probability of replanning at each timestep
                    rand_budget = false # Search budget is fixed everytime
                ),
                # Assume action noise
                act_config = act_config
            )

            # Configure world model with agent configuration and initial state prior
            world_config = WorldConfig(
                agent_config = agent_config,
                env_config = PDDLEnvConfig(domain, state_prior)
            )

            # FIRST PASS: Compute max plan length across all (g, i) pairs
            # This ensures all inference arrays have the same length
            max_plan_length = 0
            debug_println("  [DEBUG] Computing max plan length for scenario $scenario...")
            for g in 1:length(goals)
                for i in 1:length(initial_states)
                    state_i = initial_states[i]

                    if goal_type == "naive"
                        plan_temp = generate_naive_plan_if_needed(
                            domain, state_i, goals[g], blue_wizards, goal_type, agent_sym
                        )
                    else
                        planner_astar = AStarPlanner(GoalManhattan())
                        plan_temp = collect(planner_astar(domain, state_i, goals[g]))
                    end

                    max_plan_length = max(max_plan_length, length(collect(plan_temp)))
                end
            end
            debug_println("  [DEBUG] Max plan length for scenario $scenario: $max_plan_length")

            # SECOND PASS: Run inference for ALL goals (observer doesn't know agent's goal)
            # For each hypothesis (g, i), generate a plan using the agent's planning style
            # The observer knows the agent's TYPE (naive/actual) but not their goal
            for g in 1:length(goals)
                goal_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict()
                state_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict()

                for i in 1:length(initial_states)
                    state_i = initial_states[i]

                    # Generate plan for hypothesis (g, i) using the agent's known planning style
                    # - If agent is naive: use naive planning for hypothesis goal g
                    # - If agent is actual: use optimal planning for hypothesis goal g
                    # This matches inference_multi.jl but with type-aware planning
                    if goal_type == "naive"
                        # Agent is naive - use naive planning for hypothesis goal g
                        plan = generate_naive_plan_if_needed(
                            domain, state_i, goals[g], blue_wizards, goal_type, agent_sym
                        )
                    else
                        # Agent is actual/optimal - use optimal planning for hypothesis goal g
                        planner_astar = AStarPlanner(GoalManhattan())
                        plan = collect(planner_astar(domain, state_i, goals[g]))
                    end

                    debug_println("    Goal $g, State $i (type=$goal_type): $(length(collect(plan))) steps (max=$max_plan_length)")

                    t_obs_iter = act_choicemap_pairs(collect(plan))

                    # DEBUG: Print the plan being observed
                    debug_println("      Plan actions:")
                    for (step, act) in enumerate(plan)
                        debug_println("        Step $step: $act")
                    end

                    # Set up logging callback
                    local n_goals = length(goals)
                    local n_init_states = length(initial_states)

                    # DEBUG: Custom callback to inspect particle states
                    # Only print detailed debug for ground truth (g, i) pair
                    is_ground_truth = (g == goal_gem_idx)
                    plan_actions = collect(plan)  # Store for reference in callback
                    function debug_particle_states(t, pf)
                        # if !is_ground_truth
                        #     return nothing  # Skip non-ground-truth scenarios
                        # end
                        # if t > 10
                        #     return nothing  # Limit output to first 10 timesteps
                        # end
                        # observed_act = t > 0 ? plan_actions[t] : "none"
                        # println("      === Timestep $t (observed: $observed_act) ===")
                        # traces = Gen.get_traces(pf)
                        # for (idx, trace) in enumerate(traces)
                        #     # Get particle's goal and initial state
                        #     goal_id = trace[:init => :agent => :goal => :goal_id]
                        #     state_id = trace[:init => :env => :state_id]

                        #     # Get current state at this timestep
                        #     if t == 0
                        #         curr_state = trace[:init => :env]
                        #     else
                        #         curr_state = trace[:timestep => t => :env]
                        #     end

                        #     # Check if agent has key in this particle's current state
                        #     blue_keys = [k for k in PDDL.get_objects(curr_state, :key) if curr_state[pddl"(iscolor $k blue)"]]
                        #     has_key = false
                        #     if !isempty(blue_keys)
                        #         blue_key = blue_keys[1]
                        #         has_key = curr_state[pddl"(has $agent_sym $blue_key)"]
                        #     end

                        #     # Get agent location
                        #     agent_loc = (curr_state[pddl"(xloc $agent_sym)"], curr_state[pddl"(yloc $agent_sym)"])

                        #     # Get what action NaivePlanner would predict
                        #     if goal_type == "naive"
                        #         predicted_action = compute_naive_action(
                        #             domain, curr_state, goals[goal_id],
                        #             blue_wizards, agent_sym, AStarPlanner(GoalManhattan())
                        #         )
                        #     else
                        #         predicted_action = "N/A (not naive)"
                        #     end

                        #     # Get particle weight
                        #     weight = exp(Gen.get_score(trace))

                        #     println("        Particle $idx: goal=$goal_id, init_state=$state_id, has_key=$has_key, loc=$agent_loc, predicted=$predicted_action")
                        # end
                        return nothing
                    end

                    logger_cb = DataLoggerCallback(
                        t = (t, pf) -> t::Int,
                        goal_probs = pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64},
                        state_probs = pf -> probvec(pf, init_state_addr, 1:n_init_states)::Vector{Float64},
                        lml_est = pf -> log_ml_estimate(pf)::Float64,
                        debug = (t, pf) -> debug_particle_states(t, pf),
                    )
                    print_cb = PrintStatsCallback(
                        (goal_addr, 1:length(goals)),
                        (init_state_addr, 1:length(initial_states)),
                        header=("t\t" * join(goal_names, "\t") * "\t" *
                                join(state_names, "\t") * "\t")
                    )
                    callback = CombinedCallback(logger=logger_cb, print=print_cb)

                    # Configure SIPS particle filter
                    sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none)

                    # Run particle filter
                    n_samples = length(init_strata)
                    pf_state = sips(
                        n_samples,  t_obs_iter;
                        init_args = (init_strata=init_strata,),
                        callback = callback
                    );

                    # Extract goal probabilities
                    goal_probs_conditioned = reduce(hcat, callback.logger.data[:goal_probs])

                    # Extract initial state probabilities
                    state_probs_conditioned = reduce(hcat, callback.logger.data[:state_probs])

                    # PAD arrays to max_plan_length + 1 (timesteps 0 to max_plan_length)
                    # After the agent finishes their plan, probabilities stay constant
                    target_length = max_plan_length + 1
                    current_length = size(goal_probs_conditioned, 2)

                    if current_length < target_length
                        # Repeat the final column to fill remaining timesteps
                        final_goal_col = goal_probs_conditioned[:, end]
                        final_state_col = state_probs_conditioned[:, end]

                        for _ in current_length:(target_length-1)
                            goal_probs_conditioned = hcat(goal_probs_conditioned, final_goal_col)
                            state_probs_conditioned = hcat(state_probs_conditioned, final_state_col)
                        end

                        debug_println("      [DEBUG] Padded from $current_length to $target_length timesteps")
                    end

                    goal_probs_conditioned_dict[agent_name][map_id][scenario][g][i] = goal_probs_conditioned
                    state_probs_conditioned_dict[agent_name][map_id][scenario][g][i] = state_probs_conditioned

                    # DEBUG: Show sample probabilities being saved
                    if i == 1 && g <= 2
                        debug_println("      [DEBUG] Saved probs for (agent=$agent_name, map=$map_id, scenario=$scenario, g=$g, i=$i)")
                        debug_println("        goal_probs shape: $(size(goal_probs_conditioned))")
                        debug_println("        state_probs shape: $(size(state_probs_conditioned))")
                        if size(goal_probs_conditioned, 2) > 0
                            debug_println("        goal_probs[:, 1] = $(round.(goal_probs_conditioned[:, 1], digits=3))")
                        end
                        if size(state_probs_conditioned, 2) > 0
                            debug_println("        state_probs[:, 1] = $(round.(state_probs_conditioned[:, 1], digits=3))")
                        end
                    end

                    # Update progress bar
                    next!(progress)
                end
            end
            debug_println("  [DEBUG] Completed scenario $scenario")
        end
    end

    agent_elapsed = time() - agent_start_time
    agent_times[agent_name] = agent_elapsed
    println("  $agent_name completed in $(round(agent_elapsed, digits=2))s")
    debug_println("\n[DEBUG] Completed inference for $agent_name in $(round(agent_elapsed, digits=2))s\n")
end

# Finish progress bar
finish!(progress)

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
println("Average per agent: $(round(mean(values(agent_times)), digits=2))s")
println("Fastest agent: $(round(minimum(values(agent_times)), digits=2))s")
println("Slowest agent: $(round(maximum(values(agent_times)), digits=2))s")

data = Dict(
    "goal" => goal_probs_conditioned_dict,
    "state" => state_probs_conditioned_dict,
    "worlds" => possible_worlds,
    "agent_types" => agent_types_dict  # NEW: stores type per (agent, map, scenario)
)

# DEBUG: Print keys being saved
debug_println("\n[DEBUG] Data structure being saved:")
debug_println("  Keys: $(keys(data))")
debug_println("  goal_probs_conditioned_dict keys: $(keys(goal_probs_conditioned_dict))")
for agent in keys(goal_probs_conditioned_dict)
    debug_println("    $agent -> maps: $(keys(goal_probs_conditioned_dict[agent]))")
    for map_id in keys(goal_probs_conditioned_dict[agent])
        debug_println("      $map_id -> scenarios: $(keys(goal_probs_conditioned_dict[agent][map_id]))")
    end
end

output_path = joinpath(@__DIR__, "..", "..", "data", "inference", output_filename)
save(output_path, data)
debug_println("[DEBUG] Saved to: $output_path")
debug_println("[DEBUG] File size: $(round(filesize(output_path) / 1024, digits=2)) KB")

debug_println("\n=== Inference Complete ===")
debug_println("Saved to: $output_path")
debug_println("Data structure: data[agent_name][map_id][scenario][goal_id][state_id]")
debug_println("Agent types: data[\"agent_types\"][agent_name][map_id][scenario]")

close(debug_log_file)
println("\nDebug output saved to: $debug_log_path")
