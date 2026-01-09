using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JLD2, FileIO
using JSON
# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "plan_io.jl")) 
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "render.jl"))

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_exp4")

# #--- Initial Setup ---#

goal_probs_conditioned_dict = Dict()
state_probs_conditioned_dict = Dict()
possible_worlds = Dict()

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

# Helper function to plan to a wizard location (to or adjacent)
function plan_to_wizard_location(
    domain::Domain, state::State, wizard_loc::Tuple{Int,Int}, 
    agent_name::Symbol, planner
)
    agent_loc = get_obj_loc(state, Const(agent_name))
    
    # Check if already at or adjacent
    agent_at = (agent_loc[1] == wizard_loc[1] && agent_loc[2] == wizard_loc[2])
    agent_adjacent = (abs(agent_loc[1] - wizard_loc[1]) + abs(agent_loc[2] - wizard_loc[2]) == 1)
    
    if agent_at || agent_adjacent
        return Term[]
    end
    
    # Try to get to wizard location
    wizard_goal = PDDL.parse_pddl("(and (= (xloc $agent_name) $(wizard_loc[1])) (= (yloc $agent_name) $(wizard_loc[2])))")
    plan = collect(planner(domain, state, wizard_goal))
    
    # If that doesn't work, try adjacent positions
    if isempty(plan)
        for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
            adj_pos = (wizard_loc[1] + dx, wizard_loc[2] + dy)
            adj_goal = PDDL.parse_pddl("(and (= (xloc $agent_name) $(adj_pos[1])) (= (yloc $agent_name) $(adj_pos[2])))")
            try
                plan = collect(planner(domain, state, adj_goal))
                if !isempty(plan)
                    return plan
                end
            catch
                continue
            end
        end
    end
    
    return plan
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

for agent_name in agents_to_infer
    agent_sym = Symbol(agent_name)
    
    goal_probs_conditioned_dict[agent_name] = Dict()
    state_probs_conditioned_dict[agent_name] = Dict()
    possible_worlds[agent_name] = Dict()
    
    println("\n=== Running inference for $agent_name ===\n")
    
    for (map_id, agent_goals) in metadata
        println("Processing map: $map_id for $agent_name")
        
        # Get goals for this agent: [{"gem": 1, "type": "naive"}, {"gem": 3, "type": "naive"}]
        goal_info_list = agent_goals[agent_name]
        
        goal_probs_conditioned_dict[agent_name][map_id] = Dict()
        state_probs_conditioned_dict[agent_name][map_id] = Dict()
        
        # Loop over both scenarios
        for scenario in 1:2
            # Extract gem and type from metadata
            goal_info = goal_info_list[scenario]
            goal_gem_idx = goal_info["gem"]
            goal_type = goal_info["type"]  # "naive" or "actual"
            
            goal = PDDL.parse_pddl("(has $agent_sym gem$(goal_gem_idx))")
            println("  Scenario $scenario: $(agent_name) -> gem$(goal_gem_idx) ($(goal_type))")
            
            goal_probs_conditioned_dict[agent_name][map_id][scenario] = Dict()
            state_probs_conditioned_dict[agent_name][map_id][scenario] = Dict()

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

            # Define planning algorithm (RTHS uses optimal planning - observer doesn't know agent is naive)
            heuristic = GoalManhattan()
            planner = RTHS(heuristic, n_iters=1, max_nodes=2^15)

            # Define action noise model
            temperatures = 0.5

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

            # Get blue wizards for naive planning check
            blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

            # Run inference for ALL goals (observer doesn't know agent's goal)
            for g in 1:length(goals)
                goal_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict()
                state_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict()
                
                for i in 1:length(initial_states)
                    state_i = initial_states[i]
                    
                    # Generate ground truth plan
                    # If this goal (g) matches the agent's actual goal (goal_gem_idx) and goal type is naive,
                    # use naive planning. Otherwise use optimal planning.
                    if g == goal_gem_idx && goal_type == "naive"
                        # Use naive planning for the actual goal when agent is naive
                        plan = generate_naive_plan_if_needed(
                            domain, state_i, goals[g], blue_wizards, goal_type, agent_sym
                        )
                    else
                        # Use optimal planning for all other cases
                        planner_astar = AStarPlanner(GoalManhattan())
                        plan = collect(planner_astar(domain, state_i, goals[g]))
                    end

                    println("    Goal $g, State $i: $(length(collect(plan))) steps")

                    t_obs_iter = act_choicemap_pairs(collect(plan))

                    # Set up logging callback
                    n_goals = length(goals)
                    n_init_states = length(initial_states)
                    logger_cb = DataLoggerCallback(
                        t = (t, pf) -> t::Int,
                        goal_probs = pf -> probvec(pf, goal_addr, 1:n_goals)::Vector{Float64},
                        state_probs = pf -> probvec(pf, init_state_addr, 1:n_init_states)::Vector{Float64},
                        lml_est = pf -> log_ml_estimate(pf)::Float64,
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

                    goal_probs_conditioned_dict[agent_name][map_id][scenario][g][i] = goal_probs_conditioned
                    state_probs_conditioned_dict[agent_name][map_id][scenario][g][i] = state_probs_conditioned
                end
            end
        end
    end
end



data = Dict("goal" => goal_probs_conditioned_dict, "state" => state_probs_conditioned_dict, "worlds" => possible_worlds)

save("inference_data_exp4.jld2", data)

println("\n=== Inference Complete ===")
println("Saved to: inference_data_exp4.jld2")
println("Data structure: data[agent_name][map_id][scenario][goal_id][state_id]")

