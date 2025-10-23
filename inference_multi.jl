using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JLD2, FileIO
using JSON
# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/render.jl")

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_exp3")

# #--- Initial Setup ---#

goal_probs_conditioned_dict = Dict()
state_probs_conditioned_dict = Dict()
possible_worlds = Dict()

problem_files = filter(f -> endswith(f, ".pddl") && !occursin("plan", f), readdir(PROBLEM_DIR))

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Loop over both agents (agent2=Z, agent3=X, agent1=M is not inferred)
agents_to_infer = ["agent2", "agent3"]

for agent_name in agents_to_infer
    agent_sym = Symbol(agent_name)
    
    goal_probs_conditioned_dict[agent_name] = Dict()
    state_probs_conditioned_dict[agent_name] = Dict()
    possible_worlds[agent_name] = Dict()
    
    println("\n=== Running inference for $agent_name ===\n")
    
    for (map_id, agent_goals) in metadata
        println("Processing map: $map_id for $agent_name")
        
        # Get goals for this agent: [scenario1_gem, scenario2_gem]
        gem_indices = agent_goals[agent_name]
        
        goal_probs_conditioned_dict[agent_name][map_id] = Dict()
        state_probs_conditioned_dict[agent_name][map_id] = Dict()
        
        # Loop over both scenarios
        for scenario in 1:2
            goal_gem_idx = gem_indices[scenario]
            goal = PDDL.parse_pddl("(has $agent_sym gem$(goal_gem_idx))")
            println("  Scenario $scenario: $(agent_name) -> gem$(goal_gem_idx)")
            
            goal_probs_conditioned_dict[agent_name][map_id][scenario] = Dict()
            state_probs_conditioned_dict[agent_name][map_id][scenario] = Dict()

            # Load domain
            domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

            # Load problem (convert from txt to PDDL if needed)
            problem_path = joinpath(PROBLEM_DIR, "$map_id.pddl")
            if !isfile(problem_path)
                # Need to convert from txt first
                include("src/ascii.jl")
                txt_path = joinpath(PROBLEM_DIR, "$map_id.txt")
                problem = load_ascii_problem(txt_path)
            else
                problem = load_problem(problem_path)
            end

            # Initialize and compile reference state
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

            # Define planning algorithm
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

            # Run inference only for the specific goal in this scenario
            g = goal_gem_idx  # Use the specific goal from metadata
            goal_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict()
            state_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict()

            for i in 1:length(initial_states)
                    state_i = initial_states[i]
                    planner_astar = AStarPlanner(GoalManhattan())
                    plan = planner_astar(domain, state_i, goals[g])

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



data = Dict("goal" => goal_probs_conditioned_dict, "state" => state_probs_conditioned_dict, "worlds" => possible_worlds)

save("inference_data_exp3.jld2", data)

println("\n=== Inference Complete ===")
println("Saved to: inference_data_exp3.jld2")
println("Data structure: data[agent_name][map_id][scenario][goal_id][state_id]")


