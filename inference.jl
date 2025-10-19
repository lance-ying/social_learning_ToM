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
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_exp2")

# #--- Initial Setup ---#


# goal_probs_conditioned_dict = load("inference_data_temp_new.jld2", "goal")
# state_probs_conditioned_dict = load("inference_data_temp_new.jld2", "state")
# possible_worlds = load("inference_data_temp_new.jld2", "worlds")

goal_probs_conditioned_dict = Dict()
state_probs_conditioned_dict = Dict()
possible_worlds = Dict()

# possible_worlds_render = Dict()

problem_files = filter(f -> endswith(f, ".pddl") && !occursin("plan", f), readdir(joinpath(@__DIR__, "dataset", "problems_exp2")))


metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

for (map_id, v) in metadata
    for (i, goal_str) in enumerate(v)
        if (occursin("442", map_id))
            continue
        end
        filename = "$(map_id)_$(i)_plan.pddl"
        goal = PDDL.parse_pddl("(has agent2 gem$(goal_str))")
        println(filename, ": ", goal)


        goal_probs_conditioned_dict[map_id] = Dict()
        state_probs_conditioned_dict[map_id] = Dict()

        # Load domain
        domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

        # Load problem

        problem = load_problem(joinpath(PROBLEM_DIR, "$map_id.pddl"))

        # Load plan
        # plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

        # Initialize and compile reference state
        state = initstate(domain, problem)

        heuristic = GoalManhattan()
        planner = AStarPlanner(heuristic)

        domain, state = PDDL.compiled(domain, problem)

        # Render initial state

        #--- Goal Inference Setup ---#

        # Specify possible goals
        goals, goal_names = initialize_goals(state)
        
        # @pddl(
        #     "(has agent2 gem1)",
        #     "(has agent2 gem2)",
        #     "(has agent2 gem3)"
        # )

        # goal_names = ["A", "B", "C"]
        # goal_colors = gem_colors

        # Define uniform prior over possible goals
        @gen function goal_prior()
            goal_id ~ uniform_discrete(1, length(goals))
            return Specification(goals[goal_id])
        end

        # Enumerate over possible initial states
        initial_states, belief_probs, state_names = enumerate_beliefs(
            state
        )

        possible_worlds[map_id] = initial_states


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
        temperatures =0.5

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

        for g in 1:length(goals)

            goal_probs_conditioned_dict[map_id][g] = Dict()
            state_probs_conditioned_dict[map_id][g] = Dict()

            for i in 1:length(initial_states)
                state = initial_states[i]
                planner = AStarPlanner(GoalManhattan())
                plan =planner(domain, state, goals[g])

                println(plan)

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
                sips = SIPS(world_config, resample_cond=:none, rejuv_cond=:none, resample_block_fn=nothing)

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

                goal_probs_conditioned_dict[map_id][g][i] = goal_probs_conditioned
                state_probs_conditioned_dict[map_id][g][i] = state_probs_conditioned
            end
        end

    end
end



data = Dict("goal" => goal_probs_conditioned_dict, "state" => state_probs_conditioned_dict, "worlds" => possible_worlds)

save("inference_data_exp2.jld2", data)


