using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JLD2, FileIO
using JSON
# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "src", "render.jl"))

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "..", "dataset", "problems_exp2_test")

goal_probs_conditioned_dict = Dict()
state_probs_conditioned_dict = Dict()
possible_worlds = Dict()
plans_dict = Dict()  # Store the paths/plans

problem_files = filter(f -> endswith(f, ".pddl") && !occursin("plan", f), readdir(joinpath(@__DIR__, "..", "dataset", "problems_exp2_test")))

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

for (map_id, v) in metadata
    # TEMPORARY: Only test s541 (note: metadata key is "s541", not "sm541_test")
    if map_id != "sm541_test"
        continue
    end
    
    for (i, goal_str) in enumerate(v)
        # TEMPORARY: Only test scenario 1
        if i != 1
            continue
        end
        
        if (occursin("442", map_id))
            continue
        end
        
        # Use the actual file naming: sm541_test_1.pddl
        problem_filename = "sm541_test_$(i).pddl"
        filename = "sm541_test_$(i)_plan.pddl"
        goal = PDDL.parse_pddl("(has agent2 gem$(goal_str))")
        println(filename, ": ", goal)

        goal_probs_conditioned_dict[map_id] = Dict()
        state_probs_conditioned_dict[map_id] = Dict()
        plans_dict[map_id] = Dict()  # Initialize plans storage for this map

        # Load domain
        domain = load_domain(joinpath(@__DIR__, "..", "dataset", "domain.pddl"))

        # Load problem - use the correct filename with scenario number
        problem = load_problem(joinpath(PROBLEM_DIR, problem_filename))

        # Initialize and compile reference state
        state = initstate(domain, problem)

        heuristic = GoalManhattan()
        planner = AStarPlanner(heuristic)

        domain, state = PDDL.compiled(domain, problem)

        #--- Goal Inference Setup ---#

        # Specify possible goals
        goals, goal_names = initialize_goals(state)

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

        for g in 1:length(goals)

            goal_probs_conditioned_dict[map_id][g] = Dict()
            state_probs_conditioned_dict[map_id][g] = Dict()
            plans_dict[map_id][g] = Dict()  # Initialize plans storage for this goal

            for i in 1:length(initial_states)
                state = initial_states[i]
                planner = AStarPlanner(GoalManhattan())
                plan = planner(domain, state, goals[g])

                println("Plan for goal $g, state $i: ", plan)
                
                # Store the plan
                plans_dict[map_id][g][i] = collect(plan)

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

                goal_probs_conditioned_dict[map_id][g][i] = goal_probs_conditioned
                state_probs_conditioned_dict[map_id][g][i] = state_probs_conditioned
                
                println("\n=== DEBUG: Goal $g, State $i ===")
                println("Goal probabilities over time:")
                for goal_idx in 1:size(goal_probs_conditioned, 1)
                    println("  Goal $goal_idx: ", round.(goal_probs_conditioned[goal_idx, 1:min(10, end)], digits=3))
                end
                println("Total timesteps: $(size(goal_probs_conditioned, 2))")
                println("====================================\n")
            end
        end

    end
end

data = Dict("goal" => goal_probs_conditioned_dict, "state" => state_probs_conditioned_dict, "worlds" => possible_worlds, "plans" => plans_dict)

save("inference_data_exp2_test.jld2", data)

plans_json = Dict()
for (map_id, goals_dict) in plans_dict
    plans_json[map_id] = Dict()
    for (g, states_dict) in goals_dict
        plans_json[map_id][string(g)] = Dict()
        for (i, plan) in states_dict
            plans_json[map_id][string(g)][string(i)] = [string(action) for action in plan]
        end
    end
end

open("plans_exp2_test.json", "w") do f
    JSON.print(f, plans_json, 4)
end
