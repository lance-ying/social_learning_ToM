using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/translate.jl")
include("src/render.jl")

# include("paths_new.jl")
# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")

#--- Initial Setup ---#

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

# Load problem
p_id = "s362_blue_exp"
map_id = p_id[1:4]
problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
# plan = paths[p_id]
# Load plan
plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

# Initialize and compile reference state
state = initstate(domain, problem)

heuristic = GoalManhattan()
planner = AStarPlanner(heuristic)

domain, state = PDDL.compiled(domain, problem)

# Render initial state

#--- Goal Inference Setup ---#

# Specify possible goals
goals = @pddl(
    "(has agent2 gem1)",
    "(has agent2 gem2)",
    "(has agent2 gem3)"
)

goal_names = ["A", "B", "C"]
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
temperatures =2.0

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

# Construct iterator over observation timesteps and choicemaps 
t_obs_iter = act_choicemap_pairs(collect(plan))

# Set up logging callback
n_goals = length(goals)
n_init_states = length(initial_states)
logger_cb = DataLoggerCallback(
    t = (t, pf) -> t::Int,
    goal_probs = pf -> probvec(pf, goal_addr, 1:3)::Vector{Float64},
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
goal_probs = reduce(hcat, callback.logger.data[:goal_probs])

# Extract initial state probabilities
state_probs = reduce(hcat, callback.logger.data[:state_probs])

# condition = "naive"


action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

plan_cost = calculate_plan_cost(plan, action_cost)

estimate_self_exploration_cost(state, problem.goal,PDDL.get_objects(state, :wizard), action_cost)


new_state = copy(state)
t= 0

blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
wizard_candicates = blue_wizards

goal_probs_conditioned_dict = Dict()
state_probs_conditioned_dict = Dict()


for g in 1:length(goals)

    for state in initial_states
        planner = AStarPlanner(GoalManhattan())
        plan =planner(domain, state, goals[g])

        t_obs_iter = act_choicemap_pairs(collect(plan))

        # Set up logging callback
        n_goals = length(goals)
        n_init_states = length(initial_states)
        logger_cb = DataLoggerCallback(
            t = (t, pf) -> t::Int,
            goal_probs = pf -> probvec(pf, goal_addr, 1:3)::Vector{Float64},
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

        goal_probs_conditioned_dict[g] = goal_probs_conditioned
        state_probs_conditioned_dict[g] = state_probs_conditioned
    end
end


blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
wizard_candicates = blue_wizards
t=0

while !PDDL.satisfy(domain, new_state, problem.goal)

    Q_observe = 0


    for g in 1:length(goals)

        if goal_probs[g, t+1] < 0.1
            continue
        end

        # Q_T_best = Inf
        # T_best = 0


        for w in wizard_candicates
            loc = get_obj_loc(new_state, w)
            loc_x = loc[1]
            loc_y = loc[2]
            goal = pddl"(and (= (xloc agent1) $x_loc) (= (yloc agent1) $y_loc))"
            plan = planner(domain, new_state, goal)




        end

        # for i in 1:5

        #     new_wizard_candicates = []

        #     for j in 1:length(blue_wizards)
        #         if state_probs_conditioned_dict[g][j, t+i+1] > 0.1
        #             push!(new_wizard_candicates, blue_wizards[j])
        #         end
        #     end

        #     # print("checkpt1",new_wizard_candicates )

        #     Q_T = estimate_self_exploration_cost(new_state, problem.goal, new_wizard_candicates, action_cost)

        #     if Q_T < Q_T_best
        #         Q_T_best = Q_T
        #         T_best = i
        #     end
        # end

        Q_observe += goal_probs[g, t+1] * (Q_T_best + action_cost[:observe] * T_best)
    end 

    # print("checkpt2",wizard_candicates )

    Q_not_observe = estimate_self_exploration_cost(new_state, problem.goal, wizard_candicates, action_cost)

    print("Q_observe = ", Q_observe, "Q_not_observe = ", Q_not_observe)
    println()
    if Q_observe < Q_not_observe
        t+=1

        wizard_candicates = []

        for j in 1:length(blue_wizards)
            if state_probs[j, t+1] > 0.1
                push!(wizard_candicates, blue_wizards[j])
            end
        end
    else
        print("t = ", t)
        break
    end
end

using Distributions, Random

x = rand(Normal(2, 1.5), 10)