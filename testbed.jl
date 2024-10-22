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
p_id = "s221_blue_exp"
map_id = p_id[1:4]
problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
# plan = paths[p_id]
# Load plan
plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

# Load belief statements
# statements = load_statements(joinpath(STATEMENT_DIR, "$(p_id).txt"))

# Initialize and compile reference state
state = initstate(domain, problem)

heuristic = GoalManhattan()
planner = AStarPlanner(heuristic)


# w_loc = get_obj_loc(state, pddl"wizard2")
# x_loc = w_loc[1]
# y_loc = w_loc[2]
# goal = pddl"(and (= (xloc agent1) $x_loc) (= (yloc agent1) $y_loc))"

# plan = planner(domain, state, goal)

# canvas

# canvas = PDDLViz.new_canvas(RENDERER)
# anim = anim_plan!(canvas, RENDERER, domain, state, collect(plan); trail_length = 15, show_inventory=false)


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
temperatures =0.01

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

## Run goal and belief inference ##

# Whether or not to normalize the statement prior to 50/50
# NORMALIZE_STATEMENT_PRIOR = true

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

# Extract statement probabilities
# statement_probs = reduce(hcat, callback.logger.data[:statement_probs])

# Estimate the reward for not observing

function calculate_plan_cost(plan:: Vector{<:Term}, action_cost::Dict{Symbol, Real})

    cost = 0

    for act in plan
        if act.name == :interact
            cost += action_cost[:interact]
        else
            cost += action_cost[:move]
        end
    end
    return cost
    
end

function estimate_self_exploration_cost(state:: State, agent_goal::Any, wizards::Any, action_cost::Dict{Symbol, Real})

    new_state = copy(state)

    planner = AStarPlanner(GoalManhattan())
    # Extract wizard locations
    # print(state[pddl"(iscolor wizard1 blue)"])
    wizard_locs = [get_obj_loc(new_state, w) for w in wizards if state[pddl"(iscolor $w blue)"]]

    # Compute self-exploration cost
    cost = 0
    total_cost = 0

    for i in 1:length(wizards)-1
        cost = Inf
        min_distance_loc = wizard_locs[1]
        for w_loc in wizard_locs

            x_loc = w_loc[1]
            y_loc = w_loc[2]
            goal = pddl"(and (= (xloc agent1) $x_loc) (= (yloc agent1) $y_loc))"
            plan = planner(domain, new_state, goal)

            plan_cost = calculate_plan_cost(collect(plan), action_cost)

            if plan_cost < cost
                cost = plan_cost
                min_distance_loc = w_loc
            end
        end

        # print(min_distance_loc)

        # if min_distance_loc in wizard_locs
        wizard_locs = filter!(loc -> (loc[1] != min_distance_loc[1] && loc[2] != min_distance_loc[2]), wizard_locs)

        total_cost += cost
        total_cost += action_cost[:interact]
        # total_cost -= 2*action_cost[:move]

        new_state[pddl"(xloc agent1)"] = min_distance_loc[1]
        new_state[pddl"(yloc agent1)"] = min_distance_loc[2]

        # counter+=1
        # new_state[]
    end

    goal_loc = get_obj_loc(new_state, agent_goal.args[2])

    x_loc = goal_loc[1]
    y_loc = goal_loc[2]

    goal = pddl"(and (= (xloc agent1) $x_loc) (= (yloc agent1) $y_loc))"

    plan = planner(domain, new_state, goal)

    plan_cost = calculate_plan_cost(collect(plan), action_cost)

    # print(agent_goal)
    # final_plan = planner(domain, new_state, agent_goal)
    # print(collect(final_plan))
    # print(calculate_plan_cost(collect(final_plan), action_cost))
    total_cost += plan_cost
    total_cost -= action_cost[:move] *(2* length(wizards)-1)


    return total_cost
    
end



action_cost = Dict(:move => 2, :interact => 5, :observe => 0.5)

plan_cost = calculate_plan_cost(plan, action_cost)

estimate_self_exploration_cost(state, problem.goal,PDDL.get_objects(state, :wizard), action_cost)


new_state = copy(state)
t= 0

while !PDDL.satisfy(domain, new_state, goal)

    Q_observe = 0

    for g in 1:length(goals)

        if goal_probs[g, t] < 0.2
            continues
        end

        plan = collect(planner(domain, new_state, goals[g]))

        Q_T_best = Inf
        T_best = 0

        for i in 1:length(plan)
            new_state = copy(state)
            for j in 1:i
                new_state = PDDL.apply(domain, new_state, plan[j])
            end

            Q_T = estimate_self_exploration_cost(new_state, problem.goal, PDDL.get_objects(new_state, :wizard), action_cost)

            if Q_T < Q_T_best
                Q_T_best = Q_T
                T_best = i
            end
        end



        Q_observe += goal_probs[g, t] * estimate_self_exploration_cost(new_state, problem.goal, PDDL.get_objects(new_state, :wizard), action_cost)
    end 

    Q_not_observe = estimate_self_exploration_cost(new_state, problem.goal, PDDL.get_objects(new_state, :wizard), action_cost)
end