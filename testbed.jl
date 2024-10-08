using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
# using PDDLViz, GLMakie

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/translate.jl")
# include("src/render.jl")

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")
STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements")

#--- Initial Setup ---#

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

# Load problem
p_id = "1_1"
problem = load_problem(joinpath(PROBLEM_DIR, "$(p_id).pddl"))

# Load plan
plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

# Load belief statements
statements = load_statements(joinpath(STATEMENT_DIR, "$(p_id).txt"))

# Initialize and compile reference state
state = initstate(domain, problem)
domain, state = PDDL.compiled(domain, problem)

# Render initial state
# canvas = RENDERER(domain, state)

#--- Goal Inference Setup ---#

# Translate belief statements to (extended) PDDL
pddl_statements = map(statements) do stmt
    translate_statement(stmt, verbose = true)
end

# Specify possible goals
goals = @pddl(
    "(has human gem1)",
    "(has human gem2)",
    "(has human gem3)",
    "(has human gem4)"
)

goal_names = ["tri", "square", "hex", "circle"]
# goal_colors = gem_colors

# Define uniform prior over possible goals
@gen function goal_prior()
    goal_id ~ uniform_discrete(1, length(goals))
    return Specification(goals[goal_id])
end

# Enumerate over possible initial states
initial_states, belief_probs = enumerate_beliefs(
    state,
    min_keys = 1,
    max_keys = min(2, length(PDDL.get_objects(state, :box))),
    max_color_keys = [2 for _ in PDDL.get_objects(state, :color)],
    discount = 1.0
)
state_names = map(box_contents_str, initial_states)

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
temperatures = 2 .^ collect(-2:0.55:5)
act_config = HierarchicalBoltzmannActConfig(temperatures,
                                            Gen.inv_gamma, (0.5, 1.0))

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
NORMALIZE_STATEMENT_PRIOR = true

# Construct iterator over observation timesteps and choicemaps 
t_obs_iter = act_choicemap_pairs(plan)

# Set up logging callback
n_goals = length(goals)
n_init_states = length(initial_states)
# logger_cb = DataLoggerCallback(
#     t = (t, pf) -> t::Int,
#     goal_probs = pf -> probvec(pf, goal_addr, 1:4)::Vector{Float64},
#     state_probs = pf -> probvec(pf, init_state_addr, 1:n_init_states)::Vector{Float64},
#     statement_probs = (t, pf) -> begin
#         map(pddl_statements) do stmt
#             get_formula_probs(
#                 pf, domain, stmt, t;
#                 normalize_prior = NORMALIZE_STATEMENT_PRIOR,
#                 belief_probs = belief_probs
#             )
#         end::Vector{Float64}
#     end,
#     lml_est = pf -> log_ml_estimate(pf)::Float64,
# )
print_cb = PrintStatsCallback(
    (goal_addr, 1:length(goals)),
    (init_state_addr, 1:length(initial_states)),
    (belief_addr, 1:length(beliefs)),;
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
    callback = print_cb
);

# Extract goal probabilities
goal_probs = reduce(hcat, callback.logger.data[:goal_probs])

# Extract initial state probabilities
state_probs = reduce(hcat, callback.logger.data[:state_probs])

# Extract statement probabilities
statement_probs = reduce(hcat, callback.logger.data[:statement_probs])
