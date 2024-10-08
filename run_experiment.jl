using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using CSV, DataFrames, Dates
using GenGPT3
using PDDLViz, GLMakie, CairoMakie

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/translate.jl")
include("src/render.jl")

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")
STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements")
RESULTS_DIR = joinpath(@__DIR__, "results")
mkpath(RESULTS_DIR)

# Load domain
DOMAIN = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
COMPILED_DOMAINS = Dict{String, Domain}()

# Load problems
PROBLEMS = Dict{String, Problem}()
for path in readdir(PROBLEM_DIR)
    name, ext = splitext(path)
    ext == ".pddl" || continue
    PROBLEMS[name] = load_problem(joinpath(PROBLEM_DIR, path))
end

# Load plans, judgement points, and belief statements
PLAN_IDS, PLANS, _, SPLITPOINTS = load_plan_dataset(PLAN_DIR)
_, STATEMENTS = load_statement_dataset(STATEMENT_DIR)

# Action temperatures to enumerate over
ACT_TEMPERATURES = [
    # 2 .^ (-3:0.5:5); # Uncomment to do grid search over range of temperatures
    (inv_gamma, (0.5, 1.0), 2, -5:0.25:5) # InvGamma(0.5, 1.0) temperature prior 
]

# Whether to normalize the belief statement prior
NORMALIZE_STATEMENT_PRIOR = true

# Translate all statements to PDDL in advance
PDDL_STATEMENTS = translate_statement_dataset(PLAN_IDS, STATEMENTS)

## Run experiments ##

df = DataFrame(
    # Plan info
    plan_id = String[],
    true_goal = Int[],
    # Model parameters
    method = String[],
    normalize_prior = Bool[],
    act_temperature = String[],
    # Timesteps
    timestep = Int[],
    is_judgment = Bool[],
    action = String[],
    # Goal inference results
    goal_probs_1 = Float64[],
    goal_probs_2 = Float64[],
    goal_probs_3 = Float64[],
    goal_probs_4 = Float64[],
    true_goal_probs = Float64[],
    lml_est = Float64[],
    # Initial state probabilities
    n_init_states = Int[],
    state_probs_1 = Float64[],
    state_probs_2 = Float64[],
    state_probs_3 = Float64[],
    state_probs_4 = Float64[],
    state_probs_5 = Float64[],
    state_probs_6 = Float64[],
    state_probs_7 = Float64[],
    state_probs_8 = Float64[],
    state_probs_9 = Float64[],
    state_probs_10 = Float64[],
    state_probs_11 = Float64[],
    state_probs_12 = Float64[],
    state_probs_13 = Float64[],
    state_probs_14 = Float64[],
    state_probs_15 = Float64[],
    state_probs_16 = Float64[],
    state_probs_17 = Float64[],
    state_probs_18 = Float64[],
    # Belief statement probabilities
    n_statements = Int[],
    statement_probs_1 = Float64[],
    statement_probs_2 = Float64[],
    statement_1 = String[],
    statement_2 = String[]
)
datetime = Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
df_types = eltype.(eachcol(df))
df_path = NORMALIZE_STATEMENT_PRIOR ?
    "results_joint_uniform.csv" : "results_joint.csv"
df_path = joinpath(RESULTS_DIR, "models", df_path)

# Iterate over plans
for plan_id in PLAN_IDS
    println("=== Plan $plan_id ===")
    println()

    # Load problem, plan, splitpoints, and statements
    problem = PROBLEMS[plan_id]
    plan = PLANS[plan_id]
    splitpoints = SPLITPOINTS[plan_id]
    statements = STATEMENTS[plan_id]

    # Extract true goal index from problem
    true_goal = parse(Int, string(problem.goal.args[2])[end])

    # Compile domain for problem
    domain = get!(COMPILED_DOMAINS, plan_id) do
        println("Compiling domain for problem $plan_id...")
        state = initstate(DOMAIN, problem)
        domain, _ = PDDL.compiled(DOMAIN, state)
        return domain
    end

    # Initialize reference state
    state = initstate(domain, problem)

    # Lookup translated PDDL statements
    pddl_statements = PDDL_STATEMENTS[plan_id]
    pddl_statement_strs = write_pddl.(pddl_statements)

    # Specify possible goals
    goals = @pddl(
        "(has human gem1)",
        "(has human gem2)",
        "(has human gem3)",
        "(has human gem4)"
    )
    goal_names = ["tri", "square", "hex", "circle"]
    goal_colors = gem_colors

    n = 5

    belief_states = enumerate_beliefs(
        state,
        min_keys = 1,
        max_keys = min(2, length(PDDL.get_objects(state, :box))),
        # max_keys = 2,
        max_color_keys = [2 for _ in PDDL.get_objects(state, :color)],
        discount = 1.0
    )

    # evaluate_statement(domain, belief_states, [0,0,0,0,1,0], pddl_statements[1].args[2])

    belief_states, belief_probs, prior_probs = generate_beliefs(domain, state, statement.args[2], n)


    # Define uniform prior over possible goals
    @gen function goal_prior()
        goal_id ~ uniform_discrete(1, length(goals))
        return Specification(goals[goal_id])
    end

    initial_states = [state]

    # Define uniform prior over possible initial states
    @gen function state_prior()
        state_id ~ uniform_discrete(1, length(initial_states))
        return initial_states[state_id]
    end
    

    state_names = map(box_contents_str, initial_states)

    # Construct iterator over initial choicemaps for stratified sampling
    init_state_addr = :init => :env => :state_id
    goal_addr = :init => :agent => :goal => :goal_id
    belief_addr = :init => :agent => :belief => :belief_idx


    init_strata = choiceproduct(
        (goal_addr, 1:length(goals)),
        (init_state_addr, 1:length(initial_states))
    )

    # Define planning algorithm
    heuristic = GoalManhattan()
    planner = RTHS(heuristic, n_iters=1, max_nodes=2^15)

    # Iterate over possible action noise configurations
    for temp in ACT_TEMPERATURES
        println("Action temperature: $temp\n")
        
        # Convert act temperature parameter
        if temp isa Real
            act_args = ([temp],)
        elseif temp isa AbstractVector
            act_args = (temp,)
        elseif temp isa Tuple && temp[1] isa Gen.Distribution
            act_args = temp
            dist, dist_args = act_args[1:2]
            temp_base = act_args[3]
            temp_exp_range = act_args[4]
            temps = temp_base .^ collect(temp_exp_range)
            act_args = (temps, dist, dist_args)
        else
            act_args = temp
        end

        # Define action noise model
        act_config = HierarchicalBoltzmannActConfig(act_args...)

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
        t_obs_iter = act_choicemap_pairs(plan)

        # Set up logging and printing callback
        n_goals = length(goals)
        n_init_states = length(initial_states)
        logger_cb = DataLoggerCallback(
            t = (t, pf) -> t::Int,
            goal_probs = pf -> begin
                probvec(pf, goal_addr, 1:4)::Vector{Float64}
            end,
            state_probs = pf -> begin
                probvec(pf, init_state_addr, 1:n_init_states)::Vector{Float64}
            end,
            statement_probs = (t, pf) -> begin
                map(pddl_statements) do stmt
                    get_formula_probs(
                        pf, domain, stmt, t;
                        normalize_prior = NORMALIZE_STATEMENT_PRIOR,
                        belief_probs = belief_probs
                    )
                end::Vector{Float64}
            end,
            lml_est = pf -> log_ml_estimate(pf)::Float64,
        )
        print_cb = PrintStatsCallback(
            (goal_addr, 1:length(goals)),
            (init_state_addr, 1:length(initial_states)),;
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

        # Extract logged data
        timesteps = callback.logger.data[:t]
        goal_probs = reduce(hcat, callback.logger.data[:goal_probs])
        state_probs = reduce(hcat, callback.logger.data[:state_probs])
        statement_probs = reduce(hcat, callback.logger.data[:statement_probs])
        lml_est = callback.logger.data[:lml_est]

        actions = write_pddl.([PDDL.no_op; plan])
        is_judgment = [t in splitpoints && t != 0 for t in timesteps]

        println()
        println("Statement probabilities")
        display(statement_probs)

        # Create and append dataframe
        T = length(timesteps)
        # println(statement_probs[2,:])
        # println(statement_probs[2,1])
        # # println(log(1-j+0.001))
        # println(log(1-statement_probs[2,1]))
        # statement_probs[1,:] = [min(0.5*(1+log(1-statement_probs[1,1]) - log(1-j+0.001)),1) for j in statement_probs[1,:]]
        # statement_probs[2,:] = [min(0.5*(1+log(1-statement_probs[2,1]) - log(1-j+0.001)),1) for j in statement_probs[2,:]]

        new_df = DataFrame(
            "plan_id" => fill(plan_id, T),
            "true_goal" => fill(true_goal, T),
            "method" => fill("joint", T),
            "normalize_prior" => fill(NORMALIZE_STATEMENT_PRIOR, T),
            "act_temperature" => fill(string(temp), T),
            "timestep" => timesteps,
            "is_judgment" => is_judgment,
            "action" => actions,
            ("goal_probs_$i" => goal_probs[i, :] for i in 1:n_goals)...,
            "true_goal_probs" => goal_probs[true_goal, :],
            "lml_est" => lml_est,
            "n_init_states" => fill(n_init_states, T),
            ("state_probs_$i" => state_probs[i, :] for i in 1:n_init_states)...,
            "n_statements" => fill(length(statements), T),
            ("statement_probs_$i" => statement_probs[i, :] for i in 1:length(statements))...,
            ("statement_$i" => pddl_statement_strs[i] for i in 1:length(statements))...
        )

        append!(df, new_df, cols=:union)
        CSV.write(df_path, df)
        println()
    end
    GC.gc()
end

model_path = RESULTS_DIR * "/results.csv"
human_path = RESULTS_DIR * "/human.csv"

model_rating = CSV.read(model_path, DataFrame)
human_rating = CSV.read(human_path, DataFrame)

plan_index = "1_1"
# plan_ids = sort!(collect(keys(plans)))
# for plan_index in plan_ids
# m = match(r"p(\d+)_g(\d+)", plan_index).captures
problem_id = plan_index
# goal_id = parse(Int, m[2])

plan = PLANS[plan_index]
# utterance = utterances[plan_index]
times = SPLITPOINTS[plan_index][2:end]

# Filter out results for this plan stimuli
filter_fn = r -> r.plan_id == plan_index && (r.is_judgment || r.timestep == length(plan))
model_rating_sub_df = filter(filter_fn, model_rating)

filter_fn = r -> r.plan_id == plan_index
human_rating_sub_df = filter(filter_fn, human_rating)

# Use GLMakie for rendering
GLMakie.activate!()

# Initialize state, and set renderer resolution to fit state grid
state = initstate(domain, PROBLEMS[plan_index])
grid = state[pddl"(walls)"]
height, width = size(grid)
RENDERER.resolution = (width * 100, (height + 2) * 100 + 200)

# Initialize canvas for animation
canvas = new_canvas(RENDERER)
canvas = anim_initialize!(
    canvas, RENDERER, domain, state;
    # caption = "Human: " * utterance, caption_size = 40,
    trail_length = 12
)
canvas.blocks[2].titlesize = 30
# canvas.blocks[3].titlesize = 30
display(canvas)

# Animate plan
anim = anim_plan!(
    canvas, RENDERER, domain, state, plan;
    trail_length=10,
    format = "gif",
    # captions=Dict(1 => "Human: " * utterance, times[2] => "...")
)
anim_dir = mkpath(joinpath(@__DIR__, "animations"))
save(joinpath(anim_dir, "anim_p$(problem_id)_g$(goal_id).gif"), anim)

# Switch to CairoMakie for plotting
CairoMakie.activate!()

# Plot storyboard
storyboard = render_storyboard(
    anim, times;
    xlabels = ["t = $t" for t in times],
    xlabelsize = 36, xlabelfont = :italic
);