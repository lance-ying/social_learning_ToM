using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using InversePlanning
using CSV, DataFrames, Dates
using GenGPT3

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")

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
    "results_goal_uniform.csv" : "results_goal.csv"
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

        # Specify possible goals
        goals = @pddl(
            "(has human gem1)",
            "(has human gem2)",
            "(has human gem3)",
            "(has human gem4)"
        )
        goal_names = ["tri", "square", "hex", "circle"]

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
            max_color_keys = [2 for _ in PDDL.get_objects(state, :color)]
        )
        state_names = map(box_contents_str, initial_states)

        # Allocate variables for logged data
        timesteps = collect(0:length(plan))
        T = length(timesteps)
        n_goals = length(goals)
        n_init_states = length(initial_states)

        goal_probs = zeros(n_goals, T)
        lml_est = fill(-Inf, T)

        # Set initial state probabilities
        state_probs = repeat(belief_probs, 1, T)

        # Set initial statement probabilities
        if NORMALIZE_STATEMENT_PRIOR
            statement_probs = fill(0.5, length(statements), T)
        else
            statement_probs = zeros(length(statements), T)
            for (belief_state, belief_prob) in zip(initial_states, belief_probs)
                for (i, formula) in enumerate(pddl_statements)
                    if formula.name == :believes
                        formula = formula.args[2]
                    end
                    if satisfy(domain, belief_state, formula)
                        statement_probs[i, :] .+= belief_prob
                    end
                end
            end
        end

        # Construct iterator over initial choicemaps for stratified sampling
        goal_addr = :init => :agent => :goal => :goal_id
        init_strata = choiceproduct((goal_addr, 1:n_goals))

        # Run goal inference for each initial belief state
        for (belief_state, belief_prob) in zip(initial_states, belief_probs)
            println("Belief: $(box_contents_str(belief_state))")

            # Define planning algorithm
            heuristic = GoalManhattan()
            planner = RTHS(heuristic, n_iters=1, max_nodes=2^15)

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
                env_config = PDDLEnvConfig(domain, belief_state)
            )

            # Construct iterator over observation timesteps and choicemaps 
            t_obs_iter = act_choicemap_pairs(plan)

            # Set up logging and printing callback
            logger_cb = DataLoggerCallback(
                t = (t, pf) -> t::Int,
                goal_probs = pf -> begin
                    probvec(pf, goal_addr, 1:n_goals)::Vector{Float64}
                end,
                lml_est = pf -> log_ml_estimate(pf)::Float64,
            )
            print_cb = PrintStatsCallback(
                (goal_addr, 1:n_goals),
                header = ("t\t" * join(goal_names, "\t"))
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

            # Accumulate logged data
            new_goal_probs = reduce(hcat, callback.logger.data[:goal_probs])
            goal_probs .+= new_goal_probs .* belief_prob
            new_lml_est = callback.logger.data[:lml_est]
            lml_est .= log.(exp.(lml_est) .+ exp.(new_lml_est) .* belief_prob)
        end

        # Create and append dataframe
        actions = write_pddl.([PDDL.no_op; plan])
        is_judgment = [t in splitpoints && t != 0 for t in timesteps]
        new_df = DataFrame(
            "plan_id" => fill(plan_id, T),
            "true_goal" => fill(true_goal, T),
            "method" => fill("goal_only", T),
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

        append!(df, new_df, cols = :union)
        CSV.write(df_path, df)
        println()
    end
    GC.gc()
end
