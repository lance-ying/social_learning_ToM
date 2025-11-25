############ BOOTSTRAP: activate env everywhere + ensure deps ############
import Pkg
const PROJ = Base.active_project() === nothing ? joinpath(@__DIR__, "Project.toml") : Base.active_project()
Pkg.activate(PROJ)

try
    # Registered deps
    Pkg.add(["PDDL","SymbolicPlanners","Gen","GenParticleFilters","PDDLViz","GLMakie","JSON","JLD2","FileIO","ProgressMeter"])
    # Unregistered dep
    if isnothing(Base.find_package("InversePlanning"))
        Pkg.add(PackageSpec(url="https://github.com/cosilab/InversePlanning.jl.git"))
    end
    Pkg.precompile()
catch e
    @warn "Dependency setup warning" exception=e
end

using Distributed
if nprocs() == 1
    addprocs(max(1, Sys.CPU_THREADS - 1))   # use all but one core locally
end

@everywhere begin
    import Pkg
    Pkg.activate($PROJ)  # all workers use SAME project
end
############ END BOOTSTRAP ###############################################


@info "Workers" nworkers() procs=workers()

# --- Stage A: load packages and import macros on ALL workers ---
@everywhere begin
    using PDDL, SymbolicPlanners
    using Gen, GenParticleFilters
    using InversePlanning
    using JLD2, FileIO, JSON
    using ProgressMeter

    # Make the @gen macro visible in Main on workers
    import Gen: @gen

    # Optional: skip GLMakie on headless runs
    try
        using PDDLViz
        # using GLMakie   # comment out on clusters/headless
    catch e
        @warn "Skipping PDDLViz/GL backends" exception=e
    end

    const ROOT = @__DIR__
    const DATASET_DIR = joinpath(ROOT, "dataset")
    const PROBLEM_DIR = joinpath(DATASET_DIR, "problems_exp3")
    PDDL.Arrays.register!()

    include(joinpath(ROOT, "src", "plan_io.jl"))
    include(joinpath(ROOT, "src", "utils.jl"))
    include(joinpath(ROOT, "src", "heuristics.jl"))
    include(joinpath(ROOT, "src", "beliefs.jl"))
    include(joinpath(ROOT, "src", "render.jl"))

    const ASCII_PATH = joinpath(ROOT, "src", "ascii.jl")
    if isfile(ASCII_PATH)
        include(ASCII_PATH)
    end
end

    # --- Stage B: only now define any code that uses @gen on workers ---
@everywhere begin
    function load_domain_problem_states(map_id::AbstractString)
        domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
        problem_path = joinpath(PROBLEM_DIR, "$map_id.pddl")
        problem = isfile(problem_path) ? load_problem(problem_path) :
                                         load_ascii_problem(joinpath(PROBLEM_DIR, "$map_id.txt"))
        state_u = initstate(domain, problem)                    # UNCOMPILED state (for objects/goals)
        domain_c, state_c = PDDL.compiled(domain, problem)     # COMPILED (for planning/inference)
        return domain, problem, state_u, domain_c, state_c
    end
    

    function prepare_for(agent_sym::Symbol, map_id::AbstractString, scenario_goal_index::Int)
        domain, problem, state_u, domain_c, state_c = load_domain_problem_states(map_id)
        
        # Use UNCOMPILED (slower per state, but 10 workers makes total time ~4-5 hours)
        domain, state = domain, state_u
        goals, goal_names = initialize_goals(state, agent_sym)
        initial_states, belief_probs, state_names = enumerate_beliefs(state)

        @gen function goal_prior()
            goal_id ~ uniform_discrete(1, length(goals))
            return Specification(goals[goal_id])
        end
        @gen function state_prior()
            state_id ~ categorical(belief_probs)
            return initial_states[state_id]
        end

        heuristic = GoalManhattan()
        planner = RTHS(heuristic, n_iters=1, max_nodes=2^15)
        act_config = BoltzmannActConfig(0.5)

        agent_config = AgentConfig(
            domain, planner;
            goal_config = StaticGoalConfig(goal_prior),
            replan_args = (plan_at_init=true, prob_refine=1.0, prob_replan=0, rand_budget=false),
            act_config = act_config
        )

        world_config = WorldConfig(
            agent_config = agent_config,
            env_config = PDDLEnvConfig(domain, state_prior)
        )

        init_state_addr = :init => :env => :state_id
        goal_addr = :init => :agent => :goal => :goal_id
        init_strata = choiceproduct(
            (goal_addr, 1:length(goals)),
            (init_state_addr, 1:length(initial_states))
        )

        return (; domain, state, goals, goal_names, initial_states, belief_probs,
                state_names, world_config, init_state_addr, goal_addr, init_strata,
                scenario_goal_index)
    end

    function run_batch(agent_sym::Symbol, map_id::AbstractString, scenario_goal_index::Int, idxs::Vector{Int})
        println("    [Worker $(myid())] Starting batch: $(length(idxs)) states")
        flush(stdout)
        prep = prepare_for(agent_sym, map_id, scenario_goal_index)
        println("    [Worker $(myid())] Preparation complete")
        flush(stdout)
        results = Vector{Tuple{Int, Matrix{Float64}, Matrix{Float64}}}(undef, length(idxs))
        planner_astar = AStarPlanner(GoalManhattan())

        @inbounds for (k, i) in enumerate(idxs)
            println("    [Worker $(myid())] Processing state $i ($(k)/$(length(idxs)))")
            flush(stdout)
            state_i = prep.initial_states[i]
            plan = planner_astar(prep.domain, state_i, prep.goals[prep.scenario_goal_index])
            println("    [Worker $(myid())] Plan length for state $i: $(length(collect(plan))) steps")
            flush(stdout)
            t_obs_iter = act_choicemap_pairs(collect(plan))

            n_goals = length(prep.goals)
            n_states = length(prep.initial_states)
            logger_cb = DataLoggerCallback(
                t = (t, pf) -> t::Int,
                goal_probs = pf -> probvec(pf, prep.goal_addr, 1:n_goals)::Vector{Float64},
                state_probs = pf -> probvec(pf, prep.init_state_addr, 1:n_states)::Vector{Float64},
                lml_est = pf -> log_ml_estimate(pf)::Float64,
            )
            callback = CombinedCallback(logger=logger_cb)

            sips = SIPS(prep.world_config, resample_cond=:none, rejuv_cond=:none)
            n_samples = length(prep.init_strata)
            println("    [Worker $(myid())] Running inference for state $i with $n_samples particles...")
            flush(stdout)
            sips(n_samples, t_obs_iter; init_args=(init_strata=prep.init_strata,), callback=callback)

            goal_probs_conditioned  = reduce(hcat, callback.logger.data[:goal_probs])
            state_probs_conditioned = reduce(hcat, callback.logger.data[:state_probs])
            results[k] = (i, goal_probs_conditioned, state_probs_conditioned)
            println("    [Worker $(myid())] ✓ State $i complete")
            flush(stdout)
        end
        println("    [Worker $(myid())] Batch complete!")
        flush(stdout)
        return results
    end

    function chunk_indices(n::Int, nchunks::Int)
        nchunks = max(1, min(n, nchunks))
        base = div(n, nchunks); remn = n % nchunks
        chunks = Vector{Vector{Int}}(undef, nchunks)
        start = 1
        for c in 1:nchunks
            len = base + (c <= remn ? 1 : 0)
            stop = start + len - 1
            chunks[c] = (len == 0) ? Int[] : collect(start:stop)
            start = stop + 1
        end
        return [ch for ch in chunks if !isempty(ch)]
    end
end

# ----------------- Master process orchestration -----------------

# Input directories and metadata (master)
const ROOT = @__DIR__
const PROBLEM_DIR = joinpath(ROOT, "dataset", "problems_exp3")

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

agents_to_infer = ["agent2", "agent3"]

goal_probs_conditioned_dict = Dict{String, Any}()
state_probs_conditioned_dict = Dict{String, Any}()
possible_worlds = Dict{String, Any}()

# Helper: count initial states once on master (don't compile, just count)
function count_initial_states_for(agent_sym::Symbol, map_id::String)
    domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    problem_path = joinpath(PROBLEM_DIR, "$map_id.pddl")
    problem = isfile(problem_path) ? load_problem(problem_path) :
                                     (include(joinpath(ROOT, "src", "ascii.jl"));
                                      load_ascii_problem(joinpath(PROBLEM_DIR, "$map_id.txt")))
    state = initstate(domain, problem)
    initial_states, belief_probs, state_names = enumerate_beliefs(state)
    return length(initial_states)  # Only return count
end

println("\n=== Parallel Inference Start ===\n")

# Calculate total work for progress bar
total_scenarios = length(agents_to_infer) * length(metadata) * 2  # 2 scenarios per map
progress = Progress(total_scenarios, desc="Overall progress: ", barlen=50, showspeed=true)

for agent_name in agents_to_infer
    agent_sym = Symbol(agent_name)
    goal_probs_conditioned_dict[agent_name] = Dict{Any, Any}()
    state_probs_conditioned_dict[agent_name] = Dict{Any, Any}()
    possible_worlds[agent_name] = Dict{Any, Any}()

    println("\n== Agent: $agent_name ==")

    for (map_id, agent_goals_any) in metadata
        agent_goals = Dict{String, Any}(agent_goals_any)
        println("Processing map $map_id for $agent_name")

        gem_indices = agent_goals[agent_name]  # [scenario1_gem, scenario2_gem]
        goal_probs_conditioned_dict[agent_name][map_id] = Dict{Any, Any}()
        state_probs_conditioned_dict[agent_name][map_id] = Dict{Any, Any}()

        # Get n_states from master (just count, don't compile)
        n_states = count_initial_states_for(agent_sym, map_id)

        for scenario in 1:2
            g = Int(gem_indices[scenario])
            println("  Scenario $scenario: $agent_name -> gem$g  (n_states=$n_states)")

            goal_probs_conditioned_dict[agent_name][map_id][scenario] = Dict{Any, Any}()
            state_probs_conditioned_dict[agent_name][map_id][scenario] = Dict{Any, Any}()
            goal_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict{Int, Any}()
            state_probs_conditioned_dict[agent_name][map_id][scenario][g] = Dict{Int, Any}()

            # Create more chunks than workers for better balance
            batches = @fetchfrom 1 begin
                # Choose ~2×nworkers chunks but not more than n_states
                chunk_indices(n_states, min(n_states, max(2*nworkers(), 1)))
            end

            # Map batches across workers
            results_batches = pmap(b -> run_batch(agent_sym, map_id, g, b), batches)

            # Collect results and possible worlds from first batch
            for (batch_idx, batch) in enumerate(results_batches)
                for (i, gp, sp) in batch
                    goal_probs_conditioned_dict[agent_name][map_id][scenario][g][i] = gp
                    state_probs_conditioned_dict[agent_name][map_id][scenario][g][i] = sp
                end
            end
            
            # Store possible worlds from first scenario only
            if scenario == 1
                # Get initial_states from a worker
                temp_prep = @fetchfrom 2 prepare_for(agent_sym, map_id, g)
                possible_worlds[agent_name][map_id] = temp_prep.initial_states
            end

            # Update progress bar
            next!(progress, showvalues = [(:agent, agent_name), (:map, map_id), (:scenario, scenario)])
        end
    end
end

finish!(progress)

data = Dict(
    "goal" => goal_probs_conditioned_dict,
    "state" => state_probs_conditioned_dict,
    "worlds" => possible_worlds
)

save("inference_data_exp3.jld2", data)

println("\n=== Inference Complete ===")
println("Saved to: inference_data_exp3.jld2")
println("Data structure: data[agent_name][map_id][scenario][goal_id][state_id]")
println("Procs used: $(nprocs()) (workers: $(nworkers()))")
