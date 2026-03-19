using PDDL, SymbolicPlanners
using JSON
using FileIO, JLD2

PDDL.Arrays.register!()

isdefined(@__MODULE__, :GoalManhattan) || include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
isdefined(@__MODULE__, :clear_planner_cache!) || include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
isdefined(@__MODULE__, :load_ascii_problem) || include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl"))
isdefined(@__MODULE__, :NaivePlanner) || include(joinpath(@__DIR__, "..", "..", "src", "planners.jl"))

const ROOT = joinpath(@__DIR__, "..", "..")
const REPLAY_PLAN_CACHE = Dict{Any, Vector{Term}}()
const REPLAY_PLAN_CACHE_LOCK = ReentrantLock()
const REPLAY_PLAN_CACHE_HITS = Ref(0)
const REPLAY_PLAN_CACHE_MISSES = Ref(0)
const POSTERIOR_FILTER_CACHE = Dict{Any, Any}()
const POSTERIOR_FILTER_CACHE_LOCK = ReentrantLock()
const POSTERIOR_FILTER_CACHE_HITS = Ref(0)
const POSTERIOR_FILTER_CACHE_MISSES = Ref(0)

function clear_replay_plan_cache!()
    lock(REPLAY_PLAN_CACHE_LOCK) do
        empty!(REPLAY_PLAN_CACHE)
        REPLAY_PLAN_CACHE_HITS[] = 0
        REPLAY_PLAN_CACHE_MISSES[] = 0
    end
end

function get_replay_plan_cache_stats()
    total = REPLAY_PLAN_CACHE_HITS[] + REPLAY_PLAN_CACHE_MISSES[]
    hit_rate = total > 0 ? REPLAY_PLAN_CACHE_HITS[] / total : 0.0
    return Dict(
        "hits" => REPLAY_PLAN_CACHE_HITS[],
        "misses" => REPLAY_PLAN_CACHE_MISSES[],
        "hit_rate" => hit_rate,
        "entries" => length(REPLAY_PLAN_CACHE),
    )
end

function clear_posterior_filter_cache!()
    lock(POSTERIOR_FILTER_CACHE_LOCK) do
        empty!(POSTERIOR_FILTER_CACHE)
        POSTERIOR_FILTER_CACHE_HITS[] = 0
        POSTERIOR_FILTER_CACHE_MISSES[] = 0
    end
end

function get_posterior_filter_cache_stats()
    total = POSTERIOR_FILTER_CACHE_HITS[] + POSTERIOR_FILTER_CACHE_MISSES[]
    hit_rate = total > 0 ? POSTERIOR_FILTER_CACHE_HITS[] / total : 0.0
    return Dict(
        "hits" => POSTERIOR_FILTER_CACHE_HITS[],
        "misses" => POSTERIOR_FILTER_CACHE_MISSES[],
        "hit_rate" => hit_rate,
        "entries" => length(POSTERIOR_FILTER_CACHE),
    )
end

function get_cached_posterior_filter!(compute_fn, cache_key)
    lock(POSTERIOR_FILTER_CACHE_LOCK) do
        if haskey(POSTERIOR_FILTER_CACHE, cache_key)
            POSTERIOR_FILTER_CACHE_HITS[] += 1
            value = POSTERIOR_FILTER_CACHE[cache_key]
            return [copy(s) for s in value[1]], copy(value[2]), copy(value[3])
        end
        POSTERIOR_FILTER_CACHE_MISSES[] += 1
    end
    value = compute_fn()
    lock(POSTERIOR_FILTER_CACHE_LOCK) do
        POSTERIOR_FILTER_CACHE[cache_key] = ([copy(s) for s in value[1]], copy(value[2]), copy(value[3]))
    end
    return value
end

function parse_cli(args::Vector{String})
    opts = Dict{String, String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--")
            key = a[3:end]
            if i == length(args) || startswith(args[i + 1], "--")
                opts[key] = "true"
                i += 1
            else
                opts[key] = args[i + 1]
                i += 2
            end
        else
            i += 1
        end
    end
    return opts
end

function usage()
    println(
        """
        Usage:
          julia scripts/utilities/reconstruct_model_costs.jl \\
            --exp exp1|exp2|exp3|exp3_debug|exp4|exp4_wrapper \\
            [--model full_model|social_mentalizing|rational_non_mentalizing|naive_observer] \\
            --steps-file <path/to/steps_dict.json> \\
            [--replay-trace-file <path/to/replay_trace.json>] \\
            [--disable-exp4-interaction-outcome-pruning] \\
            [--inference-file <path/to/inference.jld2>] \\
            [--restrict-to-human-levels] [--human-costs-file <path/to/human_costs.json>] \\
            [--output-file <path/to/output.json>] \\
            [--problem-dir <path/to/dataset/problems_...>] \\
            [--move-cost <float>] [--interact-cost <float>] [--observe-cost <float>]

        Notes:
        - Reconstruction replays stored observations and uses model-specific internal updates.
        - For exp1/exp2, scalar t is treated as observing the single other agent t times.
        - For exp3/exp4, the stored observations list is replayed directly.
        """
    )
end

function normalize_exp(exp::String)
    exp = lowercase(exp)
    if exp == "exp3_debug"
        return "exp3"
    elseif exp == "exp4_wrapper"
        return "exp4"
    end
    return exp
end

function default_paths(exp::String)
    if exp == "exp1"
        return (
            steps_file = joinpath(ROOT, "scripts", "experiments", "experiment_outputs", "steps_dict_exp1.json"),
            inference_file = joinpath(ROOT, "data", "inference", "inference_data_exp1.jld2"),
            problem_dir = joinpath(ROOT, "dataset", "problems_exp1"),
        )
    elseif exp == "exp2"
        return (
            steps_file = joinpath(ROOT, "scripts", "experiments", "experiment_outputs", "steps_dict_exp2.json"),
            inference_file = joinpath(ROOT, "data", "inference", "inference_data_exp2.jld2"),
            problem_dir = joinpath(ROOT, "dataset", "problems_exp2"),
        )
    elseif exp == "exp3"
        return (
            steps_file = joinpath(ROOT, "scripts", "experiments", "experiment_outputs", "test_comparison.json"),
            inference_file = joinpath(ROOT, "data", "inference", "inference_data_exp3.jld2"),
            problem_dir = joinpath(ROOT, "dataset", "problems_exp3"),
        )
    elseif exp == "exp4"
        return (
            steps_file = joinpath(ROOT, "scripts", "experiments", "experiment_outputs", "steps_dict_exp4_020126_2.json"),
            inference_file = joinpath(ROOT, "data", "inference", "inference_exp4_020126_1.jld2"),
            problem_dir = joinpath(ROOT, "dataset", "problems_exp4_013026"),
        )
    else
        return (
            steps_file = "",
            inference_file = "",
            problem_dir = "",
        )
    end
end

function default_human_costs_path(exp::String)
    return joinpath(ROOT, "data_processing", "outputs", "human_costs", "$(exp)_human_costs.json")
end

function parse_real_opt(opts::Dict{String, String}, key::String)
    haskey(opts, key) || return nothing
    return parse(Float64, opts[key])
end

function default_action_cost(exp::String, steps_file::String)
    return Dict{Symbol, Real}(
        :move => 3.0,
        :interact => 5.0,
        :observe => 1.0,
    )
end

function resolve_action_cost(opts::Dict{String, String}, exp::String, steps_file::String)
    action_cost = default_action_cost(exp, steps_file)

    move_cost = parse_real_opt(opts, "move-cost")
    interact_cost = parse_real_opt(opts, "interact-cost")
    observe_cost = parse_real_opt(opts, "observe-cost")

    if move_cost !== nothing
        action_cost[:move] = move_cost
    end
    if interact_cost !== nothing
        action_cost[:interact] = interact_cost
    end
    if observe_cost !== nothing
        action_cost[:observe] = observe_cost
    end

    return action_cost
end

toint(x) = x isa Integer ? Int(x) : Int(round(parse(Float64, string(x))))

function enumerate_beliefs_local(state)
    wizards = collect(PDDL.get_objects(state, :wizard))
    belief_states = Vector{typeof(state)}()
    belief_probs = Float64[]
    belief_names = String[]
    blue_wizards = [wizard for wizard in sort!(wizards, by=x -> string(x)) if state[pddl"(iscolor $wizard blue)"]]
    belief_cnt = max(length(blue_wizards), 1)

    for wizard in blue_wizards
        push!(belief_names, string(wizard))
        base_state = copy(state)
        assign!(base_state, wizard)
        push!(belief_states, base_state)
        push!(belief_probs, 1.0 / belief_cnt)
    end

    return belief_states, belief_probs, belief_names
end

function enumerate_beliefs_quiet(state)
    redirect_stdout(devnull) do
        return enumerate_beliefs_local(state)
    end
end

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

function resolve_model_label(opts::Dict{String, String}, steps_file::String)
    if haskey(opts, "model")
        return lowercase(opts["model"])
    end
    file_label = lowercase(basename(steps_file))
    if occursin("naive", file_label)
        return "naive_observer"
    elseif occursin("nonmental", file_label)
        return "rational_non_mentalizing"
    elseif occursin("mentalize", file_label)
        return "social_mentalizing"
    end
    return "full_model"
end

function exp1_human_candidates_for_reconstruction(model_label::String, model_key::String)
    normalized_key = normalize_exp1_map_key(model_key)
    return ["mod_$(normalized_key)_ascii"]
end

function human_candidates_for_reconstruction(exp::String, model_label::String, model_key::String)
    if exp == "exp1"
        return exp1_human_candidates_for_reconstruction(model_label, model_key)
    end
    return [model_key]
end

function load_human_per_case(human_costs_file::String)
    human_costs = JSON.parsefile(human_costs_file)
    if !haskey(human_costs, "per_case")
        error("Human costs file is missing per_case: $human_costs_file")
    end
    return human_costs["per_case"]
end

function filter_steps_dict_to_human_cases(
    steps_dict::Dict,
    exp::String,
    model_label::String,
    human_per_case::Dict,
)
    filtered = Dict{String, Any}()
    matched_human_keys = String[]

    for (raw_key, value) in steps_dict
        model_key = string(raw_key)
        human_key = nothing
        for candidate in human_candidates_for_reconstruction(exp, model_label, model_key)
            if haskey(human_per_case, candidate)
                human_key = candidate
                break
            end
        end

        if human_key !== nothing
            filtered[model_key] = value
            push!(matched_human_keys, human_key)
        end
    end

    return filtered, sort!(unique!(matched_human_keys))
end

is_mentalizing_model(model_label::String) = model_label in ("full_model", "social_mentalizing")
uses_latent_hypothesis_replay(model_label::String) = is_mentalizing_model(model_label)
uses_direct_evidence_candidates(model_label::String) = model_label in ("rational_non_mentalizing", "naive_observer")

function initial_replay_hypotheses(initial_states, model_label::String)
    if uses_latent_hypothesis_replay(model_label) || uses_direct_evidence_candidates(model_label)
        return [copy(s) for s in initial_states]
    end
    return typeof(initial_states)(undef, 0)
end

function normalize_exp1_map_key(map_key::String)
    if startswith(map_key, "mod_") && endswith(map_key, "_ascii")
        return replace(replace(map_key, "mod_" => "", count=1), "_ascii" => "")
    end
    return split(map_key, "_")[1]
end

function parse_map_scenario_key(map_key::String)
    m = match(r"^(.+)_scenario(\d+)$", map_key)
    m === nothing && error("Invalid map key format: $map_key")
    return m.captures[1], parse(Int, m.captures[2])
end

function get_t(entry)
    if entry isa Dict && haskey(entry, "t")
        return toint(entry["t"])
    end
    return toint(entry)
end

function normalize_observation_event(raw, obs_idx::Int)
    if raw isa Dict
        agent = if haskey(raw, "agent")
            string(raw["agent"])
        elseif haskey(raw, "observed_agent")
            string(raw["observed_agent"])
        else
            ""
        end
        if !isempty(agent)
            action = haskey(raw, "action") ? string(raw["action"]) : ""
            interaction_outcome = haskey(raw, "interaction_outcome") ? string(raw["interaction_outcome"]) : "none"
            return Dict(
                "observation_index" => obs_idx,
                "observed_agent" => agent,
                "action" => action,
                "interaction_outcome" => interaction_outcome,
            )
        end
    end
    if raw !== nothing
        return Dict(
            "observation_index" => obs_idx,
            "observed_agent" => string(raw),
            "action" => "",
            "interaction_outcome" => "none",
        )
    end
    return nothing
end

function get_observation_events(entry)
    if entry isa Dict
        if haskey(entry, "observations")
            raw_observations = entry["observations"]
            events = Any[]
            for (obs_idx, raw) in enumerate(raw_observations)
                event = normalize_observation_event(raw, obs_idx)
                event === nothing || push!(events, event)
            end
            return events
        elseif haskey(entry, "observation_events")
            raw_events = entry["observation_events"]
            events = Any[]
            for (obs_idx, raw) in enumerate(raw_events)
                event = normalize_observation_event(raw, obs_idx)
                event === nothing || push!(events, event)
            end
            return events
        end
    end
    return Any[]
end

function merge_step_and_replay_entry(step_entry, replay_entry)
    replay_entry === nothing && return step_entry

    merged = Dict{String, Any}()
    if step_entry isa Dict
        for (k, v) in step_entry
            merged[string(k)] = v
        end
    else
        merged["t"] = step_entry
    end

    if replay_entry isa Dict
        for (k, v) in replay_entry
            merged[string(k)] = v
        end
    else
        merged["t"] = replay_entry
    end

    return merged
end

function resolve_replay_observations(entry, model_label::String)
    observation_events = get_observation_events(entry)
    observations = [event["observed_agent"] for event in observation_events]
    t = get_t(entry)
    source = "sequence"
    warnings = String[]

    if isempty(observations) && t > 0 && !uses_latent_hypothesis_replay(model_label)
        observations = fill("agent2", t)
        observation_events = [
            Dict(
                "observation_index" => obs_idx,
                "observed_agent" => "agent2",
                "action" => "",
            ) for obs_idx in 1:t
        ]
        source = "count_fallback"
        push!(
            warnings,
            "Missing ordered observations; used recorded t=$t as a count-only placeholder for $model_label replay.",
        )
    end

    return observations, observation_events, t, source, warnings
end

function build_single_context(map_id, problem_dir)
    domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    problem = load_problem(joinpath(problem_dir, "$(map_id).pddl"))
    state = initstate(domain, problem)
    initial_states, _, belief_names = enumerate_beliefs_quiet(state)
    return (
        problem = problem,
        state = state,
        domain = domain,
        initial_states = initial_states,
        belief_names = belief_names,
    )
end

function build_multi_context(map_id, problem_dir)
    txt_path = joinpath(problem_dir, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    problem = load_ascii_problem(txt_path)
    state = initstate(domain, problem)
    full_blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    initial_states, _, belief_names = enumerate_beliefs_quiet(state)

    domain_a2 = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    tmp_a2 = joinpath(problem_dir, ".temp_agent2_$(map_id).txt")
    if !isfile(tmp_a2)
        filtered_a2 = filter_ascii_agents(ascii_content, :agent2)
        write(tmp_a2, filtered_a2)
    end
    problem_a2 = load_ascii_problem(tmp_a2)
    state_a2 = initstate(domain_a2, problem_a2)
    initial_states_a2, _, _ = enumerate_beliefs_quiet(state_a2)
    s_id_a2 = findfirst(s -> check_equal_state(state_a2, initial_states_a2[s]), 1:length(initial_states_a2))
    s_id_a2 = s_id_a2 === nothing ? -1 : s_id_a2

    domain_a3 = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    tmp_a3 = joinpath(problem_dir, ".temp_agent3_$(map_id).txt")
    if !isfile(tmp_a3)
        filtered_a3 = filter_ascii_agents(ascii_content, :agent3)
        write(tmp_a3, filtered_a3)
    end
    problem_a3 = load_ascii_problem(tmp_a3)
    state_a3 = initstate(domain_a3, problem_a3)
    initial_states_a3, _, _ = enumerate_beliefs_quiet(state_a3)
    s_id_a3 = findfirst(s -> check_equal_state(state_a3, initial_states_a3[s]), 1:length(initial_states_a3))
    s_id_a3 = s_id_a3 === nothing ? -1 : s_id_a3

    return (
        problem = problem,
        state = state,
        domain = domain,
        initial_states = initial_states,
        belief_names = belief_names,
        blue_wizards = full_blue_wizards,
        s_id_a2 = s_id_a2,
        s_id_a3 = s_id_a3,
    )
end

function build_agent1_context(map_id, problem_dir)
    txt_path = joinpath(problem_dir, "$(map_id).txt")
    ascii_content = read(txt_path, String)
    tmp_a1 = joinpath(problem_dir, ".temp_agent1_$(map_id).txt")
    if !isfile(tmp_a1)
        filtered_a1 = filter_ascii_agents(ascii_content, :agent1)
        write(tmp_a1, filtered_a1)
    end

    domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    problem = load_ascii_problem(tmp_a1)
    state = initstate(domain, problem)
    initial_states, _, belief_names = enumerate_beliefs_quiet(state)

    return (
        problem = problem,
        state = state,
        domain = domain,
        initial_states = initial_states,
        belief_names = belief_names,
    )
end

function candidate_support_at_count(blue_wizards, state_probs, obs_count::Int)
    max_t = size(state_probs, 2) - 1
    nrows = size(state_probs, 1)
    nw = min(length(blue_wizards), nrows)
    t_use = min(obs_count, max_t)
    if t_use <= 0
        return copy(blue_wizards), 0
    end

    idx = t_use + 1
    wizard_candidates = Any[]
    for j in 1:nw
        if state_probs[j, idx] > 0.1
            push!(wizard_candidates, blue_wizards[j])
        end
    end
    return wizard_candidates, t_use
end

function replay_wizard_candidates_single(blue_wizards, state_probs, t::Int)
    return candidate_support_at_count(blue_wizards, state_probs, t)
end

function replay_wizard_candidates_multi(blue_wizards, state_probs_agent2, state_probs_agent3, observations::AbstractVector, t::Int)
    wizard_candidates = copy(blue_wizards)
    t_use = min(t, length(observations))
    agent2_obs_count = 0
    agent3_obs_count = 0

    for k in 1:t_use
        obs_k = string(observations[k])
        current_support = nothing

        if obs_k == "agent2"
            agent2_obs_count += 1
            current_support, _ = candidate_support_at_count(blue_wizards, state_probs_agent2, agent2_obs_count)
        elseif obs_k == "agent3"
            agent3_obs_count += 1
            current_support, _ = candidate_support_at_count(blue_wizards, state_probs_agent3, agent3_obs_count)
        end

        if current_support !== nothing
            support_names = Set(string(w) for w in current_support)
            wizard_candidates = [w for w in wizard_candidates if string(w) in support_names]
        end
    end

    return wizard_candidates, t_use
end

function filter_hypotheses_by_candidates(initial_states, belief_names, candidate_wizards)
    isempty(candidate_wizards) && return [copy(s) for s in initial_states], String[]
    candidate_names = Set(string(w) for w in candidate_wizards)
    filtered_states = Any[]
    dropped = String[]
    for (idx, belief_name) in enumerate(belief_names)
        if belief_name in candidate_names
            push!(filtered_states, copy(initial_states[idx]))
        else
            push!(dropped, belief_name)
        end
    end
    if isempty(filtered_states)
        return [copy(s) for s in initial_states], String[]
    end
    return filtered_states, dropped
end

function select_hypotheses_by_belief_names(initial_states, belief_names, candidate_names)
    isempty(candidate_names) && return [copy(s) for s in initial_states], String[]

    candidate_set = Set(string(name) for name in candidate_names)
    filtered_states = Any[]
    dropped = String[]
    for (idx, belief_name) in enumerate(belief_names)
        if belief_name in candidate_set
            push!(filtered_states, copy(initial_states[idx]))
        else
            push!(dropped, belief_name)
        end
    end

    if isempty(filtered_states)
        return [copy(s) for s in initial_states], ["Reduced planning projection matched no belief states; used all hypotheses."]
    end

    warnings = isempty(dropped) ? String[] : ["Projected hypotheses into reduced planning world with $(length(filtered_states)) candidate states."]
    return filtered_states, warnings
end

function choose_shortest_plan(planner, domain, states, goal, explored_state_ids)
    candidate_indices = [i for i in 1:length(states) if !(i in explored_state_ids)]
    isempty(candidate_indices) && return 0, Term[]

    best_idx = 0
    best_plan = Term[]
    best_len = typemax(Int)

    if Threads.nthreads() <= 1 || length(candidate_indices) <= 1
        for i in candidate_indices
            plan = planner(domain, states[i], goal)
            if length(plan) < best_len
                best_idx = i
                best_plan = plan
                best_len = length(plan)
            end
        end
        return best_idx, best_plan
    end

    tasks = map(candidate_indices) do i
        Threads.@spawn begin
            plan = planner(domain, states[i], goal)
            return (i, plan, length(plan))
        end
    end

    for task in tasks
        idx, plan, len = fetch(task)
        if len < best_len
            best_idx = idx
            best_plan = plan
            best_len = len
        end
    end

    return best_idx, best_plan
end

function dynamic_state_signature(state::State)
    object_types = (:agent, :wizard, :key, :gem, :door)
    loc_terms = Tuple{String, Int, Int}[]
    for obj_type in object_types
        for obj in sort!(collect(PDDL.get_objects(state, obj_type)), by=x -> string(x))
            push!(loc_terms, (string(obj), Int(state[pddl"(xloc $obj)"]), Int(state[pddl"(yloc $obj)"])))
        end
    end

    has_terms = Tuple{String, String}[]
    agents = sort!(collect(PDDL.get_objects(state, :agent)), by=x -> string(x))
    carriables = vcat(
        sort!(collect(PDDL.get_objects(state, :key)), by=x -> string(x)),
        sort!(collect(PDDL.get_objects(state, :gem)), by=x -> string(x)),
    )
    for agent in agents
        for obj in carriables
            if state[pddl"(has $agent $obj)"]
                push!(has_terms, (string(agent), string(obj)))
            end
        end
    end

    hold_terms = Tuple{String, String}[]
    for wizard in sort!(collect(PDDL.get_objects(state, :wizard)), by=x -> string(x))
        for key in sort!(collect(PDDL.get_objects(state, :key)), by=x -> string(x))
            if state[pddl"(hold $wizard $key)"]
                push!(hold_terms, (string(wizard), string(key)))
            end
        end
    end

    hidden_terms = String[]
    offgrid_terms = String[]
    for obj in carriables
        if state[pddl"(hidden $obj)"]
            push!(hidden_terms, string(obj))
        end
        if state[pddl"(offgrid $obj)"]
            push!(offgrid_terms, string(obj))
        end
    end

    return (
        locs = Tuple(loc_terms),
        has = Tuple(has_terms),
        hold = Tuple(hold_terms),
        hidden = Tuple(hidden_terms),
        offgrid = Tuple(offgrid_terms),
    )
end

function equivalent_plan_state(state1::State, state2::State)
    dynamic_state_signature(state1) == dynamic_state_signature(state2)
end

function get_cached_plan!(domain, state::State, goal, cache_scope)
    goal_key = PDDL.write_pddl(goal)
    state_key = dynamic_state_signature(state)
    cache_key = (cache_scope, goal_key, state_key)

    lock(REPLAY_PLAN_CACHE_LOCK) do
        if haskey(REPLAY_PLAN_CACHE, cache_key)
            REPLAY_PLAN_CACHE_HITS[] += 1
            return copy(REPLAY_PLAN_CACHE[cache_key])
        end
        REPLAY_PLAN_CACHE_MISSES[] += 1
    end

    planner = AStarPlanner(GoalManhattan())
    plan = collect(planner(domain, state, goal))
    lock(REPLAY_PLAN_CACHE_LOCK) do
        REPLAY_PLAN_CACHE[cache_key] = copy(plan)
    end
    return plan
end

function get_cached_naive_plan!(domain, state::State, goal, cache_scope)
    goal_key = PDDL.write_pddl(goal)
    state_key = dynamic_state_signature(state)
    cache_key = (cache_scope, :naive, goal_key, state_key)

    lock(REPLAY_PLAN_CACHE_LOCK) do
        if haskey(REPLAY_PLAN_CACHE, cache_key)
            REPLAY_PLAN_CACHE_HITS[] += 1
            return copy(REPLAY_PLAN_CACHE[cache_key])
        end
        REPLAY_PLAN_CACHE_MISSES[] += 1
    end

    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    planner = NaivePlanner(blue_wizards, :agent1, AStarPlanner(GoalManhattan()), domain)
    plan = collect(planner(domain, state, goal))
    lock(REPLAY_PLAN_CACHE_LOCK) do
        REPLAY_PLAN_CACHE[cache_key] = copy(plan)
    end
    return plan
end

function classify_action(act::Term)
    if act.name == :interact
        return "interact"
    elseif act.name == :observe
        return "observe"
    else
        return "move"
    end
end

function replay_case(domain, state, goal, initial_states, observation_trace, action_cost; cache_scope=nothing, case_label="")
    true_state = copy(state)
    hypotheses = [copy(s) for s in initial_states]
    observations = [string(x) for x in observation_trace]
    observe_steps = length(observations)
    observe_cost = action_cost[:observe] * observe_steps
    warnings = String[]
    replay_start_time = time()
    last_heartbeat_time = replay_start_time
    replan_count = 0
    initial_plan_time = 0.0
    replan_time_total = 0.0

    function timed_get_cached_plan!(state_for_plan, bucket::Symbol)
        start_time = time()
        plan = get_cached_plan!(domain, state_for_plan, goal, cache_scope)
        elapsed = time() - start_time
        if bucket == :initial
            initial_plan_time += elapsed
        else
            replan_time_total += elapsed
        end
        return plan
    end

    function timed_choose_shortest_plan(states_for_plan, explored_state_ids, bucket::Symbol)
        start_time = time()
        best_idx, plan = choose_shortest_plan(
            (d, s, g) -> get_cached_plan!(d, s, g, cache_scope),
            domain, states_for_plan, goal, explored_state_ids
        )
        elapsed = time() - start_time
        if bucket == :initial
            initial_plan_time += elapsed
        else
            replan_time_total += elapsed
        end
        return best_idx, plan
    end

    if isempty(hypotheses)
        plan = Term[]
        curr_state = copy(state)
        explored_state_ids = Set{Int}()
        push!(warnings, "No latent-state hypothesis available; skipped planning rather than using true-state planning.")
    else
        best_idx, plan = timed_choose_shortest_plan(hypotheses, Set{Int}(), :initial)
        if best_idx == 0
            plan = Term[]
            curr_state = copy(hypotheses[1])
            explored_state_ids = Set{Int}()
            push!(warnings, "No latent-state hypothesis plan available; skipped planning rather than using true-state planning.")
        else
            curr_state = copy(hypotheses[best_idx])
            explored_state_ids = Set([best_idx])
        end
    end

    if initial_plan_time >= 30
        label = isempty(case_label) ? "replay" : case_label
        println(
            "  planning [$label]: initial_plan=$(round(initial_plan_time, digits=2))s, " *
            "start_hypotheses=$(length(hypotheses))"
        )
    end

    executed_actions = String[]
    executed_terms = Term[]
    planning_action_counts = Dict("move" => 0, "interact" => 0, "observe" => 0)

    while !PDDL.satisfy(domain, true_state, goal)
        now = time()
        if now - last_heartbeat_time >= 30
            elapsed = round(now - replay_start_time, digits=2)
            label = isempty(case_label) ? "replay" : case_label
            println(
                "  heartbeat [$label]: elapsed=$(elapsed)s, executed=$(length(executed_actions)), " *
                "remaining_hypotheses=$(length(hypotheses)), replans=$(replan_count)"
            )
            last_heartbeat_time = now
        end

        if isempty(plan)
            plan = timed_get_cached_plan!(curr_state, :replan)
            if isempty(plan)
                best_idx, plan = timed_choose_shortest_plan(hypotheses, explored_state_ids, :replan)
                replan_count += 1
                if best_idx > 0
                    push!(explored_state_ids, best_idx)
                    curr_state = copy(hypotheses[best_idx])
                else
                    push!(warnings, "Exhausted hypotheses during replay; stopped planning rather than using true-state replanning.")
                end
            end
            isempty(plan) && break
        end

        action = plan[1]
        if !PDDL.available(domain, true_state, action)
            best_idx, plan = timed_choose_shortest_plan(hypotheses, explored_state_ids, :replan)
            replan_count += 1
            if best_idx > 0
                push!(explored_state_ids, best_idx)
                curr_state = copy(hypotheses[best_idx])
                push!(warnings, "Replanned before execution because $(PDDL.write_pddl(action)) was not applicable in the true state.")
            else
                plan = Term[]
                push!(warnings, "Stopped planning because $(PDDL.write_pddl(action)) was not applicable in the true state and no latent-state hypothesis remained.")
                break
            end
            continue
        end
        if !PDDL.available(domain, curr_state, action)
            plan = timed_get_cached_plan!(curr_state, :replan)
            if isempty(plan)
                best_idx, plan = timed_choose_shortest_plan(hypotheses, explored_state_ids, :replan)
                replan_count += 1
                if best_idx > 0
                    push!(explored_state_ids, best_idx)
                    curr_state = copy(hypotheses[best_idx])
                    push!(warnings, "Current hypothesis could not execute $(PDDL.write_pddl(action)); replanned using another latent-state hypothesis.")
                else
                    push!(warnings, "Current hypothesis could not execute $(PDDL.write_pddl(action)); stopped planning rather than using true-state replanning.")
                end
            else
                push!(warnings, "Current plan state could not execute $(PDDL.write_pddl(action)); replanned within the current latent-state hypothesis.")
            end
            continue
        end
        push!(executed_terms, action)
        push!(executed_actions, PDDL.write_pddl(action))
        action_kind = classify_action(action)
        planning_action_counts[action_kind] += 1

        true_state = PDDL.execute(domain, true_state, action)
        curr_state = PDDL.execute(domain, curr_state, action)
        next_hypotheses = typeof(hypotheses)(undef, 0)
        pruned_hypotheses = 0
        for hypothesis in hypotheses
            if PDDL.available(domain, hypothesis, action)
                push!(next_hypotheses, PDDL.execute(domain, hypothesis, action))
            else
                pruned_hypotheses += 1
            end
        end
        hypotheses = next_hypotheses
        if pruned_hypotheses > 0
            explored_state_ids = Set{Int}()
            push!(warnings, "Pruned $pruned_hypotheses latent-state hypotheses because $(PDDL.write_pddl(action)) was not applicable.")
        end

        if action.name == :interact && !equivalent_plan_state(true_state, curr_state)
            best_idx, plan = timed_choose_shortest_plan(hypotheses, explored_state_ids, :replan)
            replan_count += 1
            if best_idx > 0
                push!(explored_state_ids, best_idx)
                curr_state = copy(hypotheses[best_idx])
            else
                plan = Term[]
                push!(warnings, "Interaction revealed mismatch after all hypotheses were explored; stopped planning rather than using true-state replanning.")
            end
        else
            plan = plan[2:end]
        end
    end

    planning_steps = length(executed_actions)
    planning_cost = calculate_plan_cost(executed_terms, action_cost)
    total_steps = observe_steps + planning_steps
    replay_elapsed = time() - replay_start_time
    replay_nonplanning_time = max(replay_elapsed - initial_plan_time - replan_time_total, 0.0)

    return Dict(
        "observe_steps" => observe_steps,
        "planning_steps" => planning_steps,
        "total_steps" => total_steps,
        "observe_cost" => observe_cost,
        "planning_cost" => planning_cost,
        "total_cost" => observe_cost + planning_cost,
        "executed_actions" => executed_actions,
        "planning_action_counts" => planning_action_counts,
        "initial_plan_time" => initial_plan_time,
        "replan_time_total" => replan_time_total,
        "replay_nonplanning_time" => replay_nonplanning_time,
        "replay_elapsed" => replay_elapsed,
        "warnings" => warnings,
    )
end

function replay_case_naive_like(domain, state, goal, observation_trace, action_cost; cache_scope=nothing, case_label="")
    true_state = copy(state)
    observations = [string(x) for x in observation_trace]
    observe_steps = length(observations)
    observe_cost = action_cost[:observe] * observe_steps
    replay_start_time = time()

    initial_plan_start = time()
    plan = get_cached_naive_plan!(domain, true_state, goal, cache_scope)
    initial_plan_time = time() - initial_plan_start

    executed_terms = copy(plan)
    executed_actions = [PDDL.write_pddl(action) for action in executed_terms]
    planning_action_counts = Dict("move" => 0, "interact" => 0, "observe" => 0)
    for action in executed_terms
        planning_action_counts[classify_action(action)] += 1
    end

    planning_steps = length(executed_actions)
    planning_cost = calculate_plan_cost(executed_terms, action_cost)
    total_steps = observe_steps + planning_steps
    replay_elapsed = time() - replay_start_time
    replay_nonplanning_time = max(replay_elapsed - initial_plan_time, 0.0)

    return Dict(
        "observe_steps" => observe_steps,
        "planning_steps" => planning_steps,
        "total_steps" => total_steps,
        "observe_cost" => observe_cost,
        "planning_cost" => planning_cost,
        "total_cost" => observe_cost + planning_cost,
        "executed_actions" => executed_actions,
        "planning_action_counts" => planning_action_counts,
        "initial_plan_time" => initial_plan_time,
        "replan_time_total" => 0.0,
        "replay_nonplanning_time" => replay_nonplanning_time,
        "replay_elapsed" => replay_elapsed,
        "warnings" => String[],
    )
end

function resolve_direct_evidence_candidates(state::State, candidate_names; had_direct_evidence::Bool=false)
    blue_wizards = [
        wizard for wizard in sort!(collect(PDDL.get_objects(state, :wizard)), by=x -> string(x))
        if state[pddl"(iscolor $wizard blue)"]
    ]
    if isempty(candidate_names)
        if had_direct_evidence
            return Const[], ["Direct-evidence interaction outcomes ruled out all blue wizard candidates."]
        end
        return blue_wizards, [
            "No direct-evidence wizard update was available; searched all blue wizard candidates."
        ]
    end

    candidate_set = Set(string(name) for name in candidate_names)
    selected = [wizard for wizard in blue_wizards if string(wizard) in candidate_set]
    if isempty(selected)
        return blue_wizards, [
            "Direct-evidence candidate set matched no blue wizards in the planning world; searched all blue wizard candidates."
        ]
    end

    warnings = length(selected) < length(blue_wizards) ? [
        "Direct-evidence planner restricted self-exploration to $(length(selected)) candidate wizards."
    ] : String[]
    return selected, warnings
end

function replay_case_candidate_search(domain, state, goal, candidate_names, observation_trace, action_cost; cache_scope=nothing, case_label="", had_direct_evidence::Bool=false)
    current_state = copy(state)
    observations = [string(x) for x in observation_trace]
    observe_steps = length(observations)
    observe_cost = action_cost[:observe] * observe_steps
    warnings = String[]
    replay_start_time = time()
    planning_time_total = 0.0

    candidate_wizards, candidate_warnings = resolve_direct_evidence_candidates(current_state, candidate_names; had_direct_evidence=had_direct_evidence)
    append!(warnings, candidate_warnings)
    all_blue_wizards = [
        wizard for wizard in sort!(collect(PDDL.get_objects(current_state, :wizard)), by=x -> string(x))
        if current_state[pddl"(iscolor $wizard blue)"]
    ]
    broadened_beyond_direct_evidence = false

    fallback_planner = (d, s, g) -> get_cached_plan!(d, s, g, (cache_scope, :candidate_search))
    agent_name = :agent1
    agent_const = Const(agent_name)
    blue_keys = [k for k in PDDL.get_objects(current_state, :key) if current_state[pddl"(iscolor $k blue)"]]
    blue_key = isempty(blue_keys) ? nothing : blue_keys[1]
    visited_wizards = Set{Const}()
    for wizard in candidate_wizards
        if current_state[Compound(:visited, [agent_const, wizard])]
            push!(visited_wizards, wizard)
        end
    end

    executed_terms = Term[]
    executed_actions = String[]
    planning_action_counts = Dict("move" => 0, "interact" => 0, "observe" => 0)

    function execute_action!(action::Term)
        push!(executed_terms, action)
        push!(executed_actions, PDDL.write_pddl(action))
        planning_action_counts[classify_action(action)] += 1
        current_state = PDDL.execute(domain, current_state, action)
        return current_state
    end

    function timed_cached_plan(state_for_plan, goal_for_plan)
        start_time = time()
        plan = fallback_planner(domain, state_for_plan, goal_for_plan)
        planning_time_total += time() - start_time
        return copy(plan)
    end

    while !PDDL.satisfy(domain, current_state, goal)
        if blue_key !== nothing && current_state[pddl"(has $agent_name $blue_key)"]
            plan_to_goal = timed_cached_plan(current_state, goal)
            if isempty(plan_to_goal)
                push!(warnings, "Candidate-search planner could not find a goal-reaching plan after obtaining the blue key.")
                break
            end
            goal_plan_failed = false
            for action in plan_to_goal
                if !PDDL.available(domain, current_state, action)
                    push!(warnings, "Candidate-search goal plan became unavailable in the executed state.")
                    goal_plan_failed = true
                    break
                end
                current_state = execute_action!(action)
            end
            goal_plan_failed && break
            continue
        end

        agent_loc = get_obj_loc(current_state, agent_const)
        closest_wizard = nothing
        closest_wizard_loc = nothing
        min_dist = typemax(Int)
        for wizard in candidate_wizards
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
            if !broadened_beyond_direct_evidence && length(candidate_wizards) < length(all_blue_wizards)
                candidate_wizards = all_blue_wizards
                broadened_beyond_direct_evidence = true
                push!(
                    warnings,
                    "Direct-evidence candidate subset was exhausted before reaching the goal; broadened self-exploration to all remaining blue wizards."
                )
                continue
            end
            push!(warnings, "Candidate-search planner exhausted all candidate wizards before reaching the goal.")
            break
        end

        plan_to_wizard = plan_to_wizard_location_naive(domain, current_state, closest_wizard_loc, agent_name, fallback_planner)
        if isempty(plan_to_wizard) && min_dist != 1
            push!(visited_wizards, closest_wizard)
            push!(warnings, "Candidate-search planner could not reach $(string(closest_wizard)); removed it and continued.")
            continue
        end

        move_plan_failed = false
        for action in plan_to_wizard
            if !PDDL.available(domain, current_state, action)
                push!(warnings, "Candidate-search movement plan toward $(string(closest_wizard)) became unavailable in the executed state.")
                move_plan_failed = true
                break
            end
            current_state = execute_action!(action)
        end
        if move_plan_failed
            push!(visited_wizards, closest_wizard)
            continue
        end

        interact_action = PDDL.parse_pddl("(interact $agent_name $closest_wizard)")
        if !PDDL.available(domain, current_state, interact_action)
            push!(visited_wizards, closest_wizard)
            push!(warnings, "Candidate-search planner reached $(string(closest_wizard)) but could not interact with it.")
            continue
        end

        current_state = execute_action!(interact_action)
        push!(visited_wizards, closest_wizard)
    end

    planning_steps = length(executed_actions)
    planning_cost = calculate_plan_cost(executed_terms, action_cost)
    total_steps = observe_steps + planning_steps
    replay_elapsed = time() - replay_start_time
    replay_nonplanning_time = max(replay_elapsed - planning_time_total, 0.0)

    return Dict(
        "observe_steps" => observe_steps,
        "planning_steps" => planning_steps,
        "total_steps" => total_steps,
        "observe_cost" => observe_cost,
        "planning_cost" => planning_cost,
        "total_cost" => observe_cost + planning_cost,
        "executed_actions" => executed_actions,
        "planning_action_counts" => planning_action_counts,
        "initial_plan_time" => planning_time_total,
        "replan_time_total" => 0.0,
        "replay_nonplanning_time" => replay_nonplanning_time,
        "replay_elapsed" => replay_elapsed,
        "warnings" => warnings,
    )
end

function direct_evidence_candidate_names(belief_names, observation_events)
    belief_name_set = Set(string(name) for name in belief_names)
    remaining_candidates = Set(string(name) for name in belief_names)
    warnings = String[]
    saw_action_detail = false
    saw_interaction = false
    saw_direct_evidence = false

    for event in observation_events
        action_str = get(event, "action", "")
        isempty(action_str) && continue
        saw_action_detail = true
        action = PDDL.parse_pddl(action_str)
        if action isa Term && action.name == :interact
            saw_interaction = true
            wizard_name = string(action.args[end])
            outcome = lowercase(strip(string(get(event, "interaction_outcome", "none"))))

            if outcome == "blue_amulet_present"
                saw_direct_evidence = true
                if wizard_name in belief_name_set
                    empty!(remaining_candidates)
                    push!(remaining_candidates, wizard_name)
                else
                    push!(warnings, "Replay trace reported blue amulet at unknown wizard $(wizard_name); ignored that outcome.")
                end
            elseif outcome == "blue_amulet_absent"
                saw_direct_evidence = true
                if wizard_name in remaining_candidates
                    delete!(remaining_candidates, wizard_name)
                end
            elseif outcome != "none" && !isempty(outcome)
                push!(warnings, "Replay trace had unrecognized interaction outcome '$outcome'; ignored it.")
            end
        end
    end

    if !saw_direct_evidence
        if saw_interaction
            push!(warnings, "Replay trace includes observed interactions but no explicit interaction outcomes; direct-evidence candidate update was unavailable.")
        elseif !saw_action_detail && !isempty(observation_events)
            push!(warnings, "Replay trace lacks action-level observations; direct-evidence candidate update was unavailable.")
        end
        return String[], warnings, false
    end

    candidate_names = sort!(collect(remaining_candidates))
    if isempty(candidate_names)
        push!(warnings, "Direct-evidence interaction outcomes ruled out all blue wizard candidates.")
    elseif length(candidate_names) < length(belief_names)
        push!(warnings, "Direct-evidence interaction outcomes reduced candidates to $(length(candidate_names)) blue wizards.")
    end

    sort!(candidate_names)
    return candidate_names, warnings, true
end

function mentalizing_candidates_exp1(ctx, map_key, t, inference_data)
    cache_key = ("exp1", map_key, t)
    return get_cached_posterior_filter!(cache_key) do
        state_dict = inference_data["state"]
        map_id = normalize_exp1_map_key(map_key)
        inference_map_id = "mod_$(map_id)_ascii"
        goal_id = parse(Int, string(ctx.problem.goal.args[2])[end:end])
        s_id = findfirst(s -> check_equal_state(ctx.state, ctx.initial_states[s]), 1:length(ctx.initial_states))
        s_id === nothing && return ctx.initial_states, String[], ["Could not locate current belief state; used all hypotheses."]

        blue_wizards = [PDDL.parse_pddl(name) for name in ctx.belief_names]
        state_probs = state_dict[inference_map_id][goal_id][s_id]
        candidate_wizards, _ = replay_wizard_candidates_single(blue_wizards, state_probs, t)
        filtered_states, dropped = filter_hypotheses_by_candidates(ctx.initial_states, ctx.belief_names, candidate_wizards)
        warnings = isempty(dropped) ? String[] : ["Filtered hypotheses after observations to $(length(filtered_states)) candidate states."]
        return filtered_states, [string(w) for w in candidate_wizards], warnings
    end
end

function mentalizing_candidates_exp2(ctx, map_key, t, inference_data, metadata)
    cache_key = ("exp2", map_key, t)
    return get_cached_posterior_filter!(cache_key) do
        state_dict = inference_data["state"]
        parts = split(map_key, "_")
        map_id = parts[1]
        scenario_idx = parse(Int, parts[2])
        g_id = toint(metadata[map_id][scenario_idx])
        s_id = findfirst(s -> check_equal_state(ctx.state, ctx.initial_states[s]), 1:length(ctx.initial_states))
        s_id === nothing && return ctx.initial_states, String[], ["Could not locate current belief state; used all hypotheses."]

        blue_wizards = [PDDL.parse_pddl(name) for name in ctx.belief_names]
        state_probs = state_dict[map_id][g_id][s_id]
        candidate_wizards, _ = replay_wizard_candidates_single(blue_wizards, state_probs, t)
        filtered_states, dropped = filter_hypotheses_by_candidates(ctx.initial_states, ctx.belief_names, candidate_wizards)
        warnings = isempty(dropped) ? String[] : ["Filtered hypotheses after observations to $(length(filtered_states)) candidate states."]
        return filtered_states, [string(w) for w in candidate_wizards], warnings
    end
end

function mentalizing_candidates_exp3_or_exp4(ctx, map_key, observations, inference_data, metadata, exp4_metadata_style)
    cache_key = ((exp4_metadata_style ? "exp4" : "exp3"), map_key, Tuple(observations))
    return get_cached_posterior_filter!(cache_key) do
        state_dict = inference_data["state"]
        map_id, scenario = parse_map_scenario_key(map_key)

        if exp4_metadata_style
            agent2_gem = toint(metadata[map_id]["agent2"][scenario]["gem"])
            agent3_gem = toint(metadata[map_id]["agent3"][scenario]["gem"])
        else
            agent2_gem = toint(metadata[map_id]["agent2"][scenario])
            agent3_gem = toint(metadata[map_id]["agent3"][scenario])
        end

        if ctx.s_id_a2 < 0 || ctx.s_id_a3 < 0
            return [copy(s) for s in ctx.initial_states], String[], ["Could not locate filtered belief state; used all hypotheses."]
        end

        state_probs_agent2 = state_dict["agent2"][map_id][scenario][agent2_gem][ctx.s_id_a2]
        state_probs_agent3 = state_dict["agent3"][map_id][scenario][agent3_gem][ctx.s_id_a3]
        candidate_wizards, _ = replay_wizard_candidates_multi(ctx.blue_wizards, state_probs_agent2, state_probs_agent3, observations, length(observations))
        filtered_states, dropped = filter_hypotheses_by_candidates(ctx.initial_states, ctx.belief_names, candidate_wizards)
        warnings = isempty(dropped) ? String[] : ["Filtered hypotheses after observations to $(length(filtered_states)) candidate states."]
        return filtered_states, [string(w) for w in candidate_wizards], warnings
    end
end

function reconstruct_exp1(steps_dict, replay_trace_dict, problem_dir, action_cost, model_label, inference_data)
    out = Dict{String, Any}()
    cache = Dict{String, Any}()
    reduced_cache = Dict{String, Any}()
    use_reduced_agent1_planning = true

    for (map_key, step_entry) in steps_dict
        entry = merge_step_and_replay_entry(step_entry, get(replay_trace_dict, string(map_key), nothing))
        observations, observation_events, t, observation_source, observation_warnings = resolve_replay_observations(entry, model_label)
        map_id = normalize_exp1_map_key(map_key)
        if !haskey(cache, map_id)
            cache[map_id] = build_single_context(map_id, problem_dir)
        end
        if use_reduced_agent1_planning && !haskey(reduced_cache, map_id)
            reduced_cache[map_id] = build_agent1_context(map_id, problem_dir)
        end

        clear_planner_cache!()
        ctx = cache[map_id]
        replay_ctx = use_reduced_agent1_planning ? reduced_cache[map_id] : ctx
        candidate_names = String[]
        warnings = copy(observation_warnings)
        if is_mentalizing_model(model_label)
            hypothesis_states = initial_replay_hypotheses(replay_ctx.initial_states, model_label)
            posterior_states, candidate_names, warnings = mentalizing_candidates_exp1(ctx, map_key, t, inference_data)
            if use_reduced_agent1_planning
                hypothesis_states, projection_warnings = select_hypotheses_by_belief_names(
                    replay_ctx.initial_states, replay_ctx.belief_names, candidate_names
                )
                warnings = vcat(warnings, projection_warnings)
            else
                hypothesis_states = posterior_states
            end
            warnings = vcat(observation_warnings, warnings)
            replay = replay_case(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, hypothesis_states, observations, action_cost;
                cache_scope=("exp1", map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
            )
        elseif uses_direct_evidence_candidates(model_label)
            candidate_names, direct_warnings, had_direct_evidence = direct_evidence_candidate_names(replay_ctx.belief_names, observation_events)
            warnings = vcat(warnings, direct_warnings)
            replay = replay_case_candidate_search(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, candidate_names, observations, action_cost;
                cache_scope=("exp1", map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
                had_direct_evidence=had_direct_evidence,
            )
        else
            replay = replay_case(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, replay_ctx.initial_states[1:0], observations, action_cost;
                cache_scope=("exp1", map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
            )
        end
        replay["warnings"] = vcat(warnings, replay["warnings"])

        out[map_key] = merge(
            Dict(
                "t_recorded" => t,
                "t_replayed" => length(observations),
                "observations_count" => length(observations),
                "observation_source" => observation_source,
                "observation_trace" => observations,
                "observation_events" => observation_events,
                "model_update_mode" => is_mentalizing_model(model_label) ? "mentalizing_posterior" : (uses_direct_evidence_candidates(model_label) ? "direct_observation_evidence" : "no_mentalizing_update"),
                "planning_world" => use_reduced_agent1_planning ? "agent1_reduced_ascii" : "full_single_agent_pddl",
                "posterior_candidates" => candidate_names,
                "reconstruction_mode" => "model_faithful_replay",
            ),
            replay,
        )
    end

    return out
end

function reconstruct_exp2(steps_dict, replay_trace_dict, problem_dir, action_cost, model_label, inference_data, metadata)
    out = Dict{String, Any}()
    cache = Dict{String, Any}()
    reduced_cache = Dict{String, Any}()
    use_reduced_agent1_planning = true

    for (map_key, step_entry) in steps_dict
        entry = merge_step_and_replay_entry(step_entry, get(replay_trace_dict, string(map_key), nothing))
        observations, observation_events, t, observation_source, observation_warnings = resolve_replay_observations(entry, model_label)
        map_id = split(map_key, "_")[1]
        if !haskey(cache, map_id)
            cache[map_id] = build_single_context(map_id, problem_dir)
        end
        if use_reduced_agent1_planning && !haskey(reduced_cache, map_id)
            reduced_cache[map_id] = build_agent1_context(map_id, problem_dir)
        end

        clear_planner_cache!()
        ctx = cache[map_id]
        replay_ctx = use_reduced_agent1_planning ? reduced_cache[map_id] : ctx
        candidate_names = String[]
        warnings = copy(observation_warnings)
        if is_mentalizing_model(model_label)
            hypothesis_states = initial_replay_hypotheses(replay_ctx.initial_states, model_label)
            posterior_states, candidate_names, warnings = mentalizing_candidates_exp2(ctx, map_key, t, inference_data, metadata)
            if use_reduced_agent1_planning
                hypothesis_states, projection_warnings = select_hypotheses_by_belief_names(
                    replay_ctx.initial_states, replay_ctx.belief_names, candidate_names
                )
                warnings = vcat(warnings, projection_warnings)
            else
                hypothesis_states = posterior_states
            end
            warnings = vcat(observation_warnings, warnings)
            replay = replay_case(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, hypothesis_states, observations, action_cost;
                cache_scope=("exp2", map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
            )
        elseif uses_direct_evidence_candidates(model_label)
            candidate_names, direct_warnings, had_direct_evidence = direct_evidence_candidate_names(replay_ctx.belief_names, observation_events)
            warnings = vcat(warnings, direct_warnings)
            replay = replay_case_candidate_search(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, candidate_names, observations, action_cost;
                cache_scope=("exp2", map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
                had_direct_evidence=had_direct_evidence,
            )
        else
            replay = replay_case(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, replay_ctx.initial_states[1:0], observations, action_cost;
                cache_scope=("exp2", map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
            )
        end
        replay["warnings"] = vcat(warnings, replay["warnings"])

        out[map_key] = merge(
            Dict(
                "t_recorded" => t,
                "t_replayed" => length(observations),
                "observations_count" => length(observations),
                "observation_source" => observation_source,
                "observation_trace" => observations,
                "observation_events" => observation_events,
                "model_update_mode" => is_mentalizing_model(model_label) ? "mentalizing_posterior" : (uses_direct_evidence_candidates(model_label) ? "direct_observation_evidence" : "no_mentalizing_update"),
                "planning_world" => use_reduced_agent1_planning ? "agent1_reduced_ascii" : "full_single_agent_pddl",
                "posterior_candidates" => candidate_names,
                "reconstruction_mode" => "model_faithful_replay",
            ),
            replay,
        )
    end

    return out
end

function reconstruct_exp3_or_exp4(steps_dict, replay_trace_dict, problem_dir, action_cost, model_label, inference_data, metadata; exp4_metadata_style=false, disable_exp4_interaction_outcome_pruning=false)
    out = Dict{String, Any}()
    cache = Dict{String, Any}()
    reduced_cache = Dict{String, Any}()
    total_cases = length(steps_dict)
    case_idx = 0
    use_reduced_agent1_planning = true

    for (map_key, step_entry) in steps_dict
        case_idx += 1
        case_start_time = time()
        map_id, _ = parse_map_scenario_key(map_key)
        entry = merge_step_and_replay_entry(step_entry, get(replay_trace_dict, string(map_key), nothing))
        observations, observation_events, t, observation_source, observation_warnings = resolve_replay_observations(entry, model_label)
        println("[$case_idx/$total_cases] starting $map_key (t=$(t), obs_len=$(length(observations)))")

        if isempty(observations) && t > 0 && uses_latent_hypothesis_replay(model_label)
            error("Missing ordered observations for $map_key; cannot replay without a sequence.")
        end

        if !haskey(cache, map_id)
            cache[map_id] = build_multi_context(map_id, problem_dir)
        end
        if use_reduced_agent1_planning && !haskey(reduced_cache, map_id)
            reduced_cache[map_id] = build_agent1_context(map_id, problem_dir)
        end

        clear_planner_cache!()
        ctx = cache[map_id]
        replay_ctx = use_reduced_agent1_planning ? reduced_cache[map_id] : ctx
        candidate_names = String[]
        warnings = copy(observation_warnings)
        posterior_filter_start = time()
        if is_mentalizing_model(model_label)
            hypothesis_states = initial_replay_hypotheses(replay_ctx.initial_states, model_label)
            posterior_states, candidate_names, warnings = mentalizing_candidates_exp3_or_exp4(
                ctx, map_key, observations, inference_data, metadata, exp4_metadata_style
            )
            if use_reduced_agent1_planning
                hypothesis_states, projection_warnings = select_hypotheses_by_belief_names(
                    replay_ctx.initial_states, replay_ctx.belief_names, candidate_names
                )
                warnings = vcat(warnings, projection_warnings)
            else
                hypothesis_states = posterior_states
            end
            warnings = vcat(observation_warnings, warnings)
            replay = replay_case(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, hypothesis_states, observations, action_cost;
                cache_scope=((exp4_metadata_style ? "exp4" : "exp3"), map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
                case_label=map_key,
            )
        elseif uses_direct_evidence_candidates(model_label)
            if exp4_metadata_style && disable_exp4_interaction_outcome_pruning
                candidate_names = String[]
                direct_warnings = ["Exp4 interaction-outcome pruning disabled for direct-evidence reconstruction; searched all blue wizard candidates."]
                had_direct_evidence = false
            else
                candidate_names, direct_warnings, had_direct_evidence = direct_evidence_candidate_names(replay_ctx.belief_names, observation_events)
            end
            warnings = vcat(warnings, direct_warnings)
            replay = replay_case_candidate_search(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, candidate_names, observations, action_cost;
                cache_scope=((exp4_metadata_style ? "exp4" : "exp3"), map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
                case_label=map_key,
                had_direct_evidence=had_direct_evidence,
            )
        else
            replay = replay_case(
                replay_ctx.domain, replay_ctx.state, replay_ctx.problem.goal, replay_ctx.initial_states[1:0], observations, action_cost;
                cache_scope=((exp4_metadata_style ? "exp4" : "exp3"), map_id, model_label, use_reduced_agent1_planning ? "agent1_reduced" : "full_world"),
                case_label=map_key,
            )
        end
        posterior_filter_time = time() - posterior_filter_start
        warnings = vcat(observation_warnings, warnings, replay["warnings"])
        if t != length(observations)
            push!(warnings, "Recorded t=$t does not match observations length=$(length(observations)); used observations length.")
        end
        replay["warnings"] = warnings

        out[map_key] = merge(
            Dict(
                "t_recorded" => t,
                "t_replayed" => length(observations),
                "observations_count" => length(observations),
                "observation_source" => observation_source,
                "observation_trace" => observations,
                "observation_events" => observation_events,
                "model_update_mode" => is_mentalizing_model(model_label) ? "mentalizing_posterior" : (uses_direct_evidence_candidates(model_label) ? "direct_observation_evidence" : "no_mentalizing_update"),
                "planning_world" => use_reduced_agent1_planning ? "agent1_reduced_ascii" : "full_multi_agent_ascii",
                "posterior_candidates" => candidate_names,
                "posterior_filter_time" => posterior_filter_time,
                "reconstruction_mode" => "model_faithful_replay",
            ),
            replay,
        )
        case_elapsed = time() - case_start_time
        plan_cache_stats = get_replay_plan_cache_stats()
        posterior_cache_stats = get_posterior_filter_cache_stats()
        plan_hits = plan_cache_stats["hits"]
        plan_misses = plan_cache_stats["misses"]
        posterior_hits = posterior_cache_stats["hits"]
        posterior_misses = posterior_cache_stats["misses"]
        initial_plan_time = Float64(out[map_key]["initial_plan_time"])
        replan_time_total = Float64(out[map_key]["replan_time_total"])
        replay_nonplanning_time = Float64(out[map_key]["replay_nonplanning_time"])
        println(
            "[$case_idx/$total_cases] $map_key completed in $(round(case_elapsed, digits=2))s " *
            "(filter=$(round(posterior_filter_time, digits=2))s, " *
            "initial_plan=$(round(initial_plan_time, digits=2))s, " *
            "replans=$(round(replan_time_total, digits=2))s, " *
            "other_replay=$(round(replay_nonplanning_time, digits=2))s, " *
            "plan cache: $(plan_hits)/$(plan_misses) hits/misses, " *
            "posterior cache: $(posterior_hits)/$(posterior_misses) hits/misses)"
        )
    end

    return out
end

function summarize(cost_dict)
    total = [Float64(v["total_cost"]) for v in values(cost_dict)]
    observe = [Float64(v["observe_cost"]) for v in values(cost_dict)]
    planning = [Float64(v["planning_cost"]) for v in values(cost_dict)]
    total_steps = [Float64(v["total_steps"]) for v in values(cost_dict)]
    observe_steps = [Float64(v["observe_steps"]) for v in values(cost_dict)]
    planning_steps = [Float64(v["planning_steps"]) for v in values(cost_dict)]
    warning_cases = count(v -> !isempty(v["warnings"]), values(cost_dict))
    return Dict(
        "n_cases" => length(total),
        "mean_total_cost" => isempty(total) ? 0.0 : sum(total) / length(total),
        "mean_observe_cost" => isempty(observe) ? 0.0 : sum(observe) / length(observe),
        "mean_planning_cost" => isempty(planning) ? 0.0 : sum(planning) / length(planning),
        "mean_total_steps" => isempty(total_steps) ? 0.0 : sum(total_steps) / length(total_steps),
        "mean_observe_steps" => isempty(observe_steps) ? 0.0 : sum(observe_steps) / length(observe_steps),
        "mean_planning_steps" => isempty(planning_steps) ? 0.0 : sum(planning_steps) / length(planning_steps),
        "cases_with_warnings" => warning_cases,
    )
end

function main()
    opts = parse_cli(ARGS)
    if !haskey(opts, "exp")
        usage()
        return
    end

    exp_requested = lowercase(opts["exp"])
    exp = normalize_exp(exp_requested)
    defaults = default_paths(exp)

    steps_file = get(opts, "steps-file", defaults.steps_file)
    replay_trace_file = get(opts, "replay-trace-file", "")
    model_label = resolve_model_label(opts, steps_file)
    inference_file = get(opts, "inference-file", defaults.inference_file)
    problem_dir = get(opts, "problem-dir", defaults.problem_dir)
    disable_exp4_interaction_outcome_pruning = get(opts, "disable-exp4-interaction-outcome-pruning", "false") == "true"
    restrict_to_human_levels = get(opts, "restrict-to-human-levels", "false") == "true"
    human_costs_file = get(opts, "human-costs-file", default_human_costs_path(exp))

    if isempty(steps_file) || !isfile(steps_file)
        error("Missing/invalid --steps-file: $steps_file")
    end
    if !isempty(replay_trace_file) && !isfile(replay_trace_file)
        error("Missing/invalid --replay-trace-file: $replay_trace_file")
    end
    if isempty(problem_dir) || !isdir(problem_dir)
        error("Missing/invalid --problem-dir: $problem_dir")
    end
    if is_mentalizing_model(model_label) && (isempty(inference_file) || !isfile(inference_file))
        error("Mentalizing reconstruction requires --inference-file: $inference_file")
    end

    steps_dict = JSON.parsefile(steps_file)
    replay_trace_dict = isempty(replay_trace_file) ? Dict{String, Any}() : JSON.parsefile(replay_trace_file)
    human_level_filter = Dict{String, Any}("enabled" => false)
    if restrict_to_human_levels
        if isempty(human_costs_file) || !isfile(human_costs_file)
            error("Missing/invalid --human-costs-file: $human_costs_file")
        end
        original_case_count = length(steps_dict)
        human_per_case = load_human_per_case(human_costs_file)
        filtered_steps_dict, matched_human_keys = filter_steps_dict_to_human_cases(
            steps_dict, exp, model_label, human_per_case
        )
        println(
            "Restricting to human levels: kept $(length(filtered_steps_dict)) / $(original_case_count) " *
            "cases from $human_costs_file"
        )
        steps_dict = filtered_steps_dict
        human_level_filter = Dict(
            "enabled" => true,
            "human_costs_file" => human_costs_file,
            "original_case_count" => original_case_count,
            "kept_case_count" => length(steps_dict),
            "matched_human_keys" => matched_human_keys,
        )
    end
    metadata = isfile(joinpath(problem_dir, "metadata.json")) ? JSON.parsefile(joinpath(problem_dir, "metadata.json")) : nothing
    clear_replay_plan_cache!()
    clear_posterior_filter_cache!()
    inference_data = is_mentalizing_model(model_label) ? Dict("state" => load(inference_file, "state")) : nothing
    action_cost = resolve_action_cost(opts, exp, steps_file)
    costs = if exp == "exp1"
        reconstruct_exp1(steps_dict, replay_trace_dict, problem_dir, action_cost, model_label, inference_data)
    elseif exp == "exp2"
        reconstruct_exp2(steps_dict, replay_trace_dict, problem_dir, action_cost, model_label, inference_data, metadata)
    elseif exp == "exp3"
        reconstruct_exp3_or_exp4(
            steps_dict,
            replay_trace_dict,
            problem_dir,
            action_cost,
            model_label,
            inference_data,
            metadata;
            exp4_metadata_style=false,
            disable_exp4_interaction_outcome_pruning=disable_exp4_interaction_outcome_pruning,
        )
    elseif exp == "exp4"
        reconstruct_exp3_or_exp4(
            steps_dict,
            replay_trace_dict,
            problem_dir,
            action_cost,
            model_label,
            inference_data,
            metadata;
            exp4_metadata_style=true,
            disable_exp4_interaction_outcome_pruning=disable_exp4_interaction_outcome_pruning,
        )
    else
        error("Unsupported --exp value: $exp")
    end

    out = Dict(
        "exp" => exp_requested,
        "exp_normalized" => exp,
        "model_label" => model_label,
        "steps_file" => steps_file,
        "replay_trace_file" => replay_trace_file,
        "inference_file" => inference_file,
        "problem_dir" => problem_dir,
        "disable_exp4_interaction_outcome_pruning" => disable_exp4_interaction_outcome_pruning,
        "human_level_filter" => human_level_filter,
        "reconstruction_mode" => "model_faithful_replay",
        "action_cost" => Dict(String(k) => v for (k, v) in action_cost),
        "replay_plan_cache_stats" => get_replay_plan_cache_stats(),
        "posterior_filter_cache_stats" => get_posterior_filter_cache_stats(),
        "summary" => summarize(costs),
        "per_case" => costs,
    )

    output_file = get(opts, "output-file", joinpath(dirname(steps_file), "reconstructed_costs_$(exp).json"))
    open(output_file, "w") do io
        JSON.print(io, out, 2)
    end

    println("Saved reconstructed costs to: $output_file")
    println("Summary: ", out["summary"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
