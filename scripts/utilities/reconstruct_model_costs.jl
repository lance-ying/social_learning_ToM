using PDDL, SymbolicPlanners
using JSON
using FileIO, JLD2

PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl"))

const ROOT = joinpath(@__DIR__, "..", "..")

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
            --steps-file <path/to/steps_dict.json> \\
            [--output-file <path/to/output.json>] \\
            [--inference-file <path/to/inference.jld2>] \\
            [--problem-dir <path/to/dataset/problems_...>] \\
            [--move-cost <float>] [--interact-cost <float>] [--observe-cost <float>]

        Notes:
        - For exp1/exp2/exp3, defaults are inferred if omitted.
        - For exp4, provide both --problem-dir and --inference-file to match the run.
        - Action costs default from the experiment/steps file, but can be overridden.
        """
    )
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
    else
        return (
            steps_file = "",
            inference_file = "",
            problem_dir = "",
        )
    end
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

function blue_wizards_from_state(state)
    [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
end

function assign_blue_wizard!(state, wizard)
    for candidate in PDDL.get_objects(state, :wizard)
        state[pddl"(iscolor $candidate blue)"] = candidate == wizard
    end
    return state
end

function enumerate_beliefs_local(state)
    wizards = collect(PDDL.get_objects(state, :wizard))
    belief_states = Vector{typeof(state)}()
    belief_probs = Float64[]
    belief_names = String[]

    blue_wizards = [wizard for wizard in sort!(wizards, by=x -> string(x)) if state[pddl"(iscolor $wizard blue)"]]
    belief_cnt = length(blue_wizards)

    for wizard in blue_wizards
        push!(belief_names, string(wizard))
        base_state = copy(state)
        assign_blue_wizard!(base_state, wizard)
        push!(belief_states, base_state)
        push!(belief_probs, 1.0 / belief_cnt)
    end

    return belief_states, belief_probs, belief_names
end

function enumerate_beliefs_silent(state)
    redirect_stdout(devnull) do
        return enumerate_beliefs_local(state)
    end
end

function replay_wizard_candidates_single(blue_wizards, state_probs, t::Int)
    wizard_candidates = copy(blue_wizards)
    max_t = size(state_probs, 2) - 1
    nrows = size(state_probs, 1)
    nw = min(length(blue_wizards), nrows)
    t_use = min(t, max_t)
    for k in 1:t_use
        idx = k + 1
        wizard_candidates = Any[]
        for j in 1:nw
            if state_probs[j, idx] > 0.1
                push!(wizard_candidates, blue_wizards[j])
            end
        end
    end
    return wizard_candidates, t_use
end

function replay_wizard_candidates_multi(blue_wizards, state_probs_agent2, state_probs_agent3, observations::AbstractVector, t::Int)
    wizard_candidates = copy(blue_wizards)
    t_use = min(t, length(observations))
    nrows_agent2 = size(state_probs_agent2, 1)
    nrows_agent3 = size(state_probs_agent3, 1)
    nw_agent2 = min(length(blue_wizards), nrows_agent2)
    nw_agent3 = min(length(blue_wizards), nrows_agent3)
    for k in 1:t_use
        idx = k + 1
        wizard_candidates = Any[]
        obs_k = string(observations[k])
        if obs_k == "agent2"
            for j in 1:nw_agent2
                if idx <= size(state_probs_agent2, 2) && state_probs_agent2[j, idx] > 0.1
                    push!(wizard_candidates, blue_wizards[j])
                end
            end
        elseif obs_k == "agent3"
            for j in 1:nw_agent3
                if idx <= size(state_probs_agent3, 2) && state_probs_agent3[j, idx] > 0.1
                    push!(wizard_candidates, blue_wizards[j])
                end
            end
        end
    end
    return wizard_candidates, t_use
end

function estimate_self_exploration_stats(domain, state, agent_goal, wizards, action_cost)
    new_state = copy(state)
    planner = AStarPlanner(GoalManhattan())

    function get_plan(agent_x::Int, agent_y::Int, goal_x::Int, goal_y::Int)
        goal = pddl"(and (= (xloc agent1) $goal_x) (= (yloc agent1) $goal_y))"
        return collect(planner(domain, new_state, goal))
    end

    wizard_locs = [get_obj_loc(new_state, w) for w in wizards if state[pddl"(iscolor $w blue)"]]

    total_cost = 0.0
    planning_steps = 0

    for _ in 1:length(wizards)
        best_cost = Inf
        best_plan = Term[]
        best_loc = wizard_locs[1]

        agent_x = new_state[pddl"(xloc agent1)"]
        agent_y = new_state[pddl"(yloc agent1)"]

        for w_loc in wizard_locs
            x_loc, y_loc = w_loc
            plan = get_plan(agent_x, agent_y, x_loc, y_loc)
            plan_cost = calculate_plan_cost(plan, action_cost)
            if plan_cost < best_cost
                best_cost = plan_cost
                best_plan = plan
                best_loc = w_loc
            end
        end

        wizard_locs = filter!(loc -> ((loc[1] != best_loc[1]) || (loc[2] != best_loc[2])), wizard_locs)

        total_cost += best_cost
        total_cost += action_cost[:interact]
        planning_steps += length(best_plan) + 1

        new_state[pddl"(xloc agent1)"] = best_loc[1]
        new_state[pddl"(yloc agent1)"] = best_loc[2]
    end

    goal_loc = get_obj_loc(new_state, agent_goal.args[2])
    agent_x = new_state[pddl"(xloc agent1)"]
    agent_y = new_state[pddl"(yloc agent1)"]
    plan = get_plan(agent_x, agent_y, goal_loc[1], goal_loc[2])

    total_cost += calculate_plan_cost(plan, action_cost)
    planning_steps += length(plan)
    total_cost -= action_cost[:move] * (2 * length(wizards) - 1)

    return total_cost, planning_steps
end

function normalize_exp1_map_key(map_key::String)
    if startswith(map_key, "mod_") && endswith(map_key, "_ascii")
        return replace(replace(map_key, "mod_" => "", count=1), "_ascii" => "")
    end
    return split(map_key, "_")[1]
end

function reconstruct_exp1(steps_dict, inference_file, problem_dir, action_cost)
    data = load(inference_file)
    goal_dict = data["goal"]
    state_dict = data["state"]
    domain_render = load_domain(joinpath(ROOT, "dataset", "domain_render.pddl"))

    out = Dict{String, Any}()
    cache = Dict{String, Any}()

    for (map_key, t_raw) in steps_dict
        t = toint(t_raw)
        map_id = normalize_exp1_map_key(map_key)
        g_id = 1
        inference_map_id = "mod_$(map_id)_ascii"

        if !haskey(cache, map_id)
            domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
            problem = load_problem(joinpath(problem_dir, "$(map_id).pddl"))
            state = initstate(domain, problem)
            state_render = copy(state)
            initial_states, _, _ = enumerate_beliefs_silent(state)
            s_id = findfirst(s -> check_equal_state(state, initial_states[s]), 1:length(initial_states))
            s_id = s_id === nothing ? -1 : s_id

            cache[map_id] = (
                problem = problem,
                state_render = state_render,
                blue_wizards = blue_wizards_from_state(state),
                s_id = s_id,
            )
        end

        ctx = cache[map_id]
        state_probs = state_dict[inference_map_id][g_id][ctx.s_id]
        wizard_candidates, t_use = replay_wizard_candidates_single(ctx.blue_wizards, state_probs, t)
        planning_cost, planning_steps = estimate_self_exploration_stats(domain_render, copy(ctx.state_render), ctx.problem.goal, wizard_candidates, action_cost)
        observe_cost = action_cost[:observe] * t_use
        total_steps = planning_steps + t_use

        out[map_key] = Dict(
            "t_recorded" => t,
            "t_replayed" => t_use,
            "observe_steps" => t_use,
            "planning_steps" => planning_steps,
            "total_steps" => total_steps,
            "observe_cost" => observe_cost,
            "planning_cost" => planning_cost,
            "total_cost" => planning_cost + observe_cost,
            "wizard_candidates_count" => length(wizard_candidates),
        )
    end

    return out
end

function reconstruct_exp2(steps_dict, inference_file, problem_dir, action_cost)
    data = load(inference_file)
    goal_dict = data["goal"]
    state_dict = data["state"]
    metadata = JSON.parsefile(joinpath(problem_dir, "metadata.json"))
    domain_render = load_domain(joinpath(ROOT, "dataset", "domain_render.pddl"))

    out = Dict{String, Any}()
    cache = Dict{String, Any}()

    for (map_key, t_raw) in steps_dict
        t = toint(t_raw)
        parts = split(map_key, "_")
        map_id = parts[1]
        i = parse(Int, parts[2])
        g_id = toint(metadata[map_id][i])

        if !haskey(cache, map_id)
            domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
            problem = load_problem(joinpath(problem_dir, "$(map_id).pddl"))
            state = initstate(domain, problem)
            state_render = copy(state)
            initial_states, _, _ = enumerate_beliefs_silent(state)
            s_id = findfirst(s -> check_equal_state(state, initial_states[s]), 1:length(initial_states))
            s_id = s_id === nothing ? -1 : s_id

            cache[map_id] = (
                problem = problem,
                state_render = state_render,
                blue_wizards = blue_wizards_from_state(state),
                s_id = s_id,
            )
        end

        ctx = cache[map_id]
        state_probs = state_dict[map_id][g_id][ctx.s_id]
        wizard_candidates, t_use = replay_wizard_candidates_single(ctx.blue_wizards, state_probs, t)
        planning_cost, planning_steps = estimate_self_exploration_stats(domain_render, copy(ctx.state_render), ctx.problem.goal, wizard_candidates, action_cost)
        observe_cost = action_cost[:observe] * t_use
        total_steps = planning_steps + t_use

        out[map_key] = Dict(
            "t_recorded" => t,
            "t_replayed" => t_use,
            "observe_steps" => t_use,
            "planning_steps" => planning_steps,
            "total_steps" => total_steps,
            "observe_cost" => observe_cost,
            "planning_cost" => planning_cost,
            "total_cost" => planning_cost + observe_cost,
            "wizard_candidates_count" => length(wizard_candidates),
        )
    end

    return out
end

function build_multi_context(map_id, problem_dir)
    domain = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    problem = load_ascii_problem(joinpath(problem_dir, "$(map_id).txt"))
    state = initstate(domain, problem)
    state_render = copy(state)

    txt_path = joinpath(problem_dir, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    # Agent2 filtered
    domain_a2 = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    filtered_a2 = filter_ascii_agents(ascii_content, :agent2)
    tmp_a2 = tempname() * "_agent2.txt"
    write(tmp_a2, filtered_a2)
    problem_a2 = load_ascii_problem(tmp_a2)
    state_a2 = initstate(domain_a2, problem_a2)
    initial_states_a2, _, _ = enumerate_beliefs_silent(state_a2)
    s_id_a2 = findfirst(s -> check_equal_state(state_a2, initial_states_a2[s]), 1:length(initial_states_a2))
    s_id_a2 = s_id_a2 === nothing ? -1 : s_id_a2

    # Agent3 filtered
    domain_a3 = load_domain(joinpath(ROOT, "dataset", "domain.pddl"))
    filtered_a3 = filter_ascii_agents(ascii_content, :agent3)
    tmp_a3 = tempname() * "_agent3.txt"
    write(tmp_a3, filtered_a3)
    problem_a3 = load_ascii_problem(tmp_a3)
    state_a3 = initstate(domain_a3, problem_a3)
    initial_states_a3, _, _ = enumerate_beliefs_silent(state_a3)
    s_id_a3 = findfirst(s -> check_equal_state(state_a3, initial_states_a3[s]), 1:length(initial_states_a3))
    s_id_a3 = s_id_a3 === nothing ? -1 : s_id_a3

    return (
        problem = problem,
        state_render = state_render,
        blue_wizards = blue_wizards_from_state(state),
        s_id_a2 = s_id_a2,
        s_id_a3 = s_id_a3,
    )
end

function parse_map_scenario_key(map_key::String)
    m = match(r"^(.+)_scenario(\d+)$", map_key)
    m === nothing && error("Invalid map key format: $map_key")
    return m.captures[1], parse(Int, m.captures[2])
end

function get_obs_list(entry)
    if entry isa Dict && haskey(entry, "observations")
        return [string(x) for x in entry["observations"]]
    end
    return String[]
end

function get_count(entry, key::String)
    if entry isa Dict && haskey(entry, key)
        return Float64(entry[key])
    end
    return 0.0
end

function get_t(entry)
    if entry isa Dict && haskey(entry, "t")
        return toint(entry["t"])
    end
    return toint(entry)
end

function balanced_observations_from_counts(entry)
    t = get_t(entry)
    if t <= 0
        return String[]
    end

    agent2_raw = get_count(entry, "agent2_count")
    agent3_raw = get_count(entry, "agent3_count")
    total_raw = agent2_raw + agent3_raw

    if total_raw <= 0
        return String[]
    end

    scale = t / total_raw
    scaled_agent2 = agent2_raw * scale
    scaled_agent3 = agent3_raw * scale

    floors = Dict(
        "agent2" => floor(Int, scaled_agent2),
        "agent3" => floor(Int, scaled_agent3),
    )
    remainders = [
        ("agent2", scaled_agent2 - floors["agent2"]),
        ("agent3", scaled_agent3 - floors["agent3"]),
    ]

    remaining = t - floors["agent2"] - floors["agent3"]
    for (agent_name, _) in sort(remainders, by=x -> x[2], rev=true)
        if remaining <= 0
            break
        end
        floors[agent_name] += 1
        remaining -= 1
    end

    counts_left = Dict(
        "agent2" => floors["agent2"],
        "agent3" => floors["agent3"],
    )

    observations = String[]
    previous_agent = ""
    while length(observations) < t && (counts_left["agent2"] > 0 || counts_left["agent3"] > 0)
        candidates = sort(
            [(agent, counts_left[agent]) for agent in ("agent2", "agent3") if counts_left[agent] > 0],
            by=x -> (x[2], x[1] != previous_agent),
            rev=true,
        )
        chosen = candidates[1][1]
        push!(observations, chosen)
        counts_left[chosen] -= 1
        previous_agent = chosen
    end

    return observations
end

function reconstruct_exp3_or_exp4(steps_dict, inference_file, problem_dir, action_cost; exp4_metadata_style=false)
    data = load(inference_file)
    goal_dict = data["goal"]
    state_dict = data["state"]
    metadata = JSON.parsefile(joinpath(problem_dir, "metadata.json"))
    domain_render = load_domain(joinpath(ROOT, "dataset", "domain_render.pddl"))

    out = Dict{String, Any}()
    map_cache = Dict{String, Any}()

    for (map_key, entry) in steps_dict
        map_id, scenario = parse_map_scenario_key(map_key)
        t = get_t(entry)
        observations = get_obs_list(entry)
        synthesized_observations = false

        if isempty(observations) && entry isa Dict && (haskey(entry, "agent2_count") || haskey(entry, "agent3_count"))
            observations = balanced_observations_from_counts(entry)
            synthesized_observations = !isempty(observations)
        end

        if !haskey(map_cache, map_id)
            map_cache[map_id] = build_multi_context(map_id, problem_dir)
        end
        ctx = map_cache[map_id]

        if exp4_metadata_style
            agent2_gem = toint(metadata[map_id]["agent2"][scenario]["gem"])
            agent3_gem = toint(metadata[map_id]["agent3"][scenario]["gem"])
        else
            agent2_gem = toint(metadata[map_id]["agent2"][scenario])
            agent3_gem = toint(metadata[map_id]["agent3"][scenario])
        end

        state_probs_agent2 = state_dict["agent2"][map_id][scenario][agent2_gem][ctx.s_id_a2]
        state_probs_agent3 = state_dict["agent3"][map_id][scenario][agent3_gem][ctx.s_id_a3]

        wizard_candidates, t_use = replay_wizard_candidates_multi(
            ctx.blue_wizards, state_probs_agent2, state_probs_agent3, observations, t
        )
        planning_cost, planning_steps = estimate_self_exploration_stats(domain_render, copy(ctx.state_render), ctx.problem.goal, wizard_candidates, action_cost)
        observe_cost = action_cost[:observe] * t_use
        total_steps = planning_steps + t_use

        out[map_key] = Dict(
            "t_recorded" => t,
            "t_replayed" => t_use,
            "observe_steps" => t_use,
            "planning_steps" => planning_steps,
            "total_steps" => total_steps,
            "observe_cost" => observe_cost,
            "planning_cost" => planning_cost,
            "total_cost" => planning_cost + observe_cost,
            "wizard_candidates_count" => length(wizard_candidates),
            "observations_count" => length(observations),
            "observations_synthesized" => synthesized_observations,
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
    return Dict(
        "n_cases" => length(total),
        "mean_total_cost" => isempty(total) ? 0.0 : sum(total) / length(total),
        "mean_observe_cost" => isempty(observe) ? 0.0 : sum(observe) / length(observe),
        "mean_planning_cost" => isempty(planning) ? 0.0 : sum(planning) / length(planning),
        "mean_total_steps" => isempty(total_steps) ? 0.0 : sum(total_steps) / length(total_steps),
        "mean_observe_steps" => isempty(observe_steps) ? 0.0 : sum(observe_steps) / length(observe_steps),
        "mean_planning_steps" => isempty(planning_steps) ? 0.0 : sum(planning_steps) / length(planning_steps),
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
    inference_file = get(opts, "inference-file", defaults.inference_file)
    problem_dir = get(opts, "problem-dir", defaults.problem_dir)

    if isempty(steps_file) || !isfile(steps_file)
        error("Missing/invalid --steps-file: $steps_file")
    end

    if exp == "exp4" && (isempty(inference_file) || isempty(problem_dir))
        error("For exp4, provide both --inference-file and --problem-dir.")
    end
    if (exp == "exp1" || exp == "exp2" || exp == "exp3") && (!isfile(inference_file) || !isdir(problem_dir))
        error("Invalid inference/problem paths for $exp.\n  inference: $inference_file\n  problem_dir: $problem_dir")
    end

    steps_dict = JSON.parsefile(steps_file)
    action_cost = resolve_action_cost(opts, exp, steps_file)
    costs = if exp == "exp1"
        reconstruct_exp1(steps_dict, inference_file, problem_dir, action_cost)
    elseif exp == "exp2"
        reconstruct_exp2(steps_dict, inference_file, problem_dir, action_cost)
    elseif exp == "exp3"
        reconstruct_exp3_or_exp4(steps_dict, inference_file, problem_dir, action_cost; exp4_metadata_style=false)
    elseif exp == "exp4"
        reconstruct_exp3_or_exp4(steps_dict, inference_file, problem_dir, action_cost; exp4_metadata_style=true)
    else
        error("Unsupported --exp value: $exp")
    end

    out = Dict(
        "exp" => exp_requested,
        "exp_normalized" => exp,
        "steps_file" => steps_file,
        "inference_file" => inference_file,
        "problem_dir" => problem_dir,
        "action_cost" => Dict(String(k) => v for (k, v) in action_cost),
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

main()
