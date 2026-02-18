using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using JSON
using FileIO, JLD2

PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "beliefs.jl"))
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
            [--problem-dir <path/to/dataset/problems_...>]

        Notes:
        - For exp1/exp2/exp3, defaults are inferred if omitted.
        - For exp4, provide both --problem-dir and --inference-file to match the run.
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

toint(x) = x isa Integer ? Int(x) : Int(round(parse(Float64, string(x))))

function blue_wizards_from_state(state)
    [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
end

function enumerate_beliefs_silent(state)
    redirect_stdout(devnull) do
        return enumerate_beliefs(state)
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

function estimate_self_exploration_stats(domain::Any, state::State, agent_goal::Any, wizards::Any, action_cost::Dict{Symbol, Real})
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

function reconstruct_exp1(steps_dict, inference_file, problem_dir)
    data = load(inference_file)
    goal_dict = data["goal"]
    state_dict = data["state"]
    domain_render = load_domain(joinpath(ROOT, "dataset", "domain_render.pddl"))
    action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

    out = Dict{String, Any}()
    cache = Dict{String, Any}()

    for (map_key, t_raw) in steps_dict
        t = toint(t_raw)
        parts = split(map_key, "_")
        map_id = parts[1]
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

function reconstruct_exp2(steps_dict, inference_file, problem_dir)
    data = load(inference_file)
    goal_dict = data["goal"]
    state_dict = data["state"]
    metadata = JSON.parsefile(joinpath(problem_dir, "metadata.json"))
    domain_render = load_domain(joinpath(ROOT, "dataset", "domain_render.pddl"))
    action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

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

function get_t(entry)
    if entry isa Dict && haskey(entry, "t")
        return toint(entry["t"])
    end
    return toint(entry)
end

function reconstruct_exp3_or_exp4(steps_dict, inference_file, problem_dir; exp4_metadata_style=false)
    data = load(inference_file)
    goal_dict = data["goal"]
    state_dict = data["state"]
    metadata = JSON.parsefile(joinpath(problem_dir, "metadata.json"))
    domain_render = load_domain(joinpath(ROOT, "dataset", "domain_render.pddl"))
    action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

    out = Dict{String, Any}()
    map_cache = Dict{String, Any}()

    for (map_key, entry) in steps_dict
        map_id, scenario = parse_map_scenario_key(map_key)
        t = get_t(entry)
        observations = get_obs_list(entry)

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
    costs = if exp == "exp1"
        reconstruct_exp1(steps_dict, inference_file, problem_dir)
    elseif exp == "exp2"
        reconstruct_exp2(steps_dict, inference_file, problem_dir)
    elseif exp == "exp3"
        reconstruct_exp3_or_exp4(steps_dict, inference_file, problem_dir; exp4_metadata_style=false)
    elseif exp == "exp4"
        reconstruct_exp3_or_exp4(steps_dict, inference_file, problem_dir; exp4_metadata_style=true)
    else
        error("Unsupported --exp value: $exp")
    end

    out = Dict(
        "exp" => exp_requested,
        "exp_normalized" => exp,
        "steps_file" => steps_file,
        "inference_file" => inference_file,
        "problem_dir" => problem_dir,
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
