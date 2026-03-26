using JSON

include(joinpath(@__DIR__, "reconstruct_model_costs.jl"))

const AGENT1_NAIVE_LABEL = "agent1_naive_planner"
const ALL_EXPS = ("exp1", "exp2", "exp3", "exp4")

function default_agent1_naive_steps_file(exp::String)
    return joinpath(ROOT, "model_outputs", "baselines", exp, "step_dict_naive_observer.json")
end

function default_agent1_naive_replay_trace_file(exp::String)
    return joinpath(ROOT, "model_outputs", "baselines", exp, "replay_trace_naive_observer.json")
end

function reconstruct_agent1_naive_dedicated(exp::String, steps_dict, problem_dir, action_cost; exp4_metadata_style::Bool=false)
    out = Dict{String, Any}()
    reduced_cache = Dict{String, Any}()
    total_cases = length(steps_dict)
    case_idx = 0

    for (map_key, step_entry) in steps_dict
        case_idx += 1
        case_start_time = time()
        map_id = if exp == "exp1"
            normalize_exp1_map_key(map_key)
        elseif exp == "exp2"
            split(map_key, "_")[1]
        else
            first(parse_map_scenario_key(map_key))
        end

        if !haskey(reduced_cache, map_id)
            reduced_cache[map_id] = build_agent1_context(map_id, problem_dir)
        end

        raw_t = get_t(step_entry)
        println("[$case_idx/$total_cases] starting $map_key (t=$raw_t, obs_len=0)")

        clear_planner_cache!()
        replay_ctx = reduced_cache[map_id]
        warnings = String[]
        if raw_t > 0
            push!(warnings, "Ignored recorded observations (t=$raw_t) for non-observing planner reconstruction.")
        end

        replay = replay_case_naive_like(
            replay_ctx.domain,
            replay_ctx.state,
            replay_ctx.problem.goal,
            String[],
            action_cost;
            cache_scope=(exp, map_id, AGENT1_NAIVE_LABEL, "agent1_reduced_dedicated"),
            case_label=map_key,
            wizard_selection_mode=:candidate_search,
        )
        replay["warnings"] = vcat(warnings, replay["warnings"])

        case_payload = Dict(
            "t_recorded" => 0,
            "t_replayed" => 0,
            "observations_count" => 0,
            "observation_source" => "ignored_for_nonobserving_planner",
            "observation_trace" => String[],
            "observation_events" => Any[],
            "model_update_mode" => "agent1_naive_planner",
            "planning_world" => "agent1_reduced_ascii",
            "posterior_candidates" => String[],
            "reconstruction_mode" => "agent1_naive_dedicated",
        )
        if exp == "exp3" || exp == "exp4"
            case_payload["posterior_filter_time"] = 0.0
        end
        out[map_key] = merge(case_payload, replay)

        case_elapsed = time() - case_start_time
        plan_cache_stats = get_replay_plan_cache_stats()
        plan_hits = plan_cache_stats["hits"]
        plan_misses = plan_cache_stats["misses"]
        initial_plan_time = Float64(out[map_key]["initial_plan_time"])
        replan_time_total = Float64(out[map_key]["replan_time_total"])
        replay_nonplanning_time = Float64(out[map_key]["replay_nonplanning_time"])
        println(
            "[$case_idx/$total_cases] $map_key completed in $(round(case_elapsed, digits=2))s " *
            "(filter=0.0s, " *
            "initial_plan=$(round(initial_plan_time, digits=2))s, " *
            "replans=$(round(replan_time_total, digits=2))s, " *
            "other_replay=$(round(replay_nonplanning_time, digits=2))s, " *
            "plan cache: $(plan_hits)/$(plan_misses) hits/misses, posterior cache: 0/0 hits/misses)"
        )
    end

    return out
end

function usage_agent1_naive()
    println(
        """
        Usage:
          julia scripts/utilities/reconstruct_agent1_naive_planner.jl \\
            [--exp exp1|exp2|exp3|exp4|exp1,exp2,...] \\
            [--steps-file <path/to/steps_dict.json>] \\
            [--replay-trace-file <path/to/replay_trace.json>] \\
            [--restrict-to-human-levels true|false] [--human-costs-file <path/to/human_costs.json>] \\
            [--output-file <path/to/output.json>] \\
            [--problem-dir <path/to/dataset/problems_...>] \\
            [--move-cost <float>] [--interact-cost <float>] [--observe-cost <float>]

        Notes:
        - This dedicated runner ignores social observations for agent1_naive_planner.
        - It reconstructs directly in the reduced agent1 planning world.
        - Human-level filtering is enabled by default; pass --restrict-to-human-levels false to run all cases.
        - If --exp is omitted, it runs exp1, exp2, exp3, and exp4.
        """
    )
end

function parse_exp_list(raw_exp::String)
    cleaned = lowercase(strip(raw_exp))
    if cleaned == "" || cleaned == "all"
        return collect(ALL_EXPS)
    end
    requested = String[]
    for item in split(cleaned, ",")
        exp = normalize_exp(String(strip(item)))
        exp in ALL_EXPS || error("Unsupported --exp value: $item")
        exp in requested || push!(requested, exp)
    end
    isempty(requested) && error("No experiments selected")
    return requested
end

function run_single_exp_agent1_naive(opts::Dict{String, String}, exp_requested::String, exp::String)
    defaults = default_paths(exp)
    steps_file = get(opts, "steps-file", default_agent1_naive_steps_file(exp))
    replay_trace_file = get(opts, "replay-trace-file", default_agent1_naive_replay_trace_file(exp))
    problem_dir = get(opts, "problem-dir", defaults.problem_dir)
    human_costs_file = get(opts, "human-costs-file", default_human_costs_path(exp))
    human_level_filter = get(opts, "restrict-to-human-levels", "true") == "true"

    steps_dict = JSON.parsefile(steps_file)
    total_step_cases = length(steps_dict)
    matched_human_keys = String[]
    if human_level_filter
        human_per_case = load_human_per_case(human_costs_file)
        filtered_steps_dict, matched_human_keys = filter_steps_dict_to_human_cases(
            steps_dict, exp, AGENT1_NAIVE_LABEL, human_per_case
        )
        steps_dict = filtered_steps_dict
        println(
            "Restricting to human levels: kept $(length(steps_dict)) / $total_step_cases cases from $human_costs_file"
        )
    end

    clear_replay_plan_cache!()
    clear_posterior_filter_cache!()
    action_cost = resolve_action_cost(opts, exp, steps_file)
    costs = reconstruct_agent1_naive_dedicated(
        exp,
        steps_dict,
        problem_dir,
        action_cost;
        exp4_metadata_style=(exp == "exp4"),
    )

    out = Dict(
        "exp" => exp_requested,
        "exp_normalized" => exp,
        "model_label" => AGENT1_NAIVE_LABEL,
        "steps_file" => steps_file,
        "replay_trace_file" => replay_trace_file,
        "inference_file" => "",
        "problem_dir" => problem_dir,
        "disable_exp4_interaction_outcome_pruning" => false,
        "posterior_candidate_rule" => Dict(
            "rule" => "unused",
            "mass_threshold" => 0.0,
            "prob_threshold" => 0.0,
        ),
        "human_level_filter" => human_level_filter ? matched_human_keys : false,
        "reconstruction_mode" => "agent1_naive_dedicated",
        "action_cost" => Dict(String(k) => v for (k, v) in action_cost),
        "replay_plan_cache_stats" => get_replay_plan_cache_stats(),
        "posterior_filter_cache_stats" => Dict("hits" => 0, "misses" => 0, "hit_rate" => 0.0, "entries" => 0),
        "summary" => summarize(costs),
        "per_case" => costs,
    )

    output_file = get(
        opts,
        "output-file",
        joinpath(ROOT, "model_outputs", "reconstructed_costs", "$(exp)_$(AGENT1_NAIVE_LABEL).json"),
    )
    mkpath(dirname(output_file))
    open(output_file, "w") do io
        JSON.print(io, out, 2)
    end

    println("Saved reconstructed costs to: $output_file")
    println("Summary: ", out["summary"])
end

function main_agent1_naive()
    opts = parse_cli(ARGS)
    if haskey(opts, "help") || haskey(opts, "h")
        usage_agent1_naive()
        return
    end

    exp_requested = get(opts, "exp", "all")
    exps = parse_exp_list(exp_requested)
    if length(exps) > 1 && haskey(opts, "output-file")
        error("--output-file can only be used when reconstructing a single experiment")
    end

    for exp in exps
        println("==> $exp / $AGENT1_NAIVE_LABEL")
        run_single_exp_agent1_naive(opts, exp_requested, exp)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_agent1_naive()
end
