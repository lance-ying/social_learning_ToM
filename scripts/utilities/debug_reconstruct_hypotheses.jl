using JSON
using FileIO, JLD2

include(joinpath(@__DIR__, "reconstruct_model_costs.jl"))

function usage_debug()
    println(
        """
        Usage:
          julia scripts/utilities/debug_reconstruct_hypotheses.jl \\
            --exp exp1|exp2|exp3|exp4 \\
            [--model full_model|social_mentalizing|rational_non_mentalizing|naive_observer] \\
            [--steps-file <path/to/steps_dict.json>] \\
            [--inference-file <path/to/inference.jld2>] \\
            [--problem-dir <path/to/dataset/problems_...>] \\
            [--output-file <path/to/output.json>] \\
            [--case <single_map_key>]

        Notes:
        - Reports how many latent hypotheses survive before replay and how many
          unique dynamic-state signatures they correspond to.
        - Use this to estimate whether collapsing equivalent hypotheses would
          reduce replay work.
        """
    )
end

function count_unique_signatures(states)
    signatures = Set{Any}()
    for state in states
        push!(signatures, dynamic_state_signature(state))
    end
    return length(signatures)
end

function collect_case_metrics_exp1(steps_dict, problem_dir, model_label, inference_data)
    out = Dict{String, Any}()
    cache = Dict{String, Any}()

    for (map_key, t_raw) in steps_dict
        t = get_t(t_raw)
        map_id = normalize_exp1_map_key(map_key)
        if !haskey(cache, map_id)
            cache[map_id] = build_single_context(map_id, problem_dir)
        end

        ctx = cache[map_id]
        observations = fill("agent2", t)
        hypothesis_states = initial_replay_hypotheses(ctx.initial_states, model_label)
        candidate_names = String[]
        warnings = String[]
        if is_mentalizing_model(model_label)
            hypothesis_states, candidate_names, warnings = mentalizing_candidates_exp1(ctx, map_key, t, inference_data)
        end

        filtered_count = length(hypothesis_states)
        unique_count = count_unique_signatures(hypothesis_states)
        out[map_key] = Dict(
            "t_recorded" => t,
            "observations_count" => length(observations),
            "posterior_candidates_count" => length(candidate_names),
            "filtered_hypotheses_count" => filtered_count,
            "unique_signature_count" => unique_count,
            "duplicate_hypotheses_count" => filtered_count - unique_count,
            "warnings" => warnings,
        )
    end

    return out
end

function collect_case_metrics_exp2(steps_dict, problem_dir, model_label, inference_data, metadata)
    out = Dict{String, Any}()
    cache = Dict{String, Any}()

    for (map_key, t_raw) in steps_dict
        t = get_t(t_raw)
        map_id = split(map_key, "_")[1]
        if !haskey(cache, map_id)
            cache[map_id] = build_single_context(map_id, problem_dir)
        end

        ctx = cache[map_id]
        observations = fill("agent2", t)
        hypothesis_states = initial_replay_hypotheses(ctx.initial_states, model_label)
        candidate_names = String[]
        warnings = String[]
        if is_mentalizing_model(model_label)
            hypothesis_states, candidate_names, warnings = mentalizing_candidates_exp2(ctx, map_key, t, inference_data, metadata)
        end

        filtered_count = length(hypothesis_states)
        unique_count = count_unique_signatures(hypothesis_states)
        out[map_key] = Dict(
            "t_recorded" => t,
            "observations_count" => length(observations),
            "posterior_candidates_count" => length(candidate_names),
            "filtered_hypotheses_count" => filtered_count,
            "unique_signature_count" => unique_count,
            "duplicate_hypotheses_count" => filtered_count - unique_count,
            "warnings" => warnings,
        )
    end

    return out
end

function collect_case_metrics_exp3_or_exp4(steps_dict, problem_dir, model_label, inference_data, metadata; exp4_metadata_style=false)
    out = Dict{String, Any}()
    cache = Dict{String, Any}()

    for (map_key, entry) in steps_dict
        map_id, _ = parse_map_scenario_key(map_key)
        observations, t, observation_source, observation_warnings = resolve_replay_observations(entry, model_label)

        if isempty(observations) && t > 0 && uses_latent_hypothesis_replay(model_label)
            out[map_key] = Dict(
                "t_recorded" => t,
                "observations_count" => 0,
                "posterior_candidates_count" => 0,
                "filtered_hypotheses_count" => 0,
                "unique_signature_count" => 0,
                "duplicate_hypotheses_count" => 0,
                "warnings" => ["Missing ordered observations for mentalizing replay."],
            )
            continue
        end

        if !haskey(cache, map_id)
            cache[map_id] = build_multi_context(map_id, problem_dir)
        end

        ctx = cache[map_id]
        hypothesis_states = initial_replay_hypotheses(ctx.initial_states, model_label)
        candidate_names = String[]
        warnings = copy(observation_warnings)
        if is_mentalizing_model(model_label)
            hypothesis_states, candidate_names, posterior_warnings = mentalizing_candidates_exp3_or_exp4(
                ctx, map_key, observations, inference_data, metadata, exp4_metadata_style
            )
            append!(warnings, posterior_warnings)
        end

        filtered_count = length(hypothesis_states)
        unique_count = count_unique_signatures(hypothesis_states)
        out[map_key] = Dict(
            "t_recorded" => t,
            "observations_count" => length(observations),
            "observation_source" => observation_source,
            "posterior_candidates_count" => length(candidate_names),
            "filtered_hypotheses_count" => filtered_count,
            "unique_signature_count" => unique_count,
            "duplicate_hypotheses_count" => filtered_count - unique_count,
            "warnings" => warnings,
        )
    end

    return out
end

function summarize_metrics(case_metrics)
    values_list = collect(values(case_metrics))
    filtered = [Int(v["filtered_hypotheses_count"]) for v in values_list]
    unique = [Int(v["unique_signature_count"]) for v in values_list]
    duplicates = [Int(v["duplicate_hypotheses_count"]) for v in values_list]
    candidate_counts = [Int(v["posterior_candidates_count"]) for v in values_list]
    observation_counts = [Int(v["observations_count"]) for v in values_list]

    max_duplicate_case = nothing
    max_duplicate_count = -1
    for (case_key, metrics) in case_metrics
        dup = Int(metrics["duplicate_hypotheses_count"])
        if dup > max_duplicate_count
            max_duplicate_count = dup
            max_duplicate_case = case_key
        end
    end

    return Dict(
        "n_cases" => length(values_list),
        "mean_filtered_hypotheses" => isempty(filtered) ? 0.0 : sum(filtered) / length(filtered),
        "mean_unique_signatures" => isempty(unique) ? 0.0 : sum(unique) / length(unique),
        "mean_duplicate_hypotheses" => isempty(duplicates) ? 0.0 : sum(duplicates) / length(duplicates),
        "mean_posterior_candidates" => isempty(candidate_counts) ? 0.0 : sum(candidate_counts) / length(candidate_counts),
        "mean_observations_count" => isempty(observation_counts) ? 0.0 : sum(observation_counts) / length(observation_counts),
        "cases_with_duplicates" => count(>(0), duplicates),
        "total_duplicate_hypotheses" => sum(duplicates),
        "max_duplicate_case" => max_duplicate_case,
        "max_duplicate_hypotheses" => max_duplicate_count,
    )
end

function main_debug()
    opts = parse_cli(ARGS)
    if !haskey(opts, "exp")
        usage_debug()
        return
    end

    exp_requested = lowercase(opts["exp"])
    exp = normalize_exp(exp_requested)
    defaults = default_paths(exp)

    steps_file = get(opts, "steps-file", defaults.steps_file)
    model_label = resolve_model_label(opts, steps_file)
    inference_file = get(opts, "inference-file", defaults.inference_file)
    problem_dir = get(opts, "problem-dir", defaults.problem_dir)

    if isempty(steps_file) || !isfile(steps_file)
        error("Missing/invalid --steps-file: $steps_file")
    end
    if isempty(problem_dir) || !isdir(problem_dir)
        error("Missing/invalid --problem-dir: $problem_dir")
    end
    if is_mentalizing_model(model_label) && (isempty(inference_file) || !isfile(inference_file))
        error("Mentalizing inspection requires --inference-file: $inference_file")
    end

    steps_dict_full = JSON.parsefile(steps_file)
    steps_dict = if haskey(opts, "case")
        case_key = opts["case"]
        haskey(steps_dict_full, case_key) || error("Case not found in steps file: $case_key")
        Dict(case_key => steps_dict_full[case_key])
    else
        steps_dict_full
    end

    metadata = isfile(joinpath(problem_dir, "metadata.json")) ? JSON.parsefile(joinpath(problem_dir, "metadata.json")) : nothing
    clear_posterior_filter_cache!()
    inference_data = is_mentalizing_model(model_label) ? Dict("state" => load(inference_file, "state")) : nothing

    case_metrics = if exp == "exp1"
        collect_case_metrics_exp1(steps_dict, problem_dir, model_label, inference_data)
    elseif exp == "exp2"
        collect_case_metrics_exp2(steps_dict, problem_dir, model_label, inference_data, metadata)
    elseif exp == "exp3"
        collect_case_metrics_exp3_or_exp4(steps_dict, problem_dir, model_label, inference_data, metadata; exp4_metadata_style=false)
    elseif exp == "exp4"
        collect_case_metrics_exp3_or_exp4(steps_dict, problem_dir, model_label, inference_data, metadata; exp4_metadata_style=true)
    else
        error("Unsupported --exp value: $exp")
    end

    out = Dict(
        "exp" => exp_requested,
        "exp_normalized" => exp,
        "model_label" => model_label,
        "steps_file" => steps_file,
        "inference_file" => inference_file,
        "problem_dir" => problem_dir,
        "summary" => summarize_metrics(case_metrics),
        "per_case" => case_metrics,
    )

    output_file = get(opts, "output-file", joinpath(dirname(steps_file), "debug_hypotheses_$(exp)_$(model_label).json"))
    open(output_file, "w") do io
        JSON.print(io, out, 2)
    end

    println("Saved debug metrics to: $output_file")
    println("Summary: ", out["summary"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_debug()
end
