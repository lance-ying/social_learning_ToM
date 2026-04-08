using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JLD2, FileIO
using JSON

PDDL.Arrays.register!()

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DOMAIN_PATH = joinpath(REPO_ROOT, "dataset", "domain.pddl")
const DOMAIN_RENDER_PATH = joinpath(REPO_ROOT, "dataset", "domain_render.pddl")

include(joinpath(REPO_ROOT, "src", "plan_io.jl"))
include(joinpath(REPO_ROOT, "src", "utils.jl"))
include(joinpath(REPO_ROOT, "src", "heuristics.jl"))
include(joinpath(REPO_ROOT, "src", "beliefs.jl"))
include(joinpath(REPO_ROOT, "src", "translate.jl"))
include(joinpath(REPO_ROOT, "src", "render.jl"))
include(joinpath(REPO_ROOT, "src", "ascii.jl"))

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

serialize_wizards(wizards) = sort(string.(wizards))
serialize_observation(agent::String, action::Term, interaction_outcome::String = "none") = Dict(
    "agent" => agent,
    "action" => write_pddl(action),
    "interaction_outcome" => interaction_outcome,
)

function write_json(path::String, payload; indent::Int = 4)
    open(path, "w") do io
        JSON.print(io, payload, indent)
    end
end

function discover_map_ids(problem_dir::String)
    map_ids = Set{String}()
    for entry in readdir(problem_dir)
        if startswith(entry, ".") || endswith(entry, "_plan.pddl")
            continue
        end
        if endswith(entry, ".txt") || endswith(entry, ".pddl")
            push!(map_ids, splitext(entry)[1])
        end
    end
    return sort!(collect(map_ids))
end

function load_ascii_content(problem_dir::String, map_id::String)
    txt_path = joinpath(problem_dir, "$(map_id).txt")
    isfile(txt_path) || error("Expected ASCII problem file at $txt_path")
    return read(txt_path, String)
end

function load_full_problem(problem_dir::String, map_id::String)
    txt_path = joinpath(problem_dir, "$(map_id).txt")
    pddl_path = joinpath(problem_dir, "$(map_id).pddl")
    if isfile(txt_path)
        return load_ascii_problem(txt_path)
    elseif isfile(pddl_path)
        return load_problem(pddl_path)
    end
    error("No problem file found for $map_id in $problem_dir")
end

function require_agent1_goal(problem, map_id::String)
    if problem.goal == pddl"(true)"
        error(
            "Map $map_id does not define an agent1 goal. " *
            "In ASCII maps, mark the target gem with uppercase 'G' " *
            "instead of lowercase 'g'."
        )
    end
end

function load_filtered_ascii_problem(problem_dir::String, map_id::String, keep_agent::Symbol)
    ascii_content = load_ascii_content(problem_dir, map_id)
    filtered_ascii = filter_ascii_agents(ascii_content, keep_agent)
    temp_path = joinpath(problem_dir, ".temp_$(String(keep_agent))_$(map_id).txt")
    write(temp_path, filtered_ascii)
    return load_ascii_problem(temp_path)
end

function normalize_metadata(raw_metadata)
    metadata = Dict{String, Vector{Int}}()
    for (map_id, scenarios) in raw_metadata
        metadata[string(map_id)] = [Int(goal_id) for goal_id in scenarios]
    end
    return metadata
end

function generate_default_metadata(problem_dir::String, map_ids::Vector{String})
    metadata = Dict{String, Vector{Int}}()
    for map_id in map_ids
        domain = load_domain(DOMAIN_PATH)
        problem = load_filtered_ascii_problem(problem_dir, map_id, :agent2)
        state = initstate(domain, problem)
        goals, _ = initialize_goals(state, :agent2)
        metadata[map_id] = collect(1:length(goals))
    end
    return metadata
end

function load_or_generate_metadata(problem_dir::String)
    map_ids = discover_map_ids(problem_dir)
    isempty(map_ids) && error("No .txt or .pddl problems found in $problem_dir")

    metadata_path = joinpath(problem_dir, "metadata.json")
    if isfile(metadata_path)
        println("Using metadata from $metadata_path")
        return normalize_metadata(JSON.parsefile(metadata_path))
    end

    metadata = generate_default_metadata(problem_dir, map_ids)
    println("No metadata.json found. Generated default scenarios:")
    for map_id in sort!(collect(keys(metadata)))
        println("  $map_id => ", metadata[map_id])
    end
    return metadata
end

function agent_has_blue_item(state, agent_sym::Symbol)
    for key in PDDL.get_objects(state, :key)
        if state[pddl"(iscolor $key blue)"] && state[pddl"(has $agent_sym $key)"]
            return true
        end
    end
    return false
end

function interaction_outcome(state_before, state_after, agent_sym::Symbol, action::Term)
    if action.name != :interact
        return "none"
    end
    had_blue_before = agent_has_blue_item(state_before, agent_sym)
    has_blue_after = agent_has_blue_item(state_after, agent_sym)
    return (!had_blue_before && has_blue_after) ? "blue_amulet_present" : "blue_amulet_absent"
end

function run_inference(problem_dir::String, output_path::String, metadata::Dict{String, Vector{Int}})
    println("\n=== Running inference ===")

    goal_probs_conditioned_dict = Dict()
    state_probs_conditioned_dict = Dict()
    possible_worlds = Dict()

    for map_id in sort!(collect(keys(metadata)))
        println("\nMap: $map_id")
        goal_probs_conditioned_dict[map_id] = Dict()
        state_probs_conditioned_dict[map_id] = Dict()

        domain = load_domain(DOMAIN_PATH)
        problem = load_filtered_ascii_problem(problem_dir, map_id, :agent2)
        state = initstate(domain, problem)

        goals, goal_names = initialize_goals(state, :agent2)
        initial_states, belief_probs, state_names = enumerate_beliefs(state)
        possible_worlds[map_id] = initial_states

        @gen function goal_prior()
            goal_id ~ uniform_discrete(1, length(goals))
            return Specification(goals[goal_id])
        end

        @gen function state_prior()
            state_id ~ categorical(belief_probs)
            return initial_states[state_id]
        end

        init_state_addr = :init => :env => :state_id
        goal_addr = :init => :agent => :goal => :goal_id
        init_strata = choiceproduct(
            (goal_addr, 1:length(goals)),
            (init_state_addr, 1:length(initial_states)),
        )

        planner = RTHS(GoalManhattan(), n_iters = 2, max_nodes = 2^18)
        act_config = BoltzmannActConfig(0.5)

        agent_config = AgentConfig(
            domain, planner;
            goal_config = StaticGoalConfig(goal_prior),
            replan_args = (
                plan_at_init = true,
                prob_refine = 1.0,
                prob_replan = 0,
                rand_budget = false,
            ),
            act_config = act_config,
        )

        world_config = WorldConfig(
            agent_config = agent_config,
            env_config = PDDLEnvConfig(domain, state_prior),
        )

        for g in 1:length(goals)
            goal_probs_conditioned_dict[map_id][g] = Dict()
            state_probs_conditioned_dict[map_id][g] = Dict()

            for state_idx in 1:length(initial_states)
                state_i = initial_states[state_idx]
                plan = collect(AStarPlanner(GoalManhattan())(domain, state_i, goals[g]))
                println("  Goal $g, state $state_idx: $(length(plan)) steps")

                t_obs_iter = act_choicemap_pairs(plan)

                logger_cb = DataLoggerCallback(
                    t = (t, pf) -> t::Int,
                    goal_probs = pf -> probvec(pf, goal_addr, 1:length(goals))::Vector{Float64},
                    state_probs = pf -> probvec(pf, init_state_addr, 1:length(initial_states))::Vector{Float64},
                    lml_est = pf -> log_ml_estimate(pf)::Float64,
                )
                print_cb = PrintStatsCallback(
                    (goal_addr, 1:length(goals)),
                    (init_state_addr, 1:length(initial_states)),
                    header = (
                        "t\t" * join(goal_names, "\t") * "\t" *
                        join(state_names, "\t") * "\t"
                    ),
                )
                callback = CombinedCallback(logger = logger_cb, print = print_cb)

                sips = SIPS(world_config, resample_cond = :none, rejuv_cond = :none)
                sips(
                    length(init_strata),
                    t_obs_iter;
                    init_args = (init_strata = init_strata,),
                    callback = callback,
                )

                goal_probs_conditioned = reduce(hcat, callback.logger.data[:goal_probs])
                state_probs_conditioned = reduce(hcat, callback.logger.data[:state_probs])

                goal_probs_conditioned_dict[map_id][g][state_idx] = goal_probs_conditioned
                state_probs_conditioned_dict[map_id][g][state_idx] = state_probs_conditioned
            end
        end
    end

    save(output_path, Dict(
        "goal" => goal_probs_conditioned_dict,
        "state" => state_probs_conditioned_dict,
        "worlds" => possible_worlds,
    ))
    println("\nInference saved to: $output_path")
end

function run_experiment(
    problem_dir::String,
    inference_path::String,
    output_dir::String,
    metadata::Dict{String, Vector{Int}},
)
    println("\n=== Running exp2-style experiment ===")
    mkpath(output_dir)

    goal_probs_conditioned_dict = load(inference_path, "goal")
    state_probs_conditioned_dict = load(inference_path, "state")
    possible_worlds = load(inference_path, "worlds")
    domain_render = load_domain(DOMAIN_RENDER_PATH)

    steps_dict = Dict()
    replay_trace_dict = Dict()
    action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

    for map_id in sort!(collect(keys(metadata)))
        for (scenario_idx, g_id) in enumerate(metadata[map_id])
            map_key = "$(map_id)_$(scenario_idx)"
            println("\nScenario: $map_key (agent2 goal = gem$(g_id))")

            clear_planner_cache!()

            domain = load_domain(DOMAIN_PATH)
            problem = load_full_problem(problem_dir, map_id)
            state_raw = initstate(domain, problem)
            goals, _ = initialize_goals(state_raw)
            initial_states, _, _ = enumerate_beliefs(state_raw)
            blue_wizards = [w for w in PDDL.get_objects(state_raw, :wizard) if state_raw[pddl"(iscolor $w blue)"]]

            state = state_raw

            domain_agent1 = load_domain(DOMAIN_PATH)
            problem_agent1 = load_filtered_ascii_problem(problem_dir, map_id, :agent1)
            require_agent1_goal(problem_agent1, map_id)
            state_agent1 = initstate(domain_agent1, problem_agent1)
            state_render_agent1 = copy(state_agent1)

            domain_agent2 = load_domain(DOMAIN_PATH)
            problem_agent2 = load_filtered_ascii_problem(problem_dir, map_id, :agent2)
            state_agent2 = initstate(domain_agent2, problem_agent2)
            observed_agent_goals, _ = initialize_goals(state_agent2, :agent2)

            if g_id < 1 || g_id > length(goals)
                error("Scenario $map_key refers to gem$(g_id), but $map_id only has $(length(goals)) possible goals")
            end

            t = 0
            observation_events = Any[]
            decision_trace = Any[]
            stop_reason = ""

            wizard_candidates = copy(blue_wizards)

            s_id = findfirst(i -> check_equal_state(state_raw, initial_states[i]), eachindex(initial_states))
            isnothing(s_id) && error("Could not match initial state for $map_id")

            goal_probs = goal_probs_conditioned_dict[map_id][g_id][s_id]
            state_probs = state_probs_conditioned_dict[map_id][g_id][s_id]

            new_state = copy(state_render_agent1)
            planner = AStarPlanner(GoalManhattan())
            plan = collect(planner(domain_agent1, state_agent1, problem_agent1.goal))

            if !any(x -> x.name == :interact && x.args[end] in blue_wizards, plan)
                steps_dict[map_key] = 0
                stop_reason = "agent1_no_blue_wizard_needed"
                replay_trace_dict[map_key] = Dict(
                    "t" => 0,
                    "observations" => Any[],
                    "observation_events" => observation_events,
                    "decision_trace" => decision_trace,
                    "initial_candidates" => serialize_wizards(blue_wizards),
                    "final_candidates" => serialize_wizards(wizard_candidates),
                    "stop_reason" => stop_reason,
                )
                continue
            end

            observed_agent_plan = collect(planner(domain_agent2, state_agent2, observed_agent_goals[g_id]))
            observed_state_agent2_raw = copy(state_agent2)

            while !PDDL.satisfy(domain_agent1, state_agent1, problem_agent1.goal)
                if t + 1 > size(goal_probs, 2) || t + 1 > size(state_probs, 2)
                    steps_dict[map_key] = t
                    stop_reason = "inference_horizon_exhausted"
                    break
                end

                Q_observe = 0.0
                total_probs = 0.0
                timestep_hypotheses = Any[]

                for g in 1:length(goals)
                    for state_idx in 1:length(initial_states)
                        goal_prob_t = goal_probs[g, t + 1]
                        state_prob_t = state_probs[state_idx, t + 1]
                        joint_weight = goal_prob_t * state_prob_t

                        hypothesis_entry = Dict{String, Any}(
                            "goal_id" => g,
                            "state_idx" => state_idx,
                            "goal_prob" => goal_prob_t,
                            "state_prob" => state_prob_t,
                            "joint_weight" => joint_weight,
                            "included_in_q_observe" => false,
                        )

                        if goal_prob_t < 0.1
                            hypothesis_entry["skip_reason"] = "goal_prob_below_threshold"
                            push!(timestep_hypotheses, hypothesis_entry)
                            continue
                        end

                        if state_prob_t < 0.1
                            hypothesis_entry["skip_reason"] = "state_prob_below_threshold"
                            push!(timestep_hypotheses, hypothesis_entry)
                            continue
                        end

                        T = -1
                        for val in 1:size(state_probs_conditioned_dict[map_id][g][state_idx], 2)
                            if any(x -> x > 0.95, state_probs_conditioned_dict[map_id][g][state_idx][:, val])
                                T = val
                                break
                            end
                        end

                        if T == -1
                            for val in 1:size(goal_probs_conditioned_dict[map_id][g][state_idx], 2)
                                if any(x -> x < 0.1, goal_probs_conditioned_dict[map_id][g][state_idx][:, val])
                                    T = val
                                    break
                                end
                            end
                        end

                        if T == -1
                            T = max(t + 1, 1)
                        end
                        hypothesis_entry["inference_horizon"] = T

                        new_wizard_candidates = Const[]
                        for j in 1:length(blue_wizards)
                            if state_probs_conditioned_dict[map_id][g][state_idx][j, T] > 0.1
                                push!(new_wizard_candidates, blue_wizards[j])
                            end
                        end
                        hypothesis_entry["candidate_wizards_after_observing"] = serialize_wizards(new_wizard_candidates)

                        if isempty(new_wizard_candidates)
                            hypothesis_entry["skip_reason"] = "no_candidate_wizards_after_thresholding"
                            push!(timestep_hypotheses, hypothesis_entry)
                            continue
                        end

                        q_t_details = estimate_self_exploration_details(
                            domain_render,
                            new_state,
                            problem_agent1.goal,
                            new_wizard_candidates,
                            action_cost,
                        )
                        Q_T = q_t_details.cost
                        observed_steps = max(T - 1, 1)
                        observe_action_cost = action_cost[:observe] * observed_steps
                        total_hypothesis_cost = Q_T + observe_action_cost
                        weighted_contribution = joint_weight * total_hypothesis_cost

                        hypothesis_entry["included_in_q_observe"] = true
                        hypothesis_entry["exploration_cost_after_observing"] = Q_T
                        hypothesis_entry["observed_steps"] = observed_steps
                        hypothesis_entry["exploration_plan_after_observing"] = q_t_details.plan
                        hypothesis_entry["observe_action_cost"] = observe_action_cost
                        hypothesis_entry["total_hypothesis_cost"] = total_hypothesis_cost
                        hypothesis_entry["weighted_cost_contribution"] = weighted_contribution

                        Q_observe += weighted_contribution
                        total_probs += joint_weight
                        push!(timestep_hypotheses, hypothesis_entry)
                    end
                end

                timestep_entry = Dict{String, Any}(
                    "timestep" => t,
                    "next_observation_index" => t + 1,
                    "wizard_candidates_before" => serialize_wizards(wizard_candidates),
                    "hypotheses" => timestep_hypotheses,
                    "unnormalized_q_observe_sum" => Q_observe,
                    "total_probability_mass" => total_probs,
                )

                if total_probs <= 0
                    timestep_entry["decision"] = "stop"
                    timestep_entry["stop_reason"] = "no_probability_mass_after_thresholding"
                    push!(decision_trace, timestep_entry)
                    steps_dict[map_key] = t
                    stop_reason = "no_probability_mass_after_thresholding"
                    break
                end

                Q_observe /= total_probs
                q_not_observe_details = estimate_self_exploration_details(
                    domain_render,
                    new_state,
                    problem_agent1.goal,
                    wizard_candidates,
                    action_cost,
                )
                Q_not_observe = q_not_observe_details.cost
                timestep_entry["q_observe"] = Q_observe
                timestep_entry["q_not_observe"] = Q_not_observe
                timestep_entry["q_not_observe_plan"] = q_not_observe_details.plan

                println("  t=$t Q_observe=$Q_observe Q_not_observe=$Q_not_observe")
                println("    Q_not_observe plan: ", join(q_not_observe_details.plan, ", "))

                if Q_observe + 0.3 < Q_not_observe
                    if t + 1 > length(observed_agent_plan)
                        timestep_entry["decision"] = "stop"
                        timestep_entry["stop_reason"] = "observed_plan_exhausted"
                        push!(decision_trace, timestep_entry)
                        steps_dict[map_key] = t
                        stop_reason = "observed_plan_exhausted"
                        break
                    end

                    candidates_before = serialize_wizards(wizard_candidates)
                    t += 1

                    empty!(wizard_candidates)
                    observed_action = observed_agent_plan[t]
                    state_before_observation = copy(observed_state_agent2_raw)
                    observed_state_agent2_raw = PDDL.execute(domain_agent2, observed_state_agent2_raw, observed_action)
                    observed_outcome = interaction_outcome(
                        state_before_observation,
                        observed_state_agent2_raw,
                        :agent2,
                        observed_action,
                    )
                    timestep_entry["decision"] = "observe"
                    timestep_entry["observed_agent"] = "agent2"
                    timestep_entry["action"] = write_pddl(observed_action)
                    timestep_entry["interaction_outcome"] = observed_outcome

                    if t + 1 > size(state_probs, 2)
                        timestep_entry["stop_reason"] = "state_probs_exhausted"
                        push!(decision_trace, timestep_entry)
                        steps_dict[map_key] = t
                        stop_reason = "state_probs_exhausted"
                        break
                    end

                    for j in 1:length(blue_wizards)
                        if state_probs[j, t + 1] > 0.1
                            push!(wizard_candidates, blue_wizards[j])
                        end
                    end
                    timestep_entry["wizard_candidates_after"] = serialize_wizards(wizard_candidates)
                    push!(decision_trace, timestep_entry)

                    push!(observation_events, Dict(
                        "observation_index" => t,
                        "observed_agent" => "agent2",
                        "action" => write_pddl(observed_action),
                        "interaction_outcome" => observed_outcome,
                        "q_observe" => Q_observe,
                        "q_not_observe" => Q_not_observe,
                        "wizard_candidates_before" => candidates_before,
                        "wizard_candidates_after" => serialize_wizards(wizard_candidates),
                    ))
                else
                    timestep_entry["decision"] = "stop"
                    timestep_entry["stop_reason"] = "q_not_observe_better"
                    timestep_entry["wizard_candidates_after"] = serialize_wizards(wizard_candidates)
                    push!(decision_trace, timestep_entry)
                    steps_dict[map_key] = t
                    stop_reason = "q_not_observe_better"
                    break
                end
            end

            if !haskey(steps_dict, map_key)
                steps_dict[map_key] = t
                stop_reason = "goal_satisfied"
            end

            replay_trace_dict[map_key] = Dict(
                "t" => steps_dict[map_key],
                "observations" => [
                    serialize_observation(
                        event["observed_agent"],
                        parse_pddl(event["action"]),
                        get(event, "interaction_outcome", "none"),
                    ) for event in observation_events
                ],
                "observation_events" => observation_events,
                "decision_trace" => decision_trace,
                "initial_candidates" => serialize_wizards(blue_wizards),
                "final_candidates" => serialize_wizards(wizard_candidates),
                "stop_reason" => stop_reason,
            )
        end
    end

    steps_path = joinpath(output_dir, "steps_dict.json")
    replay_trace_path = joinpath(output_dir, "replay_trace.json")
    effective_metadata_path = joinpath(output_dir, "effective_metadata.json")

    write_json(steps_path, steps_dict)
    write_json(replay_trace_path, replay_trace_dict)
    write_json(effective_metadata_path, metadata)

    println("\nExperiment outputs:")
    println("  Steps: $steps_path")
    println("  Replay trace: $replay_trace_path")
    println("  Metadata: $effective_metadata_path")
end

function main()
    problem_dir = isempty(ARGS) ? String(@__DIR__) : abspath(ARGS[1])
    inference_path = joinpath(problem_dir, "inference_data_problem_example.jld2")
    output_dir = joinpath(problem_dir, "experiment_outputs")

    println("Problem directory: $problem_dir")
    metadata = load_or_generate_metadata(problem_dir)
    run_inference(problem_dir, inference_path, metadata)
    run_experiment(problem_dir, inference_path, output_dir, metadata)
    println("\nDone.")
end

main()
