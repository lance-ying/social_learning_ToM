using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl"))

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
serialize_observation(agent::String, action::Term, interaction_outcome::String="none") = Dict(
    "agent" => agent,
    "action" => write_pddl(action),
    "interaction_outcome" => interaction_outcome,
)

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

# include("paths_new.jl")
# Define directory paths
experiment_id = "exp2"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_$experiment_id")
LEGACY_OUTPUT_DIR = joinpath(@__DIR__, "experiment_outputs")
CANONICAL_OUTPUT_DIR = joinpath(@__DIR__, "outputs", experiment_id)
mkpath(LEGACY_OUTPUT_DIR)
mkpath(CANONICAL_OUTPUT_DIR)

function write_json_to_paths(paths, payload; indent::Int=4)
    for path in paths
        open(path, "w") do io
            JSON.print(io, payload, indent)
        end
    end
end

#--- Initial Setup ---#
problem_files = filter(f -> endswith(f, "ascii.pddl"), readdir(joinpath(@__DIR__, "..", "..", "dataset","problems_$experiment_id")))

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)



steps_dict = Dict()
replay_trace_dict = Dict()


# steps_dict = JSON.parsefile("/Users/lance/Documents/GitHub/ObserveMove/step_dict.json") 

goal_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "goal")
state_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "state")
possible_worlds = load(joinpath(@__DIR__, "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "worlds")


domain_render = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

for (map_id, v) in metadata

    println(steps_dict)
    for (i, goal_str) in enumerate(v)
        filename = "$(map_id)_$(i)_plan.pddl"
        println(filename)

        map_key = "$(map_id)_$(i)"
        
        # Clear planner cache for each scenario to avoid memory issues
        clear_planner_cache!()

        # println(map_id)


        domain = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))

        # Load problem
        # p_id = "s521_blue_exp"
        # map_id = p_id[1:4]

        problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
        # plan = paths[p_id]
        # Load plan
        # plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

        # Initialize and compile reference state
        state = initstate(domain, problem)

        state_render = copy(state)

        # heuristic = GoalManhattan()
        # planner = AStarPlanner(heuristic)

        domain, state = PDDL.compiled(domain, problem)

        txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
        ascii_content = read(txt_path, String)
        domain_agent1 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
        temp_path_agent1 = joinpath(PROBLEM_DIR, ".temp_agent1_$(map_id).txt")
        if !isfile(temp_path_agent1)
            filtered_ascii_agent1 = filter_ascii_agents(ascii_content, :agent1)
            write(temp_path_agent1, filtered_ascii_agent1)
        end
        problem_agent1 = load_ascii_problem(temp_path_agent1)
        state_agent1 = initstate(domain_agent1, problem_agent1)
        state_render_agent1 = copy(state_agent1)
        domain_agent1, state_agent1 = PDDL.compiled(domain_agent1, problem_agent1)

        domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "dataset", "domain.pddl"))
        temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
        if !isfile(temp_path_agent2)
            filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
            write(temp_path_agent2, filtered_ascii_agent2)
        end
        problem_agent2 = load_ascii_problem(temp_path_agent2)
        state_agent2 = initstate(domain_agent2, problem_agent2)
        domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)

        # Render initial state

        #--- Goal Inference Setup ---#

        # Specify possible goals
        goals, goal_names = initialize_goals(state)
        observed_agent_goals, _ = initialize_goals(state_agent2, :agent2)

        # goal_names = ["A", "B", "C"]
        # goal_colors = gem_colors


        # Enumerate over possible initial states
        initial_states, belief_probs, state_names = enumerate_beliefs(
            state
        )



        t= 0
        observation_events = Any[]
        stop_reason = ""

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
        wizard_candicates = blue_wizards


        g_id = metadata[map_id][i]
        s_id = -1


        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
            end
        end

        goal_probs = goal_probs_conditioned_dict[map_id][g_id][s_id]
        state_probs = state_probs_conditioned_dict[map_id][g_id][s_id]

        new_state = copy(state_render_agent1)


        planner = AStarPlanner(GoalManhattan())

        plan = planner(domain_agent1, state_agent1, problem_agent1.goal)

        if !any(x-> x.name == :interact && x.args[end] in blue_wizards, plan)
            print("t=", 0)
            steps_dict[map_key] = 0
            stop_reason = "agent1_no_blue_wizard_needed"
            replay_trace_dict[map_key] = Dict(
                "t" => 0,
                "observations" => Any[],
                "observation_events" => observation_events,
                "initial_candidates" => serialize_wizards(blue_wizards),
                "final_candidates" => serialize_wizards(wizard_candicates),
                "stop_reason" => stop_reason,
            )
            continue
        end

        observed_agent_plan = collect(planner(domain_agent2, state_agent2, observed_agent_goals[g_id]))
        observed_state_agent2 = copy(state_agent2)

        while !PDDL.satisfy(domain_agent1, state_agent1, problem_agent1.goal)

            Q_observe = 0
        
            planner = AStarPlanner(GoalManhattan())
        
            # plan = planner(domain, new_state, problem.goal)
        
            # if !any(x-> x.name == :interact, plan)
            #     print("t=", 0)
            #     steps_dict[p_id] = 0
            #     break
            # end

            total_probs = 0
        
        
            for g in 1:length(goals)
        
        
                if goal_probs[g, t+1] < 0.1
                    continue
                end
        
                for i in 1:length(initial_states)
        
                    if state_probs[i, t+1] < 0.1
                        continue
                    end
        
                    T = -1
        
                    for val in 1:length(state_probs_conditioned_dict[map_id][g][i][1,:])
            
                        if any(x -> x>0.95, state_probs_conditioned_dict[map_id][g][i][:,val])
                            T = val
                            break
                        end
                    end
        
                    if T == -1
        
                        for val in 1:length(goal_probs_conditioned_dict[map_id][g_id][i][1,:])
                
                            if any(x -> x<0.1, goal_probs_conditioned_dict[map_id][g_id][i][:,val])
                                T = val
                                break
                            end
                        end
                    end
        
                    new_wizard_candicates = []
        
                    for j in 1:length(blue_wizards)

                        if state_probs_conditioned_dict[map_id][g][i][j, T] > 0.1
                            push!(new_wizard_candicates, blue_wizards[j])
                        end
                    end
        
                    Q_T = estimate_self_exploration_cost(domain_render, new_state, problem_agent1.goal, new_wizard_candicates, action_cost)
        
                    Q_observe += goal_probs[g, t+1] * state_probs[i, t+1] * (Q_T + action_cost[:observe] * max(T,1))

                    total_probs += goal_probs[g, t+1] * state_probs[i, t+1]
                end

            end 

            Q_observe /= total_probs
        
            Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem_agent1.goal, wizard_candicates, action_cost)
        
            print("Q_observe = ", Q_observe, "Q_not_observe = ", Q_not_observe)
            println()
        
            if Q_observe+0.3 < Q_not_observe
                candidates_before = serialize_wizards(wizard_candicates)
                if t + 1 > length(observed_agent_plan)
                    steps_dict[map_key] = t
                    stop_reason = "observed_plan_exhausted"
                    break
                end
                t+=1
        
                wizard_candicates = []

                observed_action = observed_agent_plan[t]
                state_before_observation = copy(observed_state_agent2)
                observed_state_agent2 = PDDL.execute(domain_agent2, observed_state_agent2, observed_action)
                observed_outcome = interaction_outcome(state_before_observation, observed_state_agent2, :agent2, observed_action)
        
                for j in 1:length(blue_wizards)
                    if state_probs[j, t+1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
                push!(observation_events, Dict(
                    "observation_index" => t,
                    "observed_agent" => "agent2",
                    "action" => write_pddl(observed_action),
                    "interaction_outcome" => observed_outcome,
                    "q_observe" => Q_observe,
                    "q_not_observe" => Q_not_observe,
                    "wizard_candidates_before" => candidates_before,
                    "wizard_candidates_after" => serialize_wizards(wizard_candicates),
                ))
            else
                print("t = ", t)
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
            "observations" => [serialize_observation(event["observed_agent"], parse_pddl(event["action"]), get(event, "interaction_outcome", "none")) for event in observation_events],
            "observation_events" => observation_events,
            "initial_candidates" => serialize_wizards(blue_wizards),
            "final_candidates" => serialize_wizards(wizard_candicates),
            "stop_reason" => stop_reason,
        )
    end
end


legacy_steps_path = joinpath(LEGACY_OUTPUT_DIR, "steps_dict_$experiment_id.json")
canonical_steps_path = joinpath(CANONICAL_OUTPUT_DIR, "steps_dict.json")
write_json_to_paths((legacy_steps_path, canonical_steps_path), steps_dict)
println("Results saved to: $canonical_steps_path")
println("Legacy compatibility copy: $legacy_steps_path")

legacy_replay_trace_path = joinpath(LEGACY_OUTPUT_DIR, "replay_trace_$experiment_id.json")
canonical_replay_trace_path = joinpath(CANONICAL_OUTPUT_DIR, "replay_trace.json")
write_json_to_paths((legacy_replay_trace_path, canonical_replay_trace_path), replay_trace_dict)
println("Replay trace saved to: $canonical_replay_trace_path")
println("Legacy compatibility copy: $legacy_replay_trace_path")
