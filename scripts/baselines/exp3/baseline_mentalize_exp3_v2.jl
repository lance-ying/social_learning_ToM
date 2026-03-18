using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
# using GenGPT3
using InversePlanning
# using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "beliefs.jl"))
# include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))

# Define directory paths
experiment_id = "exp3"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

# Load inference data for both agents (agent2=X, agent3=Y)
data = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"))
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]
possible_worlds = data["worlds"]

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # 25 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing mentalizing baseline v2: ")

# Track timing
map_times = Dict()
total_start_time = time()

# Helper function to filter ASCII maps to single agent
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

"""
Find when an agent's state distributions stop being informative.
Returns the timestep T at which the observer's beliefs can no longer be
distinguished from the conditioned future distributions (i.e., further
observation of this agent won't help resolve uncertainty).

If divergence is never detected (observations remain informative throughout),
returns the max available timestep.
"""
function find_informativeness_horizon(
    goal_probs, state_probs,
    state_probs_conditioned_dict_agent, goal_probs_conditioned_dict_agent,
    map_id, scenario, gem_id, s_id,
    goals, initial_states
)
    max_t = size(goal_probs, 2) - 1

    for t in 1:max_t
        curr_state_dist = state_probs[:, t]
        flag = true

        for g in 1:length(goals)
            if goal_probs[g, t+1] > 0.1
                for s in 1:length(initial_states)
                    if state_probs[s, t+1] > 0.1
                        max_t_available = size(state_probs_conditioned_dict_agent[map_id][scenario][g][s], 2)
                        for val in t:max_t_available
                            if eval_state_dist(curr_state_dist, state_probs_conditioned_dict_agent[map_id][scenario][g][s][:, val])
                                flag = false
                                break
                            end
                        end
                    end
                    if !flag
                        break
                    end
                end
            end
            if !flag
                break
            end
        end

        if flag
            return t
        end
    end

    # If we never found divergence, observations remain informative
    # throughout the entire horizon -> return max timestep
    return max_t
end

for (map_id, agent_goals) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")

    clear_planner_cache!()

    #--- Map-level setup (shared across both scenarios) ---#

    # Load main domain (uncompiled — planner works fine without compilation, see exp1/exp2)
    domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))
    state = initstate(domain, problem)

    #--- Agent-filtered domains (compile once per map, reuse across scenarios) ---#

    txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
    ascii_content = read(txt_path, String)

    # Load and compile filtered problem for agent1
    domain_agent1 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent1 = joinpath(PROBLEM_DIR, ".temp_agent1_$(map_id).txt")
    if !isfile(temp_path_agent1)
        filtered_ascii_agent1 = filter_ascii_agents(ascii_content, :agent1)
        write(temp_path_agent1, filtered_ascii_agent1)
    end
    problem_agent1 = load_ascii_problem(temp_path_agent1)
    state_agent1 = initstate(domain_agent1, problem_agent1)
    domain_agent1, state_agent1 = PDDL.compiled(domain_agent1, problem_agent1)

    # Check if agent1 even needs blue wizards (same answer for both scenarios)
    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    planner = AStarPlanner(GoalManhattan())
    plan_agent1 = planner(domain_agent1, state_agent1, problem_agent1.goal)
    agent1_needs_wizards = any(x -> x.name == :interact && x.args[end] in blue_wizards, plan_agent1)

    if !agent1_needs_wizards
        println("    Agent1 doesn't need blue wizards -> t=0 for both scenarios")
        for scenario in 1:2
            map_key = "$(map_id)_scenario$(scenario)"
            steps_dict[map_key] = Dict(
                "observations" => [],
                "agent2_count" => 0,
                "agent3_count" => 0,
                "t" => 0
            )
            next!(progress)
        end
        map_elapsed = time() - map_start_time
        map_times[map_id] = map_elapsed
        println("  Map completed in $(round(map_elapsed, digits=2))s")
        continue
    end

    # Load and compile filtered problem for agent2
    domain_agent2 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    if !isfile(temp_path_agent2)
        filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
        write(temp_path_agent2, filtered_ascii_agent2)
    end
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2 = initstate(domain_agent2, problem_agent2)
    domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)

    # Load and compile filtered problem for agent3
    domain_agent3 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
    if !isfile(temp_path_agent3)
        filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
        write(temp_path_agent3, filtered_ascii_agent3)
    end
    problem_agent3 = load_ascii_problem(temp_path_agent3)
    state_agent3 = initstate(domain_agent3, problem_agent3)
    domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)

    # Goals and beliefs (same across scenarios — only depends on map layout)
    goals_agent2, goal_names_agent2 = initialize_goals(state_agent2, :agent2)
    goals_agent3, goal_names_agent3 = initialize_goals(state_agent3, :agent3)

    initial_states_agent2, belief_probs_agent2, state_names_agent2 = enumerate_beliefs(state_agent2)
    initial_states_agent3, belief_probs_agent3, state_names_agent3 = enumerate_beliefs(state_agent3)

    # Find current state ID for agent2 (using FILTERED state)
    s_id_agent2 = -1
    for s in 1:length(initial_states_agent2)
        if check_equal_state(state_agent2, initial_states_agent2[s])
            s_id_agent2 = s
            break
        end
    end

    # Find current state ID for agent3 (using FILTERED state)
    s_id_agent3 = -1
    for s in 1:length(initial_states_agent3)
        if check_equal_state(state_agent3, initial_states_agent3[s])
            s_id_agent3 = s
            break
        end
    end

    # Loop over both scenarios (only goal assignments differ)
    for scenario in 1:2
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"

        println("  Scenario $scenario")

        # Get which gems each agent wants in this scenario
        agent2_gem = agent_goals["agent2"][scenario]  # X's goal
        agent3_gem = agent_goals["agent3"][scenario]  # Y's goal

        # Load scenario-specific probabilities
        goal_probs_agent2 = goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
        state_probs_agent2 = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]

        goal_probs_agent3 = goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]
        state_probs_agent3 = state_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]

        #--- Find informativeness horizon independently for each agent ---#
        T_agent2 = find_informativeness_horizon(
            goal_probs_agent2, state_probs_agent2,
            state_probs_conditioned_dict["agent2"], goal_probs_conditioned_dict["agent2"],
            map_id, scenario, agent2_gem, s_id_agent2,
            goals_agent2, initial_states_agent2
        )

        T_agent3 = find_informativeness_horizon(
            goal_probs_agent3, state_probs_agent3,
            state_probs_conditioned_dict["agent3"], goal_probs_conditioned_dict["agent3"],
            map_id, scenario, agent3_gem, s_id_agent3,
            goals_agent3, initial_states_agent3
        )

        # Each agent's count is independent: observe each agent for as long
        # as their actions remain informative about the environment state
        agent2_count = T_agent2
        agent3_count = T_agent3
        T = agent2_count + agent3_count

        # Build interleaved observations list (alternate between agents)
        observations = String[]
        a2_remaining = agent2_count
        a3_remaining = agent3_count
        while a2_remaining > 0 || a3_remaining > 0
            if a2_remaining > 0
                push!(observations, "agent2")
                a2_remaining -= 1
            end
            if a3_remaining > 0
                push!(observations, "agent3")
                a3_remaining -= 1
            end
        end

        steps_dict[map_key] = Dict(
            "observations" => observations,
            "agent2_count" => agent2_count,
            "agent3_count" => agent3_count,
            "t" => T
        )

        scenario_elapsed = time() - scenario_start_time
        println("    Result: agent2=$T_agent2, agent3=$T_agent3, total=$T")
        println("    Time: $(round(scenario_elapsed, digits=2))s")

        next!(progress)
    end

    map_elapsed = time() - map_start_time
    map_times[map_id] = map_elapsed
    println("  Map completed in $(round(map_elapsed, digits=2))s")
end

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
println("Average per map: $(round(mean(values(map_times)), digits=2))s")

# Save results
output_filename = "step_dict_mentalize_exp3.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
