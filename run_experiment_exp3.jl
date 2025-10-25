using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using ProgressMeter
using Base.Threads
using Statistics
# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/render.jl")

# Define directory paths
experiment_id = "exp3"

PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

# Load inference data for both agents (agent2=Z, agent3=X)
data = load("inference_data_$experiment_id.jld2")
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]
possible_worlds = data["worlds"]

domain_render = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # 25 maps × 2 scenarios
progress = Progress(total_iterations, desc="Processing exp3: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, agent_goals) in metadata

    if map_id != "sm541"
        continue
    end
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    # Loop over both scenarios
    for scenario in 1:2
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"
        println("  Scenario $scenario")
        
        # Clear planner cache for each scenario to avoid memory issues
        clear_planner_cache!()
        
        # Get which gems each agent wants in this scenario
        agent2_gem = agent_goals["agent2"][scenario]  # Z's goal
        agent3_gem = agent_goals["agent3"][scenario]  # X's goal
        
        println("    agent2 (Z) -> gem$agent2_gem, agent3 (X) -> gem$agent3_gem")

        domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
        problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
        
        # Initialize and compile reference state for the FULL problem
        state = initstate(domain, problem)
        state_render = copy(state)
        domain, state = PDDL.compiled(domain, problem)

        #--- Goal Inference Setup ---#
        
        # Load FILTERED problem for agent2 to match inference belief states
        # (Inference was run with agent filtering, so we need to match that)
        include("src/ascii.jl")
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
        
        txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
        ascii_content = read(txt_path, String)
        
        # Load filtered problem for agent2 (use existing temp file if it exists)
        domain_agent2 = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
        temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
        if !isfile(temp_path_agent2)
            filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
            write(temp_path_agent2, filtered_ascii_agent2)
        end
        problem_agent2 = load_ascii_problem(temp_path_agent2)
        state_agent2 = initstate(domain_agent2, problem_agent2)
        domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)
        
        # Load filtered problem for agent3 (use existing temp file if it exists)
        domain_agent3 = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
        temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
        if !isfile(temp_path_agent3)
            filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
            write(temp_path_agent3, filtered_ascii_agent3)
        end
        problem_agent3 = load_ascii_problem(temp_path_agent3)
        state_agent3 = initstate(domain_agent3, problem_agent3)
        domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)
        
        # Specify possible goals for each agent (from FILTERED states)
        goals_agent2, goal_names_agent2 = initialize_goals(state_agent2, :agent2)
        goals_agent3, goal_names_agent3 = initialize_goals(state_agent3, :agent3)

        # Enumerate over possible initial states (from FILTERED states)
        initial_states_agent2, belief_probs_agent2, state_names_agent2 = enumerate_beliefs(state_agent2)
        initial_states_agent3, belief_probs_agent3, state_names_agent3 = enumerate_beliefs(state_agent3)

        t = 0


        
        # Track observations
        observations = []
        agent2_count = 0
        agent3_count = 0

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
        wizard_candicates = blue_wizards
        
        # Pre-compute blue wizards for filtered states (used in Q computation)
        blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]
        blue_wizards_agent3 = [w for w in PDDL.get_objects(state_agent3, :wizard) if state_agent3[pddl"(iscolor $w blue)"]]

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

        # Load initial probabilities from scenario-specific goals
        goal_probs_agent2 = goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
        state_probs_agent2 = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
        
        goal_probs_agent3 = goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]
        state_probs_agent3 = state_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id_agent3]


        # Pre-compute state copy and planner (moved outside loop for efficiency)
        new_state = copy(state_render)
        planner = AStarPlanner(GoalManhattan())
        plan = planner(domain, state, problem.goal)

        if !any(x-> x.name == :interact && x.args[end] in blue_wizards, plan)
            print("t=", 0)
            steps_dict[map_key] = Dict(
                "t" => 0, 
                "observations" => [],
                "agent2_count" => 0,
                "agent3_count" => 0
            )
            next!(progress)
            continue
        end

        while !PDDL.satisfy(domain, state, problem.goal)
            
            println("    [t=$t] Computing Q-values...")
            q_start_time = time()
            
            # Check if we have probability data for timestep t+1
            max_t_agent2 = size(goal_probs_agent2, 2) - 1  # -1 because we access t+1
            max_t_agent3 = size(goal_probs_agent3, 2) - 1
            
            if t >= max_t_agent2 || t >= max_t_agent3
                println("    -> Reached end of inference data at t=$t")
                steps_dict[map_key] = Dict(
                    "t" => t,
                    "observations" => observations,
                    "agent2_count" => agent2_count,
                    "agent3_count" => agent3_count
                )
                break
            end
            
            # Parallelize Q computation for both agents
            task_agent2 = Threads.@spawn begin
                # Compute Q_observe for agent2 (Z)
                Q_observe_agent2 = 0.0
                total_probs_agent2 = 0.0
                
                # Loop over ALL possible goals (observer doesn't know which goal agent has)
                for g in 1:length(goals_agent2)
                if goal_probs_agent2[g, t+1] < 0.1
                    continue
                end
                
                for i in 1:length(initial_states_agent2)
                    if state_probs_agent2[i, t+1] < 0.1
                        continue
                    end
                    
                    println("        agent2: goal=$g, state=$i (prob=$(goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1]))")
                    
                    T = -1
                    for val in 1:length(state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][1,:])
                        if any(x -> x>0.95, state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][:,val])
                            T = val
                            break
                        end
                    end
                    
                    if T == -1
                        for val in 1:length(goal_probs_conditioned_dict["agent2"][map_id][scenario][g][i][1,:])
                            if any(x -> x<0.1, goal_probs_conditioned_dict["agent2"][map_id][scenario][g][i][:,val])
                                T = val
                                break
                            end
                        end
                    end
                    
                    # Validate T is within bounds for state_probs_conditioned_dict
                    max_T_state = size(state_probs_conditioned_dict["agent2"][map_id][scenario][g][i], 2)
                    if T == -1 || T > max_T_state
                        # If T is invalid, use all wizards as candidates
                        new_wizard_candicates = copy(blue_wizards_agent2)
                    else
                        # Get blue wizards from pre-computed list
                        new_wizard_candicates = []
                        for j in 1:length(blue_wizards_agent2)
                            if state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][j, T] > 0.1
                                push!(new_wizard_candicates, blue_wizards_agent2[j])
                            end
                        end
                    end
                    
                    println("          Computing Q_T (calling planner)...")
                    Q_T = estimate_self_exploration_cost(domain_render, new_state, problem.goal, new_wizard_candicates, action_cost)
                    println("          Q_T = $Q_T")
                    Q_observe_agent2 += goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1] * (Q_T + action_cost[:observe] * max(T,1))
                    total_probs_agent2 += goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1]
                end
                end
                
                Q_observe_agent2 /= total_probs_agent2
                println("      agent2 Q computed: $Q_observe_agent2")
                (Q_observe_agent2, total_probs_agent2)
            end
            
            task_agent3 = Threads.@spawn begin
                # Compute Q_observe for agent3 (X)
                Q_observe_agent3 = 0.0
                total_probs_agent3 = 0.0
                
                # Loop over ALL possible goals (observer doesn't know which goal agent has)
                for g in 1:length(goals_agent3)
                if goal_probs_agent3[g, t+1] < 0.1
                    continue
                end
                
                for i in 1:length(initial_states_agent3)
                    if state_probs_agent3[i, t+1] < 0.1
                        continue
                    end
                    
                    println("        agent3: goal=$g, state=$i (prob=$(goal_probs_agent3[g, t+1] * state_probs_agent3[i, t+1]))")
                    
                    T = -1
                    for val in 1:length(state_probs_conditioned_dict["agent3"][map_id][scenario][g][i][1,:])
                        if any(x -> x>0.95, state_probs_conditioned_dict["agent3"][map_id][scenario][g][i][:,val])
                            T = val
                            break
                        end
                    end
                    
                    if T == -1
                        for val in 1:length(goal_probs_conditioned_dict["agent3"][map_id][scenario][g][i][1,:])
                            if any(x -> x<0.1, goal_probs_conditioned_dict["agent3"][map_id][scenario][g][i][:,val])
                                T = val
                                break
                            end
                        end
                    end
                    
                    # Validate T is within bounds for state_probs_conditioned_dict
                    max_T_state = size(state_probs_conditioned_dict["agent3"][map_id][scenario][g][i], 2)
                    if T == -1 || T > max_T_state
                        # If T is invalid, use all wizards as candidates
                        new_wizard_candicates = copy(blue_wizards_agent3)
                    else
                        # Get blue wizards from pre-computed list
                        new_wizard_candicates = []
                        for j in 1:length(blue_wizards_agent3)
                            if state_probs_conditioned_dict["agent3"][map_id][scenario][g][i][j, T] > 0.1
                                push!(new_wizard_candicates, blue_wizards_agent3[j])
                            end
                        end
                    end
                    
                    println("          Computing Q_T (calling planner)...")
                    Q_T = estimate_self_exploration_cost(domain_render, new_state, problem.goal, new_wizard_candicates, action_cost)
                    println("          Q_T = $Q_T")
                    Q_observe_agent3 += goal_probs_agent3[g, t+1] * state_probs_agent3[i, t+1] * (Q_T + action_cost[:observe] * max(T,1))
                    total_probs_agent3 += goal_probs_agent3[g, t+1] * state_probs_agent3[i, t+1]
                end
                end
                
                Q_observe_agent3 /= total_probs_agent3
                println("      agent3 Q computed: $Q_observe_agent3")
                (Q_observe_agent3, total_probs_agent3)
            end
            
            # Wait for both parallel tasks to complete
            (Q_observe_agent2, total_probs_agent2) = fetch(task_agent2)
            (Q_observe_agent3, total_probs_agent3) = fetch(task_agent3)
            
            q_elapsed = time() - q_start_time
            println("      Q-value computation time: $(round(q_elapsed, digits=2))s")
            
            # Compute Q_not_observe
            Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, wizard_candicates, action_cost)
            println("      Q_not_observe computed: $Q_not_observe")
            
            println("    t=$t: Q_observe_agent2=$Q_observe_agent2, Q_observe_agent3=$Q_observe_agent3, Q_not_observe=$Q_not_observe")
            
            # Take argmin to decide which action
            q_values = [Q_observe_agent2, Q_observe_agent3, Q_not_observe]
            best_action_idx = argmin(q_values)
            
            if best_action_idx == 1
                # Observe agent2 (Z) - it has the lowest Q-value
                println("    -> Observing agent2 (Z)")
                push!(observations, "agent2")
                agent2_count += 1
                t += 1
                wizard_candicates = []
                for j in 1:length(blue_wizards)
                    if state_probs_agent2[j, t+1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
            elseif best_action_idx == 2
                # Observe agent3 (X) - it has the lowest Q-value
                println("    -> Observing agent3 (X)")
                push!(observations, "agent3")
                agent3_count += 1
                t += 1
                wizard_candicates = []
                for j in 1:length(blue_wizards)
                    if state_probs_agent3[j, t+1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
            else
                # best_action_idx == 3: Not observing has the lowest Q-value
                println("    -> Not observing (stopping at t=$t)")
                steps_dict[map_key] = Dict(
                    "t" => t,
                    "observations" => observations,
                    "agent2_count" => agent2_count,
                    "agent3_count" => agent3_count
                )
                break
            end
        end
        
        # Update progress bar after each scenario
        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        println("    Scenario completed in $(round(scenario_elapsed, digits=2))s")
        println("    Cache stats: $(cache_stats.hits) hits, $(cache_stats.misses) misses, $(round(cache_stats.hit_rate * 100, digits=1))% hit rate")
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
println("Fastest map: $(round(minimum(values(map_times)), digits=2))s")
println("Slowest map: $(round(maximum(values(map_times)), digits=2))s")


output_filename = "steps_dict_$(experiment_id)_optimized.json"
open(output_filename, "w") do io
    JSON.print(io, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")

