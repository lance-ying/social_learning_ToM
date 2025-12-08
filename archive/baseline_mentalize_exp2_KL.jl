using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# KL divergence function: KL(P||Q) = sum_i P(i) * log(P(i) / Q(i))
function kl_divergence(P, Q)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    P_safe = P .+ epsilon
    Q_safe = Q .+ epsilon
    # Normalize
    P_safe = P_safe / sum(P_safe)
    Q_safe = Q_safe / sum(Q_safe)
    # Compute KL divergence
    kl = sum(P_safe .* log.(P_safe ./ Q_safe))
    return kl
end

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))

# Define directory paths
experiment_id = "exp2"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

goal_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "goal")
state_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "state")
possible_worlds = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "worlds")

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 3, :interact => 5, :observe => 1)

# Create progress bar
total_iterations = sum(length(v) for v in values(metadata))
progress = Progress(total_iterations, desc="Processing mentalizing baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, goal_list) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    for (i, goal_str) in enumerate(goal_list)
        scenario_start_time = time()
        map_key = "$(map_id)_$(i)"
        
        # Clear planner cache for each scenario
        clear_planner_cache!()
        
        println("  Scenario $i (goal=$goal_str)")

        domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
        
        # Initialize and compile reference state
        state = initstate(domain, problem)
        state_render = copy(state)

        # Get goal_id from problem
        goal_id = parse(Int, string(problem.goal.args[2])[end:end])

        initial_states, belief_probs, state_names = enumerate_beliefs(state)

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

        # goal_probs_conditioned_dict uses integer keys, state_probs_conditioned_dict uses string keys
        g_id = isa(goal_str, Int) ? goal_str : parse(Int, goal_str)  # Integer key for goal_probs_conditioned_dict
        g_id_int = g_id  # Same value for integer comparison in loops
        g_id_str = string(g_id)  # String key for state_probs_conditioned_dict
        s_id = -1

        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
                break
            end
        end

        goal_probs = goal_probs_conditioned_dict[map_id][g_id][s_id]
        state_probs = state_probs_conditioned_dict[map_id][g_id][s_id]

        new_state = copy(state_render)

        planner = AStarPlanner(GoalManhattan())
        plan = planner(domain, state, problem.goal)

        T = 0

        # Social Mentalizing Observer: Observe when max_T KL[P(bm_t)||P(bm_t|ao_{t:t+T})] > 0
        # P(bm_t) = current belief about environment state (marginal over goals and states)
        # P(bm_t|ao_{t:t+T}) = belief after observing other agent's actions from t to t+T
        
        max_timesteps = min(length(goal_probs[1,:]) - 1, 50)  # Limit search to prevent long loops
        
        for t in 1:max_timesteps
            # Compute P(bm_t) - current marginal belief over states at time t
            # Marginalize over goals: P(state=s|obs_t) = sum_g P(goal=g|obs_t) * P(state=s|goal=g, obs_t)
            P_bm_t = zeros(length(initial_states))
            for s in 1:length(initial_states)
                for g in 1:3
                    g_str = string(g)  # Convert integer to string for dictionary access
                    if haskey(state_probs_conditioned_dict[map_id], g_str) && 
                       haskey(state_probs_conditioned_dict[map_id][g_str], s) &&
                       t <= size(state_probs_conditioned_dict[map_id][g_str][s], 2)
                        # Weight by goal probability and state probability
                        goal_prob = (g == g_id_int && t <= size(goal_probs, 2)) ? goal_probs[g, t] : 0.0
                        state_prob = state_probs_conditioned_dict[map_id][g_str][s][s_id, t]
                        P_bm_t[s] += goal_prob * state_prob
                    end
                end
            end
            
            # Normalize P_bm_t
            if sum(P_bm_t) > 0
                P_bm_t = P_bm_t / sum(P_bm_t)
            else
                # If no probability mass, use uniform
                P_bm_t = ones(length(initial_states)) / length(initial_states)
            end
            
            # Find max_T KL[P(bm_t)||P(bm_t|ao_{t:t+T})]
            max_kl = 0.0
            max_kl_T = 0
            
            # Try different T values (how many steps ahead to look)
            for T_val in 1:min(10, max_timesteps - t)  # Limit lookahead to 10 steps
                t_future = t + T_val
                if t_future > max_timesteps
                    break
                end
                
                # Compute P(bm_t|ao_{t:t+T}) - belief after observing T steps
                # This is the marginal belief at time t_future
                P_bm_t_future = zeros(length(initial_states))
                for s in 1:length(initial_states)
                    for g in 1:3
                        g_str = string(g)  # Convert integer to string for dictionary access
                        if haskey(state_probs_conditioned_dict[map_id], g_str) && 
                           haskey(state_probs_conditioned_dict[map_id][g_str], s) &&
                           t_future <= size(state_probs_conditioned_dict[map_id][g_str][s], 2)
                            goal_prob = (g == g_id_int && t_future <= size(goal_probs, 2)) ? goal_probs[g, t_future] : 0.0
                            state_prob = state_probs_conditioned_dict[map_id][g_str][s][s_id, t_future]
                            P_bm_t_future[s] += goal_prob * state_prob
                        end
                    end
                end
                
                # Normalize P_bm_t_future
                if sum(P_bm_t_future) > 0
                    P_bm_t_future = P_bm_t_future / sum(P_bm_t_future)
                else
                    P_bm_t_future = ones(length(initial_states)) / length(initial_states)
                end
                
                # Compute KL divergence
                kl = kl_divergence(P_bm_t, P_bm_t_future)
                
                if kl > max_kl
                    max_kl = kl
                    max_kl_T = T_val
                end
            end
            
            # According to paper: observe when max_T KL > 0
            # If max_kl > 0, continue observing; otherwise stop
            if max_kl <= 0.0 || max_kl < 1e-6  # Use small threshold for numerical stability
                T = t - 1  # Stop at previous timestep
                break
            end
            
            # If we've reached the end, set T to current t
            if t == max_timesteps
                T = t
            end
        end
        
        # Ensure T is at least 1 if we never found a stopping point
        if T == 0
            T = 1
        end

        steps_dict[map_key] = T
        
        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        println("    Result: t=$T")
        println("    Time: $(round(scenario_elapsed, digits=2))s")
        println("    Cache: $(cache_stats.hits) hits, $(cache_stats.misses) misses, $(round(cache_stats.hit_rate * 100, digits=1))% hit rate")
        
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
output_filename = "step_dict_mentalize_exp2.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")