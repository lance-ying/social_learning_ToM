using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
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
include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))

experiment_id = "exp3"
model = "naive"

step_dict = JSON.parsefile(joinpath(@__DIR__, "step_dict_$(model)_$(experiment_id).json"))

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")
PLAN_DIR = joinpath(@__DIR__, "results", "plans", model)
if !isdir(PLAN_DIR)
    mkpath(PLAN_DIR)
end

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Load inference data for both agents (agent2=X, agent3=Y)
data = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"))
state_probs_conditioned_dict = data["state"]

action_cost = Dict(:move => 3, :interact => 5, :observe => 1.0)

# Create progress bar for all (map, scenario) combinations
total_iterations = length(metadata) * 2  # 25 maps × 2 scenarios
progress = Progress(total_iterations, desc="Simulating $model baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for (map_id, agent_goals) in metadata
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    # Loop over both scenarios
    for scenario in 1:2
        scenario_start_time = time()
        map_key = "$(map_id)_scenario$(scenario)"
        
        # Clear planner cache for each scenario
        clear_planner_cache!()
        
        println("  Scenario $scenario")
        
        # Get which gems each agent wants in this scenario
        agent2_gem = agent_goals["agent2"][scenario]  # X's goal
        agent3_gem = agent_goals["agent3"][scenario]  # Y's goal

        domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))
        problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))

        goal = problem.goal

        # Initialize and compile reference state
        state = initstate(domain, problem)
        domain, state = PDDL.compiled(domain, problem)

        initial_states, belief_probs, state_names = enumerate_beliefs(state)

        # Get observation count from step_dict
        # step_dict can be either a number (T) or a dict with observation info
        observe_data = step_dict[map_key]
        if isa(observe_data, Dict)
            # If it's a dict, extract total observations
            observe = get(observe_data, "t", 0)
        else
            # If it's just a number
            observe = observe_data
        end

        planner = AStarPlanner(GoalManhattan())

        # For exp3, we need to find state IDs for both agents
        # But for simulation, we'll use the full state
        s_id = -1
        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
                break
            end
        end

        # Write initial observations
        # For exp3, observations could be for agent2 or agent3, but we'll observe agent1 (main agent)
        open(joinpath(PLAN_DIR, "$(map_key).pddl"), "w") do file
            for j in 1:observe
                println(file, "(observe agent1)")
            end
        end

        # Check if we can plan directly or need to simulate
        # For exp3, we need to check both agents' state probabilities
        # Use agent2 as the primary check (can be adjusted)
        if s_id > 0
            max_timesteps_agent2 = size(state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id], 2)
            if observe == 0 || (observe < max_timesteps_agent2 && any(x -> x > 0.9, state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id][:, observe+1]))
                plan = planner(domain, state, goal)

                open(joinpath(PLAN_DIR, "$(map_key).pddl"), "w") do file
                    for j in 1:observe
                        println(file, "(observe agent1)")
                    end
                    for action in plan
                        println(file, PDDL.write_pddl(action))
                    end
                end
            else
                # Simulate execution with belief updates
                explored_state = []
                cost = 9999
                curr_state = initial_states[1]
                curr_state_id = 1
                plan = []
                
                # Find best initial plan
                for j in 1:length(initial_states)
                    plan_temp = collect(planner(domain, initial_states[j], goal))
                    if length(plan_temp) < cost
                        cost = length(plan_temp)
                        curr_state_id = j
                        plan = plan_temp
                    end
                end
                
                push!(explored_state, curr_state_id)
                curr_state = initial_states[curr_state_id]

                # Execute plan and replan when needed
                while !PDDL.satisfy(domain, state, problem.goal)
                    state = PDDL.execute(domain, state, plan[1])
                    curr_state = PDDL.execute(domain, curr_state, plan[1])
                    
                    for j in 1:length(initial_states)
                        initial_states[j] = PDDL.execute(domain, initial_states[j], plan[1])
                    end

                    open(joinpath(PLAN_DIR, "$(map_key).pddl"), "a") do file
                        println(file, PDDL.write_pddl(plan[1]))
                    end

                    # Check if we need to replan
                    if (plan[1].name == :interact) && (!check_equal_state(state, curr_state))
                        println("    Replanning due to divergence...")
                        cost = 999
                        for j in 1:length(initial_states)
                            if j in explored_state
                                continue
                            end
                            plan_temp = collect(planner(domain, initial_states[j], goal))
                            if length(plan_temp) < cost
                                cost = length(plan_temp)
                                curr_state_id = j
                                plan = plan_temp
                            end
                        end
                        push!(explored_state, curr_state_id)
                        curr_state = initial_states[curr_state_id]
                    else
                        plan = plan[2:end]
                    end
                end
            end
        else
            # Fallback: just plan directly
            plan = planner(domain, state, goal)
            open(joinpath(PLAN_DIR, "$(map_key).pddl"), "w") do file
                for j in 1:observe
                    println(file, "(observe agent1)")
                end
                for action in plan
                    println(file, PDDL.write_pddl(action))
                end
            end
        end
        
        scenario_elapsed = time() - scenario_start_time
        cache_stats = get_cache_stats()
        println("    Observations: $observe")
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

println("\n=== Simulation Complete ===")
println("Plans saved to: $PLAN_DIR")
