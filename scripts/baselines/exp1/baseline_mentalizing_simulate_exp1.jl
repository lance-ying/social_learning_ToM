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
include(joinpath(@__DIR__, "..", "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))

experiment_id = "exp1"
model = "naive"

step_dict = JSON.parsefile(joinpath(@__DIR__, "step_dict_$(model)_exp1.json"))

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")
PLAN_DIR = joinpath(@__DIR__, "results", "plans", model)
if !isdir(PLAN_DIR)
    mkpath(PLAN_DIR)
end

#--- Initial Setup ---#
state_probs_conditioned_dict = load(joinpath(@__DIR__, "..", "..", "..", "data", "inference", "inference_data_$experiment_id.jld2"), "state")

action_cost = Dict(:move => 2, :interact => 5, :observe => 1)

# Get all map files
map_files = filter(f -> endswith(f, ".pddl") && !occursin("_plan", f), readdir(PROBLEM_DIR))
map_ids = [replace(f, ".pddl" => "") for f in map_files]

# Create progress bar
total_iterations = length(map_ids)
progress = Progress(total_iterations, desc="Simulating $model baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for map_id in map_ids
    # Map map_id to inference data format (mod_XXX_ascii)
    inference_map_id = "mod_$(map_id)_ascii"
    
    # Skip maps that don't exist in inference data or step_dict
    if !haskey(state_probs_conditioned_dict, inference_map_id) || !haskey(step_dict, map_id)
        println("\nSkipping map $map_id (not in inference data as $inference_map_id or step_dict)")
        next!(progress)
        continue
    end
    
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    scenario_start_time = time()
    map_key = map_id
    
    # Clear planner cache
    clear_planner_cache!()

    domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))

    goal = problem.goal

    # Initialize and compile reference state
    state = initstate(domain, problem)
    domain, state = PDDL.compiled(domain, problem)

    initial_states, belief_probs, state_names = enumerate_beliefs(state)

    observe = step_dict[map_key]

    planner = AStarPlanner(GoalManhattan())

    # Get goal_id from problem
    goal_id = parse(Int, string(problem.goal.args[2])[end:end])
    g_id = goal_id
        s_id = -1

        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
                break
            end
        end

        # Write initial observations
        open(joinpath(PLAN_DIR, "$(map_key).pddl"), "w") do file
            for j in 1:observe
                println(file, "(observe agent1)")
            end
        end

        # Check if we can plan directly or need to simulate
        # Ensure we don't access beyond available timesteps
        max_timesteps = size(state_probs_conditioned_dict[inference_map_id][g_id][s_id], 2)
        if observe == 0 || (observe < max_timesteps && any(x -> x > 0.9, state_probs_conditioned_dict[inference_map_id][g_id][s_id][:, observe+1]))
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
        
    scenario_elapsed = time() - scenario_start_time
    cache_stats = get_cache_stats()
    println("    Observations: $observe")
    println("    Time: $(round(scenario_elapsed, digits=2))s")
    println("    Cache: $(cache_stats.hits) hits, $(cache_stats.misses) misses, $(round(cache_stats.hit_rate * 100, digits=1))% hit rate")
    
    map_elapsed = time() - map_start_time
    map_times[map_id] = map_elapsed
    println("  Map completed in $(round(map_elapsed, digits=2))s")
    
    next!(progress)
end

total_elapsed = time() - total_start_time
println("\n=== Timing Summary ===")
println("Total time: $(round(total_elapsed, digits=2))s")
if length(map_times) > 0
    println("Average per map: $(round(mean(values(map_times)), digits=2))s")
else
    println("No maps processed")
end

println("\n=== Simulation Complete ===")
println("Plans saved to: $PLAN_DIR")