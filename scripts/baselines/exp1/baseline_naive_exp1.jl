using PDDL, SymbolicPlanners
using JSON
using FileIO, JLD2
using ProgressMeter
using Statistics

# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "heuristics.jl"))
# beliefs.jl not needed for naive baseline
include(joinpath(@__DIR__, "..", "..", "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "render.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "src", "ascii.jl"))

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

# Define directory paths
experiment_id = "exp1"

PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "..", "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
steps_dict = Dict()

domain_render = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 0.5)

# Get all map files
map_files = filter(f -> endswith(f, ".pddl") && !occursin("_plan", f), readdir(PROBLEM_DIR))
map_ids = [replace(f, ".pddl" => "") for f in map_files]

# Create progress bar
total_iterations = length(map_ids)
progress = Progress(total_iterations, desc="Processing naive baseline: ")

# Track timing
map_times = Dict()
total_start_time = time()

for map_id in map_ids
    map_start_time = time()
    println("\nProcessing map: $map_id")
    
    scenario_start_time = time()
    map_key = map_id
    
    # Clear planner cache
    clear_planner_cache!()
    
    domain = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
    problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
        
        # Initialize and compile reference state
        state = initstate(domain, problem)
        state_render = copy(state)

        txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
        ascii_content = read(txt_path, String)
        domain_agent1 = load_domain(joinpath(@__DIR__, "..", "..", "..", "dataset", "domain.pddl"))
        temp_path_agent1 = joinpath(PROBLEM_DIR, ".temp_agent1_$(map_id).txt")
        if !isfile(temp_path_agent1)
            filtered_ascii_agent1 = filter_ascii_agents(ascii_content, :agent1)
            write(temp_path_agent1, filtered_ascii_agent1)
        end
        problem_agent1 = load_ascii_problem(temp_path_agent1)
        state_agent1 = initstate(domain_agent1, problem_agent1)
        
        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]

        planner = AStarPlanner(GoalManhattan())
        plan = collect(planner(domain_agent1, state_agent1, problem_agent1.goal))

        # Find first interaction with blue wizard
        T = -1
        for (idx, action) in enumerate(plan)
            if action.name == :interact && action.args[end] in blue_wizards
                T = idx
                break
            end
        end

        if T == -1
            T = length(plan)
        end

    steps_dict[map_key] = T
    
    scenario_elapsed = time() - scenario_start_time
    cache_stats = get_cache_stats()
    println("    Result: t=$T")
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
println("Average per map: $(round(mean(values(map_times)), digits=2))s")

# Save results
output_filename = "step_dict_naive_exp1.json"
open(output_filename, "w") do f
    JSON.print(f, steps_dict)
end

println("\n=== Experiment Complete ===")
println("Results saved to: $output_filename")
