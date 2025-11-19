using FileIO, JLD2
using JSON
using PDDL, SymbolicPlanners
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "ascii.jl"))

# Define enumerate_beliefs directly to avoid ParticleFilterState dependency
"Enumerate all possible beliefs about key locations in the initial state."
function enumerate_beliefs(
    state::State;
    wizards = collect(PDDL.get_objects(state, :wizard)),
)
    belief_states = Vector{typeof(state)}()
    belief_probs = Float64[]
    belief_names = String[]
    belief_cnt = length(wizards)-1

    for wizard in sort!(wizards, by = x -> string(x))
        if state[pddl"(iscolor $wizard blue)"]
            push!(belief_names, string(wizard))
            base_state = copy(state)
            assign!(base_state, wizard)
            push!(belief_states, base_state)
            push!(belief_probs, 1.0 / belief_cnt)
        end
    end
    return belief_states, belief_probs, belief_names
end

# Load inference data
experiment_id = "exp3"
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]

# Load metadata
metadata_path = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id", "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Specify the map you want (change this as needed)
map_id = "sm332"  # You can change this to any map like "sm332", "sm342", etc.
scenario = 2      # 1 or 2

println("\n" * "="^80)
println("Extracting Pathing Data for $map_id, Scenario $scenario")
println("="^80)

# Check if map exists
if !haskey(metadata, map_id)
    println("ERROR: Map $map_id not found in metadata!")
    exit(1)
end

# Get agent goals for this scenario
agent_goals = metadata[map_id]
agent2_gem = agent_goals["agent2"][scenario]
agent3_gem = agent_goals["agent3"][scenario]

println("\nAgent goals for scenario $scenario:")
println("  agent2 -> gem$agent2_gem")
println("  agent3 -> gem$agent3_gem")

# Try to load saved plan files instead of recomputing
PLAN_DIR = joinpath(@__DIR__, "..", "results", "plans", "exp3")

function load_plan_file(plan_path::String)
    if !isfile(plan_path)
        return nothing
    end
    plan_content = read(plan_path, String)
    # Parse PDDL plan file - each line is an action
    actions = []
    for line in split(plan_content, '\n')
        line = strip(line)
        if length(line) > 0 && line[1] == '('
            push!(actions, line)
        end
    end
    return actions
end

# Extract pathing for agent2
println("\n" * "="^80)
println("AGENT2 (X) PATHING for $map_id, Scenario $scenario")
println("="^80)

if !haskey(goal_probs_conditioned_dict, "agent2") || !haskey(goal_probs_conditioned_dict["agent2"], map_id) || !haskey(goal_probs_conditioned_dict["agent2"][map_id], scenario)
    println("ERROR: No data found for agent2, $map_id, scenario $scenario")
else
    agent2_data = goal_probs_conditioned_dict["agent2"][map_id][scenario]
    
    # Show all available goal/state combinations
    println("\nAvailable goal/state combinations:")
    for (g, states_dict) in agent2_data
        println("  Goal $g: $(length(states_dict)) states")
    end
    
    # Get pathing for the actual goal used in this scenario
    if haskey(agent2_data, agent2_gem)
        println("\n--- Pathing for Actual Goal: gem$agent2_gem ---")
        states_dict = agent2_data[agent2_gem]
        
        # Show pathing for all states (or just the first few)
        local state_count = 0
        for (i, goal_probs_matrix) in states_dict
            state_count += 1
            if state_count <= 3  # Show first 3 states
                wizard_probs_matrix = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][i]
                
                println("\nState $i:")
                println("  Goal probabilities shape: $(size(goal_probs_matrix)) (goals × timesteps)")
                println("  Wizard probabilities shape: $(size(wizard_probs_matrix)) (wizards × timesteps)")
                println("  Number of timesteps: $(size(goal_probs_matrix, 2))")
                
                # Try to load the actual plan/path from saved files
                plan_file = joinpath(PLAN_DIR, "$(map_id)_agent2_gem$(agent2_gem)_plan.pddl")
                plan_actions = load_plan_file(plan_file)
                if plan_actions !== nothing
                    println("\n  Actual Path (Action Sequence):")
                    println("    Plan length: $(length(plan_actions)) actions")
                    for (step, action) in enumerate(plan_actions)
                        println("    Step $step: $action")
                    end
                else
                    println("\n  Actual Path: Plan file not found at $plan_file")
                end
                
                println("\n  Goal Probabilities Over Time:")
                println("  t\t" * join(["Goal$g" for g in 1:size(goal_probs_matrix, 1)], "\t"))
                for t in 1:min(size(goal_probs_matrix, 2), 25)  # Show first 25 timesteps
                    probs = [round(goal_probs_matrix[g, t], digits=3) for g in 1:size(goal_probs_matrix, 1)]
                    println("  $t\t" * join(string.(probs), "\t"))
                end
                if size(goal_probs_matrix, 2) > 25
                    println("  ... (showing first 25 of $(size(goal_probs_matrix, 2)) timesteps)")
                end
                
                println("\n  Wizard Probabilities Over Time:")
                println("  t\t" * join(["Wizard$w" for w in 1:size(wizard_probs_matrix, 1)], "\t"))
                for t in 1:min(size(wizard_probs_matrix, 2), 25)  # Show first 25 timesteps
                    probs = [round(wizard_probs_matrix[w, t], digits=3) for w in 1:size(wizard_probs_matrix, 1)]
                    println("  $t\t" * join(string.(probs), "\t"))
                end
                if size(wizard_probs_matrix, 2) > 25
                    println("  ... (showing first 25 of $(size(wizard_probs_matrix, 2)) timesteps)")
                end
            end
        end
        if state_count > 3
            println("\n  ... (showing first 3 of $state_count states)")
        end
    else
        println("ERROR: Goal gem$agent2_gem not found in agent2 data!")
    end
end

# Extract pathing for agent3
println("\n" * "="^80)
println("AGENT3 (Y) PATHING for $map_id, Scenario $scenario")
println("="^80)

if !haskey(goal_probs_conditioned_dict, "agent3") || !haskey(goal_probs_conditioned_dict["agent3"], map_id) || !haskey(goal_probs_conditioned_dict["agent3"][map_id], scenario)
    println("ERROR: No data found for agent3, $map_id, scenario $scenario")
else
    agent3_data = goal_probs_conditioned_dict["agent3"][map_id][scenario]
    
    # Show all available goal/state combinations
    println("\nAvailable goal/state combinations:")
    for (g, states_dict) in agent3_data
        println("  Goal $g: $(length(states_dict)) states")
    end
    
    # Get pathing for the actual goal used in this scenario
    if haskey(agent3_data, agent3_gem)
        println("\n--- Pathing for Actual Goal: gem$agent3_gem ---")
        states_dict = agent3_data[agent3_gem]
        
        # Show pathing for all states (or just the first few)
        local state_count = 0
        for (i, goal_probs_matrix) in states_dict
            state_count += 1
            if state_count <= 3  # Show first 3 states
                wizard_probs_matrix = state_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][i]
                
                println("\nState $i:")
                println("  Goal probabilities shape: $(size(goal_probs_matrix)) (goals × timesteps)")
                println("  Wizard probabilities shape: $(size(wizard_probs_matrix)) (wizards × timesteps)")
                println("  Number of timesteps: $(size(goal_probs_matrix, 2))")
                
                # Try to load the actual plan/path from saved files
                plan_file = joinpath(PLAN_DIR, "$(map_id)_agent3_gem$(agent3_gem)_plan.pddl")
                plan_actions = load_plan_file(plan_file)
                if plan_actions !== nothing
                    println("\n  Actual Path (Action Sequence):")
                    println("    Plan length: $(length(plan_actions)) actions")
                    for (step, action) in enumerate(plan_actions)
                        println("    Step $step: $action")
                    end
                else
                    println("\n  Actual Path: Plan file not found at $plan_file")
                end
                
                println("\n  Goal Probabilities Over Time:")
                println("  t\t" * join(["Goal$g" for g in 1:size(goal_probs_matrix, 1)], "\t"))
                for t in 1:min(size(goal_probs_matrix, 2), 25)  # Show first 25 timesteps
                    probs = [round(goal_probs_matrix[g, t], digits=3) for g in 1:size(goal_probs_matrix, 1)]
                    println("  $t\t" * join(string.(probs), "\t"))
                end
                if size(goal_probs_matrix, 2) > 25
                    println("  ... (showing first 25 of $(size(goal_probs_matrix, 2)) timesteps)")
                end
                
                println("\n  Wizard Probabilities Over Time:")
                println("  t\t" * join(["Wizard$w" for w in 1:size(wizard_probs_matrix, 1)], "\t"))
                for t in 1:min(size(wizard_probs_matrix, 2), 25)  # Show first 25 timesteps
                    probs = [round(wizard_probs_matrix[w, t], digits=3) for w in 1:size(wizard_probs_matrix, 1)]
                    println("  $t\t" * join(string.(probs), "\t"))
                end
                if size(wizard_probs_matrix, 2) > 25
                    println("  ... (showing first 25 of $(size(wizard_probs_matrix, 2)) timesteps)")
                end
            end
        end
        if state_count > 3
            println("\n  ... (showing first 3 of $state_count states)")
        end
    else
        println("ERROR: Goal gem$agent3_gem not found in agent3 data!")
    end
end

println("\n" * "="^80)
println("Done extracting pathing data")
println("="^80)

