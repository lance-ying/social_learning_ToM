using FileIO, JLD2
using JSON

# Load inference data
experiment_id = "exp3"
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]

# Load metadata
metadata_path = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id", "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Check sm332 and sm342
for map_id in ["sm332", "sm342"]
    println("\n" * "="^60)
    println("Checking map: $map_id")
    println("="^60)
    
    agent_goals = metadata[map_id]
    println("Agent goals: agent2 -> gem$(agent_goals["agent2"]), agent3 -> gem$(agent_goals["agent3"])")
    
    # Check agent2
    println("\n--- Agent2 (X) ---")
    if !haskey(goal_probs_conditioned_dict, "agent2")
        println("ERROR: No 'agent2' key in inference data!")
    elseif !haskey(goal_probs_conditioned_dict["agent2"], map_id)
        println("ERROR: Map $map_id not found for agent2!")
        println("Available maps for agent2: ", sort(collect(keys(goal_probs_conditioned_dict["agent2"]))))
    else
        agent2_data = goal_probs_conditioned_dict["agent2"][map_id]
        println("Agent2 has scenarios: ", keys(agent2_data))
        
        # Check scenario 1
        if haskey(agent2_data, 1)
            scenario1 = agent2_data[1]
            println("  Scenario 1 has gems: ", keys(scenario1))
            gem1 = agent_goals["agent2"][1]
            if haskey(scenario1, gem1)
                gem_data = scenario1[gem1]
                println("  Gem $gem1 has states: ", keys(gem_data))
                if length(keys(gem_data)) > 0
                    first_state = first(keys(gem_data))
                    goal_probs = gem_data[first_state]
                    println("  State $first_state goal_probs shape: ", size(goal_probs))
                end
            else
                println("  ERROR: Gem $gem1 not found in scenario 1!")
            end
        end
    end
    
    # Check agent3
    println("\n--- Agent3 (Y) ---")
    if !haskey(goal_probs_conditioned_dict, "agent3")
        println("ERROR: No 'agent3' key in inference data!")
    elseif !haskey(goal_probs_conditioned_dict["agent3"], map_id)
        println("ERROR: Map $map_id not found for agent3!")
        println("Available maps for agent3: ", sort(collect(keys(goal_probs_conditioned_dict["agent3"]))))
    else
        agent3_data = goal_probs_conditioned_dict["agent3"][map_id]
        println("Agent3 has scenarios: ", keys(agent3_data))
    end
end

