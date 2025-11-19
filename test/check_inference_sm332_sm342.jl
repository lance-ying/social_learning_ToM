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
    
    if !haskey(goal_probs_conditioned_dict, map_id)
        println("ERROR: Map $map_id not found in inference data!")
        continue
    end
    
    agent_goals = metadata[map_id]
    println("Agent goals: agent2 -> gem$(agent_goals["agent2"]), agent3 -> gem$(agent_goals["agent3"])")
    
    # Check structure for agent2
    println("\n--- Agent2 (X) ---")
    if !haskey(goal_probs_conditioned_dict, map_id) || !haskey(goal_probs_conditioned_dict[map_id], "agent2")
        println("ERROR: No agent2 data for $map_id")
    else
        agent2_data = goal_probs_conditioned_dict[map_id]["agent2"]
        println("Agent2 keys: ", keys(agent2_data))
        
        # Check scenario 1
        if haskey(agent2_data, 1)
            scenario1 = agent2_data[1]
            println("  Scenario 1 keys: ", keys(scenario1))
            if length(keys(scenario1)) > 0
                first_gem = first(keys(scenario1))
                println("  First gem: $first_gem")
                if length(keys(scenario1[first_gem])) > 0
                    first_state = first(keys(scenario1[first_gem]))
                    println("  First state: $first_state")
                    goal_probs = scenario1[first_gem][first_state]
                    println("  Goal probs shape: ", size(goal_probs))
                    println("  Goal probs at t=1: ", goal_probs[:, 1])
                end
            end
        end
        
        # Check scenario 2
        if haskey(agent2_data, 2)
            scenario2 = agent2_data[2]
            println("  Scenario 2 keys: ", keys(scenario2))
        end
    end
    
    # Check structure for agent3
    println("\n--- Agent3 (Y) ---")
    if !haskey(goal_probs_conditioned_dict, map_id) || !haskey(goal_probs_conditioned_dict[map_id], "agent3")
        println("ERROR: No agent3 data for $map_id")
    else
        agent3_data = goal_probs_conditioned_dict[map_id]["agent3"]
        println("Agent3 keys: ", keys(agent3_data))
        
        # Check scenario 1
        if haskey(agent3_data, 1)
            scenario1 = agent3_data[1]
            println("  Scenario 1 keys: ", keys(scenario1))
        end
        
        # Check scenario 2
        if haskey(agent3_data, 2)
            scenario2 = agent3_data[2]
            println("  Scenario 2 keys: ", keys(scenario2))
        end
    end
    
    # Check state_probs_conditioned_dict structure
    println("\n--- State Probs Structure ---")
    if haskey(state_probs_conditioned_dict, map_id)
        if haskey(state_probs_conditioned_dict[map_id], "agent2")
            agent2_states = state_probs_conditioned_dict[map_id]["agent2"]
            println("Agent2 state keys: ", keys(agent2_states))
            if haskey(agent2_states, 1)
                println("  Scenario 1 state keys: ", keys(agent2_states[1]))
            end
        end
        if haskey(state_probs_conditioned_dict[map_id], "agent3")
            agent3_states = state_probs_conditioned_dict[map_id]["agent3"]
            println("Agent3 state keys: ", keys(agent3_states))
        end
    end
end

println("\n" * "="^60)
println("Done checking inference data")
println("="^60)

