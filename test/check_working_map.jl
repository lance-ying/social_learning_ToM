using FileIO, JLD2
using JSON

experiment_id = "exp3"
PROBLEM_DIR = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id")
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Load inference data
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
possible_worlds = data["worlds"]

# Check a working map (sm511) vs problematic ones
for map_id in ["sm511", "sm332", "sm342"]
    println("\n" * "="^60)
    println("Map: $map_id")
    println("="^60)
    
    agent_goals = metadata[map_id]
    
    if haskey(possible_worlds["agent2"], map_id)
        map_data_agent2 = possible_worlds["agent2"][map_id]
        println("Agent2 data type: ", typeof(map_data_agent2))
        if isa(map_data_agent2, Vector)
            println("Agent2 vector length: ", length(map_data_agent2))
            
            for scenario in 1:min(2, length(map_data_agent2))
                println("\n  Scenario $scenario:")
                scenario_data = map_data_agent2[scenario]
                println("    Type: ", typeof(scenario_data))
                
                # Try to understand the structure
                if isa(scenario_data, Dict)
                    println("    Keys: ", keys(scenario_data))
                    agent2_gem = agent_goals["agent2"][scenario]
                    if haskey(scenario_data, agent2_gem)
                        gem_data = scenario_data[agent2_gem]
                        println("    Gem $agent2_gem data type: ", typeof(gem_data))
                        if isa(gem_data, Dict)
                            println("    Gem $agent2_gem keys (states): ", keys(gem_data))
                            if length(keys(gem_data)) > 0
                                first_state = first(keys(gem_data))
                                state_data = gem_data[first_state]
                                println("    State $first_state data type: ", typeof(state_data))
                                if isa(state_data, Dict)
                                    println("    State $first_state keys (timesteps): ", keys(state_data))
                                    if length(keys(state_data)) > 0
                                        first_timestep = first(keys(state_data))
                                        plans = state_data[first_timestep]
                                        println("    Timestep $first_timestep: $(length(plans)) plan(s)")
                                        if length(plans) > 0
                                            plan_entry = plans[1]
                                            println("    Plan entry type: ", typeof(plan_entry))
                                            if isa(plan_entry, Tuple) && length(plan_entry) > 0
                                                plan = plan_entry[1]
                                                println("    Plan type: ", typeof(plan))
                                                println("    Plan length: ", length(plan))
                                                interact_actions = [a for a in plan if a.name == :interact]
                                                println("    Interact actions: $(length(interact_actions))")
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                else
                    # If it's not a Dict, it might be a state object (like sm332/sm342)
                    println("    WARNING: Scenario data is not a Dict - it's a state object!")
                    println("    This suggests the structure is different/missing for this map")
                end
            end
        end
    else
        println("Map $map_id not found in agent2 data")
    end
end

