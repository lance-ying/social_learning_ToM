using FileIO, JLD2
using JSON

experiment_id = "exp3"
PROBLEM_DIR = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id")
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Load inference data
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
possible_worlds = data["worlds"]

# Loop through the problematic maps
for map_id in ["sm332", "sm342"]
    println("\n" * "="^60)
    println("Checking inferred plans for map: $map_id")
    println("="^60)
    
    agent_goals = metadata[map_id]
    
    for scenario in 1:2
        println("\n--- Scenario $scenario ---")
        
        agent2_gem = agent_goals["agent2"][scenario]
        agent3_gem = agent_goals["agent3"][scenario]
        println("Agent2 goal: gem$agent2_gem, Agent3 goal: gem$agent3_gem")

        # Check Agent2's plans - structure is: possible_worlds["agent2"][map_id][scenario][gem][state_id][timestep]
        println("\n  Agent2 (X) plans:")
        if haskey(possible_worlds, "agent2") && haskey(possible_worlds["agent2"], map_id)
            map_data = possible_worlds["agent2"][map_id]
            if isa(map_data, Vector) && length(map_data) >= scenario
                scenario_data = map_data[scenario]
                println("    Type of scenario data: ", typeof(scenario_data))
                if isa(scenario_data, Dict)
                    println("    Keys in scenario data (gems): ", keys(scenario_data))
                    if haskey(scenario_data, agent2_gem)
                        gem_data = scenario_data[agent2_gem]
                        println("    Type of gem data: ", typeof(gem_data))
                        if isa(gem_data, Dict)
                            println("    State IDs available: ", keys(gem_data))
                            
                            # Check each state
                            for s_id in keys(gem_data)
                                println("      State $s_id:")
                                state_data = gem_data[s_id]
                                println("        Type: ", typeof(state_data))
                                if isa(state_data, Dict)
                                    timesteps = sort(collect(keys(state_data)))
                                    println("        Timesteps available: $(length(timesteps)) (showing first 3)")
                                    
                                    # Check first few timesteps for interact actions
                                    for t_step in timesteps[1:min(3, length(timesteps))]
                                        plans_at_t = state_data[t_step]
                                        println("        Timestep $t_step: $(length(plans_at_t)) plan(s)")
                                        if length(plans_at_t) > 0
                                            plan_entry = plans_at_t[1]
                                            if isa(plan_entry, Tuple) && length(plan_entry) > 0
                                                plan = plan_entry[1]
                                                interact_actions = [a for a in plan if a.name == :interact]
                                                if !isempty(interact_actions)
                                                    println("          Plan 1: Has $(length(interact_actions)) interact action(s)")
                                                    for act in interact_actions
                                                        println("            - $(act.name) with $(act.args[end])")
                                                    end
                                                else
                                                    println("          Plan 1: No interact actions")
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    else
                        println("    ERROR: Gem $agent2_gem not found in scenario data")
                    end
                else
                    println("    ERROR: Scenario data is not a Dict, it's a $(typeof(scenario_data))")
                end
            else
                println("    ERROR: Scenario $scenario not available (vector length: $(isa(map_data, Vector) ? length(map_data) : "N/A"))")
            end
        end

        # Check Agent3's plans
        println("\n  Agent3 (Y) plans:")
        if haskey(possible_worlds, "agent3") && haskey(possible_worlds["agent3"], map_id)
            map_data = possible_worlds["agent3"][map_id]
            if isa(map_data, Vector) && length(map_data) >= scenario
                scenario_data = map_data[scenario]
                println("    Type of scenario data: ", typeof(scenario_data))
                if isa(scenario_data, Dict)
                    println("    Keys in scenario data (gems): ", keys(scenario_data))
                    if haskey(scenario_data, agent3_gem)
                        gem_data = scenario_data[agent3_gem]
                        println("    Type of gem data: ", typeof(gem_data))
                        if isa(gem_data, Dict)
                            println("    State IDs available: ", keys(gem_data))
                            
                            # Check each state
                            for s_id in keys(gem_data)
                                println("      State $s_id:")
                                state_data = gem_data[s_id]
                                println("        Type: ", typeof(state_data))
                                if isa(state_data, Dict)
                                    timesteps = sort(collect(keys(state_data)))
                                    println("        Timesteps available: $(length(timesteps)) (showing first 3)")
                                    
                                    # Check first few timesteps for interact actions
                                    for t_step in timesteps[1:min(3, length(timesteps))]
                                        plans_at_t = state_data[t_step]
                                        println("        Timestep $t_step: $(length(plans_at_t)) plan(s)")
                                        if length(plans_at_t) > 0
                                            plan_entry = plans_at_t[1]
                                            if isa(plan_entry, Tuple) && length(plan_entry) > 0
                                                plan = plan_entry[1]
                                                interact_actions = [a for a in plan if a.name == :interact]
                                                if !isempty(interact_actions)
                                                    println("          Plan 1: Has $(length(interact_actions)) interact action(s)")
                                                    for act in interact_actions
                                                        println("            - $(act.name) with $(act.args[end])")
                                                    end
                                                else
                                                    println("          Plan 1: No interact actions")
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    else
                        println("    ERROR: Gem $agent3_gem not found in scenario data")
                    end
                else
                    println("    ERROR: Scenario data is not a Dict, it's a $(typeof(scenario_data))")
                end
            else
                println("    ERROR: Scenario $scenario not available (vector length: $(isa(map_data, Vector) ? length(map_data) : "N/A"))")
            end
        end
    end
end

println("\nDone checking inferred plans.")
