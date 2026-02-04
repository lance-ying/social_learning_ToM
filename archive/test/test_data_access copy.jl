using FileIO, JLD2

println("=== Loading exp2 data ===")
data_exp2 = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_exp2.jld2"))
goal_dict_exp2 = data_exp2["goal"]
state_dict_exp2 = data_exp2["state"]

println("\nexp2 top-level keys (goal dict): ", keys(goal_dict_exp2))
first_map_exp2 = first(keys(goal_dict_exp2))
println("First map: $first_map_exp2")
println("Keys for $first_map_exp2: ", keys(goal_dict_exp2[first_map_exp2]))

first_goal = first(keys(goal_dict_exp2[first_map_exp2]))
println("Keys for goal $first_goal: ", keys(goal_dict_exp2[first_map_exp2][first_goal]))

first_state = first(keys(goal_dict_exp2[first_map_exp2][first_goal]))
println("Sample data shape for goal=$first_goal, state=$first_state: ", size(goal_dict_exp2[first_map_exp2][first_goal][first_state]))

println("\n" * "="^60)

println("\n=== Loading exp3 data ===")
data_exp3 = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_exp3.jld2"))
goal_dict_exp3 = data_exp3["goal"]
state_dict_exp3 = data_exp3["state"]

println("\nexp3 top-level keys (goal dict): ", keys(goal_dict_exp3))
if "agent2" in keys(goal_dict_exp3)
    println("Keys for agent2: ", keys(goal_dict_exp3["agent2"]))
    first_map_exp3 = first(keys(goal_dict_exp3["agent2"]))
    println("First map: $first_map_exp3")
    println("Keys for agent2/$first_map_exp3: ", keys(goal_dict_exp3["agent2"][first_map_exp3]))
    
    first_scenario = first(keys(goal_dict_exp3["agent2"][first_map_exp3]))
    println("Keys for scenario $first_scenario: ", keys(goal_dict_exp3["agent2"][first_map_exp3][first_scenario]))
    
    first_goal_exp3 = first(keys(goal_dict_exp3["agent2"][first_map_exp3][first_scenario]))
    println("Keys for goal $first_goal_exp3: ", keys(goal_dict_exp3["agent2"][first_map_exp3][first_scenario][first_goal_exp3]))
    
    first_state_exp3 = first(keys(goal_dict_exp3["agent2"][first_map_exp3][first_scenario][first_goal_exp3]))
    println("Sample data shape for goal=$first_goal_exp3, state=$first_state_exp3: ", 
            size(goal_dict_exp3["agent2"][first_map_exp3][first_scenario][first_goal_exp3][first_state_exp3]))
    
    # Test the exact access pattern from run_experiment_exp3.jl
    println("\n=== Testing exp3 access pattern ===")
    map_id = first_map_exp3
    scenario = first_scenario
    g = first_goal_exp3
    i = first_state_exp3
    
    println("Accessing: state_dict_exp3[\"agent2\"][$map_id][$scenario][$g][$i]")
    test_data = state_dict_exp3["agent2"][map_id][scenario][g][i]
    println("Success! Shape: ", size(test_data))
    println("Sample values (first 3 rows, first 3 cols):")
    println(test_data[1:min(3, size(test_data, 1)), 1:min(3, size(test_data, 2))])
else
    println("ERROR: 'agent2' not found in exp3 data!")
end

