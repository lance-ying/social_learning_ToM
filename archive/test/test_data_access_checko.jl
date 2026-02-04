using FileIO, JLD2

# Load data
data_exp2 = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_exp2.jld2"))
goal_dict = data_exp2["goal"]
state_dict = data_exp2["state"]

# Explore the structure
println("Top-level keys in goal dict: ", keys(goal_dict))

# Get first map
first_map = first(keys(goal_dict))
println("First map: $first_map")
println("Keys for $first_map: ", keys(goal_dict[first_map]))

# Get first goal
first_goal = first(keys(goal_dict[first_map]))
println("Keys for goal $first_goal: ", keys(goal_dict[first_map][first_goal]))

# Get first state and sample data
first_state = first(keys(goal_dict[first_map][first_goal]))
println("Sample data shape: ", size(goal_dict[first_map][first_goal][first_state]))