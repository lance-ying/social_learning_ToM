using FileIO, JLD2
using JSON

# Load inference data
experiment_id = "exp3"
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
goal_probs_conditioned_dict = data["goal"]

# Load metadata
metadata_path = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id", "metadata.json")
metadata = JSON.parsefile(metadata_path)

println("Maps in metadata: ", sort(collect(keys(metadata))))
println("\nMaps in inference data: ", sort(collect(keys(goal_probs_conditioned_dict))))

println("\nMissing from inference data:")
for map_id in keys(metadata)
    if !haskey(goal_probs_conditioned_dict, map_id)
        println("  - $map_id")
    end
end

println("\nExtra in inference data (not in metadata):")
for map_id in keys(goal_probs_conditioned_dict)
    if !haskey(metadata, map_id)
        println("  - $map_id")
    end
end

