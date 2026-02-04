using FileIO, JLD2
using JSON

experiment_id = "exp3"
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
possible_worlds = data["worlds"]

# Compare a working map (sm431) with problematic ones (sm332, sm342)
for map_id in ["sm431", "sm332", "sm342"]
    println("\n" * "="^60)
    println("Map: $map_id")
    println("="^60)
    
    if haskey(possible_worlds["agent2"], map_id)
        map_data = possible_worlds["agent2"][map_id]
        println("Type: ", typeof(map_data))
        if isa(map_data, Vector)
            println("Length: ", length(map_data))
            for i in 1:min(2, length(map_data))
                println("  Element $i type: ", typeof(map_data[i]))
                if isa(map_data[i], Dict)
                    println("    Keys: ", keys(map_data[i]))
                end
            end
        end
    end
end

