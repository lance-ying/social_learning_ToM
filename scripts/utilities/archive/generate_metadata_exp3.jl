using JSON

# Parse all txt files in problems_exp3 to extract agent goals
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_exp3")

metadata = Dict()

txt_files = filter(f -> endswith(f, ".txt"), readdir(PROBLEM_DIR))

for txt_file in txt_files
    map_id = splitext(txt_file)[1]  # e.g., "sm211"
    
    # Read the file
    content = read(joinpath(PROBLEM_DIR, txt_file), String)
    lines = split(content, '\n')
    
    # Find the lines with agent goals (they're at the bottom)
    agent2_goals = nothing
    agent3_goals = nothing
    
    for line in lines
        line = strip(line)
        if startswith(line, "X:")
            # Parse "X: 1, 3" -> [1, 3]
            goal_str = replace(line, "X:" => "")
            agent2_goals = parse.(Int, split(strip(goal_str), ","))
        elseif startswith(line, "Y:")
            # Parse "Y: 2, 1" -> [2, 1]
            goal_str = replace(line, "Y:" => "")
            agent3_goals = parse.(Int, split(strip(goal_str), ","))
        end
    end
    
    if agent2_goals !== nothing && agent3_goals !== nothing
        metadata[map_id] = Dict(
            "agent2" => agent2_goals,
            "agent3" => agent3_goals
        )
        println("$map_id: agent2=$agent2_goals, agent3=$agent3_goals")
    else
        println("Warning: Could not parse goals for $map_id")
    end
end

# Write metadata.json
output_path = joinpath(PROBLEM_DIR, "metadata.json")
open(output_path, "w") do f
    JSON.print(f, metadata, 4)
end

println("\nMetadata saved to: $output_path")
println("Total maps: $(length(metadata))")





