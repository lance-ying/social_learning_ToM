using JSON

# Parse all txt files in problems_exp4 to extract agent goals with naive/actual types
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_exp4")

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
            # Parse "X: 1n, 3n" -> [Dict("gem" => 1, "type" => "naive"), Dict("gem" => 3, "type" => "naive")]
            goal_str = replace(line, "X:" => "")
            goal_parts = split(strip(goal_str), ",")
            agent2_goals = []
            for part in goal_parts
                part = strip(part)
                # Extract number and suffix (n or a)
                if endswith(part, "n")
                    gem_num = parse(Int, part[1:end-1])
                    push!(agent2_goals, Dict("gem" => gem_num, "type" => "naive"))
                elseif endswith(part, "a")
                    gem_num = parse(Int, part[1:end-1])
                    push!(agent2_goals, Dict("gem" => gem_num, "type" => "actual"))
                else
                    # Fallback: if no suffix, assume actual
                    gem_num = parse(Int, part)
                    push!(agent2_goals, Dict("gem" => gem_num, "type" => "actual"))
                end
            end
        elseif startswith(line, "Y:")
            # Parse "Y: 2a, 1a" -> [Dict("gem" => 2, "type" => "actual"), Dict("gem" => 1, "type" => "actual")]
            goal_str = replace(line, "Y:" => "")
            goal_parts = split(strip(goal_str), ",")
            agent3_goals = []
            for part in goal_parts
                part = strip(part)
                # Extract number and suffix (n or a)
                if endswith(part, "n")
                    gem_num = parse(Int, part[1:end-1])
                    push!(agent3_goals, Dict("gem" => gem_num, "type" => "naive"))
                elseif endswith(part, "a")
                    gem_num = parse(Int, part[1:end-1])
                    push!(agent3_goals, Dict("gem" => gem_num, "type" => "actual"))
                else
                    # Fallback: if no suffix, assume actual
                    gem_num = parse(Int, part)
                    push!(agent3_goals, Dict("gem" => gem_num, "type" => "actual"))
                end
            end
        end
    end
    
    if agent2_goals !== nothing && agent3_goals !== nothing
        metadata[map_id] = Dict(
            "agent2" => agent2_goals,
            "agent3" => agent3_goals
        )
        println("$map_id: agent2=$(agent2_goals), agent3=$(agent3_goals)")
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


