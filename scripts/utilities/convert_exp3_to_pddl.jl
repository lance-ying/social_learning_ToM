using PDDL
# Register PDDL array theory
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "..", "src", "ascii.jl")

# Define directory path
PROBLEM_DIR = joinpath(@__DIR__, "..", "..", "dataset", "problems_exp3")

# Get all txt files
txt_files = filter(f -> endswith(f, ".txt") && !startswith(f, "."), readdir(PROBLEM_DIR))

println("Found $(length(txt_files)) txt files to convert")
println("="^60)

for txt_file in txt_files
    map_id = splitext(txt_file)[1]  # e.g., "sm211"
    
    println("Converting: $txt_file")
    
    # Load the ASCII problem
    txt_path = joinpath(PROBLEM_DIR, txt_file)
    problem = load_ascii_problem(txt_path)
    
    # Write to PDDL file (using string() to convert GenericProblem)
    pddl_path = joinpath(PROBLEM_DIR, "$(map_id).pddl")
    write(pddl_path, string(problem))
    
    println("  -> Created: $(map_id).pddl")
end

println("="^60)
println("Conversion complete! Created $(length(txt_files)) PDDL files")





