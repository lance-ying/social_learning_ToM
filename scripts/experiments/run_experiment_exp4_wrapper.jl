# Wrapper script that runs scenario 1 and scenario 2 separately
# This avoids PDDL.compiled() type conflicts between scenarios

using JSON

#==============================================================================#
#                           CONFIGURATION                                       #
#==============================================================================#

# Experiment identifier - used for problem directory and naming
experiment_id = "exp4_012626"

# Inference data file (relative to project root or absolute path)
inference_file = "inference_data_exp4_012626.jld2"

# Output prefix for results files (will have _scenario1.json, _scenario2.json, .json appended)
output_prefix = "steps_dict_exp4_012626"

#==============================================================================#

OUTPUT_DIR = joinpath(@__DIR__, "experiment_outputs")

println("=== Running Experiment $experiment_id ===")
println("Inference file: $inference_file")
println("Output prefix: $output_prefix")
println("This wrapper runs each scenario separately to avoid type conflicts.\n")

# Run scenario 1
println("=" ^ 50)
println("PHASE 1: Running all maps for Scenario 1")
println("=" ^ 50)
scenario1_script = joinpath(@__DIR__, "run_experiment_exp4_scenario1.jl")
run(`julia $scenario1_script $experiment_id $inference_file $output_prefix`)

# Run scenario 2
println("\n" * "=" ^ 50)
println("PHASE 2: Running all maps for Scenario 2")
println("=" ^ 50)
scenario2_script = joinpath(@__DIR__, "run_experiment_exp4_scenario2.jl")
run(`julia $scenario2_script $experiment_id $inference_file $output_prefix`)

# Merge results
println("\n" * "=" ^ 50)
println("PHASE 3: Merging results")
println("=" ^ 50)

results1_path = joinpath(OUTPUT_DIR, "$(output_prefix)_scenario1.json")
results2_path = joinpath(OUTPUT_DIR, "$(output_prefix)_scenario2.json")
merged_path = joinpath(OUTPUT_DIR, "$(output_prefix).json")

results1 = JSON.parsefile(results1_path)
results2 = JSON.parsefile(results2_path)

# Merge the two dictionaries
merged = merge(results1, results2)

open(merged_path, "w") do io
    JSON.print(io, merged, 4)
end

println("Scenario 1 results: $results1_path")
println("Scenario 2 results: $results2_path")
println("Merged results: $merged_path")
println("\n=== Experiment Complete ===")
