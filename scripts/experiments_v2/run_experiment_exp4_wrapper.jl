# Wrapper script that runs scenario 1 and scenario 2 separately
# This avoids PDDL.compiled() type conflicts between scenarios

using JSON

#==============================================================================#
#                           CONFIGURATION                                       #
#==============================================================================#

# Experiment identifier - used for problem directory and naming
experiment_id = "exp4_013026"

# Inference data file (relative to project root or absolute path)
inference_file = "inference_exp4_020126_1.jld2"

# Output prefix for results files (will have _scenario1.json, _scenario2.json, .json appended)
output_prefix = "steps_dict_exp4_031726_2"

#==============================================================================#

REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
OUTPUT_DIR = joinpath(REPO_ROOT, "model_outputs", "experiments_v2", "exp4")
CANONICAL_OUTPUT_DIR = OUTPUT_DIR
DETAIL_OUTPUT_DIR = joinpath(OUTPUT_DIR, "_wrapper_artifacts")
mkpath(DETAIL_OUTPUT_DIR)
mkpath(CANONICAL_OUTPUT_DIR)

function relocate_top_level_wrapper_artifacts!(output_dir::String, detail_output_dir::String)
    for name in readdir(output_dir)
        src = joinpath(output_dir, name)
        if !isfile(src)
            continue
        end
        if startswith(name, "steps_dict_exp4_")
            mv(src, joinpath(detail_output_dir, name); force=true)
        end
    end
end

relocate_top_level_wrapper_artifacts!(OUTPUT_DIR, DETAIL_OUTPUT_DIR)

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

results1_path = joinpath(DETAIL_OUTPUT_DIR, "$(output_prefix)_scenario1.json")
results2_path = joinpath(DETAIL_OUTPUT_DIR, "$(output_prefix)_scenario2.json")
merged_path = joinpath(DETAIL_OUTPUT_DIR, "$(output_prefix).json")
trace1_path = joinpath(DETAIL_OUTPUT_DIR, "$(output_prefix)_scenario1_replay_trace.json")
trace2_path = joinpath(DETAIL_OUTPUT_DIR, "$(output_prefix)_scenario2_replay_trace.json")
merged_trace_path = joinpath(DETAIL_OUTPUT_DIR, "$(output_prefix)_replay_trace.json")

results1 = JSON.parsefile(results1_path)
results2 = JSON.parsefile(results2_path)
trace1 = JSON.parsefile(trace1_path)
trace2 = JSON.parsefile(trace2_path)

# Merge the two dictionaries
merged = merge(results1, results2)
merged_trace = merge(trace1, trace2)

open(merged_path, "w") do io
    JSON.print(io, merged, 4)
end

open(merged_trace_path, "w") do io
    JSON.print(io, merged_trace, 4)
end

open(joinpath(CANONICAL_OUTPUT_DIR, "steps_dict.json"), "w") do io
    JSON.print(io, merged, 4)
end

open(joinpath(CANONICAL_OUTPUT_DIR, "replay_trace.json"), "w") do io
    JSON.print(io, merged_trace, 4)
end

println("Scenario 1 results: $results1_path")
println("Scenario 2 results: $results2_path")
println("Merged results: $merged_path")
println("Merged replay trace: $merged_trace_path")
println("Canonical exp4 outputs: $CANONICAL_OUTPUT_DIR")
println("\n=== Experiment Complete ===")
