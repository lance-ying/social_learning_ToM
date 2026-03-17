# Wrapper script v2: adds inference phase and optional map filtering
# Scenarios run in separate Julia processes to avoid PDDL.compiled() type conflicts

using JSON

#==============================================================================#
#                           CONFIGURATION                                       #
#==============================================================================#

# Experiment identifier - used for problem directory and naming
experiment_id = "exp4_013026"

# Inference data file (relative to project root or absolute path)
inference_file = "inference_exp4_020126_1.jld2"

# Output prefix for results files (will have _scenario1.json, _scenario2.json, .json appended)
output_prefix = "steps_dict_exp4_031726_1"

#==============================================================================#

# Parse CLI flags
maps_arg = ""       # comma-separated map IDs, empty = all maps
skip_inference = false

let i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--maps" && i + 1 <= length(ARGS)
            global maps_arg = ARGS[i + 1]
            i += 2
        elseif ARGS[i] == "--skip-inference"
            global skip_inference = true
            i += 1
        else
            error("Unknown argument: $(ARGS[i])\nUsage: julia run_experiment_exp4_wrapper_v2.jl [--maps sm211,sm221,...] [--skip-inference]")
        end
    end
end

OUTPUT_DIR = joinpath(@__DIR__, "experiment_outputs")

println("=== Running Experiment $experiment_id (v2 wrapper) ===")
println("Inference file: $inference_file")
println("Output prefix: $output_prefix")
println("Map filter: $(isempty(maps_arg) ? "all maps" : maps_arg)")
println("Skip inference: $skip_inference")
println("This wrapper runs each phase separately to avoid type conflicts.\n")

# Phase 0 — Inference
if !skip_inference
    println("=" ^ 50)
    println("PHASE 0: Running inference")
    println("=" ^ 50)
    inference_script = joinpath(@__DIR__, "..", "utilities", "inference_multi_exp4.jl")
    run(`julia $inference_script $maps_arg $inference_file`)
else
    println("PHASE 0: Skipping inference (--skip-inference)\n")
end

# Phase 1 — Scenario 1
println("\n" * "=" ^ 50)
println("PHASE 1: Running all maps for Scenario 1")
println("=" ^ 50)
scenario1_script = joinpath(@__DIR__, "run_experiment_exp4_scenario1.jl")
run(`julia $scenario1_script $experiment_id $inference_file $output_prefix $maps_arg`)

# Phase 2 — Scenario 2
println("\n" * "=" ^ 50)
println("PHASE 2: Running all maps for Scenario 2")
println("=" ^ 50)
scenario2_script = joinpath(@__DIR__, "run_experiment_exp4_scenario2.jl")
run(`julia $scenario2_script $experiment_id $inference_file $output_prefix $maps_arg`)

# Phase 3 — Merge results
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
