#!/usr/bin/env julia
"""
Print state probabilities over time for sm221 scenario 2 agent3
"""

using JLD2, FileIO

# Load the inference data
data_path = joinpath(@__DIR__, "..", "..", "data", "inference", "inference_exp4_020126_1.jld2")
data = load(data_path)

# Extract state probs for sm221, scenario 2, agent3, goal 1, state 1
# Indices: [agent][map][scenario][goal][initial_state]
state_probs = data["state"]["agent3"]["sm221"][2][1][1]

println("Extracting state probabilities for:")
println("  Map: sm221")
println("  Scenario: 2")
println("  Agent: agent3")
println("  Goal: 1 (gem1)")
println("  Initial state: 1")
println()

# Print as CSV
println("timestep,state_1_prob,state_2_prob")
for t in 1:size(state_probs, 2)
    println("$(t-1),$(round(state_probs[1,t], digits=3)),$(round(state_probs[2,t], digits=3))")
end
