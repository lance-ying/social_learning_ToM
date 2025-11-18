using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JLD2, FileIO
using JSON
# Register PDDL array theory
PDDL.Arrays.register!()

include("../../src/ascii.jl")

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_exp2_test")

problem = convert_ascii_problem(joinpath("/Users/heyodogo/Documents/labs/ryan_lab/work/Social_Learning/social_learning_ToM/dataset/problems_exp2_test/sm541_test.txt"))

println(problem)    