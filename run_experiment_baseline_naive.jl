using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
using JSON
# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/translate.jl")
include("src/render.jl")

include("paths_new.jl")
# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_new")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")

#--- Initial Setup ---#
exp_pids = [p_id for p_id in keys(other_agent_goal) if !occursin("naive",p_id)]

steps_dict = Dict()


for p_id in exp_pids

    if occursin("naive",p_id)
        continue
    end

    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

    # Load problem
    # p_id = "s521_blue_exp"
    map_id = p_id[1:4]


    problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))

    goal_id = parse(Int, string(problem.goal.args[2])[end:end])
    # plan = paths[p_id]
    # Load plan
    # plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

    # Initialize and compile reference state
    state = initstate(domain, problem)


    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]


    plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

    steps_dict[p_id] = length(plan)

    for (i, action) in enumerate(plan)
        if action.name == :interact && action.args[end] in blue_wizards
            steps_dict[p_id] = i
        end
    end

end

# using JSON
open("step_dict_naive.json","w") do f
    JSON.print(f, steps_dict)
end

