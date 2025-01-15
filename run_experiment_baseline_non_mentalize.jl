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
exp_pids = [p_id for p_id in keys(paths) if !occursin("naive",p_id)]

steps_dict = Dict()

# steps_dict = JSON.parsefile("/Users/lance/Documents/GitHub/ObserveMove/step_dict.json") 

goal_probs_conditioned_dict = load("inference_data_temp_new.jld2", "goal")
state_probs_conditioned_dict = load("inference_data_temp_new.jld2", "state")
possible_worlds = load("inference_data_temp_new.jld2", "worlds")


domain_render = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 0.5)

for p_id in exp_pids

    if occursin("naive",p_id)
        continue
    end

    println(p_id)

    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

    # Load problem
    # p_id = "s521_blue_exp"
    map_id = p_id[1:4]


    problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
    # plan = paths[p_id]
    # Load plan
    # plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

    # Initialize and compile reference state
    state = initstate(domain, problem)

    state_render = copy(state)


    # Render initial state

    #--- Goal Inference Setup ---#

    # Specify possible goals
    goals = @pddl(
        "(has agent2 gem1)",
        "(has agent2 gem2)",
        "(has agent2 gem3)"
    )

    # Enumerate over possible initial states
    # initial_states, belief_probs, state_names = enumerate_beliefs(
    #     state
    # )

    t= 0

    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    # wizard_candicates = blue_wizards


    g_id = other_agent_goal[p_id]
    s_id = -1

    for s in 1:length(initial_states)
        if check_equal_state(state, initial_states[s])
            s_id = s
        end
    end

    new_state = copy(state_render)


    Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, blue_wizards, action_cost)

    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]


    plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

    steps_dict[p_id] = length(plan)

    T= -1

    for (i, action) in enumerate(plan)
        if action.name == :interact && action.args[end] in blue_wizards
            T = i
        end
    end

    if T == -1
        T = length(plan)
    end

    plan_main = collect(planner(domain, state, problem.goal))

    Q_observe = calculate_plan_cost(plan_main, action_cost)

    Q_observe = Q_observe + action_cost[:observe]*T

    if Q_observe < Q_not_observe
        steps_dict[p_id] = T
    else
        steps_dict[p_id] = 0
    end


end

# using JSON
open("step_dict_nonmentalize.json","w") do f
    JSON.print(f, steps_dict)
end

