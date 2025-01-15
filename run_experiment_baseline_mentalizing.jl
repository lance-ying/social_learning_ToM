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


# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_new")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")

#--- Initial Setup ---#
exp_pids = [p_id for p_id in keys(other_agent_goal) if !occursin("naive",p_id)]

include("paths_new.jl")


steps_dict = Dict()

# steps_dict = JSON.parsefile("/Users/lance/Documents/GitHub/ObserveMove/step_dict.json") 

goal_probs_conditioned_dict = load("inference_data_temp_new.jld2", "goal")
state_probs_conditioned_dict = load("inference_data_temp_new.jld2", "state")
possible_worlds = load("inference_data_temp_new.jld2", "worlds")


domain_render = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 1)

for p_id in exp_pids
# for p_id in ["s411_blue_exp"]

    if occursin("naive",p_id)
        continue
    end

    println(p_id)

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

    state_render = copy(state)

    # heuristic = GoalManhattan()
    # planner = AStarPlanner(heuristic)

    # domain, state = PDDL.compiled(domain, problem)

    # Render initial state

    #--- Goal Inference Setup ---#

    # Specify possible goals
    goals = @pddl(
        "(has agent2 gem1)",
        "(has agent2 gem2)",
        "(has agent2 gem3)"
    )


    initial_states, belief_probs, state_names = enumerate_beliefs(
        state
    )

    # print(possible_worlds[map_id])

    # initial_states, belief_probs, state_names = possible_worlds[map_id]


    # print(initial_states)

    t= 0

    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    # wizard_candicates = blue_wizards


    g_id = other_agent_goal[p_id]
    s_id = -1

    # println(check_equal_state(state, initial_states[4]))

    for s in 1:length(initial_states)
        if check_equal_state(state, initial_states[s])
            s_id = s
        end
    end

    # println(s_id)

    goal_probs = goal_probs_conditioned_dict[map_id][g_id][s_id]
    state_probs = state_probs_conditioned_dict[map_id][g_id][s_id]

    new_state = copy(state_render)


    planner = AStarPlanner(GoalManhattan())

    plan = planner(domain, state, problem.goal)

    plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))


    T = 1

    # for val in 1:min(length(goal_probs_conditioned_dict[map_id][g_id][s_id][1,:]), 10)
    #     if goal_probs_conditioned_dict[map_id][g_id][s_id][goal_id,val+1] < 0.1
    #         print("t=", val)
    #         T = val
    #         break
    #     end
    # end





    for t in 1:length(plan)

        curr_state_dist = state_probs[:,t]

        flag = true

        for g in 1:3
            if goal_probs[g, t+1] > 0.1
                for s in 1: length(initial_states)
                    if state_probs[s, t+1] > 0.1
                        for val in t:length(state_probs_conditioned_dict[map_id][g][s][1,:])
                            println(state_probs_conditioned_dict[map_id][g][s][:,val], curr_state_dist)
                            if eval_state_dist(curr_state_dist, state_probs_conditioned_dict[map_id][g][s][:,val])
                                flag = false
                                break
                            end
                        end
                    end
                end
            end
        end

        if flag
            T = t
            break
        end
        
    end

    steps_dict[p_id] = T

    print(p_id, ", t=", T)

end

# using JSON
open("step_dict_mentalize.json","w") do f
    JSON.print(f, steps_dict)
end

