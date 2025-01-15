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

model = "naive"


step_dict = JSON.parsefile(joinpath(@__DIR__, "step_dict_$(model).json"))
# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_new")
PLAN_DIR = joinpath(@__DIR__, "results", "plans", model)
if !isdir(PLAN_DIR)
    mkpath(PLAN_DIR)
end
#--- Initial Setup ---#
exp_pids = [p_id for p_id in keys(other_agent_goal) if !occursin("naive",p_id)]

state_probs_conditioned_dict = load("inference_data_temp_new.jld2", "state")

# domain_render = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 1)

for p_id in exp_pids
# for p_id in ["s321_other_exp"]

    if occursin("naive",p_id)
        continue
    end

    println(p_id)

    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

    map_id = p_id[1:4]


    problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))

    goal= problem.goal

    # Initialize and compile reference state
    state = initstate(domain, problem)

    domain, state = PDDL.compiled(domain, problem)

    initial_states, belief_probs, state_names = enumerate_beliefs(
        state
    )

    observe = step_dict[p_id]

    planner = AStarPlanner(GoalManhattan())

    g_id = other_agent_goal[p_id]
    s_id = -1

    open("/Users/lance/Documents/GitHub/ObserveMove/results/plans/$(model)/$(p_id).pddl", "w") do file

        for i in 1:observe
            println(file, "(observe agent1)")
        end
    end

    print(initial_states[2])

    for s in 1:length(initial_states)
        if check_equal_state(state, initial_states[s])
            s_id = s
        end
    end


        # println(length(state_probs_conditioned_dict[map_id][g_id][s_id][1,:]))
    
    
    if observe == 0 || any(x -> x>0.9, state_probs_conditioned_dict[map_id][g_id][s_id][:,observe+1])
        plan = planner(domain, state, goal)

        open("/Users/lance/Documents/GitHub/ObserveMove/results/plans/$(model)/$(p_id).pddl", "w") do file

            for i in 1:observe
                println(file, "(observe agent1)")
            end
            for action in plan
                println(file, PDDL.write_pddl(action))
            end
        end
    else
        explored_state = []
        cost = 9999
        curr_state = initial_states[1]
        curr_state_id = 1
        plan = []
        for i in 1:length(initial_states)
            plan_temp = collect(planner(domain, initial_states[i], goal))
            if length(plan_temp) < cost
                cost = length(plan_temp)
                curr_state_id = i
                plan = plan_temp
            end
        end
        
        push!(explored_state, curr_state_id)
        curr_state = initial_states[curr_state_id]
        println(curr_state_id, check_equal_state(state, curr_state))
        # t = 1
        
        # plan = []
        while !PDDL.satisfy(domain, state, problem.goal)
            # println(plan[1])
            state = PDDL.execute(domain, state, plan[1])
            curr_state = PDDL.execute(domain, curr_state, plan[1])
            for i in 1:length(initial_states)
                initial_states[i] = PDDL.execute(domain, initial_states[i], plan[1])
            end

            open("/Users/lance/Documents/GitHub/ObserveMove/results/plans/$(model)/$(p_id).pddl", "a") do file
                println(file,  PDDL.write_pddl(plan[1]))
                # write(file, "smt")
            end

            # println(plan[1].name)
            # println(state[pddl"(hold wizard2 key1)"])
            # println(curr_state[pddl"(hold wizard2 key1)"])
            # println(check_equal_state(state, curr_state))
            
            if (plan[1].name == :interact) && (!check_equal_state(state, curr_state))

                println("flag")
                cost = 999
                for i in 1:length(initial_states)
                    # println(explored_state)
                    if i in explored_state
                        continue
                    end
                    plan_temp = collect(planner(domain, initial_states[i], goal))
                    if length(plan_temp) < cost
                        cost = length(plan_temp)
                        curr_state_id = i
                        plan = plan_temp
        
                    end
                end
                    push!(explored_state, curr_state_id)
                    curr_state = initial_states[curr_state_id]
                    print(curr_state_id, check_equal_state(state, curr_state))
            else
                plan = plan[2:end]
            end
        end
    end


end

