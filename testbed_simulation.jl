using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/beliefs.jl")
include("src/translate.jl")
include("src/render.jl")
include("paths_new.jl")

# include("paths_new.jl")
# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_new")
PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")

#--- Initial Setup ---#

# Load domain


action_cost = Dict(:move => 2, :interact => 4, :observe =>1.0)

# Load problem
p_id = "s342_other_exp"
map_id = p_id[1:4]
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
# plan = paths[p_id]
# Load plan
# plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

# Initialize and compile reference state
state = initstate(domain, problem)

state_render = copy(state)
domain_render = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))
# heuristic = GoalManhattan()
# planner = AStarPlanner(heuristic)

domain, state = PDDL.compiled(domain, problem)

# Render initial state

#--- Goal Inference Setup ---#

# Specify possible goals
goals = @pddl(
    "(has agent2 gem1)",
    "(has agent2 gem2)",
    "(has agent2 gem3)"
)

# goal_names = ["A", "B", "C"]
# goal_colors = gem_colors


# Enumerate over possible initial states
initial_states, belief_probs, state_names = enumerate_beliefs(
    state
)


g_id = other_agent_goal[p_id]
s_id = -1


for s in 1:length(initial_states)
    if check_equal_state(state, initial_states[s])
        s_id = s
    end
end


# goal_probs_conditioned_dict = load("inference_data_temp_new.jld2", "goal")
# state_probs_conditioned_dict = load("inference_data_temp_new.jld2", "state")
# possible_worlds = load("inference_data_temp_new.jld2", "worlds")

# possible_worlds_render = load("possible_worlds_render.jld2")

goal_probs = goal_probs_conditioned_dict[map_id][g_id][s_id]
state_probs = state_probs_conditioned_dict[map_id][g_id][s_id]

new_state = copy(state_render)

t= 0

blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
wizard_candicates = blue_wizards


planner = AStarPlanner(GoalManhattan())

plan = planner(domain, state, problem.goal)


if !any(x-> x.name == :interact && x.args[end] in blue_wizards, plan)
    print("t=", 0)
else
    while true

        Q_observe = 0

        print(goal_probs[:, t+1])

        for g in 1:length(goals)

            if goal_probs[g, t+1] < 0.01
                continue
            end

            for i in 1:length(initial_states)

                print()

                if state_probs[i, t+1] < 0.1
                    continue
                end

                T = -1

                for val in 1:length(state_probs_conditioned_dict[map_id][g][i][1,:])

                    # print(state_probs_conditioned_dict[map_id][g][i][:,val])
        
                    if any(x -> x>0.95, state_probs_conditioned_dict[map_id][g][i][:,val])
                        T = val
                        break
                    end
                end

                if T == -1

                    for val in 1:length(goal_probs_conditioned_dict[map_id][g_id][i][g,:])

                        print(goal_probs_conditioned_dict[map_id][g][i][g_id,val])
            
                        if goal_probs_conditioned_dict[map_id][g][i][g_id,val] < 0.1
                            T = val
                            break
                        end
                    end
                end

                new_wizard_candicates = []


                for j in 1:length(blue_wizards)

                    # print(j, T)
                    if state_probs_conditioned_dict[map_id][g][i][j, T] > 0.1
                        push!(new_wizard_candicates, blue_wizards[j])
                    end
                end

                Q_T = estimate_self_exploration_cost(domain_render, new_state, problem.goal, new_wizard_candicates, action_cost)

                # println(g," ",i, " ",goal_probs[g, t+1]," ", state_probs[i, t+1]," ",T-1," ", new_wizard_candicates, " ",Q_T + action_cost[:observe] * T)

                Q_observe += goal_probs[g, t+1] * state_probs[i, t+1] * (Q_T + action_cost[:observe] * max(T-1,1))

            end

        end 

        print("checkpt2",wizard_candicates )

        Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, wizard_candicates, action_cost)

        print("Q_observe = ", Q_observe, "Q_not_observe = ", Q_not_observe)
        println()

        if Q_observe+0.5 < Q_not_observe
            t+=1

            wizard_candicates = []

            for j in 1:length(blue_wizards)
                if state_probs[j, t+1] > 0.1
                    push!(wizard_candicates, blue_wizards[j])
                end
            end
        else
            print("t = ", t)
            break
        end
    end
end




