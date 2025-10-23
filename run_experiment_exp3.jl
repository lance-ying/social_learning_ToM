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

# Define directory paths
experiment_id = "exp3"

PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_$experiment_id")

#--- Initial Setup ---#
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

steps_dict = Dict()

# Load inference data for both agents (agent2=Z, agent3=X)
data = load("inference_data_$experiment_id.jld2")
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]
possible_worlds = data["worlds"]

domain_render = load_domain(joinpath(@__DIR__, "dataset", "domain_render.pddl"))

action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

for (map_id, agent_goals) in metadata
    println("Processing map: $map_id")
    
    # Loop over both scenarios
    for scenario in 1:2
        map_key = "$(map_id)_scenario$(scenario)"
        println("  Scenario $scenario")
        
        # Get which gems each agent wants in this scenario
        agent2_gem = agent_goals["agent2"][scenario]  # Z's goal
        agent3_gem = agent_goals["agent3"][scenario]  # X's goal
        
        println("    agent2 (Z) -> gem$agent2_gem, agent3 (X) -> gem$agent3_gem")

        domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
        problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
        
        # Initialize and compile reference state
        state = initstate(domain, problem)
        state_render = copy(state)
        domain, state = PDDL.compiled(domain, problem)

        #--- Goal Inference Setup ---#
        
        # Specify possible goals for each agent
        goals_agent2, goal_names_agent2 = initialize_goals(state, :agent2)
        goals_agent3, goal_names_agent3 = initialize_goals(state, :agent3)

        # Enumerate over possible initial states
        initial_states, belief_probs, state_names = enumerate_beliefs(state)

        t = 0

        blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
        wizard_candicates = blue_wizards

        # Find current state ID
        s_id = -1
        for s in 1:length(initial_states)
            if check_equal_state(state, initial_states[s])
                s_id = s
                break
            end
        end

        # Load goal/state probabilities for both agents
        goal_probs_agent2 = goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id]
        state_probs_agent2 = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id]
        
        goal_probs_agent3 = goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id]
        state_probs_agent3 = state_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][s_id]

        new_state = copy(state_render)
        planner = AStarPlanner(GoalManhattan())
        plan = planner(domain, state, problem.goal)

        if !any(x-> x.name == :interact && x.args[end] in blue_wizards, plan)
            print("t=", 0)
            steps_dict[map_key] = Dict("t" => 0, "observed" => "none")
            continue
        end

        while !PDDL.satisfy(domain, state, problem.goal)
            
            # Compute Q_observe for agent2 (Z)
            Q_observe_agent2 = 0
            total_probs_agent2 = 0
            
            for g in 1:length(goals_agent2)
                if goal_probs_agent2[g, t+1] < 0.1
                    continue
                end
                
                for i in 1:length(initial_states)
                    if state_probs_agent2[i, t+1] < 0.1
                        continue
                    end
                    
                    T = -1
                    for val in 1:length(state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][1,:])
                        if any(x -> x>0.95, state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][:,val])
                            T = val
                            break
                        end
                    end
                    
                    if T == -1
                        for val in 1:length(goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][i][1,:])
                            if any(x -> x<0.1, goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][i][:,val])
                                T = val
                                break
                            end
                        end
                    end
                    
                    new_wizard_candicates = []
                    for j in 1:length(blue_wizards)
                        if state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][j, T] > 0.1
                            push!(new_wizard_candicates, blue_wizards[j])
                        end
                    end
                    
                    Q_T = estimate_self_exploration_cost(domain_render, new_state, problem.goal, new_wizard_candicates, action_cost)
                    Q_observe_agent2 += goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1] * (Q_T + action_cost[:observe] * max(T,1))
                    total_probs_agent2 += goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1]
                end
            end
            Q_observe_agent2 /= total_probs_agent2
            
            # Compute Q_observe for agent3 (X)
            Q_observe_agent3 = 0
            total_probs_agent3 = 0
            
            for g in 1:length(goals_agent3)
                if goal_probs_agent3[g, t+1] < 0.1
                    continue
                end
                
                for i in 1:length(initial_states)
                    if state_probs_agent3[i, t+1] < 0.1
                        continue
                    end
                    
                    T = -1
                    for val in 1:length(state_probs_conditioned_dict["agent3"][map_id][scenario][g][i][1,:])
                        if any(x -> x>0.95, state_probs_conditioned_dict["agent3"][map_id][scenario][g][i][:,val])
                            T = val
                            break
                        end
                    end
                    
                    if T == -1
                        for val in 1:length(goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][i][1,:])
                            if any(x -> x<0.1, goal_probs_conditioned_dict["agent3"][map_id][scenario][agent3_gem][i][:,val])
                                T = val
                                break
                            end
                        end
                    end
                    
                    new_wizard_candicates = []
                    for j in 1:length(blue_wizards)
                        if state_probs_conditioned_dict["agent3"][map_id][scenario][g][i][j, T] > 0.1
                            push!(new_wizard_candicates, blue_wizards[j])
                        end
                    end
                    
                    Q_T = estimate_self_exploration_cost(domain_render, new_state, problem.goal, new_wizard_candicates, action_cost)
                    Q_observe_agent3 += goal_probs_agent3[g, t+1] * state_probs_agent3[i, t+1] * (Q_T + action_cost[:observe] * max(T,1))
                    total_probs_agent3 += goal_probs_agent3[g, t+1] * state_probs_agent3[i, t+1]
                end
            end
            Q_observe_agent3 /= total_probs_agent3
            
            # Compute Q_not_observe
            Q_not_observe = estimate_self_exploration_cost(domain_render, new_state, problem.goal, wizard_candicates, action_cost)
            
            println("    t=$t: Q_observe_agent2=$Q_observe_agent2, Q_observe_agent3=$Q_observe_agent3, Q_not_observe=$Q_not_observe")
            
            # Take argmin to decide which action
            q_values = [Q_observe_agent2, Q_observe_agent3, Q_not_observe]
            best_action_idx = argmin(q_values)
            
            if best_action_idx == 1 && Q_observe_agent2 + 0.3 < Q_not_observe
                # Observe agent2 (Z)
                println("    -> Observing agent2 (Z)")
                t += 1
                wizard_candicates = []
                for j in 1:length(blue_wizards)
                    if state_probs_agent2[j, t+1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
            elseif best_action_idx == 2 && Q_observe_agent3 + 0.3 < Q_not_observe
                # Observe agent3 (X)
                println("    -> Observing agent3 (X)")
                t += 1
                wizard_candicates = []
                for j in 1:length(blue_wizards)
                    if state_probs_agent3[j, t+1] > 0.1
                        push!(wizard_candicates, blue_wizards[j])
                    end
                end
            else
                # Don't observe
                println("    -> Not observing (t=$t)")
                observed_agent = best_action_idx == 1 ? "agent2" : (best_action_idx == 2 ? "agent3" : "none")
                steps_dict[map_key] = Dict("t" => t, "observed" => observed_agent)
                break
            end
        end
    end
end


open("steps_dict_$experiment_id.json", "w") do io
    JSON.print(io, steps_dict, 4)
end

println("\n=== Experiment Complete ===")
println("Results saved to: steps_dict_$experiment_id.json")

