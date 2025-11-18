using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON
using FileIO, JLD2
PDDL.Arrays.register!()

include(joinpath(@__DIR__, "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "src", "translate.jl"))
include(joinpath(@__DIR__, "..", "src", "render.jl"))

experiment_id = "exp3"
PROBLEM_DIR = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id")

metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

# Load inference data
data = load(joinpath(@__DIR__, "..", "data", "inference", "inference_data_$experiment_id.jld2"))
goal_probs_conditioned_dict = data["goal"]
state_probs_conditioned_dict = data["state"]
possible_worlds = data["worlds"]

domain_render = load_domain(joinpath(@__DIR__, "..", "dataset", "domain_render.pddl"))
action_cost = Dict(:move => 2, :interact => 5, :observe => 1.0)

# Test only the FIRST map
map_id = first(keys(metadata))
agent_goals = metadata[map_id]

println("Testing map: $map_id")
println("Agent goals: $agent_goals")

scenario = 1
agent2_gem = agent_goals["agent2"][scenario]
agent3_gem = agent_goals["agent3"][scenario]

println("  Scenario $scenario: agent2 -> gem$agent2_gem, agent3 -> gem$agent3_gem")

domain = load_domain(joinpath(@__DIR__, "..", "dataset", "domain.pddl"))
problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))

state = initstate(domain, problem)
state_render = copy(state)
domain, state = PDDL.compiled(domain, problem)

# Load FILTERED problems to match inference data
include(joinpath(@__DIR__, "..", "src", "ascii.jl"))
function filter_ascii_agents(ascii_content::String, keep_agent::Symbol)
    agent_chars = Dict(:agent1 => 'M', :agent2 => 'Z', :agent3 => 'X')
    filtered = ascii_content
    for (agent_sym, char) in agent_chars
        if agent_sym != keep_agent
            filtered = replace(filtered, char => '.')
        end
    end
    return filtered
end

txt_path = joinpath(PROBLEM_DIR, "$(map_id).txt")
ascii_content = read(txt_path, String)

# Load the SAME filtered problem that inference used
domain_agent2 = load_domain(joinpath(@__DIR__, "..", "dataset", "domain.pddl"))
temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")

# If it doesn't exist, create it (same as inference did)
if !isfile(temp_path_agent2)
    filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
    write(temp_path_agent2, filtered_ascii_agent2)
end

problem_agent2 = load_ascii_problem(temp_path_agent2)
state_agent2 = initstate(domain_agent2, problem_agent2)
domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)

goals_agent2, goal_names_agent2 = initialize_goals(state_agent2, :agent2)
goals_agent3, goal_names_agent3 = initialize_goals(state, :agent3)

initial_states_agent2, belief_probs_agent2, state_names_agent2 = enumerate_beliefs(state_agent2)

println("Number of goals: $(length(goals_agent2))")
println("Number of initial states (agent2 filtered): $(length(initial_states_agent2))")

t = 0
blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
blue_wizards_agent2 = [w for w in PDDL.get_objects(state_agent2, :wizard) if state_agent2[pddl"(iscolor $w blue)"]]
println("Number of blue wizards (full): $(length(blue_wizards))")
println("Number of blue wizards (agent2 filtered): $(length(blue_wizards_agent2))")

# Find state ID using FILTERED agent2 state
s_id_agent2 = -1
println("\nDEBUG: Checking state equality...")
println("  state_agent2 type: $(typeof(state_agent2))")
println("  initial_states_agent2[1] type: $(typeof(initial_states_agent2[1]))")

for s in 1:length(initial_states_agent2)
    global s_id_agent2  # Fix Julia scoping issue
    matches = check_equal_state(state_agent2, initial_states_agent2[s])
    if s <= 3  # Print first 3 for debugging
        println("  State $s matches: $matches")
    end
    if matches
        s_id_agent2 = s
        break
    end
end

println("Current state ID (agent2): $s_id_agent2")

if s_id_agent2 == -1
    println("\nERROR: Could not find matching state!")
    println("This means the current state doesn't match any enumerated belief state.")
    println("Possible reasons:")
    println("  1. The filtered problem loaded here differs from inference")
    println("  2. The states are using different object orderings")
    println("  3. The temp file was modified between inference and now")
    error("State matching failed - cannot continue")
end

# Load initial probabilities
goal_probs_agent2 = goal_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]
state_probs_agent2 = state_probs_conditioned_dict["agent2"][map_id][scenario][agent2_gem][s_id_agent2]

println("goal_probs_agent2 shape: $(size(goal_probs_agent2))")
println("state_probs_agent2 shape: $(size(state_probs_agent2))")

new_state = copy(state_render)
wizard_candicates = blue_wizards

println("\n=== Starting Q-value computation at t=$t ===")

# Compute Q_observe for agent2
Q_observe_agent2 = 0
total_probs_agent2 = 0

println("Computing Q_observe_agent2...")
iteration_count = 0

for g in 1:length(goals_agent2)
    if goal_probs_agent2[g, t+1] < 0.1
        println("  Skipping goal $g (prob=$(goal_probs_agent2[g, t+1]))")
        continue
    end
    
    for i in 1:length(initial_states_agent2)
        global iteration_count, Q_observe_agent2, total_probs_agent2  # Fix Julia scoping
        
        if state_probs_agent2[i, t+1] < 0.1
            continue
        end
        
        iteration_count += 1
        println("  [$iteration_count] agent2: goal=$g, state=$i (prob=$(goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1]))")
        
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
        
        println("      T = $T")
        
        new_wizard_candicates = []
        for j in 1:length(blue_wizards_agent2)
            if state_probs_conditioned_dict["agent2"][map_id][scenario][g][i][j, T] > 0.1
                push!(new_wizard_candicates, blue_wizards_agent2[j])
            end
        end
        
        println("      Wizard candidates: $(length(new_wizard_candicates))")
        println("      Calling estimate_self_exploration_cost...")
        
        start_time = time()
        Q_T = estimate_self_exploration_cost(domain_render, new_state, problem.goal, new_wizard_candicates, action_cost)
        elapsed = time() - start_time
        
        println("      Q_T = $Q_T (took $(round(elapsed, digits=2))s)")
        
        Q_observe_agent2 += goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1] * (Q_T + action_cost[:observe] * max(T,1))
        total_probs_agent2 += goal_probs_agent2[g, t+1] * state_probs_agent2[i, t+1]
    end
end

Q_observe_agent2 /= total_probs_agent2
println("\nQ_observe_agent2 = $Q_observe_agent2 (total iterations: $iteration_count)")

println("\n=== Test completed successfully! ===")

