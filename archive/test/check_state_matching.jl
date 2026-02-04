using PDDL, SymbolicPlanners
using FileIO, JLD2
using JSON
include(joinpath(@__DIR__, "..", "src", "plan_io.jl"))
include(joinpath(@__DIR__, "..", "src", "utils.jl"))
include(joinpath(@__DIR__, "..", "src", "heuristics.jl"))
include(joinpath(@__DIR__, "..", "src", "beliefs.jl"))
include(joinpath(@__DIR__, "..", "src", "ascii.jl"))

PDDL.Arrays.register!()

experiment_id = "exp3"
PROBLEM_DIR = joinpath(@__DIR__, "..", "dataset", "problems_$experiment_id")
metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)

for map_id in ["sm332", "sm342"]
    println("\n" * "="^60)
    println("Checking map: $map_id")
    println("="^60)
    
    agent_goals = metadata[map_id]
    scenario = 1
    agent2_gem = agent_goals["agent2"][scenario]
    agent3_gem = agent_goals["agent3"][scenario]
    
    println("Scenario $scenario: agent2 -> gem$agent2_gem, agent3 -> gem$agent3_gem")
    
    # Load full problem
    domain = load_domain(joinpath(@__DIR__, "..", "dataset", "domain.pddl"))
    problem = load_ascii_problem(joinpath(PROBLEM_DIR, "$(map_id).txt"))
    state = initstate(domain, problem)
    domain, state = PDDL.compiled(domain, problem)
    
    # Check blue wizards
    blue_wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    println("\nBlue wizards in full state: $blue_wizards")
    
    # Check agent1's plan
    planner = AStarPlanner(GoalManhattan())
    plan = planner(domain, state, problem.goal)
    println("Agent1 plan length: $(length(plan))")
    interact_actions = [x for x in plan if x.name == :interact]
    println("Interact actions: $interact_actions")
    wizard_interacts = [x for x in interact_actions if x.args[end] in blue_wizards]
    println("Blue wizard interactions: $wizard_interacts")
    
    # Load filtered problem for agent2
    ascii_content = read(joinpath(PROBLEM_DIR, "$(map_id).txt"), String)
    function filter_ascii_agents(ascii_content::String, keep_agent::Symbol)
        agent_chars = Dict(:agent1 => 'M', :agent2 => 'X', :agent3 => 'Y')
        filtered = ascii_content
        for (agent_sym, char) in agent_chars
            if agent_sym != keep_agent
                filtered = replace(filtered, char => '.')
            end
        end
        return filtered
    end
    
    filtered_ascii_agent2 = filter_ascii_agents(ascii_content, :agent2)
    temp_path_agent2 = joinpath(PROBLEM_DIR, ".temp_agent2_$(map_id).txt")
    write(temp_path_agent2, filtered_ascii_agent2)
    
    domain_agent2 = load_domain(joinpath(@__DIR__, "..", "dataset", "domain.pddl"))
    problem_agent2 = load_ascii_problem(temp_path_agent2)
    state_agent2 = initstate(domain_agent2, problem_agent2)
    domain_agent2, state_agent2 = PDDL.compiled(domain_agent2, problem_agent2)
    
    # Enumerate states for agent2
    initial_states_agent2, belief_probs_agent2, state_names_agent2 = enumerate_beliefs(state_agent2)
    println("\nAgent2: $(length(initial_states_agent2)) possible initial states")
    
    # Find matching state
    s_id_agent2 = -1
    for s in 1:length(initial_states_agent2)
        if check_equal_state(state_agent2, initial_states_agent2[s])
            s_id_agent2 = s
            println("  Matched state: $s_id_agent2")
            break
        end
    end
    
    if s_id_agent2 == -1
        println("  ERROR: No matching state found for agent2!")
    end
    
    # Same for agent3
    filtered_ascii_agent3 = filter_ascii_agents(ascii_content, :agent3)
    temp_path_agent3 = joinpath(PROBLEM_DIR, ".temp_agent3_$(map_id).txt")
    write(temp_path_agent3, filtered_ascii_agent3)
    
    domain_agent3 = load_domain(joinpath(@__DIR__, "..", "dataset", "domain.pddl"))
    problem_agent3 = load_ascii_problem(temp_path_agent3)
    state_agent3 = initstate(domain_agent3, problem_agent3)
    domain_agent3, state_agent3 = PDDL.compiled(domain_agent3, problem_agent3)
    
    initial_states_agent3, belief_probs_agent3, state_names_agent3 = enumerate_beliefs(state_agent3)
    println("\nAgent3: $(length(initial_states_agent3)) possible initial states")
    
    s_id_agent3 = -1
    for s in 1:length(initial_states_agent3)
        if check_equal_state(state_agent3, initial_states_agent3[s])
            s_id_agent3 = s
            println("  Matched state: $s_id_agent3")
            break
        end
    end
    
    if s_id_agent3 == -1
        println("  ERROR: No matching state found for agent3!")
    end
    
    println("\nBoth agents have s_id=$s_id_agent2: ", s_id_agent2 == s_id_agent3)
end

