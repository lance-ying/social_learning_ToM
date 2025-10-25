using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie
using JSON

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("src/ascii.jl")

# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_exp3")
PLAN_DIR = joinpath(@__DIR__, "results", "plans", "exp3")

# Create the plans directory if it doesn't exist
if !isdir(PLAN_DIR)
    mkpath(PLAN_DIR)
end

#--- Helper Functions ---#

"""
Filter ASCII content to remove other agents, keeping only the target agent.
This creates a simplified problem with only one agent for faster planning.
"""
function filter_ascii_agents(ascii_content::String, keep_agent::Symbol)
    agent_chars = Dict(:agent1 => 'M', :agent2 => 'X', :agent3 => 'Y')
    agent_metadata = Dict(:agent1 => "M:", :agent2 => "X:", :agent3 => "Y:")
    
    lines = split(ascii_content, '\n')
    filtered_lines = []
    
    for line in lines
        # Filter the grid content (replace other agents with dots)
        if !startswith(strip(line), "agent") && !startswith(strip(line), "X:") && !startswith(strip(line), "Y:") && !startswith(strip(line), "M:")
            filtered_line = line
            for (agent_sym, char) in agent_chars
                if agent_sym != keep_agent
                    filtered_line = replace(filtered_line, char => '.')
                end
            end
            push!(filtered_lines, filtered_line)
        # Filter metadata lines (keep only the target agent's metadata)
        elseif startswith(strip(line), agent_metadata[keep_agent])
            push!(filtered_lines, line)
        end
        # Skip metadata lines for other agents
    end
    
    return join(filtered_lines, '\n')
end

#--- Initial Setup ---#

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
action_cost = Dict(:move => 2, :interact => 4, :observe => 1.0)

# Load metadata for exp3 scenarios
metadata = JSON.parsefile(joinpath(PROBLEM_DIR, "metadata.json"))

# Get all stimulus IDs from metadata
stimulus_ids = collect(keys(metadata))

plan_dict = Dict()

for stimulus_id in stimulus_ids
    println("Processing stimulus: $stimulus_id")
    
    # Load the ASCII content for filtering
    txt_path = joinpath(PROBLEM_DIR, "$(stimulus_id).txt")
    ascii_content = read(txt_path, String)
    
    # Get agent goals from metadata
    agent_goals = metadata[stimulus_id]
    
    # Process each agent (agent2 and agent3)
    for (agent_id, gem_numbers) in agent_goals
        println("  Processing $agent_id with goals: $gem_numbers")
        
        # Create filtered problem for this agent (following run_experiment_exp3.jl pattern)
        domain_agent = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
        temp_path = joinpath(PROBLEM_DIR, ".temp_$(agent_id)_$(stimulus_id).txt")
        
        # Create filtered ASCII content (remove other agents)
        filtered_ascii = filter_ascii_agents(ascii_content, Symbol(agent_id))
        write(temp_path, filtered_ascii)
        
        # Load the filtered problem
        problem_agent = load_ascii_problem(temp_path)
        state_agent = initstate(domain_agent, problem_agent)
        domain_agent, state_agent = PDDL.compiled(domain_agent, problem_agent)
        
        println("    Created agent-specific problem for $agent_id")
        
        # Create planner for this agent
        heuristic = GoalManhattan()
        planner = AStarPlanner(heuristic)
        
        # Create goals for each gem number
        for gem_num in gem_numbers
            goal = PDDL.parse_pddl("(has $agent_id gem$gem_num)")
            
            println("    Planning for goal: $goal")
            plan = planner(domain_agent, state_agent, goal)
            
            # Create unique identifier for this plan
            plan_id = "$(stimulus_id)_$(agent_id)_gem$(gem_num)_plan"
            plan_dict[plan_id] = collect(plan)
            
            println("    Generated plan with $(length(collect(plan))) actions")
        end
        
        # Clean up temporary file
        rm(temp_path)
    end
end

# Write all plans to files
for (plan_id, plan_actions) in plan_dict
    # Create the filename using the new convention: {stimulus_id}_{agent_id}_gem{gem_num}_plan.pddl
    # Extract stimulus_id, agent_id, and gem_num from plan_id
    parts = split(plan_id, "_")
    stimulus_id = parts[1]
    agent_id = parts[2]
    gem_num = parts[3]
    filename = "$(stimulus_id)_$(agent_id)_$(gem_num)_plan.pddl"
    
    println("Writing plan: $filename")
    
    open(joinpath(PLAN_DIR, filename), "w") do file
        for action in plan_actions
            println(file, PDDL.write_pddl(action))
        end
    end
end

println("Plan generation completed!")
println("Generated $(length(plan_dict)) plans for exp3 scenarios.")
