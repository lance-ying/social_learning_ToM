using PDDL, SymbolicPlanners
using Gen, GenParticleFilters
using GenGPT3
using InversePlanning
using PDDLViz, GLMakie

# Register PDDL array theory
PDDL.Arrays.register!()

include("src/plan_io.jl")
include("src/utils.jl")
include("src/heuristics.jl")
include("paths_new.jl")
# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems")
# PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")
# STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements")

#--- Initial Setup ---#

# Load domain
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))


p_id = "s221_blue_exp"

goal = PDDL.parse_pddl("(has agent2 gem"*string(other_agent_goal[p_id]*")"))
domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
problem = load_problem(joinpath(PROBLEM_DIR, "$(p_id[1:4]).pddl"))
state = initstate(domain, problem)

domain, state = PDDL.compiled(domain, problem)

heuristic = GoalManhattan()
planner = AStarPlanner(heuristic)
plan = planner(domain, state, pddl"(has agent2 gem2)")

plan_dict = Dict()

for p_id in keys(other_agent_goal)
    if occursin("blue_naive",p_id)
        continue
    end

    # if !startswith(p_id,"s521")
    #     continue
    # end



    goal = PDDL.parse_pddl("(has agent2 gem"*string(other_agent_goal[p_id])*")")
    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
    problem = load_problem(joinpath(PROBLEM_DIR, "$(p_id[1:4]).pddl"))
    state = initstate(domain, problem)
    heuristic = GoalManhattan()
    planner = AStarPlanner(heuristic)

    print(p_id, goal)
    # print(goal)
    print("\n")
    plan = planner(domain, state, goal)

    plan_dict[p_id] = collect(plan)

end


for (k,v) in plan_dict

    if occursin("blue_naive",k)
        continue
    end
    if occursin("other_naive",k)
        continue
    end   

    open("/Users/lance/Documents/GitHub/ObserveMove/dataset/plans/$(k).pddl", "w") do file
        for action in v
            println(file, PDDL.write_pddl(action))
        end
    end
    # print("\n")
end


for (k,v) in paths

    if !occursin("blue_naive",k)
        continue
    end

    open("/Users/lance/Documents/GitHub/ObserveMove/dataset/plan_naive/$(k).pddl", "w") do file
        for action in v
            println(file, PDDL.write_pddl(action))
        end
    end
    # print("\n")
end



# Load problem
p_id = "s221_blue_exp"
map_id = p_id[1:4]
problem = load_problem(joinpath(PROBLEM_DIR, "$(map_id).pddl"))
plan = paths[p_id]
# Load plan
# plan, _, splitpoints = load_plan(joinpath(PLAN_DIR, "$(p_id).pddl"))

# Load belief statements
# statements = load_statements(joinpath(STATEMENT_DIR, "$(p_id).txt"))

# Initialize and compile reference state
state = initstate(domain, problem)

heuristic = GoalManhattan()
planner = AStarPlanner(heuristic)

# plan = planner(domain, state, @pddl("(has agent2 gem2)"))

canvas

canvas = PDDLViz.new_canvas(RENDERER)
anim = anim_plan!(canvas, RENDERER, domain, state, collect(plan); trail_length = 15, show_inventory=false)


domain, state = PDDL.compiled(domain, problem)