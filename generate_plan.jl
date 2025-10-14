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
# include("paths_new.jl")
# Define directory paths
PROBLEM_DIR = joinpath(@__DIR__, "dataset", "problems_exp2")
# PLAN_DIR = joinpath(@__DIR__, "dataset", "plans")
# STATEMENT_DIR = joinpath(@__DIR__, "dataset", "statements")

#--- Initial Setup ---#

# Load domain
# domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
action_cost = Dict(:move => 2, :interact => 4, :observe => 1.0)

problem_files = filter(f -> endswith(f, ".pddl"), readdir(joinpath(@__DIR__, "dataset","problems_exp2")))



metadata_path = joinpath(PROBLEM_DIR, "metadata.json")
metadata = JSON.parsefile(metadata_path)
for (k, v) in metadata
    for (i, goal_str) in enumerate(v)
        filename = "$(k)_$(i)_plan.pddl"
        goal = PDDL.parse_pddl("(has agent2 gem$(goal_str))")
        println(filename, ": ", goal)

        problem_name = "$(k).pddl"


        domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

        problem = load_problem(joinpath(PROBLEM_DIR, problem_name))
        state = initstate(domain, problem)

        # domain, state = PDDL.compiled(domain, problem)
        
        heuristic = GoalManhattan()
        planner = AStarPlanner(heuristic)
        plan = planner(domain, state, goal)
        open(PROBLEM_DIR*filename, "w") do file
            for action in plan
                println(file, PDDL.write_pddl(action))
            end
        end


        # You can add code here to use `filename` and `goal` as needed
    end
end

for problem_name in problem_files
    println("Now processing: ", problem_name)
    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

    problem = load_problem(joinpath(PROBLEM_DIR, problem_name))
    state = initstate(domain, problem)

    # domain, state = PDDL.compiled(domain, problem)
    
    heuristic = GoalManhattan()
    planner = AStarPlanner(heuristic)
    plan = planner(domain, state, goal)
    p_id = splitext(problem_name)[1]
    open(PROBLEM_DIR*"/$(p_id)_plan.pddl", "w") do file
        for action in plan
            println(file, PDDL.write_pddl(action))
        end
    end
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

for p_id in keys(other_agent_goal)
    if !occursin("blue_naive",p_id)
        continue
    end

    # if !startswith(p_id,"s521")
    #     continue
    # end
    println(p_id)


    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))

    goal = PDDL.parse_pddl("(has agent2 gem"*string(other_agent_goal[p_id])*")")
    domain = load_domain(joinpath(@__DIR__, "dataset", "domain.pddl"))
    problem = load_problem(joinpath(PROBLEM_DIR, "$(p_id[1:4]).pddl"))
    state = initstate(domain, problem)

    domain, state = PDDL.compiled(domain, problem)

    heuristic = GoalManhattan()
    planner = AStarPlanner(heuristic)
    # plan = planner(domain, state, pddl"(has agent2 gem2)")


    wizards = [w for w in PDDL.get_objects(state, :wizard) if state[pddl"(iscolor $w blue)"]]
    # wizard_candicates = blue_wizards
    new_state =copy(state)


    action_cost = Dict(:move => 2, :interact => 4, :observe => 1.0)
    naive_plan = []

    wizard_locs = [get_obj_loc(new_state, w) for w in wizards if new_state[pddl"(iscolor $w blue)"]]

    keys = [w for w in PDDL.get_objects(new_state, :key) if new_state[pddl"(iscolor $w blue)"]]
    print(keys)
    key_loc = get_obj_loc(new_state, keys[1])

    for i in 1:length(wizards)
        cost = Inf

        # print(wizard_locs)

        min_distance_loc = wizard_locs[1]
        best_plan = []

        for w_loc in wizard_locs

            x_loc = w_loc[1]
            y_loc = w_loc[2]
            goal = pddl"(and (= (xloc agent2) $x_loc) (= (yloc agent2) $y_loc))"
            plan = planner(domain, new_state, goal)

            plan_cost = calculate_plan_cost(collect(plan), action_cost)

            # println(plan_cost)

            if plan_cost < cost
                cost = plan_cost
                min_distance_loc = w_loc
                best_plan = collect(plan)
                # print(best_plan)
            end
        end

    

        # print(min_distance_loc)

        # if min_distance_loc in wizard_locs
        wizard_locs = filter!(loc -> ((loc[1] != min_distance_loc[1]) || (loc[2] != min_distance_loc[2])), wizard_locs)

        # total_cost += cost
        # total_cost += action_cost[:interact]
        # total_cost -= 2*action_cost[:move]

        new_state[pddl"(xloc agent2)"] = min_distance_loc[1]
        new_state[pddl"(yloc agent2)"] = min_distance_loc[2]

        # print(min_distance_loc)
        if length(naive_plan) == 0
            naive_plan = best_plan[1:end-1]
        else
            naive_plan = [naive_plan; best_plan[2:end-1]]
        end

        if min_distance_loc == key_loc
            break
        end
    end

    plan = planner(domain, new_state, goal)
    naive_plan = [naive_plan; collect(plan)[2:end-1]]
    plan_dict[p_id] =naive_plan
end
