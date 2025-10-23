using PDDL, SymbolicPlanners
using IterTools
using Distances

"Returns the color of an object."
function get_obj_color(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        if state[Compound(:iscolor, Term[obj, color])]
            return color
        end
    end
    return Const(:none)
end

"Returns the location of an object."
function get_obj_loc(state::State, obj::Const; check_has::Bool=false)
    x = state[Compound(:xloc, Term[obj])]
    y = state[Compound(:yloc, Term[obj])]
    # Check if object is held by an agent, and return agent's location if so
    if check_has && PDDL.get_objtype(state, obj) in (:gem, :key)
        agents = (PDDL.get_objects(state, :human)...,
                  PDDL.get_objects(state, :robot)...)
        for agent in agents
            if state[Compound(:has, Term[agent, obj])]
                x, y = get_obj_loc(state, agent)
                break
            end
        end
    end
    return (x, y)
end

"Sets the location of an object."
function set_obj_loc!(state::State, obj::Const, loc::Tuple{Int,Int})
    state[pddl"(xloc $obj)"] = loc[1]
    state[pddl"(yloc $obj)"] = loc[2]
    return loc
end

"Removes the color of an object."
function remove_color!(state::State, obj::Const)
    for color in PDDL.get_objects(state, :color)
        state[pddl"(iscolor $obj $color)"] = false
    end
    return state
end

"Sets the color of an object."
function set_color!(state::State, obj::Const, color::Const)
    remove_color!(state, obj)
    state[pddl"(iscolor $obj $color)"] = true
    return color
end

"Empties a box of all keys."
function assign!(state::State, wizard::Const)
    for key in PDDL.get_objects(state, :key)
        if state[pddl"(iscolor $key blue)"]
            wizard_loc = get_obj_loc(state, wizard)
            for w in PDDL.get_objects(state, :wizard)
                if w != wizard
                    state[pddl"(hold $w $key)"] = false
                end
            end
            state[pddl"(hold $wizard $key)"] = true
            set_obj_loc!(state, key, wizard_loc)
        end
    end
    return state
end

"Places a key in a box."
function place_key_in_box!(state::State, key::Const, box::Const)
    box_loc = get_obj_loc(state, box)
    set_obj_loc!(state, key, box_loc)
    state[pddl"(inside $key $box)"] = true
    state[pddl"(hidden $key)"] = true
    state[pddl"(offgrid $key)"] = false
    return state
end

"Extracts keys and boxes relevant to a goal or plan."
function extract_relevant_keys_and_boxes(
    domain::Domain, state::State, plan::AbstractVector{<:Term}
)
    plan_keys = [act.args[2] for act in plan if act.name == :pickup]
    filter!(k -> PDDL.get_objtype(state, k) == :key, plan_keys)
    filter!(k -> state[pddl"(hidden $k)"], plan_keys)
    key_colors = [get_obj_color(state, k) for k in plan_keys]
    plan_boxes = map(plan_keys) do k
        for b in PDDL.get_objects(state, :box)
            if state[pddl"(inside $k $b)"]
                return b
            end
        end
        error("Could not find box for key $k")
    end
    return key_colors, plan_keys, plan_boxes
end

function extract_relevant_keys_and_boxes(
    domain::Domain, state::State, goal;
    planner = AStarPlanner(GoalCountHeuristic())
)
    sol = planner(domain, state, goal)
    plan = collect(sol)
    return extract_relevant_keys_and_boxes(domain, state, plan)
end



function calculate_plan_cost(plan:: Vector{<:Term}, action_cost::Dict{Symbol, Real})

    cost = 0

    for act in plan
        if act.name == :interact
            cost += action_cost[:interact]
        elseif act.name == :observe
            cost += action_cost[:observe]
        else
            cost += action_cost[:move]
        end
    end
    return cost
    
end

function estimate_self_exploration_cost(domain:: Any, state:: State, agent_goal::Any, wizards::Any, action_cost::Dict{Symbol, Real})

    new_state = copy(state)

    planner = AStarPlanner(GoalManhattan())


    # Extract wizard locations
    # print(state[pddl"(iscolor wizard1 blue)"])
    wizard_locs = [get_obj_loc(new_state, w) for w in wizards if state[pddl"(iscolor $w blue)"]]

    # Compute self-exploration cost
    cost = 0
    total_cost = 0

    # print(wizard_locs)

    for i in 1:length(wizards)
        cost = Inf
        # print(wizard_locs)
        min_distance_loc = wizard_locs[1]
        for w_loc in wizard_locs

            x_loc = w_loc[1]
            y_loc = w_loc[2]
            goal = pddl"(and (= (xloc agent1) $x_loc) (= (yloc agent1) $y_loc))"
            plan = planner(domain, new_state, goal)

            # print(collect(plan))

            plan_cost = calculate_plan_cost(collect(plan), action_cost)

            if plan_cost < cost
                cost = plan_cost
                min_distance_loc = w_loc
            end
        end

        # print(wizard_locs)

        # print(min_distance_loc)

        # if min_distance_loc in wizard_locs
        wizard_locs = filter!(loc -> ((loc[1] != min_distance_loc[1]) || (loc[2] != min_distance_loc[2])), wizard_locs)

        total_cost += cost
        total_cost += action_cost[:interact]
        # total_cost -= 2*action_cost[:move]

        new_state[pddl"(xloc agent1)"] = min_distance_loc[1]
        new_state[pddl"(yloc agent1)"] = min_distance_loc[2]

        # print(new_state[pddl"(xloc agent1)"], " ")
        # print(new_state[pddl"(yloc agent1)"])
        # print("\n")

        # counter+=1
        # new_state[]
    end

    goal_loc = get_obj_loc(new_state, agent_goal.args[2])

    x_loc = goal_loc[1]
    y_loc = goal_loc[2]

    goal = pddl"(and (= (xloc agent1) $x_loc) (= (yloc agent1) $y_loc))"

    plan = planner(domain, new_state, goal)

    # print(collect(plan))

    plan_cost = calculate_plan_cost(collect(plan), action_cost)

    # print(agent_goal)
    # final_plan = planner(domain, new_state, agent_goal)
    # print(collect(final_plan))
    # print(calculate_plan_cost(collect(final_plan), action_cost))
    total_cost += plan_cost
    total_cost -= action_cost[:move] *(2* length(wizards)-1)


    return total_cost
    
end

function check_equal_state(state1::State, state2::State)
    for wizard in PDDL.get_objects(state1, :wizard)
        for key in PDDL.get_objects(state1, :key)
            if state1[pddl"(hold $wizard $key)"]!= state2[pddl"(hold $wizard $key)"]
                return false
            end
        end
    end
    return true
end

function eval_state_dist(dist1, dist2)
    if euclidean(dist1, dist2) < 0.1
        return false
    end
    return true
end

function initialize_goals(state::State, agent_name::Symbol=:agent2)
    goals = []
    goal_names = []

    gems = PDDL.get_objects(state, :gem)
    for (i, gem) in enumerate(gems)
        push!(goals, pddl"(has $agent_name $gem)")
        push!(goal_names, string(Char('A' + i - 1)))
    end
    return goals, goal_names
end