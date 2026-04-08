using PDDL, SymbolicPlanners
using IterTools
using Distances

# Global cache for planner results
const PLANNER_CACHE = Dict{Tuple{Int, Int, Int, Int}, Vector{Term}}()
const CACHE_HITS = Ref(0)
const CACHE_MISSES = Ref(0)

"Clear the planner cache (call between maps/scenarios)"
function clear_planner_cache!()
    empty!(PLANNER_CACHE)
    CACHE_HITS[] = 0
    CACHE_MISSES[] = 0
end

"Get planner cache statistics"
function get_cache_stats()
    total = CACHE_HITS[] + CACHE_MISSES[]
    hit_rate = total > 0 ? CACHE_HITS[] / total : 0.0
    return (hits=CACHE_HITS[], misses=CACHE_MISSES[], hit_rate=hit_rate)
end

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



function calculate_plan_cost(plan::Vector{<:Term}, action_cost::AbstractDict{Symbol, <:Real})

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

function estimate_self_exploration_details(
    domain::Any,
    state::State,
    agent_goal::Any,
    wizards::Any,
    action_cost::AbstractDict{Symbol, <:Real},
)

    new_state = copy(state)

    planner = AStarPlanner(GoalManhattan())

    # Helper function to get plan with caching
    function get_cached_plan(agent_x::Int, agent_y::Int, goal_x::Int, goal_y::Int)
        cache_key = (agent_x, agent_y, goal_x, goal_y)
        if haskey(PLANNER_CACHE, cache_key)
            CACHE_HITS[] += 1
            return PLANNER_CACHE[cache_key]
        else
            CACHE_MISSES[] += 1
            goal = pddl"(and (= (xloc agent1) $goal_x) (= (yloc agent1) $goal_y))"
            plan = collect(planner(domain, new_state, goal))
            PLANNER_CACHE[cache_key] = plan
            return plan
        end
    end

    function best_interaction_plan(agent_x::Int, agent_y::Int, wizard_loc::Tuple{Int, Int})
        if abs(agent_x - wizard_loc[1]) + abs(agent_y - wizard_loc[2]) == 1
            return Term[], (agent_x, agent_y)
        end

        best_plan = nothing
        best_pos = nothing
        best_cost = Inf

        for (dx, dy) in ((0, -1), (0, 1), (-1, 0), (1, 0))
            adj_x = wizard_loc[1] + dx
            adj_y = wizard_loc[2] + dy
            try
                plan = get_cached_plan(agent_x, agent_y, adj_x, adj_y)
                isempty(plan) && (agent_x != adj_x || agent_y != adj_y) && continue
                plan_cost = calculate_plan_cost(plan, action_cost)
                if plan_cost < best_cost
                    best_plan = plan
                    best_pos = (adj_x, adj_y)
                    best_cost = plan_cost
                end
            catch
                continue
            end
        end

        best_plan === nothing && error("No reachable interaction position found for wizard at $wizard_loc")
        return best_plan, best_pos
    end

    wizard_targets = [(w, get_obj_loc(new_state, w)) for w in wizards if state[pddl"(iscolor $w blue)"]]

    total_cost = 0
    full_plan_strings = String[]

    for _ in 1:length(wizards)
        cost = Inf
        min_distance_wizard, min_distance_loc = wizard_targets[1]

        agent_x = new_state[pddl"(xloc agent1)"]
        agent_y = new_state[pddl"(yloc agent1)"]
        min_distance_pos = (agent_x, agent_y)
        min_distance_plan = Term[]

        for (wizard_obj, w_loc) in wizard_targets
            plan, interact_pos = best_interaction_plan(agent_x, agent_y, w_loc)
            plan_cost = calculate_plan_cost(plan, action_cost)

            if plan_cost < cost
                cost = plan_cost
                min_distance_wizard = wizard_obj
                min_distance_loc = w_loc
                min_distance_pos = interact_pos
                min_distance_plan = plan
            end
        end

        wizard_targets = filter!(target -> target[1] != min_distance_wizard, wizard_targets)

        append!(full_plan_strings, write_pddl.(min_distance_plan))
        push!(full_plan_strings, "(interact agent1 $(string(min_distance_wizard)))")
        total_cost += cost
        total_cost += action_cost[:interact]

        new_state[pddl"(xloc agent1)"] = min_distance_pos[1]
        new_state[pddl"(yloc agent1)"] = min_distance_pos[2]
    end

    goal_loc = get_obj_loc(new_state, agent_goal.args[2])

    x_loc = goal_loc[1]
    y_loc = goal_loc[2]

    agent_x = new_state[pddl"(xloc agent1)"]
    agent_y = new_state[pddl"(yloc agent1)"]

    final_plan = get_cached_plan(agent_x, agent_y, x_loc, y_loc)
    append!(full_plan_strings, write_pddl.(final_plan))

    if isempty(final_plan)
        total_cost += 0
    else
        total_cost += calculate_plan_cost(final_plan[1:end-1], action_cost)
    end

    return (cost = total_cost, plan = full_plan_strings)
end

function estimate_self_exploration_cost(
    domain::Any,
    state::State,
    agent_goal::Any,
    wizards::Any,
    action_cost::AbstractDict{Symbol, <:Real},
)
    return estimate_self_exploration_details(domain, state, agent_goal, wizards, action_cost).cost
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

    # Preserve semantic gem numbering by sorting in map order:
    # top-to-bottom first, then left-to-right within a row.
    gems = sort!(collect(PDDL.get_objects(state, :gem)), by=g -> begin
        x, y = get_obj_loc(state, g)
        (y, x, string(g))
    end)
    for (i, gem) in enumerate(gems)
        push!(goals, pddl"(has $agent_name $gem)")
        push!(goal_names, string(Char('A' + i - 1)))
    end
    return goals, goal_names
end
