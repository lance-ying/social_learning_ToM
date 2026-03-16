using PDDL, SymbolicPlanners

import SymbolicPlanners:
    compute, precompute!, filter_available,
    get_goal_terms, set_goal_terms

include("utils.jl")

"""
    GoalManhattan

Custom relaxed distance heuristic to goal objects. Estimates the cost of 
collecting all goal objects by computing the distance between all goal objects
and the agent, then returning the minimum distance plus the number of remaining
goals to satisfy.
"""
struct GoalManhattan <: Heuristic end

function compute(heuristic::GoalManhattan,
                 domain::Domain, state::State, spec::Specification)
    # Count number of remaining goals to satisfy
    goal_count = GoalCountHeuristic()(domain, state, spec)
    # Determine goal objects to collect
    goals = get_goal_terms(spec)
    isempty(goals) && return goal_count
    # Compute minimum distance to goal objects
    min_dist = minimum(goals) do g
        g.name != :has && return 0.0f0
        state[g] && return 0.0f0
        agent, obj = g.args[1], g.args[2]
        agent_loc = get_obj_loc(state, agent)
        obj_loc = get_obj_loc(state, obj)
        return Float32(sum(abs.(agent_loc .- obj_loc)))
    end
    return min_dist + goal_count
end

"""
    GoalLandmarkMST

Admissible heuristic for the doors-keys-gems domain that uses:
- obstacle-aware shortest-path distances on the static wall grid
- landmark cells for required pickups/interactions
- an MST lower bound over remaining landmarks

Doors are treated as open in the relaxed grid, so this remains a lower bound on
the true action count used by `AStarPlanner`.
"""
struct GoalLandmarkMST <: Heuristic end

mutable struct RelaxedGridCache
    passable::BitMatrix
    dist_cache::Dict{Tuple{Int, Int}, Matrix{Int}}
end

const RELAXED_GRID_CACHES = Dict{Tuple{UInt64, Tuple{Int, Int}}, RelaxedGridCache}()

function relaxed_grid_cache(state::State)
    walls = state[pddl"(walls)"]
    cache_key = (UInt64(objectid(walls)), size(walls))
    return get!(RELAXED_GRID_CACHES, cache_key) do
        passable = BitMatrix(.!Matrix{Bool}(walls))
        RelaxedGridCache(passable, Dict{Tuple{Int, Int}, Matrix{Int}}())
    end
end

function in_relaxed_bounds(cache::RelaxedGridCache, cell::Tuple{Int, Int})
    x, y = cell
    height, width = size(cache.passable)
    return 1 <= x <= width && 1 <= y <= height
end

function normalize_landmark_cells(
    cache::RelaxedGridCache, cells::AbstractVector{Tuple{Int, Int}}
)
    seen = Set{Tuple{Int, Int}}()
    normalized = Tuple{Int, Int}[]
    for cell in cells
        in_relaxed_bounds(cache, cell) || continue
        x, y = cell
        cache.passable[y, x] || continue
        cell in seen && continue
        push!(seen, cell)
        push!(normalized, cell)
    end
    return normalized
end

function relaxed_distances_from!(
    cache::RelaxedGridCache, source::Tuple{Int, Int}
)
    return get!(cache.dist_cache, source) do
        height, width = size(cache.passable)
        dist = fill(typemax(Int), height, width)
        in_relaxed_bounds(cache, source) || return dist
        sx, sy = source
        cache.passable[sy, sx] || return dist

        queue = Vector{Tuple{Int, Int}}()
        push!(queue, source)
        dist[sy, sx] = 0
        head = 1

        while head <= length(queue)
            x, y = queue[head]
            head += 1
            base_dist = dist[y, x]

            for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                nx, ny = x + dx, y + dy
                1 <= nx <= width || continue
                1 <= ny <= height || continue
                cache.passable[ny, nx] || continue
                next_dist = base_dist + 1
                if next_dist < dist[ny, nx]
                    dist[ny, nx] = next_dist
                    push!(queue, (nx, ny))
                end
            end
        end

        dist
    end
end

function relaxed_min_distance(
    cache::RelaxedGridCache,
    from_cells::AbstractVector{Tuple{Int, Int}},
    to_cells::AbstractVector{Tuple{Int, Int}},
)
    best = typemax(Int)
    for source in from_cells
        dist = relaxed_distances_from!(cache, source)
        for (x, y) in to_cells
            best = min(best, dist[y, x])
        end
    end
    return best
end

function interaction_landmark_cells(state::State, wizard::Const)
    x, y = get_obj_loc(state, wizard)
    return Tuple{Int, Int}[
        (x, y),
        (x - 1, y),
        (x + 1, y),
        (x, y - 1),
        (x, y + 1),
    ]
end

function holding_wizard(state::State, item::Const)
    for wizard in PDDL.get_objects(state, :wizard)
        if state[pddl"(hold $wizard $item)"]
            return wizard
        end
    end
    return nothing
end

function landmark_for_goal_term(state::State, goal_term::Term)
    goal_term.name == :has || return nothing
    state[goal_term] && return nothing

    agent, obj = goal_term.args[1], goal_term.args[2]
    obj_type = PDDL.get_objtype(state, obj)

    if obj_type in (:gem, :key)
        wizard = holding_wizard(state, obj)
        if wizard !== nothing
            return (
                agent = agent,
                cells = interaction_landmark_cells(state, wizard),
                action_lb = 1,
            )
        end
        return (
            agent = agent,
            cells = Tuple{Int, Int}[get_obj_loc(state, obj; check_has=true)],
            action_lb = 1,
        )
    end

    return nothing
end

function landmark_mst_lower_bound(
    cache::RelaxedGridCache,
    agent_loc::Tuple{Int, Int},
    landmark_cells::Vector{Vector{Tuple{Int, Int}}},
)
    isempty(landmark_cells) && return 0

    agent_start = Tuple{Int, Int}[agent_loc]
    start_lb = minimum(relaxed_min_distance(cache, agent_start, cells) for cells in landmark_cells)
    start_lb == typemax(Int) && return typemax(Int)
    length(landmark_cells) == 1 && return start_lb

    n = length(landmark_cells)
    pairwise = fill(typemax(Int), n, n)
    for i in 1:n
        pairwise[i, i] = 0
        for j in i+1:n
            d = relaxed_min_distance(cache, landmark_cells[i], landmark_cells[j])
            pairwise[i, j] = d
            pairwise[j, i] = d
        end
    end

    in_tree = falses(n)
    best_edge = fill(typemax(Int), n)
    in_tree[1] = true
    for j in 2:n
        best_edge[j] = pairwise[1, j]
    end

    mst_cost = 0
    for _ in 2:n
        next_idx = 0
        next_cost = typemax(Int)
        for j in 1:n
            in_tree[j] && continue
            if best_edge[j] < next_cost
                next_idx = j
                next_cost = best_edge[j]
            end
        end
        next_idx == 0 && return typemax(Int)
        mst_cost += next_cost
        in_tree[next_idx] = true
        for j in 1:n
            in_tree[j] && continue
            best_edge[j] = min(best_edge[j], pairwise[next_idx, j])
        end
    end

    return start_lb + mst_cost
end

function compute(heuristic::GoalLandmarkMST,
                 domain::Domain, state::State, spec::Specification)
    goal_count = GoalCountHeuristic()(domain, state, spec)
    goals = get_goal_terms(spec)
    isempty(goals) && return goal_count

    cache = relaxed_grid_cache(state)
    supported_landmarks = Vector{Vector{Tuple{Int, Int}}}()
    mandatory_action_lb = 0
    unsupported_goal_lb = 0
    goal_agent = nothing

    for goal_term in goals
        state[goal_term] && continue
        landmark = landmark_for_goal_term(state, goal_term)
        if landmark === nothing
            unsupported_goal_lb += 1
            continue
        end

        if isnothing(goal_agent)
            goal_agent = landmark.agent
        elseif goal_agent != landmark.agent
            return compute(GoalManhattan(), domain, state, spec)
        end

        cells = normalize_landmark_cells(cache, landmark.cells)
        isempty(cells) && return compute(GoalManhattan(), domain, state, spec)
        push!(supported_landmarks, cells)
        mandatory_action_lb += landmark.action_lb
    end

    isempty(supported_landmarks) && return goal_count
    isnothing(goal_agent) && return goal_count

    agent_loc = get_obj_loc(state, goal_agent)
    move_lb = landmark_mst_lower_bound(cache, agent_loc, supported_landmarks)
    move_lb == typemax(Int) && return compute(GoalManhattan(), domain, state, spec)

    # `AStarPlanner` uses unit action costs, so the lower bound here is in action
    # count rather than the external weighted costs used in downstream analyses.
    return Float32(max(goal_count, move_lb + mandatory_action_lb + unsupported_goal_lb))
end
