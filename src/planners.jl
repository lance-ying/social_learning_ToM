using PDDL, SymbolicPlanners, InversePlanning

"""
    NaivePlanner

A planner that implements naive wizard-visiting strategy:
- Visit blue wizards in order of distance until blue key is obtained
- Then plan optimally to the goal

This models the behavior of a "naive" agent who doesn't know which wizard
has the key and visits them greedily by distance.
"""
struct NaivePlanner <: SymbolicPlanners.Planner
    blue_wizards::Vector{Const}
    agent_name::Symbol
    fallback_planner::Any  # AStarPlanner instance
    domain::Any  # Store domain for dynamic action computation
    wizard_selection_mode::Symbol
end

# Constructor without domain (for backward compatibility)
NaivePlanner(blue_wizards::Vector{Const}, agent_name::Symbol, fallback_planner::Any) =
    NaivePlanner(blue_wizards, agent_name, fallback_planner, nothing, :manhattan)

# Constructor with domain and default selection mode (for backward compatibility)
NaivePlanner(blue_wizards::Vector{Const}, agent_name::Symbol, fallback_planner::Any, domain::Any) =
    NaivePlanner(blue_wizards, agent_name, fallback_planner, domain, :manhattan)

"""
    NaivePlannerSolution

Solution that can compute naive actions dynamically for any state.
Stores the planner reference to enable state-based action computation.
"""
mutable struct NaivePlannerSolution <: SymbolicPlanners.Solution
    status::Symbol
    plan::Vector{Term}
    trajectory::Vector{State}
    # Store references needed for dynamic action computation
    planner::NaivePlanner
    domain::Any
    spec::Any  # Goal specification
    # Cache: state hash -> action index for O(1) lookup
    state_action_cache::Dict{UInt64, Int}
    # Cache: state hash -> action values dict for O(1) get_action_values
    action_values_cache::Dict{UInt64, Dict{Term, Float64}}
end

Base.iterate(sol::NaivePlannerSolution) = iterate(sol.plan)
Base.iterate(sol::NaivePlannerSolution, state) = iterate(sol.plan, state)
Base.length(sol::NaivePlannerSolution) = length(sol.plan)
Base.collect(sol::NaivePlannerSolution) = sol.plan
Base.copy(sol::NaivePlannerSolution) = NaivePlannerSolution(
    sol.status, copy(sol.plan), copy(sol.trajectory),
    sol.planner, sol.domain, sol.spec, copy(sol.state_action_cache),
    copy(sol.action_values_cache)
)

# Helper: compute the next naive action from a given state
# 
# Naive agent behavior:
# 1. Go to CLOSEST wizard (from current position)
# 2. Interact with it
# 3. If got key → plan optimally to goal
# 4. If no key → go to NEXT CLOSEST wizard (excluding ones already visited)
#
# Key insight for particle filter:
# - If agent is adjacent to a wizard and DOESN'T have the key, they must have
#   ALREADY interacted with it (because naive agent always interacts when adjacent)
# - So we should skip that wizard and find the next closest
function compute_naive_action(
    domain::Domain, state::State, spec::Any,
    blue_wizards::Vector{Const}, agent_name::Symbol, fallback_planner::Any,
    wizard_selection_mode::Symbol
)
    # Find blue key
    blue_keys = [k for k in PDDL.get_objects(state, :key) if state[pddl"(iscolor $k blue)"]]
    if isempty(blue_keys)
        plan = collect(fallback_planner(domain, state, spec))
        return isempty(plan) ? missing : plan[1]
    end
    blue_key = blue_keys[1]
    
    # Check if agent already has the blue key
    if state[pddl"(has $agent_name $blue_key)"]
        # Already have key - plan optimally to goal
        plan = collect(fallback_planner(domain, state, spec))
        return isempty(plan) ? missing : plan[1]
    end
    
    # Check if optimal plan requires interacting with blue wizards
    plan_optimal = collect(fallback_planner(domain, state, spec))
    needs_blue_wizard = any(x -> x.name == :interact && x.args[end] in blue_wizards, plan_optimal)
    
    if !needs_blue_wizard || isempty(blue_wizards)
        return isempty(plan_optimal) ? missing : plan_optimal[1]
    end
    
    agent_const = Const(agent_name)

    # Find wizards that have been visited using the (visited ?a ?w) predicate
    visited_wizards = Set{Const}()
    for wizard in blue_wizards
        if state[Compound(:visited, [agent_const, wizard])]
            push!(visited_wizards, wizard)
        end
    end
    
    closest_wizard, plan_to_wizard, is_adjacent = choose_next_wizard_naive(
        domain, state, blue_wizards, visited_wizards, agent_name, fallback_planner, wizard_selection_mode
    )
    
    if closest_wizard === nothing
        # All wizards visited - shouldn't happen, fall back to optimal
        return isempty(plan_optimal) ? missing : plan_optimal[1]
    end
    
    if is_adjacent
        # Interact with the wizard
        return PDDL.parse_pddl("(interact $agent_name $closest_wizard)")
    else
        return isempty(plan_to_wizard) ? missing : plan_to_wizard[1]
    end
end

# State-based get_action - compute naive action dynamically
function SymbolicPlanners.get_action(sol::NaivePlannerSolution, state::State)
    # FIRST: Try hash cache lookup (O(1))
    state_hash = hash(state)
    if haskey(sol.state_action_cache, state_hash)
        idx = sol.state_action_cache[state_hash]
        if idx <= length(sol.plan)
            return sol.plan[idx]
        end
    end

    # SECOND: If state not in cache and we have domain, compute dynamically (slow path)
    if sol.domain !== nothing
        return compute_naive_action(
            sol.domain, state, sol.spec,
            sol.planner.blue_wizards, sol.planner.agent_name, sol.planner.fallback_planner,
            sol.planner.wizard_selection_mode
        )
    end

    return missing
end

# Time-based get_action
SymbolicPlanners.get_action(sol::NaivePlannerSolution, t::Int, state::State) = 
    SymbolicPlanners.get_action(sol, state)

# get_action_prob - probability of taking an action
# Returns softmax probability based on Q-values (consistent with Boltzmann)
function SymbolicPlanners.get_action_prob(sol::NaivePlannerSolution, state::State, act::Term)
    action_values = SymbolicPlanners.get_action_values(sol, state)
    if isempty(action_values)
        return 0.0
    end
    
    # Check if action is available
    if !haskey(action_values, act)
        return 0.0
    end
    
    # Compute softmax probability (temperature=1.0 for base solution)
    max_val = maximum(values(action_values))
    exp_vals = Dict(a => exp(v - max_val) for (a, v) in action_values)
    total = sum(values(exp_vals))
    
    return exp_vals[act] / total
end

# rand_action - sample an action (deterministic)
function SymbolicPlanners.rand_action(sol::NaivePlannerSolution, state::State)
    return SymbolicPlanners.get_action(sol, state)
end

# get_action_values - return Q-values for ALL available actions
# This is critical for Boltzmann action selection to work properly
function SymbolicPlanners.get_action_values(sol::NaivePlannerSolution, state::State)
    # FIRST: Check cache for O(1) lookup
    state_hash = hash(state)
    if haskey(sol.action_values_cache, state_hash)
        return sol.action_values_cache[state_hash]
    end

    if sol.domain === nothing
        # Fallback if no domain
        action = SymbolicPlanners.get_action(sol, state)
        if ismissing(action)
            return Dict{Term, Float64}()
        end
        return Dict{Term, Float64}(action => 0.0)
    end

    # Get the naive action (best action according to naive strategy)
    naive_action = SymbolicPlanners.get_action(sol, state)

    # Get all available actions in this state
    available_actions = PDDL.available(sol.domain, state)

    # Assign Q-values: naive action gets 0.0 (best), others get -1.0 (worse)
    # This way Boltzmann softmax gives highest prob to naive action
    # but non-zero prob to other actions
    values = Dict{Term, Float64}()
    for act in available_actions
        if !ismissing(naive_action) && act == naive_action
            values[act] = 0.0  # Best action
        else
            values[act] = -1.0  # Suboptimal action
        end
    end

    # Cache the result
    sol.action_values_cache[state_hash] = values

    return values
end

# has_cached_action_values
SymbolicPlanners.has_cached_action_values(sol::NaivePlannerSolution, state::State) = 
    !ismissing(SymbolicPlanners.get_action(sol, state))

# InversePlanning interface - check if we can compute an action for this state
function InversePlanning.has_action(sol::NaivePlannerSolution, t::Int, state::State)
    action = SymbolicPlanners.get_action(sol, state)
    return !ismissing(action)
end

InversePlanning.get_action(sol::NaivePlannerSolution, t::Int, state::State) = 
    SymbolicPlanners.get_action(sol, state)

# Helper function to plan to an adjacent position of a wizard (never on top)
function plan_to_wizard_location_naive(
    domain::Domain, state::State, wizard_loc::Tuple{Int,Int},
    agent_name::Symbol, planner
)
    agent_loc = get_obj_loc(state, Const(agent_name))

    # Check if already adjacent (Manhattan distance = 1)
    agent_adjacent = (abs(agent_loc[1] - wizard_loc[1]) + abs(agent_loc[2] - wizard_loc[2]) == 1)

    if agent_adjacent
        return Term[]  # Already adjacent, no movement needed
    end

    # Try to plan to each adjacent position, return on first success (fast)
    for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]
        adj_pos = (wizard_loc[1] + dx, wizard_loc[2] + dy)
        adj_goal = PDDL.parse_pddl("(and (= (xloc $agent_name) $(adj_pos[1])) (= (yloc $agent_name) $(adj_pos[2])))")
        try
            plan = collect(planner(domain, state, adj_goal))
            if !isempty(plan)
                return plan  # Return immediately on first success
            end
        catch
            continue
        end
    end

    return Term[]
end

function shortest_plan_to_wizard_location_naive(
    domain::Domain, state::State, wizard_loc::Tuple{Int,Int},
    agent_name::Symbol, planner
)
    agent_loc = get_obj_loc(state, Const(agent_name))
    agent_adjacent = (abs(agent_loc[1] - wizard_loc[1]) + abs(agent_loc[2] - wizard_loc[2]) == 1)
    agent_adjacent && return true, Term[]

    best_plan = nothing
    for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)]
        adj_pos = (wizard_loc[1] + dx, wizard_loc[2] + dy)
        adj_goal = PDDL.parse_pddl("(and (= (xloc $agent_name) $(adj_pos[1])) (= (yloc $agent_name) $(adj_pos[2])))")
        try
            plan = collect(planner(domain, state, adj_goal))
            isempty(plan) && continue
            if best_plan === nothing || length(plan) < length(best_plan)
                best_plan = plan
            end
        catch
            continue
        end
    end

    best_plan === nothing && return false, Term[]
    return true, best_plan
end

function choose_next_wizard_naive(
    domain::Domain, state::State, blue_wizards::Vector{Const}, visited_wizards::Set{Const},
    agent_name::Symbol, fallback_planner::Any, wizard_selection_mode::Symbol
)
    if wizard_selection_mode == :candidate_search
        closest_wizard = nothing
        chosen_plan = Term[]
        chosen_length = typemax(Int)
        for wizard in blue_wizards
            wizard in visited_wizards && continue
            wizard_loc = get_obj_loc(state, wizard)
            reachable, plan = shortest_plan_to_wizard_location_naive(domain, state, wizard_loc, agent_name, fallback_planner)
            reachable || continue
            plan_length = length(plan)
            if closest_wizard === nothing || plan_length < chosen_length ||
               (plan_length == chosen_length && string(wizard) < string(closest_wizard))
                closest_wizard = wizard
                chosen_plan = plan
                chosen_length = plan_length
            end
        end
        return closest_wizard, chosen_plan, closest_wizard !== nothing && chosen_length == 0
    elseif wizard_selection_mode == :manhattan
        agent_loc = get_obj_loc(state, Const(agent_name))
        closest_wizard = nothing
        closest_wizard_loc = nothing
        min_dist = Inf

        for wizard in blue_wizards
            wizard in visited_wizards && continue
            wizard_loc = get_obj_loc(state, wizard)
            dist = sum(abs.(agent_loc .- wizard_loc))
            if dist < min_dist
                min_dist = dist
                closest_wizard = wizard
                closest_wizard_loc = wizard_loc
            end
        end

        if closest_wizard === nothing
            return nothing, Term[], false
        end

        if min_dist == 1
            return closest_wizard, Term[], true
        end

        plan_to_wizard = plan_to_wizard_location_naive(domain, state, closest_wizard_loc, agent_name, fallback_planner)
        return closest_wizard, plan_to_wizard, false
    end

    error("Unsupported NaivePlanner wizard_selection_mode: $wizard_selection_mode")
end

"""
Generate full naive plan for trajectory generation (used for ground truth).
"""
function generate_naive_plan(
    domain::Domain, state::State, goal::Any, 
    blue_wizards::Vector{Const}, agent_name::Symbol, fallback_planner::Any,
    wizard_selection_mode::Symbol=:manhattan
)
    plan_optimal = collect(fallback_planner(domain, state, goal))
    needs_wizards = any(x -> x.name == :interact && x.args[end] in blue_wizards, plan_optimal)
    
    if needs_wizards && !isempty(blue_wizards)
        blue_keys = [k for k in PDDL.get_objects(state, :key) if state[pddl"(iscolor $k blue)"]]
        if isempty(blue_keys)
            return plan_optimal
        end
        blue_key = blue_keys[1]
        
        current_state = copy(state)
        full_naive_plan = Term[]
        agent_const = Const(agent_name)

        # Initialize visited_wizards from state's (visited) predicates
        visited_wizards = Set{Const}()
        for wizard in blue_wizards
            if current_state[Compound(:visited, [agent_const, wizard])]
                push!(visited_wizards, wizard)
            end
        end

        while !current_state[pddl"(has $agent_name $blue_key)"] && length(visited_wizards) < length(blue_wizards)
            closest_wizard, plan_to_wizard, _ = choose_next_wizard_naive(
                domain, current_state, blue_wizards, visited_wizards, agent_name, fallback_planner, wizard_selection_mode
            )
            
            if closest_wizard === nothing
                break
            end
            
            append!(full_naive_plan, plan_to_wizard)
            
            for action in plan_to_wizard
                current_state = PDDL.execute(domain, current_state, action)
            end
            
            interact_action = PDDL.parse_pddl("(interact $agent_name $closest_wizard)")
            push!(full_naive_plan, interact_action)
            current_state = PDDL.execute(domain, current_state, interact_action)
            
            push!(visited_wizards, closest_wizard)
        end
        
        if !current_state[pddl"(has $agent_name $blue_key)"]
            return plan_optimal
        end
        
        plan_to_goal = collect(fallback_planner(domain, current_state, goal))
        append!(full_naive_plan, plan_to_goal)
        
        return full_naive_plan
    else
        return plan_optimal
    end
end

"""
Execute the naive planner - returns a solution that can compute actions dynamically.
"""
function (planner::NaivePlanner)(domain::Domain, state::State, goal)
    plan = generate_naive_plan(
        domain, state, goal,
        planner.blue_wizards, planner.agent_name, planner.fallback_planner, planner.wizard_selection_mode
    )

    # Build trajectory and hash cache for O(1) action lookup
    trajectory = [state]
    state_action_cache = Dict{UInt64, Int}()
    state_action_cache[hash(state)] = 1  # First state maps to first action

    current_state = copy(state)
    for (i, action) in enumerate(plan)
        current_state = PDDL.execute(domain, current_state, action)
        push!(trajectory, current_state)
        # Map this state to the next action (i+1)
        if i < length(plan)
            state_action_cache[hash(current_state)] = i + 1
        end
    end

    # Create solution with domain reference for dynamic action computation
    # Initialize empty action_values_cache (populated on demand)
    action_values_cache = Dict{UInt64, Dict{Term, Float64}}()
    return NaivePlannerSolution(:success, plan, trajectory, planner, domain, goal, state_action_cache, action_values_cache)
end

# refine! - regenerate plan from current state
function SymbolicPlanners.refine!(
    sol::NaivePlannerSolution, planner::NaivePlanner,
    domain::Domain, state::State, spec::SymbolicPlanners.Specification
)
    new_plan = generate_naive_plan(
        domain, state, spec,
        planner.blue_wizards, planner.agent_name, planner.fallback_planner, planner.wizard_selection_mode
    )

    empty!(sol.plan)
    append!(sol.plan, new_plan)

    empty!(sol.trajectory)
    push!(sol.trajectory, state)

    # Rebuild hash caches
    empty!(sol.state_action_cache)
    empty!(sol.action_values_cache)
    sol.state_action_cache[hash(state)] = 1

    current_state = copy(state)
    for (i, action) in enumerate(new_plan)
        current_state = PDDL.execute(domain, current_state, action)
        push!(sol.trajectory, current_state)
        if i < length(new_plan)
            sol.state_action_cache[hash(current_state)] = i + 1
        end
    end

    # Update stored references
    sol.domain = domain
    sol.spec = spec

    return sol
end
