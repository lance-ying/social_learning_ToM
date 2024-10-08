using PDDL
# using Gen, GenParticleFilters

include("utils.jl")

"Enumerate all possible beliefs about key locations in the initial state."
function enumerate_beliefs(
    state::State;
    boxes = PDDL.get_objects(state, :box),
    colors = PDDL.get_objects(state, :color),
    min_keys = 1,
    max_keys = min(2, length(boxes)),
    max_color_keys = ones(Int, length(colors)),
    discount = 1.0,
)
    belief_states = Vector{typeof(state)}()
    belief_probs = Float64[]
    n_colors, n_boxes = length(colors), length(boxes)
    max_keys = min(max_keys, sum(max_color_keys))
    # Extract keys that correspond to boxes
    keys = [k for k in PDDL.get_objects(state, :key)
            if state[pddl"(offgrid $k)"] || state[pddl"(hidden $k)"]]
    if length(keys) < length(boxes)
        error("Not enough keys to fill boxes.")
    elseif length(keys) > length(boxes)
        resize!(keys, length(boxes))
    end
    # Create base state with no boxes
    base_state = copy(state)
    for box in boxes
        empty_box!(base_state, box)
    end
    # Enumerate over number of keys
    for n_keys in min_keys:max_keys
        color_iter = Iterators.product((colors for _ in 1:n_keys)...)
        weight = discount ^ n_keys
        # Enumerate over subsets of n keys
        for key_idxs in IterTools.subsets(1:n_boxes, n_keys)
            # Enumerate over color assignments to keys within subset
            for key_colors in color_iter
                # Skip if too many of any one color
                color_counts = [sum(==(c), key_colors) for c in colors]
                if any(c > m for (c, m) in zip(color_counts, max_color_keys))
                    continue
                end
                # Create new state and set key colors and locations
                s = copy(base_state)
                for (idx, color) in zip(key_idxs, key_colors)
                    key = keys[idx]
                    set_color!(s, key, color)
                    place_key_in_box!(s, key, boxes[idx])
                end
                push!(belief_states, s)
                push!(belief_probs, weight)
            end
        end
    end
    belief_probs = belief_probs ./ sum(belief_probs)
    return belief_states, belief_probs
end

"Returns a string representation of the box contents in the state."
function box_contents_str(state::State)
    keys = PDDL.get_objects(state, :key)
    boxes = sort(collect(PDDL.get_objects(state, :box)), by=string)
    str = map(boxes) do box
        for k in keys
            state[pddl"(inside $k $box)"] || continue
            color = get_obj_color(state, k)
            return first(string(color))
        end
        return '_'
    end |> join
    return str
end

"""
    get_formula_probs(pf, domain, formula, t = 0;
                      normalize_prior = false,
                      belief_probs = nothing)

Evaluate the probability of the given formula in the given state.

# Keyword Arguments

- `normalize_prior`: If true, normalize the probability of the formula such that
    the prior probability of the formula being true is 0.5.
- `belief_probs`: Prior probabilities of initial belief states. Required for
    normalization when the belief prior is not uniform.
"""
function get_formula_probs(
    pf::ParticleFilterState, domain::Domain, formula::Term, t::Int = 0;
    normalize_prior::Bool = false,
    belief_probs = nothing
)
    # Extract inner formula from belief formula
    if formula.name == :believes
        formula = formula.args[2]
    end
    state_addr = t == 0 ? (:init => :env) : (:timestep => t => :env)
    if normalize_prior
        # Compute prior probability of statement being true
        if isnothing(belief_probs)
            formula_count = sum(get_traces(pf)) do trace
                state = trace[state_addr]
                return satisfy(domain, state, formula) ? 1 : 0
            end
            formula_prior = formula_count / length(get_traces(pf))
        else
            state_id_addr = :init => :env => :state_id
            formula_count = sum(get_traces(pf)) do trace
                state_id = trace[state_id_addr]
                state = trace[state_addr]
                return satisfy(domain, state, formula) ? belief_probs[state_id] : 0
            end
            total_count = sum(get_traces(pf)) do trace
                state_id = trace[state_id_addr]
                return belief_probs[state_id]
            end
            formula_prior = formula_count / total_count
        end
        # Adjust logprobs of each trace
        log_weights = copy(pf.log_weights)
        for (i, trace) in enumerate(get_traces(pf))
            state = trace[state_addr]
            if satisfy(domain, state, formula)
                log_weights[i] -= log(formula_prior) + log(0.5)
            else
                log_weights[i] -= log(1 - formula_prior) + log(0.5)
            end
        end
        # Compute normalized posterior probability of statement being true
        trace_probs = GenParticleFilters.softmax(log_weights)
        formula_prob = 0.0
        for (i, trace) in enumerate(get_traces(pf))
            state = trace[state_addr]
            formula_prob += satisfy(domain, state, formula) ? trace_probs[i] : 0
        end
        return formula_prob
    else
        # Compute posterior probability of statement being true
        return mean(pf, state_addr) do state
            satisfy(domain, state, formula) ? 1.0 : 0.0
        end
    end
end
