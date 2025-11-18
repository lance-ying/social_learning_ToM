using PDDL
# using Gen, GenParticleFilters

include("utils.jl")

"Enumerate all possible beliefs about key locations in the initial state."
function enumerate_beliefs(
    state::State;
    wizards = collect(PDDL.get_objects(state, :wizard)),
    # colors = PDDL.get_objects(state, :color),
)
    print(wizards)
    belief_states = Vector{typeof(state)}()
    belief_probs = Float64[]
    belief_names = String[]
    # Create base state with no boxes
    belief_cnt = length(wizards)-1
    # wizard_str = [string(w) for w in wizards]

    for wizard in sort!(wizards, by = x -> string(x))
        # print(wizard)
        if state[pddl"(iscolor $wizard blue)"]
            push!(belief_names, string(wizard))
            base_state = copy(state)
            assign!(base_state, wizard)
            push!(belief_states, base_state)
            push!(belief_probs, 1.0 / belief_cnt)
            
        end
    end
    return belief_states, belief_probs, belief_names
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
