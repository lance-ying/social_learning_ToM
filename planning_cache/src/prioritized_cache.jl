"""
Prioritized cache implementation inspired by Prioritized Experience Replay (PER).

Based on: Schaul, T., et al. (2015). "Prioritized Experience Replay." ICLR 2016.
"""

"""
    PrioritizedEntry{V}

Entry with priority for sampling.

# Fields
- `value::V`: Cached value
- `priority::Float64`: Priority score (higher = more important)
- `timestamp::Float64`: Creation timestamp
"""
mutable struct PrioritizedEntry{V}
    value::V
    priority::Float64
    timestamp::Float64

    function PrioritizedEntry{V}(value::V, priority::Float64) where V
        new{V}(value, priority, time())
    end
end

"""
    PrioritizedCache{K,V}

Cache with priority-based sampling, inspired by Prioritized Experience Replay.

Useful for:
- Reinforcement learning experience replay
- Caching important plans for reuse
- Prioritizing expensive computations

# Type Parameters
- `K`: Key type
- `V`: Value type

# Fields
- `store::Dict{K,PrioritizedEntry{V}}`: Internal storage
- `max_size::Int`: Maximum capacity
- `α::Float64`: Priority exponent (0 = uniform, 1 = full prioritization)
- `metrics::CacheMetrics`: Performance metrics

# Examples
```julia
# Create prioritized cache
cache = PrioritizedCache{State, Experience}(capacity=1000, α=0.6)

# Store with priority (e.g., TD-error)
set_cache!(cache, state, experience, priority=0.8)

# Sample batch with priorities
batch = sample_batch(cache, 32)
```
"""
mutable struct PrioritizedCache{K,V}
    store::Dict{K,PrioritizedEntry{V}}
    max_size::Int
    α::Float64  # Priority exponent
    metrics::CacheMetrics

    function PrioritizedCache{K,V}(;
        capacity::Int=10000,
        α::Float64=0.6
    ) where {K,V}
        new{K,V}(
            Dict{K,PrioritizedEntry{V}}(),
            capacity,
            α,
            CacheMetrics()
        )
    end
end

"""
    set_cache!(cache::PrioritizedCache{K,V}, key::K, value::V; priority::Float64=1.0)

Store a value with priority in the cache.

Higher priority values are more likely to be sampled.
"""
function set_cache!(cache::PrioritizedCache{K,V}, key::K, value::V; priority::Float64=1.0) where {K,V}
    # Evict if full
    if length(cache.store) >= cache.max_size && !haskey(cache.store, key)
        # Evict lowest priority entry
        evict_key = argmin(kv -> kv[2].priority, cache.store)[1]
        delete!(cache.store, evict_key)
        record_eviction!(cache.metrics)
    end

    cache.store[key] = PrioritizedEntry{V}(value, priority)
    return value
end

"""
    get_cache(cache::PrioritizedCache{K,V}, key::K) -> Union{V, Nothing}

Retrieve a value from the cache.
"""
function get_cache(cache::PrioritizedCache{K,V}, key::K) where {K,V}
    if haskey(cache.store, key)
        record_hit!(cache.metrics)
        return cache.store[key].value
    else
        record_miss!(cache.metrics)
        return nothing
    end
end

"""
    update_priority!(cache::PrioritizedCache{K}, key::K, priority::Float64)

Update the priority of an entry.

Useful for updating based on new information (e.g., updated TD-error).
"""
function update_priority!(cache::PrioritizedCache{K}, key::K, priority::Float64) where {K}
    if haskey(cache.store, key)
        cache.store[key].priority = priority
    end
end

"""
    sample_batch(cache::PrioritizedCache{K,V}, n::Int) -> Vector{Tuple{K,V}}

Sample n entries based on priorities.

Entries with higher priorities are more likely to be sampled.
Uses stochastic prioritization: P(i) ∝ p_i^α
"""
function sample_batch(cache::PrioritizedCache{K,V}, n::Int) where {K,V}
    if length(cache.store) == 0
        return Tuple{K,V}[]
    end

    n = min(n, length(cache.store))

    # Compute sampling probabilities
    keys = collect(keys(cache.store))
    priorities = [cache.store[k].priority^cache.α for k in keys]
    total_priority = sum(priorities)

    if total_priority == 0
        # Uniform sampling if all priorities are zero
        probs = fill(1.0 / length(keys), length(keys))
    else
        probs = priorities ./ total_priority
    end

    # Sample without replacement
    sampled_indices = sample_without_replacement(probs, n)
    sampled_keys = keys[sampled_indices]

    return [(k, cache.store[k].value) for k in sampled_keys]
end

"""
    sample_without_replacement(probs::Vector{Float64}, n::Int) -> Vector{Int}

Sample n indices without replacement based on probabilities.
"""
function sample_without_replacement(probs::Vector{Float64}, n::Int)
    indices = Int[]
    remaining_probs = copy(probs)

    for _ in 1:n
        # Sample one index
        r = rand() * sum(remaining_probs)
        cumsum_prob = 0.0
        selected_idx = 1

        for (i, p) in enumerate(remaining_probs)
            cumsum_prob += p
            if cumsum_prob >= r
                selected_idx = i
                break
            end
        end

        push!(indices, selected_idx)
        remaining_probs[selected_idx] = 0.0  # Remove from future sampling
    end

    return indices
end

"""
    has_cache(cache::PrioritizedCache{K}, key::K) -> Bool

Check if a key exists in the cache.
"""
function has_cache(cache::PrioritizedCache{K}, key::K) where {K}
    return haskey(cache.store, key)
end

"""
    clear_cache!(cache::PrioritizedCache)

Clear all entries and reset metrics.
"""
function clear_cache!(cache::PrioritizedCache)
    empty!(cache.store)
    reset_metrics!(cache.metrics)
end

"""
    get_stats(cache::PrioritizedCache) -> CacheStats

Get performance statistics.
"""
function get_stats(cache::PrioritizedCache)
    return compute_stats(cache.metrics, length(cache.store), cache.max_size)
end

# Pretty printing
function Base.show(io::IO, cache::PrioritizedCache{K,V}) where {K,V}
    stats = get_stats(cache)
    println(io, "PrioritizedCache{$K, $V}")
    println(io, "  Size: $(length(cache.store)) / $(cache.max_size)")
    println(io, "  α: $(cache.α)")
    println(io, "  Hit Rate: $(round(stats.hit_rate * 100, digits=2))%")
end
