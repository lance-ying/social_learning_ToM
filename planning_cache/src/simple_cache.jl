"""
Simple cache implementation with basic memoization and statistics.

This is the foundational cache type, similar to the original implementation
in `src/utils.jl` but with enhanced features.
"""

"""
    SimpleCache{K,V}

A simple cache with statistics tracking and optional eviction policy.

# Type Parameters
- `K`: Key type (e.g., Tuple{Int,Int,Int,Int} for pathfinding)
- `V`: Value type (e.g., Vector{Term} for plans)

# Fields
- `store::Dict{K,CacheEntry{V}}`: Internal storage
- `max_size::Int`: Maximum number of entries (0 = unlimited)
- `policy::EvictionPolicy`: Eviction policy to use
- `metrics::CacheMetrics`: Performance metrics

# Examples
```julia
# Create cache for pathfinding
cache = SimpleCache{Tuple{Int,Int,Int,Int}, Vector{Term}}()

# Store a plan
set_cache!(cache, (x1, y1, x2, y2), plan)

# Retrieve a plan
plan = get_cache(cache, (x1, y1, x2, y2))

# Check statistics
stats = get_stats(cache)
println("Hit rate: \$(stats.hit_rate)")
```
"""
mutable struct SimpleCache{K,V}
    store::Dict{K,CacheEntry{V}}
    max_size::Int
    policy::EvictionPolicy
    metrics::CacheMetrics

    function SimpleCache{K,V}(;
        max_size::Int=0,
        policy::EvictionPolicy=LRUPolicy()
    ) where {K,V}
        new{K,V}(
            Dict{K,CacheEntry{V}}(),
            max_size,
            policy,
            CacheMetrics()
        )
    end
end

"""
    set_cache!(cache::SimpleCache{K,V}, key::K, value::V)

Store a value in the cache.

If the cache is full, evicts an entry according to the eviction policy.
"""
function set_cache!(cache::SimpleCache{K,V}, key::K, value::V) where {K,V}
    # Check if we need to evict
    if cache.max_size > 0 && length(cache.store) >= cache.max_size && !haskey(cache.store, key)
        # Select and evict entry
        evict_key = select_eviction_candidate(cache.store, cache.policy)
        delete!(cache.store, evict_key)
        record_eviction!(cache.metrics)
    end

    # Store the entry
    cache.store[key] = CacheEntry{V}(value)
    return value
end

"""
    get_cache(cache::SimpleCache{K,V}, key::K) -> Union{V, Nothing}

Retrieve a value from the cache.

Returns `nothing` if the key is not found.
"""
function get_cache(cache::SimpleCache{K,V}, key::K) where {K,V}
    if haskey(cache.store, key)
        record_hit!(cache.metrics)
        entry = cache.store[key]
        update_access!(entry)
        return entry.value
    else
        record_miss!(cache.metrics)
        return nothing
    end
end

"""
    has_cache(cache::SimpleCache{K}, key::K) -> Bool

Check if a key exists in the cache.
"""
function has_cache(cache::SimpleCache{K}, key::K) where {K}
    return haskey(cache.store, key)
end

"""
    clear_cache!(cache::SimpleCache)

Clear all entries from the cache and reset metrics.
"""
function clear_cache!(cache::SimpleCache)
    empty!(cache.store)
    reset_metrics!(cache.metrics)
    return nothing
end

"""
    get_stats(cache::SimpleCache) -> CacheStats

Get performance statistics for the cache.
"""
function get_stats(cache::SimpleCache)
    return compute_stats(cache.metrics, length(cache.store), cache.max_size)
end

"""
    get_or_compute!(f::Function, cache::SimpleCache{K,V}, key::K) -> V

Get cached value or compute it using function `f`.

This is a convenience function for lazy evaluation with caching.

# Examples
```julia
plan = get_or_compute!(cache, (x1, y1, x2, y2)) do
    expensive_planning_computation(x1, y1, x2, y2)
end
```
"""
function get_or_compute!(f::Function, cache::SimpleCache{K,V}, key::K) where {K,V}
    cached = get_cache(cache, key)
    if cached !== nothing
        return cached
    else
        value = f()
        set_cache!(cache, key, value)
        return value
    end
end

# Pretty printing
function Base.show(io::IO, cache::SimpleCache{K,V}) where {K,V}
    stats = get_stats(cache)
    println(io, "SimpleCache{$K, $V}")
    println(io, "  Size: $(length(cache.store))")
    println(io, "  Max Size: $(cache.max_size == 0 ? "unlimited" : cache.max_size)")
    println(io, "  Policy: $(typeof(cache.policy))")
    println(io, "  Hit Rate: $(round(stats.hit_rate * 100, digits=2))%")
end
