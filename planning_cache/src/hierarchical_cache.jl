"""
Hierarchical cache with hot/cold storage tiers.

Inspired by CPU cache hierarchies and tiered storage systems.
"""

"""
    HierarchicalCache{K,V}

Multi-tier cache with hot (fast) and cold (slow) storage.

Frequently accessed items stay in hot tier, while less frequently
accessed items move to cold tier.

# Type Parameters
- `K`: Key type
- `V`: Value type

# Fields
- `hot::Dict{K,CacheEntry{V}}`: Hot tier (fast access)
- `cold::Dict{K,CacheEntry{V}}`: Cold tier (larger capacity)
- `hot_size::Int`: Maximum hot tier size
- `cold_size::Int`: Maximum cold tier size
- `metrics::CacheMetrics`: Performance metrics

# Examples
```julia
# Create hierarchical cache
cache = HierarchicalCache{Key, Value}(
    hot_size=100,    # Small, fast tier
    cold_size=1000   # Larger, slower tier
)

# Use like normal cache
set_cache!(cache, key, value)
value = get_cache(cache, key)
```
"""
mutable struct HierarchicalCache{K,V}
    hot::Dict{K,CacheEntry{V}}
    cold::Dict{K,CacheEntry{V}}
    hot_size::Int
    cold_size::Int
    metrics::CacheMetrics

    function HierarchicalCache{K,V}(;
        hot_size::Int=100,
        cold_size::Int=1000
    ) where {K,V}
        new{K,V}(
            Dict{K,CacheEntry{V}}(),
            Dict{K,CacheEntry{V}}(),
            hot_size,
            cold_size,
            CacheMetrics()
        )
    end
end

"""
    promote_to_hot!(cache::HierarchicalCache{K,V}, key::K)

Move an entry from cold to hot tier.
"""
function promote_to_hot!(cache::HierarchicalCache{K,V}, key::K) where {K,V}
    if !haskey(cache.cold, key)
        return
    end

    # Get entry from cold
    entry = cache.cold[key]
    delete!(cache.cold, key)

    # Evict from hot if necessary
    if length(cache.hot) >= cache.hot_size
        # Evict least recently used from hot to cold
        lru_key = argmin(kv -> kv[2].access_time, cache.hot)[1]
        lru_entry = cache.hot[lru_key]
        delete!(cache.hot, lru_key)

        # Move to cold (evict from cold if necessary)
        if length(cache.cold) >= cache.cold_size
            cold_lru_key = argmin(kv -> kv[2].access_time, cache.cold)[1]
            delete!(cache.cold, cold_lru_key)
            record_eviction!(cache.metrics)
        end
        cache.cold[lru_key] = lru_entry
    end

    # Add to hot
    cache.hot[key] = entry
end

"""
    demote_to_cold!(cache::HierarchicalCache{K,V}, key::K)

Move an entry from hot to cold tier.
"""
function demote_to_cold!(cache::HierarchicalCache{K,V}, key::K) where {K,V}
    if !haskey(cache.hot, key)
        return
    end

    entry = cache.hot[key]
    delete!(cache.hot, key)

    # Evict from cold if necessary
    if length(cache.cold) >= cache.cold_size
        lru_key = argmin(kv -> kv[2].access_time, cache.cold)[1]
        delete!(cache.cold, lru_key)
        record_eviction!(cache.metrics)
    end

    cache.cold[key] = entry
end

"""
    set_cache!(cache::HierarchicalCache{K,V}, key::K, value::V)

Store a value in the hierarchical cache (starts in hot tier).
"""
function set_cache!(cache::HierarchicalCache{K,V}, key::K, value::V) where {K,V}
    # Remove from cold if present
    if haskey(cache.cold, key)
        delete!(cache.cold, key)
    end

    # Evict from hot if necessary
    if length(cache.hot) >= cache.hot_size && !haskey(cache.hot, key)
        # Move LRU from hot to cold
        lru_key = argmin(kv -> kv[2].access_time, cache.hot)[1]
        demote_to_cold!(cache, lru_key)
    end

    # Add to hot tier
    cache.hot[key] = CacheEntry{V}(value)
    return value
end

"""
    get_cache(cache::HierarchicalCache{K,V}, key::K) -> Union{V, Nothing}

Retrieve a value from the hierarchical cache.

If found in cold tier, promotes to hot tier.
"""
function get_cache(cache::HierarchicalCache{K,V}, key::K) where {K,V}
    # Check hot tier
    if haskey(cache.hot, key)
        record_hit!(cache.metrics)
        entry = cache.hot[key]
        update_access!(entry)
        return entry.value
    end

    # Check cold tier
    if haskey(cache.cold, key)
        record_hit!(cache.metrics)
        # Promote to hot tier
        promote_to_hot!(cache, key)
        entry = cache.hot[key]  # Now in hot tier
        update_access!(entry)
        return entry.value
    end

    # Not found
    record_miss!(cache.metrics)
    return nothing
end

"""
    has_cache(cache::HierarchicalCache{K}, key::K) -> Bool

Check if a key exists in either tier.
"""
function has_cache(cache::HierarchicalCache{K}, key::K) where {K}
    return haskey(cache.hot, key) || haskey(cache.cold, key)
end

"""
    clear_cache!(cache::HierarchicalCache)

Clear both tiers and reset metrics.
"""
function clear_cache!(cache::HierarchicalCache)
    empty!(cache.hot)
    empty!(cache.cold)
    reset_metrics!(cache.metrics)
end

"""
    get_stats(cache::HierarchicalCache) -> CacheStats

Get performance statistics for the hierarchical cache.
"""
function get_stats(cache::HierarchicalCache)
    total_size = length(cache.hot) + length(cache.cold)
    max_size = cache.hot_size + cache.cold_size
    return compute_stats(cache.metrics, total_size, max_size)
end

"""
    get_or_compute!(f::Function, cache::HierarchicalCache{K,V}, key::K) -> V

Get cached value or compute it using function `f`.
"""
function get_or_compute!(f::Function, cache::HierarchicalCache{K,V}, key::K) where {K,V}
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
function Base.show(io::IO, cache::HierarchicalCache{K,V}) where {K,V}
    stats = get_stats(cache)
    println(io, "HierarchicalCache{$K, $V}")
    println(io, "  Hot Tier: $(length(cache.hot)) / $(cache.hot_size)")
    println(io, "  Cold Tier: $(length(cache.cold)) / $(cache.cold_size)")
    println(io, "  Total: $(stats.size) / $(stats.max_size)")
    println(io, "  Hit Rate: $(round(stats.hit_rate * 100, digits=2))%")
end
