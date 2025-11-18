"""
Distributed cache implementation for parallel planning tasks.

Thread-safe caching using sharded locks to minimize contention.
"""

"""
    CacheShard{K,V}

A shard of the distributed cache with its own lock.

# Fields
- `store::Dict{K,CacheEntry{V}}`: Storage for this shard
- `lock::ReentrantLock`: Lock for thread-safe access
"""
mutable struct CacheShard{K,V}
    store::Dict{K,CacheEntry{V}}
    lock::ReentrantLock

    CacheShard{K,V}() where {K,V} = new{K,V}(Dict{K,CacheEntry{V}}(), ReentrantLock())
end

"""
    DistributedCache{K,V}

Thread-safe cache for parallel planning using sharded locks.

Useful for:
- Multi-threaded planning algorithms
- Parallel scenario exploration
- Concurrent agent simulations

# Type Parameters
- `K`: Key type (must be hashable)
- `V`: Value type

# Fields
- `shards::Vector{CacheShard{K,V}}`: Cache shards
- `num_shards::Int`: Number of shards
- `metrics::CacheMetrics`: Performance metrics
- `metrics_lock::ReentrantLock`: Lock for metrics updates

# Examples
```julia
# Create distributed cache
cache = DistributedCache{Tuple{Int,Int,Int,Int}, Vector{Term}}(num_shards=8)

# Use in parallel planning
Threads.@threads for scenario in scenarios
    plan = get_or_compute!(cache, scenario.key) do
        compute_plan(scenario)
    end
end
```
"""
mutable struct DistributedCache{K,V}
    shards::Vector{CacheShard{K,V}}
    num_shards::Int
    metrics::CacheMetrics
    metrics_lock::ReentrantLock

    function DistributedCache{K,V}(; num_shards::Int=8) where {K,V}
        shards = [CacheShard{K,V}() for _ in 1:num_shards]
        new{K,V}(shards, num_shards, CacheMetrics(), ReentrantLock())
    end
end

"""
    get_shard_index(cache::DistributedCache{K}, key::K) -> Int

Determine which shard a key belongs to.
"""
function get_shard_index(cache::DistributedCache{K}, key::K) where {K}
    return (hash(key) % cache.num_shards) + 1
end

"""
    set_cache!(cache::DistributedCache{K,V}, key::K, value::V)

Store a value in the distributed cache (thread-safe).
"""
function set_cache!(cache::DistributedCache{K,V}, key::K, value::V) where {K,V}
    shard_idx = get_shard_index(cache, key)
    shard = cache.shards[shard_idx]

    lock(shard.lock) do
        shard.store[key] = CacheEntry{V}(value)
    end

    return value
end

"""
    get_cache(cache::DistributedCache{K,V}, key::K) -> Union{V, Nothing}

Retrieve a value from the distributed cache (thread-safe).
"""
function get_cache(cache::DistributedCache{K,V}, key::K) where {K,V}
    shard_idx = get_shard_index(cache, key)
    shard = cache.shards[shard_idx]

    result = lock(shard.lock) do
        if haskey(shard.store, key)
            entry = shard.store[key]
            update_access!(entry)
            return entry.value
        else
            return nothing
        end
    end

    # Update metrics (locked separately to avoid holding shard lock)
    lock(cache.metrics_lock) do
        if result !== nothing
            record_hit!(cache.metrics)
        else
            record_miss!(cache.metrics)
        end
    end

    return result
end

"""
    has_cache(cache::DistributedCache{K}, key::K) -> Bool

Check if a key exists in the distributed cache (thread-safe).
"""
function has_cache(cache::DistributedCache{K}, key::K) where {K}
    shard_idx = get_shard_index(cache, key)
    shard = cache.shards[shard_idx]

    return lock(shard.lock) do
        haskey(shard.store, key)
    end
end

"""
    clear_cache!(cache::DistributedCache)

Clear all shards and reset metrics (thread-safe).
"""
function clear_cache!(cache::DistributedCache)
    for shard in cache.shards
        lock(shard.lock) do
            empty!(shard.store)
        end
    end

    lock(cache.metrics_lock) do
        reset_metrics!(cache.metrics)
    end
end

"""
    get_stats(cache::DistributedCache) -> CacheStats

Get performance statistics (thread-safe).
"""
function get_stats(cache::DistributedCache)
    total_size = sum(shard -> length(shard.store), cache.shards)

    return lock(cache.metrics_lock) do
        compute_stats(cache.metrics, total_size, 0)
    end
end

"""
    get_or_compute!(f::Function, cache::DistributedCache{K,V}, key::K) -> V

Get cached value or compute it using function `f` (thread-safe).
"""
function get_or_compute!(f::Function, cache::DistributedCache{K,V}, key::K) where {K,V}
    # Try to get from cache first
    cached = get_cache(cache, key)
    if cached !== nothing
        return cached
    end

    # Compute value
    value = f()

    # Store in cache
    set_cache!(cache, key, value)

    return value
end

# Pretty printing
function Base.show(io::IO, cache::DistributedCache{K,V}) where {K,V}
    stats = get_stats(cache)
    println(io, "DistributedCache{$K, $V}")
    println(io, "  Shards: $(cache.num_shards)")
    println(io, "  Total Size: $(stats.size)")
    println(io, "  Hit Rate: $(round(stats.hit_rate * 100, digits=2))%")

    # Show per-shard sizes
    shard_sizes = [length(shard.store) for shard in cache.shards]
    println(io, "  Shard sizes: $shard_sizes")
end
