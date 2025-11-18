"""
Eviction policies for cache management.
"""

"""
    EvictionPolicy

Abstract type for eviction policies.
"""
abstract type EvictionPolicy end

"""
    LRUPolicy <: EvictionPolicy

Least Recently Used eviction policy.
"""
struct LRUPolicy <: EvictionPolicy end

"""
    LFUPolicy <: EvictionPolicy

Least Frequently Used eviction policy.
"""
struct LFUPolicy <: EvictionPolicy end

"""
    TTLPolicy <: EvictionPolicy

Time-To-Live eviction policy.

# Fields
- `ttl::Float64`: Time-to-live in seconds
"""
struct TTLPolicy <: EvictionPolicy
    ttl::Float64
end

"""
    FIFOPolicy <: EvictionPolicy

First-In-First-Out eviction policy.
"""
struct FIFOPolicy <: EvictionPolicy end

"""
    CacheEntry{V}

Entry in the cache with metadata for eviction policies.

# Fields
- `value::V`: Cached value
- `access_time::Float64`: Last access time
- `create_time::Float64`: Creation time
- `access_count::Int`: Number of accesses
"""
mutable struct CacheEntry{V}
    value::V
    access_time::Float64
    create_time::Float64
    access_count::Int

    function CacheEntry{V}(value::V) where V
        t = time()
        new{V}(value, t, t, 1)
    end
end

"""
    update_access!(entry::CacheEntry)

Update access metadata for an entry.
"""
function update_access!(entry::CacheEntry)
    entry.access_time = time()
    entry.access_count += 1
end

"""
    should_evict(entry::CacheEntry, policy::TTLPolicy) -> Bool

Check if entry should be evicted based on TTL.
"""
function should_evict(entry::CacheEntry, policy::TTLPolicy)
    return (time() - entry.create_time) > policy.ttl
end

"""
    select_eviction_candidate(entries::Dict, policy::LRUPolicy) -> K

Select entry to evict based on LRU policy.
"""
function select_eviction_candidate(entries::Dict{K,CacheEntry{V}}, policy::LRUPolicy) where {K,V}
    return argmin(kv -> kv[2].access_time, entries)[1]
end

"""
    select_eviction_candidate(entries::Dict, policy::LFUPolicy) -> K

Select entry to evict based on LFU policy.
"""
function select_eviction_candidate(entries::Dict{K,CacheEntry{V}}, policy::LFUPolicy) where {K,V}
    return argmin(kv -> kv[2].access_count, entries)[1]
end

"""
    select_eviction_candidate(entries::Dict, policy::FIFOPolicy) -> K

Select entry to evict based on FIFO policy.
"""
function select_eviction_candidate(entries::Dict{K,CacheEntry{V}}, policy::FIFOPolicy) where {K,V}
    return argmin(kv -> kv[2].create_time, entries)[1]
end
