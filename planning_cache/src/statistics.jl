"""
Statistics tracking for cache performance analysis.
"""

"""
    CacheStats

Statistics about cache performance.

# Fields
- `hits::Int`: Number of cache hits
- `misses::Int`: Number of cache misses
- `evictions::Int`: Number of entries evicted
- `total_queries::Int`: Total number of queries
- `hit_rate::Float64`: Ratio of hits to total queries
- `size::Int`: Current number of entries
- `max_size::Int`: Maximum capacity
"""
struct CacheStats
    hits::Int
    misses::Int
    evictions::Int
    total_queries::Int
    hit_rate::Float64
    size::Int
    max_size::Int
end

"""
    CacheMetrics

Mutable container for tracking cache metrics.
"""
mutable struct CacheMetrics
    hits::Int
    misses::Int
    evictions::Int

    CacheMetrics() = new(0, 0, 0)
end

"""
    record_hit!(metrics::CacheMetrics)

Record a cache hit.
"""
function record_hit!(metrics::CacheMetrics)
    metrics.hits += 1
end

"""
    record_miss!(metrics::CacheMetrics)

Record a cache miss.
"""
function record_miss!(metrics::CacheMetrics)
    metrics.misses += 1
end

"""
    record_eviction!(metrics::CacheMetrics)

Record an eviction event.
"""
function record_eviction!(metrics::CacheMetrics)
    metrics.evictions += 1
end

"""
    compute_stats(metrics::CacheMetrics, size::Int, max_size::Int) -> CacheStats

Compute statistics from metrics.
"""
function compute_stats(metrics::CacheMetrics, size::Int, max_size::Int)
    total = metrics.hits + metrics.misses
    hit_rate = total > 0 ? metrics.hits / total : 0.0

    return CacheStats(
        metrics.hits,
        metrics.misses,
        metrics.evictions,
        total,
        hit_rate,
        size,
        max_size
    )
end

"""
    reset_metrics!(metrics::CacheMetrics)

Reset all metrics to zero.
"""
function reset_metrics!(metrics::CacheMetrics)
    metrics.hits = 0
    metrics.misses = 0
    metrics.evictions = 0
end

# Pretty printing
function Base.show(io::IO, stats::CacheStats)
    println(io, "CacheStats:")
    println(io, "  Hits: $(stats.hits)")
    println(io, "  Misses: $(stats.misses)")
    println(io, "  Evictions: $(stats.evictions)")
    println(io, "  Total Queries: $(stats.total_queries)")
    println(io, "  Hit Rate: $(round(stats.hit_rate * 100, digits=2))%")
    println(io, "  Size: $(stats.size) / $(stats.max_size)")
end
