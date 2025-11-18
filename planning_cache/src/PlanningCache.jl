"""
# PlanningCache.jl

A modular caching system for planning algorithms and reinforcement learning.

Provides efficient memoization and experience replay capabilities inspired by:
- Prioritized Experience Replay (Schaul et al., 2015)
- Pathfinding caching (US Patent 6377887B1)
- Combined Experience Replay (Zhang & Sutton, 2017)

## Exports

### Cache Types
- `SimpleCache`: Basic memoization with statistics
- `PrioritizedCache`: Priority-based experience replay
- `DistributedCache`: Thread-safe caching for parallel tasks
- `HierarchicalCache`: Multi-tier caching system

### Core Functions
- `set_cache!`, `get_cache`, `has_cache`, `clear_cache!`
- `get_or_compute!`: Lazy evaluation with caching
- `get_stats`: Performance statistics

### Advanced Functions
- `sample_batch`: Sample entries (with priorities)
- `update_priority!`: Update entry priorities
- `evict_policy!`: Apply eviction policies
"""
module PlanningCache

using Statistics
using DataStructures

# Export cache types
export SimpleCache, PrioritizedCache, DistributedCache, HierarchicalCache

# Export core functions
export set_cache!, get_cache, has_cache, clear_cache!
export get_or_compute!, get_stats

# Export advanced functions
export sample_batch, update_priority!, evict_policy!

# Export statistics types
export CacheStats

# Include submodules
include("statistics.jl")
include("eviction_policies.jl")
include("simple_cache.jl")
include("prioritized_cache.jl")
include("distributed_cache.jl")
include("hierarchical_cache.jl")

end # module
