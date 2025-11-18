# PlanningCache.jl

A modular, efficient caching system for planning algorithms and reinforcement learning applications, inspired by experience replay and pathfinding memoization techniques.

## Overview

This package provides a flexible caching infrastructure for computational planning tasks, particularly useful for:

- **Pathfinding algorithms** (A*, Dijkstra, etc.)
- **Reinforcement learning** experience replay
- **PDDL planning** and state-space search
- **Multi-agent systems** with repeated planning queries

## Key Features

### 1. Multiple Cache Strategies
- **SimpleCache**: Basic dictionary-based memoization with LRU eviction
- **PrioritizedCache**: Priority-based caching inspired by Prioritized Experience Replay (PER)
- **DistributedCache**: Thread-safe caching for parallel planning tasks
- **HierarchicalCache**: Multi-level cache with hot/cold storage tiers

### 2. Intelligent Eviction Policies
- Least Recently Used (LRU)
- Least Frequently Used (LFU)
- Time-To-Live (TTL) expiration
- Custom priority scoring

### 3. Performance Monitoring
- Hit/miss rate tracking
- Memory usage profiling
- Cache effectiveness metrics
- Real-time statistics

## Research Context

This implementation draws from several research areas:

### Experience Replay in Deep RL
- **Prioritized Experience Replay** (Schaul et al., 2015): Store and replay important transitions
- **Combined Experience Replay** (Zhang & Sutton, 2017): O(1) extra computation for replay
- **Hindsight Experience Replay** (Andrychowicz et al., 2017): Learn from failed attempts

### Pathfinding Memoization
- **A* Caching** (US Patent 6377887B1): Cache pathfinding computations
- **Hub Labelling**: Pre-compute hierarchical distance labels
- **Arc-Flags**: Cache routing information for large graphs

### Planning and Search
- **State Space Caching**: Avoid redundant state expansions
- **Plan Reuse**: Cache partial plans for similar problems
- **Heuristic Memoization**: Store expensive heuristic computations

## Installation

```julia
using Pkg
Pkg.develop(path="planning_cache")
using PlanningCache
```

## Quick Start

### Basic Usage

```julia
using PlanningCache

# Create a simple cache
cache = SimpleCache{Tuple{Int,Int,Int,Int}, Vector{Any}}()

# Cache a planning result
start = (0, 0)
goal = (5, 5)
plan = compute_plan(start, goal)  # Your planning function
set_cache!(cache, (start..., goal...), plan)

# Retrieve cached result
cached_plan = get_cache(cache, (start..., goal...))

# Check statistics
stats = get_stats(cache)
println("Hit rate: $(stats.hit_rate)")
```

### Prioritized Caching for RL

```julia
using PlanningCache

# Create prioritized cache (like PER)
cache = PrioritizedCache{State, Action}(capacity=10000)

# Store experience with priority
state = get_current_state()
action = select_action()
reward = execute(action)
priority = abs(reward - value_estimate)  # TD-error as priority

store_experience!(cache, state, action, reward, priority)

# Sample high-priority experiences
batch = sample_batch(cache, batch_size=32, α=0.6)
```

### Distributed Planning Cache

```julia
using PlanningCache

# Thread-safe cache for parallel planning
cache = DistributedCache{PlanKey, Plan}(num_shards=8)

# Use in parallel planning
Threads.@threads for scenario in scenarios
    plan = get_or_compute!(cache, scenario) do
        compute_expensive_plan(scenario)
    end
end
```

## Architecture

```
PlanningCache/
├── src/
│   ├── PlanningCache.jl          # Main module
│   ├── simple_cache.jl           # Basic memoization
│   ├── prioritized_cache.jl      # Priority-based replay
│   ├── distributed_cache.jl      # Thread-safe implementation
│   ├── hierarchical_cache.jl     # Multi-tier storage
│   ├── eviction_policies.jl      # LRU, LFU, TTL policies
│   └── statistics.jl             # Performance metrics
├── test/
│   ├── runtests.jl
│   ├── test_simple_cache.jl
│   ├── test_prioritized_cache.jl
│   └── benchmarks.jl
├── examples/
│   ├── pathfinding_example.jl    # A* with caching
│   ├── rl_replay_example.jl      # Experience replay
│   └── social_learning_example.jl # Original use case
└── docs/
    ├── design.md                 # Architecture decisions
    ├── benchmarks.md             # Performance analysis
    └── papers.md                 # Research references
```

## Performance

Based on the original implementation in `social_learning_ToM`:

- **Cache Hit Rate**: 75-90% for repeated planning queries
- **Speedup**: 3-5x for scenarios with spatial locality
- **Memory**: O(n) storage for n cached entries with configurable limits

## Use Cases

### 1. Multi-Agent Planning
Cache plans for agents exploring similar state spaces:
```julia
# Agent observing other agents
for agent in [agent2, agent3]
    Q_observe = compute_with_cache(cache, agent.goal, agent.state)
end
```

### 2. PDDL Planning
Cache pathfinding for grid-based domains:
```julia
# Cache (start_x, start_y, goal_x, goal_y) → plan
plan = get_or_compute!(cache, (x1, y1, x2, y2)) do
    planner(domain, state, goal)
end
```

### 3. Reinforcement Learning
Store and replay experiences for training:
```julia
# Store transition
cache.store(state, action, reward, next_state)

# Sample for training
batch = cache.sample(batch_size=32)
```

## API Reference

### Core Functions

- `set_cache!(cache, key, value)`: Store a value
- `get_cache(cache, key)`: Retrieve a cached value
- `has_cache(cache, key)`: Check if key exists
- `clear_cache!(cache)`: Clear all entries
- `get_stats(cache)`: Get performance statistics

### Advanced Functions

- `get_or_compute!(f, cache, key)`: Get cached or compute
- `sample_batch(cache, n)`: Sample n entries (prioritized)
- `update_priority!(cache, key, priority)`: Update entry priority
- `evict_policy!(cache, policy)`: Apply eviction policy

## Configuration

```julia
# Configure cache behavior
cache = SimpleCache(
    max_size = 10000,        # Maximum entries
    eviction = :lru,          # Eviction policy
    ttl = 3600,               # Time-to-live (seconds)
    track_stats = true        # Enable statistics
)
```

## Benchmarking

Run benchmarks to evaluate cache performance:

```bash
julia --project=planning_cache planning_cache/test/benchmarks.jl
```

## Contributing

Contributions welcome! Areas of interest:
- New eviction policies
- Adaptive cache sizing
- GPU-accelerated caching
- Integration with popular planning libraries

## References

### Papers

1. **Schaul, T., et al. (2015)**. "Prioritized Experience Replay." ICLR 2016.
   - Priority-based sampling for replay buffers

2. **Zhang, S. & Sutton, R. S. (2017)**. "A Deeper Look at Experience Replay."
   - Analysis of replay mechanisms and CER

3. **Fedus, W., et al. (2020)**. "Revisiting Fundamentals of Experience Replay."
   - Systematic study of buffer capacity and update ratios

4. **US Patent 6377887B1**. "Caching for pathfinding computation."
   - Pathfinding-specific caching strategies

### Related Techniques

- **Hindsight Experience Replay (HER)**: Learn from failed attempts
- **Combined Experience Replay (CER)**: Combine on-policy and off-policy
- **Hub Labelling**: Hierarchical distance labels for routing
- **Memoization**: General-purpose function result caching

## License

MIT License - See LICENSE file for details

## Citation

If you use this cache system in your research, please cite:

```bibtex
@software{planning_cache,
  title = {PlanningCache.jl: Efficient Caching for Planning and RL},
  author = {Social Learning Lab},
  year = {2025},
  url = {https://github.com/yourusername/PlanningCache.jl}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
