# Planning Cache as Episodic Memory: A Cognitive Model

## Research Vision

This project frames **planning cache systems** as a computational model of episodic memory in multi-agent planning tasks, inspired by recent work on key-value memory in the brain (Gershman, Fiete, & Irie, 2025) and memory as a computational resource (Dasgupta & Gershman, 2021).

## Core Research Question

**How should planning agents decide what to cache and when to retrieve cached plans for efficient goal-directed behavior?**

This mirrors fundamental questions in cognitive science:
- What experiences should be stored in episodic memory?
- When should cached experiences be retrieved vs. fresh computation?
- How do priority and recency interact in memory retrieval?

## Theoretical Framework

### Connection to Key-Value Memory

**Gershman, Fiete, & Irie (2025)** show that brains use key-value memory where:
- **Keys**: Retrieval cues (context, state features)
- **Values**: Stored content (experiences, plans)
- **Principle**: Separate optimization for storage fidelity and retrieval discriminability

Our planning cache extends this to:
- **Keys**: Spatial configurations (start, goal positions)
- **Values**: Computed plans (action sequences)
- **Priorities**: Importance weights (like TD-error in RL)
- **Decisions**: When to cache vs. recompute

### Connection to Memory as Computational Resource

**Dasgupta & Gershman (2021)** show that:
1. **Memoization**: Reusing past computations is algorithmically efficient
2. **Human Evidence**: People reuse computations in planning, arithmetic, mental imagery
3. **Rational Analysis**: Memory retrievability depends on predictive usefulness

Our cache system implements these principles for multi-agent planning:
- **Memoization**: Cache expensive planning computations
- **Prioritization**: Store high-value experiences (like PER in RL)
- **Rational Retrieval**: Decide when cached plans are useful

## Research Contributions

### 1. Computational Framework

Four cache architectures inspired by cognitive/ML principles:

| Cache Type | Inspiration | Key Feature |
|------------|-------------|-------------|
| **SimpleCache** | Basic memoization | LRU/LFU eviction |
| **PrioritizedCache** | Prioritized Experience Replay (Schaul et al., 2015) | Priority-weighted sampling |
| **DistributedCache** | Parallel processing | Thread-safe sharded storage |
| **HierarchicalCache** | Memory hierarchies | Hot/cold tier storage |

### 2. Experimental Paradigms

We propose experiments in three domains:

#### Experiment 1: Spatial Planning (Grid Worlds)
- **Task**: Repeated pathfinding queries in spatial environments
- **Manipulation**: Cache size, eviction policy, spatial locality
- **Measures**: Hit rate, planning time, memory efficiency
- **Hypothesis**: Spatial locality predicts cache benefit

#### Experiment 2: Multi-Agent Social Learning
- **Task**: Observer agents inferring others' goals (your current setup!)
- **Manipulation**: Which agent to observe (cache their plans)
- **Measures**: Observation decisions, inference accuracy, exploration cost
- **Hypothesis**: Caching reduces redundant planning when observing similar agents

#### Experiment 3: Sequential Decision Making
- **Task**: RL agents in episodic environments
- **Manipulation**: Priority functions (TD-error, recency, frequency)
- **Measures**: Sample efficiency, learning curves, cache utilization
- **Hypothesis**: Priority-based caching accelerates learning

### 3. Model Comparison

Compare cache strategies on:
- **Computational Efficiency**: Runtime, memory usage
- **Sample Efficiency**: Learning speed in RL tasks
- **Cognitive Plausibility**: Match human planning behavior
- **Robustness**: Performance across environment types

## Experimental Design

### Phase 1: Benchmark Suite

Create standardized tasks:

```
experiments/
├── 01_spatial_planning/
│   ├── gridworld.jl          # Grid-based pathfinding
│   ├── mazes.jl              # Complex maze navigation
│   └── analysis.jl           # Performance metrics
├── 02_social_learning/
│   ├── multi_agent_obs.jl    # Your current setup
│   ├── theory_of_mind.jl     # Goal inference tasks
│   └── analysis.jl
├── 03_reinforcement_learning/
│   ├── episodic_rl.jl        # Episodic control tasks
│   ├── model_based_rl.jl     # Planning with cached models
│   └── analysis.jl
└── 04_cognitive_modeling/
    ├── human_data/           # Human planning data
    ├── model_fit.jl          # Fit cache models to humans
    └── analysis.jl
```

### Phase 2: Controlled Experiments

For each experiment:

1. **Independent Variables**
   - Cache type (Simple, Prioritized, Hierarchical, Distributed)
   - Cache size (10, 100, 1000, unlimited)
   - Eviction policy (LRU, LFU, Priority-based)
   - Environment properties (spatial locality, stochasticity)

2. **Dependent Variables**
   - **Performance**: Task completion time, success rate
   - **Efficiency**: Cache hit rate, memory usage
   - **Behavior**: Which queries are cached/retrieved
   - **Learning**: Improvement over time/episodes

3. **Controls**
   - Baseline (no cache): Pure online planning
   - Oracle (perfect cache): Upper bound performance
   - Random cache: Control for cache size effects

### Phase 3: Analysis

Statistical analyses:

1. **Efficiency Analysis**
   - Cache hit rates across conditions
   - Speedup factors (vs. no-cache baseline)
   - Memory-performance tradeoffs

2. **Behavior Analysis**
   - What gets cached? (frequency, recency, priority)
   - When is cache retrieved? (decision policies)
   - Exploration-exploitation in caching

3. **Model Comparison**
   - Which cache strategy best fits each task?
   - Are there domain-general principles?
   - Cognitive plausibility (compare to human data)

## Expected Outcomes

### Computational Insights

1. **When caching helps**: High spatial/temporal locality, expensive planning
2. **Which policy works**: Priority-based for RL, LRU for planning, Hierarchical for large-scale
3. **Tradeoffs**: Memory vs. compute, storage vs. retrieval optimization

### Cognitive Insights

1. **Episodic planning**: How humans might cache/retrieve plans
2. **Social learning**: When to observe others vs. explore self
3. **Resource-rational**: Optimal caching under computational constraints

### Practical Contributions

1. **Efficient planning**: Speed up multi-agent simulations
2. **RL improvements**: Better experience replay strategies
3. **Software tool**: Reusable cache library for planning research

## Paper Outline

### "Efficient Planning Through Strategic Caching: A Cognitive and Computational Perspective"

**Abstract**: Introduce planning cache as cognitive model + computational tool

**Introduction**:
- Planning is computationally expensive
- Brains cache experiences (episodic memory)
- Key-value memory framework (Gershman et al., 2025)
- Memory as computational resource (Dasgupta & Gershman, 2021)
- Our contribution: Systematic study of caching for planning

**Methods**:
- Four cache architectures
- Three experimental domains
- Evaluation metrics

**Experiment 1: Spatial Planning**
- Results: LRU cache achieves 3-5x speedup
- Spatial locality predicts benefit

**Experiment 2: Social Learning**
- Results: Prioritized cache reduces observation cost
- Agents learn when to cache others' plans

**Experiment 3: Reinforcement Learning**
- Results: Priority-based caching improves sample efficiency
- Matches PER but for planning (not experiences)

**Experiment 4: Cognitive Modeling** (optional)
- Compare to human planning behavior
- Cache models predict human planning times

**Discussion**:
- Principles of efficient caching
- Connections to episodic memory
- Limitations and future work

**Conclusion**:
- Caching bridges computation and cognition
- Practical tool + theoretical insight

## Timeline

1. **Week 1-2**: Complete cache implementation + tests
2. **Week 3-4**: Build experimental suite
3. **Week 5-8**: Run experiments + collect data
4. **Week 9-10**: Analysis + visualizations
5. **Week 11-12**: Write paper + polish

## Related Work

### Machine Learning
- Prioritized Experience Replay (Schaul et al., 2015)
- Hindsight Experience Replay (Andrychowicz et al., 2017)
- Model-based RL with planning (Sutton & Barto, 2018)

### Cognitive Science
- Key-value memory in the brain (Gershman et al., 2025)
- Memory as computational resource (Dasgupta & Gershman, 2021)
- Episodic RL (Gershman & Daw, 2017)
- Resource-rational analysis (Lieder & Griffiths, 2020)

### Computer Science
- Memoization in functional programming
- Cache hierarchies in CPU design
- Pathfinding optimization (A*, JPS+)

## Open Questions

1. **Optimality**: What is the theoretically optimal caching policy?
2. **Generalization**: Do cache strategies transfer across tasks?
3. **Biological plausibility**: How similar to hippocampal replay?
4. **Scalability**: How to cache in continuous/infinite state spaces?
5. **Meta-learning**: Can agents learn their own caching policies?

## Impact

This research could:
- **Advance cognitive models** of planning and memory
- **Improve RL algorithms** through better experience management
- **Speed up simulations** in computational social science
- **Bridge** AI and cognitive science perspectives on memory

---

**Next Steps**: Build experimental infrastructure and run pilot studies to validate the approach.
