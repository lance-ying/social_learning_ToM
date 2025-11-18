# Novelty Analysis & Human Experiment Design

## What Exists in the Literature

Based on comprehensive search, here's what's already been studied:

### 1. **Memory as Computational Resource** (Dasgupta & Gershman, 2021)
- ✅ **Exists**: Evidence that humans reuse computations in planning, arithmetic, mental imagery
- ✅ **Exists**: Memoization speeds up repeated computations
- ❌ **Gap**: No systematic study of *when* people choose to retrieve vs. recompute

### 2. **Resource-Rational Planning** (Callaway et al., 2022, Nature Human Behaviour)
- ✅ **Exists**: People balance planning depth with computational cost
- ✅ **Exists**: Hierarchical planning reduces computational burden
- ❌ **Gap**: Doesn't model memory/cache as a resource to manage

### 3. **Key-Value Memory** (Gershman, Fiete, & Irie, 2025)
- ✅ **Exists**: Framework for storage vs. retrieval representations
- ✅ **Exists**: Applications to neural/cognitive phenomena
- ❌ **Gap**: Not applied to *planning* tasks or *caching decisions*

### 4. **Social Learning** (Bandura, 1961+)
- ✅ **Exists**: People learn by observing others (reduces personal cost)
- ✅ **Exists**: Selective attention to prestigious/successful models
- ❌ **Gap**: No models of *when to observe* as a caching decision

### 5. **Space-Time Tradeoffs** (Computer Science)
- ✅ **Exists**: Well-studied in algorithms (gradient checkpointing, dynamic programming)
- ❌ **Gap**: Not connected to human cognition or planning behavior

### 6. **Cognitive Offloading** (Risko & Gilbert, 2016)
- ✅ **Exists**: People offload memory to external devices when internal memory is costly
- ❌ **Gap**: Doesn't study *strategic caching* of computations for reuse

## What's Novel About Your Project

### 🎯 **Core Novel Contributions**

| Contribution | Novelty |
|-------------|---------|
| **1. Cache-or-Compute Decision** | First systematic study of *when* humans/agents choose to cache vs. recompute plans |
| **2. Social Caching** | Framing observation in multi-agent settings as "caching others' plans" |
| **3. Unified Framework** | Connecting key-value memory, experience replay, and resource-rational planning |
| **4. Planning-Specific** | Most work on memory reuse is for simple computations; you focus on *planning* |
| **5. Priority Mechanisms** | Testing prioritized caching (from RL) in cognitive/planning contexts |

### Specific Novel Questions

1. **When do people retrieve from memory vs. replan?**
   - Existing work: Shows people *do* reuse plans
   - Your contribution: Model *when* they choose to reuse vs. recompute

2. **How do people decide what to cache?**
   - Existing work: Shows limited memory capacity affects behavior
   - Your contribution: Test priority-based caching (importance, recency, frequency)

3. **In social learning, when do you observe vs. explore alone?**
   - Existing work: Shows observation reduces cost
   - Your contribution: Frame as strategic caching decision with explicit cost-benefit

4. **Do humans use hierarchical caching strategies?**
   - Existing work: Hierarchical planning is efficient
   - Your contribution: Test hot/cold memory tiers for plan storage

## Human Experiment Designs

### Experiment 1: **Cache-or-Recompute in Spatial Planning**

#### Task Design
```
Participants plan paths in a grid world:
- Phase 1 (Training): Solve 40 pathfinding problems
  - Some problems repeat (A→B appears 4 times across trials)
  - Track planning time on each trial
- Phase 2 (Test): Same problems, measure retrieval vs. recomputation
```

#### Manipulations
1. **Repetition frequency**: Some paths repeated 1x, 2x, 4x, 8x
2. **Time delay**: Immediate vs. 5min vs. 1hr between repetitions
3. **Problem similarity**: Identical vs. slightly different start/goal

#### Measures
- **Planning time**: Fast = retrieved from cache; Slow = recomputed
- **Accuracy**: Identical to previous solution = cached; Different = recomputed
- **Explicit reports**: "Did you remember your previous path or replan?"
- **Eye tracking**: Fixations on previous path = retrieval

#### Predictions (Cache Model)
- ✅ Faster planning times for repeated problems (cache hit)
- ✅ Cache hit rate depends on recency (recent = more likely cached)
- ✅ High-frequency problems more likely cached (LFU policy)
- ✅ Very similar problems show "partial cache hits" (reuse subplans)

#### Implementation
```julia
# Task: Grid world pathfinding
# Display: 10×10 grid, walls, start (green), goal (red)
# Response: Click sequence of tiles to define path
# Feedback: Show optimal path after each trial

function run_experiment_1()
    # 40 trials: 10 unique problems × 4 repetitions each
    problems = generate_problems(n_unique=10, n_reps=4)

    for trial in shuffle(problems)
        rt, path, confidence = present_trial(trial)

        # Check if cached (fast + identical to previous)
        is_cached = (rt < median_rt * 0.7) && (path == prev_path[trial])

        save_data(trial, rt, path, confidence, is_cached)
    end
end
```

---

### Experiment 2: **Priority-Based Caching in Sequential Planning**

#### Task Design
```
Tower of Hanoi with resource constraints:
- Each problem costs "mental effort" (explicit points)
- Can store 3 solutions in "memory bank" (limited cache)
- Must decide which solutions to keep when cache is full
```

#### Manipulations
1. **Priority cue**: Some problems labeled "important" (high reward)
2. **Repetition probability**: "This problem may appear again" (likelihood)
3. **Complexity**: Easy (3 disks) vs. Hard (5 disks) problems

#### Measures
- **Cache decisions**: Which problems do people choose to cache?
- **Eviction strategy**: LRU? LFU? Priority-based?
- **Performance**: Success rate, total effort spent

#### Predictions (Prioritized Cache Model)
- ✅ People cache high-reward problems (priority-based)
- ✅ Cache hard problems more than easy ones (save computation)
- ✅ People use mixed strategy (recency + priority + difficulty)

#### Implementation
```julia
# Task: Tower of Hanoi with explicit cache
# Display: Problem + "Memory Bank" showing 3 cached solutions
# Decision: After each problem, "Save to memory?" or "Delete which one?"

function run_experiment_2()
    cache = CacheBank(capacity=3)

    for problem in problems
        # Solve problem
        solution, effort = solve_tower_of_hanoi(problem)

        # Cache decision
        if length(cache) < 3
            decision = ask_cache_decision(problem, solution)
        else
            # Must evict one
            decision = ask_eviction_decision(cache, problem)
        end

        update_cache!(cache, decision)
        save_data(problem, solution, effort, decision)
    end

    # Analyze eviction patterns
    analyze_eviction_strategy(data)
end
```

---

### Experiment 3: **Social Caching - When to Observe Others**

#### Task Design
```
Multi-agent foraging task (like your current setup!):
- You control one agent exploring a grid world
- Two other agents (X and Y) visible on screen, pursuing their own goals
- You can "observe" an agent (costs time) to learn their goal/strategy
- Goal: Collect gems efficiently
```

#### Manipulations
1. **Agent similarity**: Other agents have similar vs. different goals
2. **Observation cost**: Cheap (1 action) vs. Expensive (5 actions)
3. **Agent expertise**: One agent is "expert" (optimal), one is "novice" (random)

#### Measures
- **Observation decisions**: When do people choose to observe?
- **Agent selection**: Do they observe expert more?
- **Benefit**: Does observation actually help performance?

#### Predictions (Social Cache Model)
- ✅ People observe when agents have relevant goals (cache useful plans)
- ✅ Observe expert more than novice (priority-based caching)
- ✅ Less observation when cost is high (cost-benefit tradeoff)
- ✅ Cache others' plans and reuse for own goals

#### Implementation
```julia
# Task: Multi-agent grid world (like your exp3!)
# Display: You (green), Agent X (blue), Agent Y (red), Gems (yellow)
# Actions: Move, Observe X, Observe Y, Collect Gem

function run_experiment_3()
    world = GridWorld(size=15, n_gems=5, n_agents=3)

    while !goal_reached(world)
        # Show current state
        display(world)

        # Get action
        action = get_human_action()  # Move / Observe X / Observe Y

        if action == :observe_X || action == :observe_Y
            # Show agent's goal and planned path
            show_agent_plan(action)
            record_observation(action, world.state)
        else
            execute_action(world, action)
        end

        # Update other agents
        update_agents!(world)
    end

    # Analyze: When did they observe? Was it helpful?
    analyze_observation_decisions(data)
end
```

---

### Experiment 4: **Metacognitive Cache Control**

#### Task Design
```
Arithmetic problems with explicit cache:
- Solve math problems (e.g., 37 × 24)
- Can store 5 intermediate results in "scratchpad" (working memory)
- Must decide what to store for later reuse
```

#### Manipulations
1. **Problem structure**: Some problems reuse intermediate results
2. **Cache size**: 3 vs. 5 vs. 7 slots
3. **Incentive**: Speed bonus vs. accuracy bonus

#### Measures
- **Cache usage**: What do people store?
- **Retrieval patterns**: When do they retrieve vs. recompute?
- **Metacognition**: "How confident are you this will be useful?"

#### Predictions
- ✅ People cache frequently-used intermediate results
- ✅ Confidence predicts cache benefit (good metacognition)
- ✅ Under speed pressure, cache more aggressively

---

## Data Collection Plan

### Sample Size
- **N = 60** per experiment (based on power analysis for medium effect sizes)
- **Power**: 80% to detect d=0.5 effect size at α=0.05

### Platform
- **Online**: Prolific + custom JavaScript/Julia task
- **Lab**: In-person with eye-tracking (for Exp 1)

### Analysis Plan

#### Primary Analyses
1. **Cache hit rate**: How often do people retrieve vs. recompute?
2. **Planning time**: RT for cached vs. non-cached problems
3. **Strategy classification**: LRU? LFU? Priority? Mixed?
4. **Cost-benefit**: Does caching improve performance?

#### Computational Modeling
Fit cache models to individual participants:
```julia
# Model comparison
models = [
    SimpleCacheModel(policy=:LRU),
    SimpleCacheModel(policy=:LFU),
    PrioritizedCacheModel(α=0.6),
    HierarchicalCacheModel(hot_size=3, cold_size=10),
    NoCache Baseline()
]

for participant in participants
    for model in models
        LL = log_likelihood(model, participant.data)
        BIC = compute_BIC(LL, model.n_params, participant.n_trials)
        save_fit(participant, model, LL, BIC)
    end
end

# Best-fitting model per participant
best_models = get_best_models(fits)
```

### Statistical Tests
- **ANOVA**: Cache condition × Repetition × Problem type
- **Mixed models**: Random effects for participants and items
- **Model comparison**: BIC/AIC for computational models

---

## Key Testable Predictions

| Hypothesis | Test | Expected Result |
|------------|------|-----------------|
| **H1: Humans cache computations** | Planning time for repeated problems | RT decreases with repetition |
| **H2: Cache follows rational policy** | Cache decisions vs. optimal | Correlation with cost-benefit |
| **H3: Priority-based caching** | Cache high-value problems more | Higher cache rate for important items |
| **H4: Social caching is strategic** | Observation decisions | Observe when relevant + expert |
| **H5: Metacognitive control** | Confidence predicts cache benefit | High confidence = better caching |

---

## Why This Is Novel

### Compared to Existing Work

| Prior Work | Your Contribution |
|------------|-------------------|
| **Dasgupta & Gershman**: Humans reuse computations | ➡️ **When** do they decide to reuse? Model the decision |
| **Callaway et al.**: Resource-rational planning | ➡️ Memory/cache as explicit resource to manage |
| **Gershman et al.**: Key-value memory framework | ➡️ Apply to planning tasks and caching decisions |
| **Bandura**: Social learning via observation | ➡️ Frame as strategic caching of others' plans |
| **RL/ML**: Experience replay | ➡️ Test in human cognition, not just agents |

### Bridging Multiple Fields

Your project uniquely connects:
1. **Cognitive science**: Human memory and planning
2. **Reinforcement learning**: Experience replay and PER
3. **Computer science**: Caching algorithms and memoization
4. **Social learning**: Observation decisions
5. **Neuroscience**: Key-value memory and hippocampus

---

## Publication Strategy

### Target Journals

1. **Cognitive Science** (top tier, interdisciplinary)
   - Emphasize cognitive modeling + human experiments

2. **Psychological Review** (theory-focused)
   - Emphasize unified framework for memory in planning

3. **Nature Human Behaviour** (high impact)
   - Emphasize resource-rational caching + novel predictions

4. **PLOS Computational Biology** (computational focus)
   - Emphasize modeling and simulations + human data

5. **Cognition** (solid cognitive journal)
   - Emphasize experimental results

### Conference Presentations

- **CogSci** (Cognitive Science Society): Human experiments + modeling
- **NeurIPS** (ML focus): Computational methods + RL connections
- **CCN** (Computational Cognitive Neuroscience): Bridge cognition and ML

---

## Next Steps

1. ✅ **Week 1-2**: Pilot Experiment 1 (spatial planning) with N=20
2. ✅ **Week 3**: Analyze pilot data, refine task design
3. ✅ **Week 4-6**: Run full Experiment 1 (N=60)
4. ✅ **Week 7-8**: Build and test Experiments 2-3
5. ✅ **Week 9-12**: Collect all data
6. ✅ **Week 13-16**: Analysis + modeling + writing

---

## Summary

### Is it novel? **YES!**

- ✅ No prior work systematically studies *cache-or-compute* decisions in humans
- ✅ No work applies key-value memory framework to *planning*
- ✅ No work frames social observation as *strategic caching*
- ✅ Novel connection between cognitive science, RL, and algorithms

### Can you collect human data? **YES!**

- ✅ Clear experimental paradigms (4 experiments designed above)
- ✅ Measurable dependent variables (RT, decisions, accuracy)
- ✅ Strong predictions from computational models
- ✅ Feasible to run online (Prolific) or in lab

### Will it make an impact? **VERY LIKELY!**

- ✅ Bridges multiple fields (cognition, ML, neuroscience)
- ✅ Practical applications (efficient AI planning)
- ✅ Theoretical contributions (memory as planning resource)
- ✅ Timely (builds on hot Gershman/Irie paper)
