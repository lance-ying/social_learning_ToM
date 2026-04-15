# Boltzmann Action Scoring and SIPS

This note explains, at a low level, how the observed-agent action probabilities
are produced in this repo and how those probabilities are used by SIPS.

## 1. Where the Boltzmann model is configured

In the problem example script, the observed agent is given a Boltzmann action
model here:

- `planner = RTHS(GoalManhattan(), n_iters = 2, max_nodes = 2^18)`
- `act_config = BoltzmannActConfig(0.5)`

Source:

- `dataset/problem_example/run_inference_and_experiment.jl`

So the action-selection temperature is:

- `T = 0.5`

## 2. What is being softmaxed

The key implementation is in `SymbolicPlanners`.

`BoltzmannPolicy` defines:

```text
P(a | s) is proportional to exp(Q(s,a) / T)
```

Source:

- `/Users/heyodogo/.julia/packages/SymbolicPlanners/6oEf5/src/solutions/boltzmann_policy.jl`

The actual probability computation is:

```julia
probs = softmax(q_values ./ sol.temperature)
```

So the thing being softmaxed is:

- the vector of action Q-values in the current state
- divided by temperature

Not:

- costs directly
- posterior probabilities
- timesteps `T`

Another way to write the same thing is:

```text
score(a) = Q(s,a) / T
P(a | s) = exp(score(a)) / sum_b exp(score(b))
```

So the softmax is not "just dividing by T".

It is:

1. scale each action value by `1 / T`
2. exponentiate those scaled values
3. normalize them into probabilities

Why divide by `T` at all:

- smaller `T` makes Q-value differences matter more
- larger `T` makes Q-value differences matter less

Example:

```text
Q(left) = 8
Q(up)   = 6
```

If `T = 1.0`, then the gap is:

```text
(8 - 6) / 1.0 = 2
```

If `T = 0.5`, then the gap becomes:

```text
(8 - 6) / 0.5 = 4
```

So a lower temperature makes the policy more peaked, because the better action
gets a much larger exponentiated score relative to the others.

## 3. Where the Q-values come from

The Boltzmann wrapper does not invent the Q-values itself. It asks the
underlying planner solution for them:

```julia
action_values = get_action_values(sol.policy, state)
```

In this codepath, the underlying solution comes from the planner stored in the
agent's `plan_state.sol`.

At a high level:

- the planner evaluates the current state relative to the goal
- it assigns each available action a value `Q(s,a)`
- Boltzmann turns those values into action probabilities

So "utility" here means:

- planner-provided action value for taking action `a` in state `s`

## 3.5. What that means in this exact script

In `dataset/problem_example/run_inference_and_experiment.jl`, the goal prior
returns:

```julia
Specification(goals[goal_id])
```

In `SymbolicPlanners`, that constructor means:

```julia
Specification(goal::Term) = MinStepsGoal(goal)
```

So the observed agent is solving a **minimum number of steps to the goal**
problem.

For `MinStepsGoal`:

- each action has reward `-1`
- discount is `1.0`

So the planner action value is:

```text
Q(s,a) = -1 + V(s')
```

where `s'` is the next state after taking action `a`.

Since `V(s)` is stored as a negative remaining-cost-to-go estimate, this means:

- actions that get to the goal faster have **higher** Q-values
- actions that move away or waste time have **lower** Q-values

At initialization, `RTHS` uses a heuristic value policy whose state value is:

```text
V(s) = -h(s)
```

with `h` given here by `GoalManhattan()`.

In this repo, `GoalManhattan()` is:

- Manhattan distance from the agent to the goal object
- plus the number of remaining goal terms

So before search refinement, the action values are roughly:

```text
Q(s,a) ~= -1 - h(s')
```

After `RTHS` runs search, those values are refined using the search tree, so
they become a better estimate of the true step cost-to-go than the raw
heuristic alone.

So the short answer is:

- **yes**, in this script the action values are fundamentally based on estimated
  remaining steps to the goal
- but they are not just "read off the final plan length"
- they come from a heuristic/search value function over successor states

## 4. How the observed action gets a likelihood

`InversePlanning` uses:

```julia
log(SymbolicPlanners.get_action_prob(policy, state, act))
```

Source:

- `/Users/heyodogo/.julia/packages/InversePlanning/WrXNa/src/modeling/agents/actions.jl`

That means:

1. Compute probabilities for all available actions with Boltzmann softmax.
2. Look up the probability of the action that was actually observed.
3. Use that probability as the likelihood under that particle/hypothesis.

## 5. Tiny numerical example

Suppose a hypothesis says the available actions have these Q-values:

| Action | Q-value |
| --- | ---: |
| `left` | 8 |
| `up` | 6 |
| `right` | 1 |

and temperature is `T = 0.5`.

Then the softmax is over:

| Action | Q/T |
| --- | ---: |
| `left` | 16 |
| `up` | 12 |
| `right` | 2 |

So the probabilities are:

```text
P(a | s) = exp(Q(s,a)/T) / sum_b exp(Q(s,b)/T)
```

This makes:

- `left` very likely
- `up` somewhat likely
- `right` very unlikely

If you want concrete numbers:

```text
Q(left)  = 8   -> 8 / 0.5 = 16
Q(up)    = 6   -> 6 / 0.5 = 12
Q(right) = 1   -> 1 / 0.5 = 2
```

Then softmax compares:

```text
exp(16), exp(12), exp(2)
```

Since `exp(16)` is much larger than `exp(12)`, and both are much larger than
`exp(2)`, `left` gets most of the probability mass.

If the observed action was `up`, then the likelihood for this hypothesis is:

- `P(up | s, hypothesis)`

If the observed action was `right`, the likelihood is much smaller.

## 6. How particles use that likelihood

In this repo, each particle corresponds to a hidden explanation:

- observed agent goal
- latent initial world state

In the simple example, that is effectively a `(goal, state)` pair.

For each newly observed action:

1. Each particle predicts action probabilities using Boltzmann.
2. Each particle gets the likelihood of the observed action.
3. The particle's weight is multiplied by that likelihood.
4. We normalize across particles.

So if a particle strongly predicts the observed action, it keeps more weight.
If it predicts the observed action poorly, it loses weight.

## 7. What SIPS is doing with those particles

SIPS is a sequential importance-sampling style inference procedure.

At a high level, after each observed action:

1. Extend/update each hidden hypothesis forward by one timestep.
2. Score the observed action under that hypothesis.
3. Reweight the particles.
4. Read out marginals over:
   - goal
   - latent state

So the posterior tables you later use in social learning are built from repeated
applications of:

- prior particle weight
- times Boltzmann action likelihood
- then normalized

## 8. Why this matters for the social-learning `Q_observe`

The social-learning code does not use Boltzmann directly.

Instead, it uses the posterior trajectories produced by SIPS:

- `goal_probs_conditioned_dict[...]`
- `state_probs_conditioned_dict[...]`

Then, for each current timestep:

1. Use those posterior probabilities as weights on hypotheses.
2. For each hypothesis, estimate how much future observation would help.
3. Add:
   - self-exploration cost after observing
   - observe cost
4. Average across hypotheses to get `Q_observe`.

So the chain is:

1. Planner produces action values.
2. Boltzmann converts action values into action probabilities.
3. SIPS uses those action probabilities to update particle weights.
4. Posterior probabilities over goals/states are extracted.
5. Social-learning code uses those probabilities to compute `Q_observe`.

## 9. Short version

If you want the one-line intuition:

- the planner says how good each action is
- Boltzmann turns "goodness" into a probability distribution
- SIPS uses the observed action's probability to reweight hidden hypotheses
- those posterior weights are later used inside `Q_observe`
