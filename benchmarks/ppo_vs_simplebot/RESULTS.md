# PPO Baseline Benchmark Results

## Run Configuration

- **Map:** `maps/1v1/beginner.csv` (6x6)
- **Opponent:** SimpleBot
- **Algorithm:** MaskablePPO (sb3-contrib)
- **Max steps per episode:** 500
- **Eval episodes per checkpoint:** 50
- **Parallel training envs:** 4
- **Seed:** 42

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| learning_rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |

## Results

| Timesteps | Win Rate (%) | Avg Reward | Avg Length | Wins | Losses | Draws | Train Time (s) |
|-----------|-------------|------------|-----------|------|--------|-------|----------------|
| 10,000 | 0.0 | -4965.1 | 500.0 | 0 | 0 | 50 | 85.3 |
| 50,000 | 0.0 | -4965.1 | 500.0 | 0 | 0 | 50 | 208.4 |
| 200,000 | 0.0 | -4978.1 | 500.0 | 0 | 0 | 50 | 836.7 |
| 1,000,000 | 0.0 | -1066.9 | 10.0 | 0 | 50 | 0 | 4229.5 |

### Expected Results (from notebook reference)

| Timesteps | Expected Win Rate |
|-----------|------------------|
| 10,000 | 0-15% |
| 50,000 | 15-40% |
| 200,000 | 40-70% |
| 1,000,000 | 60-90%+ |

## Root Cause Analysis

### The core problem: per-dimension action masking over-approximation

The action space is `MultiDiscrete([10, 8, 6, 6, 6, 6])` representing
`(action_type, unit_type, from_x, from_y, to_x, to_y)`.

`MaskablePPO` with `MultiDiscrete` spaces applies **per-dimension** masks: each
of the 6 dimensions gets an independent boolean mask indicating which values are
valid. These masks are the **union** of valid values across all legal actions.

For example, if the only legal actions are:
- `(move, -, 1, 0, 2, 0)` (move unit at (1,0) to (2,0))
- `(attack, -, 3, 2, 4, 2)` (attack from (3,2) to (4,2))

The per-dimension masks would allow:
- `action_type`: {move, attack, end_turn}
- `from_x`: {1, 3}, `from_y`: {0, 2}
- `to_x`: {2, 4}, `to_y`: {0, 2}

This creates `3 x 8 x 2 x 2 x 2 x 2 = 384` "mask-valid" combinations, but
only **3** are actually game-valid (the 2 actions + end_turn). That means
**~99.2% of sampled actions are invalid**.

### Numerical breakdown of the results

**10K-200K checkpoints (all draws, avg reward ~-4965):**
- Episodes hit `max_steps=500` without either player winning (draws)
- With `invalid_action` penalty = -10 and ~99% invalid action rate:
  `500 steps x 0.99 x (-10) = -4,950` from penalties alone
- Plus small turn penalties and shaping noise: matches the observed ~-4965
- The agent cannot learn useful behavior because the reward is dominated by
  invalid-action noise

**1M checkpoint (all losses, avg length 10, reward ~-1067):**
- After enough training, the agent learns to avoid the -10 penalty by spamming
  `end_turn` (action_type=5), which is **always valid**
- But ending turns without acting means SimpleBot plays freely
- SimpleBot creates units, captures structures, and wins quickly
- Reward ~= -1000 (loss) + small penalties = -1067

### Summary of failure modes

1. **Phase 1 (10K-200K):** Agent randomly samples from mask-valid space, hits
   invalid actions ~99% of the time, accumulates massive penalties, episodes
   time out as draws
2. **Phase 2 (1M):** Agent discovers `end_turn` avoids penalties, collapses to
   always ending turn, SimpleBot wins immediately

## Improvement Suggestions

### 1. Switch to flat Discrete action space with exact masking (highest impact)

Replace `MultiDiscrete([10, 8, 6, 6, 6, 6])` with a `Discrete(N)` space where
each index maps to a specific legal action tuple. The mask becomes exact: only
truly valid actions are maskable.

```python
# Enumerate all legal actions into a flat list each step
legal_actions = game_state.get_legal_actions(player)
flat_actions = []  # List of (action_type, unit_type, from_x, from_y, to_x, to_y)
for key, actions in legal_actions.items():
    for action in actions:
        flat_actions.append(encode(key, action))
flat_actions.append(END_TURN_ACTION)  # Always available

# Mask: exactly len(flat_actions) are valid
mask = np.zeros(MAX_ACTIONS, dtype=bool)
mask[:len(flat_actions)] = True
```

This eliminates invalid actions entirely. The agent only ever samples from
game-valid actions. The tradeoff is a variable-size action space that requires
padding to a fixed max size, but `MaskablePPO` handles this naturally.

**Expected impact:** Eliminates ~99% of wasted samples and removes the
invalid-action penalty noise from the reward signal.

### 2. Reduce or eliminate the invalid action penalty

If keeping `MultiDiscrete`, reduce `invalid_action` from -10 to something much
smaller (e.g., -0.1) or 0. The current -10 penalty creates a reward signal ~50x
larger than any useful game reward, making it impossible for the agent to
learn from actual gameplay signals.

```python
reward_config = {
    'invalid_action': -0.1,  # Was -10.0
    # ... rest unchanged
}
```

Alternatively, treat invalid actions as no-ops (skip them, no penalty). The
agent will still learn to avoid them over time because valid actions yield
positive rewards.

### 3. Auto-retry with valid action fallback

Instead of penalizing invalid actions, automatically re-sample or fall back to
`end_turn` when the agent picks an invalid combination:

```python
def step(self, action):
    action_dict = self._encode_action(action)
    reward, is_valid = self._execute_action(action_dict)
    if not is_valid:
        # Fall back to end_turn instead of penalizing
        self._execute_action({'action_type': 5, ...})
        reward = self.reward_config['turn_penalty']
    ...
```

This keeps the training loop moving with valid game progression rather than
stalling on repeated invalid actions.

### 4. Increase entropy coefficient for exploration

With `ent_coef=0.01`, the policy can collapse early (e.g., to spamming
`end_turn`). Increasing entropy encourages broader exploration:

```python
PPO_CONFIG = dict(
    ent_coef=0.05,  # Was 0.01; try 0.02-0.1
    # ... rest unchanged
)
```

Consider using entropy coefficient annealing: start high (0.1) and decay to
0.01 over training.

### 5. Simplify the action space with a two-stage architecture

Split action selection into two stages:
1. **Stage 1:** Select which unit to act with (or create/end_turn) - small
   Discrete space with exact masking
2. **Stage 2:** Given the unit, select what to do - another small Discrete
   space with exact masking

This is easier to mask exactly and reduces the combinatorial explosion.

### 6. Curriculum learning on unit types

Start training with a reduced set of units (e.g., just Warriors) against a
random opponent, then gradually add unit types and increase opponent difficulty:

```python
# Phase 1: Warriors only vs random
env = make_maskable_env(
    enabled_units=['W'],
    opponent='random',
    reward_config={'invalid_action': -0.1}
)

# Phase 2: Add Archers
# Phase 3: Full unit set vs SimpleBot
```

### 7. Reduce max_steps and increase turn_penalty

500 steps is very long for a 6x6 map. A typical game should resolve in
50-100 agent steps. Reducing `max_steps` limits wasted computation on
degenerate episodes:

```python
max_steps = 200    # Was 500
turn_penalty = -1.0  # Was -0.1; stronger incentive to act, not stall
```

### 8. Normalize observations

The `global_features` observation has values ranging from 0-10000 (gold) mixed
with 0-500 (turn number) and 0-10 (unit counts). Adding observation
normalization would help the neural network learn:

```python
# Wrap with VecNormalize
from stable_baselines3.common.vec_env import VecNormalize
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
```

## Recommended Priority Order

1. **Switch to flat Discrete action space** (suggestion 1) - fixes the root cause
2. **Reduce invalid_action penalty** (suggestion 2) - quick mitigation if keeping MultiDiscrete
3. **Increase ent_coef** (suggestion 4) - prevents early policy collapse
4. **Reduce max_steps** (suggestion 7) - faster iteration
5. **Curriculum learning** (suggestion 6) - helps with complex action space
6. **Observation normalization** (suggestion 8) - general training improvement
7. **Two-stage architecture** (suggestion 5) - longer-term architectural improvement
8. **Auto-retry fallback** (suggestion 3) - alternative to suggestion 2
