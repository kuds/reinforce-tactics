# Feudal RL Setup Review

## Overall Assessment
The architecture is solid — clean separation between Manager and Worker networks,
proper temporal hierarchy with configurable horizons, and separate GAE computation
with segment-length-aware discounting. Below are potential bugs, functionality gaps,
and quality-of-life improvements worth addressing.

---

## Potential Bugs

### 1. Shared feature extractor in two optimizers causes double updates
**Location:** `reinforcetactics/rl/feudal_rl.py:555-567`

Both `worker_optimizer` and `manager_optimizer` include
`self.feature_extractor.parameters()`. During each PPO epoch the feature extractor
receives gradient updates from *both* optimizers sequentially. The second optimizer's
update is computed on stale feature representations since the first optimizer already
changed the weights.

**Fix options:**
- (a) Use a single shared optimizer with all three parameter groups.
- (b) Detach features for one of the two networks.
- (c) Alternate which network's gradients flow through the feature extractor each epoch.

### 2. No action masking in feudal mode
**Location:** `reinforcetactics/rl/feudal_rl.py:641`, `train/train_feudal_rl.py:191`

The worker samples actions freely without consulting the `action_mask` from the
environment. The flat PPO baseline supports `MaskablePPO`, but the feudal agent
ignores masking entirely. The worker will frequently sample invalid actions,
receiving -10.0 penalties and wasting training signal.

### 3. `_last_obs` not initialized before first `collect_rollout`
**Location:** `reinforcetactics/rl/feudal_rl.py:606`

`collect_rollout` reads `self._last_obs` at line 606, but this is only set at
line 571 (`setup_training`) as `None`. The training script manually sets
`agent._last_obs = obs`, but there's no guard — calling `collect_rollout` without
this manual initialization will crash inside `_obs_to_tensor`.

### 4. Intrinsic reward uses `state` parameter but never reads it
**Location:** `reinforcetactics/rl/feudal_rl.py:914-916`

The `state` argument is accepted but only `next_state` is ever used. If the intent
is to compute a *delta* reward (progress toward goal), both states should be compared.
Currently the function signature is misleading.

### 5. `evaluate` method accesses `info` after the loop
**Location:** `reinforcetactics/rl/feudal_rl.py:899`

```python
if info.get("winner") == self.agent_player:
```

The variable `info` is assigned inside the `while not done` loop. If the environment
instantly terminates on reset, `info` would be unbound. For `truncated=True` episodes,
the `info` may not contain the `winner` key, silently counting those as losses.

---

## Functionality Gaps

### 1. No learning rate scheduling
The training loop runs for up to 10M timesteps at a fixed learning rate. Most
successful PPO implementations use linear or cosine annealing.

### 2. No reward normalization or clipping
Extrinsic rewards span from -1000 (loss) to +1000 (win), while intrinsic rewards
are roughly [-15, +10]. Even with `alpha=0.5` weighting, extrinsic reward dominates
at terminal states, potentially causing large value function targets and unstable
training.

### 3. No multi-environment support for feudal mode
The flat baseline supports `SubprocVecEnv` with `n_envs`, but feudal training is
locked to a single environment. This significantly slows data collection. Per-env
goal state could be maintained to enable vectorized environments.

### 4. No goal-conditioned exploration bonus
The manager explores via entropy bonus alone. There's no mechanism to encourage
diverse goals over time (e.g., goal coverage tracking, count-based exploration,
or goal novelty reward).

### 5. No curriculum or adaptive horizon
The `manager_horizon` is fixed at 10. Early in training when the worker can barely
execute actions, 10 is too long. An adaptive horizon could improve learning across
training stages.

### 6. W&B integration not used in feudal mode
The `--wandb` flag is parsed and initialized in `main()`, but `train_feudal_rl()`
only writes to TensorBoard. The W&B writer is never used by the feudal training
function.

---

## Quality-of-Life Improvements

### 1. Add goal visualization/logging
Log goal distributions (heatmaps of goal_x/goal_y frequencies, goal type distribution)
to TensorBoard to make debugging easier.

### 2. Track intrinsic vs. extrinsic reward separately
Currently only `worker_mean_reward` (combined) is logged. Separating components
would help diagnose whether intrinsic reward is shaping behavior effectively.

### 3. Add gradient norm logging
Log gradient norms for manager and worker separately to detect training
instabilities early.

### 4. Add `from_config` / `to_config` to `FeudalRLAgent`
A serializable config would make checkpoint loading more robust and enable
hyperparameter sweeps.

### 5. Seed the feudal training path
`set_random_seed` is called for flat mode but the feudal training path has no
`torch.manual_seed`, `np.random.seed`, or `env.reset(seed=...)`.

### 6. Add early stopping
No mechanism to stop training if performance plateaus. An early stopping check
based on eval reward would save compute.

### 7. Expose `max_steps` as a CLI arg
The environment's `max_steps=500` is hardcoded in `train_feudal_rl` — making it
configurable would be useful for faster iteration with shorter episodes.

### 8. Resume training from checkpoint
The `load_checkpoint` method exists but the training script has no `--resume` flag.

---

## Priority Summary

| Priority | Issue | Type |
|----------|-------|------|
| High | Double-updated feature extractor via two optimizers | Bug |
| High | No action masking in feudal mode | Gap |
| High | No reward normalization/clipping | Gap |
| Medium | `_last_obs` not safely initialized | Bug |
| Medium | Unused `state` param in intrinsic reward | Bug |
| Medium | No multi-env support for feudal training | Gap |
| Medium | Seeding not applied in feudal training | QoL |
| Low | No LR scheduling | Gap |
| Low | No goal visualization logging | QoL |
| Low | No resume-from-checkpoint flag | QoL |
