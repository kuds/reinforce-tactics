# PPO Bootstrap — Lessons Learned

Retrospective on the curriculum-bootstrap work for `notebooks/ppo_bootstrap.ipynb`
and `reinforcetactics/rl/bootstrap.py`. Captures what we learned the hard way
about cold-starting PPO on this strategy game, especially the long arc trying
to teach the agent to capture the enemy HQ.

These lessons came out of multiple long Colab training runs that stalled in
ways the obvious fixes didn't resolve. Read this before tweaking the
curriculum or reaching for "easier" opponents — many of the moves that *seem*
easier are actively harder for PPO.

## TL;DR

1. **Don't curriculum-train PPO against deterministic opponents.** A NoopBot
   produces zero state variance, which produces zero return variance, which
   produces zero advantage signal. PPO can't learn from constant rewards.
2. **The "easier" opponent isn't always easier for the algorithm.** Random
   opponents are the actually-easy training target because their actions
   inject the variance PPO needs.
3. **HQ capture is hard to learn directly.** It requires a 7+ step
   committed sequence (build → walk → seize ×4) on a sparse-reward terrain.
   Easier to teach via curriculum stages where elimination is also a viable
   win path.
4. **Match reward shape to map geometry.** A 5×elimination-vs-HQ-capture
   bonus that worked on a tiny map (`starter.csv`) becomes a perverse
   incentive on a bigger map (`beginner.csv`) where HQ capture is
   geometrically impractical.
5. **Watch `std=0.0` on eval episodes.** It's the single most diagnostic
   signal — it almost always means the policy is locked, not that the
   environment is deterministic by nature.

## The cold-start problem on this env

PPO starts with a near-uniform policy (orthogonal init scaled by 0.01). The
agent can't yet do anything useful, so its rollouts are mostly losing
trajectories. Three specific failure modes showed up:

### Failure mode 1: dead-policy trap

**Symptom**: every eval episode produces *exactly* the same reward,
length, and W/L/D breakdown. `std_reward = 0.0` across 30 episodes for
hundreds of thousands of steps. `approx_kl ≈ 0` in TensorBoard.

**Cause**: returns are constant across all rollouts → value function fits
the constant → advantages collapse to ≈ 0 → policy gradient ≈ 0 → no
updates → no exploration → no new training data → loop. The policy is
permanently stuck at whatever its initialisation argmax happened to be.

**Fix**: introduce *anything* that gives returns variance. The simplest
source is a stochastic opponent. Bigger entropy bonus alone doesn't help if
the rewards collapse to constants regardless of action choice.

### Failure mode 2: catastrophic value-function dislocation on map shifts

**Symptom**: agent reaches 100% WR on `starter.csv` (reward `std=0.0` on
eval — fully deterministic policy, totally overfit), then on first
exposure to `beginner.csv` collapses to 0% WR with the policy spamming
`end_turn`. Training-time `train/explained_variance` drops from ~0.9 to
~0.05 at the transition.

**Cause**: the value head learned to predict "starter-map state values."
Beginner-map states are dimensionally similar but produce wildly
different actual returns, so the predictions are catastrophically wrong.
PPO's gradient signal goes haywire trying to fix a value function that's
predicting a totally different distribution; meanwhile the policy
reverts toward whatever's "safe" (ending the turn) under the noise.

**Fix**: don't let the starter policy crystallise too hard. Use
`patience: 1` on starter stages so they promote at first hit of the
threshold rather than confirming with a second eval. This still produces
a competent policy but leaves it less perfectly converged, which transfers
better. Also bump `ent_coef` (e.g. 0.05 → 0.10) on the first beginner
stage to inject exploration noise during the transition.

### Failure mode 3: structural reward bias toward `end_turn`

**Symptom**: on noop stages, PPO gradient drift over time pushes the
policy toward `end_turn`. End_turn sample probability rises from a
~20% uniform prior to ~26% over 50k steps and keeps creeping. Even after
the policy briefly samples `create_unit` or `move`, the gradient noise
from low-explained-variance updates pushes it back toward `end_turn`.

**Cause**: `turn_penalty = -20` per turn applied unconditionally creates
a constant negative drag on V(s) for every state. The agent doesn't
experience this as an "end_turn cost" — it experiences it as part of
V(s) for every state, which makes "advance the turn faster" structurally
attractive. Over many slow updates this compounds, even though end_turn
doesn't actually shorten the fixed-length episode.

**Fix**: remove the structural bias. On stages without a real opponent
putting time pressure on the agent, set `turn_penalty: 0` and rely on
positive per-action shaping (e.g. `create_unit: +30`, `move: +5`) to
establish a strict ordering: `seize_progress > capture > create_unit >
move > end_turn`. Or, more robustly: don't run noop stages at all
(see below).

## Why we removed the `noop` stages

The original idea was reasonable on paper: introduce stage 0 against a
literal-no-op opponent so the agent could learn navigation and HQ
capture in isolation, before adding combat dynamics. In practice it
produced the worst PPO failure modes we encountered.

`NoopBot.take_turn()` calls `end_turn()` and exits — zero opponent
actions, zero state perturbation. Combined with deterministic evaluation
(`MaskablePPO.predict(deterministic=True)` picks argmax), this means
*every* eval episode produces an identical trajectory, identical reward,
identical length. Across 30 episodes, `std=0.0`.

The chain of consequences:

```
deterministic opponent
    ↓ no state variance across episodes
constant return distribution
    ↓ value function fits the constant
advantages collapse to ≈ 0
    ↓ policy gradient ≈ 0
no policy updates
    ↓ no exploration of new actions
training rollouts stay near initialisation
    ↓ same constant returns
[loop: stuck]
```

The agent eventually shifts the argmax of *one* state (the empty-board
state at episode start, where positive `create_unit` reward fires every
time it samples a build) but never shifts the argmax of post-build
states. Result: 1 build, then 20 `end_turns`. Across 250k steps the
policy mode literally did not change.

Random / SimpleBot / MediumBot opponents avoid this because their
actions cause the game state to differ across episodes, which makes
returns vary, which gives PPO advantage signal to update on. **The
"easier" opponent is the harder learning target.**

If we ever bring noop-style training back, it would have to be paired
with a non-RL warm-up (behaviour cloning from scripted demos) so the
policy enters PPO already knowing the build-and-seize pattern. Pure RL
can't bootstrap on a deterministic-opponent stage.

## Why we kept (and didn't change) the original 6-stage layout

The pre-bootstrap configuration that *worked* was:

```
starter map:  random → simple → medium
beginner map: random → simple → medium
```

We added per-stage overrides (`max_turns`, `max_steps`, `ent_coef`,
`reward_config`) to handle the starter→beginner map shift, plus a
`balanced_random` stepping stone on beginner for slightly easier
opponent dynamics. But we kept opponent randomness as the primary
exploration driver throughout.

Specifically, the things to *not* skip:

- **Random as the very first opponent.** PPO needs the variance.
- **`patience: 1` on starter stages.** Prevents over-fitting that
  doesn't transfer.
- **Entropy bump on the first beginner stage.** Map shift is the
  dangerous transition; cooling back to 0.05 over the next stages.
- **Reward reshape on beginner stages with opponents.** HQ capture is
  geometrically impractical on the bigger map; equalize HQ-capture and
  elimination terminal rewards (was 5000/1000, now 3000/3000) and drop
  `seize_progress` from 300 to 50 so the agent stops being paid to
  start seizes it can't finish in contested space.

## Things that helped along the way

Even after dropping noop, several other interventions stayed:

- **Bigger MLP**: SB3 default `net_arch=[64, 64]` is undersized for a
  ~734-dim Dict observation feeding a 512-dim flat-discrete action
  head. The hidden layer has fewer dims than action choices. Bumped to
  `[256, 256]`, total ~640K params. No spatial extractor (yet) — a
  small CNN over the grid/units channels would probably help further
  but not strictly required.
- **`policy_kwargs` field on `PPOConfig`**: makes net_arch and other
  policy customisations configurable from YAML rather than buried in
  `_default_model_factory`.
- **`reset_num_timesteps=False` per stage**: keeps a global timestep
  axis for cross-stage WR plots. Note: `model.learn(total_timesteps=N,
  reset_num_timesteps=False)` actually trains *until*
  `num_timesteps >= num_timesteps + N`, i.e. SB3 adds the current
  count to the requested total at setup time. Each stage's printed
  budget is "N additional steps," not absolute.
- **`PromotionCallback` with patience**: the right way to advance
  stages is "WR ≥ threshold for `patience` consecutive evals," not
  "first eval that hits threshold." Avoids one-eval lucky promotions.
- **`CurriculumStalled` exception with a clear message**: when a stage
  exhausts its budget without promoting, fail loudly rather than
  silently shipping a non-promoting checkpoint to the next stage.

## Diagnostic plays that worked

When a stage stalls, in this order:

1. **Eyeball the W/L/D breakdown.** `0/0/30` (all draws) means episodes
   are timing out. `0/30/0` means the opponent is killing the agent.
   Different fixes for each.
2. **Check `std_reward`.** `0.0` = locked policy on a deterministic env.
   Anything > 0 = the policy is doing different things in different
   episodes (good, even if WR is low).
3. **Decompose the reward.** `mean − terminal_baseline` tells you
   whether the agent is collecting any shaping reward. If shaping is
   ~0, the agent isn't doing anything productive (no builds, no moves,
   no seize attempts). If shaping is large but terminal is bad,
   exploration is happening but the agent can't finish.
4. **Look at `action_counts` over time.** A flat `end_turn` band at
   100% is mode collapse. Healthy training has `create_unit` early,
   transitioning to `move` and eventually `seize`.
5. **Watch `train/explained_variance` in TensorBoard.** Below 0.1
   means the value function is barely fitting; below 0.0 means it's
   actively wrong. That blocks all PPO gradient signal.

## When PPO genuinely cannot learn the task

If after collapsing back to "random opponents only" the agent still
doesn't reach a winning trajectory by chance within a few hundred k
steps, the next escalation is **not** more entropy or fancier reward
shaping — it's **behaviour cloning** from scripted demonstrations.

The pattern: write a demo generator that programmatically plays a
winning episode (build a Warrior, walk it to the enemy HQ, call seize
×4), record the (obs, action) pairs, and run a few epochs of
supervised learning on the policy network before starting PPO. After
BC the policy already "knows" the build → move → seize pattern; PPO
then refines under reward signal. The `imitation` library
(sb3-companion) integrates with SB3 policies cleanly.

We didn't ship this — the cheaper alternative (drop noop stages, use
random opponents) worked. But it's the next lever to reach for if
sparse-reward exploration genuinely fails.

## Things that are **not** worth chasing again

- **Bigger entropy** as a primary fix. We tried `ent_coef=0.10` (and
  earlier prototypes that pushed higher). Useful as a one-shot bump
  on map transitions, useless against the dead-policy trap. The
  problem there isn't "policy distribution too peaky," it's "no
  gradient signal at all."
- **Bigger per-action rewards** as a primary fix. We bumped
  `create_unit` 10 → 30 → 30 and `move` 0.5 → 5. The agent never
  responded. The issue isn't reward magnitude; it's reward
  *differentiation* across rollouts (which requires opponent
  variance).
- **Negative `end_turn` rewards.** Reintroduces a constant baseline
  bias in the opposite direction and doesn't help the underlying
  cold-start problem.
- **More training steps**. Stalls don't unstall by waiting. If the
  policy hasn't shifted in 250k steps, it won't shift in 2.5M.

## Future work

- A **CNN features extractor** over the `grid` and `units` channels
  would probably accelerate convergence, especially as we move to
  bigger maps. Mid-effort (~80 LOC).
- **Drop `action_mask` from the observation dict for MaskablePPO**
  via a small custom features extractor. It's redundant with
  MaskablePPO's masking pipeline and currently consumes ~70% of the
  network's input dimensionality. Note: AZ and Feudal RL pipelines
  *do* use the obs-mask and would break if we removed it globally —
  has to be a per-policy feature extractor, not an env change.
- **Behaviour-cloning warm-up** as described above, if the random-
  opponent curriculum fails on a future bigger map.
- **`start_fresh` flag on `CurriculumStage`** to optionally
  reinitialise the model at a stage boundary instead of always
  reusing weights — useful when a map shift is too steep for the
  starter policy to bridge.
