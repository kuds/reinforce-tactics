# JEPA × Reinforce Tactics — Design Document

**Status:** Design only. No code yet. This document is the single artifact produced
by the first pass of the JEPA exploration. It (a) audits the existing repository,
(b) honestly evaluates whether a Joint-Embedding Predictive Architecture is the
right tool for this game, and (c) describes a concrete design that we would
implement *if* we decide to proceed.

**Branch:** `claude/jepa-reinforcement-learning-UXPcr`

**Audience:** the Reinforce Tactics maintainers and any future collaborator
working on representation learning, hierarchical RL, or model-based RL in this
repo. Equations render in MathJax.

---

## 1. TL;DR

- The repo's observation is a structured $(H, W, 6)$ grid plus a 6-dim global
  vector, with maps as small as $6 \times 6$. This is **not** the high-dimensional
  noisy-pixel regime where JEPA's "predict embeddings, not pixels" advantage is
  decisive. A small CNN (already implemented in
  `reinforcetactics/rl/feudal_rl.py:18` as `SpatialFeatureExtractor`) gets you
  most of the representational capacity you need.
- The current Feudal RL pain points — documented in
  [`docs/feudal_rl_review.md`](../feudal_rl_review.md) — are **not**
  representation-learning problems. They are: shared optimizer interference, no
  action masking in the worker, no reward normalization, no multi-env support,
  no LR schedule. JEPA does not fix any of these.
- The **fog-of-war-as-masking** framing is the strongest standalone research
  contribution we identified: the existing `visibility` observation channel
  (0/1/2 per cell, see `reinforcetactics/rl/gym_env.py:159`) is a natural
  ground-truth mask for a JEPA-style predictor. "Predict the embeddings of
  tiles you can't see" is a clean self-supervised objective that maps directly
  onto the partial-observability structure of the game.
- The MuZero-flavored latent dynamics angle is **the weakest** of the proposed
  uses. The simulator is fast and deterministic, so the speed and
  reconstruction-cost arguments that motivate MuZero in Atari/Go don't apply
  here. We recommend deferring it.
- **Recommendation:** if we proceed, scope the first iteration to (i) spatial
  JEPA pretraining with fog-of-war-shaped masking as the headline ablation,
  (ii) a `JEPAFeudalAgent` that uses the pretrained encoder as a frozen
  feature extractor (not as a continuous goal space), and (iii) tournament
  registration so results are credible. We should also run two strong
  baselines we expect to be competitive: end-to-end PPO with the
  `feudal_rl_review.md` fixes, and behavior cloning from MCTS policies.

---

## 2. Codebase Audit

The numbers in this section are the ones any later design decision must
respect. File paths are verified against `HEAD`.

### 2.1 Observation space

Defined in `reinforcetactics/rl/gym_env.py:150`:

```python
spaces.Dict({
    "grid":            Box(0, 255, (H, W, 3), float32),
    "units":           Box(0, 255, (H, W, 3), float32),
    "global_features": Box(0, 10000, (6,),   float32),
    "action_mask":     Box(0, 1, (action_space_size,), float32),
    # only present when fog_of_war=True:
    "visibility":      Box(0, 2, (H, W),     uint8),
})
```

Per-channel meaning (from `reinforcetactics/rl/observation.py`):

| Tensor | Channels | Meaning |
|--------|----------|---------|
| `grid[:, :, 0]` | 1 | Terrain type (0–9 enum: grass, forest, mountain, road, water, ocean, building, tower, HQ, …) |
| `grid[:, :, 1]` | 1 | Tile owner (0 = neutral, 1 = player, 2 = opponent) |
| `grid[:, :, 2]` | 1 | Structure HP (0–255) |
| `units[:, :, 0]` | 1 | Unit type (W=0, M=1, C=2, A=3, K=4, R=5, S=6, B=7) |
| `units[:, :, 1]` | 1 | Unit owner (0 = empty, 1 = self, 2 = enemy) |
| `units[:, :, 2]` | 1 | Unit HP percent (0–100) |
| `global_features` | 6 | `[own_gold, opp_gold, turn, own_units, opp_units, current_player]` |
| `visibility` | 1 | 0 = unexplored, 1 = shrouded (fog), 2 = currently visible |

**Map sizes encountered.** `maps/1v1/` contains everything from `6x6_beginner.csv`
up to ~$20 \times 20$. The Gym env reads `H = self.game_state.grid.height` and
`W = self.game_state.grid.width` at `__init__` and uses them throughout.
Implication: **any JEPA encoder must handle variable $(H, W)$**, ideally by
keeping per-cell processing translation-invariant and using a positional
embedding indexed up to a fixed `max_grid_size` (e.g. $32 \times 32$).

**Channel scale.** Channels are stored as `float32` in the range $[0, 255]$ but
represent integer enums or HP. They are **not** normalized to $[0, 1]$ at the
env boundary. A JEPA encoder should normalize internally (divide by 255 or by
the per-channel max) or use a learned per-channel embedding for the categorical
channels (terrain type, unit type, owners) — these are 8-way categoricals, not
intensities, and treating them as continuous floats throws away structure.

### 2.2 Action space

Three modes (`reinforcetactics/rl/gym_env.py:169`):

| Mode | Space |
|------|-------|
| `multi_discrete` (default) | `MultiDiscrete([10, 8, W, H, W, H])` — `(action_type, unit_type, from_xy, to_xy)` |
| `flat_discrete` | `Discrete(max_flat_actions)` — flat index over a per-step legal-action list |
| `hierarchical` | `Dict(goal=Discrete(64), primitive=MultiDiscrete([10, 8, W, H, W, H]))` |

AlphaZero (`reinforcetactics/rl/alphazero_net.py:142`) outputs logits over a
fixed flat space of size $10 \cdot W \cdot H$ where the index is
`action_type * W * H + y * W + x`. This is the canonical "global" action
space for the repo.

**The action space is not auto-regressive in the AlphaStar sense** — there is no
sequential decomposition of `unit_type | action_type, from_xy | unit_type, ...`.
The MultiDiscrete decoding samples each dimension independently and the env
silently rejects invalid combinations (`reward_config["invalid_action"] = -10.0`).
This is good news for JEPA: an action-conditioned predictor only needs to embed
a single 6-tuple per step, not unroll a per-token decoder.

Action masking is exposed in two ways (`reinforcetactics/rl/gym_env.py:410`):
per-dimension boolean masks for MaskablePPO (over-approximation of legality), or
a single flat boolean mask in `flat_discrete` mode (exact). **The Feudal worker
currently ignores both** (see `feudal_rl_review.md` issue #2) and will sample
illegal actions during training.

### 2.3 Feudal RL scaffold

`reinforcetactics/rl/feudal_rl.py`. Key classes and their integration points:

- `SpatialFeatureExtractor` (`feudal_rl.py:18`): 3-layer CNN over the 6-channel
  spatial obs, concatenated with `global_features`, projected to a 512-dim
  feature. **This is the natural drop-in replacement target for a JEPA
  encoder.**
- `ManagerNetwork` (`feudal_rl.py:90`): MLP head producing **discrete** goals
  $(g_x, g_y, g_t)$ where $g_x \in [0, W)$, $g_y \in [0, H)$,
  $g_t \in \{0, 1, 2, 3\}$ for `{attack, defend, capture, expand}`.
  Goals are sampled from independent categoricals.
- `WorkerNetwork` (`feudal_rl.py:197`): conditions on goal via **concatenation**
  (not FiLM, not cross-attention). The goal $(g_x, g_y, g_t) \in \mathbb{R}^3$ is
  embedded by a 2-layer MLP to 64-dim, concatenated with the 512-dim feature,
  then fed to per-action-dimension Categorical heads.
- `compute_intrinsic_reward` (`feudal_rl.py:914`): goal-type-conditioned shaped
  reward based on Manhattan distance from the closest own unit to the goal,
  plus type-specific bonuses (proximity to enemies for `attack`, on-tile for
  `defend`, on-structure for `capture`, unit density in radius for `expand`).
  **It is not the FuN-style cosine-similarity reward** that JEPA goal embeddings
  would naturally use.
- `FeudalRolloutBuffer` (`feudal_rl.py:345`): stores worker steps and manager
  segments separately, with segment-length-aware GAE in `_compute_gae`.
  Reusable as-is for any JEPA-conditioned variant.

**Therefore there are two clean integration points** for a JEPA-derived
representation:

1. **Frozen feature extractor.** Replace `SpatialFeatureExtractor` with a frozen
   pretrained JEPA encoder; the rest of the Feudal training loop is unchanged.
   Lowest-risk option, smallest delta, easiest ablation.
2. **JEPA-embedding goal space.** Replace the discrete $(g_x, g_y, g_t)$ goal
   with a continuous $D$-dim embedding produced by the manager and consumed by
   the worker via concat or FiLM, with cosine-similarity intrinsic reward
   between the worker's reached-state embedding and the manager's goal vector.
   This is the deeper change. It is what the original brief was asking for, but
   it has the highest "what if it just makes things worse" risk because we lose
   the discrete-goal interpretability and inherit the Feudal review doc's open
   issues.

### 2.4 AlphaZero / MCTS

`reinforcetactics/rl/alphazero_net.py:39` defines a small ResNet (Conv → 6
ResBlocks @ 128 channels → policy & value heads). Forward signature is
`(grid, units, global_features) -> (policy_logits, value)` over a flat action
space of size $10 \cdot W \cdot H$. Self-play examples
(`alphazero_trainer.py:38`) are 6-tuples
`(grid, units, global_features, action_mask, mcts_policy, value_target)`. The
replay buffer is a `deque` and is sampled uniformly at training time.

**Implication for any "MuZero-with-JEPA" extension:** we already have the data
format and the MCTS skeleton. The minimum change to go MuZero-style would be to
add a `latent_state -> (latent_state, reward)` predictor and route MCTS rollouts
through it. **We're not recommending we do this in the first iteration**; it is
a stretch goal at best, for reasons in §3.

### 2.5 Tournament / ELO

`reinforcetactics/tournament/bots.py` defines `BotType` (an enum) and
`BotDescriptor` (a dataclass with factory methods). A new bot type requires:

1. Adding `JEPA = "jepa"` (or similar) to the `BotType` enum.
2. Adding a `BotDescriptor.jepa_bot(name, model_path, ...)` factory.
3. Adding a `BotType.JEPA` arm to `create_bot_instance` that imports and
   instantiates the JEPA bot class.
4. Adding a `discover_jepa_bots(jepa_dir)` function and wiring it into
   `discover_all_bots`.

`TournamentRunner` (`tournament/runner.py:37`) consumes `List[BotDescriptor]`
and runs round-robin schedules; ELO updates are handled by
`tournament/elo.py`. **No deeper changes are needed in the tournament system
itself** — the JEPA bot just has to expose the same `take_turn(self)`
interface that `ModelBot` already does (`game/model_bot.py:94`).

### 2.6 Self-play data format

`alphazero_trainer.self_play_game` (`alphazero_trainer.py:69`) returns
`(examples, winner)` where each example is the 6-tuple above. **Trajectories
themselves — i.e. ordered sequences of (state, action, next_state) tuples —
are not currently persisted to disk in any structured format.** For temporal
JEPA we would need either:

- A new collection script that dumps `(grid_t, units_t, global_t, action_t)`
  sequences (one file per game) in NPZ or HDF5 form, or
- An `AlphaZeroTrainer` extension that emits trajectories as a side product.

The cleanest path is the new collection script — it doesn't perturb the
existing AlphaZero training loop and lets us collect from any policy
(rule-based bot, random, AlphaZero, mixed) without entanglement.

---

## 3. Is JEPA the Right Tool Here?

### 3.1 Why JEPA on natural images works

Standard I-JEPA / V-JEPA make four bets:

1. **Pixels are noisy and locally redundant**, so reconstruction-based SSL
   wastes capacity on photorealistic detail.
2. **Targets in embedding space** sidestep this and let the model focus on
   semantically meaningful structure.
3. **Masked prediction** is a strong pretext task because the masked region
   has plausible alternatives that share semantics with the visible region.
4. **EMA targets** prevent representation collapse without negatives.

### 3.2 How those bets transfer to Reinforce Tactics

| Bet | Transfers to a $(H, W, 6)$ enum-coded grid? |
|-----|---------------------------------------------|
| Pixels are noisy / redundant | **No.** Channels are clean integer enums. There is nothing to denoise. |
| Reconstruction wastes capacity | **No.** A reconstruction loss on 6 enum channels is cheap and well-defined. |
| Masked prediction is informative | **Partially.** The fog-of-war structure makes "predict masked tiles" *very* natural — but the same is true for an MAE-style reconstruction loss. JEPA is not uniquely positioned here. |
| EMA prevents collapse | **Yes**, but so do contrastive methods. Not a JEPA-specific advantage. |

The dominant pitch reduces to: **"JEPA-as-masked-prediction is a clean SSL
pretext that respects the partial-observability structure of the game."** That
is true — but it does not necessarily mean JEPA beats MAE / SimSiam / a
forward-dynamics world model on the metrics we actually care about (ELO,
sample efficiency).

### 3.3 What problem are we actually trying to solve?

There are at least three distinct goals that have been conflated:

(a) **Better representations for the manager.** The current Feudal manager
emits discrete spatial goals because they're easy to learn from sparse reward.
The hypothesis is that a continuous goal space pretrained on game trajectories
would let the manager express richer subgoals.
*Counterargument:* the limit on Feudal performance right now is engineering
(see review doc), not goal expressiveness. Discrete spatial goals are a
strong inductive bias for a tactical grid game where "go to (x, y) and do
X" really is the primitive of strategic play.

(b) **A pretraining warm start for the worker.** Pretrain on offline self-play
data, fine-tune with PPO. This is a defensible sample-efficiency play.
*Counterargument:* AlphaZero already produces `(state, MCTS-policy)` pairs.
**Behavior cloning on those pairs is a stronger warm start** than any
self-supervised objective — it directly transfers MCTS's search-improved
policy into the network.

(c) **A novel ICML angle.** "JEPA on a discrete-domain grid game with
fog-of-war as the natural mask" is a defensible contribution.
*Counterargument:* it is a narrow contribution. The paper would live or die on
the fog-of-war ablation, not on raw ELO improvements.

**My honest read:** goal (c) is the most credible motivation; goal (b) is
better-served by BC-from-MCTS; goal (a) is unproven and the original brief
overstates its likelihood of success.

### 3.4 Strongest alternatives we should A/B against, not skip past

If we go ahead, the experiments must include these baselines, otherwise the
results are not interpretable:

1. **Behavior cloning on AlphaZero MCTS policies, then PPO-finetune.** Same
   compute budget, same final policy size. This is the obvious "warm start"
   competitor.
2. **End-to-end Feudal PPO with the `feudal_rl_review.md` issues fixed**
   (action masking, separate optimizers, reward normalization, multi-env).
   This is what the *current* Feudal scaffold should be, and it's the real
   "no pretraining" baseline — not the buggy version that exists today.
3. **MAE-style reconstruction pretraining** with the same masking schedule.
   This is the direct ablation that isolates "JEPA's embedding-target trick"
   vs "any masked-prediction pretext".

If JEPA does not beat (1), (2), and (3) reliably across seeds, the
"JEPA-helps-Feudal" claim does not hold. The fog-of-war ablation can still be
interesting on its own as a representation-learning study even if downstream
RL gains are flat.

---

## 4. Proposed JEPA Design (If We Proceed)

This is what we would build in iteration 1. Bias toward small, modular,
testable; no `[jepa]` install extra needed yet (only `torch`, `numpy`,
`einops` if we want it).

### 4.1 The encoder

A small ViT-style patch encoder treating each grid cell as one patch. Per-cell
input has 6 channels (3 grid + 3 units), optionally augmented with the
visibility channel.

- **Tokenization.** Each cell $(y, x)$ becomes a token. Categorical channels
  (terrain type, unit type, owner-of-tile, owner-of-unit) are passed through
  small learned embedding tables (e.g. $\text{Embed}(10, d_e)$ for terrain),
  and continuous channels (structure HP, unit HP) are linearly projected.
  All concatenated and mapped to dim $D$ via a small linear layer.
- **Positional embedding.** Learned 2D embedding indexed by $(y, x)$ with a
  fixed `max_grid_size = 32`. For variable-size maps, only the prefix
  $H \times W$ is used.
- **Global token.** `global_features` (gold, turn, etc.) are projected to dim
  $D$ and prepended as a `[GLOBAL]` token, à la `[CLS]`.
- **Backbone.** $L = 4$ transformer blocks, $D = 128$, $H_{\text{heads}} = 4$,
  MLP ratio 4. About 1M params.

We deliberately keep this **smaller than the AlphaZero ResNet** (~3M params)
so that any improvements are not just "we made the network bigger." Target:
under 5M params for the JEPA encoder + predictor combined, per the original
brief.

### 4.2 Masking

Three masking strategies, selectable at training time:

1. **Random rectangular masking** (I-JEPA style): sample $K \in [3, 5]$
   non-overlapping rectangles covering 40–60% of cells. The visible cells are
   the *context*, the masked cells are the *target*.
2. **Random per-tile masking**: each cell masked independently with probability
   $p \in \{0.15, 0.5, 0.75\}$ (the ablation knob).
3. **Fog-of-war-shaped masking**: for observations with `visibility` available,
   cells with `visibility < 2` are masked. This mirrors the partial
   observability the agent actually faces during play.

A `MaskingStrategy` interface keeps these swappable. Mask is a boolean tensor
of shape $(H, W)$; tokens at masked positions are replaced with a learned
`[MASK]` embedding before being fed to the predictor. The encoder always sees
the *full* unmasked observation when producing targets (inside `torch.no_grad`
on the EMA copy), and only sees the *visible* cells when producing context.

### 4.3 The predictor

A small transformer (depth 2, dim 128) that takes:

- The encoder's context tokens (visible cells + `[GLOBAL]` token)
- Mask-position tokens with their positional embeddings
- (Temporal variant only) An action embedding per step

It predicts the EMA-target encoder's embeddings at the masked positions. The
loss is smooth-L1 between predicted and target embeddings over the masked set:

$$
\mathcal{L}_{\text{spatial}} = \frac{1}{|\mathcal{M}|} \sum_{(y, x) \in \mathcal{M}} \|\hat{z}_{y,x} - \mathrm{sg}(z_{y,x}^{\text{EMA}})\|_{\text{smooth-L1}}
$$

where $\mathcal{M}$ is the masked set and $\mathrm{sg}$ is stop-gradient.

For the **temporal variant**, the predictor takes context tokens from frame
$t$ and an action $a_t$, and predicts the EMA encoder's embeddings at frame
$t + k$ for some $k \geq 1$:

$$
\mathcal{L}_{\text{temporal}} = \mathbb{E}_{(s_t, a_t, s_{t+k})} \left[ \frac{1}{HW} \sum_{(y, x)} \|\hat{z}^{t+k}_{y,x} - \mathrm{sg}(z^{t+k, \text{EMA}}_{y,x})\|_{\text{smooth-L1}} \right]
$$

The action embedding is just `Embed(num_action_types) + Embed(num_unit_types)
+ project([from_x, from_y, to_x, to_y])` — a single 128-dim vector per step.
We do not unroll an auto-regressive decoder; one-shot $k$-step prediction is
sufficient for the manager's planning horizon (which is also $\sim 10$ steps
in the existing `manager_horizon`).

### 4.4 EMA target updates

Standard exponential moving average:

$$
\theta^{\text{EMA}} \leftarrow \tau \cdot \theta^{\text{EMA}} + (1 - \tau) \cdot \theta
$$

with $\tau$ scheduled from $0.996 \to 1.0$ linearly across pretraining. The
target encoder is in eval mode and never sees gradients.

### 4.5 What we explicitly are NOT doing

- **No image augmentations.** This is not an image domain; flips/crops would
  break the game's spatial semantics (a flipped HQ is a different game state).
  Symmetry-aware augmentation (e.g. swap player 1 / player 2 channels) might
  be worth a future ablation but is not in iteration 1.
- **No multi-block / multi-target prediction** as in I-JEPA's full recipe.
  Single context-target split keeps the loss interpretable for the first pass.
- **No latent-space MCTS rollouts.** This is the MuZero stretch goal, deferred.

---

## 5. Two Downstream Uses

### 5.1 Feudal RL — frozen-encoder variant (Iteration 1)

The minimum-risk integration: train JEPA as in §4, then use the encoder as a
**frozen** drop-in replacement for `SpatialFeatureExtractor` inside the
existing `FeudalRLAgent`. Manager output stays discrete $(g_x, g_y, g_t)$;
intrinsic reward stays the existing distance-based shaped reward.

- File: `reinforcetactics/jepa/feudal_integration.py` (planned).
- Class: `JEPAFrozenFeatureExtractor(BaseFeaturesExtractor)` — wraps a loaded
  JEPA encoder, freezes its parameters, exposes the same `(B, features_dim)`
  output contract as `SpatialFeatureExtractor`.
- Touches `feudal_rl.py` only via dependency injection (the agent constructor
  already takes an `observation_space`; we'd add an optional
  `feature_extractor` kwarg).
- **Ablation A:** SpatialFeatureExtractor (current baseline, with
  `feudal_rl_review.md` fixes applied).
- **Ablation B:** Frozen JEPA encoder (this section).
- **Ablation C:** JEPA encoder fine-tuned with the policy gradient.

This isolates the question: **does pretraining the encoder help downstream
RL?** without simultaneously perturbing the goal space.

### 5.2 Feudal RL — JEPA-goal-space variant (Iteration 2, if 5.1 looks good)

Only worth doing if Iteration 1 shows a real signal. This is the design from
the original brief:

- Manager outputs a continuous goal $g \in \mathbb{R}^D$ in the JEPA
  embedding space (a linear head off the encoder features, normalized to the
  unit sphere).
- Worker is conditioned on $(z, g)$ via concatenation (matches existing
  worker conditioning style; FiLM is a follow-up ablation).
- Intrinsic reward: cosine similarity between the worker's reached-state
  embedding $z_{t+k}$ and the manager's goal direction $g$:
  $$
  r^{\text{int}}_t = \cos\!\big(z_{t+k}^{\text{EMA}}, g_t\big).
  $$
  This is the FuN-style reward that the brief asked for, and it requires the
  embedding to be meaningfully geometric — which is exactly what the JEPA
  pretraining is supposed to produce.

The risk in Iteration 2 is that we lose the discrete-goal interpretability
without gaining downstream ELO. We will only commit to this iteration if (a)
Iteration 1 frozen-encoder Feudal beats end-to-end Feudal across seeds, **and**
(b) probing experiments (k-NN classification of game outcomes from
embeddings) show the JEPA embedding is meaningfully structured.

### 5.3 MuZero-flavored latent dynamics — deferred

For all the reasons in §3.2: the simulator is fast, the observation is small,
the reconstruction-cost argument is weak. We document the design here so it's
not lost, but we don't recommend implementing it before Feudal results are in.

If we ever do: add `(latent_z, action) -> (latent_z', reward, value, policy)`
heads on top of the JEPA encoder, train with the standard MuZero objectives
plus the JEPA representation loss, and route MCTS through latent rollouts
instead of `GameState.deepcopy`. The expected wallclock win is small because
the simulator is already cheap.

---

## 6. Evaluation Protocol

We evaluate on two axes: **representation quality** (intrinsic to the JEPA
training) and **downstream RL** (extrinsic, the only thing that matters for
the ICML claim).

### 6.1 Representation-quality probes (intrinsic)

These do not require RL and are cheap. They are also what we'd publish even
if RL gains are flat.

- **Masked-tile prediction loss** on a held-out validation split of
  trajectories.
- **Linear probe** on frozen embeddings predicting the eventual game outcome
  from a mid-game state. Reports accuracy / log loss vs. baseline encoders
  (random init, `SpatialFeatureExtractor` from a trained PPO agent).
- **Linear probe for unit count / gold delta** — checks whether the embedding
  preserves task-relevant scalars.
- **Fog-of-war probe** (the headline experiment): how well does the predictor
  recover a fogged-out unit's `(unit_type, owner)` from context alone?
  Compare random-mask training vs. fog-shaped-mask training — does training
  on fog-shaped masks transfer to other masking distributions?

### 6.2 Downstream RL (extrinsic)

The original brief listed four conditions; we keep them, with the caveat that
we add the alternatives from §3.4 as additional baselines:

| Tag | Condition |
|-----|-----------|
| **D** | Flat MaskablePPO (current baseline) |
| **A** | Feudal RL, current scaffold (no pretraining) |
| **A'** | Feudal RL with `feudal_rl_review.md` fixes (the *real* baseline) |
| **B** | Feudal RL with frozen JEPA encoder (Iteration 1 of §5.1) |
| **C** | Feudal RL with JEPA-pretrained-then-fine-tuned encoder |
| **+BC** | Feudal RL with BC-from-MCTS warm start (§3.4 baseline) |
| **+MAE** | Feudal RL with MAE-pretrained encoder, same masking schedule (§3.4 baseline) |

**Metric:** ELO from round-robin tournaments using
`reinforcetactics/tournament/runner.py:TournamentRunner` against a fixed bot
pool: `SimpleBot`, `MediumBot`, `AdvancedBot`, plus the strongest existing
trained model. Three seeds minimum. Map suite: at least three maps per format
($6 \times 6$, $10 \times 10$, $20 \times 20$ from `maps/1v1/`).

**No reinventing the tournament runner.** All evaluation must go through the
existing infrastructure so the numbers are comparable to numbers people
already trust.

### 6.3 Ablations (compute-permitting)

- Masking ratio sweep $\{0.15, 0.50, 0.75\}$ for spatial JEPA.
- With vs. without action conditioning for temporal JEPA.
- Goal-embedding dim $\{32, 64, 128, 256\}$ for the §5.2 manager.
- Random-mask vs. fog-shaped-mask pretraining, evaluated by both
  representation probes (§6.1) and downstream RL (§6.2).
- Encoder size $\{1\text{M}, 3\text{M}, 5\text{M}\}$ params — to verify any
  gains are not just "we used more compute."

---

## 7. Risks & Stop Conditions

We commit in advance to the following stop conditions so we don't sink
indefinite compute into a negative result:

- **R1: JEPA pretraining loss does not converge below a random-mask
  reconstruction baseline within 24h of training on a single A100.** Stop and
  re-examine the encoder/masking design.
- **R2: Linear probe on game-outcome prediction from JEPA embeddings is no
  better than random-init or PPO-trained `SpatialFeatureExtractor`.** Stop;
  the embeddings aren't capturing task-relevant structure and there's no
  reason to expect downstream RL gains.
- **R3: Frozen-JEPA Feudal (B) does not beat end-to-end Feudal (A') across 3
  seeds at 5M timesteps.** Stop; do not advance to §5.2.
- **R4: BC-from-MCTS (+BC) beats both Frozen-JEPA (B) and Fine-tuned-JEPA (C)
  at the same compute.** Then the JEPA pretraining isn't pulling its weight as
  a warm start; rewrite the paper to be about the fog-of-war representation
  ablation, not about RL gains.

---

## 8. Open Questions Resolved by This Audit

The original brief listed four questions. Status after the audit:

| Question | Resolved? | Answer |
|----------|-----------|--------|
| What is the current observation tensor shape? Is it amenable to patch encoding? | **Yes.** | Dict obs, spatial tensors of shape $(H, W, 3)$ each for grid and units, plus 6 globals, optional $(H, W)$ visibility. Trivially patch-encodable as one token per cell. Variable $(H, W)$ requires either dynamic positional embedding or a fixed `max_grid_size`. |
| How does the existing Feudal scaffold define manager outputs and worker conditioning? Where does a JEPA goal vector plug in? | **Yes.** | Manager outputs discrete $(g_x, g_y, g_t)$; worker conditions via 64-dim goal embedding concatenated with 512-dim feature. JEPA goal vector plugs in by replacing the manager's three categorical heads with a single continuous head into $\mathbb{R}^D$, replacing the worker's `Linear(3, 64)` goal embedder with a goal-projection MLP, and replacing `compute_intrinsic_reward` with cosine similarity. |
| Is the action space auto-regressive (per AlphaStar)? | **Yes.** | No, it is not. MultiDiscrete with independently-sampled dimensions plus env-side rejection of invalid combinations. A temporal predictor only needs to embed a single 6-tuple per action, no per-token decoder. |
| How much self-play data exists, and is it diverse enough? | **Partially.** | No persistent self-play trajectory dump exists; AlphaZero stores only flattened `(obs, mcts_policy, value)` tuples in an in-memory deque. We will need a `scripts/collect_jepa_data.py` (planned) that saves ordered $(s_t, a_t)$ sequences — collected from a mix of `SimpleBot` self-play and trained-PPO self-play to maximize diversity without requiring exploration policy work upfront. |

Still open after this audit:

- **Q5: How sensitive is JEPA to the categorical-channel encoding choice?**
  Embedding tables vs. one-hot vs. raw float. We'll pick embedding tables for
  iteration 1 (matches the categorical structure) but it's worth ablating.
- **Q6: Should the predictor be conditioned on the visibility channel itself
  during training?** Probably yes for the fog-of-war variant — without it,
  the predictor cannot distinguish "tile is masked because random-mask" from
  "tile is masked because fog-of-war."
- **Q7: For variable map sizes, do we train one encoder per map size, or a
  single encoder with positional masking up to `max_grid_size`?** Prefer the
  latter (transfer is the whole point), but it needs explicit testing on the
  smallest maps.

---

## 9. Phased Implementation Plan (If We Proceed)

Each phase is a distinct PR. Stop after any phase if its acceptance criteria
fail.

### Phase 0 — design (this document)

- ✅ DESIGN.md committed.
- 🔲 Decision: proceed to Phase 1, narrow scope to fog-of-war study, or pivot
  to a different approach.

### Phase 1 — JEPA encoder + spatial pretraining

Files: `reinforcetactics/jepa/{__init__.py, encoder.py, masking.py,
predictor.py, ema.py, dataset.py, train_jepa.py}`,
`scripts/collect_jepa_data.py`, `tests/test_jepa_*.py`,
`docs/jepa/RESULTS.md` (skeleton), `[jepa]` extra in `pyproject.toml`.

**Acceptance:** R1 (loss converges) + R2 (linear probe beats random-init).

### Phase 2 — Frozen-JEPA Feudal integration

Files: `reinforcetactics/jepa/feudal_integration.py`,
`train/train_feudal_jepa.py`, additional tests.

**Acceptance:** R3 (beats end-to-end Feudal with review-doc fixes applied).

### Phase 3 — Tournament registration + ablations

Files: extend `reinforcetactics/tournament/bots.py` with `BotType.JEPA`,
add `discover_jepa_bots`, write the `notebooks/jepa_comparison.ipynb`,
populate `docs/jepa/RESULTS.md` with tables, write `docs/jepa/ICML_NOTES.md`.

**Acceptance:** JEPA agent is a first-class tournament participant; results
table reproduces with `pytest tests/test_jepa_* && python
scripts/tournament.py --include-jepa`.

### Phase 4 (optional) — Continuous goal-space variant (§5.2)

Only if Phase 2 acceptance criterion clears decisively across seeds.

### Phase 5 (deferred) — MuZero-flavored latent dynamics (§5.3)

Only if there's a specific reason to believe latent rollouts beat
`GameState.deepcopy` rollouts on this game. Current evidence suggests they
won't.

---

## 10. What This Document Is Not

- It is **not** a paper draft. The ICML write-up will live in
  `docs/jepa/ICML_NOTES.md` once we have results.
- It is **not** a defense of JEPA. §3 is the honest negative case.
- It is **not** a commitment to all of §9. We re-evaluate after each phase.

The intent is to produce one well-considered artifact before writing any code,
so that whoever does the implementation (or whoever reads this in six months)
understands both the design and the tradeoffs that were considered and not
silently elided.
