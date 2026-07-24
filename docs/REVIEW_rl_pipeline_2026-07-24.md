# RL pipeline review — 2026-07-24

Scope: `reinforcetactics/rl/` at `faac8c8`, all 65 YAML configs under `configs/`,
and the Drive run archive (`benchmarks/bootstrap/runs_summary.csv`,
`runs_per_stage.csv`, per-run `eval_results.json`). Every number below was
recomputed from the archive; every code claim carries a `file:line`.

This builds on `docs/REVIEW_ppo_training.md` (2026-07-12) rather than repeating
it. Section 5 states explicitly which of that review's recommendations landed
and which did not.

---

## TL;DR

The engineering around the pipeline is genuinely good — per-stage config
capture, incremental eval persistence, run metadata, stall traces, 75 bootstrap
tests. The blocker is not code hygiene. It is that **59 sweep configs varied
reward shaping, curriculum shape and entropy, and never once varied the
optimizer or the representation.** `gamma`, `learning_rate`, `n_steps`,
`batch_size`, `clip_range`, `pool` and the action-head design are byte-identical
across every run ever logged.

Five findings I would act on before running another sweep variant:

1. **The potential-based shaping charges the agent ~190 per episode for being
   ahead, while winning pays 50.** `(gamma-1)*Phi` per micro-action, ~1900
   micro-actions per episode. The archive's measured `shaping_delta` matches the
   arithmetic to within 0.3%. Symmetrically, a *losing* agent is paid to stall.
   That is a bistable incentive, and the archive shows exactly that bistability.
2. **Every gate decision is measured on the wrong object.** The eval runs
   `deterministic=True` (PPO optimizes the stochastic policy) against an
   **80-episode benchmark that is redrawn on every eval**. So `patience`
   consecutive crossings compares two different problem sets, and
   `best_model.zip` is an argmax over ~100 noisy estimates — the luckiest draw,
   not the best policy.
3. **The agent has never once won by HQ capture.** 480 eval episodes across the
   deepest run: `captures_by_type.hq == 0` in every single eval, 100% of wins
   `by_elimination`. The reward has been paying 80 for an HQ win vs 50 for
   elimination since v49. Paying more for something the policy never does is not
   a reward-tuning problem — it is a capability problem.
4. **`pool: masked_avg` compresses the whole board to 64 numbers** before the
   policy head (`extractors.py:189-195`). With `flat_discrete`'s positional
   action indices on top, the policy cannot express "move *this* unit to *that*
   tile". Every config in the repo uses `masked_avg`.
5. **`gamma: 0.99` over ~2000-step episodes makes the terminal reward
   arithmetically invisible** (`0.99^2000 = 1.9e-9`). This is documented in the
   July 12 review and was never applied to a single config, including v54.

---

## 1. What the archive says, recomputed

| metric | value |
| --- | --- |
| usable runs | 56 of 109 run dirs |
| stalled stages | 41 |
| stalled stages that burned **>=95%** of their budget | **41 / 41** |
| stalled stages that **peaked at or above their gate** first | **28 / 41** |
| median final WR of those 28 | **0.106** |
| stalls ending with **positive avg reward at WR <= 0.2** | 18 / 41 |
| 33-stage curriculum budget | **108M env steps** |
| deepest run ever | **7.9M steps (7%)** |
| runs at `seed: 42`, n=1 | 50 / 56 |
| `multi_discrete` runs, peak WR | 3 / 3 at **0.00** |

The four rows flagged `completed_curriculum` are all 1/5/6-stage probe
curricula. **No run has ever finished the real 33-stage ladder**, and 13 of its
33 stages have never been attempted by anything.

### The collapse has a measured shape

From the deepest run's `eval_results.json` (six consecutive 50k-step evals):

| step | WR | avg_len | W built | M built | hq caps | truncated |
| --- | --- | --- | --- | --- | --- | --- |
| 5.25M | 1.00 | 1965 | 6650 | 0 | 0 | 0/80 |
| 5.30M | 0.60 | 2639 | 6877 | 2225 | 0 | 27/80 |
| 5.35M | 0.45 | 2698 | 3246 | 5773 | 0 | 38/80 |
| 5.40M | 0.21 | 2812 | **407** | **8989** | 0 | 42/80 |
| 5.45M | 0.75 | 2474 | 7984 | 4 | 0 | 17/80 |
| 5.50M | 0.99 | 1773 | 6791 | 0 | 0 | 1/80 |

This is not gradual drift. It is a **bistable flip in build order** — Warriors
to Mages and back — inside 150k steps, with win rate following it. The policy
has two modes and the optimizer keeps knocking it between them.

Reward mass per 80-episode eval: `action` **+26,011**, `shaping_delta`
**-15,162**, `terminal` **+4,000**. The terminal signal is under 10% of the
total magnitude even before discounting.

---

## 2. Findings, ranked

### 2.1 The shaping term pays the agent to be behind and taxes it for being ahead

`reinforcetactics/rl/gym_env.py:1446-1456`

```python
if not terminal:
    current_potential = self._compute_potential()
    delta = self.gamma * current_potential - self._prev_potential
    reward += delta
```

The shaping is applied **per micro-action**, and `Phi` is a *level*, not a rate:

```python
potential += (structures_agent - structures_opp) * self.reward_config["structure_control"]
```
(`:1406-1409`; v52a sets `structure_control: 1.0`, `income_diff: 0.05`,
`unit_diff: 0.0`)

When the board is quiet, `Phi(s') == Phi(s)` and the delta is
`(gamma - 1) * Phi = -0.01 * Phi` **every step**. So:

- **Agent ahead by 10 structures** -> `Phi = +10` -> **-0.1 per step**.
  Over the measured ~1900 steps/episode: **-190 per episode**.
- **Agent behind by 10 structures** -> `Phi = -10` -> **+0.1 per step**, i.e.
  **+190 for stalling to the clock while losing.**

Against `win: 50.0`. **Holding a winning position costs ~3.8x what winning
pays.**

This is not a back-of-envelope estimate. The archive's `reward_components`
for the deepest run's WR-1.00 eval reads `shaping_delta: -15,162.5` over 80
episodes = **-189.5 per episode**. The predicted -190 and the measured -189.5
agree to 0.3%.

Two things make this bite rather than cancel out:

- **The terminal `-Phi(s_T)` is never charged.** Skipping the shaping on
  terminal steps (`if not terminal:`) is *not* the same as `Phi(terminal) = 0`;
  the telescoping sum is left with a dangling `+gamma^(T-1) * Phi(s_(T-1))`.
  Ng et al.'s policy-invariance guarantee formally does not apply.
- **GAE cannot see the compensation.** With `gamma=0.99, lambda=0.95` the
  advantage window is ~17 steps (see 2.3). The compensating terminal term sits
  ~1900 steps away. Locally, the only thing the optimizer sees is
  "shed material lead -> reward goes up".

That is a clean mechanistic account of every symptom in the archive: the
draw-machine attractor, positive average reward at WR <= 0.2 (18/41 stalls),
and the Warrior->Mage->Warrior bistability in the table above — a policy that
is ahead is pushed down, and a policy that is behind is pushed to stall.

**Fix**, cheapest first:
1. Charge the terminal step: on `terminal`, add `-self._prev_potential`
   (i.e. `F = gamma*0 - Phi(s_prev)`) instead of skipping. Restores invariance.
2. Apply shaping **once per game turn**, not per micro-action — the drain is
   proportional to step count, and step count is ~25x turn count.
3. Or make `Phi` bounded/normalized so a large lead cannot dominate the
   terminal reward.

### 2.2 Every gate decision is made on a policy PPO never optimizes, against a benchmark that is resampled each time

Two independent problems in `PeriodicEvalCallback._do_eval` (`callbacks.py:235-254`),
both of which land directly on promotion, `best_model.zip` and stall.

**(a) The eval measures the deterministic-argmax policy.**

```python
m = evaluate_model(
    self.model,
    self.eval_env,
    n_episodes=self.n_eval_episodes,
    seed=eval_seed,
    track_breakdown=self.track_breakdown,
    **eval_kwargs,
)
```

No `deterministic` argument is passed, and `evaluate_model`'s default is
`deterministic: bool = True` (`evaluation.py:97`). PPO optimizes the *stochastic*
policy; with `ent_coef` held at 0.05-0.10 for most of a stage those two policies
are far apart. Every promotion, every `best_model.zip` save, and every stall
verdict is therefore a measurement of an object the optimizer is not improving.

It also makes the metric brittle in exactly the observed way: argmax over a
positionally-shuffled action list (2.6) can flip wholesale on a small weight
change, which a stochastic policy would average over.

**(b) The 80-episode benchmark is redrawn on every eval.**

```python
eval_seed = self.eval_seed_base + 1000 * self._last_eval_block
```

with `_last_eval_block = num_timesteps // eval_freq` (`:229`) and per-episode
seeds `seed + ep_idx` (`evaluation.py:217`). Consecutive evals therefore draw
**completely disjoint** 80-seed problem sets. Two consequences:

- `patience` counts consecutive crossings of a gate whose *problem set changes
  between the two measurements* — it is measuring benchmark noise as if it were
  policy change.
- `best_model.zip` is `argmax` over ~30-100 noisy estimates on different
  benchmarks. That is a textbook winner's curse: the saved "best" is
  systematically the luckiest draw, not the best policy — and 2.8 then
  propagates that lucky checkpoint into the next stage.

**Fix**: pass `deterministic=False` for the promotion metric (or report both);
and hold the eval seed set **fixed** across a stage (`eval_seed_base` alone, or
a fixed seed block rotated only per stage) so consecutive evals are comparable.
Both are one-line changes and together they make the gate mean what it says.

### 2.3 `gamma: 0.99` puts the terminal outside the horizon — never varied

`configs/**/*.yaml` — **all 65 configs**, including the newest
`v54_uncapped_frontier.yaml:593`.

- One env step = one unit micro-action. Measured `avg_length` is **1965-2812
  env steps** (`max_steps: 3000`).
- Effective horizon `1/(1-gamma)` = **100 steps ~ 5 game turns**.
  `0.99^2000 = 1.9e-9`.
- GAE `lambda: 0.95` gives an advantage window of `1/(1-gamma*lambda)` =
  **17 steps** — about one game turn.

The draw attractor is therefore not a bug in the reward table; it is the
*correct optimum of the objective PPO can actually see*. Nine sweep variants
(v16, v27a-c, v34, v42, v44, v46, v49, v52a) attacked it from the reward side.
None touched the discount.

**Fix**: raise `gamma` to 0.997-0.999 (horizon 330-1000 steps) *and* cut steps
per episode — `max_actions_per_turn: 20-30` would take a 75-turn game from
~2000 steps to ~1500. Do both; gamma alone asks the value head to represent a
1000-step return over unnormalized rewards.

### 2.4 Truncation is charged the draw penalty *and* gets the time-limit bootstrap

`gym_env.py:1504-1506, 1573-1577`

```python
terminated = self.game_state.game_over
truncated  = self.current_step >= self.max_steps
...
elif truncated:
    # Truncation penalty: agent hit step limit without finishing the game.
    terminal_bonus = self.reward_config.get("draw", 0.0)      # -50
```

Returning `truncated=True` is the correct Gymnasium signal, and SB3's on-policy
rollout collector responds to it by adding `gamma * V(terminal_obs)` to that
step's reward (the `TimeLimit.truncated` path in `collect_rollouts`). The env
has *already* added `draw: -50` to the same transition.

So the value target for a truncated step is `-50 + gamma*V(s_T)` — the agent is
simultaneously told "this is a terminal draw, take the penalty" and "this is an
artificial cutoff, keep your future value". The two are contradictory, and the
bug fires on **27-42 of every 80 eval episodes** (2.10), so it is not a corner
case.

**Fix**: pick one semantic.
- If step-limit exhaustion *is* a draw by house rule, mirror the `max_turns`
  handling — set `terminated=True` (`:1529-1532` already treats a max-turns draw
  that way) so no bootstrap happens.
- If it is a genuine time limit, drop the `draw` bonus from the `elif truncated`
  branch and let the bootstrap carry the value.

### 2.5 The policy is spatially blind — never varied

`reinforcetactics/rl/extractors.py:189-195`

```python
elif self.pool == "masked_avg":
    live = self._live_cell_mask(grid)
    features = (features * live).sum(dim=(2, 3), keepdim=True) / denom
    features = features.flatten(1)
```

The CNN's `(B, 64, H, W)` output is averaged over live cells to **64 numbers**,
concatenated with 5 global features, and projected to 256. Every unit position,
HP, terrain tile and ownership bit on a 10x12 board arrives at the policy head
as 69 spatially-averaged scalars.

`extractors.py:46-50` already says `flatten` "preserves positional info" and is
"required when downstream heads consume per-cell features". `masked_avg` was
chosen for pad-invariance — but `pad_to_size` is resolved once per curriculum
(`bootstrap.py:626-629`), so the padded shape is *fixed* and `flatten` is safe.
At (10, 12) that is `64*10*12 + 5 = 7685` inputs, a 1.97M-parameter first
layer. Cheap.

**This is the best explanation on the table for `hq: 0`.** HQ capture requires
routing a specific unit to a specific tile and seizing. Elimination is
position-agnostic attrition. The policy does exactly the one it can represent.

**Fix**: run one variant with `pool: flatten`, everything else held at v52a.
It is a one-line config change and it tests the never-tested axis.

### 2.6 `flat_discrete` action indices are positional and rebuilt every step

`gym_env.py:1065-1069` and `_build_flat_actions` (`gym_env.py:247-302`)

Index `i` is "the i-th entry of the legal-action list rebuilt this step". That
list is built by iterating `self.units` (`core/game_state.py:1334`) and
`grid.get_capturable_tiles` (`:1327`). When a unit dies, `self.units.remove()`
shifts every index after it; when gold crosses a unit's cost, the whole
`create_unit` prefix changes length and shifts everything.

So the logit-to-action mapping is state-dependent and volatile *within* an
episode. Combined with 2.5 the policy cannot infer the enumeration from the
observation — it can only learn coarse regularities ("low indices are
create_unit, the last is end_turn"). That is a mechanical explanation for why
deterministic-argmax eval swings 1.00 -> 0.21 -> 0.99 across adjacent evals.

**Fix (larger)**: a pointer/per-cell action head — score each (unit, target)
pair from its own spatial feature vector — removes 2.5 and 2.6 at once.
`pool: flatten` is the prerequisite.

### 2.7 No learning-rate schedule reaches PPO at all

`reinforcetactics/rl/config.py:116` declares `lr_schedule: str = "constant"`,
but `:137` drops it:

```python
skip = {"use_action_masking", "lr_schedule", "purchase_explore_eps"}
```

with the comment "feudal-only (`lr_schedule`)". So the bootstrap/PPO path runs
a **constant 3e-4 for all 22M planned steps** and there is no supported way to
change that from a config. All 56 archived runs confirm `learning_rate: 0.0003`.

Given 2.1's bistability, a late-stage LR anneal is one of the cheapest possible
interventions — and the codebase already has the exact pattern to copy
(`callbacks.py:434-520`, `EntropyScheduleCallback`, which mutates a live model
attribute per stage and correctly handles `reset_num_timesteps=False` via
`_on_training_start`).

**Fix**: add an `LRScheduleCallback` mirroring `EntropyScheduleCallback`, and
either honour `lr_schedule` in `as_sb3_kwargs` or add
`CurriculumStage.learning_rate_schedule`.

### 2.8 `best_model.zip` is usually the *carry-in* policy, so promotion can undo the stage

`callbacks.py:224, 229-232, 324-331` interacting with `bootstrap.py:997-1013`

`PeriodicEvalCallback` is constructed fresh per stage with
`self._last_eval_block = -1`, but gates on the **cumulative** counter:

```python
block = self.num_timesteps // self.eval_freq
if block > self._last_eval_block:
```

The runner passes `reset_num_timesteps=False` (`bootstrap.py:841`), so at stage
entry `num_timesteps` is already in the millions and `block` is a large positive
number. **The first `_on_step` of every stage fires an eval immediately** — the
"first eval of every stage is at `t = stage_steps + 4`" already noted in
`docs/bootstrap_runs_review.md:67-70`.

That eval measures the **carry-in policy**, and it is fully eligible to become
the stage best (`best_win_rate` starts at `-1.0`, `:214`). So when the stage
later promotes and the runner does

```python
model.set_parameters(str(best_ckpt), exact_match=True)  # :1011
```

it can restore the policy *as it was at stage entry*, discarding everything the
stage trained. This is not hypothetical: the July 12 review measured a **median
time-to-clear of 50k steps — a single eval interval**. A stage that clears on
evals #1 and #2 with the carry-in policy scoring highest reverts to the carry-in
weights.

Stacked across a 20-stage run, the policy can advance most of the ladder while
absorbing far less training than the step counter implies — and then meet the
first genuinely hard stage under-trained. The runner already *measures* this
(`best_checkpoint_stage_steps`, `bootstrap.py:845-851`, and the "skip-ahead"
comment at `:1000-1003`); the measurement was added, the fix was not.

**Fix**: add `best_eligible_after` (stage-relative steps, default `eval_freq`)
to `PeriodicEvalCallback` — record the stage-entry eval as a *baseline* row but
skip the `model.save` for it. Two lines, and it makes
`restore_best_checkpoint_between_stages` mean what its name says.

### 2.9 A stall is run-fatal, and the good checkpoint is left on disk

`bootstrap.py:934-981`

On stall the runner writes `run_status.json`, saves the collapsed policy, and
raises `CurriculumStalled` — killing the whole run. The stage's
`best_model.zip` was **at or above the gate in 28 of 41 cases** and is
referenced in the exception (`:956-957`, added by `7fedf3e`) but nothing ever
loads it.

Meanwhile `restore_best_checkpoint_between_stages` (`:997-1013`) only runs on
the **promotion** path:

```python
if not promote_cb.promoted:
    ...
    raise CurriculumStalled(...)
# Stage promoted (the stall branch above raises).
if cfg.curriculum.restore_best_checkpoint_between_stages:
```

So the one mechanism that fixes drift is unreachable in exactly the case that
needs it. There is also no start-at-stage-K: with 108M steps of budget against
~6M per Colab session, the ladder cannot be finished by any amount of reward
tuning.

**Fix, in order**:
1. On stall, `model.set_parameters(best_model.zip)`, re-warm entropy, and retry
   the stage once before raising.
2. Add a **within-stage** regression guard: if WR drops more than X below the
   stage best for N consecutive evals, restore `best_model.zip` and continue.
   That directly converts the 28 peak-then-collapse stalls into promotions.
3. `run_curriculum(start_stage=K)` + a run manifest, so a killed session
   resumes instead of restarting.

### 2.10 `max_steps: 3000` is a truncation cliff that the metric can't see

`gym_env.py:1505` `truncated = self.current_step >= self.max_steps`;
`:1573-1577` scores truncation with the **draw** reward; `evaluation.py:366`
computes `win_rate = wins / n_episodes`, so a truncation counts against the
gate exactly like a loss.

Measured: **27-42 of 80** eval episodes truncated in normal evals of the
deepest run. `v52a_maxturn_scaled_draw.yaml:11-13` still asserts "3000 is well
above any realistic game length ... so truncation never fires under normal
play". It fires on a third to half of episodes.

Compounding it, `max_actions_per_turn: null` in both `bootstrap.yaml:49` and
`v54:474` disables the only guard against the never-end-turn attractor — the
guard whose own docstring (`gym_env.py:501`) describes the failure mode being
observed.

**Fix**: scale `max_steps` from `max_turns` (`max_turns * max_actions_per_turn
* 1.5`) rather than pinning it at a constant; set `max_actions_per_turn` to a
real value; and log `truncated` separately from `max_turns_draw` in the
promotion metric so the gate stops conflating "slow" with "bad".

### 2.11 Promotion gates are a coin flip near threshold

`callbacks.py:415-430` requires `patience` **consecutive** evals over the gate.
At WR 0.75 with 80 episodes the standard error is 4.8pp, so two consecutive
crossings of a 0.75 gate is roughly a coin flip for a genuinely-0.75 policy —
and 41/41 stalls burned their full budget waiting for it.

**Fix**: gate on a Wilson lower bound of the win rate, or on a rolling mean of
the last K evals, instead of raw consecutive point estimates.

### 2.12 The BC / imitation subsystem targets an action space that cannot learn

`gym_env.py:1059-1060` documents multi_discrete masks as a
"per-dimension boolean masks (**union over-approximation**)" — per-dimension
masking cannot express joint legality, so illegal joint actions are sampled.

Measured on run `20260525_050151`: `invalid_penalty` = **-2358 per eval** at
`invalid_action: -0.1`, i.e. **~23,580 invalid actions per 240,000 steps
(~10%)**, with **80/80** episodes hitting `max_steps_truncate` and WR pinned at
0.00 for the entire run.

`imitation.py:25` and `scripts/build_bc_warmstart.py:160` support
**multi_discrete only**. That is ~1,600 lines of BC infrastructure plus three
configs wired to the action space that empirically cannot learn, while every
production run uses `flat_discrete`.

**Fix**: port BC to `flat_discrete` — the label is just the index of the
demonstrated action inside `build_flat_actions(...)`, which is already a
module-level pure function (`gym_env.py:230-302`) shared with `ModelBot`. Or
retire the subsystem and delete the configs, so it stops being a trap.

Two more reasons a BC warm-start would underperform even after that port:

- **The value head is left at random init on top of a BC-shifted shared trunk.**
  `imitation.py:1331` selects `bc_params = [p for n, p in
  policy.named_parameters() if not n.startswith("value_net")]`, so BC trains the
  `features_extractor` but not the final value head. SB3's
  `MaskableActorCriticPolicy` defaults to `share_features_extractor=True` and no
  config overrides it, so PPO's first updates see an enormous value loss and
  push it back through the shared extractor at `vf_coef: 0.5` — undoing the
  clone within a few updates. Freeze the extractor for the first N updates, or
  regress the value head on demo returns before handing off.
- **Demonstrations are collected on a different engine.** `imitation.py:707-713`
  constructs `GameState(map_data, num_players=2, max_turns=..., enabled_units=...,
  fog_of_war=...)` and omits both `engine_overrides` and `rng`. Any config from
  v50 onward (`damage_model: hp_scaled`, `W: {cost: 300}`) therefore clones a
  bot that was playing a different game.

### 2.13 Self-play RNG is unseeded; opponent inference is on the hot path

`self_play.py:542`

```python
if self.swap_players and random.random() < 0.5:
```

Module-global `random`, so `reset(seed=...)` does not control it. Same at
`:355` / `:365` (`np.random.randint`, `np.random.choice`). Under forked
`SubprocVecEnv` every worker inherits an identical global RNG state, so the
side-swap decision is **correlated across all 8 envs** and self-play evals are
not reproducible. This is the exact bug class the env already fixed for combat
RNG at `gym_env.py:1663-1669` — the fix just never reached the wrapper.

`self_play.py:317-340` copies the full policy `state_dict` to numpy,
`load_state_dict`s the opponent, predicts one action, then `load_state_dict`s
the original back — **two full parameter loads per opponent action**. Hold two
policy instances instead.

`self_play.py:533-548`: `self.env.reset()` runs at `:533` (which sets
`_prev_potential` from `agent_player`'s seat, `gym_env.py:1692`) and
`agent_player` is only flipped at `:544`. The first shaping delta of a swapped
episode is computed against the other seat's potential. Low severity but free
to fix.

Three further defects in the same file are more serious than the RNG issue, and
each independently makes self-play not-self-play. I verified all three against
the source:

**(a) The opponent is never given action masks.** `self_play.py:332`:

```python
action, _ = self.opponent_model.predict(obs, deterministic=self.opponent_deterministic)
```

`MaskablePPO.predict` treats a missing `action_masks` as "no masking", so the
opponent samples freely over the full `MultiDiscrete([10, 8, W, H, W, H])`.
Nearly every such joint action is illegal (2.12), so the opponent is effectively
a pass-bot. The "self-play" agent is training against noise.

**(b) `swap_players` never reaches the game.** `make_self_play_env` composes
`SelfPlayEnv(ActionMaskedEnv(StrategyGameEnv(...)))` (`:751-754`), so
`self.env` is the wrapper. `SelfPlayEnv.reset` then does:

```python
self.agent_player = 2
self.env.agent_player = 2  # :545
```

`gymnasium.Wrapper` overrides `__getattr__` but **not** `__setattr__`, so this
creates a fresh attribute on the *wrapper* that shadows the delegation.
`StrategyGameEnv.agent_player` stays `1`. On every swapped episode the base env
scores rewards, computes the potential and picks the terminal bonus for the
wrong seat while `SelfPlayEnv` believes the agent is player 2.

**(c) `SubprocVecEnv` silently disables opponent updates entirely.**
`_SelfPlayCallback._get_self_play_envs` (`:627-643`) reaches the envs only via
`hasattr(self.env, "envs")`. `SubprocVecEnv` has no `.envs` — the envs live in
child processes — and the `isinstance(self.env, SelfPlayEnv)` fallbacks don't
match a VecEnv either, so the method returns `[]` and `_update_opponents` /
`_add_to_pool` are no-ops. Every bootstrap config sets `use_subprocess: true`.

Given (a)-(c), no self-play result in the archive should be trusted, and
`opponent: "self"` in a curriculum degrades further still: `gym_env.py:1751-1760`
falls back to `self.opponent = None` when no self-play factory was registered,
and the bootstrap path never registers one — a silently do-nothing opponent that
config validation accepts (`"self"` is in `_BOT_OPPONENT_TYPES`, `:76`).

### 2.14 Single-seed sweeps

50/56 runs used `seed: 42`, n=1. v21 re-ran a *just-cleared* stage on identical
settings and went from cleared to peak 0.86 / final 0.0125. Most single-knob
verdicts recorded in `bootstrap_lessons_learned.md` sit inside the noise band
they were meant to resolve.

---

### 2.15 The final sanity eval does not use the stage's env

`scripts/train/train_bootstrap.py:301-312` hand-rolls its env:

```python
env = make_maskable_env(
    map_file=...,
    opponent=...,
    max_steps=...,
    max_turns=...,
    reward_config=...,
    enabled_units=...,
    action_space_type=...,
    seed=cfg.seed + 9999,
    opponent_kwargs=...,
    pad_to_size=cfg.env.pad_to_size,
)
```

It drops `engine_overrides`, `max_flat_actions`, `max_actions_per_turn`, `gamma`
and the three tanh scale factors. For every config from v50 onward that means
the sanity eval runs on **default engine rules** — no `damage_model:
hp_scaled`, no `W: {cost: 300}` — so its win rate is not comparable to the
in-training evals it is meant to cross-check.

`bootstrap.py:1036-1053` documents this exact bug class and
`make_stage_env` (`:1055`) exists to prevent it. The fix is one line:

```python
from reinforcetactics.rl.bootstrap import make_stage_env

env = make_stage_env(stage, cfg.env, seed=cfg.seed + 9999)
```

(`make_stage_env` should also forward `gamma`; it currently does not — see
section 3.)

### 2.16 The alternative algorithms have their own correctness bugs

These are not on the bootstrap critical path, but they are load-bearing for
anyone who runs `train_feudal_rl.py` or `train_alphazero.py`, and each would
silently produce a plausible-looking but wrong training curve.

**Feudal RL**
- `feudal_rl.py:1258-1262` vs `:1295-1296` — `manager_segment_open`,
  `manager_reward_accum` and `manager_step_count` are **locals** reset on every
  `collect_rollout()` call, but `self.current_goal` / `self.goal_step_counter`
  persist on the agent. Reward earned under goal `g_k` is credited to `g_(k+1)`
  across every rollout boundary.
- `feudal_rl.py:1324-1325` — `done = terminated or truncated` is stored as both
  `w_dones` and `m_dones`, and `_compute_gae` uses it as the terminal indicator.
  Every time-limit episode gets its value bootstrap zeroed.
- `feudal_rl.py:1358` — the manager critic regresses on an **undiscounted**
  intra-segment reward sum while GAE applies `gamma^k` to the bootstrap; the two
  are inconsistent.
- `feudal_rl.py:823, 1983-2003` — worker reward mixes an intrinsic term in
  roughly `[-10, +15]` with the raw env reward, unnormalized; and two goal
  bonuses scale linearly in nearby-unit count with no cap.

**AlphaZero**
- `alphazero_trainer.py:315-326` — `self.network.train()` is set before
  `_training_phase()` and never switched back before `_evaluation_phase`.
  BatchNorm running statistics are then overwritten by batch-size-1 MCTS
  forwards during evaluation.
- `alphazero_trainer.py:522-525` — `_evaluation_phase` returns `0.5` when
  `total_decided == 0`, and `_play_eval_game` builds its `GameState` with **no
  `max_turns`**, so an all-truncated eval returns exactly the value that
  guarantees permanent rejection of every candidate network. Rejection also
  reverts weights but not the Adam/scheduler state.
- `alphazero_trainer.py:98-139` — `self_play_game` caps on `max_steps=400`
  counted in *individual actions*, with no `max_turns` on the `GameState`. Given
  the ~2000-step episodes measured for PPO, essentially every self-play game
  truncates and is labelled a draw, training the value head to predict 0
  everywhere.
- `alphazero_trainer.py:574, 612` — `_save_checkpoint` persists five config keys
  and `load_checkpoint` does `cls(**config)`, so `map_file`, `enabled_units` and
  `lr` are silently dropped on resume.

---

## 3. Smaller things worth fixing

- `configs/ppo/bootstrap.yaml` — the **canonical** config — still ships
  `win_speed_bonus: 50.0` (`:157`), `enemy_owned_capture: -15.0` (`:171`) and
  `turn_penalty: 0.0` (`:149`): exactly the terms the v27 ablations isolated as
  causing the wall. v54 has the corrected values; the default anyone runs does
  not. Three-line back-port.
- `callbacks.py:324-331` — `best_win_rate` is only updated inside
  `if self.save_dir is not None`. With `save_dir=None` it stays at `-1.0` and
  `CurriculumStalled.achieved_win_rate` would report `-1.0`. Not live in
  bootstrap (which always passes a stage dir), but it is a trap for any other
  caller.
- `bootstrap.py:866-896`, `:918-921` — three `except Exception: pass` blocks
  around metadata writes. Justified for the checkpoint-is-load-bearing
  argument, but they currently swallow the *reason*; a one-line
  `logger.warning` would make a Drive-quota failure visible.
- No `VecNormalize` anywhere in the PPO path. With unnormalized returns and
  `vf_coef: 0.5`, value loss magnitude is set by reward scale. The 5000 -> 50
  rescale fixed the worst of it; `norm_reward=True` (or `clip_range_vf`) would
  decouple the two permanently.
- `viz.py:262` already carries a comment about scaling rewards / adding
  `clip_range_vf` — the diagnostic exists, the knob was never wired.
- `bootstrap.py:1055` `make_stage_env` does not forward `gamma`, so any replay
  or re-eval env computes `shaping_delta` with the default 0.99 even when the
  run trained at a different discount. This becomes wrong the moment
  recommendation 2.3 is applied.
- `bootstrap.py:752` reads `stage.n_eval_episodes`, whose default is `30`
  (`config.py:255`). `cfg.eval.n_eval_episodes` is **never read** by the
  curriculum runner. Today every shipped stage sets the override, so nothing is
  broken — but a new stage that forgets it silently evaluates on 30 episodes
  while the config says 80. Make the stage field `int | None` and resolve
  against `cfg.eval`.
- `bootstrap.py:836` — `model.ep_info_buffer` is not cleared at stage
  boundaries, so the first ~100 episodes of each stage's `train_metrics.csv`
  rows describe the *previous* stage while carrying the new stage's `context`
  label.
- `scripts/train/train_bootstrap.py:399-415` — a `CurriculumStalled` is caught,
  printed, and then the script prints `✅ Done` and returns **0**. Any CI or
  scheduler treating exit code as ground truth records a failed run as a
  success. Return a non-zero code on stall.
- `callbacks.py:390-409` — `min_timesteps_before_promotion` is now
  stage-relative; it used to be compared against the cumulative counter. That is
  the better semantic, but `v31_production_minsteps_gate.yaml`'s archived run no
  longer reproduces from its own config. Worth stamping a
  `promotion_gate_version` into `_write_stage_config`'s output.
- `bootstrap.py:699-703` — the purchase-exploration hook is installed once from
  the **global** `cfg.ppo.purchase_explore_eps`. A stage-level override is
  applied by writing the live attribute, but if the global is `0.0` on
  `flat_discrete` the install returns early (`purchase_exploration.py:207-211`)
  and the per-stage value is a silent no-op.
- `bootstrap.py:469-480` — per-stage `config.json` omits `pad_to_size`,
  `gold_scale`, `turn_scale`, `unit_count_scale` and `n_envs`. `pad_to_size` is
  the one field the runner *derives* (`:626-629`), so the observation space
  cannot be reconstructed from the run record.
- `bootstrap.py:97-101` — the `CurriculumStalled` message always reads "best
  win_rate X did not reach threshold Y", even when the peak exceeded the
  threshold (28/41 cases). "Peaked at 100%, never held it for 2 consecutive
  evals" is a different failure and should read differently.
- `scripts/train/train_bootstrap.py:373-384` — the run directory gets
  `shutil.copy2(config_path, ...)`, i.e. the **source** YAML, *after*
  `_apply_set_overrides` and the device resolution have already mutated `cfg`.
  A run launched with `--set ppo.gamma=0.997` records `gamma: 0.99`. Dump the
  resolved config instead — this matters the moment you start sweeping via
  `--set`.
- `bootstrap.py:764` — per-step eval tracing is hard-wired on
  (`trace_dir=stage_dir / "traces"`) with no config knob and no size cap, and
  `evaluation.py:223` allocates the per-step buffer for *every* episode, not
  just matching ones. A stalled stage writing to a Drive-backed run dir can
  produce a lot of JSONL for episodes that are 2000+ steps long.
- `evaluation.py:215-235` — eval is a strictly sequential single-env loop with
  one inference call per step. At 80 episodes x ~2000 steps that is 160k
  sequential GPU round-trips per eval, against 50k *vectorized* training steps
  between evals. Eval, not training, may well be the wall-clock bottleneck —
  which matters directly because wall-clock death is one of the two run-fatal
  modes. Vectorize it or drop `n_eval_episodes` between promotions.
- `gym_env.py:420` — `map_file: null` draws a random map once in `__init__` via
  the module-global numpy RNG, before any seeding exists, so `reset(seed=...)`
  cannot reproduce it and each vec worker gets a different board. Not used by
  the bootstrap configs, but it is a live trap.
- Preemption: `train_bootstrap.py` writes to `benchmarks/bootstrap/<run_id>` and
  uploads once from a `finally` block, while the container's periodic uploader
  syncs a different hardcoded directory list. There is no SIGTERM handler. On a
  Vertex/Colab preemption the whole run directory is lost — for a project whose
  deepest run died to wall-clock, this is worth an hour of work.

---

## 4. Suggested order of work

Correctness first, then the two never-tested axes, then the plumbing that lets a
run survive. Each experiment is one variant off v52a so attribution stays clean.

> **Status:** items 1-5 are implemented. See "Landed" at the bottom of this
> section for exactly what changed and what it does *not* fix.

**Correctness (do before any new sweep — these change what the numbers mean):**

1. **Charge the terminal `-Phi(s_prev)`** (2.1). ~3 lines in
   `_calculate_reward`. Removes the "being ahead is taxed" gradient that the
   archive confirms to 0.3%.
2. **Pick one truncation semantic** (2.4). Either `terminated=True` on
   `max_steps`, or drop the `draw` bonus from the truncation branch. ~2 lines.
3. **Fix the eval methodology** (2.2): hold the eval seed set fixed across a
   stage, and gate on the stochastic policy (`deterministic=False`) or report
   both. Two lines, and every subsequent number becomes comparable.
4. **Make the stage-entry eval baseline-only** (2.8). ~5 lines in
   `PeriodicEvalCallback`. Stops `restore_best_checkpoint_between_stages` from
   reverting a stage's own training.
5. **Point `_final_sanity_eval` at `make_stage_env`** (2.15), and forward
   `gamma` from `make_stage_env`. ~3 lines. Dump the *resolved* config rather
   than the source YAML (section 3) before running any `--set` sweep.

**Then the two axes nobody has swept:**

6. **`pool: flatten`** (one line). Watch `captures_by_type.hq` — if it leaves 0,
   that is the answer to the biggest open question in the archive.
7. **`gamma: 0.997` + `max_actions_per_turn: 25`** (two lines). Brings the
   terminal inside the horizon and shortens the episode at the same time.
8. **Stage-relative LR anneal** (~40 lines, copy `EntropyScheduleCallback`).
   The other half of the bistability fix.

**Then the plumbing that decides whether a run can ever finish:**

9. **Within-stage best-checkpoint restore on regression** (~30 lines). Converts
   the dominant stall mode into promotions using a checkpoint already on disk.
10. **Retry-from-best on stall, and resume-from-stage-K** (~60 lines). Without
   it 108M steps of curriculum cannot be reached at ~6M steps per session, no
   matter what else is fixed.
11. **Wilson lower bound in `PromotionCallback`** (~10 lines).

**Then hygiene:**

12. Back-port the v49/v52a reward values into `configs/ppo/bootstrap.yaml`.
13. Non-zero exit on stall in `train_bootstrap.py`.
14. Port BC to `flat_discrete`, or retire it and delete the three configs.
15. Fix the three self-play defects in 2.13 — opponent masks, the
    `agent_player` write that lands on the wrapper, and `SubprocVecEnv`
    returning no envs — before trusting any self-play result. Then seed the
    wrapper's RNG off `np_random` and hold two policy objects instead of
    swapping `state_dict`s per opponent action.
16. Re-run any conclusion you intend to keep at **3 seeds**.

### Landed (items 1-5)

| change | file |
| --- | --- |
| Terminal step charges `F = gamma*0 - Phi(s_prev)` instead of skipping the shaping term. A step-limit truncation keeps the ordinary delta — its successor state is real and gets bootstrapped. | `gym_env.py` `_calculate_reward` (`terminal` -> `terminated`) |
| Truncation no longer charges the `draw` terminal. New opt-in `reward_config['truncation']` (default 0.0) if an explicit penalty is wanted. | `gym_env.py` step |
| Eval replays one fixed problem set per stage. `EvalConfig.resample_eval_seeds` restores the old rotate-every-eval behaviour. | `callbacks.py`, `config.py` |
| Evals inside `best_eligible_after` stage-relative steps are recorded but cannot claim `best_model.zip`. Bootstrap passes `eval_freq`; `EvalConfig.best_eligible_after` overrides. Each eval row now carries `stage_steps` and `best_eligible`. | `callbacks.py`, `bootstrap.py`, `config.py` |
| `make_stage_env` forwards `gamma`; `_final_sanity_eval` goes through it instead of hand-rolling an env. | `bootstrap.py`, `train_bootstrap.py`, `imitation.py` |
| `resolved_config.yaml` written next to the source YAML so `--set` overrides are in the run record. | `train_bootstrap.py` |

Eight tests were added, each verified to fail against the previous behaviour —
including a telescoping check that the discounted shaping return equals
`-Phi(s_0)` exactly, on a deliberately asymmetric start so it cannot pass
vacuously.

**What this does *not* fix.** The terminal charge restores Ng et al.
invariance, but it does **not** remove the ~190/episode drain measured in 2.1.
That drain is `(1 - gamma) * Phi` accumulated over ~1900 micro-actions, and its
size is set by `gamma` and by steps-per-episode — not by the terminal term,
whose discounted weight at `gamma=0.99, T=1900` is ~1e-9. Removing the drain
needs item 7 (`gamma` up, episode length down) or per-game-turn shaping cadence.
Treat item 1 as the correctness precondition that makes raising `gamma` safe,
not as the fix for the draw attractor on its own.

**Comparability.** These change the reward function and the eval protocol, so
numbers from here on are not comparable with the archived runs. Re-run v52a
unchanged as the new anchor before reading any treatment effect.

---

## 5. Status of the 2026-07-12 review's recommendations

**Landed** (commits `026b8a0`, `7fedf3e`, `a552bc1`, `0587759`):
observability — `train_metrics.csv`, `eval_results.jsonl`, best-checkpoint
fields in `run_status.json`; `BUILD_BC_WARMSTART` default; eval-env drift fix;
auto-heal diagnostics; `v54_uncapped_frontier` with `max_steps: 4500` and
`max_flat_actions: 1024`.

**Not landed** — and these are the two that decide whether a run finishes:

- **Rec #2, checkpoint/resume.** No `start_stage`, no retry-from-best on stall.
  Both observed death modes (wall-clock, stall) are still run-fatal.
- **Rec #5, the optimizer axis.** `gamma`, `learning_rate`, `n_steps`,
  `batch_size` and `clip_range` remain identical across every config in the
  repo, v54 included. `lr_schedule` is still dropped before it reaches SB3
  (`config.py:137`).

To which this review adds a third never-touched axis: **the representation**
(`pool`, and the flat positional action head).
