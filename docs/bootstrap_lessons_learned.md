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
6. **Carry the peak forward, but don't force overtraining.** PPO drifts
   within a stage; the in-memory end-of-stage policy is often weaker than
   the stage's `best_model.zip` — so restore the best between stages
   (`restore_best_checkpoint_between_stages`, default on). But on
   stochastic stages with the draw-with-shaping attractor, *additional
   training past the policy's natural peak erodes the winning behaviour*.
   v31 tested forcing more training via `min_timesteps_before_promotion`
   and stalled one stage earlier than v28. The gate is kept in the code
   as an opt-in feature but is **not** set on `*_random_N` stages in
   production; PPO's natural promotion rhythm beats forced overtraining
   on drift-attractor stages.
7. **BC dataset quality > dataset size.** N episodes of a fully-deterministic
   bot-vs-bot matchup produce **1 unique trajectory copied N times**, not N
   distinct demos. Without the new `stochastic_tiebreak` opt-in (or
   stochastic opponents), most of a BC dataset is duplicates and the
   policy memorises three game scripts instead of learning a
   distribution. See "BC warm-start: what we learned" near the bottom
   of this doc.
8. **Player-side asymmetries observed in deterministic bot-vs-bot play
   are usually artifacts, not structural.** AdvancedBot-vs-AdvancedBot on
   skirmish.csv looked like a 0/60 sweep for player 2 — recording demos
   from player 2 looked like the obvious fix. With stochastic tiebreak,
   the same matchup is 40/60 *for player 1* (i.e. roughly symmetric).
   Don't conclude "the map favours side X" from runs where ties resolve
   by data-structure iteration order.
9. **Ad-hoc eval envs MUST forward every kwarg the production env uses
   — not just the obvious ones.** Especially `reward_config` and
   `max_actions_per_turn`. The env's built-in defaults are 10-1000×
   off from any well-tuned YAML, and missing `max_actions_per_turn`
   disables the never-end-turn safety net that production trains
   against. We hit this in 20260524_225835 and almost misdiagnosed
   "BC is broken" when the real issue was a 1-line eval env bug. See
   "BC warm-start: what we learned" → Failure mode G.
10. **Better BC training metrics ≠ different gameplay.** Run
    20260525_015401 bumped `end_turn_weight` 10× → 30× (3× the
    auto-balance) and got measurably better BC training (loss
    2.73 → 1.21, action_type_acc 0.78 → 0.83, full_action_acc
    0.21 → 0.27) — but BC's sanity-eval result vs SimpleBot was
    *byte-identical*. The class re-weight upweights gradient
    contribution from end_turn samples, which makes the policy
    predict end_turn more accurately *when end_turn is the
    correct demonstrator action*, but doesn't shift the
    argmax prior at the OTHER states (where the demonstrator
    would have made a productive move). Loss / accuracy curves
    can improve invisibly to gameplay. Sanity-eval BC against
    the bot ladder *before* trusting training metrics. See
    Failure mode H.
11. **Mono-strategy attractors are env-structural, not curriculum-
    tweakable.** v34-v37 ran four independent levers — combat-shaping
    reward, mild Warrior stat nerf (atk 10→8, def 6→4), 2× budget
    on the stall stage, MixedBot mid-curriculum bridge, higher
    promotion thresholds — and got **100% mono-Warrior policies on
    every cleared stage** in all of them. v37b cleared 15 stages at
    100% WR each, every one mono-W with zero ability use and zero
    HQ captures. If the default unit balance has one unit winning
    HP/$, Atk/$, AND a defense advantage at the cheapest tier, PPO
    finds it and never leaves. The fix is cost-curve geometry
    (v38/v39 changed Warrior's price), not curriculum pressure.
12. **Don't run balance experiments on the tightest-clock map.** v38
    (structural stat nerf) and v39 (cost-only nerf) both stalled at
    `starter_random` (51% / 35% WR) — *not* because the nerfs were
    wrong, but because starter's $250 starting gold + max_turns=20
    has no headroom for the strategic tradeoff the nerfs are
    supposed to teach. Same nerf on beginner (max_turns=75, $400
    start) cleared the first stage at 100% WR in 200k steps. Starter
    is a navigation warm-up; drop it from the curriculum when the
    experiment is about balance, not navigation.
13. **Composition tracking belongs in the eval pipeline.** `units_built`
    per stage tells you whether the policy learned strategy or just
    exploited the cheapest cost-efficient unit. WR + W/L/D alone
    looks identical between "won via mono-Warrior elimination" and
    "won via diverse composition with HQ rushes". `viz.plot_curriculum_
    composition_summary` (added in this branch) makes this visible
    at a glance — without it, four iterations would have looked
    indistinguishable on the WR curves.

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

## ⚠️ CORRECTION — the v15–v23 conclusions below were confounded

**Read this before the sweep section.** The analysis in
"The curriculum-tuning sweep (v15–v23)" (below) was built on a
`runs_summary.csv` whose `enabled_units` column was wrong. The
analysis notebook sourced the roster from the run-root YAML and fell
back to "all 8" for every run without one — which was almost every
run. A corrected audit reading the authoritative
`config.json:env_config.enabled_units` overturned two headline
conclusions:

1. **"Reducing the unit roster fails catastrophically" (from v22) is
   FALSE.** v22 used `[W,K,A,M]` (Knight, *no Cleric*) + `tp=-0.5` +
   the post-v16 engine. The *deepest-progressing runs in the entire
   history* used a **restricted** roster:
   - `20260511_132922`: roster `[W,M,C,A,K]`, `tp=-0.2`, rt 0.2.5 →
     **reached `skirmish_simple` — 17 curriculum stages**, ~3× deeper
     than v19, the supposed "high-water mark."
   - `20260512_163309` / `_195716`: roster `[W,M,C,A,K]`, `tp=0.0` →
     cleared `beginner_random_15` at **100% WR in 50k steps** (the
     stage v17–v23 spent millions of steps stalling on).
   - The deep `[W,M,C,A]` (4-unit, *with* Cleric) runs reached
     `beginner_advanced` (12 stages).
   Reduced rosters are the *best* configs on record. v22 failed
   because it dropped Cleric and stacked the v16 economy on top, not
   because it had fewer units.

2. **"turn_penalty=-0.5 alone introduced the random_15 wall" is
   INCOMPLETE.** v16 changed *three* things at once: roster
   (restricted → all-8), `turn_penalty` (0 → -0.5), and engine
   economy (Knight defence 5→7, HQ income 150→100). Every
   all-8 + tp=-0.5 run stalled; every deep run was restricted-roster
   + tp≤0 + pre-v16 economy. The wall correlates with the *triple*
   change, not turn_penalty in isolation.

Caveat preserved for honesty: roster + tp are not *sufficient* —
`20260513_033055` (v15) used `[W,M,C,A,K]` + `tp=0.0` and still
stalled at random_10 (its v15 entropy-floor change is the
differentiator). And the deep runs *stopped* at 6–17 stages (Colab
disconnects / short sessions), so we cannot prove they wouldn't have
stalled later. But "every run since v17 stalled at random_15" is a
statement about the all-8 + tp=-0.5 branch only — a branch the
corrected data says was a regression off a working configuration.

**v24 (`v24_reproduce_deep_config.yaml`) reproduces the 17-stage
config**: roster `[W,M,C,A,K]`, `turn_penalty=-0.2`, engine reverted
to Knight def=5 / HQ income=150 (constants.py, committed alongside),
on the v18 base (standard curriculum + the proven patience=2 fix, no
consolidate experiment). If v24 progresses deep, the entire v17–v23
entropy/patience/threshold detour was treating a self-inflicted
regression.

The original sweep section is kept below **unedited** because its
*mechanistic* observations (the draw-with-shaping attractor, the
drift-vs-plateau diagnostic, GPU non-determinism, warm-start
non-reproducibility) remain valid and useful — only the
roster-blind comparative conclusions are superseded by this
correction.

## ✅ EXPERIMENT A RESULT — the deep config is REAL; the regression is in the code path

Run `20260515_213159` was a faithful reproduction: commit
`6eb0566` (rt 0.2.5) checked out unedited, `ppo_bootstrap.ipynb`
from that commit Run-All'd, economy verified at install
(`STARTING_GOLD=250 HQ_INCOME=150 Warrior_atk=10 Knight_def=5`),
roster `[W,M,C,A,K]` via the active cell-11 override,
`turn_penalty=-0.2`, 24-stage rt-0.2.5 curriculum. **It reproduced
the deep progression and then some:**

- `starter_random/simple/medium` → `beginner_balanced_random` →
  **`beginner_random_10` cleared on the first two evals (WR
  1.0, 1.0)** — the exact stage every v16–v24 run dies on.
- `beginner_random_15` (0.99/1.0), `beginner_random_20`
  (0.89/0.90), `beginner_simple/mixed/medium/advanced` — all
  cleared.
- `skirmish_balanced_random` → `skirmish_random_10/15/20` cleared
  (each via a collapse→recover through the draw attractor — the
  policy *escapes* it at this commit).
- Stalled at **`skirmish_simple` (stage 17)**, 0.0 WR from 5.8M to
  8.8M timesteps where the session ended — exactly matching the
  historical deep run's recovered signature.

This is **decisive**. It rules out the remaining hypotheses:

1. **Not the economy.** v24 reverted the economy and still stalled
   at stage 5. Confirmed by an independent axis here.
2. **Not a phantom / not a curriculum-length artifact.** The deep
   progression is real, end-to-end, and repeatable — 16 stages
   cleared, not a Colab-stop count.
3. **It IS the rt 0.2.5 → 0.2.7 code path.** The regression that
   makes `beginner_random_10` unpassable on current code lives in
   the engine/training code that drifted between `6eb0566` and
   HEAD — *not* config, *not* economy, *not* roster.

Mechanistic note: the skirmish stages all showed the
draw-with-shaping collapse (1.0 → 0.0 draws for ~1M steps → snap
back to 1.0). At `6eb0566` the agent **escapes** that attractor;
on current code at `beginner_random_10` it never does. The
regression likely didn't add a hard breakage — it made the
draw-equilibrium attractor *inescapable*. Prime suspect:
`c7001bf` ("Replace per-end_turn cost with terminal speed bonus")
— a reward-landscape change directly over the draw/win incentive
the deep config's `turn_penalty=-0.2` was shaping.

Next step was a **code bisect** over `6eb0566..HEAD` — but it
turned out **not** to be a code regression at all. See the
RESOLVED section immediately below.

## ✅ RESOLVED — the regression is the post-`6eb0566` reward additions (code exonerated)

The "0.2.5 → 0.2.7 code path" framing above was **wrong**, and the
reason it was wrong is instructive: v24 was assumed to be a faithful
config port, but it silently carried **three reward terms the deep
run never had** — `win_speed_bonus: 50` (added by `c7001bf`) and
`enemy_neutral_capture: -8` / `enemy_owned_capture: -15` (added by
`922aa29`). Every other reward key was byte-identical to `6eb0566`.

`v26_faithful_deep_reward_on_head.yaml` settled it (run
`20260516_194038`): **modern HEAD code** (rt 0.2.7, modern
obs/extractor/masking) + the byte-faithful `6eb0566` reward shape
(those 3 terms zeroed) + faithful economy via `engine_overrides` →
**cleared `beginner_random_10`** (`promoted: true`,
`best_win_rate 0.9625`, WR 96.25%/83.75%; `run_status:
completed_curriculum 5/5`). Not bit-identical to Experiment A
(reward/turns differ — the config genuinely landed, unlike the
inert Experiment-B test #1), so it's a real, independent clear.

| Run | Code | Reward shape | `beginner_random_10` |
|-----|------|--------------|----------------------|
| Experiment A | 6eb0566 | deep (faithful) | clears, 16 stages |
| v24 | modern | deep economy **+ win_speed_bonus 50 + capture penalties** | **stalls @ stage 5** |
| **v26** | **modern** | **deep, those 3 terms zeroed** | **clears** |

The *only* delta between the v24 stall and the v26 clear is those
three reward terms. **That is the regression.** It is **not** a
code regression — modern obs slimming / CNN extractor / masking /
RandomBot / curriculum all clear `random_10` fine once the reward
shape is faithful. The fix is **config-only**. The whole v17–v23
entropy/patience/threshold sweep, and the planned code bisect, were
chasing a self-inflicted *reward* regression — no code bisect was
ever needed.

### Final isolation — TWO independent culprit terms (`v27a/b/c`)

`v27a/b/c` re-enabled exactly **one** of the three terms each on
the proven v26 baseline (modern code, faithful economy via
`engine_overrides`, only the one reward term varying — all three
runs verified `git b1926c2`, rt 0.2.7):

| Run | Term re-enabled | `beginner_random_10` | best WR |
|-----|-----------------|----------------------|---------|
| `20260516_202350` v27a | `win_speed_bonus: 50` | **stalled** (`curriculum_stalled`) | 0.80 |
| `20260516_222009` v27b | `enemy_neutral_capture: -8` | **cleared** (`completed_curriculum 5/5`) | 0.9875 |
| `20260516_230751` v27c | `enemy_owned_capture: -15` | **stalled** (`curriculum_stalled`) | 0.90 |

**Two terms independently re-create the wall: `win_speed_bonus`
(`c7001bf`) and `enemy_owned_capture` (`922aa29`).**
`enemy_neutral_capture` is **benign** (v27b cleared at 98.75%).

Mechanistic note: v27a/c did not hard-collapse to 0% — they peaked
*above* the 0.75 gate (0.80 / 0.90) but couldn't hold two
consecutive evals within budget. This is the draw-with-shaping
**policy-drift instability** documented above: each term perturbs
the reward landscape enough to prevent *stable* convergence at
`random_10`, even when the policy transiently reaches competence.
Consistent with v26 (all three zeroed → stable clear) and v24 (all
three present → hard 0% stall).

**Shipping fix:** on modern code, the production reward config must
set `win_speed_bonus: 0` and `enemy_owned_capture: 0` and
`turn_penalty: -0.2` (the deep-run value, which works);
`enemy_neutral_capture` may stay at `-8` if it has design value on
neutral-contested maps, or be zeroed for minimalism. This is the
basis for `configs/ppo/bootstrap_sweep/v28_production_reward_fixed.yaml`
(full 33-stage production curriculum, modern intended balance — see
its header for the balance-vs-trainability note).

Lesson reinforced: "faithful repro" must be verified key-by-key
against the original, not assumed from "same economy + roster." A
config that differs by three additive reward terms is a different
experiment — and cost two weeks of code-regression archaeology.

## ✅ ENGINE-CONSTANT CONFOUND CLASS — now closed via `env.engine_overrides`

The root cause of the two-week reproduction thread was not any one
config: it was that **balance lived in `constants.py`
(`UNIT_DATA`, `STARTING_GOLD`, `*_INCOME`) — outside the config
surface entirely.** No YAML edit and no `apply_overrides` call
could reach it; it silently drifted across commits (`a596c15`
Warrior/Barbarian/gold, `f4dc50e` Knight defence 5→7, `6f64745`
HQ income 150→100) and was unrecorded in any run artifact. Every
historical comparison was confounded by an invisible variable only
recoverable via `git show <sha>:constants.py`.

This is now structurally closed:

1. **`env.engine_overrides`** — a sparse, optional YAML overlay
   (`starting_gold`, `*_income`, `unit_data: {CODE: {field: val}}`).
   `GameState` deep-merges it over the module constants into
   `self.unit_data` / `self.income_rates` / `self.starting_gold`;
   units and income read those resolved tables, never the globals,
   so an override can't leak or be half-applied (module constants
   stay pristine; unknown code/field fails loud). Absent ⇒
   byte-identical to before.
2. **Auto-recorded** — `config.json` now logs `engine_overrides`,
   the resolved `effective_engine_economy`, an
   `effective_balance_profile_hash`, and a verbatim
   `engine_constants_hash` over the *full* `UNIT_DATA` (catches
   drift in fields the 5-key projection misses).
3. **Faithful repro is now pure config.** `v26_faithful_deep_reward_on_head.yaml`
   carries the byte-faithful `6eb0566` `[W,M,C,A,K]` stat block +
   economy as an `engine_overrides` block — no `git checkout
   constants.py` pin, no notebook surgery, runs on modern HEAD.

Lesson: **anything that affects outcomes must live on the config
surface and be snapshotted into `config.json`.** `enabled_units`,
`reward_config`, and now the engine economy/stats are all closed.
The remaining members of this class to stay alert to: map content
(closed via `map_sha256`), curriculum structure (`curriculum_hash`),
and library/code drift (`git.commit` + `libraries`). If a future
result-affecting knob is added, it goes in the dataclass and the
`config.json` meta block in the same change — not in a module
constant.

## ✅ CURRICULUM HANDOFF + SKIP-AHEAD — two coupled curriculum bugs, both fixed

After the reward and economy classes were closed, v28 (the full
33-stage production config: modern intended balance via
`engine_overrides`, byte-faithful 6eb0566 reward, roster
`[W,M,C,A,K]`) still stalled at `beginner_random_15`. Two further
mechanistic bugs in the curriculum runner itself surfaced — neither
about config, both about *how the runner moves a policy between
stages*. Both are now fixed (defaults preserve legacy behaviour).

### Bug 1 — within-stage drift was propagating to the next stage

The curriculum loop carried the **in-memory end-of-stage** model
across stages via `model.set_env(...)`. PPO drifts off the winning
attractor *within* a stage after it first clears the threshold (the
documented draw-with-shaping policy drift); by the time the stage
promotes (patience-2) or exhausts its budget, the in-memory policy
is often the drifted post-peak version, not the peak itself.
Meanwhile `<stage>/best_model.zip` *was* being saved by the eval
callback but **never reloaded** between stages.

v29 (deep economy + reward-fixed + `max_turns=100` + modern code,
all confounds controlled) entered `random_15` from a drifted
post-`random_10` policy and stalled identically to v28. v30
(single-stage `random_15` warm-started from v29's *peak*
`random_10` checkpoint via `set_parameters`) cleared `random_15`
in ~50k steps. Same code, same stage, same opponent — the only
difference was which `random_10` policy entered. Conclusive
evidence that drifted-handoff was the bug, not the stage.

**Fix:** `CurriculumConfig.restore_best_checkpoint_between_stages`
(default `True`). After a stage promotes, the runner reloads
`<stage>/best_model.zip` into the in-memory model
(`set_parameters(..., exact_match=True)`) before the next stage.
Hands forward the peak, not the drift. Also improves the final
stage: `final_model.zip` becomes the best of the last stage rather
than the drifted end-of-run policy.

### Bug 2 — strong handoffs cause stages to "skip ahead" past their own learning

The handoff fix was necessary but not sufficient. v28 run
`20260522_163958` (handoff fix live + modern balance + reward
fix, all verified via `config.json`) still stalled at
`beginner_random_15`. The CSV revealed the mechanism:

| stage | timesteps | WR | note |
|--|--|--|--|
| `beginner_balanced_random` | end | promoted | best handed to next stage |
| **`beginner_random_10`** | **@250,008** (first eval, ~8 steps into the stage) | **1.0** | random_10 trivially won by the carry-in policy |
| `beginner_random_10` | 300,000 | 0.01 | PPO updates drifted the policy |
| `beginner_random_10` | 350,000 / 400,000 | 0.925 / 0.84 | recovered → promoted |
| `beginner_random_15` | first eval | 0.91 | inherited *the @250,008 snapshot* (best ever in the stage) |
| `beginner_random_15` | next eval | 0.0 → 3M-step chaos | collapse |

`best_model.zip` for `random_10` was the *@250,008* snapshot — the
moment random_10 began, before any random_10-specific learning. So
the handoff into `random_15` was effectively `balanced_random`'s
best policy with ~0 random_10 refinement. A policy strong enough
to beat random_10 by transfer isn't necessarily strong enough for
the harder `random_15` opponent, and PPO updates against
random_15's reward signal destabilized it immediately.

**Stages can "skip ahead" past their own learning when the carry-in
policy is already strong.** The promotion gate (patience=2) doesn't
guard against this — it only requires sustained competence, which
a strong handoff can produce instantly without any stage-specific
training.

**Fix:** `CurriculumStage.min_timesteps_before_promotion: int = 0`
(default 0; legacy behaviour). When `> 0`, `PromotionCallback`
ignores eval results — and refuses to promote — until
`model.num_timesteps >= min_timesteps_before_promotion`. The
streak counter is reset and pre-window evals are discarded
entirely, so promotion can only build from post-gate evals. Set
to `500_000` on the noisy stochastic `*_random_N` stages in
`v31_production_minsteps_gate.yaml` (the successor to v28; v28
left untouched as the historical record of the bug that motivated
the gate).

### Lessons

1. **Handoff ≠ progress.** Carrying a peak policy forward only
   helps if each stage has *actually trained* before the
   handoff. The promotion gate measures sustained competence,
   not amount of learning — those are different things and the
   curriculum's correctness depends on both.
2. **Inspect the *content* of `best_model.zip`, not just its
   existence.** The handoff fix appeared to work (the next stage
   started at 91% — looked great). It only failed under sustained
   training on the harder stage, where the under-trained-for-this-
   opponent policy couldn't hold up. Don't trust early-eval WR
   transfer as evidence of robust learning.
3. **Strong carry-in is a double-edged sword.** A great handoff
   makes the *current* stage trivially passable on first eval,
   which short-circuits learning *for the next stage*. The
   `min_timesteps_before_promotion` gate makes "amount of stage-
   specific learning" a first-class, configurable curriculum
   property — alongside threshold and patience.
4. **One v-number per isolated variable.** v28 (handoff only, no
   gate) and v31 (handoff + gate) are kept as separate sweep
   configs so the experimental narrative is legible: each
   v-number corresponds to one set of run records and one
   variable change vs its predecessor.

### ⚠️ CORRECTION TO THE SKIP-AHEAD DIAGNOSIS — the gate hurt more than it helped

The "Bug 2" section above is preserved as the *hypothesis we
tested*. v31 (`min_timesteps_before_promotion: 500_000` on
`beginner_random_{10,15,20}`) was run on 2026-05-23 to validate it
and **stalled one stage earlier than v28 — at `beginner_random_10`
itself.** The skip-ahead premise was wrong.

**The data (run `20260523_042220`, `beginner_random_10`):**

| timesteps | WR | note |
|--|--|--|
| 250,008 | **1.00** | pre-gate (discarded by callback) |
| 300,000 | 0.01 | pre-gate (discarded) |
| 350,000 | **0.925** | pre-gate (would-have-promoted with v28) |
| 400,000 | **0.8375** | pre-gate (would-have-promoted with v28) |
| 450,000 | 0.0375 | pre-gate; policy starting to drift |
| 500,000 | 0.5875 | gate opens; policy already degraded |
| → 1.75M | 0.0 – 0.5, never two ≥ 0.75 | post-gate chaos |

Under v28's natural rhythm (no gate), `random_10` promoted on the
`@350k / @400k` pair (0.925 + 0.8375). v31 discarded that pair as
pre-gate and **kept training past the policy's peak**, which
**degraded** the working policy via the draw-with-shaping
attractor — the same mechanism the doc has warned about all along,
but happening *within* `random_10` itself rather than at the
handoff. The policy peaked early, additional training erased it.

**Reconciling with v30 (the diagnostic that worked):** v30
warm-started from **v29's** `random_10` best, which sat at
cumulative ~600k env-steps (v29's longer probe prefix). v28's
`random_10` best was at cumulative ~250k. The difference between
"holds `random_15`" and "collapses on `random_15`" was **total
cumulative training over the curriculum**, not stage-specific
`random_10` training. The skip-ahead diagnosis aimed the gate at
the wrong axis.

**Also revealed: an implementation flaw, kept for the record.**
`PromotionCallback.min_timesteps` is compared against
`model.num_timesteps`, which is cumulative (since
`reset_num_timesteps=False`). So a 500k value on `random_10` only
forced ~250k of stage-specific training, not the 500k intended.
Fixing this to be stage-relative would not help (and would in
fact hurt more — the empirical signal is that *less* post-peak
training is better, not more).

**Updated lessons (supersede points 3–4 above):**

3. **PPO's natural promotion rhythm beats forced additional
   training on stages with the drift attractor.** Once a policy
   crosses the competence threshold on a `*_random_N` stage,
   continued PPO updates tend to erode the winning behaviour
   rather than improve it. Don't lengthen these stages
   artificially.
4. **The amount-of-training-needed axis is *cumulative*, not
   per-stage.** v30 cleared `random_15` not because it had more
   `random_10` training, but because the policy entering
   `random_15` had more total experience across the curriculum.
   If a later stage needs a more robust entry policy, the lever
   is the *earlier* stages or a separate BC warm-start — not
   forcing the immediately-prior stage to overtrain past its
   peak.
5. **`min_timesteps_before_promotion` is kept in the code as an
   opt-in feature, default 0, NOT set by any production config.**
   It may be useful for non-stochastic stages where overtraining
   doesn't risk the drift attractor — but `*_random_N` stages
   should not use it.

**Current production target (`v32_drop_gate_higher_eval.yaml`):**
v28 with no gate, plus the long-deferred eval-noise lever applied
to `beginner_random_15` (`n_eval_episodes: 80 → 160`, per-stage
override). σ on the WR estimate halves, so a competent-but-noisy
random_15 policy can sustain the patience-2 streak. v28 and v31
are both retained as historical records ("no gate" and
"with gate" experiments respectively).

## The curriculum-tuning sweep (v15–v23): what 9 variants taught us

After the original 6-stage layout shipped, a long sweep
(`configs/ppo/bootstrap_sweep/v15…v23`) tried to push the agent past the
`beginner_random_*` stages on a longer curriculum that adds
`balanced_random` stepping stones, a `consolidate` stage, and a tail
of intermediate / skirmish / corner_points maps. **Every run since v17
has stalled at `beginner_random_15`.** v19 is the high-water mark: 6
stages cleared, stalled at random_15 with peak WR 0.74. v20, v21, v22
all regressed. The lessons below are as expensive as the cold-start
ones above — read before touching the curriculum again.

### TL;DR (sweep edition)

1. **v19 is the best config. Every "make it simpler" change regressed.**
   v20 (lower entropy), v21 (split-entropy consolidate), v22 (reduced
   unit roster) each removed something the agent implicitly relied on
   and each cleared *fewer* stages than v19. The careful balance is
   load-bearing; do not assume simplification is free.
2. **Entropy is load-bearing for *generalization*, not just exploration.**
   Lowering `beginner_balanced_random` start entropy 0.07 → 0.05 (v20)
   caused a catastrophic regression at random_10 (90% → 0% in one eval,
   all draws). The high entropy was producing a policy robust to the
   opponent-class change, not merely "noisy."
3. **The entropy *step* at a stage boundary is NOT the consolidate
   shock cause.** v21 split v19's single 0.05→0.003 consolidate into
   two smaller steps (≈no boundary jump) and the policy dropped HARDER
   (86% → 8.75%). Disproved the boundary-step hypothesis cleanly.
4. **Reducing the unit roster is confounded with opponent strength.**
   v22 (`enabled_units: [W,K,A,M]`) made `starter_random` — the
   *easiest* stage — unlearnable. Removing Cleric/Sorcerer/Rogue/
   Barbarian stopped RandomBot from wasting ~25% of its builds on
   units it can't use randomly, turning the "random" opponent into a
   much stronger one. The early curriculum's easiness was partly an
   artifact of opponents handicapping themselves with weak units.
5. **`patience` against stochastic opponents must be low.** patience=4
   demands 4 consecutive evals ≥ threshold against a noisy RandomBot —
   effectively asking for skill *and* luck. v17/v18 dropped random_N
   stages to patience=2; that fixed random_10 (cleared in 50k steps
   where v16 stalled the full 1.5M budget). It did not fix random_15
   (the policy never sustains the bar there).
6. **Lowering the gate ≠ fixing the policy, but it answers a different
   question.** v23 relaxes random_15/20 promotion 0.70 → 0.60 (a level
   v19 demonstrably sustained) purely to learn whether the
   post-random_15 curriculum is *reachable at all*, or whether the
   stall is a genuine capability ceiling. It is explicitly the last
   curriculum-config experiment.

### The draw-with-shaping equilibrium (the random_15 stall mechanism)

Across v17/v18/v19/v21, the random_15 stall has one consistent
signature: the policy reaches a winning composition (Warrior+Knight,
sometimes +Mage), then **drifts off it** over ~hundreds of k steps
into a max-turns-draw attractor, and never recovers within the stage
budget.

The mechanism: against a *static stochastic* opponent there is no
curriculum gradient *inside* the stage. Once PPO finds a winning
policy, nothing pulls it back if it wanders, because the value
function correctly estimates that "stall to max_turns" has a higher
variance-adjusted return than "risk a loss attempting to win" given
the current weights. `turn_penalty = -0.5` (added in v16) reprices
drawing from ≈+45 expected to ≈−40, which *partially* counters this —
but the policy still drifts there because its *alternative* (commit to
winning) scores even worse when it isn't a competent winner. The
penalty changed the value, not the behaviour.

### The decisive diagnostic: drift vs plateau

The single most useful discriminator we found this sweep:

| Signal | random_15 stall (what we see) | capacity-bound (what it'd be) |
|--------|-------------------------------|-------------------------------|
| `explained_variance` | ~0.85–0.95 | low, ~0.3–0.5 |
| `value_loss` | low, stable | high, possibly climbing |
| `approx_kl` | ~0.005–0.01 (barely updating) | higher (thrashing) |
| WR trajectory | **drifts** off a winning policy | **plateaus** below optimal |

The observed signature is "value function fits returns very well,
policy barely updating, but the policy is *wandering*." That is an
**optimization / attractor** problem, not a representational-capacity
problem. A bigger network would fit the same returns and reach the
same "drawing is best" conclusion. This is why we did *not* jump to
architecture changes during the sweep — the data didn't support
capacity-bound. (When curriculum tuning is exhausted after v23, the
discriminating control is: train a fresh policy on random_15 alone,
no curriculum, 3M steps. If it clears, random_15 is solvable at this
capacity and the curriculum failures are interference/forgetting; if
it can't, it's a genuine ceiling and the network must grow.)

### GPU non-determinism amplifies divergence

v19 and v21 share an identical config through `beginner_random_10`
(same seed). v19's r10 promote eval hit 100%; v21's hit 90%. CUDA
atomic-add non-determinism over ~50k training steps + ~24 PPO updates
is enough to send otherwise-identical runs into different basins. When
comparing runs, treat ±10% at a promotion eval as noise, not signal —
and don't over-interpret a single run. This is a known deep-RL
reproducibility limitation, not a config bug.

### Warm-starting is not shareable

We added a dormant `warm_start_path` to `TrainingConfig` /
`run_curriculum` (loads policy+optimizer via `set_parameters` before
stage 1). It works, but a config that points at a *specific prior
run's* checkpoint is not reproducible from a clean checkout — other
people can't run it. Keep warm-start as opt-in infrastructure; do not
build a sweep variant that *depends* on it. This is why v23 pivoted
from "warm-start the post-consolidate policy into random_15" to the
self-contained gate-relaxation.

### Knight buff: invisible to scripted bots, visible to RL

A separate but related finding from the balance work: bumping Knight
defence 5→7 produced *byte-identical* bot-tournament outcomes (scripted
bots use static heuristic priorities and never perceive a stat change)
but a measurable +3.6pt Knight survival-rate change in replay analysis,
and RL agents *do* build more Knights post-buff (gradient discovers the
durability). **`balance_analysis` (bot tournaments) cannot validate
RL-relevant stat changes.** Use replay-level metrics or RL training
deltas, not bot win/loss, to assess unit-stat tuning.

**Update (post-`rng_seed` plumbing):** the "byte-identical" framing
above was *partly* a deterministic-replay artifact, not solely a
function of static-priority bots. With `games_per_side=1` and no rng,
the entire tournament was 12 unique games × 8 replicas, so any stat
change that didn't flip a deterministic decision branch produced
literally identical bytes. With stochastic mode the bot-tournament
*outputs* (winrates, capture counts) now have real statistical content
— but the static-priority point still holds: a stat change that
doesn't alter any sort key won't shift scripted-bot behaviour, no
matter how many independent samples you collect. The new contribution
is that replay-level metrics (gold shares, build counts, survival
rates) are now statistically meaningful at the per-game level, which
makes them usable for stat-tuning signal. See
[`docs/balance_analysis_lessons_learned.md`](balance_analysis_lessons_learned.md)
for the full balance-analysis retrospective.

### What is *not* worth chasing again (sweep edition)

- **Entropy-schedule variants.** Two clean negatives (v20, v21). The
  problem is not entropy shape. Lower entropy reduces generalization;
  smaller entropy steps don't soften the consolidate shock.
- **Unit-roster reduction as a "simplification."** It changes opponent
  strength (v22). If unit-complexity must be tested, it requires
  *asymmetric* rosters (agent-only restriction) which needs an env-
  factory code change, not a YAML toggle.
- **More `patience` / threshold permutations beyond v23.** patience is
  already 2 on random_N; v23 is the last meaningful gate move. After
  v23 the curriculum-config search space is spent.
- **Re-running for luck.** Same as the cold-start lesson: a stall that
  holds for ~1M steps does not unstall by extending the budget. v17's
  random_15 ran the full 3M and never recovered.

## Future work

- **Decide network changes off v23 + a single-stage control.** v23 is
  the terminal curriculum experiment. If it stalls (or clears random_15
  but a downstream stage walls with the same drift signature), widen
  the MLP `net_arch [256,256] → [512,512]` first (cheapest meaningful
  capacity bump, ~1.3M params, ~1.5–2× wall-clock), then `features_dim`
  + CNN width, then a unit-attention block. Run the single-stage
  random_15-only control in parallel to discriminate capacity-bound
  from reward-shaping before committing.
- **Opponent diversity in random stages (Option C).** The diagnosed
  draw-attractor mechanism is "static opponent → no in-stage gradient
  → drift." A per-episode *mixed* RandomBot (e.g. max_actions sampled
  from {10,15,20}) would give a multi-modal gradient that resists
  collapse. Requires extending `MixedBot._build_inner` to support
  `random`/`balanced_random` sub-bots with kwargs (it currently only
  supports simple/medium/advanced). Mid-effort; the most principled
  remaining lever if v23 + bigger network don't crack random_15.
- A **CNN features extractor** over the `grid` and `units` channels
  would probably accelerate convergence, especially as we move to
  bigger maps. Mid-effort (~80 LOC). *(Note: a `SpatialFeatureExtractor`
  was subsequently shipped — this item is largely done; the open part
  is unit-attention on top of it.)*
- **Drop `action_mask` from the observation dict for MaskablePPO**
  via a small custom features extractor. It's redundant with
  MaskablePPO's masking pipeline and currently consumes ~70% of the
  network's input dimensionality. Note: AZ and Feudal RL pipelines
  *do* use the obs-mask and would break if we removed it globally —
  has to be a per-policy feature extractor, not an env change.
- **Behaviour-cloning warm-up** is now built (`reinforcetactics/rl/
  imitation.py` + section 3c of `ppo_bootstrap.ipynb`) and validated
  on the beginner curriculum (v33 cleared random_15 with BC). See
  "BC warm-start: what we learned" below for the failure modes that
  surfaced during validation. Open follow-ups: pad_to_size threading
  for multi-size-map BC, and decoupling BC's dataset config from the
  curriculum's first-stage map.
- **`start_fresh` flag on `CurriculumStage`** to optionally
  reinitialise the model at a stage boundary instead of always
  reusing weights — useful when a map shift is too steep for the
  starter policy to bridge.

## BC warm-start: what we learned

Behaviour cloning (BC) was originally listed as future work; we built it
during the v33 cycle to break the random_15 wall, then refined it on a
single-map skirmish probe before the second wall (BC -> AdvancedBot
directly, 0% WR for 750k steps). The validation work surfaced several
failure modes that aren't obvious from "just clone the bot's actions"
intuition and that recur if the diagnostics aren't built in.

### Failure mode A: deterministic bots produce 1 unique trajectory per N episodes

**Symptom**: BC dataset has 60 demos from a scenario, but the per-scenario
`avg_turns` comes back as an *exact integer* (18.0, 21.0, 14.0). The 60
episodes were 60 byte-identical replays of one game. Compare to scenarios
with a stochastic opponent (random, balanced_random), where `avg_turns`
is fractional (21.9, 10.6) — those produced 60 actually-distinct games.

**Cause**: `SimpleBot` / `MediumBot` / `AdvancedBot` / `MasterBot` are
fully deterministic on game state. Their sort / max / best-tracking
sites resolve ties by data-structure iteration order (Python's stable
sort, leftmost-on-ties `max()`). Two games from the same starting state
produce the same decisions at every step. `_make_bot` seeds them an
`rng`, but they don't consume it.

**Diagnostic**: per-scenario `ScenarioStats` table (added in this
branch, prints from notebook section 3d) — the W/L/D + avg_turns
columns reveal duplicate-game scenarios at a glance via the
exact-integer-avg-turns signature.

**Fix**: opt-in `stochastic_tiebreak: true` on each
`DemonstrationScenario`. The bots receive a per-episode rng and
`_maybe_shuffle()` the input to every sort / best-tracking site
before the deterministic comparison runs. Scoring logic is unchanged
— the bot still picks among its top-rated options — but ties resolve
randomly, so N episodes give ~N unique trajectories.

**Why it matters for BC**: a duplicate-heavy dataset lets the supervised
policy *memorise* a handful of game scripts. Action-type accuracy
looks great (81% on the first run) but reflects pattern matching, not
generalisation. With stochastic-tiebreak diversity, action-type
accuracy drops to 76% — that's the *real* ceiling on a representative
distribution.

### Failure mode B: deterministic-tiebreak artifacts look like structural map bias

The first skirmish BC build showed `advanced_vs_advanced` as **0 wins
/ 60 losses for player 1**. We diagnosed this as "the skirmish map
favours player 2" and flipped `demonstrator_player: 2` to record from
the winning side.

With stochastic tiebreak enabled on the next build, the same matchup
came back as **20 wins / 40 losses for player 2** — i.e. **player 1
wins ~67% of the time**. The original 0/60 was not structural map
bias; it was a single fixed tiebreak sequence that happened to
deterministically punish player 1's move order. Tiny perturbations
flipped the matchup wholesale.

**Lesson**: do not infer "map X favours side Y" from
deterministic-bot-vs-deterministic-bot outcomes. The deterministic
tiebreaks are an unmodelled adversarial parameter, and changing it
can invert the conclusion.

### Failure mode C: `make_warm_started_model` single-source path silently ignores `stochastic_tiebreak`

When the BC infra was first wired, `stochastic_tiebreak` lived on
`DemonstrationScenario` but **not** on `make_warm_started_model`'s
single-source kwargs. Calling `make_warm_started_model(env,
demonstrator='advanced', opponent='advanced', n_episodes=50)` without
`scenarios=` silently produced 50 byte-identical games regardless of
the flag — the exact failure the flag is supposed to fix, with no
way to opt in. The notebook uses `scenarios=` so it was fine, but any
ad-hoc caller hit the duplicate trap.

**Fix**: thread `stochastic_tiebreak` through every BC entry point.
Default `False` everywhere preserves backwards compatibility, but the
flag has to be reachable from every caller.

### Failure mode D: MixedBot's inner bots were deterministic

`MixedBot(easy=medium, hard=advanced)` resamples easy-vs-hard per
episode (`self.use_hard = self._rng.random() < p_hard`) but
`MixedBot._build_inner` originally constructed the chosen inner bot
without forwarding `rng`. So even with `stochastic_tiebreak=True`,
each "Medium episode" played out identically to every other "Medium
episode" — you got ~2 unique trajectories total (one per inner
choice), not N. The PPO curriculum's `skirmish_mixed_medium_advanced`
bridge stage was affected too (each "side" of MixedBot was
deterministic).

**Fix**: `_build_inner` now accepts and forwards `rng`. Caught only
because the per-scenario stats showed identical `avg_turns` on a
MixedBot scenario after we'd "fixed" the deterministic-bot issue
elsewhere.

### Failure mode E: action-loop timeouts silently inflate the "draws" count

`_play_episode`'s outer step-budget cap (`max_turns * 4 + 50`) is a
defense against bots stalling in an intra-turn action loop. When it
fires, `game_state.game_over` is still False and `winner is None` —
which `EpisodeOutcome.is_draw` correctly treats as a draw. But this
conflates "game legitimately ended without a winner at the
`max_turns` cap" (a property of the matchup) with "bot got stuck in
an action loop" (a bot bug). The per-scenario draws column inflates,
and the user can't tell which.

**Fix**: when the step-budget cap fires before `end_game` runs, set
`end_reason = "step_budget_exhausted"` and bump a separate
`step_budget_exhausted` counter on `ScenarioStats`. The formatter's
`T` column makes timeouts visible alongside W/L/D so a
duplicate-game scenario or bot bug surfaces immediately.

### Failure mode F: shared rng desyncs stochastic-bot trajectories when toggling `stochastic_tiebreak`

When `stochastic_tiebreak=True`, the same per-episode rng powered both
deterministic-bot tiebreak shuffles and stochastic-bot action draws
(RandomBot/BalancedRandomBot/MixedBot's coin flip). Toggling the
flag on the same seed shifted RandomBot's `.choice()` draws because
AdvancedBot's prior `_maybe_shuffle` consumed rng state. The two
modes were not reproducibility-comparable.

**Fix**: `_play_episode` derives a separate `tiebreak_rng =
random.Random(seed ^ 0xC0FFEEBAD)` for the deterministic-bot
tiebreaks. The stochastic-bot stream is unchanged regardless of the
flag.

### Failure mode G: sanity-eval env construction omits production env kwargs → measurement artifact looks like a BC failure

This one cost us a full iteration almost-misdiagnosed. The newly-added
section 3e ("BC sanity eval") in `ppo_bootstrap.ipynb` built its eval
env via:

```python
sanity_env = make_maskable_env(
    map_file=first_stage.map_file,
    opponent=opp,
    max_turns=first_stage.resolve_max_turns(cfg.env),
    max_steps=first_stage.resolve_max_steps(cfg.env),
    enabled_units=cfg.env.enabled_units,
    action_space_type=cfg.env.action_space_type,
    seed=cfg.seed + 7777,
)
```

Notice what's missing: **`reward_config` and `max_actions_per_turn`**.
When those aren't passed, `make_maskable_env` →
`StrategyGameEnv.__init__` falls back to the env's
*built-in defaults* (`gym_env.py:406-454`):

| Knob | YAML (skirmish_bc_selfplay) | Env default | Ratio |
|---|---|---|---|
| `invalid_action` | -0.01 | **-10.0** | 1000× more punitive |
| `draw` | -50 | **-200** | 4× more punitive |
| `loss` | -50 | **-1000** | 20× more punitive |
| `max_actions_per_turn` | 60 | **None** (disabled) | Safety net gone |

The first three made the reward magnitudes uninterpretable. The fourth
caused a structural eval failure: without `max_actions_per_turn`,
nothing forces end_turn after N agent actions. The BC policy
under-predicts end_turn at greedy decode and produces per-dim-legal-
but-jointly-invalid action tuples (Failure mode 4 from the agent
audit -- per-dim MaskablePPO masking is an over-approximation). The
agent loops on those invalids for the full `max_steps=3000` budget
without ever advancing a single game-turn, then truncates as
`max_steps_truncate` (classified as a draw).

**Observed symptoms in run 20260524_225835**:
```
BC vs simple   WR= 0.0%   reward=-30169.0   W/L/D=0/0/30
```

Arithmetic checks: `3000 invalid_actions × -10 = -30,000` plus
`-200` default draw terminal plus small potential-shaping bias
≈ `-30,200`, matching observed `-30,169` to within shaping noise.
The 0/0/30 outcome was every episode looping on invalids until
`max_steps`, not "BC can't beat SimpleBot."

**The misdiagnosis we almost shipped**: the obvious read of the
output was "BC is fundamentally broken; bump `end_turn_weight` from
auto-balance to 30 to force more end_turn prediction." That's the
exact wrong fix -- the BC policy *was* under-predicting end_turn,
but with the production env's `max_actions_per_turn=60` safety net
the cap would have forced end_turn after 60 invalids and the agent
would have actually played the game. Disabling the safety net in
the sanity eval inverted the diagnosis.

**Fix**: section 3e now mirrors what section 6's post-curriculum
sanity eval and `run_curriculum`'s in-training eval already do
correctly -- forward `reward_config = first_stage.resolve_reward_config(cfg.env)`,
`max_actions_per_turn = cfg.env.max_actions_per_turn`,
`pad_to_size = cfg.env.pad_to_size`. Also added `avg_length`,
`avg_turns`, and `end_reasons` to the printed output so the
never-end-turn failure mode is visible inline rather than buried in
the reward magnitude.

**General lesson**: any "ad-hoc" env construction outside of
`run_curriculum` (sanity evals, replay videos, hand-rolled
debugging probes) has to forward the *same* kwargs the production
env uses, not just the obvious ones (map / opponent / max_turns).
`reward_config` is the easy one to forget because it doesn't
change *behavior*, only *measurement* -- but the env's built-in
defaults are 10-1000× off from the YAML values across most keys,
so any reward number from a not-forwarded env is uninterpretable.
`max_actions_per_turn` is the dangerous one to forget because it
*does* change behavior in a way that masquerades as a different
failure mode.

The kwargs that must be forwarded for measurement consistency:

```python
make_maskable_env(
    map_file=stage.map_file,
    opponent=opp,
    max_steps=stage.resolve_max_steps(cfg.env),
    max_turns=stage.resolve_max_turns(cfg.env),
    reward_config=stage.resolve_reward_config(cfg.env),  # essential
    enabled_units=cfg.env.enabled_units,
    action_space_type=cfg.env.action_space_type,
    max_actions_per_turn=cfg.env.max_actions_per_turn,    # essential
    pad_to_size=cfg.env.pad_to_size,
    opponent_kwargs=stage.opponent_kwargs,                # if applicable
    seed=cfg.seed + <fresh offset>,
)
```

### Failure mode H: end_turn_weight upweights loss, not argmax priors

The class-imbalance auto-rebalance in `behavior_clone` (default
`end_turn_weight = n_non_end / n_end ≈ 10`) was originally added to
counteract the ~10:1 imbalance between non-end_turn and end_turn
demonstrations. v33's notes describe BC's "never-end-turn attractor"
as a consequence of the imbalance, and the auto-rebalance was meant
to neutralise it. Skirmish run 20260525_015401 tested whether
*explicitly bumping* the weight to 30 (3× the auto-balance value)
would push the BC policy out of the attractor.

**It did not.** Same map, same seed, same scenarios, only
`end_turn_weight=30.0` changed. Sanity-eval results vs SimpleBot
were **byte-identical** to the auto-balance run:

```
                end_turn=auto (~10×)    end_turn_weight=30.0
BC vs simple    +796.6 / 50 turns       +796.5 / 50 turns
                W/L/D 0/0/30            W/L/D 0/0/30
                end=max_steps_truncate  end=max_steps_truncate
```

But the BC *training* metrics improved substantially across the same
60 epochs:

```
                Loss        action_type_acc   full_action_acc
auto-balance    5.0 → 2.73  0.60 → 0.78       0.08 → 0.21
weight=30.0     3.0 → 1.21  0.57 → 0.83       0.09 → 0.27
```

Lower loss, higher accuracies on every axis. The supervised policy
is **objectively better trained** under `end_turn_weight=30`. It
just doesn't play any differently at greedy decode against the
passive opponent.

**Why the gap**: the loss weight increases the gradient contribution
from samples where `end_turn` is the correct label, so the policy
predicts `end_turn` more accurately *when faced with states where
end_turn is the correct answer in the demonstrations*. It does NOT
shift the argmax prior in states where no specific demonstrator
action is strongly correct. At sanity-eval time, the BC policy
encounters game states the demonstrator would have played a
move/attack/capture on (because scripted bots are aggressive); the
argmax over those high-logit non-end_turn actions still wins, and
the agent never voluntarily ends its turn.

The asymmetry is structural: **upweighting samples ≠ shifting
argmax priors at unrelated states**. The class-imbalance argument
assumes the policy must "learn that end_turn is a possible
action"; but the policy already knows that — it just ranks end_turn
below the most likely non-end_turn action at most states, and
weight scaling alone doesn't flip that ordering on the states where
the demonstrator was making a productive move.

What might actually shift the argmax (untested, ordered by
intrusiveness):

1. **Much higher weight (100, 500, 1000).** Diminishing returns
   expected — the loss eventually saturates and gradients to
   non-end_turn samples vanish, but the relative logit ordering at
   inference may still favor whichever non-end_turn action had the
   most demos. Quick to try (a hyperparam sweep), low-risk fallback.
2. **Asymmetric / margin-based loss.** Penalize wrong-end_turn
   predictions more than wrong-non_end_turn predictions, or add a
   margin term that pushes end_turn's logit up by at least δ above
   the next-best non-end_turn action when end_turn is correct.
   Bigger code change in `behavior_clone`.
3. **Different inference rule.** At eval time, pick end_turn if
   `max(non_end_turn_logit) - end_turn_logit < threshold` — i.e.,
   "no high-confidence productive action available, just end the
   turn". This bypasses the BC training issue entirely but is a
   post-hoc behavioral hack, not a model fix.
4. **`flat_discrete` action space.** End_turn becomes one of N
   equally-weighted action tokens; per-action masking instead of
   per-dim masking; the cross-entropy is well-defined over a flat
   distribution. Probably the cleanest model-level fix but
   requires extending `imitation.py` to record flat_discrete
   demonstrations.

**Lesson**: don't conclude that a hyperparameter "didn't work"
purely from supervised metrics — also don't conclude it "worked"
from supervised metrics alone. The training-curve view (loss
falling, accuracy rising) can hide an unchanged argmax behavior
when the metric and the gameplay policy diverge. The
`evaluate_bc_against_bot_ladder` sanity eval is the diagnostic
that catches this — without it, we'd have happily shipped
`end_turn_weight=30` and discovered the same 0% WR at PPO eval
@ step 8 of the next curriculum run.

**Concrete sequence for future BC iterations on this codebase**:
sanity eval first (cheap, ~1 min), training metrics second (load
JSON), curriculum run third (expensive). If sanity-eval WR vs
Simple is unchanged after a hyperparam tweak, abort before the
curriculum run — additional PPO compute won't fix what BC didn't.

### Diagnostic discipline: print per-scenario stats *before* PPO

Three of the six failure modes above (A, B, E) were caught by the
`format_scenario_stats_table` print in notebook section 3d, which
exists *only because we built it*. A BC checkpoint with the wrong
demonstrator side, or with 12% duplicate-game labels, or with most
"draws" actually being bot-stall timeouts, will all train without
error and produce a checkpoint that looks superficially fine. The
diagnostic columns (`W/L/D/T` plus `avg_turns`) are the cheapest
signal we have for catching these *before* a multi-hour PPO run.

**Rule of thumb**: an exact-integer `avg_turns` value on any
all-deterministic-bot scenario means N copies of one game. Action-loop
timeouts in the `T` column mean a bot bug, not a map property. A
demonstrator WR substantially below 50% on a same-bot matchup
(`X vs X`) means you're recording from the losing perspective.

### What BC's training metrics actually tell you

Per-epoch BC stats (`bc_training_stats.json` + `bc_training_curves.png`)
have a specific shape we now understand:

- **`loss`** falls cleanly. Always. If it doesn't, the dataset / model
  config is broken before BC-specific concerns apply.
- **`action_type_acc`** rises sharply for the first 3-4 epochs then
  **plateaus around 70-80%** on diversified data. More epochs do not
  help this metric meaningfully. It is the ceiling of the action-type
  classifier on the dataset's action distribution, not an
  under-training signal.
- **`full_action_acc`** rises slowly throughout training. The (from,
  to) coordinate predictions take longer than the type classification
  to converge. Strict-match metric so the absolute number is low
  (~14% at epoch 8) — *don't read this as bad*; even a near-optimal
  policy can pick a different-but-valid target than the demonstrator.

**Implication for `BC_EPOCHS`**: stop tuning by action_type_acc — it
caps fast on diversified data. 8 epochs is fine; 16 mostly helps
full_action_acc and at diminishing returns.

### What BC's value head doesn't learn

`behavior_clone` updates the policy network (features extractor +
shared MLP + action head) but **not** the value head. BC has
supervised labels for actions but no labels for state values; we
could estimate values from terminal outcomes but they'd be noisy and
would bias PPO's bootstrapping. The value head starts fresh and PPO
fits it during the first few updates.

**Consequence**: expect a `value_loss` spike in the first 1-2 PPO
updates that decays as the critic catches up. Healthy. The
diagnostic is when the spike *doesn't* decay — that signals either
reward-scale issues or a value-head capacity ceiling.

### Single-map probe pattern (do this every time)

Before any multi-hour curriculum run on a new map / new BC mix:

1. Drop `total_timesteps` and stage `max_timesteps` to ~1M total.
   That's ~60 PPO updates at our hyperparams, ~20 eval points.
2. Confirm `WR > 0` with `std > 0` within the first 3-5 evals.
3. Confirm reward is not falling (yellow flag) and avg_turns isn't
   monotonically climbing toward `max_turns` (the stall-attractor
   signature from the v15-v23 sweep).
4. Only after the probe shows upward signal, raise budgets back to
   production scale.

The first skirmish run was a 10M-budget run that held WR=0.0% with
std=0.0 for 750k steps before we stopped it. The probe pattern catches
that signature in ~15 minutes of wall-clock instead of multiple hours,
and the saved diagnostic artifacts (`scenario_stats.txt`,
`bc_training_curves.png`, `bc_demo_outcomes.png`) live on Drive so
they survive Colab disconnects and are comparable across iterations.

## Composition diversity: the mono-Warrior attractor (v34–v40)

After the BC pipeline cleared `beginner_random_15` (v33), the next
goal was teaching the agent to **play with composition** — use
Mages/Clerics/Knights, capture HQs, exercise the status-effect channels
added in PR #383. v34 cold-started instead and surfaced a different
problem: PPO can clear most of the curriculum without ever learning
composition at all. v34–v40 chased the fix; the lessons below are
what those seven configs cost.

### The v34 finding: 14 stages cleared, zero strategy learned

`v34_aggressive_combat.yaml` re-enabled the combat-shaping reward at
the rescaled magnitudes the docs recommended (`damage_scale=0.002`,
`kill=0.2`, `turn_penalty=-0.5`, `draw=-10`). Run `20260526_145412`
cleared **14 stages** cold — through `beginner_advanced` and into
intermediate — then stalled at `intermediate_random_20` at 60% WR.
The headline number was good. The composition data wasn't:

- `units_built["W"]` was **95–100%** of total builds on **every**
  cleared stage. M/C/A/K/R/S/B were single digits or zero.
- `captures_by_type.hq = 0` across every stage. The policy won
  exclusively by elimination, never by HQ rush.
- Status-effect observation channels (paralyze/haste/buffs) had
  zero signal because the policy never built the units that produce
  them.
- `avg_turns` pegged at the 60-turn cap on the stalled stage:
  intermediate's 7×7 map at max_turns=60 (smaller clock than 6×6
  beginner's 75) ran out as draws (80/80 max_turns_draw on the
  stall).

The policy had learned a working but trivial strategy — "spam the
cheapest unit with the best HP/$ + Atk/$, eliminate the opponent
before the clock runs out" — which works on small maps with
non-aggressive opponents and stops working immediately when geometry
or opponent strength changes.

### Why the default balance is a one-unit local optimum

Before v34, default Warrior stats made it pareto-dominant per gold
spent: cost=200 (cheapest), HP=15 (tied highest), atk=10 (tied
highest), def=6 (highest of any unit). PPO finds this immediately
and never explores. The status of the cost-efficiency table mattered:

```
  W: cost 200  HP/$ 0.075  Atk/$ 0.050  Def 6   <- top of every axis
  K: cost 350  HP/$ 0.051  Atk/$ 0.023  Def 5
  M: cost 300  HP/$ 0.033  Atk/$ 0.033  Def 4
  A: cost 250  HP/$ 0.060  Atk/$ 0.020  Def 1
```

Once a single unit wins every cost-efficiency axis at the cheapest
tier, there is no in-distribution policy gradient toward diversity.
Entropy bonus generates random *exploration*, not random
*exploitation* — the policy will sample a Mage occasionally, the
Mage will under-perform vs spending the same gold on more Warriors,
and the gradient pushes back toward W.

### The four levers we tried (v35–v37)

| Config | Lever | First stall | Mono-W on cleared stages |
|---|---|---|---|
| v35 | Warrior nerf atk 10→8, def 6→4; intermediate r10/r15 added; intermediate max_turns 60→75 | `beginner_random_10` (composition diversifying) | drift in progress |
| v36 | v35 + `beginner_random_10` budget 1.5M → 3M (recovery time) | `beginner_random_10` (7 stages cleared) | 100% (5479/5479 = W on r10) |
| v37a | revert nerf + MixedBot bridge stage between r15 and r20 | `beginner_random_15` (CUDA non-det stall) | 100% |
| v37b | revert nerf + higher promotion thresholds (r10:0.75→0.90, r15:0.70→0.85, r20:0.70→0.75) | `intermediate_random_20` after **15 stages cleared at 100% WR each** | 100% on every cleared stage |

**v37b is the decisive negative result.** A policy that clears 15
stages at 100% WR — better than v34's deep run — with 0% composition
diversity confirms the mono-W attractor is **not** a curriculum
problem. No amount of mid-curriculum pressure (MixedBot bridge),
no threshold tightening (forcing more stage-specific training),
no soft stat nerf escapes it. The default balance is the cause.

### v38 (structural nerf) and v39 (cost-only nerf) hit a new wall

`v38_structural_warrior_nerf.yaml`: atk 10→7, def 6→3, HP 15→13.
This makes Knight strictly better than Warrior on every combat
axis. Run `20260527_200511` stalled at `starter_random` with **51%
peak WR** vs the 0.90 threshold. Diagnosis: combat math broken at
the smallest scale. With Warrior atk=7 facing def=4 units, kills
took 3–4 turns each, and starter's max_turns=20 ran out before
either side could resolve the fight — 50–70 draws per 80-episode
eval. The nerf was too aggressive *and* the test map was too small
to support the strategic tradeoff it was meant to teach.

`v39_cost_only_nerf.yaml`: Warrior stats unchanged, cost 200→300.
The motivation was surgical — change only the price (Warrior's
last advantage axis), keep combat math intact. Run
`20260528_005831` stalled at `starter_random` with **35% peak WR
— worse than v38**. Diagnosis was different and more fundamental:
starter has $250 starting gold and max_turns=20. At W cost=300,
the policy literally **cannot build a Warrior on turn 1**, and
gets ~1 unit per 2 turns of income afterward. Games run out as
draws regardless of whether the policy was "winning" the early
game.

### Diagnostic: balance experiments are incompatible with starter

The two stalls have the same root cause: **starter is too tight to
test balance**. Starter's role in the curriculum is to teach the
agent navigation cheaply — that's all. With the default balance
PPO can speedrun starter (v34 cleared it in 50–100k each), so
nothing about starter's economic structure ever mattered. Once
you change the cost curve or the combat math, starter's hard
limits (20 turns, $250 gold) collide with the new mechanics
before the policy can express the strategic shift the curriculum
is supposed to elicit.

The shape of the failure is general: **a curriculum stage that
worked at one balance setting may be infeasible at another, and
the failure looks like a policy bug** (low WR, lots of draws)
when it's actually an environment-clock bug. The diagnostic that
distinguishes them: look at the *gold-spent / units-built* ratio
on the stall — if it shows the policy spending almost everything
on units but games still draw out, the clock is the problem; if
gold is accumulating un-spent, the policy never learned to build.
v38/v39 both showed the former.

### v40's resolution

`v40_skip_starter.yaml` drops `starter_random` / `starter_simple` /
`starter_medium` entirely. The curriculum opens on
`beginner_balanced_random` (6×6 map, max_turns=75, $400 start
gold), keeps v39's W cost=300 nerf, and inherits the rest of v39
unchanged. Run `20260528_020718` cleared stage 1 in 200k env-steps
at 100% WR (W/L/D=80/0/0) on the first eval after the early-
exploration dip — the cost-curve hypothesis is testable here in a
way it wasn't on starter. Remaining stages are still TBD as the
run progresses; the diagnostic question is whether the composition
diversifies through `beginner_random_{10,15,20}` and especially
through `beginner_mixed_r15_simple`.

### Generalizable lessons

1. **Default-balance composition is the policy's prior.** If the
   default unit table has a clear cost-efficiency winner, every
   PPO run on this env converges to mono-that-unit composition.
   The curriculum doesn't push back — patience gates and stage
   thresholds reward winning, not winning *with composition*. If
   you want diversity, the cost curve has to break the per-axis
   monopoly. (Three units winning different axes is the v39 design:
   Knight HP/$, Mage Atk/$, Archer cheap-tier HP/$. Whether this
   actually breaks the attractor is the v40 test.)

2. **Stat nerfs and cost nerfs fail differently.** Stat nerfs
   (v38) break combat *math* and propagate to "the policy can't
   win at all" on tight-clock maps. Cost nerfs (v39) preserve
   combat math but break the *economy* at low-gold maps.
   Cost nerfs are reversible at finer granularity (300 → 275 → 250)
   and don't interact with combat dynamics; prefer them when the
   goal is composition shaping rather than per-unit rebalancing.

3. **Combat-shaping reward (v34) is **not** a diversity lever.**
   `damage_scale + kill` rewards finishing fights. The unit that
   wins per gold spent on fights is the cheapest competent
   combatant — i.e. the same unit the un-shaped policy was
   already converging on. Combat shaping unlocked v34's depth
   (14 stages vs prior ~7) but didn't change *what* the policy
   built; it just made the policy more aggressive about it.

4. **Promotion-threshold tightening doesn't force composition.**
   v37b raised r10 to 0.90 and r15 to 0.85 hoping the bar would
   force more stage-specific learning. The policy met the higher
   bars by **getting better at mono-Warrior on the same opponents**,
   not by diversifying. Patience and thresholds measure WR; WR is
   measurable without composition diversity; therefore tighter
   gates don't pressure for composition diversity.

5. **MixedBot bridges are useful for opponent transitions, not
   composition transitions.** The `beginner_mixed_r15_simple`
   stage (50% RandomBot at max_actions=15, 50% SimpleBot) was
   designed to punish mono-elimination plays via SimpleBot's
   capture-greedy tempo. v37a passed through it cleanly *still
   mono-W* — SimpleBot's pressure on contested structures wasn't
   enough to make composition more valuable than more Warriors.
   The bridge still has value as an opponent-difficulty step;
   it just doesn't independently force diversity.

6. **The composition-summary viz is the cheapest catch.** Adding
   `plot_curriculum_composition_summary` (horizontal stacked
   per-stage bar of units built + HQ captures + ability use)
   surfaced "100% mono-W on every cleared stage" in v37b at a
   glance. The same data was in `eval_log.csv` all along
   (`units_built` is a per-stage JSON column) but no one was
   parsing it. **Build the diagnostic that summarises *strategy*
   per stage, not just outcomes** — and look at it after every
   run, especially the runs that "succeed."

7. **`units_built` shape decides whether new features are
   exercised.** PR #383 added 4 new observation channels for
   status-effect debuffs (paralyze, haste, defence_buff,
   attack_buff). On a mono-Warrior policy those channels are
   permanently zero in the observation stream — Mages/Sorcerers/
   Clerics are never built, so no abilities are ever cast. New
   observation features only pay off if the *policy* will end
   up producing the data they encode. If the cost curve incentivises
   ignoring those units, the channels are inert. Track this
   coupling: an "improved observation space" that the policy never
   exercises is dead weight.

### Open questions for v40+

- **Does W cost=300 actually break the attractor on beginner?**
  v39 couldn't test this — it died on starter before reaching
  any stage where the cost change had room to matter. v40's
  beginner ladder is the first real test. Watch
  `beginner_random_{10,15,20}` units_built distributions.
- **If v40 clears the beginner/intermediate ladder mono-W
  anyway**, then cost-curve geometry isn't the binding lever
  either, and the next step is either (a) explicit diversity-
  bonus reward (a small positive reward per Nth distinct unit
  type built per episode) or (b) BC warm-start from a
  demonstrator that uses composition. Both are higher-effort
  than balance tuning.
- **Capacity has not been tested.** Every v34–v40 run uses
  `net_arch=[256, 256]`. If diversity *needs* a wider hidden
  layer to represent multi-unit composition planning, none of
  the balance/curriculum levers can fix that. A `[512, 512]`
  control run on a config that otherwise reaches the
  `intermediate_random_20` wall would discriminate "balance
  bottleneck" from "capacity bottleneck."
- **The starter ladder is currently un-trained.** v40's policy
  enters `beginner_balanced_random` cold. If a future iteration
  needs starter for navigation pre-training (e.g. if beginner's
  bigger map slows cold-start convergence), the cleaner pattern
  is probably a *separate* starter-only pre-curriculum run that
  ships a checkpoint, rather than splicing starter back into a
  config tuned for balance experiments.

## The kill-farm draw-plateau at `beginner_random_15` (v41, run `20260528_135412`)

`v41_r10_stability.yaml` applied the documented triple-fix (gate
0.75→0.65, budget 3M→5M, entropy floor 0.03→0.01) to
`beginner_random_10`. **It worked for its stated target** — r10
cleared at 100% / 93.75% — but the curriculum stalled one stage later
at **`beginner_random_15` (stage 3 of 33), best WR 0.65 vs the 0.70
gate**, after exhausting ~3M steps (`run_status: curriculum_stalled`).
This is the most-instrumented r15 stall to date (five diagnostic
charts) and it pins the mechanism more precisely than the v15–v32
sweep could.

### It is the known draw-with-shaping drift — confirmed, not a new bug

`eval_curves` match the "drift vs plateau" table exactly:
**explained_variance ≈ 0.85, value_loss low and stable, approx_kl ≈
0.005.** The value head fits returns; the policy barely updates per
step yet wanders. So this is an **optimization/attractor** problem,
not capacity — and it is **not** the v27a/v27c reward regression
either: v41 already ships `win_speed_bonus: 0` and
`enemy_owned_capture: 0`. The cause is elsewhere in the reward shape.

### New detail: it's a combat-farm, and capture behaviour died at random_10

Three charts the earlier sweeps lacked:

- **Composition drifts wildly across stages; captures die early.**
  `balanced_random`: W55%, captures HQ:4 / B:21 (seizing is part of
  the plan). `random_10`: **W81%, HQ:0 / B:0 — cleared 100% purely by
  ELIMINATION.** `random_15`: re-specs to **~70% Knight + ~20%
  Barbarian**, kills 10,324/eval (highest of any stage) but only
  HQ:1 / B:5. Capture-to-win behaviour evaporated at the
  **balanced_random → random_10 transition**, not at r15.
- **Action mix on r15 is a kill-farm.** ~50% move, ~22% attack, ~10%
  create, ~10% end_turn, ~9% heal, and **seize only ~2–3%** — and
  seize was already ~1–2% at r10. The agent moves and trades attacks;
  it almost never works toward a capture.
- **Reward decomposition shows why the farm is stable.** Summed per
  80-ep eval: `action` (combat) +2000–5000, `shaping_delta`
  +1000–2000, `terminal` a chronic −500 to −3500 (draw penalty). A
  stalled 75-turn draw therefore nets **≈ +19/episode**: the dense
  combat+potential farm out-pays the draw penalty. Winning ends the
  farm and is rarely reached (HQ:1/eval), so the gradient points at
  fighting-and-stalling, not finishing. W/L/D ≈ 3/1/76, turns pinned
  at the 75-turn cap.

This is the env reward docstring's own warning ("kill-farm local
optimum that never finishes the game") and **lesson #3 of the v34–v40
section** ("combat shaping is NOT a diversity lever — it rewards
finishing fights, so the policy just gets more aggressive about the
cheapest combatant"). The v34 combat shaping unlocked depth via
elimination on easy stages; on r15 the more-active opponent
(max_actions 10→15) can't be eliminated within 75 turns, so the same
aggression draws.

### Triple-fix side effect: r10 cleared *less* diverse than v40

Worth flagging: v40's r10 *stalled* but at **28% Warrior** (diverse);
v41's r10 **cleared at 81% Warrior, elimination-only**. The lower
entropy floor (0.01) + extra budget let the policy commit to mono-W
elimination and clear — trading the composition v40 had for a
promotion. Clearing a `*_random_N` stage and clearing it *with
composition* remain different things (lesson #4).

### v42: remove the combat farm (single-variable test)

`v42_remove_combat_farm.yaml` zeroes `damage_scale` and `kill` (the
*only* delta vs v41, verified by config diff). With no per-step combat
farm, a draw is unambiguously net-negative (`turn_penalty −0.5 × ~73
turns` + `draw −10`, nothing to offset it), so capture/win is the only
positive-return path. Supporting evidence beyond this run: the deep
config (Experiment A / v26) cleared *past* r15 with no combat shaping,
and the in-config note records that zeroed-combat runs reached
`random_20` (past r15) — combat shaping only mattered for the final
r20 push. If v42 *still* stalls at r15 with seize ≈ 2%, the bottleneck
is capture **exploration** (the sparse, committed multi-step seize
sequence), not the reward ratio — at which point reward tweaks are
exhausted and the escalation is the higher-confidence levers already
on the books: **BC warm-start from a capture-using demonstrator** (v33
cleared r15 this way) and the **untested `net_arch [256,256]→[512,512]`
capacity control**.

### Lessons

1. **A profitable draw is a reward bug, not a policy bug.** If the
   per-step shaping (combat + potential) can out-earn the draw
   terminal, PPO discovers the farm-and-stall equilibrium and the
   draw-with-shaping attractor becomes a rational resting point, not a
   transient. Check the reward decomposition: if `action +
   shaping_delta` exceeds `|terminal|` on draws, the draw *pays*.
2. **"Cleared" hides which win condition was learned.** r10 cleared at
   100% with zero HQ/building captures — pure elimination. The
   curriculum then hands an elimination-only policy to a stage where
   elimination can't finish in time. Read `captures_by_type` on
   cleared stages, not just WR.
3. **The triple-fix relocates the stall; it doesn't remove the
   mechanism.** Gate/budget/entropy got r10 past the bar (by
   committing to mono-W elimination) and the identical drift reappeared
   one stage later. The mechanism is the reward landscape, and it
   travels with the policy down the curriculum.
4. **Seize starvation is the through-line.** Across r10 and r15 the
   agent seizes ~2% of the time. Every "can't close the game" stall on
   the beginner ladder reduces to "the policy never learned the
   committed seize sequence." That points the durable fix at capture
   exploration (BC, or a denser/earlier capture curriculum) rather
   than at opponent/gate/entropy tuning.

## v42 (run `20260528_212658`): removing the combat farm — right direction, same wall

`v42_remove_combat_farm.yaml` zeroed the combat shaping (`damage_scale`
0.002→0, `kill` 0.2→0) — the single-variable test the v42 entry above
predicted. It **stalled at `beginner_random_15` again**, best WR
**0.7125** (vs v41's 0.65 — it *crossed* the 0.70 gate once but couldn't
sustain patience-2). Three findings:

1. **It over-sparsified the early ladder.** With the dense combat reward
   gone, `random_10`'s first eval **collapsed to 0% / ~1 action-per-turn
   / reward −47.6** (pure draw penalty — the stage-shift end-turn
   collapse, with no combat gradient to climb back out), then took
   **~2.9M steps** of wild oscillation to barely clear (0.65/0.75 pair at
   ~3.0M) — vs v41 clearing it in **150k**. v41's combat-shaped policy was
   an elimination machine that transferred instantly; v42's
   capture-oriented policy didn't transfer and had no fallback aggression.

2. **It traded one degenerate attractor for another: Cleric-spam.** r15
   drifted to **~74% Cleric** (W21% / C74% / K5%), HQ:0 / B:0. The
   mechanism is the cost curve: the v39/v40 nerf made Warrior cost 300,
   so **Cleric (200) is now the cheapest unit**, and removing combat
   shaping removed the only reason to prefer fighters over the cheap
   support unit → the **spam-the-cheapest-unit attractor** (the v34–v40
   lesson) resurfaced with a *new* cheapest unit. A Cleric-heavy army
   can't capture or eliminate reliably → draws.

3. **The drift itself is untouched.** Same healthy-but-drifting PPO
   signature (explained_variance ~0.8, approx_kl low, value_loss stable).
   r15 peaked 0.7125 @ 3.9M then *genuinely* collapsed (0.46, 0.60, → ~0
   for a long stretch). Capture behaviour never survived past stage 1:
   `balanced_random` reached **HQ:11** captures (vs v41's HQ:4 — removing
   the farm *did* diversify + capture on the easy stage), but r10 and r15
   were both **HQ:0**.

**Verdict: the combat-reward axis is exhausted.** v41 *added* combat
shaping → Knight kill-farm, stall at r15. v42 *removed* it → Cleric
spam, stall at r15. Both directions wall at the same stage with the same
drift. The accumulated evidence (v15→v42) now spans **both adding and
removing** the reward levers — all stall at r15. **Reward shaping is not
the lever.** This lands on the v42 config's own decision-tree branch:
"stalls at r15 with seize ~2% → reward tweaks exhausted, escalate to the
non-reward levers."

### Generalizable lesson

**Removing a dense shaping term doesn't redirect behaviour toward the
sparse goal — it just relocates the cheapest-unit attractor.** The hope
was that killing the combat farm would push the policy toward capturing.
Instead it pushed the policy onto whatever the *new* lowest-friction
behaviour was (spam the now-cheapest unit, Cleric, and draw). The
capture sequence is sparse and committed; it isn't reached by *removing*
an incentive, only by *adding* a path to experience it (denser capture
curriculum or BC). Corollary: **a cost-curve nerf changes which unit the
"spam the cheapest" attractor lands on** — nerfing Warrior's price made
Cleric the cheapest, so the attractor moved to Cleric. If you keep the
W=300 nerf, the cheapest-unit identity (Cleric) is now its own problem.

## v43a / v43b: opponent diversity ± capacity (testing the non-reward levers)

Since the reward axis is spent, v43 holds reward at **v41's** setting
(combat shaping ON — the *fast* early ladder; v42's combat-off reward
over-sparsified r10 to ~2.9M, which would bottleneck a parallel run
before it reaches the r15 wall) and varies only the non-reward levers,
as a parallel A/B:

- **`v43a_opponent_diversity.yaml`** = v41 + **opponent diversity** on the
  beginner `random_{10,15,20}` cluster. Each static RandomBot is replaced
  by a per-episode `MixedBot(random@A, random@B)` (config-only — MixedBot
  already forwards `easy_kwargs`/`hard_kwargs` to `RandomBot(max_actions)`):
  r10→{10,15}, r15→{10,20} (brackets the 15 it stalled on), r20→{15,20}.
  This is the doc's **"Option C"** anti-drift lever: a *static* stochastic
  opponent gives no in-stage gradient, so PPO drifts off any winning
  policy; varying the opponent per episode supplies one. `net_arch`
  unchanged `[256,256]`.
- **`v43b_opponent_diversity_capacity.yaml`** = v43a + `net_arch`
  `[256,256]→[512,512]` (the *only* delta vs v43a). Capacity is paired
  with opponent diversity rather than tested alone, because the
  "drift vs plateau" diagnostic says r15 is an attractor problem, not
  capacity-bound — so capacity-alone is the weak experiment. The narrower
  bet: IF diverse multi-unit composition needs more width to represent,
  the bigger net can exploit the stability diversity provides.

**Reading the parallel pair:** v43a clears → diversity alone fixes the
drift (capacity unneeded, save ~1.5–2× compute); v43b clears but v43a
doesn't → capacity adds real value on top of the drift fix; neither
clears → escalate to **BC warm-start** (the only lever that has ever
cleared r15, v33). **Caveat:** v41's reward keeps a stalled draw
marginally net-positive (combat+potential farm ~+57/ep vs ~−47 draw
penalty), so opponent diversity fixes the *no-in-stage-gradient* root but
not the *profitable-draw* root; if v43a/b still draw at r15, the
follow-up is opponent diversity **+ v42's combat-off reward** (draws
net-negative).

### Results (v43a `20260529_142749`, v43b `20260529_144521`)

A clean, decisive A/B.

| Run | Lever | Stalled at | Stages cleared | Best WR |
|---|---|---|---|---|
| **v43a** | opp diversity, `[256,256]` | `beginner_random_15` | **3** | **0.8375** |
| **v43b** | opp diversity + `[512,512]` | `beginner_random_10` | **2** | 0.8375 |

1. **Capacity is out.** v43b got *less* far — stalled a full stage earlier
   (r10 vs r15). Both peaked at 0.8375, so `[512,512]` isn't incompetent; it
   **drifts the same way and stalls wherever the drift catches it**, exactly
   as the "not capacity-bound" diagnostic predicted. No benefit, ~1.5–2× the
   compute, and a stage lost here. Don't use it. (Part of the stage gap is
   run-to-run noise, but the signal — "capacity provides no benefit" — is
   clear.)

2. **Opponent diversity is the single most effective lever in the whole
   sweep.** v43a (a) **cleared the now-*harder* mixed r10** —
   `MixedBot(random@10, random@15)`, tougher than v41's static r10 and the
   stage v42 nearly died on — and (b) pushed r15 to **0.8375, the highest r15
   WR of the modern cold-start line** (v41 0.65, v42 0.71; only the
   old-code/-reward deep config ever did better). The in-stage-gradient
   hypothesis is validated: varying the opponent per episode keeps the policy
   competent at a far higher level. **Keep it.**

3. **It still didn't crack the *sustain*.** Despite the 0.84 ceiling, v43a
   stalled — the policy oscillates between evals (peaks ~0.84, dips below
   0.70) and never strings **two consecutive ≥0.70**. The drift is *reduced*
   (higher mean, competent longer) but not *eliminated*. The ~0.20
   eval-to-eval swing is too large for sampling noise (80-ep σ≈0.05), so it's
   genuine residual drift — and the W/L/D columns show **it oscillates between
   *winning* and *drawing*, not losing** (the 0.8375 eval was 67W/1L/12D; the
   low evals are ~0W/1L/79D). It's wandering into the profitable draw, exactly
   the second root the v43 caveat flagged.

**Conclusion + next step (v44).** Opponent diversity killed the
*no-in-stage-gradient* root; the *profitable-draw* root (v41 reward keeps a
stalled draw net-positive) is what's left. So the natural follow-up combines
the two levers that each kill a different root:

> **v44 = v43a (opponent diversity, `[256,256]`) + draws-made-unprofitable.**

The single cleanest added variable is an **HQ-income cut**
(`headquarters_income: 150 → 100`): it shrinks the guaranteed base income a
policy can coast on, making a stalled draw economically worse, and
simultaneously raises the *relative* value of contested buildings/towers
(pushes toward capture). It's left as a single-variable delta vs v43a so its
effect is attributable. (If it under-delivers, stack v42's combat-off reward
on top, or escalate to BC warm-start — still the only lever that has ever
cleared r15.) Caveat carried forward from the v38/v39 lesson: don't
over-tighten the economy, and prefer cutting HQ income over building/tower
income (cutting the latter would *discourage* the capture we want).

### Results: v44 + v45 (runs `20260530_034840` = v44, `20260530_034656` = v45)

Both off the v43a base (opponent diversity, `[256,256]`); each a single-variable
delta. **Neither improved on v43a, and v44 regressed.**

| Run | Single delta vs v43a | Stalled at | Cleared | Best WR |
|---|---|---|---|---|
| v43a (baseline) | — | `random_15` | 3 | **0.8375** |
| **v44** | `headquarters_income 150→100` | **`random_10`** | **2** | 0.65 |
| **v45** | r15 `ent_coef end 0.03→0.01` | `random_15` | 3 | 0.7375 |

**v44 (HQ-income cut) backfired — the v38/v39 economy-too-tight failure,
exactly as the caveat above warned.** It stalled a stage *earlier* (r10,
never reached r15). The agent's win path is mass-elimination, so a leaner
economy → smaller army → it can't close r10 against the mixed opponent. The
CSV shows real **losses** at the dips (0W/18L/62D, 11W/16L/53D), not just
draws — the symmetric income cut hurt the agent's economy-dependent strategy
*more* than the random opponent's gold-independent play. The "less gold →
more capture" hope did not materialize; less income just made the early game
weaker. **HQ-income cut is out.**

**v45 (entropy floor 0.01) was a wash.** It reached r15 like v43a but stalled
the same way (peak 0.7375 vs v43a's 0.8375 — within CUDA-noise range, so not
attributable to the change). Committing harder late did **not** stop the
win↔draw oscillation. Consistent with the doc's "lower entropy reduces
exploration" — possibly trading ceiling for commitment without gaining
stability.

**The converging signal (v43a/b, v44, v45).** Which stage it stalls on (r10
vs r15) is partly noise — the policy is *marginal* right at the 0.70 gate.
Opponent diversity genuinely raised the ceiling (0.84 achievable), but **every
cheap config lever on top of it has now failed: capacity (v43b, worse),
economy (v44, worse), entropy (v45, wash).** v43a remains the high-water mark.
That's strong evidence that reward/economy/entropy/capacity tweaks won't make
this policy *reliably* close out a win — the missing piece is a robust,
repeatable **capture** win instead of fragile mass-elimination, which those
tweaks keep failing to teach.

**Next:** v46 (`unit_diff 0.3→0.0`, the last untested cheap lever — stop paying
for mass) is queued to close out the cheap-lever sweep, but expectations are
modest. The indicated escalation is **BC warm-start from a capture-using
demonstrator** (keeping opponent diversity) — the only lever that has ever
cleared r15 (v33), and the one that can teach the capture win directly.

## ✅ BREAKTHROUGH — v49→v50: the reward landscape WAS the root; cold-start now reaches skirmish

This is the chapter the v15–v45 sweep was circling. After the cheap config
levers were exhausted (entropy, patience, capacity, opponent diversity,
economy), a set of **diagnostics** built in this branch pinned the actual
mechanism, and a **combined reward fix (v49) + decisive-combat engine change
(v50)** took the modern cold-start line from "stalls at `beginner_random_15`"
(every config since v17) to **~20 stages, reaching `skirmish_random_20`** —
matching the historical deep config's frontier, from a cold start.

### The diagnostics that cracked it

The sweep had been guessing at *why* the policy stalls. Three additions made
it observable (eval log + TensorBoard + per-step info):

- **`seize_available_rate`** — fraction of decision points where a capture
  action was legal. This separates "the policy never reaches a capturable
  tile" (navigation/exploration) from "it reaches one and declines"
  (reward). On the `beginner_random_15` stall it sat at **40–57% even at 0%
  WR** → the agent *can* capture and *won't*. That ruled out BC/navigation as
  the next lever and pointed squarely at the reward landscape.
- **`max_legal_actions`** — peak legal-action-set size. Guardrail for the
  `flat_discrete` `max_flat_actions` cap; later caught a real 512 overflow on
  skirmish (see v52).
- **`best_checkpoint_timestep` / stage-relative steps** — how far into a stage
  the handed-forward best checkpoint actually was (the skip-ahead handoff
  tell).

A **latent truncation bug** surfaced alongside: `_build_flat_actions` appended
`end_turn` *before* the `max_flat_actions` cap and head-truncated the
overflow — silently dropping `end_turn` (appended last) and `seize`
(action_type 3, built after create/move/attack). Fixed to always preserve
both. Latent on 6×6; **load-bearing on skirmish** (v50 hit the cap).

### v49 (run `20260531_051101`): the combined reward fix broke the r15 wall

The decisive read from the v43a run: at 0% WR the policy drew 78–80/80 games
to the 75-turn cap while collecting **+28..+55 reward** — *the draw was
positively rewarded*. A safe paying harbor. v49 is the combined fix the
single-axis sweep never tried (all off v43a's opponent-diversity base):

1. **`draw: −10 → −50`** — remove the positive safe-harbor draw (== loss, so
   never prefer losing to drawing).
2. **`damage_taken_scale: −0.002` (NEW env term)** — charge for damage taken
   so combat shaping is net-zero-sum; a mutual trade nets ~0, only decisive
   combat pays. Kills the trade-to-the-clock farm without zeroing combat (v42's
   combat-OFF over-sparsified r10).
3. **`unit_diff: 0.3 → 0.0`** — stop paying per-unit-*count* (the confirmed
   cheapest-unit-spam subsidy).
4. **`hq_capture: 25 → 60`, `win_by_hq_capture: 50 → 80`** — the HQ was paying
   *less* than a building (40); make it the best target and prefer the
   transferable win.

Result: draws flipped **negative** (0%-WR evals now −20..−48), mono-Warrior
spam gone (Warrior share 23–65%, not 95–100%), and it **cleared `random_10`,
`random_15`, `mixed_r15_simple`** and reached `beginner_random_20` (5 stages,
best 0.7875) — the first modern config to get past the r15 wall. Wins were by
**elimination** (HQ captures 0 past stage 1); the negative draw forced
*finishing*, not *capturing*.

### v50 (run `20260531_165459`): hp_scaled decisive combat → the breakthrough

`v50 = v49 + damage_model: hp_scaled` (engine: outgoing damage and counters
scale with the attacker's current HP fraction — consistent with seize, which
was always HP-based; config-surfaced via `engine_overrides`, recorded in
`config.json`).

The flat-damage model is an **attrition** model — a 1-HP unit hits as hard as
a full one, so armies grind down evenly and games drift to the max-turn draw
*at the mechanics level*. HP-scaling makes combat **decisive** (focus-fire
compounds, fights resolve). It also shrinks the farm on its own.

Result — the deepest modern run by a wide margin:

- `random_10` cleared **3× faster** (750k vs v49's ~2.3M) — decisiveness in
  action.
- Cleared the **entire beginner ladder** including beating Simple/Medium/
  **Advanced** bots, then **transferred to intermediate (7×7)** and **skirmish
  (8×8)**, clearing through `skirmish_random_15` and stalling at
  `skirmish_random_20` (~20 stages) when Colab timed out at ~8M steps.

**hp_scaled is a keeper.** It is the engine-side twin of v49's reward fix:
v49 removed the safe draw, v50 made the agent win the fights it now has to.

### v51 (run `20260601_031012`): budget floor + patience-2 — and a variance bombshell

`v51 = v50 + 3M budget floor on every stage + uniform patience-2`. Motivation:
v50's `intermediate_random_20` was budgeted **1.5M/patience-3** — half the
budget and a stricter gate than `beginner_random_20` (3M/patience-2), the same
drift-attractor stage type — and it cleared only via a late recovery right at
its budget edge. The floor is a **ceiling** raise (allows recovery), *not* the
`min_timesteps_before_promotion` gate that backfired (that forced
overtraining); easy stages still promote early and ignore it.

**But v51 stalled at `beginner_random_15` (3 stages, 0.6625)** — far worse than
v50. This is **NOT a regression from the fix**, and the reason is the most
important methodological lesson in this whole document:

> **Run-to-run variance on this curriculum is enormous: the same config and
> the same seed produced 14 stages (v50) vs 3 stages (v51).**

Proof it's variance, not the change: (a) `random_15` had *identical* budget
(3M) and patience (2) in both — the fix didn't touch it; (b) v50 and v51's
**first eval (`@ step 8`) is byte-identical** (WR 0.5, reward 91.8589625) —
same seed, same weight init; (c) they diverge by `@ 50k`. Everything is
seeded — **opponent bots** (per-episode `random.Random(bot_seed)` from
`np_random`), **weight init** (`MaskablePPO(seed=cfg.seed)` → `set_random_seed`
before the policy is built), **eval RNG** — so the *only* uncontrolled variable
is **CUDA floating-point non-determinism** (non-associative atomic adds),
which the `random_10/15` drift attractor amplifies into completely different
trajectories.

**Consequences:**
1. **Single-run comparisons across configs are unreliable.** v49's 5-stage
   stall, v50's 20-stage run, and v51's 3-stage stall are samples of a
   high-variance process. Conclusions need **2–3 seeds per config**.
2. The doc's old "treat ±10% at a promote eval as noise" massively understated
   it — here it's the difference between a stall and a breakthrough.
3. If exact reproducibility is ever needed, also set
   `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG` +
   disable cuDNN autotune (slower, reproduces *one* path). For *comparison*,
   do the opposite — vary the seed.
4. One residual global-RNG leak: **Rogue evade uses `random.random()`** in
   `mechanics.attack_unit` (not the env/bot rng). Minor (Rogues rarely built)
   but worth closing for clean multi-seed work.

### v52a / v52b (queued): the skirmish stall is draw-economy + truncation, not capture

The v50 `skirmish_random_20` stall exposed two skirmish-specific (`max_turns=
120`) blockers — and notably **`seize_avail` was *high* (75–82%) on skirmish**,
so it is *not* the elimination-can't-transfer capture wall:

1. **The negative-draw fix doesn't scale to the longer clock.** On skirmish the
   per-step combat/seize farm scales with `max_turns` and the larger structure
   count while `draw:-50` is fixed — so 0/0/80 draws returned to
   break-even/positive (+19..+54). v49's draw fix was calibrated for
   `max_turns=75`.
2. **Truncation fired** — legal actions exceeded `max_flat_actions=512`
   (warnings at 517/526/528; `max_legal` pinned at 512), dropping legal
   moves/attacks. The diagnostic + truncation fix caught and contained it, but
   the cap is genuinely too small for 8×8.

Both off v51 (carry the budget/patience fix forward):
- **v52a** = `turn_penalty −0.5 → −1.0`. `turn_penalty` is the natural
  *max-turns scaler* (charged per `end_turn`, so a stall accrues
  `turn_penalty × max_turns`): a 120-turn skirmish stall now costs −120 (was
  −60), shifting expected draw return clearly negative, while fast wins end
  early (+100..+220 reward) and stay positive. Isolates the reward fix.
- **v52b** = v52a + `max_flat_actions 512 → 1024`. Adds the truncation fix.
- Reading the A/B: v52a clears → draw economy was the blocker; v52b clears but
  v52a doesn't → truncation contributed; neither → deeper capability wall →
  structural v53.
- Escalation if `−1.0` under-delivers: `turn_penalty −1.5` (wins have ample
  headroom).

### Reusable levers now on the config surface (default-inert)

All snapshotted into `config.json`, all byte-identical to legacy when unset:
- `reward_config.damage_taken_scale` — symmetric combat (kill the trade farm).
- `engine_overrides.damage_model: flat|hp_scaled` — decisive combat.
- `engine_overrides.{tower,building,headquarters}_health` — capture-difficulty
  lever (e.g. HQ@30 = 2 Warrior-turns vs 4). The direct knob for the capture
  problem when it becomes the binding wall.

### What's still open (→ v53 structural)

Across the whole arc, **wins are by elimination; HQ captures stay ~0 past the
first easy stage.** The reward now *points* at capture (hq_capture 60, win 80)
and the draw no longer pays — but the **capture gauntlet** is still
mechanically hard: seize damage = the seizer's *current* HP (decays under
fire), a defended HQ takes 3–4 uninterrupted turns to take, a garrison unit
**physically blocks** the tile (you can't step onto an occupied tile) and
**base-heals** on it, and a killed seizer lets the structure regen 50%/turn. So
capture only happens once the fight is already won — i.e. it collapses into
elimination, which doesn't transfer to bigger maps where elimination can't
close in time.

The structural levers (when capture becomes the binding wall, likely on
skirmish+ after v52): **`headquarters_health` reduction** (make the HQ a 1–2
turn act so it's reachable by exploration) and/or **BC warm-start from a
capture-using demonstrator** (the only thing that ever taught capture, v33).
Both now compose with a reward that finally rewards finishing-by-capture and a
draw that no longer pays.

### Lessons (this chapter)

1. **Build the diagnostic before tuning the lever.** `seize_available_rate`
   converted years of "is it can't-reach or won't-finish?" guessing into a
   one-number answer (won't), which redirected effort from BC to reward.
2. **The profitable draw was the root all along.** Every cheap lever (entropy,
   patience, capacity, opponent diversity) failed because they nudged the
   policy *around* an attractor the reward made positive. Pricing the draw
   negative (v49) + decisive combat (v50) removed the attractor and 4× the
   curriculum depth fell out.
3. **A draw fix calibrated for one `max_turns` doesn't transfer to a longer
   clock.** The farm scales with the clock; the penalty must too
   (`turn_penalty` does this naturally — prefer it over a flat `draw` constant
   on multi-`max_turns` curricula).
4. **Run-to-run variance dwarfs most config deltas.** Same seed → 3 vs 14
   stages from CUDA non-determinism alone. Compare configs across **multiple
   seeds**, not single runs. This recontextualizes the *entire* v15–v48
   single-run sweep: some "stalls" and "clears" were luck.
5. **Budget is a ceiling, not a floor.** Raising `max_timesteps` lets a
   stall-prone stage recover without forcing overtraining (unlike the
   `min_timesteps` gate). Keep the random_N / drift-attractor stages at ≥3M.
6. **Engine balance belongs on the config surface.** `damage_model`,
   structure-HP, and `damage_taken_scale` close the last hardcoded confounds —
   a balance change is now a recorded config delta, not a `constants.py` edit.

## v52a complete run + the action-space / economy work (the skirmish_random_20 wall)

The v52a run (`20260601_172412`, `turn_penalty −1.0`, `max_flat_actions 512`)
ran to the Colab timeout and gave a decisive, *negative* result that reframes
the remaining work: **skirmish_random_20 is a hard structural wall, and on
512-cap it is truncation-saturated.**

### What the finished run showed

- **Deepest frontier yet**: cleared the entire beginner + intermediate ladders
  and skirmish through `skirmish_random_15`, then stalled at
  `skirmish_random_20`.
- **Not a timing stall**: the policy sat on `skirmish_random_20` from ~5.85M to
  the ~8.05M timeout — **~2.2M steps** — and never sustained ≥70%. WR
  oscillated 0%↔37% indefinitely. A full stage budget could not crack it, so
  the blocker is structural (capture/economy), not budget.
- **Truncation saturated**: every warning reads `max_flat_actions (512)`;
  demand at the stall ran **513–744 continuously**, so ~30–45% of the legal set
  (moves/attacks) was dropped on most decisions, every game. This is the
  cleanest proof yet that **512 is a real binding constraint on the 8×8 map** —
  and the direct justification for `max_flat_actions: 1024` (v52b/v53/v53b).
- **Policy-collapse signature**: eval @ 7.75M = `WR 0% / len 121 / turns 120 /
  seize_avail 0% / max_legal 20` — the agent built ~1 unit and drew out, then
  swung back to max_legal=512 a few evals later. Violent mass↔nothing swings =
  late-stage entropy/drift instability (keeps the entropy-respike lever live).
- **Drift confirmed system-wide**: several stages logged `restoring best
  checkpoint … peak @ 8 stage steps` — the best policy for the stage was the
  *transfer point*, and training drifted off it (beginner_random_10 needed
  ~850k steps of 100%↔2% oscillation; intermediate_random_15 went fully to 0%
  for ~800k before recovering).

### The action-space balloon: diagnosed, not guessed

Reviewing `get_legal_actions`, the balloon is **movement-dominated**, not
ability/attack-dominated: moves are enumerated **one per (unit, reachable
tile)**, so the legal-set size scales as `units × per-unit footprint`. A
correction worth recording: the Mage/Sorcerer `{"adjacent","range"}` numbers
are **damage at distance 1 / 2**, *not* a 12-tile reach — ranged attack range
is only ≤2 (Archer 2–3/4). So ranged multi-target is a minor contributor;
**army size × move-fan is the whole story.** 744 legal ≈ ~24 units on skirmish.

### Board-density is the right lens for the unit cap

`20 units/side` against per-map **walkable** tiles:

| map | walkable | 20 = 1 side | both at 20 |
|-----|------|------|------|
| beginner 6×6 | 36 | 56% | 111% (impossible) |
| intermediate 7×7 | 43 | 47% | 93% (gridlock) |
| skirmish 8×8 | 62 | 32% (~⅓) | 65% |
| corner_points 12×10 | 114 | 18% | 35% |

So a cap of **20 is generous-to-free on the small maps** (they physically
can't hold ~40 units — they self-cap ~15–18/side) and **bites only at genuine
skirmish gridlock** (the ~24-unit peak the stall sat at) or to **force
precision on corner_points** (the one map where 20 is an active downward
constraint, and even there it's only 18% density — playable). The measured
~24-unit skirmish peak is therefore the *pathology*, not normal play to
preserve headroom for.

### What shipped this session (all default-inert / config-surfaced)

- **Legal-action correctness fixes** (`get_legal_actions`): paralyze no longer
  offered against already-paralyzed enemies (matches heal/cure/buff guards);
  `health > 0` guards on the source-unit loop and ranged-attack target loop
  (safe today via synchronous removal, fragile otherwise).
- **Army-economy telemetry**: per-episode `peak_own_units` / `mean_own_units` /
  `peak_gold_banked` / `mean_gold_banked`, surfaced in eval results, tensorboard
  (`eval/*`), and the eval print line (`army(pk/mn)=… gold(pk/mn)=…`). Separates
  "economy funds mass" (high peak army + ~0 banked gold) from "reward funds
  mass" — and the reward in this lineage already pays **nothing** for units
  (`create_unit 0`, `unit_diff 0`, damage net-zero, `kill 0.2`), so a persistent
  big army implicates the **economy** (uncapped per-turn income, no upkeep, free
  structure-healing), not the reward.
- **Per-player unit cap** (`constants.MAX_UNITS_PER_PLAYER = 50`, override
  `engine_overrides.max_units_per_player`): enforced in **both** `create_unit`
  and `get_legal_actions` so the cap shows up in the action mask (no
  offered-then-rejected creates). Bounds both the action-space balloon and the
  convert-all-gold economy.
- **v53b config** = v53 structure-HP **+ economy lever**: `headquarters_income
  150 → 120` (trim the largest gold faucet without touching relative unit value)
  and `max_units_per_player 20` (active lever; code default stays 50 as a
  never-bind guardrail). v53 (accommodate the army via `max_flat 1024`) vs v53b
  (shrink the army via the economy) is the clean A/B.

### Lessons (this chapter)

1. **A saturated `max_legal` at the cap is a censored measurement.** Eval
   `max_legal_actions` is `len(_current_actions)`, itself capped at `max_flat`,
   so it pins at 512 and *hides* true demand — you must read the truncation
   **warnings** (raw counts, up to 744) to see the real size. Don't infer army
   size from the (censored) eval scalar.
2. **The action-space balloon is `units × move-footprint`, dominated by
   movement.** Nerfing a unit's *abilities* or *attack range* barely touches it;
   only army size (economy) or per-unit move enumeration (action representation)
   moves the needle.
3. **Density, not a raw count, sizes a unit cap.** The same cap is free on a 6×6
   board and binding on a 12×10 one. Size caps against **walkable tiles per
   map**; ~⅓-board one-side coverage is the gridlock onset.
4. **When the reward already de-subsidizes mass but the agent still masses, look
   at the economy, not the reward.** Uncapped income + no upkeep + free healing
   makes units a free-to-hold instrumental good; the policy converts all gold to
   bodies even though nothing pays for bodies. Fix the *supply* (income/upkeep/
   cap), not the reward weights.
5. **A full-budget stall that never sustains is structural, not under-budgeting.**
   2.2M steps on `skirmish_random_20` with no sustained promotion rules out
   "needs more steps" — escalate to the structural lever (capture HP / economy),
   not a bigger ceiling.
6. **Full-ladder runs are the unit of cost.** Reaching skirmish consumes a whole
   Colab session, so engine-only changes (structure HP, income, cap, damage
   model) are warm-start-friendly from a near-skirmish checkpoint, but a
   `max_flat` change resizes the `Discrete` action head and needs a fresh run.

## Tuning roadmap — the levers worth pulling next (and their priority)

After the v52a skirmish_random_20 wall, the open knobs sort into scalar *tunes*,
structural *refactors*, and *methodology*. Ordered by expected leverage:

### 1. `gamma` / effective horizon — the under-rated scalar (do this near-first)

`gamma` is per **env-step**, not per game-turn. Episodes run to `max_steps=3000`
(skirmish ≈ 20 env-steps/turn × 120 turns ≈ 2400 steps). At `gamma=0.99` the
effective horizon is `1/(1−0.99)=100` env-steps ≈ **~5 game-turns**. So the
terminal `hq_capture (+60)` / `win (+80)` rewards earned on turn ~50 of a
capture push are discounted to **near-zero** by the time they reach the
maneuvering that set them up — which is the cleanest mechanistic explanation
for **"wins by elimination, HQ captures ≈ 0, can't close skirmish in the
clock."** The dense potential terms (income/structure_control) carry local
signal, but the big *terminal* capture incentives barely propagate.

- Lever: `ppo.gamma 0.99 → 0.995–0.997` (horizon ~200–330 steps ≈ 10–17 turns),
  optionally `gae_lambda → ~0.97`. One line; warm-startable from a near-skirmish
  checkpoint so the effect shows fast.
- Caveat: higher gamma raises value-function variance → interacts with the
  drift instability (§ collapse @ 7.75M). Watch `explained_variance`/`value_loss`.
- Rank this **above** starting-entropy: it's a root-cause lever for the capture
  failure, not a stabilizer.

### 2. Multi-seed — methodology, not a parameter (prerequisite for trusting 3–4)

Run-to-run variance (CUDA non-determinism) dwarfs most config deltas (same seed
→ 3 vs 14 stages). Until economy / entropy / HP / gamma are compared across
**≥3 seeds**, any single-run "this helped" is suspect. Not a knob — the thing
that makes every other knob's result trustworthy.

### 3. Capture-reachability bundle (make a push able to *finish*)

Capture is a two-part gauntlet; tune both, not just HP:
- **Structure HP** (v53, config-surfaced): HQ@30 = ~2 Warrior-turns vs 4.
- **Regen rate** (NOT yet surfaced — hardcoded 50%/turn in `mechanics`): a lone
  seizer's damage = its *current* HP (decays under fire), so 50%/turn regen
  often out-heals a single seizer — capture only lands once the fight is already
  won. Lowering regen (≈25%/turn) or **suspending regen while an enemy occupies
  the tile** is arguably more targeted than HP. Would need the same
  `engine_overrides` treatment structure-HP got.
- **Economy** (v53b, config-surfaced): `headquarters_income 150→120` + unit cap
  shrink the army off the gridlock peak.

### 4. Starting entropy re-spike (stabilizer, not a root fix)

The 7.75M collapse (`max_legal=20`, built ~1 unit, drew out) and the
system-wide `peak @ transfer-point` drift say late-stage instability is real.
A gentle ent_coef re-spike when entering a new stage may keep the policy from
drifting off the transferred competence — but it treats the *symptom*; rank it
below gamma and the capture bundle.

### 5. Action-representation refactor (kills the balloon at the source)

`max_flat 1024` is a band-aid on a `units × move-footprint` balloon. The durable
fix is changing how moves are represented — select-unit-then-direction, a
per-tile move head, or top-K-destinations-per-unit pruning — which removes the
truncation confound permanently and makes precise multi-unit play *learnable*
(the policy can see all its options). A refactor, not a scalar, but the
highest-leverage non-tuning change on the board.

### Honorable mentions (lower priority)

- **`win_speed_bonus` vs `turn_penalty`**: speed pressure is currently all in
  `turn_penalty −1.0` (dense, per-turn). `win_speed_bonus` is terminal-only and
  *can't* be farmed; it may be a cleaner "win fast" signal that doesn't also
  punish legitimately long captures. Cheap A/B.
- **Garrison / seize mechanics**: a defender physically blocks the tile *and*
  base-heals on it, forcing kill-then-capture (→ elimination). Letting capture
  progress while adjacent, or partial seize from multiple units, attacks the
  gauntlet at the mechanic level — invasive; only if HP+regen aren't enough.
- **Opponent strength at the wall**: skirmish_random_20 is vs `random@20`; not
  the binding issue now (truncation/economy/horizon are) — leave it.

### Suggested order

`gamma`/horizon → multi-seed harness → (economy + structure-HP + regen) bundle
→ entropy re-spike → action-representation refactor.
