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
shape is faithful. The fix is **config-only**: ship modern code
with `win_speed_bonus` + the opponent-capture penalties removed (or
re-tuned). The whole v17–v23 entropy/patience/threshold sweep, and
the planned code bisect, were chasing a self-inflicted *reward*
regression. `docs/experiment_b_bisect_plan.md` is superseded; no
code bisect is needed.

Final isolation in flight: `v27a/b/c` re-enable exactly **one** of
the three terms each, to name the precise culprit (`win_speed_bonus`
is the prime suspect — it sits on the draw/win incentive that
`turn_penalty` was shaping). Whichever flips `random_10` back to a
stall is the single term to drop from the shipping config.

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

## The curriculum-tuning sweep (v15–v23): what 9 variants taught us

After the original 6-stage layout shipped, a long sweep
(`configs/bootstrap_sweep/v15…v23`) tried to push the agent past the
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
- **Behaviour-cloning warm-up** as described above, if the random-
  opponent curriculum fails on a future bigger map.
- **`start_fresh` flag on `CurriculumStage`** to optionally
  reinitialise the model at a stage boundary instead of always
  reusing weights — useful when a map shift is too steep for the
  starter policy to bridge.
