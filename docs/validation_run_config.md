# Validation-run configuration — 2026-09

This is the configuration for the 3-seed validation run in
[`REVIEW_full_2026-09-26.md`](REVIEW_full_2026-09-26.md) §2.1: `bootstrap.yaml`
with the reward back-port, on the fixed code (opponent-freeze fix, legal-only
rule bots, the MDP fixes, the stochastic gate). It explains every change to
three configs and gives the evidence for each. §10 lists the judgment calls
you may want to revisit. How to launch, monitor, resume and aggregate the run
is the runbook, [`validation_run.md`](validation_run.md).

| File | What changed |
|---|---|
| [`configs/ppo/bootstrap.yaml`](../configs/ppo/bootstrap.yaml) | Reward and env back-port (§2); 24 stages, re-ordered by the scripted-bot ladder, instead of 33 (§3); Wilson gate on the stochastic policy in both seats (§4); eval cost settings and budgets (§5) |
| [`configs/ppo/bootstrap_validation.yaml`](../configs/ppo/bootstrap_validation.yaml) | Now the first 8 stages of `bootstrap.yaml` with about half the budgets (§6) |
| [`configs/self_play/self_play.yaml`](../configs/self_play/self_play.yaml) | Trains the bootstrap MDP; `latest_opponent_prob` 0.5 and a pool that actually fills (§7) |

Tests pin the parts that must not drift. [`tests/test_shipped_configs.py`](../tests/test_shipped_configs.py)
checks the `max_steps` bound, that the slice mirrors the canonical file (as
the runner resolves both, padding included) and that self-play trains the
same MDP.
[`tests/test_curriculum_resume_hardening.py`](../tests/test_curriculum_resume_hardening.py)
checks that every threshold matches the Wilson rule.
[`tests/test_bootstrap_ladder_order.py`](../tests/test_bootstrap_ladder_order.py)
checks, on a cached ladder run, that no stage is a step down on the ladder
and that no gate asks for more than the scripted bots achieve against the
stage's opponent in both seats (§4).

---

## 1. How to run it

One run per seed, with the same config, launched as one seed group by
`scripts/train/run_seeds.py` (the runbook, [`validation_run.md`](validation_run.md)
§4, has the local, Colab and Vertex variants):

```
python scripts/train/run_seeds.py --config configs/ppo/bootstrap.yaml --n-seeds 3 --tag val -- \
    --strict --skip-videos
```

The seeds are 42, 1042 and 2042. Consecutive seeds (42, 43, 44) would share
7 of the 8 training env streams and 59 of the 60 eval episodes per seat, and
the launcher refuses them. Each seed runs `train_bootstrap.py --seed N
--resume-if-exists`, so continuing the group (`run_seeds.py --group <id>`,
runbook §6) resumes a killed seed from its rolling checkpoint (written every
50k stage steps). Run
`configs/ppo/bootstrap_validation.yaml` first, about 2 h; its purpose is in §6
and the command in the runbook (§3.1).

Report per seed and per stage. `scripts/eval/summarize_seeds.py` (runbook §8)
reads these from each stage's `eval_results.jsonl` and tabulates them; the
field names below are the raw ones.

- **Win / draw / loss with the stochastic policy.** Use `win_rate`,
  `draw_rate` and `loss_rate`, plus `by_seat` for each seat. The gate uses the
  stochastic policy, so these are the gate's own numbers. (Report: stochastic
  W/D/L and the gate win rate per seat.)
- **Captures by type:** `captures_by_type` (tower / building / hq), per
  episode.
- **Shaping share of return** (report: `dense_share_abs`, with
  `reward_per_ep_turn_penalty` and `reward_per_ep_shaping_delta` as separate
  columns). Decompose each eval's mean return using `reward_components` and
  `action_counts`:
  - terminal: `reward_components.terminal`;
  - turn penalty: `action_counts.end_turn × turn_penalty`;
  - seize: `combat_stats.seize_attempts × seize_progress`;
  - captures: `captures_by_type` × the per-type weights;
  - combat: `damage_dealt × damage_scale + damage_taken × damage_taken_scale + kills × kill`;
  - the potential term: `reward_components.shaping_delta`.

  Report the shaping share on the action stream with the turn penalty
  taken out: dense = `action` − turn penalty, and
  share = |dense| / (|dense| + |terminal|). Report the potential term and
  the turn penalty as separate columns. Don't compute the share on the raw
  return: the potential term telescopes in the discounted sum, but the
  undiscounted eval sum still carries −(1−γ)·ΣΦ(s_t). That drain reached
  −1508 in one 8872-step corner_points game played by a random policy.
  The report's archive-comparable `shaping_share_abs` is computed on the raw
  components (turn penalty and potential term included), so read it only
  against the archive.

For the v52a comparison, v52a's evals were greedy and seat-1 only. Evaluate
each stage's `best_model.zip` in both modes and seat 1 after the run. The
built-in sanity eval only covers the last stage, and there is no script for
the per-stage eval yet (runbook §8).

---

## 2. Reward and env back-port (review `rltrain-14` / `prior-4`)

The rule is **parity with v52a** (`configs/ppo/bootstrap_sweep/v52a_maxturn_scaled_draw.yaml`).
v52a and v50 are the deepest archived runs: 20 of 33 stages, stopping at
`skirmish_random_15`. The run re-baselines the fixed code, so it should
change as little as possible beyond the fixes themselves.

Every one of the 31 keys the env reads is now set explicitly. No key falls
back to `DEFAULT_REWARD_CONFIG`, whose terminals are 1000-scale.

| Key | Old | New | Why |
|---|---|---|---|
| `win_by_hq_capture` | 50 | **80** | v49/v52a value. Harmless, but it bought nothing measurable: HQ endings stayed at 0.3–4% of eval episodes after v49 raised it, and 96–100% of archived wins are eliminations. |
| `win_by_elimination`, `win`, `loss` | 50, 50, −50 | same | Unchanged in v52a. |
| `draw` | −50 | −50 | Equal to loss, as in v49/v52a. The clock scaling comes from `turn_penalty`: a full-clock draw costs −70 on starter, −125 on beginner and intermediate, −170 on skirmish and −250 on corner_points. |
| `truncation` | (absent → 0) | **0.0, explicit** | The fixed code pays this key plus the bootstrapped value on truncation, not the draw. The `max_steps` values below make truncation unreachable. |
| `income_diff`, `structure_control` | 0.05, 1.0 | same | Potential terms, unchanged in v52a. |
| `unit_diff` | 0.3 | **0.0** | v49 dropped the per-unit-count subsidy for cheap-unit spam. |
| `damage_scale` / `damage_taken_scale` | 0 / (absent) | **0.002 / −0.002** | v34+ combat shaping, made symmetric (v49). The fixed code measures HP actually removed and also charges counter-attacks. It is a small stream: +5..+11 per episode on beginner, +10..+24 on skirmish. |
| `kill` | 0.0 | **0.2** | v34+/v52a. |
| `unit_lost` | (absent) | **0.0, explicit** | The key did not exist in v52a; it stays off for parity. −0.2 (the mirror of `kill`) would make combat fully zero-sum, but it is untested. |
| `seize_progress` | 3.0 | 3.0 | 3.0 in every archived config. It is also the largest dense stream and a known farm (see §10.1). |
| `capture` / `tower_capture` / `building_capture` | 10 / 15 / 40 | same | `capture` is only a fallback: all three per-type keys are set. |
| `hq_capture` | 25 | **60** | v49/v52a. The HQ used to pay less than a building. |
| support abilities | 0.0 | 0.0 | The env's defaults are nonzero, so these are kept explicit. |
| `invalid_action` | −0.1 | −0.1 | Unchanged. |
| `turn_penalty` | 0.0 | **−1.0** | v52a. It widens the win–draw gap (§2.1). Safe only together with the per-turn action cap. |
| `win_speed_bonus` | 50 | **0.0** | See the archive evidence below. |
| `enemy_neutral_capture` | −8 | −8 | Shown benign: v27b cleared with it. |
| `enemy_owned_capture` | −15 | **0.0** | Adding it back alone (v27c) stalled `beginner_random_10`: peak 0.90, final 0.01. |

### 2.1 What the archive shows

The archive has 109 runs, all seed 42 and all with the opponent-freeze bug.
Per-stage `eval_results.json` files were read from v49, v50, v51, v52a, v52b
and v53c.

- **The gate to depth is `beginner_random_20`.** Only 4 runs ever cleared
  it: v34, v37b, v50 and v52a. All four had `win_speed_bonus` 0,
  `enemy_owned_capture` 0 and `turn_penalty` ≤ −0.5. In the family with
  `win_speed_bonus` 50 and `enemy_owned_capture` −15 (the old
  `bootstrap.yaml`), 0 of 20 runs cleared it.
- **The v26/v27 bisection** ran on the same code, n = 1 each. v26 and v27b
  cleared 5 of 5 stages. Adding back `win_speed_bonus` 50 (v27a) or
  `enemy_owned_capture` −15 (v27c) each stalled at `beginner_random_10`.
- **Results by `turn_penalty`** (clearing `beginner_random_15`):
  - −0.2: 0 of 9 runs;
  - −0.5: 5 of 19;
  - −1.0: 1 of 3.
- **Draw economics.** At −0.5, all-draw evals returned −19 per episode on
  beginner. At −1.0 they returned −39 to −80 on beginner and intermediate,
  while winning evals returned +60..+92.
- **Skirmish, v50, `turn_penalty` −0.5.** Evals with a 0.45–0.60 win rate
  returned +247 per episode, against +187 for winning evals: stalling
  out-earned winning.
- **Same-config noise is larger than any config effect.** v50 reached 20
  stages and v51 (a budgets-only change) reached 2. v52a reached 20 and v52b
  reached 2. So treat every archived conclusion as a prior, not a fact.

### 2.2 Per-turn cap and `max_steps`

- **`max_actions_per_turn: 60`** (was `null`). After 60 actions in one game
  turn, the mask offers only `end_turn`. This is a safety net against the
  "never end the turn" attractor, sized to rarely bind:
  - A trained v50 policy on skirmish took a median of 26, p90 37 and max 41
    steps per turn.
  - Random play maxes at 57 per turn on skirmish.
  - Random play hits the cap on about 21% of corner_points turns.
  - The per-turn counter is not in the observation, so the cap is not meant
    to shape play. 25 belongs only to the `rltrain-23` arm, which pairs it
    with γ 0.997.
- **`max_steps` per stage = ⌈(max_turns · 61 + 60) / 100⌉ · 100**: 1300 on
  starter, 4700 on beginner and intermediate, 7400 on skirmish, 12300 on
  corner_points.
  - Why the bound holds: a policy that follows the mask spends at most
    60 + 1 steps per turn, so `max_steps` can never end an episode before
    the `max_turns` clock does.
  - Why it matters: on the fixed code, truncation pays 0 plus the
    bootstrapped value. If truncation came first, it would be a cheaper exit
    than a max-turns draw.
  - Checked on the fixed code in both seats, with a random-legal policy and
    a policy that never ends its turn while it has another legal action.
    Every game ended by elimination, HQ capture or the max-turns clock, with
    `truncated=False`.
  - Random play on corner_points used 9251 and 9396 steps over 200 turns.
    The old flat `max_steps: 3000` would have truncated those games long
    before the clock.
  - A policy that ignores the mask (a uniformly random index over the whole
    512-action head) can still stall past any cap: the cap narrows the mask
    but does not force `end_turn`. MaskablePPO never samples a masked action.
  - `tests/test_shipped_configs.py` enforces the formula for every stage.
- **Intermediate `max_turns` 60 → 75,** the v5x value:
  - 25–34% of v52a's decisive intermediate games ended after turn 60.
  - v34, the only run at 60, stalled at `intermediate_random_20` with every
    game drawn at turn 60.
  - The ladder was re-measured at 75 (§3.2).
- **`max_flat_actions` stays 512** (the decision already recorded). The v2
  decode tables drop moves first and keep every attack, heal, cast, seize
  and `end_turn`. An untrained policy reaches 770–805 legal actions on
  corner_points (the runbook's smoke test), and 15–17% of its decision
  points there are cut, so watch `flat_truncated_rate` once a seed trains
  on that map.
- **`pad_to_size: [10, 12]`, now explicit** in both curriculum files. The
  runner derives the padding from the stage maps, and it would resolve the
  same value for `bootstrap.yaml`. The slice has only 6×6 maps, so without
  the setting it trained unpadded: `SpatialFeatureExtractor`'s `coord_conv`
  planes span the padded grid, so a 6×6 map sits at [0, 0.56] × [0, 0.45]
  in the canonical run but spans [0, 1]² unpadded, and one forward and
  backward pass at batch 256 took 153 ms padded against 41.5 ms unpadded
  on CPU.

### 2.3 Not back-ported

- **`engine_overrides.damage_model: hp_scaled`** (v50–v52a). It is an engine
  rule change: it makes the training game differ from the deployed engine
  and from the ladder. Its evidence is n = 1: the hp_scaled runs reached 20,
  2, 20, 2 and 1 stages.
- **`unit_data.W.cost: 300`.** It broke starter: v38 and v39 stalled at
  `starter_random`, with a best win rate of 35–51%. The curriculum keeps the
  starter block.
- **`max_flat_actions: 1024`** (v52b/v54), and **a flat `max_steps: 4500`**
  (v54, which never ran).

`ppo` is unchanged. It is identical to v52a's: lr 3e-4, n_steps 2048,
batch 256, γ 0.99, GAE 0.95, clip 0.2, ent 0.05, `SpatialFeatureExtractor`.

---

## 3. Curriculum: 24 stages ordered by the ladder

After the rule-bot legality fix, 5 of the 33 old stages faced an opponent
weaker than the previous stage's on the same map. Three more added no
measurable step ([`bot_ladder_2026-09.md`](bot_ladder_2026-09.md)). The new
list keeps each map block's entry stage, puts each map's stages in ladder
order, adds mixed-bot bridges where a step was large, and drops the stages
that measure nothing.

**t** is the intended stochastic win rate, pooled over both seats, and
**T** is the Wilson threshold that matches it (§4). Budget and anneal are
env steps; the anneal is where the entropy schedule ends.

| # | Stage | Opponent | t → T | Budget / anneal | ent_coef |
|---|---|---|---|---|---|
| 1 | starter_simple | simple | 0.80 → 0.73 | 500k / 300k | 0.10→0.05 |
| 2 | starter_mixed_random_simple | mix(random/simple, 0.5) | 0.75 → 0.68 | 500k / 300k | 0.05 |
| 3 | starter_random | random (20) | 0.75 → 0.68 | 750k / 400k | 0.05 |
| 4 | starter_mixed_random_medium | mix(random/medium, 0.5) | 0.60 → 0.53 | 1M / 500k | 0.05 |
| 5 | starter_medium | medium | 0.50 → 0.43 | 1.5M / 750k | 0.05 |
| 6 | beginner_balanced_random | balanced_random | 0.85 → 0.79 | 1M / 300k | 0.10→0.03 |
| 7 | beginner_random_10 | random (10) | 0.75 → 0.68 | 3M / 1.5M | 0.10→0.03 |
| 8 | beginner_random_15 | random (15) | 0.70 → 0.63 | 3M / 1.5M | 0.10→0.01 |
| 9 | beginner_mixed_50 | mix(simple/medium, 0.5) | 0.60 → 0.53 | 3M / 1.5M | 0.05 |
| 10 | beginner_mixed_med_adv_50 | mix(medium/advanced, 0.5) | 0.60 → 0.53 | 3M / 1.5M | 0.05 |
| 11 | beginner_medium | medium | 0.50 → 0.43 | 4M / 2M | 0.05 |
| 12 | intermediate_balanced_random | balanced_random | 0.85 → 0.79 | 1M / 300k | 0.10→0.03 |
| 13 | intermediate_mixed_br_simple | mix(balanced_random/simple, 0.5) | 0.70 → 0.63 | 2M / 1M | 0.07→0.03 |
| 14 | intermediate_simple | simple | 0.65 → 0.58 | 3M / 1.5M | 0.05 |
| 15 | intermediate_mixed_simple_medium | mix(simple/medium, 0.5) | 0.60 → 0.53 | 3M / 1.5M | 0.05 |
| 16 | intermediate_medium | medium | 0.50 → 0.43 | 4M / 2M | 0.05 |
| 17 | skirmish_balanced_random | balanced_random | 0.85 → 0.79 | 1M / 300k | 0.10→0.03 |
| 18 | skirmish_mixed_br_simple | mix(balanced_random/simple, 0.5) | 0.70 → 0.63 | 2M / 1M | 0.07→0.03 |
| 19 | skirmish_simple | simple | 0.70 → 0.63 | 3M / 1.5M | 0.05 |
| 20 | skirmish_mixed_simple_advanced | mix(simple/advanced, 0.5) | 0.70 → 0.63 | 3M / 1.5M | 0.05 |
| 21 | skirmish_advanced | advanced | 0.60 → 0.53 | 4M / 2M | 0.05 |
| 22 | corner_points_balanced_random | balanced_random | 0.85 → 0.79 | 1.5M / 500k | 0.10→0.03 |
| 23 | corner_points_mixed_50 | mix(balanced_random/simple, 0.5) | 0.65 → 0.58 | 2.5M / 1.25M | 0.07→0.03 |
| 24 | corner_points_simple | simple | 0.50 → 0.43 | 4M / 2M (no retry) | 0.05 |

Every stage has patience 2 and `min_timesteps_before_promotion: 25_000`.
The minimum only keeps the stage-entry eval of the carry-in policy from
counting toward promotion. The lessons doc showed that 500k hurt.

### 3.1 What moved, what was added, what was dropped

- **Re-ordered.**
  - `starter_simple` now comes before `starter_random`: RandomBot beats
    SimpleBot on starter, and a cold-start policy plays like RandomBot.
  - `beginner_mixed_med_adv_50` now comes before `beginner_medium`: since
    the legality fix, MediumBot beats AdvancedBot on every curriculum map.
- **Bridges added** where the step was large:
  - starter: mix(random/simple) and mix(random/medium);
  - intermediate: mix(balanced_random/simple) and mix(simple/medium). The
    second splits simple → medium, which scored 0.91 head to head.
  - skirmish: mix(balanced_random/simple) and mix(simple/advanced). The
    second splits a 0.97 step. `skirmish_advanced` is now the map top.
  - corner_points: none. mix(balanced_random/simple) → simple is a 0.73
    step (350 games, §3.2).
- **Dropped.**
  - `beginner_simple` and `beginner_advanced`: flagged as steps down.
  - `beginner_random_20`: 0.54 head to head against random_15 on 100 fresh
    games; the next step, → mix(simple/medium), was 0.50.
  - `intermediate_random_20` and `intermediate_mixed_random_simple`: 0.51,
    then 0.52 against SimpleBot. Both draw.
  - `skirmish_random_10/15/20`, `skirmish_mixed_25`, `skirmish_mixed_50`
    (random/simple) and `skirmish_medium`:
    - RandomBot with `max_actions` 10, 15 or 20 is one opponent there;
      they draw each other.
    - v52a cleared the skirmish random stages at 1.0 within 50–150k steps.
    - MediumBot only ties SimpleBot on skirmish (0.51 over 300 fresh games),
      so a medium stage after advanced is flagged on rating.
  - `corner_points_random_10/15/20`, `corner_points_mixed_25` and
    `corner_points_medium`:
    - The random stages add no measurable step: random_10, 15 and 20 rate
      the same there and draw each other.
    - SimpleBot beats MediumBot head to head there (0.74), while MediumBot
      out-rates SimpleBot by beating the random bots. So a MediumBot stage
      is flagged whether it comes before or after simple.
    - The research draft kept MediumBot through a mix(medium/simple) stage
      before simple. The ladder run on seeds 7000–7024 flagged that order
      (§3.2): the corner top bots (medium, simple, advanced, master and
      their mixes) rate within a few dozen Elo of each other, so which of
      two of them rates higher changes with the seed set. SimpleBot rated +133
      against the mix's +121 on seeds 5000–5024, and +130 against +143 on
      seeds 7000–7024. The block now enters that cluster once, at its
      hardest-to-beat bot, SimpleBot.
- **The beginner random ramp stays** (balanced_random → random_10 →
  random_15). RandomBots draw each other in 99–100% of games, so their head
  to head measures nothing. The step shows in how hard each is to beat: the
  strong scripted bots win 0.958 of games against balanced_random, 0.719
  against random_10 and 0.610 against random_15 (z = +16.3 and +5.8, 250
  games per cell). `beginner_random_15` is the archive's wall stage, and the
  ramp's other purpose, learning to convert games a surviving opponent would
  draw, is a question for training, not for the ladder.

### 3.2 Ladder check of this order

On the final config, with fresh seeds, both seats, 25 seeds (50 games per
pairing) and the tool's default pool of 15 bots per board:

```
python scripts/eval/bot_ladder.py --config configs/ppo/bootstrap.yaml \
    --seed-base 8000 --seeds 25 --workers 4 --fail-on-flag --out-dir DIR
```

**Exit 0: no stage flagged** (37 min on 4 CPUs).
`tests/test_bootstrap_ladder_order.py` re-runs the same check on this run's
games, cached in `tests/data/bootstrap_ladder.json`. A slow test replays the
first three seeds of every step to show the cache still matches the engine.

| # | Stage | Opponent (rating) | Previous opponent (rating) | Head to head, W-D-L | Score [95% CI] |
|---|---|---|---|---|---|
| 1 | starter_simple | simple (−107) | — | — | first on map |
| 2 | starter_mixed_random_simple | mix(random_20/simple) (−6) | simple (−107) | 16-31-3 | 0.63 [0.49, 0.75] |
| 3 | starter_random | random_20 (+102) | mix(random_20/simple) (−6) | 29-15-6 | 0.73 [0.59, 0.83] |
| 4 | starter_mixed_random_medium | mix(random_20/medium) (+252) | random_20 (+102) | 34-3-13 | 0.71 [0.57, 0.82] |
| 5 | starter_medium | medium (+368) | mix(random_20/medium) (+252) | 33-0-17 | 0.66 [0.52, 0.78] |
| 6 | beginner_balanced_random | balanced_random (−371) | — | — | first on map |
| 7 | beginner_random_10 | random_10 (−208) | balanced_random (−371) | 0-49-1 | 0.49 [0.36, 0.62] |
| 8 | beginner_random_15 | random_15 (−113) | random_10 (−208) | 0-50-0 | 0.50 [0.37, 0.63] |
| 9 | beginner_mixed_50 | mix(simple/medium) (+106) | random_15 (−113) | 20-27-3 | 0.67 [0.53, 0.78] |
| 10 | beginner_mixed_med_adv_50 | mix(medium/advanced) (+352) | mix(simple/medium) (+106) | 30-0-20 | 0.60 [0.46, 0.72] |
| 11 | beginner_medium | medium (+451) | mix(medium/advanced) (+352) | 30-0-20 | 0.60 [0.46, 0.72] |
| 12 | intermediate_balanced_random | balanced_random (−282) | — | — | first on map |
| 13 | intermediate_mixed_br_simple | mix(balanced_random/simple) (−184) | balanced_random (−282) | 20-27-3 | 0.67 [0.53, 0.78] |
| 14 | intermediate_simple | simple (−75) | mix(balanced_random/simple) (−184) | 26-22-2 | 0.74 [0.60, 0.84] |
| 15 | intermediate_mixed_simple_medium | mix(simple/medium) (+118) | simple (−75) | 22-25-3 | 0.69 [0.55, 0.80] |
| 16 | intermediate_medium | medium (+423) | mix(simple/medium) (+118) | 32-1-17 | 0.65 [0.51, 0.77] |
| 17 | skirmish_balanced_random | balanced_random (−247) | — | — | first on map |
| 18 | skirmish_mixed_br_simple | mix(balanced_random/simple) (−118) | balanced_random (−247) | 22-25-3 | 0.69 [0.55, 0.80] |
| 19 | skirmish_simple | simple (−5) | mix(balanced_random/simple) (−118) | 24-24-2 | 0.72 [0.58, 0.83] |
| 20 | skirmish_mixed_simple_advanced | mix(simple/advanced) (+130) | simple (−5) | 36-12-2 | 0.84 [0.71, 0.92] |
| 21 | skirmish_advanced | advanced (+291) | mix(simple/advanced) (+130) | 38-0-12 | 0.76 [0.63, 0.86] |
| 22 | corner_points_balanced_random | balanced_random (−241) | — | — | first on map |
| 23 | corner_points_mixed_50 | mix(balanced_random/simple) (−29) | balanced_random (−241) | 17-33-0 | 0.67 [0.53, 0.78] |
| 24 | corner_points_simple | simple (+143) | mix(balanced_random/simple) (−29) | 28-21-1 | 0.77 [0.64, 0.86] |

Every mix is p_hard 0.5. Ratings are Bradley-Terry Elo on that board. The
score is the stage opponent's against the previous one, with a draw
counting half.

**Two more checks:**

- **Seeds 7000–7024** were played on the research draft's order, which had
  `corner_points_mixed_medium_simple` before `corner_points_simple`. That
  run flagged `corner_points_simple` on rating: +130 against +143, although
  SimpleBot won the head-to-head 22-26-2. This is why the stage was dropped
  (§3.1). A game's result does not depend on the rest of the pool, so the
  same games re-rated without that bot are exactly what the tool plays for
  the final config. `bot_ladder.py --from-json … --config
  configs/ppo/bootstrap.yaml --fail-on-flag` on them exits 0: no flags.
- **The research runs** used seeds 2000–2049 and 3000–3049 with smaller bot
  pools per map, 2100–2149 for the skirmish bridge, and 5000–5024 with the
  default pool. Re-checked with `--from-json` against the final config, the
  final order is not flagged in any of them. They played intermediate at 60
  turns, so only the other maps are checked there. Under v52a's engine
  overrides (hp_scaled damage, W cost 300; seeds 4000–4024) it is not
  flagged either, but the lower steps shrink.

**Every step, pooled over all seed sets.** Intermediate uses only seeds
7000–7024 and 8000–8024, because the research runs played it at 60 turns.
At 60 turns its steps were 0.68, 0.74, 0.71 and 0.70.

| Stage | Opponent vs previous | Games | Pooled W-D-L | Score [95% CI] | Draws |
|---|---|---|---|---|---|
| starter_mixed_random_simple | mix(random_20/simple) vs simple | 350 | 140-196-14 | 0.68 [0.63, 0.73] | 56% |
| starter_random | random_20 vs mix(random_20/simple) | 350 | 192-89-69 | 0.68 [0.62, 0.72] | 25% |
| starter_mixed_random_medium | mix(random_20/medium) vs random_20 | 350 | 206-39-105 | 0.64 [0.59, 0.69] | 11% |
| starter_medium | medium vs mix(random_20/medium) | 350 | 229-3-118 | 0.66 [0.61, 0.71] | 1% |
| beginner_random_10 | random_10 vs balanced_random | 350 | 1-347-2 | 0.50 [0.45, 0.55] | 99% |
| beginner_random_15 | random_15 vs random_10 | 350 | 1-349-0 | 0.50 [0.45, 0.55] | 100% |
| beginner_mixed_50 | mix(simple/medium) vs random_15 | 350 | 155-177-18 | 0.70 [0.65, 0.74] | 51% |
| beginner_mixed_med_adv_50 | mix(medium/advanced) vs mix(simple/medium) | 350 | 222-1-127 | 0.64 [0.58, 0.68] | 0% |
| beginner_medium | medium vs mix(medium/advanced) | 350 | 232-0-118 | 0.66 [0.61, 0.71] | 0% |
| intermediate_mixed_br_simple | mix(balanced_random/simple) vs balanced_random | 100 | 38-59-3 | 0.68 [0.58, 0.76] | 59% |
| intermediate_simple | simple vs mix(balanced_random/simple) | 100 | 51-45-4 | 0.73 [0.64, 0.81] | 45% |
| intermediate_mixed_simple_medium | mix(simple/medium) vs simple | 100 | 46-48-6 | 0.70 [0.60, 0.78] | 48% |
| intermediate_medium | medium vs mix(simple/medium) | 100 | 71-1-28 | 0.71 [0.62, 0.79] | 1% |
| skirmish_mixed_br_simple | mix(balanced_random/simple) vs balanced_random | 450 | 193-242-15 | 0.70 [0.65, 0.74] | 54% |
| skirmish_simple | simple vs mix(balanced_random/simple) | 450 | 247-179-24 | 0.75 [0.71, 0.79] | 40% |
| skirmish_mixed_simple_advanced | mix(simple/advanced) vs simple | 350 | 184-131-35 | 0.71 [0.66, 0.76] | 37% |
| skirmish_advanced | advanced vs mix(simple/advanced) | 350 | 259-0-91 | 0.74 [0.69, 0.78] | 0% |
| corner_points_mixed_50 | mix(balanced_random/simple) vs balanced_random | 350 | 122-228-0 | 0.67 [0.62, 0.72] | 65% |
| corner_points_simple | simple vs mix(balanced_random/simple) | 350 | 176-158-16 | 0.73 [0.68, 0.77] | 45% |

- **Every step but the beginner random ramp scores 0.64–0.75,** with a
  lower 95% bound of at least 0.58.
- **The ramp's step shows in how often the strong bots beat each opponent**
  (simple, mix(simple/medium), medium, advanced and master; 500 games per
  cell over seeds 7000–7024 and 8000–8024):
  - balanced_random: 0.966;
  - random_10: 0.712;
  - random_15: 0.616.

  The research's 250 games per cell gave 0.958 / 0.719 / 0.610.

---

## 4. Promotion gate and evaluation

- **Gate: `promotion_criterion: wilson`, one-sided 95%, `promotion_score:
  win_rate`, patience 2.** A stage promotes when the Wilson lower bound of the
  pooled two-seat win rate reaches T on two consecutive evals. Draws count
  as losses.
- **Thresholds.** With n = 60 episodes per seat × 2 seats = 120:
  T = round(`wilson_lower_bound(ceil(120·t), 120, z_for_confidence(0.95))`, 2).

  | t | 0.85 | 0.80 | 0.75 | 0.70 | 0.65 | 0.60 | 0.50 |
  |---|---|---|---|---|---|---|---|
  | T (n = 120) | 0.79 | 0.73 | 0.68 | 0.63 | 0.58 | 0.53 | 0.43 |
  | wins needed of 120 | 103 | 96 | 91 | 85 | 79 | 73 | 61 |
  | T (n = 80) | 0.77 | 0.72 | 0.66 | 0.61 | 0.56 | 0.51 | 0.41 |
  | T (n = 160) | 0.80 | 0.74 | 0.69 | 0.64 | 0.59 | 0.54 | 0.44 |

  If you change `n_eval_episodes` or `eval_seats`, recompute T. The test
  fails until you do.
- **How t was set.** The gate pools the seats, so T asks for a number of
  wins out of 120 (the table's second row), and neither seat can supply
  them alone: even 60 of 60 seat-1 wins leave 13 to find in seat 2 at
  t 0.60, and 25 at t 0.70. So t is capped by what the scripted bots
  achieve against the stage's opponent in both seats, measured on the
  ladder (win rates counting wins only, as the gate does):
  - **Bridge stages** ask at most the best scripted bot's pooled win rate
    against their opponent. Stages that existed before keep their old point
    threshold as t where that fits (0.85 for balanced_random, 0.75 for
    random_10, 0.70–0.65 for most others); the starter block drops from
    0.90 to 0.80 / 0.75 / 0.75.
  - **The four bridges into MediumBot get t 0.60**: `starter_mixed_random_medium`,
    `beginner_mixed_50`, `beginner_mixed_med_adv_50` and
    `intermediate_mixed_simple_medium`. The best scripted bot wins 0.62–0.66,
    0.68–0.70, 0.60–0.70 and 0.66–0.70 of its games against these opponents
    (the cached seeds 8000–8024 and fresh seeds 11000–11024, 50 games per
    pairing each). On starter and beginner most of those wins come moving
    first: MediumBot wins 0.92–1.00 of its seat-1 games against these mixes
    and 0.24–0.40 of its seat-2 games; on intermediate AdvancedBot does best
    (0.76 / 0.64). At t 0.70 (85 of 120) a policy would have to beat every
    scripted bot's two-seat record against them.
  - **Map tops get t 0.50** (`starter_medium`, `beginner_medium`,
    `intermediate_medium`, `corner_points_simple`). No scripted bot wins
    more than 0.40 of its games against these opponents (0.34–0.36,
    0.36–0.40, 0.36–0.38 and 0.32–0.34), and MediumBot moving first is
    beaten in at most 24% of games on starter and beginner (RandomBot) and
    4–8% on intermediate. t 0.50 needs 61 of 120 wins, which a policy that
    wins nearly every seat-1 game reaches with one or two seat-2 wins; at
    t 0.60 it would need 13. It still asks for more than any scripted bot
    achieves: that is what a map top is for.
  - **`skirmish_advanced` keeps t 0.60**: there MediumBot wins 0.66 of its
    games against AdvancedBot, 0.56 moving second.

  `tests/test_bootstrap_ladder_order.py` checks these caps on the cached
  ladder: a bridge's wins needed / 120 may exceed the best bot's pooled
  rate by at most 0.01 (T's rounding), and a map top asks at most
  max(best bot, 0.50) and no more than a perfect seat 1 plus the best
  scripted seat-2 record. The best of 14 bots over 50 games per pairing is
  biased upward, which is why the four MediumBot bridges sit at t 0.60,
  below both seed sets, rather than at the 0.70 the cache alone would
  allow for two of them.
- **Gate power.** Modelled as 120 Bernoulli episodes per eval, with the
  repo's `wilson_lower_bound`:
  - a policy at exactly t passes one eval 46–47% of the time, and 55% at
    t 0.80 (`starter_simple`). Elsewhere rounding T to two decimals raises
    it just above the Wilson bound of 120·t wins, so one more win is
    needed; at t 0.80 it lowers it, and 96 = 120·t wins suffice;
  - at t + 0.05, 84–95%;
  - at t + 0.10, two consecutive passes 97–100%.
  - A policy at t − 0.05 passes one eval with probability 0.07–0.12, and
    promotes within a 10-eval (1M-step) plateau with probability
    0.035–0.115. At t − 0.10 it essentially never promotes.
  - For comparison, the old point gate at 80 episodes passed a policy
    sitting at its threshold 55–58% of the time, about as often. But it
    promoted a policy 0.05 below the threshold within 10 evals 20–29% of the
    time (the review's `rltrain-12`).
- **Both seats.** `env.agent_seat: random` trains the agent as player 1 in
  half the episodes and as player 2 in the other half. `eval_seats: [1, 2]`
  evaluates 60 episodes per seat on the same seeds.
  - The review (`critic-gaps-2`) found every archived checkpoint was trained
    and gated as the first mover only, but deployed in both seats.
  - **`seat_aggregate: mean`, not `min`.** MediumBot moving first is rarely
    beaten by any scripted bot: in at most 24% of games on starter and
    beginner (RandomBot, cached and fresh seeds) and 4–8% on intermediate.
    The best bot's pooled win rate against MediumBot is only 0.34–0.40, and
    against corner SimpleBot 0.32–0.34.
  - That is why the map tops use t = 0.50 pooled. With `min`, a policy at
    0.90 / 0.70 in the two seats passes a t = 0.70 eval 34–45% of the time;
    with `mean` it passes 100% of the time.
- **Stochastic only.** `eval_deterministic: false` and
  `eval_both_modes: false`.
  - Eval was 72–94% of v52a's wall-clock.
  - Recording the greedy number too takes the expected run from about 24 h
    to about 42 h per seed.
  - Get the greedy comparison from a post-run eval of each `best_model.zip`
    (§1).
- **Recovery.**
  - `max_retries: 1`: a stalled stage gets one more budget, restarted from
    its best checkpoint. The capstone gets 0.
  - `regression_guard_evals: 3` and `regression_guard_drop: 0.25`: after 3
    consecutive evals whose win rate is more than 0.25 below the stage's
    best, the stage restores `best_model.zip` and keeps training.
- **Eval cost.**
  - `eval_freq: 100_000`, so a stage gets one eval per ~6 PPO updates.
  - `n_eval_envs: 8` with `eval_use_subprocess: true`, stepped with one
    batched predict.
  - `checkpoint_freq: 50_000` is the rolling `latest.zip` that `--resume`
    continues from.

---

## 5. Budgets and compute

- **Budget.** Sum of `max_timesteps`: 55.25M env steps. If every stage also
  uses its retry: 106.5M. (The old file's header said ~22M worst case; its
  33 stages actually summed to 87.5M.)
- **Expected steps.** As a prior, assume a stage stops at 40% of its budget,
  20% on map-entry stages: about 21M steps per seed. There is no data on
  the fixed bots yet: in v52a, the scripted-bot stages promoted within 100k
  steps because the bots were crippled then.
- **Wall-clock model** for one L4 with 12 vCPUs and 8 training envs.
  - It is fitted to v52a's run: 5.85M training steps and 12.74M serial
    greedy eval agent-steps in 14.7 h.
  - Assumptions: about 1000 training steps/s; 270 agent-steps/s for serial
    eval; the 8 subprocess eval envs ~2.5× faster than serial.
  - Average eval-episode length on a trained agent: 250 / 900 / 1300 /
    1800 / 3000 agent steps on starter / beginner / intermediate (at 75
    turns) / skirmish / corner_points.

| Per seed | Steps | Central | Range (eval ×1.6–4, train 600–2000/s) |
|---|---|---|---|
| Expected | ~21M | **~24 h** | 14–38 h |
| Every stage runs out its budget | 55.25M | ~58 h | 35–93 h |
| … and every stage also uses its retry | 106.5M | ~106 h | 63–169 h |

- **Per map** (central model, expected / full budget): starter 0.8 / 1.8 h,
  beginner 5.3 / 12.8 h, intermediate 5.2 / 12.6 h, skirmish 6.7 / 16.1 h,
  corner_points 6.0 / 15.0 h. One eval takes about 0.7 / 2.7 / 3.9 / 5.3 /
  8.9 min on the same maps.
- **Other eval settings,** modelled at expected steps:

  | Setting | Wall-clock per seed |
  |---|---|
  | 100k stochastic only (shipped) | 24 h |
  | 100k, both modes | 42 h |
  | 50k, both modes | 72 h |
  | v52a-style (50k, 80 greedy episodes, seat 1, serial) | 61 h |

- **The 3-seed run** needs about 3 × 24 ≈ 72 L4-hours expected, and
  3 × 58 ≈ 175 if every stage runs out its budget.
- **The ×2.5 eval speed-up is an assumption for the L4.** On an idle
  4-vCPU CPU box with an untrained policy, 8 subprocess eval envs ran 2.3×
  (starter_medium) and 2.5× (beginner_random_10) faster than serial, and
  1.9–2.2× on the larger maps; in-process `n_eval_envs: 4` ran 1.5–1.7×.
  (The 1.0–1.5× this section quoted before did not reproduce there.) If it
  underperforms, the in-process setting gives about 34–38 h expected.
- **Calibrate from the slice's report** (§6; runbook §3.1), not from
  TensorBoard `time/fps` or stage wall-times, which mix training, PPO
  updates and evals. `summarize_seeds.py` reports, per map, eval agent
  steps/s (`sum(lengths) / eval_seconds` over the eval rows) against the
  model's 675/s (270 × 2.5), and training env steps/s
  (Δtimesteps / (Δwall_time − eval_seconds) between rows) against its
  1000/s, and each seed's `active h` (session time, not the span between
  its first and last eval). The slice runs no serial eval, so the ×2.5
  itself is not measured there; the absolute eval rate is what the model
  needs. Unmeasured so far: a trained agent's episode lengths on beginner
  to corner_points, which dominate the estimate (an untrained policy
  played about 6800-agent-step corner_points games, against the modelled
  3000).

---

## 6. The validation slice (`bootstrap_validation.yaml`)

The slice is the first 8 canonical stages: all of starter, then
`beginner_balanced_random`, `beginner_random_10` and `beginner_random_15`.
The env, reward, PPO, eval and gate settings are the same as in
`bootstrap.yaml`, and so are the thresholds. The env includes the padding:
the slice sets `pad_to_size: [10, 12]` itself, because its maps are all 6×6
and the runner would otherwise not pad them (§2.2). Only three things
differ:

- **Budgets are about halved.** The total is 4.75M env steps.

  | Stage | Budget / anneal (canonical) |
  |---|---|
  | starter_simple | 300k / 200k (500k / 300k) |
  | starter_mixed_random_simple | 300k / 200k (500k / 300k) |
  | starter_random | 400k / 250k (750k / 400k) |
  | starter_mixed_random_medium | 500k / 300k (1M / 500k) |
  | starter_medium | 750k / 400k (1.5M / 750k) |
  | beginner_balanced_random | 500k / 300k (1M / 300k) |
  | beginner_random_10 | 1M / 600k (3M / 1.5M) |
  | beginner_random_15 | 1M / 600k (3M / 1.5M) |

- **`eval_freq: 50_000`,** so short stages get 3 or more evals.
- **`max_retries: 0`.** The probe contract is "abort and diagnose a stall,
  don't extend it".

Expect about 2 h, or about 4.5 h if every stage runs out its budget. What
to look at:

- the per-seat win rates on `starter_mixed_random_medium` and
  `starter_medium` (seat 2 against MediumBot moving first is the expected
  hard half; at t 0.60 and 0.50 a strong seat 1 carries most of the gate);
- `avg_turns` climbing toward `max_turns` (the draw attractor);
- `flat_truncated_rate`;
- whether `beginner_random_15` trends up.

`tests/test_shipped_configs.py` fails if the slice drifts from the
canonical file in anything but these three settings, comparing the configs
as the runner resolves them (`bootstrap.resolve_config`), so a derived
difference such as the padding fails it too. The command that runs
it is in the runbook ([`validation_run.md`](validation_run.md) §3.1). The
slice restates the canonical file in full; §10.12 notes the `extends:` form
it could take instead.

---

## 7. Self-play (`self_play.yaml`)

- **`latest_opponent_prob: 0.0 → 0.5`.** Once the pool holds a snapshot,
  each episode plays the latest snapshot with probability 0.5, and a
  uniform pool sample otherwise.
  - **At 0.0**, the latest snapshot, pushed every 10k steps, only finished
    the episodes already in flight. Every other episode faced a uniformly
    drawn gated snapshot up to a whole window old: a stale, early-weak
    target.
  - **At 1.0**, the pool is dead weight, and pure self-play risks cycling.
    The archive already shows composition cycling: v40 went mass-Archer ↔
    Warrior+Knight ↔ Knight, abandoning each within 50–100k steps.
  - **Published mixes.** FSP/NFSP best-respond to the average of past
    policies. OpenAI Five played about 80% against the latest and 20%
    against past selves. AlphaStar's main agents played about 35%
    self-play, 50% prioritized past players and 15% forgotten
    players/exploiters. Bansal et al. (2018) found that sampling from a
    history window beat always-latest.
  - 0.5 sits in that range and leans toward history because unit
    compositions are non-transitive.
- **`pool_size: 10 → 20` and `add_to_pool_freq: 50_000 → 100_000`.** The
  history window grows from about 500k steps to about 2M, 40% of the 5M
  run. Each worker process holds its own copy of the pool, so pool memory
  roughly doubles per worker.
- **`min_win_rate_for_pool: 0.55 → 0.0`.** This deviates from the research
  recommendation (0.50); see §10.7.
  - The admission gate counts the agent's wins over all games finished since
    the last check. Draws and truncations count as non-wins.
  - Until the pool holds its first snapshot, every game is against the
    latest snapshot, a copy of the agent at most 10k steps old. The expected
    win rate is (1 − draw rate)/2, which is never above 0.5.
  - With a 0.55 or 0.50 threshold, the first snapshot is admitted only by
    sampling noise. The pool then stays empty and `latest_opponent_prob`
    does nothing.
  - 0.0 turns the pool into a plain history window (every snapshot is
    admitted, oldest evicted first), the standard FSP / Bansal et al.
    design.
  - A real gate needs a code change: score pool-opponent games only, with
    wins + 0.5 · draws. Watch `self_play/pool_size` in TensorBoard.
- **`pool_strategy: uniform`,** unchanged. `prioritized` weights a
  snapshot by its own admission win rate, not by how hard it is for the
  current agent, so it is not PFSP.
- **Precondition, fixed in the same change: the env block now matches the
  bootstrap env.** The old block had `max_steps: 200` and no `map_file`, so
  every env generated its own random 20×20 map. It also used
  `multi_discrete`, no `max_turns` and the 1000-scale default rewards.
  Nearly every game truncated at 200 steps, so wins and admissions stayed
  near 0 whatever the knob said. Now:
  - `maps/1v1/skirmish.csv` at 120 turns: the map where the deepest runs
    stopped, and one where SimpleBot, the unchanged eval opponent, is a
    real bar (MediumBot only ties it there, 0.51 over 300 games);
  - `flat_discrete` with 512 actions, `max_actions_per_turn: 60` and
    `max_steps: 7400` (the §2.2 formula);
  - `pad_to_size: [10, 12]`, the curriculum's padding, so `--resume-from` a
    bootstrap checkpoint loads;
  - the bootstrap reward block and `ppo` block, including `policy_kwargs`
    and batch 256;
  - `agent_seat: random` for the eval env. The workers already alternate
    seats (`swap_players`).
- **Eval.** `eval_freq` went from 10k to 100k and `n_eval_episodes` from 10
  to 20. The old cadence was sized for 200-step episodes; a skirmish game
  runs about 1000–2500 agent steps.

`tests/test_shipped_configs.py` checks that the self-play env, rewards,
padding and policy match `bootstrap.yaml`. The self-play run is a follow-up
and is not part of §2.1.

---

## 8. Checks run

All checks were run on this branch, on the fixed code (`864e62c` plus
these config changes):

- **Config loading.** Every config loads, passes
  `validate(check_files=True)` and sets no field its entry point ignores
  (`tests/test_shipped_configs.py`). `bootstrap.yaml` and
  `bootstrap_validation.yaml` are clean under the curriculum runner's
  `--strict` check. `self_play.yaml` parses through
  `train_self_play.py --strict`. All 31 reward keys are explicit.
- **Ladder.** See §3.2: seeds 8000–8024 on the final config exit 0 with no
  stage flagged.
- **`max_steps` bound.** Starter, beginner, intermediate, skirmish and
  corner_points were played in both seats with a random-legal policy and
  with a policy that never ends its turn while another legal action exists.
  Every game ended by elimination, HQ capture or the max-turns clock, never
  by truncation. The longest were the corner_points random games: 9251 and
  9396 steps over 200 turns, against `max_steps` 12300.
- **Training smoke run.** `train_bootstrap.py` ran on
  `bootstrap_validation.yaml` with `--strict --device cpu` and these
  overrides: `--set env.n_envs=2 ppo.n_steps=128 ppo.batch_size=64
  ppo.n_epochs=2 eval.eval_freq=1024 eval.n_eval_episodes=2
  eval.n_eval_envs=2 eval.checkpoint_freq=512` (each passed as its own
  `--set`).
  - It built the 8-stage slice and reached `starter_simple`'s evals at
    steps 2 (the carry-in eval), 1,024 and 2,048. Each ran with the
    stochastic policy, both seats and the Wilson gate against 73%, with
    `max_steps` 1300 and `max_turns` 20.
  - Every eval row carries the fields §1 reports: `by_seat`,
    `captures_by_type`, `reward_components`, `action_counts.end_turn` and
    `end_reasons`. One eval already had an `hq_capture` ending, and there
    were no `max_steps_truncate` endings.
  - The run was interrupted, then `--resume RUN_DIR` continued it from its
    rolling checkpoint at step 2,304 and ran the next eval at 3,072.
- **Self-play smoke run.** `train_self_play.py --config
  configs/self_play/self_play.yaml --strict` ran with tiny budgets (3,072
  steps, 2 envs) on skirmish, subprocess workers and the pool enabled.
  With `min_win_rate_for_pool: 0.0`, the first pool check that saw a
  finished game admitted the snapshot (pool size 1).
- **Lint, types and tests.** `ruff check .`, `ruff format --check .` and
  `python -m mypy .` are clean. The default `pytest` run: 3068 passed, 9 skipped.
  `pytest -m slow --no-cov`: 62 passed.
- **After merging the seed tooling** (`06f597a`, the runbook's launcher and
  aggregator):
  - `train_bootstrap.py --strict --check-only` loads both configs through
    the new loader: 24 stages, 55.25M (106.5M with retries); the slice
    8 stages, 4.75M.
  - A 2-seed group (42, 1042) of the slice ran end to end through
    `run_seeds.py --parallel 2` on 4 CPUs with tiny `--set` budgets (2 train
    and 2 eval envs, 2,048 steps per stage, gate zeroed on the first seven
    stages): both seeds cleared 7 stages and stalled at
    `beginner_random_15` (exit 3, as the probe contract wants), and
    `summarize_seeds.py` reported both with per-seat gate win rates and
    both shaping shares.
  - The runbook's smoke test (§3 there) on `bootstrap.yaml`, one seed, with
    2 train and 2 eval envs on 4 CPUs: every stage's envs and evals built,
    24/24 stages cleared at their first eval, every one flagged
    `skip_ahead`, exit 0, in 9 minutes.
  - Lint and types clean; the default `pytest` run: 3250 passed, 9 skipped;
    `pytest -m slow --no-cov`: 64 passed.
- **After the review of this prep** (2026-09-28):
  - The review's fresh ladder (seeds 11000–11024, `--fail-on-flag`, 3184 s
    on 4 CPUs) flagged no stage; its best scripted win rates per stage sit
    in the ranges of §4 next to the cached run's.
  - The gates in §3–4; `tests/test_bootstrap_ladder_order.py` checks them
    against the cached ladder, and the first draft's thresholds fail it.
  - Both curriculum files resolve to `pad_to_size` (10, 12).
  - The seed tooling with the real trainer (a tiny config, 4 CPUs): after
    SIGKILL of a `--parallel 2` launcher, `status` showed both orphaned
    seeds `running`, `--group <id> --dry-run` left them alone, the original
    launch command was refused and named the group, and a second
    `train_bootstrap.py` on a live run dir exited 1 without writing. After
    the orphans got SIGTERM, both read `interrupted` and `--group <id>`
    alone queued them with the stored `--parallel 2 --no-pin --device cpu`.
    SIGHUP to a launcher: the child checkpointed and exited 143, the
    launcher exited 129, and the seed was recorded `interrupted`.
    SIGKILL of a teed launcher: the child had written its log itself, so it
    kept logging and ran to completion.
  - Lint and types clean; the default `pytest` run (with main's pygame
    changes merged in): 3348 passed, 9 skipped; `pytest -m slow --no-cov`:
    64 passed.

---

## 9. What this changes for comparisons with the archive

- **The comparison with v52a is qualitative.** The code fixes changed
  several things at once:
  - opponents no longer freeze;
  - the rule bots got stronger, MediumBot most;
  - combat shaping is symmetric and counts counter-attacks;
  - truncation no longer pays the draw;
  - every archived promotion was gated on the greedy policy in seat 1.
- **v52a's curriculum deltas are not reproduced:** the starter skip,
  MixedBot random pairs on the beginner random stages, extra intermediate
  random stages and a 3M budget floor. This config follows the ladder
  instead.
- **`turn_penalty: -1.0` with `draw == loss`** makes a fast loss cheaper
  than a full-clock draw: a loss at turn t costs −50 − t; a draw costs
  −50 − max_turns. v52a had the same property. If a seed's loss rate rises
  while its episodes get shorter, this is the first suspect.

---

## 10. Judgment calls you may want to revisit

1. **`seize_progress` 3.0.** Kept for parity; it is the largest farmable
   stream.
   - It paid 35–68 per episode on beginner/intermediate and 147–282 on
     skirmish in the archive.
   - Partial seizes pay without a capture.
   - On the fixed code, a random policy nets +34 per episode on skirmish at
     3.0 and −82 at 1.0. Completed captures barely change: a 40-HP building
     taken by a 15-HP unit pays 9 + 40 at 3.0 and 3 + 40 at 1.0.
   - **Queue `seize_progress: 1.0` as the first one-knob arm.** Start it
     early if any seed reaches skirmish with draw-heavy evals returning > 0.
2. **`max_actions_per_turn` 60.** It never binds on skirmish and below, but
   random play hits it on 21% of corner_points turns.
   - If a seed reaches corner_points, raise it to about 80 and let
     `max_steps` follow the formula.
   - A per-episode "budget-capped turns" counter in `episode_stats` would
     show whether it binds; none exists yet.
3. **Intermediate `max_turns` 75.** The minimal-diff alternative is 60 with
   `max_steps: 3800`, at the cost of turning about 25–34% of decisive games
   into draws.
4. **`win_by_hq_capture` 80 over 50.** Parity only. It replaces a test
   invariant that HQ ≤ elimination; the evidence says it doesn't matter
   either way.
5. **Map tops at t = 0.50, the MediumBot bridges at t = 0.60, gate on the
   seat mean** (§4). They were 0.60 and 0.70 in the first draft, carried
   over from the old seat-1 thresholds; no scripted bot reached them in
   both seats, so the slice would probably have stalled at stage 4 or 5.
   - If a map top still plateaus with seat-1 win rate ≥ 0.9 and seat 2 near
     0, the next step down is t = 0.45 (T = 0.38, 55 of 120 wins).
   - The alternative is `agent_seat: 1` everywhere: comparable with v52a,
     but it leaves seat 2 untrained.
6. **Starter block kept.** v52a and v40 skipped starter. Starting at
   `beginner_balanced_random` would save 4.25M of budget, and starter_medium
   carries the same first-mover risk as the other medium stages.
7. **`min_win_rate_for_pool` 0.0.** See §7; it deviates from the research
   recommendation of 0.50. Revert to a real gate once the code scores pool
   games only.
8. **Self-play on skirmish.** Beginner (75 turns, `max_steps: 4700`) would
   be cheaper, but SimpleBot is a weak eval opponent there (RandomBot 15/20
   out-rate it), and a test pins the shipped eval opponent to SimpleBot.
   Beginner would need `eval_opponent: medium` as well.
9. **Eval cost.**
   - `eval_both_modes: false` saves about 1.8× in compute.
   - For long maps, `n_eval_episodes: 40` per seat on skirmish/corner (with
     T recomputed at n = 80: 0.85 → 0.77, 0.70 → 0.61, 0.65 → 0.56,
     0.60 → 0.51, 0.50 → 0.41) saves about a third of their eval cost. But a policy
     0.05 below t would then promote about 20–50% of the time instead of
     about 10%.
10. **Non-transitive map tops.**
    - Skirmish ends at advanced, with no MediumBot stage, though MediumBot
      beats AdvancedBot 0.62 head to head.
    - Corner_points has no MediumBot stage at all. The research draft's
      mix(medium/simple) bridge before `corner_points_simple` was dropped
      because its order against SimpleBot is a coin flip on rating (§3.1),
      though SimpleBot beats it head to head (0.69, 300 games). If you
      want MediumBot on corner_points, make that mix the capstone instead of
      pure SimpleBot. It is a 0.70 step over mix(balanced_random/simple),
      but SimpleBot is the harder bot to beat (beaten 10–11% against
      20–21%).
    - Every other ordering of the top bots is flagged by the ladder's rating
      check on some seed set.
    - A stronger final opponent has to come from outside the scripted
      tiers: self-play or a frozen checkpoint.
11. **Budgets are a prior.** Run the slice first and calibrate them, and
    the eval throughput, from its report's per-map throughput and active
    hours (§5).
12. **`extends:` and `curriculum.stage_defaults` are not used.** The seed
    tooling added both to the config loader
    ([`configs/README.md`](../configs/README.md)). The configs were left as
    literal files for this run, but two places would read more clearly:
    - The slice as `extends: bootstrap.yaml` plus `eval.eval_freq: 50_000`,
      `curriculum.max_retries: 0`, a `drop_stages` list of the 16 later
      stages and the 8 budget / anneal pairs, about 35 lines. Checked: that
      file resolves to exactly `bootstrap_validation.yaml` (no differences
      in the loaded config). `tests/test_shipped_configs.py` would then
      compare two resolved configs instead of two copies.
    - In `bootstrap.yaml`, `stage_defaults: {patience: 2,
      min_timesteps_before_promotion: 25_000}` would drop 48 repeated lines.
      The per-map `max_turns` / `max_steps` pairs cannot move there: a
      default applies to every stage.
