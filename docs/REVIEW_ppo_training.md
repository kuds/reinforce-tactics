# PPO Bootstrap Training Review — why runs don't clear the curriculum (2026-07-12)

Scope: `notebooks/ppo_bootstrap.ipynb`, `notebooks/ppo_training.ipynb`,
`reinforcetactics/rl/` (env, callbacks, bootstrap runner), `configs/ppo/`
(bootstrap.yaml + all 58 sweep variants), the bot ladder in
`reinforcetactics/game/bot.py`, and the full run history on Drive
(`benchmarks/bootstrap/runs_summary.csv` / `runs_per_stage.csv` /
`runs_detail.json`: 109 run dirs, 2026-05-07 → 2026-06-02, 56 usable runs, 44
distinct configs, 274 attempted stage-rows). Every claim below was verified
against the code at HEAD (`5178e57`) and recomputed from the CSVs; several
were checked with fresh bot-vs-bot simulations.

## TL;DR — ranked causes

1. **Policy collapse into a reward-positive draw attractor** is the dominant
   observed stall mechanism — not inability to beat the bots. 24/32
   wall-stage stalls *peaked at or above their promotion threshold* and then
   crashed to a median final WR of ~0.06–0.14, with games pinned at
   ~73/75 turns and **positive average reward while losing** (29/41 stalls).
   Dense shaping still pays when you never win.
2. **`configs/ppo/bootstrap.yaml` still ships the two reward terms the v27
   ablations proved to independently cause that wall**
   (`win_speed_bonus: 50`, `enemy_owned_capture: -15`), plus
   `turn_penalty: 0`. The v28/v49/v52a fixes that produced every deep run
   were never back-ported to the canonical config.
3. **Several promotion gates sit at or above the scripted-bot ceiling.**
   Fresh simulations from the agent's seat (P1, stage `max_turns`): on
   `beginner_random_15` (gate 0.70/75 turns) the best any repo bot manages is
   AdvancedBot at 0.63 (SimpleBot 0.00 with 90% draws — the rebuild-economy
   meat-wall is a draw machine, and `win_rate` counts draws as losses,
   `evaluation.py:357`). `intermediate_medium`'s 0.65 gate is ~2× the best
   scripted WR (0.30). PPO *has* cleared these gates, but only transiently —
   the gates sit on a statistical knife edge under ±10pp eval noise.
4. **Credit assignment cannot see the terminal reward.** One env step = one
   unit micro-action, so episodes run ~700–2400 steps; with γ=0.99 the ±50
   terminal is discounted to ~1e-3 from early-game decisions and the GAE
   window (1/(1−γλ) ≈ 17 steps) spans 1–2 game turns. The dense event
   economy is the de-facto objective. γ, lr, n_steps, batch, clip were
   **identical across all 59 configs** — the optimization side of the
   collapse was never attacked.
5. **Wall-clock + no resume caps the best runs.** The 33-stage budget sums
   to 87.5M env steps; the deepest run ever reached 7.9M. Both 20-stage runs
   (v50, v52a) ended **on a just-cleared stage, still promoting**, killed by
   the session, and `run_curriculum` has no resume-from-stage-K. 13 of 33
   stages (skirmish_random_20 onward + all corner_points) have never been
   attempted by any run. Separately, the first `CurriculumStalled` aborts
   the whole run — there is no retry-from-best-checkpoint even when the
   stage's `best_model.zip` was above threshold.
6. **Latent representation ceilings** (not yet the binding constraint, but
   waiting at the frontier): positional `flat_discrete` action indices with
   per-step-rebuilt semantics; the 512-action cap truncates 728–744-action
   skirmish states *dropping all attack/heal/cast actions* (moves enumerate
   first); `masked_avg` pooling collapses the map to 64 numbers before the
   action head; `max_steps: 3000` binds before `corner_points`' 200-turn
   clock, converting long games into −50 truncation draws.

Also: the committed notebook defaults are currently broken on Run-All
(`BUILD_BC_WARMSTART = True` raises `RuntimeError` against the default
flat_discrete config before training starts), PPO train diagnostics
(approx_kl, clip_fraction, explained_variance) are collected but never
persisted, and every one of the 59 sweep variants ran with `seed: 42`, n=1 —
same-config reruns flip between clear and stall, so most single-knob sweep
verdicts are inside the noise band.

---

## 1. What the run data actually shows

### Funnel (attempted = evals_run > 0; 56 usable runs)

| stage | att | clear | stall | clear% | med peak WR | med final WR |
|---|---|---|---|---|---|---|
| starter_random | 38 | 32 | 6 | 84% | .975 | .956 |
| starter_simple / starter_medium | 32 | 32 | 0 | 100% | 1.0 | 1.0 |
| beginner_balanced_random | 47 | 47 | 0 | 100% | 1.0 | 1.0 |
| **beginner_random_10** | **43** | **28** | **15** | **65%** | 1.0 | .838 |
| **beginner_random_15** | **23** | **10** | **13** | **43%** | .888 | .488 |
| beginner_random_20 | 6 | 4 | 2 | 67% | .819 | .769 |
| beginner_simple/mixed/medium/advanced | 4 each | all | 0 | 100% | .94–1.0 | .94–1.0 |
| intermediate_* (through random_15) | 3–4 | all | 0 | 100% | 1.0 | 1.0 |
| **intermediate_random_20** | 4 | 2 | 2 | 50% | .913 | .469 |
| intermediate_simple/medium/mixed | 2 | 2 | 0 | 100% | .80–.83 | .77–.81 |
| skirmish (through random_15) | 2 | 2 | 0 | 100% | ~1.0 | ~1.0 |
| skirmish_random_20 → corner_points_medium (13 stages) | **0** | — | — | — | — | — |

- The `random_10` / `random_15` pair accounts for **28 of 41 stalls**. The 6
  starter_random stalls are all self-inflicted configs (warrior nerfs,
  reduced units, BC warm-start, patience=4).
- Runs that get *past* the wall clear everything they touch — medium and
  advanced bots included (38/38 medium-stage clears, 4/5 advanced) — until
  the session dies. The late curriculum is not "too hard"; it is
  **unexplored**.
- Deepest runs: `20260601_172412` (v52a) and `20260531_165459` (v50), both
  20/33 stages at 5.5–5.85M steps, both ended with `skirmish_random_15`
  **CLEARED at ~1.0 WR** — wall-clock deaths, not stalls.

### Stall forensics — collapse, not incapacity

| stalled stage | n | peak WR q25/50/75 | med final WR | peak ≥ gate | med avg_turns | med avg_reward |
|---|---|---|---|---|---|---|
| beginner_random_10 | 15 | .79/.90/.99 | **.06** | 14/15 | **72.8** / 75 | **+18.2** |
| beginner_random_15 | 13 | .71/.75/.90 | **.14** | 10/13 | **73.1** / 75 | **+23.6** |
| intermediate_random_20 | 2 | .69/.79/.88 | .02 | 1/2 | 67.1 | +10.5 |

- Every stalled stage consumed **100% of its step budget**: "stalled" means
  the policy oscillated at/above threshold, never got `patience=2`
  *consecutive* 80-episode evals over the bar, and continued PPO updates
  eventually pushed it into the draw attractor.
- **29/41 stalls end with positive average reward at WR ≤ 0.2.** Extreme:
  v40_skip_starter finished at WR .0375 with avg_reward **+58.5**;
  v53c pinned avg_turns at exactly 75.0 (100% draws) for 5M steps.
- Restart proof that collapse is training-induced: v21's
  `consolidate_a` re-ran the *just-cleared* random_10 stage and regressed
  from cleared to peak .8625 / final **.0125** on identical settings.
- Promotion is bimodal: cleared stages have a median time-to-clear of
  **50k steps** (one eval interval — the policy arrives already competent);
  stages not cleared almost immediately are usually never cleared.

### What the two 20-stage runs share (and their stalled siblings lack)

1. **Opponent diversity**: random_10/15/20 stages train against
   `mixed(easy=random, hard=random_harder, p_hard=0.5)` (v43a+ family)
   instead of a single opponent.
2. **Anti-draw reward geometry**: `draw: -50` (v52a: scaled by turns used),
   `win_speed_bonus: 0`, `enemy_owned_capture: 0`, `turn_penalty: -0.5/-1.0`.
3. `ent_coef` 0.10 → 0.01 linear (all 10 random_15 clears had ent_start
   0.10; the 0.025–0.05 configs populate the stall lists; v53c global 0.05
   was the worst run of the sweep).
4. Bigger random_10 budgets (5M).

Siblings with identical PPO knobs but more shaping or less entropy
(v43b/v44/v46/v51/v52b/v53c) still stalled with the classic
peak-then-collapse signature — the reward geometry and opponent mix are
what mattered, since nothing else varied.

---

## 2. Root causes in the code and config

### 2.1 The canonical config still contains the proven-guilty rewards

`docs/bootstrap_lessons_learned.md` (§"RESOLVED", v26/v27 table) shows the
bisection: zeroing {win_speed_bonus, enemy_owned_capture, turn_penalty}
clears the beginner block (v26); reintroducing `win_speed_bonus: 50` alone
(v27a) or `enemy_owned_capture: -15` alone (v27c) each independently
recreate the random_10/15 stall; `v28_production_reward_fixed.yaml` codifies
the fix. Yet `configs/ppo/bootstrap.yaml` — last touched *after* that
resolution, and the default of `scripts/train/train_bootstrap.py` — still
ships `win_speed_bonus: 50.0` (line 150), `enemy_neutral_capture: -8.0` /
`enemy_owned_capture: -15.0` (lines 163–164), `turn_penalty: 0.0` (line
142). **A fresh run of the canonical config today re-runs the configuration
the sweep already proved stalls.**

### 2.2 The draw attractor is a reward-geometry theorem, not bad luck

Per-step event income (combat shaping, captures, `seize_progress: 3`,
income/unit/structure potentials) scales with `max_turns`, while the draw
terminal is a fixed −50 and shaping is skipped on the terminal step
(`gym_env.py:1488–1496`). On a 75-turn beginner draw the farmable
non-terminal income (~+60–95/episode in stalled runs) exceeds |−50|, so a
"safe paying harbor" exists (v49's own header: draws collect +28..+55 —
"the draw is POSITIVELY rewarded"). Each map block re-opens the attractor
because `max_turns` grows (75 → 120 → 200) while terminals stay fixed;
v52a's turn-scaled draw penalty is the structural counter and should be the
default. Since draw == loss == −50 while `turn_penalty` is negative, a fast
loss can return-dominate a long draw *and* rival a slow win — the reward
ordering `win > draw > loss`, net of per-turn tax, is not actually enforced.

### 2.3 Promotion machinery: noisy gate, no floor, winner's curse, no retry

- `PromotionCallback` (`callbacks.py:309–344`) promotes on `patience`
  consecutive evals ≥ threshold; `min_timesteps_before_promotion` defaults
  to 0 (`config.py:254`) and is unset in bootstrap.yaml, and the first eval
  fires within one vec-step of stage entry (`callbacks.py:160–169` +
  `reset_num_timesteps=False`), so 117/233 clears happened in ≤2 evals.
  Most fast clears are genuine (86/117 at ~1.0 WR), but the gate has no
  guard against promoting a policy that then meets a cliff.
- 80-episode evals give ±10pp CIs at p≈0.7. Gates within ~5pp of the
  attainable ceiling (random_15's 0.70 vs 0.63 scripted ceiling) make
  promotion a luck lottery; `patience: 4` (v13/v15/v16) made even
  starter_random's 0.90 gate unreachable (v16 peaked .9125 and stalled).
- Between stages the runner restores the best-by-eval-WR checkpoint
  (`bootstrap.py:957–971`) — argmax over noisy evals, a winner's-curse
  selection. But on **stall** it raises `CurriculumStalled`
  (`bootstrap.py:908–941`) *without* using that same `best_model.zip` —
  28/41 stalled stages had a peak ≥ threshold sitting on disk, and the run
  just dies. The notebook then `runtime.unassign()`s (cell 40).
- Draws count as losses in `win_rate` (`evaluation.py:357`). On the
  meat-wall random stages, most non-wins are draws; gating on
  win-rate-excluding-draws is the strictest possible reading.

### 2.4 Horizon: γ=0.99 over micro-action episodes

One env step = one unit action; `end_turn` executes the entire opponent
turn inside the step (`gym_env.py:1280–1309`). Beginner episodes run
~700–1500 env steps (75 turns × ~20 steps/turn), skirmish up to ~2400.
Effective horizon 1/(1−γ) = 100 steps ≈ **5 game turns**; 0.99^800 ≈ 3e-4.
Long-horizon consequences (economy snowball, HQ escort, the value of *not*
drawing) are invisible to the return from early-game states; only dense
event rewards are learnable signal at this horizon. This is why reward
re-weighting (v50/v52a) worked at all, and why nothing else could: **γ,
gae_lambda, lr, n_steps, batch_size, clip_range, n_epochs, vf_coef, net
size are literally identical in all 59 configs** (only ent_coef and, once,
net_arch varied). γ≈0.997–0.999 with rescaled terminals, or per-turn
(rather than per-action) decision granularity, was never tried.

### 2.5 Session economics and the missing resume path

- Total curriculum worst-case budget: 87.5M steps. Observed max: 7.9M
  (median 2.1M). Sequential single-env evals eat ~30–50% of session
  wall-clock (80 episodes × every 50k steps × 33 stages).
- 15/56 usable runs ended on a just-cleared stage (13 of them ≤750k total
  steps — early Colab cuts). Both 20-stage runs died this way.
- `run_curriculum` (`bootstrap.py:637`) always starts at stage 0;
  `warm_start_path` only seeds the initial model. There is no
  resume-from-stage-K, so every session repays the full ladder from
  scratch — ~1–2M steps of already-solved stages before reaching the
  frontier.

### 2.6 Latent ceilings at the current frontier (skirmish_random_20+)

- **512-action truncation**: skirmish states were measured at 728–744 legal
  actions; `_build_flat_actions` truncates at `max_flat_actions=512` and the
  enumeration order (create → move → attack → …, `gym_env.py:82–93`) means
  **attack/heal/cast actions are silently dropped first** on big armies.
  An army large enough to win cannot attack.
- **Positional action semantics**: `Discrete(512)` index i = "the i-th entry
  of a per-step rebuilt list" (`gym_env.py:1461–1468`); the mapping shifts
  every step and depends on unobservable unit-insertion order
  (`game_state.py:1319, 1351`) — true observation aliasing. Policies still
  reached 1.0 WR vs medium/advanced on small maps, so it's a
  sample-efficiency tax today, but it scales badly with army size.
- **`masked_avg` pooling** (`extractors.py:189–195`) reduces the CNN map to
  64 dims (+5 globals → Linear(69, 256)) before the action head — precise
  spatial target selection has to squeeze through a global average. All
  sweep configs use it (`pool: flatten` is unused).
- **`max_steps: 3000` vs corner_points' `max_turns: 200`**: simulation shows
  army-heavy games accumulate 3000 agent steps by turn ~90–135, so the
  yaml's "truncation never fires" comment (lines 22–24, still citing
  "max_turns 20/60/120") is stale — corner_points games will end as −50
  truncation draws regardless of board state, and `win_speed_bonus` is
  calibrated to a 200-turn horizon the episode can't reach.
- **Gate-vs-ceiling violations ahead**: from-the-agent's-seat simulations
  (fresh, 30–40 games/matchup): `intermediate_medium` gate 0.65 vs best
  scripted 0.30; `skirmish_medium` gate 0.70 vs AdvancedBot exactly 0.70;
  bot tiers are non-monotone per map (AdvancedBot *loses* to MediumBot on
  intermediate, 0.30, but beats it on skirmish, 0.70). First-move advantage
  is enormous (MediumBot mirror: P1 wins 93–97%), and P2 gets an extra
  income tick (`game_state.py:1246`) — the agent always plays P1, so
  training silently leans on seat advantage.

### 2.7 Notebook and tooling defects

- **Run-All is currently broken**: cell "3c" ships
  `BUILD_BC_WARMSTART = True`, which raises
  `RuntimeError("BC warm-start requires action_space_type=multi_discrete…")`
  against the default `bootstrap.yaml` (flat_discrete) before training
  starts. (Run data agrees BC warm-start should stay off: v33 and
  skirmish_bc_selfplay scored 0.000 WR.) A successful BC build would also
  clobber any manually set `cfg.warm_start_path`.
- **Train diagnostics are never persisted**: `TrainingMetricsCallback`
  collects approx_kl / clip_fraction / explained_variance / value_loss
  (`callbacks.py:53–62`) into memory; `bootstrap.py` writes eval JSON/CSV
  but never the train records, and the notebook only plots them live — they
  vanish with the Colab VM. The collapse dynamic (KL blowup? value
  divergence? entropy collapse?) is therefore undiagnosable from archived
  runs, and 59 variants were designed blind to it.
- **`run_status.json` is written but never read**: the analysis notebook
  classifies stages purely from `bootstrap_results.csv` tails. (This does
  not corrupt the stall statistics — aborted-in-progress stages simply show
  `not_started` — but truncation-vs-stall must be inferred from
  `total_env_steps`, and the summary CSVs would misclassify a
  cleared-then-killed run as "ended cleared".)
- **No `Monitor` wrapper on training envs** → `rollout/ep_rew_mean`,
  `ep_len_mean` silently absent from all training charts.
- **Single seed everywhere**: all 59 configs use `seed: 42`, one run per
  variant. Where configs were rerun, outcomes flipped (bootstrap:
  4 cleared / 1 stalled on random_10; v43a: 1/1; v16: 1 cleared /
  2 stalled; steps-to-clear varies 3× for identical config+stage). Most
  single-knob sweep conclusions are within run-to-run noise.
- **`ppo_training.ipynb` has diverged**: pre-rescale ±5000 rewards, trains a
  fresh model, and the nominal "self-play from bootstrap checkpoint"
  handoff was never implemented — a successful bootstrap currently has no
  working downstream consumer.
- Mechanics oddities worth knowing: min-1-damage clamp grants phantom
  counter-attacks from units that can't legally attack back; wounded units
  parked on own structures auto-heal and silently spend agent gold.

---

## 3. Recommendations, in order of expected value

1. **Back-port the proven reward geometry into `bootstrap.yaml`** (it is the
   config every new run inherits): `win_speed_bonus: 0`,
   `enemy_owned_capture: 0`, `turn_penalty: -0.5..-1.0`, draw penalty scaled
   by turns used (v52a), keep `draw` ≤ `loss` net of turn tax so
   win > draw > loss ordering holds at the return level. Adopt the v43+
   mixed-random opponents for all random_N stages.
2. **Add stage resume + stall-retry.** Persist (stage index, model,
   optimizer, entropy-schedule position) every stage; let `run_curriculum`
   start at stage K; on `CurriculumStalled`, reload the stage's
   `best_model.zip` (it was above threshold in 28/41 stalls) and retry once
   with re-warmed entropy before giving up. This converts both observed
   death modes (wall-clock and stall) from run-fatal to recoverable. With
   87.5M steps of budget vs ~6M steps/session, no reward fix alone can
   finish the ladder.
3. **Fix the gates where they exceed the attainable ceiling.** Count draws
   separately (gate on win+draw or on Elo-style score) or raise `max_turns`
   /add reinforcement-breaking mechanics on the meat-wall random stages; as
   configured, `beginner_random_15/20`'s 0.70 gate is above every scripted
   bot's WR from the agent seat (best 0.63) and `intermediate_medium`'s
   0.65 is ~2× the scripted ceiling (0.30). Add
   `min_timesteps_before_promotion` (v31-style) to production, and gate on
   a Wilson lower bound instead of raw WR to kill the ±10pp lottery.
4. **Persist train diagnostics** (metrics_callback.records → CSV per stage,
   flushed with each eval) so the peak-then-collapse dynamic can finally be
   attributed (KL/clip/entropy/value). Cheap, and every future sweep
   variant becomes diagnosable.
5. **Attack the collapse from the optimizer side** — the only never-touched
   axis: lower/annealed LR late in a stage, tighter clip_range, larger
   batch, and γ 0.997–0.999 (or per-turn macro-actions) to bring terminals
   inside the horizon. Also stop annealing entropy over `max_timesteps`
   when stages promote at 50k — anneal over expected-steps-to-promote or
   keep a floor.
6. **Raise `max_flat_actions` (≥1024) for skirmish/corner_points and
   reorder enumeration so attacks survive truncation**; raise `max_steps`
   for corner_points (≥4500) or scale it with `max_turns`. Longer term,
   replace the positional flat head with a pointer/per-cell action head and
   `pool: flatten`, which removes both representation ceilings at once.
7. **Run ≥3 seeds for any conclusion you intend to keep**, and re-eval a
   fixed opponent battery (e.g. random_15 + simple + medium on the current
   map) at every promotion so retention/forgetting is measured rather than
   assumed.
8. Small hygiene: default `BUILD_BC_WARMSTART = False` (it currently breaks
   Run-All against flat_discrete configs), Monitor-wrap training envs, make
   the analysis notebook read `run_status.json`, refresh the stale
   `max_turns` comments in bootstrap.yaml, and either wire the
   `ppo_training.ipynb` self-play handoff or retire the notebook.

## Appendix: deepest-run trajectories

- **20260601_172412 / v52a** — 20/33, 5.85M steps, ended CLEARED
  (session cut): balanced_random 100k@1.0 → r10 850k (1.0/.99) → r15 350k
  (.96/.76) → mixed 50k@1.0 → r20 1.45M (.85/.79) → simple→advanced
  50–100k each @.85–1.0 → intermediate block cleared (r15 1.1M .98/.89,
  r20 850k .93/.91) → skirmish balanced/r10 50k@1.0 → skirmish_r15
  150k@1.0. **END (wall-clock).**
- **20260531_165459 / v50** — 20/33, 5.5M steps, same shape; r20 slowest
  (2.1M, .75/.75); ended CLEARED at skirmish_random_15 (1.0/.99).
- **20260527_150915 / v37b** — 15 cleared at .85–.90 thresholds, nearly all
  at 1.0 WR, then intermediate_random_20: peak .975 → final .0375,
  avg_turns 74.2/75, full budget burned. Single-opponent training with
  legacy shaping: the collapse was postponed, not fixed.
