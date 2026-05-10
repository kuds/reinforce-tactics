# Bootstrap Runs Review — 2026-05-08 → 2026-05-10

Findings from the last four completed `ppo_bootstrap` runs in
`benchmarks/bootstrap/`. All four ran the full 6-stage curriculum on
`maps/1v1/beginner.csv` against the random opponent, with identical core PPO
hyperparameters and the same seed (42). The intentional changes between them
were the per-step combat reward shaping (runs 1 → 2/3) and the structure-capture
reward shaping (runs 3 → 4). See `docs/bootstrap_lessons_learned.md` for the
broader context this builds on.

## Runs covered

| # | Run dir | git | Reward-shape change | Final-stage best win rate |
| --- | --- | --- | --- | --- |
| 1 | `20260508_222916` | `2dbb19a` | `damage_scale: 0.5`, `kill: 50.0` | **76.7%** ✅ promoted |
| 2 | `20260509_024930` | `ef2d8a3` | combat shaping removed (`damage_scale: 0`, `kill: 0`) | 66.7% ❌ stalled |
| 3 | `20260509_153111` | `5abbcbf` | combat shaping removed (repeat) | 71.7% ❌ stalled |
| 4 | `20260509_201218` | `8f821b5` | added `tower_capture: 1500`, `building_capture: 4000`, `hq_capture: 2500` | 61.7% ❌ stalled |

"Final stage" = `beginner_random_20` (random opponent with `max_actions=20`,
the hardest of the six bootstrap stages). Promotion threshold is 75% with
`patience: 2`. Hardware was identical in all four runs (NVIDIA L4, sb3 2.8.0,
torch 2.10).

## Shared settings

These were unchanged across all four runs:

- **PPO**: lr `3e-4`, `n_steps 2048`, `batch_size 64`, `n_epochs 10`,
  γ `0.99`, λ `0.95`, clip `0.2`, `vf_coef 0.5`, `max_grad_norm 0.5`.
- **Network**: separate pi/vf MLPs `[256, 256]`.
- **Entropy**: linear schedule `0.10 → 0.03` per stage.
- **Action masking**: enabled.
- **Curriculum control**: 6 stages, `max_timesteps: 2_000_000` per stage
  (since bumped to 3M for `beginner_random_20` in commit `d3a272b`),
  `n_envs: 4`, `eval_freq: 50_000`, `n_eval_episodes: 60`,
  `promotion_win_rate: 0.75`, `patience: 2`.
- **Beginner opponent**: `random` with `opponent_kwargs.max_actions: 20`
  (vs the much weaker default-action random used in the starter stages).
- **Reward (always present)**: `win 5000`, `loss -5000`, `draw -5000`,
  `win_by_hq_capture 3000`, `win_by_elimination 3000`, `capture 2000`,
  `seize_progress 50`, `structure_control 10`, `unit_diff 1.0`,
  `income_diff 0.5`, `invalid_action -10`, `turn_penalty -20`.

## Per-stage trajectories (consistent across all four runs)

The earlier curriculum stages converge cleanly and look the same in every
run. From the consolidated `bootstrap_results.csv` of run 4 (which uses the
same curriculum as runs 2–3):

| Stage | Opponent | Cleared at | Final win rate |
| --- | --- | --- | --- |
| `starter_random` | random | ~50–100k | 88–93% |
| `starter_simple` | simple | first eval (100k) | 100% |
| `starter_medium` | medium | 200k (after 0% at 100k & 150k) | 100% |
| `beginner_balanced_random` | balanced_random | 200k | 100% |
| `beginner_random_10` | random (max_actions=10) | 250–300k | 97–98% |
| `beginner_random_20` | random (max_actions=20) | (see below) | varies |

Two trajectory features show up in every run:

- **`starter_medium` is a step function.** Win rate stays at 0.0 with
  episodes hitting the time-limit (avg reward ≈ -5000) for the first 150k
  steps, then jumps to 1.0 at 200k. The agent has to discover HQ capture
  before it can win at all against `medium`; nothing in the dense reward
  gradually approaches the solution.
- **The first eval of every stage is at `t = stage_steps + 4`** (a 4-step
  warm-eval from the curriculum runner). For `starter_simple` and
  `beginner_balanced_random` the agent already wins immediately on stage
  entry, indicating positive transfer between stages.

## What's different on `beginner_random_20`

This is where the four runs diverge.

### Run 1 — `20260508_222916` (with combat shaping)

`damage_scale: 0.5`, `kill: 50.0`. `eval_curves.png` shows the avg-reward
y-axis topping out around **15,000** — the dense combat terms add real
magnitude.

- Best win rate **76.7%** → cleared the 75% bar.
- `combat_summary` shows **captures and kills both rising over training**
  (~40–60 captures and 40+ kills per game by the end, with damage dealt
  significantly exceeding damage taken — net dealt is positive).
- `outcome_breakdown` includes a `losses_by_hq_capture` slice — the random
  opponent occasionally captures the agent's HQ.

### Run 2 — `20260509_024930` (combat shaping removed)

`damage_scale: 0.0`, `kill: 0.0`. Avg-reward axis tops out around
**10,000** (consistent with the dense combat terms being zero).

- Best win rate **66.7%** — well below the 75% bar; ran out of budget.
- `outcome_breakdown` still includes `losses_by_hq_capture`, so the agent
  is still vulnerable to HQ rushes.
- avg-reward curve is *lower-magnitude and noisier* than run 1 — the
  policy has less dense gradient to follow.

### Run 3 — `20260509_153111` (combat shaping removed, repeat)

Same reward config as run 2, same seed.

- Best win rate **71.7%** — closer to threshold than run 2 but still
  below.
- `outcome_breakdown` legend **omits `losses_by_hq_capture`** — across
  this run's evals the agent never lost by HQ capture, only by
  elimination/draws. Defense is solid; the gap is in finishing wins.
- ~5pp better than run 2 with no config differences highlights how noisy
  60-episode evaluation is at this performance level.

### Run 4 — `20260509_201218` (structure-capture rewards added)

Adds three new reward terms to the no-combat baseline of runs 2–3:
`tower_capture: 1500`, `building_capture: 4000`, `hq_capture: 2500`. Combat
shaping (`damage_scale`, `kill`) is still zero. The training run continued
~300k beyond the 2M `max_timesteps` cap (the runner finished the in-flight
collection cycle), giving 41 evals from 300k → 2.3M.

- Best win rate **61.7%** at step 2.2M — worst final number of the four
  runs *but the trajectory was still climbing* when the budget ran out:

  | Step range | Mean win rate | Max in window |
  | --- | --- | --- |
  | 0.3–0.5M | 0.343 | 0.400 |
  | 0.5–0.75M | 0.300 | 0.317 |
  | 0.75–1.0M | 0.350 | 0.383 |
  | 1.0–1.25M | 0.373 | 0.450 |
  | 1.25–1.5M | 0.420 | 0.467 |
  | 1.5–1.75M | 0.450 | 0.517 |
  | 1.75–2.0M | 0.527 | 0.600 |
  | 2.0–2.3M | 0.537 | 0.617 |

- **Defense fully solved by ~1.1M.** Across the first ~1M of the stage,
  the agent loses 5–10 games per 60-episode eval (mostly by elimination).
  Starting at step **1.1M, every subsequent eval has *zero* losses** —
  24 consecutive evals through 2.3M with no losses. The post-1.1M policy
  is uncrackable by random_20 but converts wins slowly.
- **Real bottleneck = closing.** Aggregate over the whole stage:
  1026 wins / 96 losses / **1338 draws** (54% of episodes are draws). The
  agent ties most games, doesn't lose them, but only sometimes captures.
- **Avg episode length ~1100–1300 steps** (close to but below the 1500
  cap) and stays there throughout — games are running long, racking up
  `turn_penalty: -20` per turn.
- **`approx_kl` axis on `eval_curves.png` reaches ~0.14** (vs ≤0.05 in
  runs 1–3). PPO updates are markedly larger this run — likely a
  side-effect of the new `building_capture: 4000` reward dwarfing
  pre-existing dense terms and producing higher-variance advantages.
- **`value_loss` axis spans 10⁵–10⁶**, also higher than earlier runs,
  consistent with the larger reward magnitudes.
- `combat_summary` shows captures and attacks rising; `outcome_breakdown`
  shows the loss categories collapsing to zero in the second half of the
  stage with draws absorbing the deficit.

## Trends and weaknesses

1. **Removing combat shaping cost ~5–10 percentage points on the hardest
   stage.** This is still the single largest signal across the four runs.
   Run 1 (with `damage_scale: 0.5`, `kill: 50`) was the only one to clear
   75%. Replacing combat shaping with structure-capture shaping (run 4)
   did not recover the gap.
2. **The bottleneck on `beginner_random_20` shifted from defense to
   offense.** Earlier runs (1–3) had occasional `losses_by_hq_capture` or
   `losses_by_elimination`. Run 4 *eliminates losses entirely* after
   ~1.1M steps but doesn't replace them with wins — they become draws.
   The new structure-capture rewards likely teach defensive map control
   without creating enough gradient to push for the kill.
3. **`max_timesteps: 2M` was under-budget.** Run 4's win-rate curve was
   monotonically improving through the cap (0.34 → 0.54 → 0.62 by stage
   end). The 3M bump in commit `d3a272b` is justified — extrapolating
   linearly, run 4-style policy could plausibly reach 70%+ at 3M.
4. **Draws are punished as harshly as losses (-5000).** Combined with
   `turn_penalty: -20` over `max_turns: 100`, the agent is strongly
   discouraged from any cautious "stall and stabilize" play, but ironically
   that is exactly what run 4's policy converged to (54% draw rate, no
   losses). The reward shape is fighting the actual learned policy.
5. **Eval noise is non-trivial near threshold.** The 5pp gap between runs
   2 and 3 (identical configs) at 60 eval episodes is ~1 standard error
   (`sqrt(0.7*0.3/60) ≈ 5.9%`). With `patience: 2` you need two
   consecutive crossings of 75%, so a true win rate of ~74% will fail
   promotion most of the time even if the policy is genuinely close.
6. **Entropy floor of 0.03 is high for a converged policy.** The
   schedule ends at 0.03 regardless of how long the stage runs. With
   `max_timesteps` now 3M, the agent is still being told to inject
   meaningful exploration noise at step 3M — exactly when it should be
   committing.
7. **Reward magnitudes are unbalanced and may be inducing high `approx_kl`.**
   `building_capture: 4000` is the same order of magnitude as `win: 5000`
   but fires much more often, dwarfing the `unit_diff: 1.0` /
   `income_diff: 0.5` dense terms by 4 orders. Run 4's `approx_kl` peaked
   ~0.14 (vs ≤0.05 in runs 1–3), suggesting the policy is getting yanked
   around by these large dense rewards.

## Behaviors observed in each run

- **All four runs**: same `starter_medium` step-function (0% → 100% at
  200k); positive transfer into `starter_simple` and
  `beginner_balanced_random` (1.0 win rate at first eval); fast
  convergence through `beginner_random_10` (≥97% within 50–100k of stage
  entry).
- **Run 1 (with combat shaping)**: rising captures *and* rising kills on
  `beginner_random_20`; net damage positive; closes out games with
  combat. Cleared promotion at 76.7%.
- **Run 2 (no shaping)**: stalled at 66.7%, still losing some games to
  HQ capture, lower-magnitude reward curve.
- **Run 3 (no shaping)**: stalled at 71.7%, no losses by HQ capture in
  the eval slice — the agent learned defense without learning to close.
- **Run 4 (structure-capture rewards)**: stalled at 61.7% but still
  climbing. Policy splits clearly into a pre-1.1M "still loses sometimes"
  regime and a post-1.1M "never loses, ties most games" regime. Highest
  `approx_kl` of any run (~0.14). Run was budget-limited, not
  performance-limited.

## Recommendations

In rough order of expected impact:

1. **Re-enable combat shaping at lower magnitude alongside the
   structure-capture rewards.** Try `damage_scale: 0.2`, `kill: 20.0`
   layered on top of run 4's reward config. The data argues that
   combat shaping is what creates the gradient to *finish* games, while
   the new capture rewards are what create the gradient to *control*
   the map. Run 1 had only the former and won; run 4 had only the
   latter and ties.
2. **Already done**: `beginner_random_20.max_timesteps` bumped from 2M
   to 3M (commit `d3a272b`). Run 4's monotonically-climbing trajectory
   makes this clearly the right call.
3. **Lower the entropy floor on `beginner_random_20`** to `end: 0.01`
   so the policy can sharpen during the last ~1M of the new 3M budget.
4. **Soften the draw penalty.** Make `draw` less negative than `loss`
   (e.g. `draw: -1000`) — run 4's converged-but-tying policy is being
   pushed away from the only stable strategy it learned.
5. **Consider lowering the learning rate or clip range on this stage
   only** if the next run also shows `approx_kl > 0.1`. The new
   capture rewards produce larger advantages and may need smaller
   updates to stay stable.
6. **Consider bumping `n_eval_episodes` to 100** *only* if a future
   run sits in the 73–77% band — at 100 episodes the standard error
   drops from ~5.9% to ~4.3%, materially reducing false-negative
   promotion checks. Don't change it preemptively.
7. **Vary the seed once** (e.g. seed=43, 44) to confirm the
   reward-shaping regression is structural, not a single unlucky draw.
