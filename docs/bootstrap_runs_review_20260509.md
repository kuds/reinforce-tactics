# Bootstrap Runs Review — 2026-05-08 → 2026-05-09

Findings from the last three completed `ppo_bootstrap` runs in
`benchmarks/bootstrap/`. All three ran the full 6-stage curriculum on
`maps/1v1/beginner.csv` against the random opponent, with identical PPO
hyperparameters and the same seed (42); the only intentional change between
them was the per-step combat reward shaping. See
`docs/bootstrap_lessons_learned.md` for the broader context this builds on.

## Runs covered

| Run dir | git | Reward shape | Final-stage best win rate |
| --- | --- | --- | --- |
| `20260508_222916` | `2dbb19a` | `damage_scale: 0.5`, `kill: 50.0` | **76.7%** ✅ promoted |
| `20260509_024930` | `ef2d8a3` | `damage_scale: 0.0`, `kill: 0.0` | 66.7% ❌ stalled |
| `20260509_153111` | `5abbcbf` | `damage_scale: 0.0`, `kill: 0.0` | 71.7% ❌ stalled |

"Final stage" = `beginner_random_20` (random opponent with `max_actions=20`,
the hardest of the six bootstrap stages). Promotion threshold is 75% with
`patience: 2`. Hardware was identical in all three runs (NVIDIA L4, sb3 2.8.0,
torch 2.10).

## Shared settings

These were unchanged across the three runs:

- **PPO**: lr `3e-4`, `n_steps 2048`, `batch_size 64`, `n_epochs 10`,
  γ `0.99`, λ `0.95`, clip `0.2`, `vf_coef 0.5`, `max_grad_norm 0.5`.
- **Network**: separate pi/vf MLPs `[256, 256]`.
- **Entropy**: linear schedule `0.10 → 0.03` per stage.
- **Action masking**: enabled.
- **Curriculum control**: 6 stages, `max_timesteps: 2_000_000` per stage,
  `n_envs: 4`, `eval_freq: 50_000`, `n_eval_episodes: 60`,
  `promotion_win_rate: 0.75`, `patience: 2`.
- **Beginner opponent**: `random` with `opponent_kwargs.max_actions: 20`
  (vs the much weaker default-action random used in the starter stages).
- **Reward (shared)**: `win 5000`, `loss -5000`, `draw -5000`,
  `win_by_hq_capture 3000`, `win_by_elimination 3000`, `capture 2000`,
  `seize_progress 50`, `structure_control 10`, `unit_diff 1.0`,
  `income_diff 0.5`, `invalid_action -10`, `turn_penalty -20`.

## Per-stage trajectories (consistent across runs)

The earlier curriculum stages converge cleanly and look the same in all three
runs. From the `bootstrap_results.csv` of the in-progress run (which uses the
same curriculum and shape as runs 1 and 2):

| Stage | Opponent | Cleared at | Final win rate |
| --- | --- | --- | --- |
| `starter_random` | random | ~50–100k | 88–93% |
| `starter_simple` | simple | first eval (100k) | 100% |
| `starter_medium` | medium | 200k (after 0% at 100k & 150k) | 100% |
| `beginner_balanced_random` | balanced_random | 200k | 100% |
| `beginner_random_10` | random (max_actions=10) | 250–300k | 97–98% |
| `beginner_random_20` | random (max_actions=20) | (see below) | varies |

Two trajectory features show up in every run:

- **`starter_medium` is a step function.** Win rate stays at 0.0 with episodes
  hitting the time-limit (avg reward ≈ -5000) for the first 150k steps, then
  jumps to 1.0 at 200k. The agent has to discover HQ capture before it can
  win at all against `medium`; nothing in the dense reward gradually
  approaches the solution.
- **The first eval of every stage is at `t = stage_steps + 4`** (a 4-step
  warm-eval from the curriculum runner). For `starter_random` this is
  ~5000 (-loss); for `starter_simple` and `beginner_balanced_random` the
  agent already wins immediately on stage entry, indicating positive
  transfer between stages.

## What's different on `beginner_random_20`

This is where the three runs diverge.

### Run 3 — `20260508_222916` (with combat shaping)

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

`damage_scale: 0.0`, `kill: 0.0`. Avg-reward axis tops out around **10,000**
(consistent with the dense combat terms being zero).

- Best win rate **66.7%** — well below the 75% bar; ran out of budget.
- `outcome_breakdown` still includes `losses_by_hq_capture`, so the agent is
  still vulnerable to HQ rushes.
- avg-reward curve is *lower-magnitude and noisier* than run 3 — the policy
  has less dense gradient to follow.

### Run 1 — `20260509_153111` (combat shaping removed, repeat)

Same reward config as run 2, same seed.

- Best win rate **71.7%** — closer to threshold than run 2 but still below.
- `outcome_breakdown` legend **omits `losses_by_hq_capture`** — across this
  run's evals the agent never lost by HQ capture, only by elimination/draws.
  Defense is solid; the gap is in finishing wins.
- ~5pp better than run 2 with no config differences highlights how noisy
  60-episode evaluation is at this performance level.

## Trends and weaknesses

1. **Removing combat shaping cost ~5–10 percentage points on the hardest
   stage.** This is the single largest signal across the three runs. With
   no `damage_scale` or `kill` reward, the only positive dense terms left
   are `seize_progress` (50) and `structure_control` (10), both of which
   reward ignoring the opponent and rushing capture. Against
   `random` with 20 actions/turn that strategy is too brittle — the
   opponent has enough actions to interrupt the rush often enough to
   knock win rate below 75%.
2. **Draws are punished as harshly as losses (-5000).** Combined with
   `turn_penalty: -20` over `max_turns: 100`, the agent is discouraged from
   any cautious "stall and stabilize" play, which is exactly what a
   variance-reduction strategy against a random opponent would look like.
   On the easier stages this doesn't matter; on `beginner_random_20` it
   may be locking out a viable policy class.
3. **Eval noise is non-trivial near threshold.** The 5pp gap between runs 1
   and 2 (identical configs) at 60 eval episodes is ~1 standard error
   (`sqrt(0.7*0.3/60) ≈ 5.9%`). With `patience: 2` you need two consecutive
   crossings of 75%, so a true win rate of ~74% will fail promotion most of
   the time even if the policy is genuinely close.
4. **Entropy floor of 0.03 is high for a converged policy.** The schedule
   ends at 0.03 regardless of how long the stage runs. With
   `max_timesteps` now 3M, the agent is still being told to inject
   meaningful exploration noise at step 3M — exactly when it should be
   committing.
5. **Reward magnitudes are an order of magnitude apart between sparse and
   dense terms.** `win 5000` / `loss -5000` / `capture 2000` dominate
   anything `unit_diff: 1.0` or `income_diff: 0.5` produce; the dense
   economy signal is essentially noise next to outcome rewards. Whether
   this matters depends on stage — for `starter_medium` it definitely does
   (the agent has nothing to gradient-descend toward HQ capture except the
   sparse reward), and for `beginner_random_20` it leaves the agent with
   no shaping for the actual battle.

## Behaviors observed in each run

- **All three runs**: same `starter_medium` step-function (0% → 100% at
  200k); positive transfer into `starter_simple` and
  `beginner_balanced_random` (1.0 win rate at first eval); fast convergence
  through `beginner_random_10` (≥97% within 50–100k of stage entry).
- **Run 3 (with combat shaping)**: rising captures *and* rising kills on
  `beginner_random_20`; net damage positive; closes out games with
  combat. Cleared promotion.
- **Run 2 (no shaping)**: stalled at 66.7%, still losing some games to
  HQ capture, lower-magnitude reward curve.
- **Run 1 (no shaping)**: stalled at 71.7%, but no losses by HQ capture in
  the eval slice — the agent learned defense without learning to close.

## Recommendations

In rough order of expected impact:

1. **Re-enable combat shaping at lower magnitude** (start with
   `damage_scale: 0.2`, `kill: 20.0`). This is the single change with the
   most evidence behind it. Run 3 showed it was sufficient to clear
   `beginner_random_20`.
2. **Already done**: `beginner_random_20.max_timesteps` bumped from 2M to
   3M (commit `d3a272b`). Both stalled runs may have been close to the
   threshold but out of budget.
3. **Lower the entropy floor on `beginner_random_20`** to `end: 0.01` so
   the policy can sharpen during the last ~1M of the new 3M budget.
4. **Soften the draw penalty.** Make `draw` less negative than `loss`
   (e.g. `draw: -1000`) so the agent isn't forced to gamble for wins on
   the hardest stage.
5. **Consider bumping `n_eval_episodes` to 100** *only* if a future run
   sits in the 73–77% band — at 100 episodes the standard error drops
   from ~5.9% to ~4.3%, which materially reduces false-negative
   promotion checks. Don't change it preemptively (it's a 1.7× eval cost
   for no benefit if the policy is genuinely below threshold).
6. **Vary the seed once** (e.g. seed=43, 44) to confirm the
   reward-shaping regression is structural, not a single unlucky draw.
