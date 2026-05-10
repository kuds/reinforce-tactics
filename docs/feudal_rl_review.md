# Feudal RL Setup Review

## Overall Assessment
The architecture is in good shape: clean Manager/Worker split with a shared
encoder driven by a single optimizer (manager grads detach from the encoder),
segment-length-aware GAE, full mask plumbing for both the legacy 6-head and
the AlphaStar-style autoregressive worker, and a YAML/CLI surface that exposes
the full feature set of `FeudalRLAgent`.

This pass closes the gaps the previous review flagged. What follows is the
current state — what's fixed, what's still open, and the priority of the
remaining work.

---

## Fixed in this pass

| Issue | Where | Notes |
|---|---|---|
| Unused `state` arg in `compute_intrinsic_reward` | `feudal_rl.py:1599` | Signature now `(next_state, goal)`; `agent_player` removed (obs is agent-relative). |
| `_last_obs` not safely initialized | `feudal_rl.py:1219` | `collect_rollout` auto-resets the env on first call when `_last_obs is None`. |
| `evaluate` may reference unbound `info` | `feudal_rl.py:1605` | `info = {}` initialized before the `while not done` loop. |
| Worker advantage normalization with `n_worker <= 1` | `feudal_rl.py:1418` | Now guarded the same way `m_adv` was. |
| AR worker silent when env lacks `structured_action_masks` | `feudal_rl.py:1226` | One-shot `RuntimeWarning`; falls back to unmasked AR sampling explicitly. |
| Wasted per-dim mask capture in AR mode | `feudal_rl.py:1248` | `env_supports_masks` is now AND-ed with `not use_ar_masks`. |
| Checkpoints lost runtime config (`manager_horizon`, `agent_player`) | `feudal_rl.py:1525` | `save_checkpoint` writes a `hyperparams` dict; `load_checkpoint` restores `manager_horizon`/`agent_player` and refuses to load if grid dims mismatch. |
| Stale backward-compat branch in `load_checkpoint` | `feudal_rl.py` | Dead `optimizer` fallback removed. |
| `autoregressive_worker` unreachable from training script | `train_feudal_rl.py:401`, `configs/feudal_rl.yaml:30` | New `--autoregressive-worker` CLI flag and `feudal.autoregressive_worker` YAML key, plumbed into `FeudalRLAgent(...)`. |
| Missing seeding in feudal training path | `train_feudal_rl.py:198-205` | Seeds python `random`, numpy, torch, CUDA, and both env resets. |
| W&B logged-but-unused in feudal mode | `train_feudal_rl.py:255-265` | New `_log()` helper writes every metric to TensorBoard *and* W&B (when enabled). |
| `reward_breakdown` / `end_reasons` not surfaced | `train_feudal_rl.py:282-291` | Per-component reward sums and per-rollout end-reason counters logged each update. |
| `max_steps` hardcoded | `train_feudal_rl.py` | New `--max-steps` CLI flag, mirrored in YAML under `env.max_steps`. |

Tests added in `tests/test_feudal_rl_integration.py`:
- `test_checkpoint_restores_manager_horizon`
- `test_checkpoint_rejects_grid_dim_mismatch`
- `test_collect_rollout_auto_initializes_last_obs`
- `test_ar_worker_warns_without_structured_masks`
- Existing `test_feature_extractor_*` test renamed to reflect the
  worker-only ownership of the encoder.

The notebook `notebooks/feudal_rl_training.ipynb` is updated to match the new
APIs (no manual `_last_obs` priming, new `compute_intrinsic_reward` signature,
checkpoint-roundtrip validation now also asserts `manager_horizon` survived).

---

## Pass 3: system integration (Tier 1 #1 + Tier 2)

The remaining functionality gaps that blocked feudal from being a first-class
trainer in the project — tournament/GUI integration, self-play, multi-env
rollout, AR validation, and intrinsic-reward visibility — are now closed.

| Issue | Where | Notes |
|---|---|---|
| ModelBot couldn't load feudal `.pt` | `reinforcetactics/game/model_bot.py:51-174` | `_load_model` dispatches on `.zip` vs `.pt`. Feudal path reconstructs `FeudalRLAgent` from the checkpoint's `hyperparams` blob and refuses on grid-dim mismatch. `take_turn` routes through stage-conditional / per-dim masks built from the live `game_state`. |
| Tournament didn't see feudal checkpoints | `reinforcetactics/tournament/bots.py:389-416` | `discover_model_bots` globs `*.pt` alongside `*.zip`; `_test_model_file` accepts either `bot.model` or `bot.is_feudal`. Fixed a long-standing broken hardcoded `6x6_beginner.csv` path while there. |
| Mask-builder coupling to env | `reinforcetactics/rl/gym_env.py:60-203` | `build_per_dim_masks` and `build_structured_masks` factored into pure functions taking `(game_state, grid_w, grid_h, ...)`. Env methods now thin wrappers; ModelBot uses the same canonical layout. |
| No self-play for feudal | `gym_env.py:262-279` (env), `train_feudal_rl.py:248-322` (script) | `set_self_play_opponent_factory` lets the trainer rebind the env opponent on each reset. Trainer adds `--opponent self`, `--opponent-snapshot-freq`, `--opponent-pool-size`, `--eval-opponent`. Snapshots roll over a fixed-size pool; opponents are loaded as `ModelBot` instances. Eval keeps a fixed opponent so scores don't drift. |
| Single-env rollout was the throughput bottleneck | `feudal_rl.py:1438-1612` (`collect_rollout_vec`), `feudal_rl.py:1002-1062` (`merge_finalized_buffers`) | Vectorized rollout over N envs with per-env goal / segment state and a single batched feature-extractor forward per step. Per-env GAE; merged via concatenation post-finalize. Trainer uses it whenever `--n-envs > 1`. |
| AR worker had no validation harness | `scripts/ab_feudal_ar.py` | A/B harness trains legacy + AR variants from the same seed with identical hyperparameters. Prints a side-by-side eval table (win-rate, mean reward, goal-reached rate) and a final-step verdict. ROADMAP Phase 3.7. |
| Intrinsic-reward calibration was blind | `feudal_rl.py:837-902` (buffer fields), `train_feudal_rl.py:434-465` (logging) | Buffer now stores per-step `w_intrinsic`, `w_extrinsic`, `w_reached_goal`. Trainer logs `train/worker_intrinsic_mean`, `train/worker_extrinsic_mean`, `train/goal_reached_rate` — telling you whether the worker is being shaped by the manager or the env, and what fraction of steps actually achieve the goal. |

Tests added: feudal `.pt` loader (extension dispatch, AR vs legacy, grid-mismatch
guard, take_turn smoke), tournament discovery (`*.pt` pickup + bogus-checkpoint
reject), self-play factory hook (factory called per reset; safe no-op without
factory). 398 tests in the feudal-relevant suites pass.

---

## Pass 2: training-loop functionality

This pass closed the operational gaps that were blocking actual training
runs (value-loss explosion, no resume, brittle eval scheduling, no LR
schedule, no grad-norm visibility).

| Issue | Where | Notes |
|---|---|---|
| Value-loss explosion from raw reward magnitudes | `feudal_rl.py:1278`, `train_feudal_rl.py:401` | `collect_rollout` accepts `reward_scale`; new `--reward-scale` CLI flag (and `feudal.reward_scale` YAML key). With Direction-A `±5000` terminals, `reward_scale=0.001` keeps `vf_coef * value_loss` in the same ballpark as the policy/entropy terms. |
| Resume-from-checkpoint not wired | `train_feudal_rl.py:248-256`, `feudal_rl.py:1525,1583` | `save_checkpoint` accepts a `training_state` dict; `load_checkpoint` returns it. Training script's `--resume PATH` restores weights, optimizer state, `total_timesteps`, `best_eval_*`, and the eval/checkpoint high-watermarks so cadence picks up cleanly. |
| Brittle eval/checkpoint scheduling (`% eval_freq < n_steps`) | `train_feudal_rl.py:354,392` | Replaced with high-watermark gating (`total_timesteps - last_eval_step >= eval_freq`), plus a forced eval on the final update. Robust to any `n_steps` that doesn't divide `eval_freq` evenly. |
| No LR scheduling | `train_feudal_rl.py:268-289`, `configs/feudal_rl.yaml:24` | New `--lr-schedule {constant,linear}` with linear-anneal-to-zero across `total_timesteps`. Multiplier is logged as `train/lr_mult`. |
| No per-network gradient-norm visibility | `feudal_rl.py:1479,1503` | `update()` now returns `worker_grad_norm` and `manager_grad_norm` (the pre-clip total norms returned by `clip_grad_norm_`). Surfaced to TensorBoard / W&B and printed in the per-update progress line. |
| Best-checkpoint criterion was reward-only | `train_feudal_rl.py:367-385` | Best model now picked on `(win_rate, mean_reward)` tuple — wins are dominant in this env's reward shape; mean-reward is a tiebreak. |

Tests added in `tests/test_feudal_rl_integration.py`:
- `test_checkpoint_training_state_roundtrip`
- `test_checkpoint_without_training_state_returns_none`
- `test_reward_scale_shrinks_extrinsic_signal`
- `test_update_reports_gradient_norms`

The notebook now exposes `REWARD_SCALE = 0.001`, threads it through both
sanity-check and training rollouts, and the metrics figure plots both
gradient norms (log scale) and value losses (log scale) so the value-target
range is immediately visible.

---

## Still open

These are research / exploration items, not blockers. All system-integration
gaps from the prior pass are closed.

### Functionality

1. **Run the AR A/B at scale.** `scripts/ab_feudal_ar.py` is wired up but
   nobody has produced a verdict yet. Same goes for an
   `--worker-reward-alpha` sweep using the new `train/goal_reached_rate`
   diagnostic to find the intrinsic-vs-extrinsic sweet spot.
2. **Subprocess vec envs.** The current vectorized rollout runs envs
   sequentially in-process. For env-bound runs a `SubprocVecEnv`-style
   wrapper would buy real wall-clock speedup; the per-env state inside
   `FeudalRLAgent` already supports it.
3. **No goal-coverage exploration bonus.** Manager exploration is entropy-
   only; nothing rewards covering the goal space.
4. **No curriculum on `manager_horizon`.** Fixed throughout training;
   shorter early, longer later is plausible.
5. **No early stopping on plateau.** Would save compute on long runs.

### Quality-of-life

1. **No goal-distribution heatmap.** Goal-type histograms reach TensorBoard,
   but spatial coverage (which `(x, y)` cells the manager picks) is not
   visualized.
2. **No `from_config` / `to_config` on `FeudalRLAgent`.** Checkpoints now
   carry the runtime subset, but a full config serializer would simplify
   hyperparameter sweeps.

---

## Priority Summary

| Priority | Issue | Type |
|----------|-------|------|
| Medium | Run AR A/B + intrinsic-reward sweep | Validation |
| Low | Subprocess vec envs | Perf |
| Low | Goal-coverage exploration / heatmap | Gap / QoL |
| Low | Curriculum on manager_horizon | Gap |
| Low | Early stopping | Gap |
| Low | `from_config` / `to_config` serializer | QoL |
