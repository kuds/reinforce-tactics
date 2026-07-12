"""
Reusable SB3 callbacks for RL training.

- ``TrainingMetricsCallback`` captures per-rollout PPO metrics
  (rollout/* and train/*) into an in-memory list, working around SB3's
  Logger.dump() clearing values between rollouts.
- ``PeriodicEvalCallback`` runs ``evaluate_model`` every ``eval_freq``
  env steps, mirroring SB3's ``EvalCallback`` contract while capturing
  the project's full win/loss/draw breakdown plus optional per-step
  action-type counts and reward-component sums.
- ``PromotionCallback`` watches a paired ``PeriodicEvalCallback`` and
  exits ``model.learn()`` early once a configurable win-rate threshold
  is sustained for ``patience`` consecutive evaluations. Used by the
  bootstrap-curriculum runner to advance between stages.
- ``EntropyScheduleCallback`` mutates ``model.ent_coef`` over the
  course of a stage so exploration noise can be cooled as the policy
  approaches its win-rate threshold. SB3 reads ``ent_coef`` fresh in
  every ``train()`` step so live mutation works without rebuilding.

Both callbacks are designed to work with ``MaskablePPO`` from sb3-contrib
as well as plain ``PPO`` from stable-baselines3 — they don't import
sb3-contrib, so the module remains usable in non-masked training too.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

from reinforcetactics.rl.evaluation import evaluate_model


class TrainingMetricsCallback(BaseCallback):
    """Capture per-rollout PPO training metrics during ``model.learn()``.

    SB3's ``Logger.dump()`` clears ``name_to_value`` between rollouts, so
    rollout/* values are gone by the time the next callback hook fires. To
    work around this we capture rollout/* directly from
    ``self.model.ep_info_buffer`` at ``_on_rollout_end`` (the value source
    the logger itself uses) and read train/* from the logger at the *next*
    ``_on_rollout_start`` — by which point ``train()`` has run for the
    previous iteration and populated those keys. ``_on_training_end``
    picks up the final iteration that has no follow-up rollout.

    The accumulated records persist across multiple ``model.learn()``
    invocations, so multi-stage training produces one continuous timeline.

    When ``csv_path`` is provided, every committed record is also appended
    to that CSV immediately (header written on first row). In-memory
    ``records`` vanish with the process — on Colab exactly the deep runs
    that die to disconnects lose their optimization diagnostics, leaving
    stall collapses unattributable (KL blowup vs value divergence vs
    entropy collapse). Append-mode writes survive the kill and keep one
    continuous file across stages. ``context`` (when set by the caller,
    e.g. the curriculum runner setting the stage name before each
    ``learn()``) is stamped onto each record as ``stage``.
    """

    TRAIN_KEYS = (
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/learning_rate",
        "train/loss",
        "train/policy_gradient_loss",
        "train/value_loss",
    )

    # Fixed CSV schema: identity/context first, then rollout/* and train/*.
    # ``train/ent_coef`` is read live off the model (schedules mutate the
    # attribute) because the logger copy is cleared by dump() before the
    # commit hook runs.
    _CSV_COLUMNS = (
        "timesteps",
        "stage",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        *TRAIN_KEYS,
        "train/ent_coef",
    )

    def __init__(self, csv_path: str | Path | None = None) -> None:
        super().__init__()
        self.records: list[dict] = []
        self._pending: dict | None = None
        self.csv_path = Path(csv_path) if csv_path is not None else None
        # Free-form context label (typically the curriculum stage name);
        # stamped onto records committed while it is set.
        self.context: str | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        snapshot: dict = {"timesteps": self.num_timesteps}
        if self.context is not None:
            snapshot["stage"] = self.context
        ep_buffer = getattr(self.model, "ep_info_buffer", None)
        if ep_buffer:
            rewards = [ep["r"] for ep in ep_buffer if "r" in ep]
            lengths = [ep["l"] for ep in ep_buffer if "l" in ep]
            if rewards:
                snapshot["rollout/ep_rew_mean"] = float(safe_mean(rewards))
            if lengths:
                snapshot["rollout/ep_len_mean"] = float(safe_mean(lengths))
        self._pending = snapshot

    def _commit_pending(self) -> None:
        if self._pending is None:
            return
        for key in self.TRAIN_KEYS:
            val = self.model.logger.name_to_value.get(key)
            if val is not None:
                self._pending[key] = float(val)
        ent_coef = getattr(self.model, "ent_coef", None)
        if isinstance(ent_coef, (int, float)):
            self._pending["train/ent_coef"] = float(ent_coef)
        # Only emit records that contain at least one metric beyond
        # timesteps and the context label.
        min_len = 1 + ("stage" in self._pending) + ("train/ent_coef" in self._pending)
        if len(self._pending) > min_len:
            self.records.append(self._pending)
            self._append_csv_row(self._pending)
        self._pending = None

    def _append_csv_row(self, record: dict) -> None:
        """Best-effort incremental persistence; must never break training."""
        if self.csv_path is None:
            return
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self._CSV_COLUMNS))
                if write_header:
                    writer.writeheader()
                writer.writerow({col: record.get(col, "") for col in self._CSV_COLUMNS})
        except Exception:  # noqa: BLE001
            pass

    def _on_rollout_start(self) -> None:
        # train() of the previous iteration has run by this hook, so
        # train/* values are now populated in the logger.
        self._commit_pending()

    def _on_training_end(self) -> None:
        self._commit_pending()


class PeriodicEvalCallback(BaseCallback):
    """Run ``evaluate_model`` every ``eval_freq`` env steps.

    Mirrors SB3's ``EvalCallback`` contract — a single ``model.learn()``
    call drives evaluation at a fixed cadence — but captures the full
    win / loss / draw breakdown via the project's ``evaluate_model``
    helper (SB3's built-in callback only logs mean reward and episode
    length). When ``track_breakdown=True`` the callback also passes
    through to ``evaluate_model``'s breakdown mode, so each entry in
    ``self.results`` carries ``action_counts`` and ``reward_components``
    suitable for stacked-area "what is the agent doing over time" plots.

    The callback gates on ``num_timesteps`` (total env steps across all
    sub-envs) so the cadence is independent of ``n_envs``: ``eval_freq=
    100_000`` fires every 100,000 env steps regardless of how many
    parallel envs are rolling. Best model (by win rate, with avg reward
    as tiebreaker) is saved to ``save_dir/best_model.zip`` when
    ``save_dir`` is provided.

    When ``results_jsonl_path`` is provided, each eval result is also
    appended to that file as one JSON line the moment it is produced.
    ``self.results`` lives only in memory until the *stage* finishes
    (the curriculum runner writes ``eval_results.json`` at stage end),
    so a run killed mid-stage — the normal Colab death — used to leave
    zero on-disk evidence of everything the in-progress stage measured.
    The JSONL sibling closes that gap without changing the end-of-stage
    JSON contract.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int = 30,
        eval_seed_base: int = 0,
        save_dir: Any = None,
        track_breakdown: bool = True,
        trace_dir: Any = None,
        results_jsonl_path: Any = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.track_breakdown = bool(track_breakdown)
        # ``trace_dir`` (when set) is the root under which per-eval-block
        # subdirectories are created -- each block's stall-episode JSONL
        # files land in ``trace_dir/eval_<timesteps>/``. Forwarded to
        # ``evaluate_model``; only ``max_steps_truncate`` episodes are
        # dumped, so healthy evals leave no artefacts on disk.
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.results_jsonl_path = Path(results_jsonl_path) if results_jsonl_path is not None else None

        self.results: list[dict] = []
        self.best_win_rate: float = -1.0
        self._best_reward: float = float("-inf")
        # Cumulative ``num_timesteps`` at which the current best_model.zip was
        # saved. -1 until a best is recorded. Exposed so the curriculum runner
        # can report how far *into the stage* the saved peak actually was --
        # a peak at the stage's first eval means the carry-in policy was
        # already strong and the stage did ~0 stage-specific learning (the
        # "skip-ahead" handoff failure documented in bootstrap_lessons_learned).
        self.best_timestep: int = -1
        self._last_eval_block: int = -1

    def _on_step(self) -> bool:
        # Trigger when num_timesteps crosses an eval_freq boundary. Using
        # block index (not modulo) avoids missing/double-firing when
        # num_timesteps jumps by n_envs > 1 each step.
        block = self.num_timesteps // self.eval_freq
        if block > self._last_eval_block:
            self._last_eval_block = block
            self._do_eval()
        return True

    def _do_eval(self) -> None:
        eval_seed = self.eval_seed_base + 1000 * self._last_eval_block
        eval_kwargs: dict[str, Any] = {}
        if self.trace_dir is not None:
            # One subdir per eval block, named by the timestep at which
            # the block fires, so traces from different evals don't
            # collide and a stalled episode is easy to map back to the
            # eval row in the printed log.
            eval_kwargs["trace_dir"] = self.trace_dir / f"eval_{int(self.num_timesteps):09d}"
        m = evaluate_model(
            self.model,
            self.eval_env,
            n_episodes=self.n_eval_episodes,
            seed=eval_seed,
            track_breakdown=self.track_breakdown,
            **eval_kwargs,
        )
        m["timesteps"] = int(self.num_timesteps)
        m["eval_seed"] = eval_seed
        self.results.append(m)

        # Incremental persistence: append the row now so a mid-stage kill
        # (Colab disconnect / OOM) doesn't erase every eval this stage ran.
        # Best-effort — a disk hiccup must not abort training.
        if self.results_jsonl_path is not None:
            try:
                self.results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.results_jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(m, default=float) + "\n")
            except Exception:  # noqa: BLE001
                pass

        # Tensorboard: log the most useful scalars so they show up alongside
        # the SB3-internal train/* and rollout/* curves.
        self.logger.record("eval/win_rate", m["win_rate"])
        self.logger.record("eval/mean_reward", m["avg_reward"])
        self.logger.record("eval/mean_ep_length", m["avg_length"])
        self.logger.record("eval/mean_ep_turns", m["avg_turns"])
        # Action-space diagnostics. ``seize_available_rate`` is the fraction
        # of decision points where a capture was legal -- a low value means
        # the policy rarely even reaches a capturable tile (navigation
        # bottleneck) vs. reaches one but declines (reward/exploration).
        # ``max_legal_actions`` guards the flat_discrete truncation: if it
        # approaches max_flat_actions, end_turn/seize were being crowded out
        # before the truncation fix and the cap should be raised.
        if "seize_available_rate" in m:
            self.logger.record("eval/seize_available_rate", m["seize_available_rate"])
        if "max_legal_actions" in m:
            self.logger.record("eval/max_legal_actions", m["max_legal_actions"])
        # Army-economy diagnostics: peak/mean army size and unspent gold.
        # A high peak army with near-zero banked gold means the economy is
        # funding mass (convert-all-gold-to-units) -- the "slow-walk a big
        # army" signature -- vs. a small army winning with banked gold to
        # spare, which is the precise/efficient regime.
        if "peak_own_units" in m:
            self.logger.record("eval/peak_own_units", m["peak_own_units"])
            self.logger.record("eval/mean_own_units", m["mean_own_units"])
            self.logger.record("eval/peak_gold_banked", m["peak_gold_banked"])
            self.logger.record("eval/mean_gold_banked", m["mean_gold_banked"])

        if self.verbose:
            print(
                f"  [eval @ {m['timesteps']:>9,}]  "
                f"WR={m['win_rate'] * 100:5.1f}%  "
                f"reward={m['avg_reward']:+8.1f} (+/-{m['std_reward']:5.1f})  "
                f"len={m['avg_length']:5.1f}  "
                f"turns={m['avg_turns']:5.1f}  "
                f"W/L/D={m['wins']}/{m['losses']}/{m['draws']}  "
                f"seize_avail={m.get('seize_available_rate', 0.0) * 100:4.1f}%  "
                f"max_legal={m.get('max_legal_actions', 0)}  "
                f"army(pk/mn)={m.get('peak_own_units', 0)}/{m.get('mean_own_units', 0.0):.1f}  "
                f"gold(pk/mn)={m.get('peak_gold_banked', 0.0):.0f}/{m.get('mean_gold_banked', 0.0):.0f}"
            )

        # Save best by win rate, with avg_reward as a tiebreaker so we don't
        # latch onto the first 0%-WR snapshot.
        if self.save_dir is not None:
            score = (m["win_rate"], m["avg_reward"])
            best = (self.best_win_rate, self._best_reward)
            if score > best:
                self.best_win_rate = m["win_rate"]
                self._best_reward = m["avg_reward"]
                self.best_timestep = int(self.num_timesteps)
                self.model.save(str(self.save_dir / "best_model.zip"))


class PromotionCallback(BaseCallback):
    """Stop ``model.learn()`` early when a paired :class:`PeriodicEvalCallback`
    reports sustained win-rate above a threshold.

    The callback consumes ``eval_callback.results`` rather than running its
    own evaluation, so ordering matters: pass ``PeriodicEvalCallback`` first
    in the SB3 ``CallbackList`` (or as an earlier list element) so this
    callback sees freshly-appended results on the same step.

    Returns ``False`` from :meth:`_on_step` once ``patience`` consecutive
    evaluations have ``win_rate >= threshold``. SB3 honours the ``False``
    return by exiting the current ``learn()`` call cleanly. The bootstrap
    curriculum runner inspects :attr:`promoted` afterwards to decide whether
    to advance to the next stage or raise ``CurriculumStalled``.

    ``min_timesteps`` is *stage-relative*: it counts env steps since this
    stage's ``learn()`` call began (the offset is captured in
    ``_on_training_start``), not against the cumulative ``num_timesteps``
    counter — the bootstrap runner trains with ``reset_num_timesteps=False``,
    so the absolute counter at stage entry already exceeds any reasonable
    per-stage minimum from the second stage onward.
    """

    def __init__(
        self,
        eval_callback: PeriodicEvalCallback,
        threshold: float,
        patience: int = 2,
        verbose: int = 1,
        min_timesteps: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if min_timesteps < 0:
            raise ValueError(f"min_timesteps must be >= 0, got {min_timesteps}")
        self.eval_callback = eval_callback
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.min_timesteps = int(min_timesteps)
        self._consumed: int = 0
        self._streak: int = 0
        self.promoted: bool = False
        # Timestep count at the start of this stage's ``learn()`` call.
        # ``num_timesteps`` is cumulative across stages (the bootstrap
        # runner passes ``reset_num_timesteps=False``), so ``min_timesteps``
        # must be measured stage-relative — comparing against the absolute
        # counter would make the gate a silent no-op on every stage after
        # the first (cumulative steps at stage entry already exceed any
        # plausible per-stage minimum). Captured in ``_on_training_start``;
        # initialized to 0 so direct ``_on_step`` driving (unit tests)
        # keeps from-zero semantics.
        self._stage_start_step: int = 0

    def _on_training_start(self) -> None:
        # See ``EntropyScheduleCallback._on_training_start`` — same pattern:
        # snapshot the cumulative counter so the pre-window gate below
        # measures steps trained *within this stage*.
        self._stage_start_step = int(self.num_timesteps)

    def _on_step(self) -> bool:
        # Pre-window: stage hasn't trained enough yet for promotion to
        # fire. Advance the consumed pointer past any pre-window evals
        # (they don't contribute to the post-window streak) and reset
        # the streak so post-window counting starts fresh. Guards against
        # the "skip-ahead" failure where a strong carry-in policy passes
        # threshold on the first eval and promotes a stage with ~0
        # stage-specific learning. ``min_timesteps`` is stage-relative
        # (steps since this stage's learn() began), not cumulative.
        stage_elapsed = int(self.num_timesteps) - self._stage_start_step
        if self.min_timesteps > 0 and stage_elapsed < self.min_timesteps:
            self._consumed = len(self.eval_callback.results)
            self._streak = 0
            return True
        # Consume any results the eval callback has appended since we last
        # looked. Iterating handles the unusual case of multiple new results
        # in a single step (shouldn't happen in practice but is cheap to
        # support and keeps the streak accounting correct).
        results = self.eval_callback.results
        while self._consumed < len(results):
            wr = float(results[self._consumed]["win_rate"])
            if wr >= self.threshold:
                self._streak += 1
            else:
                self._streak = 0
            self._consumed += 1
            if self._streak >= self.patience:
                self.promoted = True
                if self.verbose:
                    print(
                        f"  [promote] win_rate >= {self.threshold:.0%} for "
                        f"{self._streak} consecutive evals at "
                        f"{self.num_timesteps:,} steps — advancing"
                    )
                return False
        return True


class EntropyScheduleCallback(BaseCallback):
    """Anneal ``model.ent_coef`` from ``start`` to ``end`` over a stage.

    Use case: PPO benefits from elevated exploration on map-shift /
    opponent-shift transitions, but holding a high entropy coefficient
    for the entire stage prevents the policy from committing as it
    approaches the promotion threshold (eval WR oscillates ±15%
    between adjacent evals because sampled actions remain noisy). A
    schedule that starts high and cools to a small commitment-phase
    value gives both: early exploration plus late convergence.

    SB3 reads ``self.ent_coef`` fresh inside every ``train()`` step
    (see ``stable_baselines3.ppo.ppo.PPO.train``), so writing the
    attribute in ``_on_step`` is the documented way to drive a
    schedule without subclassing PPO. The bootstrap runner installs
    this callback per stage and removes it on stage exit.

    Progress is computed against ``total_timesteps`` (the stage's
    own budget), starting from whatever ``num_timesteps`` was when
    the stage's ``learn()`` call began. That matters because the
    bootstrap runner uses ``reset_num_timesteps=False``, so
    ``num_timesteps`` is cumulative across stages.

    Args:
        start: Initial entropy coefficient.
        end: Final entropy coefficient at the end of the stage.
        total_timesteps: Stage budget (matches ``learn(total_timesteps=...)``).
        schedule: ``"linear"`` (default) or ``"cosine"`` (smooth half-cosine
            from ``start`` to ``end``).
    """

    _SCHEDULES = ("linear", "cosine")

    def __init__(
        self,
        start: float,
        end: float,
        total_timesteps: int,
        schedule: str = "linear",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if start < 0 or end < 0:
            raise ValueError(f"start/end must be >= 0, got start={start}, end={end}")
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
        if schedule not in self._SCHEDULES:
            raise ValueError(f"schedule must be one of {self._SCHEDULES}, got '{schedule}'")
        self.start = float(start)
        self.end = float(end)
        self.total_timesteps = int(total_timesteps)
        self.schedule = schedule
        self._stage_start_step: int | None = None

    def _on_training_start(self) -> None:
        # ``num_timesteps`` is cumulative across stages because the
        # bootstrap runner passes ``reset_num_timesteps=False``; capture
        # the stage's starting offset here so progress is computed per
        # stage rather than per run.
        self._stage_start_step = int(self.num_timesteps)

    def _value_at(self, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        if self.schedule == "linear":
            return self.start + (self.end - self.start) * progress
        # cosine: smooth ease from start -> end across [0, 1].
        return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * progress))

    def _on_step(self) -> bool:
        if self._stage_start_step is None:
            # _on_training_start should always run first, but be defensive
            # in case a caller invokes _on_step directly (e.g. unit tests).
            self._stage_start_step = int(self.num_timesteps)
        elapsed = int(self.num_timesteps) - self._stage_start_step
        progress = elapsed / self.total_timesteps if self.total_timesteps > 0 else 1.0
        new_value = float(self._value_at(progress))
        # Setting the attribute is cheap; SB3 picks it up on the next
        # train() iteration. Use setattr so mypy doesn't complain about
        # ``ent_coef`` not being declared on ``BaseAlgorithm`` -- it's
        # a PPO/MaskablePPO-specific field, not part of the base class.
        setattr(self.model, "ent_coef", new_value)
        # Tensorboard: emit the live coefficient so the schedule shows
        # up alongside other train/* curves. ``record`` is buffered
        # until the next logger.dump(), which SB3 calls after train().
        self.logger.record("train/ent_coef", new_value)
        return True
