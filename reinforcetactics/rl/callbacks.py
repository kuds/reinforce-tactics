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
  ``LRScheduleCallback`` does the same for the learning rate (through
  ``model.lr_schedule``, which is what SB3's optimizer update reads).
- ``RollingCheckpointCallback`` keeps a rolling ``latest.zip`` for
  resuming an interrupted stage; ``RegressionGuardCallback`` restores a
  stage's best checkpoint after a sustained within-stage regression.

The callbacks are designed to work with ``MaskablePPO`` from sb3-contrib
as well as plain ``PPO`` from stable-baselines3 — they don't import
sb3-contrib, so the module remains usable in non-masked training too.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import safe_mean

from reinforcetactics.cloud.storage import PARTIAL_SUFFIX
from reinforcetactics.rl.evaluation import (
    EvalEnvPool,
    evaluate_model,
    evaluate_model_vec,
    wilson_lower_bound,
    z_for_confidence,
)

logger = logging.getLogger(__name__)


def save_model_atomically(model: Any, path: str | os.PathLike[str]) -> None:
    """``model.save(path)``, except ``path`` never holds a half-written checkpoint.

    SB3 writes the zip in place. Interrupted mid-save (the SIGTERM that
    scripts/train/train_bootstrap.py turns into ``SystemExit`` on a Vertex
    cancel or preemption, an OOM kill), it left a zip missing entries where
    the last good checkpoint had been, and the GCS sync then uploaded that
    over the good remote copy. Writing a ``.partial`` sibling and renaming it
    into place means ``path`` is always the old checkpoint or the new one,
    including for a sync reading it from another process mid-save; uploads
    skip ``.partial`` files.
    """
    final = Path(path)
    if not final.suffix:  # SB3 appends ".zip" to a suffix-less path
        final = final.with_name(final.name + ".zip")
    partial = final.with_name(final.name + PARTIAL_SUFFIX)
    try:
        model.save(str(partial))
        os.replace(partial, final)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _report_write_failure(hook: Callable[[str], None] | None, what: str) -> None:
    """Report a failed best-effort write: to ``hook`` when set, else as a logged warning.

    Call it inside the ``except`` block, so the traceback is still current
    for ``exc_info``.
    """
    if hook is not None:
        hook(what)
    else:
        logger.warning("could not %s", what, exc_info=True)


class AtomicCheckpointCallback(CheckpointCallback):
    """SB3's ``CheckpointCallback`` with the model zip written by ``save_model_atomically``.

    Every trainer whose output directory a sync reads while it runs (e.g.
    vertex_train.py uploading logs/ periodically and on SIGTERM) needs this:
    the stock callback writes the zip in place, so a sync or a kill mid-save
    can publish a truncated checkpoint.
    """

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        model_path = self._checkpoint_path(extension="zip")
        save_model_atomically(self.model, model_path)
        if self.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")
        if self.save_replay_buffer and getattr(self.model, "replay_buffer", None) is not None:
            self.model.save_replay_buffer(self._checkpoint_path("replay_buffer_", extension="pkl"))  # type: ignore[attr-defined]
        vec_normalize = self.model.get_vec_normalize_env()
        if self.save_vecnormalize and vec_normalize is not None:
            vec_normalize.save(self._checkpoint_path("vecnormalize_", extension="pkl"))
        return True


class SaveModelAtomicallyCallback(BaseCallback):
    """Save the model to ``path`` with ``save_model_atomically`` each time it is called.

    Meant as an eval callback's ``callback_on_new_best`` in place of its
    ``best_model_save_path``, which SB3 writes in place.
    """

    def __init__(self, path: str | os.PathLike[str], verbose: int = 0):
        super().__init__(verbose)
        self.path = path

    def _on_step(self) -> bool:
        save_model_atomically(self.model, self.path)
        if self.verbose >= 1:
            print(f"Saved new best model to {self.path}")
        return True


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
        # Called (inside the ``except``) with a description when a CSV append
        # fails; the curriculum runner counts these (review rltrain-21).
        # Unset, the failure is logged with its traceback.
        self.on_write_failure: Callable[[str], None] | None = None

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
        except Exception:  # noqa: BLE001 - best effort; reported, never raised
            _report_write_failure(self.on_write_failure, f"append a row to {self.csv_path}")

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
    parallel envs are rolling. Best model (by gate win rate, with avg
    reward as tiebreaker) is saved to ``save_dir/best_model.zip`` when
    ``save_dir`` is provided; the best is tracked either way.

    Which policy is measured (review rltrain-4 / prior-5):
    ``deterministic=False`` samples actions from the policy PPO trains,
    ``True`` (the default here, as it always was for direct callers such as
    ppo_training.ipynb) takes its argmax. The curriculum passes
    ``eval.eval_deterministic``, whose default is the stochastic policy.
    With ``eval_both_modes`` the other mode is evaluated too, on the same
    seeds, and every row carries ``win_rate_stochastic`` and
    ``win_rate_greedy`` (plus the other mode's W/L/D under ``other_mode``).
    The row's own fields (``win_rate``, ``wins``, ...) always describe the
    gate mode. A stochastic eval samples from its own seeded torch stream
    (see :mod:`reinforcetactics.rl.evaluation`), so it neither perturbs the
    training run nor depends on it: a row can be re-derived from its
    checkpoint and ``eval_seed``.

    An eval is committed -- its row appended, ``best`` / ``peak`` updated,
    its eval block marked done -- only after it has finished and
    ``best_model.zip`` has been written, so an interrupt in the middle of
    an eval leaves the bookkeeping (and a resume's snapshot of it) as it
    was before the eval, and the eval runs again after a resume.

    Which seats are measured (critic-gaps-2): ``seats=None`` plays the eval
    env's own seat; ``seats=[1, 2]`` plays ``n_eval_episodes`` per seat and
    records ``by_seat`` / ``win_rate_by_seat``. ``gate_win_rate`` -- what
    best_model.zip, the stage's peak and (by default) promotion compare --
    pools the seats (``seat_aggregate='mean'``) or takes the weakest
    (``'min'``).

    ``eval_env`` is a single env (the serial ``evaluate_model`` path) or an
    :class:`~reinforcetactics.rl.evaluation.EvalEnvPool` (the batched
    ``evaluate_model_vec`` path, review rltrain-11).

    When ``results_jsonl_path`` is provided, each eval result is also
    appended to that file as one JSON line the moment it is produced.
    ``self.results`` lives only in memory until the *stage* finishes
    (the curriculum runner writes ``eval_results.json`` at stage end),
    so a run killed mid-stage — the normal Colab death — used to leave
    zero on-disk evidence of everything the in-progress stage measured.
    The JSONL sibling closes that gap without changing the end-of-stage
    JSON contract.

    ``row_hooks`` (callables taking the row) run on every new row before it
    is persisted; :class:`PromotionCallback` uses one to stamp its gate
    statistic onto the row. ``row_extra`` is merged into every row.
    ``on_write_failure`` (called inside the ``except`` with a description)
    receives a failed JSONL append; unset, the failure is logged.
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
        resample_eval_seeds: bool = False,
        best_eligible_after: int = 0,
        verbose: int = 1,
        *,
        deterministic: bool = True,
        eval_both_modes: bool = False,
        seats: Sequence[int] | None = None,
        seat_aggregate: str = "mean",
        start_step: int | None = None,
        last_eval_block: int | None = None,
        best_state: Mapping[str, Any] | None = None,
        row_extra: Mapping[str, Any] | None = None,
        on_write_failure: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(verbose=verbose)
        if seat_aggregate not in ("mean", "min"):
            raise ValueError(f"seat_aggregate must be 'mean' or 'min', got {seat_aggregate!r}")
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.eval_seed_base = int(eval_seed_base)
        # When False (the default) every eval in this stage replays the *same*
        # ``n_eval_episodes`` seeds, so consecutive evals measure the same
        # problem set and are directly comparable. Resampling per eval block --
        # the old behaviour -- meant ``PromotionCallback``'s "patience
        # consecutive crossings" compared two different benchmarks, and
        # ``best_model.zip`` became an argmax over ~100 noisy estimates on
        # different problems (winner's curse). Set True to restore the old
        # rotate-every-eval behaviour.
        self.resample_eval_seeds = bool(resample_eval_seeds)
        # Stage-relative step count before an eval may claim ``best_model.zip``.
        # The callback is constructed fresh per stage but gates on the
        # *cumulative* ``num_timesteps`` (the curriculum runner passes
        # ``reset_num_timesteps=False``), so the very first ``_on_step`` of a
        # stage always fires an eval -- of the *carry-in* policy. Letting that
        # eval win "best" meant ``restore_best_checkpoint_between_stages``
        # could revert the stage's own training. 0 keeps the legacy behaviour.
        self.best_eligible_after = int(best_eligible_after)
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.track_breakdown = bool(track_breakdown)
        # ``trace_dir`` (when set) is the root under which per-eval-block
        # subdirectories are created -- each block's stall-episode JSONL
        # files land in ``trace_dir/eval_<timesteps>/``. Forwarded to
        # ``evaluate_model``; only ``max_steps_truncate`` episodes are
        # dumped, so healthy evals leave no artefacts on disk.
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        self.results_jsonl_path = Path(results_jsonl_path) if results_jsonl_path is not None else None
        self.deterministic = bool(deterministic)
        self.eval_both_modes = bool(eval_both_modes)
        self.seats = [int(s) for s in seats] if seats is not None else None
        self.seat_aggregate = seat_aggregate
        self.row_extra = dict(row_extra or {})
        self.row_hooks: list[Callable[[dict], None]] = []
        self.on_write_failure = on_write_failure

        self.results: list[dict] = []
        # Best eligible eval so far: ``None`` until one exists (it used to
        # start at -1.0 and reach config.json as the "best" win rate of a
        # stage that promoted on ineligible evals; review rltrain-6).
        best_state = dict(best_state or {})
        self.best_win_rate: float | None = best_state.get("best_win_rate")
        best_reward = best_state.get("best_reward")
        self._best_reward: float = float(best_reward) if best_reward is not None else float("-inf")
        # Cumulative ``num_timesteps`` at which the current best_model.zip was
        # saved. -1 until a best is recorded. Exposed so the curriculum runner
        # can report how far *into the stage* the saved peak actually was --
        # a peak at the stage's first eval means the carry-in policy was
        # already strong and the stage did ~0 stage-specific learning (the
        # "skip-ahead" handoff failure documented in bootstrap_lessons_learned).
        self.best_timestep: int = int(best_state.get("best_timestep", -1))
        # Highest gate win rate of *any* eval (carry-in evals included), with
        # its timestep: what a stall message reports as the stage's peak.
        self.peak_win_rate: float | None = best_state.get("peak_win_rate")
        self.peak_timestep: int = int(best_state.get("peak_timestep", -1))
        self._last_eval_block: int = int(last_eval_block) if last_eval_block is not None else -1
        # Cumulative counter at this stage's ``learn()`` entry, so
        # ``best_eligible_after`` is measured stage-relative. Mirrors
        # ``PromotionCallback._stage_start_step`` / ``EntropyScheduleCallback``.
        # Initialized to 0 so direct ``_on_step`` driving (unit tests) keeps
        # from-zero semantics; ``start_step`` pins it (a resumed stage).
        self._fixed_start_step = start_step
        self._stage_start_step: int = int(start_step) if start_step is not None else 0

    def best_state(self) -> dict[str, Any]:
        """The best / peak bookkeeping, for carrying across a retry or a resume."""
        return {
            "best_win_rate": self.best_win_rate,
            "best_reward": self._best_reward if math.isfinite(self._best_reward) else None,
            "best_timestep": self.best_timestep,
            "peak_win_rate": self.peak_win_rate,
            "peak_timestep": self.peak_timestep,
        }

    def _on_training_start(self) -> None:
        if self._fixed_start_step is None:
            self._stage_start_step = int(self.num_timesteps)
        # Fail at the start of learn(), not at the first eval: a flat_discrete
        # policy scored on another decode table measures actions it never
        # chose (evaluate_model makes the same check).
        from reinforcetactics.rl.gym_env import check_flat_action_version

        check_flat_action_version(self.model, self.eval_env, what="the eval env")

    def _on_training_end(self) -> None:
        # The last eval of a learn() that ran out of budget would otherwise
        # stay in the logger's buffer: SB3 dumps before train(), not after
        # the final rollout (review rltrain-19).
        dump = getattr(self.logger, "dump", None)
        if dump is not None:
            dump(self.num_timesteps)

    def _on_step(self) -> bool:
        # Trigger when num_timesteps crosses an eval_freq boundary. Using
        # block index (not modulo) avoids missing/double-firing when
        # num_timesteps jumps by n_envs > 1 each step.
        block = self.num_timesteps // self.eval_freq
        if block > self._last_eval_block:
            # ``_do_eval`` marks the block done only once the eval is
            # committed; an interrupt mid-eval leaves it pending.
            self._do_eval(block)
        return True

    def _evaluate(self, *, deterministic: bool, seed: int, trace_dir: Path | None, track_breakdown: bool) -> dict:
        kwargs: dict[str, Any] = {
            "n_episodes": self.n_eval_episodes,
            "seed": seed,
            "track_breakdown": track_breakdown,
            "deterministic": deterministic,
        }
        if self.seats is not None:
            kwargs["seats"] = self.seats
        if trace_dir is not None:
            kwargs["trace_dir"] = trace_dir
        if isinstance(self.eval_env, EvalEnvPool):
            return evaluate_model_vec(self.model, self.eval_env, **kwargs)
        return evaluate_model(self.model, self.eval_env, **kwargs)

    def _gate_win_rate(self, m: Mapping[str, Any]) -> float:
        by_seat = m.get("by_seat") or {}
        if self.seat_aggregate == "min" and len(by_seat) > 1:
            return float(min(v["win_rate"] for v in by_seat.values()))
        return float(m["win_rate"])

    def _do_eval(self, block: int | None = None) -> None:
        if block is None:
            block = int(self.num_timesteps) // self.eval_freq
        # Fixed problem set by default -- see ``resample_eval_seeds``.
        if self.resample_eval_seeds:
            eval_seed = self.eval_seed_base + 1000 * block
        else:
            eval_seed = self.eval_seed_base
        # One subdir per eval block, named by the timestep at which the block
        # fires, so traces from different evals don't collide and a stalled
        # episode is easy to map back to the eval row in the printed log.
        trace_dir = self.trace_dir / f"eval_{int(self.num_timesteps):09d}" if self.trace_dir is not None else None
        m = self._evaluate(
            deterministic=self.deterministic, seed=eval_seed, trace_dir=trace_dir, track_breakdown=self.track_breakdown
        )
        m["timesteps"] = int(self.num_timesteps)
        m["eval_seed"] = eval_seed
        m["deterministic"] = self.deterministic
        m.update(self.row_extra)
        # The gate mode's win rate, and the other mode's when evaluated on the
        # same seeds (without traces or breakdown: only its outcomes matter).
        mode_wr = {self.deterministic: float(m["win_rate"])}
        if self.eval_both_modes:
            other = self._evaluate(deterministic=not self.deterministic, seed=eval_seed, trace_dir=None, track_breakdown=False)
            mode_wr[not self.deterministic] = float(other["win_rate"])
            m["other_mode"] = {
                "deterministic": not self.deterministic,
                "gate_win_rate": self._gate_win_rate(other),
                **{
                    k: other[k]
                    for k in ("win_rate", "wins", "losses", "draws", "episodes", "draw_rate", "avg_reward")
                    if k in other
                },
                "win_rate_by_seat": {s: v["win_rate"] for s, v in (other.get("by_seat") or {}).items()},
            }
        m["win_rate_stochastic"] = mode_wr.get(False)
        m["win_rate_greedy"] = mode_wr.get(True)
        m["win_rate_by_seat"] = {s: v["win_rate"] for s, v in (m.get("by_seat") or {}).items()}
        m["gate_win_rate"] = self._gate_win_rate(m)
        # Stage-relative position of this eval, and whether it may claim
        # ``best_model.zip``. Both are persisted so post-hoc analysis can tell
        # a carry-in baseline row from a row this stage actually earned.
        stage_elapsed = int(self.num_timesteps) - self._stage_start_step
        best_eligible = stage_elapsed >= self.best_eligible_after
        m["stage_steps"] = stage_elapsed
        m["best_eligible"] = bool(best_eligible)

        gate_wr = m["gate_win_rate"]
        new_peak = self.peak_win_rate is None or gate_wr > self.peak_win_rate
        # Best by gate win rate, with avg_reward as a tiebreaker so we don't
        # latch onto the first 0%-WR snapshot. Evals inside the
        # ``best_eligible_after`` window are recorded but cannot claim the
        # best -- they measure the policy this stage inherited, not one it
        # produced. Tracked with or without a save_dir (review prior-16).
        new_best = False
        if best_eligible:
            best = (self.best_win_rate, self._best_reward) if self.best_win_rate is not None else None
            new_best = best is None or (gate_wr, m["avg_reward"]) > best
        m["saved_best"] = bool(new_best and self.save_dir is not None)
        for hook in self.row_hooks:
            hook(m)
        if new_best and self.save_dir is not None:
            save_model_atomically(self.model, self.save_dir / "best_model.zip")

        # Incremental persistence: append the row now so a mid-stage kill
        # (Colab disconnect / OOM) doesn't erase every eval this stage ran.
        # Best-effort — a disk hiccup must not abort training. Written before
        # the commit below: a row whose eval block a checkpoint did not
        # commit is dropped (and the eval replayed) by a resume, and a row
        # that says it saved best_model.zip keeps the resumed best record
        # true to that file.
        if self.results_jsonl_path is not None:
            try:
                self.results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.results_jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(m, default=float) + "\n")
            except Exception:  # noqa: BLE001 - best effort; reported, never raised
                _report_write_failure(self.on_write_failure, f"append an eval row to {self.results_jsonl_path}")

        # Commit. Nothing above changed this callback's state, so an
        # interrupt before this point (mid-eval, or mid-save of
        # best_model.zip, which is atomic) leaves the best record pointing at
        # the weights best_model.zip holds, and the eval block pending.
        if new_peak:
            self.peak_win_rate = gate_wr
            self.peak_timestep = int(self.num_timesteps)
        if new_best:
            self.best_win_rate = gate_wr
            self._best_reward = float(m["avg_reward"])
            self.best_timestep = int(self.num_timesteps)
        self.results.append(m)
        self._last_eval_block = max(self._last_eval_block, int(block))

        self._log(m)

    def _log(self, m: Mapping[str, Any]) -> None:
        # Tensorboard: log the most useful scalars so they show up alongside
        # the SB3-internal train/* and rollout/* curves.
        self.logger.record("eval/win_rate", m["win_rate"])
        self.logger.record("eval/gate_win_rate", m["gate_win_rate"])
        if m.get("win_rate_stochastic") is not None:
            self.logger.record("eval/win_rate_stochastic", m["win_rate_stochastic"])
        if m.get("win_rate_greedy") is not None:
            self.logger.record("eval/win_rate_greedy", m["win_rate_greedy"])
        if "draw_rate" in m:
            self.logger.record("eval/draw_rate", m["draw_rate"])
            self.logger.record("eval/loss_rate", m.get("loss_rate", 0.0))
        for seat, wr in (m.get("win_rate_by_seat") or {}).items():
            self.logger.record(f"eval/win_rate_seat{seat}", wr)
        self.logger.record("eval/mean_reward", m["avg_reward"])
        self.logger.record("eval/mean_ep_length", m["avg_length"])
        self.logger.record("eval/mean_ep_turns", m["avg_turns"])
        # Action-space diagnostics. ``seize_available_rate`` is the fraction
        # of decision points where a capture was legal -- a low value means
        # the policy rarely even reaches a capturable tile (navigation
        # bottleneck) vs. reaches one but declines (reward/exploration).
        # ``max_legal_actions`` (counted before truncation) and
        # ``flat_truncated_rate`` (share of decision points whose
        # flat_discrete table was cut to max_flat_actions) show when the
        # cap bites and should be raised.
        if "seize_available_rate" in m:
            self.logger.record("eval/seize_available_rate", m["seize_available_rate"])
        if "max_legal_actions" in m:
            self.logger.record("eval/max_legal_actions", m["max_legal_actions"])
        if "flat_truncated_rate" in m:
            self.logger.record("eval/flat_truncated_rate", m["flat_truncated_rate"])
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
        # Structure auto-heal economics (sums over the eval in
        # ``combat_stats``; normalised per episode here). own_heal_gold is
        # the agent's silent gold drain from parking wounded units on its
        # structures; opp_heal_hp is the free durability the opponent's
        # rebuild economy received -- the meat-wall / draw-machine probe.
        combat = m.get("combat_stats") or {}
        heal_eps = max(1, int(m.get("episodes", 0) or self.n_eval_episodes or 1))
        if "own_heal_gold" in combat:
            self.logger.record("eval/own_heal_gold_per_ep", combat["own_heal_gold"] / heal_eps)
            self.logger.record("eval/own_heal_hp_per_ep", combat.get("own_heal_hp", 0.0) / heal_eps)
            self.logger.record("eval/opp_heal_gold_per_ep", combat.get("opp_heal_gold", 0.0) / heal_eps)
            self.logger.record("eval/opp_heal_hp_per_ep", combat.get("opp_heal_hp", 0.0) / heal_eps)

        if self.verbose:
            mode = "greedy" if self.deterministic else "stoch"
            other = ""
            other_wr = m.get("win_rate_stochastic") if self.deterministic else m.get("win_rate_greedy")
            if self.eval_both_modes and other_wr is not None:
                other = f" ({'stoch' if self.deterministic else 'greedy'} {other_wr * 100:5.1f}%)"
            seats = ""
            if len(m.get("win_rate_by_seat") or {}) > 1:
                seats = "  seats=" + "/".join(f"P{s}:{wr * 100:.0f}%" for s, wr in m["win_rate_by_seat"].items())
            print(
                f"  [eval @ {m['timesteps']:>9,}]  "
                f"WR({mode})={m['win_rate'] * 100:5.1f}%{other}{seats}  "
                f"reward={m['avg_reward']:+8.1f} (+/-{m['std_reward']:5.1f})  "
                f"len={m['avg_length']:5.1f}  "
                f"turns={m['avg_turns']:5.1f}  "
                f"W/L/D={m['wins']}/{m['losses']}/{m['draws']}  "
                f"seize_avail={m.get('seize_available_rate', 0.0) * 100:4.1f}%  "
                f"max_legal={m.get('max_legal_actions', 0)}  "
                f"army(pk/mn)={m.get('peak_own_units', 0)}/{m.get('mean_own_units', 0.0):.1f}  "
                f"gold(pk/mn)={m.get('peak_gold_banked', 0.0):.0f}/{m.get('mean_gold_banked', 0.0):.0f}  "
                f"heal$(own/opp)={combat.get('own_heal_gold', 0.0) / heal_eps:.0f}/{combat.get('opp_heal_gold', 0.0) / heal_eps:.0f}"
            )


# ---------------------------------------------------------------------------
# Promotion gate (review rltrain-12 / prior-5)
# ---------------------------------------------------------------------------


def _seat_counts(row: Mapping[str, Any], seat_aggregate: str) -> list[tuple[float, float, int]] | None:
    """``[(wins, draws, episodes), ...]``: one entry per seat for ``min``, one pooled entry for ``mean``.

    ``None`` when the row carries no counts (an older row, or a test stub
    holding only ``win_rate``).
    """
    by_seat = row.get("by_seat")
    groups: list[tuple[float, float, int]] = []
    if isinstance(by_seat, Mapping) and by_seat:
        groups = [(float(v.get("wins", 0)), float(v.get("draws", 0)), int(v.get("episodes", 0))) for v in by_seat.values()]
    elif row.get("episodes"):
        groups = [(float(row.get("wins", 0)), float(row.get("draws", 0)), int(row["episodes"]))]
    if not groups or any(n <= 0 for _, _, n in groups):
        return None
    if seat_aggregate == "min" or len(groups) == 1:
        return groups
    return [(sum(g[0] for g in groups), sum(g[1] for g in groups), sum(g[2] for g in groups))]


def gate_statistic(
    row: Mapping[str, Any],
    *,
    criterion: str = "point",
    score: str = "win_rate",
    z: float = 1.6448536269514722,
    seat_aggregate: str = "mean",
) -> float:
    """The per-eval number a promotion criterion compares to the threshold.

    ``point`` / ``rolling``: the point estimate of the score (wins, or wins
    plus half the draws, over episodes); ``wilson``: the Wilson lower bound
    of it at ``z``. With several seats, ``seat_aggregate='mean'`` pools the
    episodes and ``'min'`` takes the weakest seat. A row without counts
    falls back to its ``gate_win_rate`` / ``win_rate``.
    """
    groups = _seat_counts(row, seat_aggregate)
    if groups is None:
        return float(row.get("gate_win_rate", row["win_rate"]))
    values = []
    for wins, draws, n in groups:
        successes = wins + 0.5 * draws if score == "win_plus_half_draw" else wins
        values.append(wilson_lower_bound(successes, n, z) if criterion == "wilson" else successes / n)
    return float(min(values))


class PromotionCallback(BaseCallback):
    """Stop ``model.learn()`` early when a paired :class:`PeriodicEvalCallback`
    reports sustained win-rate above a threshold.

    The callback consumes ``eval_callback.results`` rather than running its
    own evaluation, so ordering matters: pass ``PeriodicEvalCallback`` first
    in the SB3 ``CallbackList`` (or as an earlier list element) so this
    callback sees freshly-appended results on the same step.

    Returns ``False`` from :meth:`_on_step` once ``patience`` consecutive
    evaluations pass the ``criterion`` (see :func:`gate_statistic`):

    * ``point`` (default, the historical gate): the eval's point estimate
      ``>= threshold``;
    * ``wilson``: the Wilson score lower bound of successes / episodes at
      one-sided ``confidence`` ``>= threshold``;
    * ``rolling``: the mean point estimate of the last ``rolling_k``
      post-window evals ``>= threshold`` (no pass before ``rolling_k``
      evals).

    ``score='win_plus_half_draw'`` scores a draw as half a win instead of a
    loss. SB3 honours the ``False`` return by exiting the current
    ``learn()`` call cleanly; the eval that promoted is flushed to
    TensorBoard first (review rltrain-19). The bootstrap curriculum runner
    inspects :attr:`promoted` afterwards to decide whether to advance to the
    next stage or raise ``CurriculumStalled``.

    ``min_timesteps`` is *stage-relative*: it counts env steps since this
    stage's ``learn()`` call began (the offset is captured in
    ``_on_training_start``, or pinned with ``start_step`` for a resumed
    stage), not against the cumulative ``num_timesteps`` counter — the
    bootstrap runner trains with ``reset_num_timesteps=False``, so the
    absolute counter at stage entry already exceeds any reasonable
    per-stage minimum from the second stage onward.

    ``initial_state`` (from :meth:`state_dict`) restores the streak, the
    rolling window, the gate record and whether the stage already promoted
    (a resume whose checkpoint landed on the promoting eval); the eval rows
    already in ``eval_callback.results`` are then treated as consumed.
    ``record`` (from :meth:`record`) carries only the gate record, across a
    retry: the attempt starts a new streak, but a stall message covers
    every attempt.

    The gate record is what a stall is reported from (review rltrain-7):
    the peak of the value the criterion compared with the threshold (the
    point estimate, the Wilson bound or the rolling mean, of the chosen
    score, seat-aggregated), how many evals passed, the longest run of
    passes, the same for the evals before ``min_timesteps``, which do not
    count, and the peak pooled ``win_rate`` (``peak_pooled_win_rate``: with
    ``seat_aggregate='min'`` over several seats, :attr:`weaker_seat`, the
    gate compares the weaker seat's rate instead). Every row also gets
    ``gate_value`` / ``gate_passed`` (what the gate compared on it and
    whether it passed; ``None`` / False before
    ``min_timesteps`` and while a rolling window is still filling, when the
    gate compares nothing).
    """

    def __init__(
        self,
        eval_callback: PeriodicEvalCallback,
        threshold: float,
        patience: int = 2,
        verbose: int = 1,
        min_timesteps: int = 0,
        *,
        criterion: str = "point",
        rolling_k: int = 3,
        confidence: float = 0.95,
        score: str = "win_rate",
        seat_aggregate: str = "mean",
        start_step: int | None = None,
        initial_state: Mapping[str, Any] | None = None,
        record: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(verbose=verbose)
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if min_timesteps < 0:
            raise ValueError(f"min_timesteps must be >= 0, got {min_timesteps}")
        if criterion not in ("point", "wilson", "rolling"):
            raise ValueError(f"criterion must be 'point', 'wilson' or 'rolling', got {criterion!r}")
        if score not in ("win_rate", "win_plus_half_draw"):
            raise ValueError(f"score must be 'win_rate' or 'win_plus_half_draw', got {score!r}")
        if rolling_k < 1:
            raise ValueError(f"rolling_k must be >= 1, got {rolling_k}")
        self.eval_callback = eval_callback
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.min_timesteps = int(min_timesteps)
        self.criterion = criterion
        self.rolling_k = int(rolling_k)
        self.confidence = float(confidence)
        self.score = score
        self.seat_aggregate = seat_aggregate
        # The seats each eval plays (None: the eval env's own seat). With
        # several and ``seat_aggregate='min'`` the gate compares the weaker
        # seat's rate, not the pooled ``win_rate``.
        seats = getattr(eval_callback, "seats", None)
        self.seats: list[int] | None = [int(s) for s in seats] if seats is not None else None
        self.weaker_seat = seat_aggregate == "min" and len(self.seats or []) > 1
        self._z = z_for_confidence(self.confidence) if criterion == "wilson" else 0.0
        self._consumed: int = 0
        self._streak: int = 0
        self._window: deque[float] = deque(maxlen=self.rolling_k)
        self.promoted: bool = False
        # Last per-eval statistic and the value the gate compared (the
        # rolling mean for 'rolling'); None before the first post-window eval
        # and while a rolling window is still filling.
        self.last_statistic: float | None = None
        self.last_gate_value: float | None = None
        # Timestep count at the start of this stage's ``learn()`` call.
        # ``num_timesteps`` is cumulative across stages (the bootstrap
        # runner passes ``reset_num_timesteps=False``), so ``min_timesteps``
        # must be measured stage-relative — comparing against the absolute
        # counter would make the gate a silent no-op on every stage after
        # the first (cumulative steps at stage entry already exceed any
        # plausible per-stage minimum). Captured in ``_on_training_start``;
        # initialized to 0 so direct ``_on_step`` driving (unit tests)
        # keeps from-zero semantics.
        self._fixed_start_step = start_step
        self._stage_start_step: int = int(start_step) if start_step is not None else 0
        # The gate record (see the class docstring), over every attempt.
        self.peak_gate_value: float | None = None
        self.peak_gate_timestep: int = -1
        self.evals_judged: int = 0
        self.passes: int = 0
        self.longest_streak: int = 0
        self.prewindow_evals: int = 0
        self.prewindow_passes: int = 0
        self.prewindow_peak: float | None = None
        # The peak pooled ``win_rate`` over every eval (pre-window ones too),
        # which a weaker-seat gate's stall report names next to its own peak.
        self.peak_pooled_win_rate: float | None = None
        carried = dict(record or {})
        if initial_state is not None:
            self._streak = int(initial_state.get("streak", 0))
            self._window.extend(float(v) for v in initial_state.get("window", []))
            self.promoted = bool(initial_state.get("promoted", False))
            self._consumed = len(getattr(eval_callback, "results", []))
            carried.update(initial_state.get("record") or {})
        self._load_record(carried)
        hooks = getattr(eval_callback, "row_hooks", None)
        if hooks is not None:
            hooks.append(self._annotate)

    _RECORD_KEYS = (
        "peak_gate_value",
        "peak_gate_timestep",
        "evals_judged",
        "passes",
        "longest_streak",
        "prewindow_evals",
        "prewindow_passes",
        "prewindow_peak",
        "peak_pooled_win_rate",
    )

    _FLOAT_RECORD_KEYS = frozenset({"peak_gate_value", "prewindow_peak", "peak_pooled_win_rate"})

    def _load_record(self, record: Mapping[str, Any]) -> None:
        for key in self._RECORD_KEYS:
            value = record.get(key)
            if value is not None:
                setattr(self, key, float(value) if key in self._FLOAT_RECORD_KEYS else int(value))

    def record(self) -> dict[str, Any]:
        """The gate record (see the class docstring) plus the gate's settings, for a stall report / config.json."""
        return {
            "criterion": self.criterion,
            "score": self.score,
            "threshold": self.threshold,
            "patience": self.patience,
            "rolling_k": self.rolling_k,
            "confidence": self.confidence,
            "min_timesteps": self.min_timesteps,
            "seat_aggregate": self.seat_aggregate,
            "seats": self.seats,
            **{key: getattr(self, key) for key in self._RECORD_KEYS},
        }

    def statistic(self, row: Mapping[str, Any]) -> float:
        """This gate's per-eval statistic for ``row`` (see :func:`gate_statistic`)."""
        return gate_statistic(row, criterion=self.criterion, score=self.score, z=self._z, seat_aggregate=self.seat_aggregate)

    def _judge(self, value: float, window: Sequence[float]) -> tuple[float, bool, bool, list[float]]:
        """``(gate_value, passed, compared, new_window)`` for a post-window eval whose statistic is ``value``.

        ``compared`` is False while a rolling window is still filling (the
        gate compares nothing yet).
        """
        if self.criterion == "rolling":
            new_window = [*window, value][-self.rolling_k :]
            gate_value = sum(new_window) / len(new_window)
            compared = len(new_window) >= self.rolling_k
            return gate_value, compared and gate_value >= self.threshold, compared, new_window
        return value, value >= self.threshold, True, list(window)

    def _in_prewindow(self, timesteps: int) -> bool:
        return self.min_timesteps > 0 and int(timesteps) - self._stage_start_step < self.min_timesteps

    def _annotate(self, row: dict) -> None:
        # Runs inside the eval callback's step, before this callback consumes
        # the row on the same step, so the window it judges with is the one
        # the consumption below uses.
        value = self.statistic(row)
        row["gate_statistic"] = value
        row["gate_criterion"] = self.criterion
        if self._in_prewindow(int(row.get("timesteps", 0))):
            row["gate_value"], row["gate_passed"] = None, False
            return
        gate_value, passed, compared, _ = self._judge(value, list(self._window))
        # A rolling window still filling compared nothing: no gate value (the
        # partial mean would chart as a comparison that never happened).
        row["gate_value"], row["gate_passed"] = (gate_value if compared else None), bool(passed)

    def state_dict(self) -> dict[str, Any]:
        """Streak, rolling window, promotion and gate record, for a resume (see ``initial_state``)."""
        return {"streak": self._streak, "window": list(self._window), "promoted": self.promoted, "record": self.record()}

    def _on_training_start(self) -> None:
        # See ``EntropyScheduleCallback._on_training_start`` — same pattern:
        # snapshot the cumulative counter so the pre-window gate below
        # measures steps trained *within this stage*.
        if self._fixed_start_step is None:
            self._stage_start_step = int(self.num_timesteps)

    def _note_pooled(self, row: Mapping[str, Any]) -> None:
        pooled = row.get("win_rate")
        if pooled is not None and (self.peak_pooled_win_rate is None or float(pooled) > self.peak_pooled_win_rate):
            self.peak_pooled_win_rate = float(pooled)

    def _row_statistic(self, row: Mapping[str, Any]) -> float:
        if row.get("gate_criterion") == self.criterion and "gate_statistic" in row:
            return float(row["gate_statistic"])
        return self.statistic(row)

    def _on_step(self) -> bool:
        if self.promoted:
            # Restored from a checkpoint taken on the promoting eval: the
            # stage is done (the runner does not even call learn()).
            return False
        # Pre-window: stage hasn't trained enough yet for promotion to
        # fire. Advance the consumed pointer past any pre-window evals
        # (they don't contribute to the post-window streak) and reset
        # the streak so post-window counting starts fresh. Guards against
        # the "skip-ahead" failure where a strong carry-in policy passes
        # threshold on the first eval and promotes a stage with ~0
        # stage-specific learning. ``min_timesteps`` is stage-relative
        # (steps since this stage's learn() began), not cumulative.
        results = self.eval_callback.results
        if self._in_prewindow(int(self.num_timesteps)):
            while self._consumed < len(results):
                value = self._row_statistic(results[self._consumed])
                self._note_pooled(results[self._consumed])
                self._consumed += 1
                self.prewindow_evals += 1
                self.prewindow_passes += int(value >= self.threshold)
                if self.prewindow_peak is None or value > self.prewindow_peak:
                    self.prewindow_peak = value
            self._streak = 0
            self._window.clear()
            return True
        # Consume any results the eval callback has appended since we last
        # looked. Iterating handles the unusual case of multiple new results
        # in a single step (shouldn't happen in practice but is cheap to
        # support and keeps the streak accounting correct).
        while self._consumed < len(results):
            row = results[self._consumed]
            value = self._row_statistic(row)
            self._note_pooled(row)
            self.last_statistic = value
            gate_value, passed, compared, window = self._judge(value, list(self._window))
            self._window.clear()
            self._window.extend(window)
            self.last_gate_value = gate_value if compared else None
            self._streak = self._streak + 1 if passed else 0
            self._consumed += 1
            if compared:
                self.evals_judged += 1
                self.passes += int(passed)
                if self.peak_gate_value is None or gate_value > self.peak_gate_value:
                    self.peak_gate_value = gate_value
                    self.peak_gate_timestep = int(row.get("timesteps", self.num_timesteps))
            self.longest_streak = max(self.longest_streak, self._streak)
            if self._streak >= self.patience:
                self.promoted = True
                if self.verbose:
                    what = {"point": "win_rate", "wilson": "Wilson LB", "rolling": f"rolling-{self.rolling_k} mean"}
                    seat = " (weaker seat)" if self.weaker_seat else ""
                    print(
                        f"  [promote] {what[self.criterion]}{seat} {gate_value:.1%} >= {self.threshold:.0%} for "
                        f"{self._streak} consecutive evals at "
                        f"{self.num_timesteps:,} steps — advancing"
                    )
                # Flush the promoting eval to TensorBoard before learn()
                # returns (review rltrain-19): SB3 dumps before train(), and
                # there is no train() after this step. (No model: a unit
                # test driving _on_step directly.)
                if getattr(self, "model", None) is not None:
                    self.logger.record("eval/promoted", 1.0)
                    dump = getattr(self.logger, "dump", None)
                    if dump is not None:
                        dump(self.num_timesteps)
                return False
        return True


# ---------------------------------------------------------------------------
# Stall recovery and resume support (review rltrain-5 / rltrain-7, prior-3)
# ---------------------------------------------------------------------------


class RollingCheckpointCallback(BaseCallback):
    """Save the model to ``path`` (atomically) every ``save_freq`` stage steps.

    The curriculum runner's resume point (``<stage>/latest.zip``, review
    rltrain-5): ``on_save(num_timesteps)`` runs after each save, which is
    where the runner writes its manifest. Placed after the eval and
    promotion callbacks in the callback list, a save lands after any eval of
    the same step, so the manifest's snapshot of eval / promotion state is
    consistent with the saved weights. ``start_step`` (the resumed
    checkpoint's timestep) keeps the cadence of a resumed stage.

    The checkpoint is a resume convenience, so a periodic save that fails
    (a transient I/O error) is reported -- to ``on_error`` inside the
    ``except``, else as a logged warning -- and training goes on; the next
    save is attempted a ``save_freq`` later. ``save_now`` (the interrupt
    path) raises, and its caller decides.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        save_freq: int,
        *,
        on_save: Callable[[int], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        start_step: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if save_freq <= 0:
            raise ValueError(f"save_freq must be > 0, got {save_freq}")
        self.path = Path(path)
        self.save_freq = int(save_freq)
        self.on_save = on_save
        self.on_error = on_error
        self._fixed_start_step = start_step
        self._last_save: int = int(start_step) if start_step is not None else 0
        self.saves = 0
        self.failures = 0

    def _on_training_start(self) -> None:
        if self._fixed_start_step is None:
            self._last_save = int(self.num_timesteps)

    def save_now(self) -> None:
        """Save immediately (an interrupted run's last checkpoint).

        Stamped with the model's own ``num_timesteps``: on an interrupt this
        callback's cached counter can be a step behind (it runs last in the
        callback list), and the manifest must name the step the zip holds.
        """
        timesteps = int(self.model.num_timesteps)
        save_model_atomically(self.model, self.path)
        self._last_save = timesteps
        self.saves += 1
        if self.on_save is not None:
            self.on_save(timesteps)

    def _on_step(self) -> bool:
        if int(self.num_timesteps) - self._last_save >= self.save_freq:
            try:
                self.save_now()
            except Exception:  # noqa: BLE001 - a resume convenience must not end the run
                self.failures += 1
                self._last_save = int(self.num_timesteps)
                _report_write_failure(self.on_error, f"save the rolling checkpoint {self.path}")
                return True
            if self.verbose:
                print(f"  [checkpoint] {self.path} @ {self.num_timesteps:,}")
        return True


class RegressionGuardCallback(BaseCallback):
    """Restore the stage's best checkpoint after a sustained within-stage regression.

    Watches the paired :class:`PeriodicEvalCallback`'s rows: after ``n_evals``
    consecutive evals whose ``gate_win_rate`` is more than ``drop`` below
    the stage's best (``eval_callback.best_win_rate``), loads
    ``best_path`` into the model with ``set_parameters`` and keeps
    training. The load happens at the start of the next rollout, so no
    rollout mixes the two policies. ``restores`` lists each restore.
    """

    def __init__(
        self,
        eval_callback: PeriodicEvalCallback,
        n_evals: int,
        drop: float,
        best_path: str | os.PathLike[str],
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if n_evals < 1:
            raise ValueError(f"n_evals must be >= 1, got {n_evals}")
        if not 0.0 < drop <= 1.0:
            raise ValueError(f"drop must be in (0, 1], got {drop}")
        self.eval_callback = eval_callback
        self.n_evals = int(n_evals)
        self.drop = float(drop)
        self.best_path = Path(best_path)
        self._consumed = len(getattr(eval_callback, "results", []))
        self._below = 0
        self._pending: dict[str, Any] | None = None
        self.restores: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        results = self.eval_callback.results
        while self._consumed < len(results):
            row = results[self._consumed]
            self._consumed += 1
            best = self.eval_callback.best_win_rate
            wr = float(row.get("gate_win_rate", row["win_rate"]))
            if best is None or wr >= best - self.drop:
                self._below = 0
                continue
            self._below += 1
            if self._below >= self.n_evals and self.best_path.exists():
                self._below = 0
                self._pending = {
                    "timesteps": int(row.get("timesteps", self.num_timesteps)),
                    "gate_win_rate": wr,
                    "best_win_rate": best,
                    "best_timestep": self.eval_callback.best_timestep,
                }
        return True

    def _on_rollout_start(self) -> None:
        if self._pending is None:
            return
        event, self._pending = self._pending, None
        self.model.set_parameters(str(self.best_path), exact_match=True)
        event["restored_at"] = int(self.num_timesteps)
        self.restores.append(event)
        self.logger.record("eval/regression_restores", len(self.restores))
        if self.verbose:
            print(
                f"  [regression-guard] {self.n_evals} evals more than {self.drop:.0%} below the stage best "
                f"{event['best_win_rate']:.1%} (last {event['gate_win_rate']:.1%}): restored {self.best_path.name} "
                f"at {self.num_timesteps:,} steps"
            )


class ScheduledAttrCallback(BaseCallback):
    """Shared machinery for annealing a model attribute over a stage budget.

    Subclasses set ``_target_attr`` (the ``model`` attribute written every
    step), ``_tb_key`` (the tensorboard series name), and override
    ``_validate_range`` for their domain. Concrete schedules:
    :class:`EntropyScheduleCallback`, :class:`LRScheduleCallback` and
    :class:`~reinforcetactics.rl.purchase_exploration.PurchaseExploreScheduleCallback`.
    Deliberately a common base rather than sibling subclassing so
    ``isinstance(cb, EntropyScheduleCallback)`` stays False for the
    purchase-ε schedule (the bootstrap tests rely on that).

    Progress is computed against ``total_timesteps`` (the stage's own
    budget, or a shorter anneal horizon), starting from whatever
    ``num_timesteps`` was when the stage's ``learn()`` call began -- or
    from ``start_step`` when given (a resumed stage continues its
    schedule). That matters because the bootstrap runner uses
    ``reset_num_timesteps=False``, so ``num_timesteps`` is cumulative
    across stages. Past the horizon the value holds at ``end``.
    """

    _SCHEDULES: tuple[str, ...] = ("linear", "cosine")
    _target_attr: str
    _tb_key: str

    def __init__(
        self,
        start: float,
        end: float,
        total_timesteps: int,
        schedule: str = "linear",
        verbose: int = 0,
        *,
        start_step: int | None = None,
    ) -> None:
        super().__init__(verbose=verbose)
        self._validate_range(start, end)
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
        if schedule not in self._SCHEDULES:
            raise ValueError(f"schedule must be one of {self._SCHEDULES}, got '{schedule}'")
        self.start = float(start)
        self.end = float(end)
        self.total_timesteps = int(total_timesteps)
        self.schedule = schedule
        self._fixed_start_step = start_step
        self._stage_start_step: int | None = int(start_step) if start_step is not None else None

    def _validate_range(self, start: float, end: float) -> None:
        raise NotImplementedError

    def _on_training_start(self) -> None:
        # ``num_timesteps`` is cumulative across stages because the
        # bootstrap runner passes ``reset_num_timesteps=False``; capture
        # the stage's starting offset here so progress is computed per
        # stage rather than per run.
        if self._fixed_start_step is None:
            self._stage_start_step = int(self.num_timesteps)

    def progress(self) -> float:
        """Fraction of the horizon elapsed (clamped to [0, 1])."""
        if self._stage_start_step is None:
            return 0.0
        elapsed = int(self.num_timesteps) - self._stage_start_step
        return max(0.0, min(1.0, elapsed / self.total_timesteps if self.total_timesteps > 0 else 1.0))

    def _value_at(self, progress: float) -> float:
        progress = max(0.0, min(1.0, progress))
        if self.schedule == "linear":
            return self.start + (self.end - self.start) * progress
        # cosine: smooth ease from start -> end across [0, 1].
        return self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * progress))

    def _apply(self, value: float) -> None:
        # Setting the attribute is cheap; SB3 picks it up on the next
        # train() iteration. Use setattr so mypy doesn't complain about
        # e.g. ``ent_coef`` not being declared on ``BaseAlgorithm`` -- the
        # targets are PPO/MaskablePPO-specific fields, not base-class ones.
        setattr(self.model, self._target_attr, value)

    def _on_step(self) -> bool:
        if self._stage_start_step is None:
            # _on_training_start should always run first, but be defensive
            # in case a caller invokes _on_step directly (e.g. unit tests).
            self._stage_start_step = int(self.num_timesteps)
        new_value = float(self._value_at(self.progress()))
        self._apply(new_value)
        # Tensorboard: emit the live coefficient so the schedule shows
        # up alongside other train/* curves. ``record`` is buffered
        # until the next logger.dump(), which SB3 calls after train().
        self.logger.record(self._tb_key, new_value)
        return True


class EntropyScheduleCallback(ScheduledAttrCallback):
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

    Args:
        start: Initial entropy coefficient.
        end: Final entropy coefficient at the end of the stage.
        total_timesteps: Anneal horizon (the stage budget unless the stage
            sets ``anneal_horizon`` or the schedule its ``horizon``).
        schedule: ``"linear"`` (default) or ``"cosine"`` (smooth half-cosine
            from ``start`` to ``end``).
    """

    _target_attr = "ent_coef"
    _tb_key = "train/ent_coef"

    def _validate_range(self, start: float, end: float) -> None:
        if start < 0 or end < 0:
            raise ValueError(f"start/end must be >= 0, got start={start}, end={end}")


class LRScheduleCallback(ScheduledAttrCallback):
    """Anneal the learning rate from ``start`` to ``end`` over a stage (review rltrain-8 / prior-6).

    SB3 does not read ``model.learning_rate`` during training: each
    ``train()`` calls ``_update_learning_rate``, which sets every optimizer
    param group to ``model.lr_schedule(progress_remaining)`` -- and that
    progress runs over the whole ``learn()`` call, the cumulative counter
    the curriculum never resets. So this callback writes both
    ``model.lr_schedule`` (a constant schedule holding the current value,
    which is what the optimizer sees) and ``model.learning_rate`` (which a
    saved checkpoint reloads its schedule from). Stage-relative like the
    other schedules. ``schedule='constant'`` holds ``start``.
    """

    _SCHEDULES = ("linear", "cosine", "constant")
    _target_attr = "learning_rate"
    _tb_key = "train/learning_rate"

    def _validate_range(self, start: float, end: float) -> None:
        if start <= 0 or end < 0:
            raise ValueError(f"learning rate start must be > 0 and end >= 0, got start={start}, end={end}")

    def _value_at(self, progress: float) -> float:
        if self.schedule == "constant":
            return self.start
        return super()._value_at(progress)

    def _apply(self, value: float) -> None:
        set_learning_rate(self.model, value)


def set_learning_rate(model: Any, value: float) -> None:
    """Make ``value`` the learning rate SB3 applies at the next ``train()``."""
    from stable_baselines3.common.utils import ConstantSchedule

    model.learning_rate = float(value)
    model.lr_schedule = ConstantSchedule(float(value))
