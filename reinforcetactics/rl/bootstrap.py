"""
Curriculum-based PPO bootstrapping.

Trains a single MaskablePPO policy through a sequence of stages
(map x opponent combinations) before handing the resulting checkpoint
off to self-play. Each stage runs until ``PromotionCallback`` reports
that the win rate has held above the stage's threshold for
``patience`` evaluations, at which point ``model.learn()`` returns
early and the runner moves to the next stage.

If a stage exhausts its ``max_timesteps`` budget without promoting, it
is retried up to ``max_retries`` times (default 1) from its best
checkpoint with a fresh budget; a stage that stalls on its last attempt
raises :class:`CurriculumStalled`. Bumping the budget alone usually
masks a real issue (reward shaping, hyperparams), so failing loud after
the retries is the default.

What the gate measures is ``cfg.eval``: the stochastic policy by default
(``eval_deterministic``), with the greedy win rate recorded alongside
(``eval_both_modes``), on the seats ``eval_seats`` names. How it decides is
the stage's promotion criterion (``point`` / ``wilson`` / ``rolling``).

A run that was killed can continue where it stopped: every stage keeps a
rolling ``<stage>/latest.zip`` (``eval.checkpoint_freq``) and the run a
``run_manifest.json``; ``run_curriculum(..., resume=True)`` (the CLI's
``--resume``) skips the promoted stages and resumes the interrupted one.

Configuration lives in :class:`reinforcetactics.rl.config.TrainingConfig`:
``cfg.curriculum.stages`` defines the curriculum, ``cfg.env`` / ``cfg.ppo``
are shared across stages, and each stage may override ``max_steps``,
``max_turns``, ``ent_coef``, ``reward_config``, or ``opponent_kwargs``
on a per-stage basis. ``cfg.eval.eval_freq`` and ``cfg.env.n_envs``
drive eval cadence and parallelism respectively.

Usage:

    from reinforcetactics.rl.bootstrap import run_curriculum
    from reinforcetactics.rl.config import load_config

    cfg = load_config("configs/ppo/bootstrap.yaml")
    result = run_curriculum(cfg, output_dir="benchmarks/bootstrap")
    # result["final_model_path"] -> ready for self-play warm start
"""

from __future__ import annotations

import copy
import csv
import functools
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reinforcetactics.rl.config import CurriculumStage, TrainingConfig, check_ignored_config_fields

logger = logging.getLogger(__name__)

ConfigPath = str | Path

# Fallback discount for ``make_stage_env`` when a caller doesn't pass one.
# Mirrors ``StrategyGameEnv``'s default; only correct while ``ppo.gamma`` is
# left at 0.99, which is why the parameter is documented as "pass cfg.ppo.gamma".
_DEFAULT_SHAPING_GAMMA = 0.99


class CurriculumStalled(RuntimeError):
    """Raised when a stage exhausts its budget without promoting.

    Attributes:
        stage_name: Name of the failing stage.
        achieved_win_rate: Peak gate win rate observed during the stage
            (any eval, every attempt).
        threshold: Promotion threshold the stage was required to clear.
        timesteps: Stage timestep budget that was exhausted.
        patience: Consecutive passing evals the gate needed.
        retries: Retries the stage used before giving up.
        history: Per-stage results gathered up to (and including) the
            stalled stage. Same shape as ``run_curriculum``'s
            ``result["history"]`` so callers can recover diagnostics
            from a partial run by reading
            ``CurriculumStalled.partial_result()``.
        final_model_path: Path to ``final_model.zip`` saved at the
            point of stall (the in-progress policy at the moment the
            stalled stage gave up). ``None`` for older callers that
            didn't pass it.
        best_model_path: Path to the stalled stage's ``best_model.zip``
            when one was saved -- the peak-WR snapshot from before the
            collapse. Most stalls peak at or above the promotion
            threshold before crashing, so this checkpoint (not the
            collapsed ``final_model_path``) is usually the one worth
            warm-starting or replaying. ``None`` if the stage never
            saved a best.
        metrics_callback: ``TrainingMetricsCallback`` accumulated over
            the partial run, exposed for the same reason as
            ``history``.
        gate_record: The promotion gate's record over every attempt
            (:meth:`PromotionCallback.record`): the criterion, the peak of
            the value it compared with the threshold (``peak_gate_value``),
            how many evals passed, the longest run of passes, and the evals
            before ``min_timesteps``. The message is worded from it; without
            it (older callers) from ``achieved_win_rate``.
        trained_timesteps: Env steps the stage trained over all its
            attempts (``timesteps`` is the per-attempt budget).
    """

    def __init__(
        self,
        stage_name: str,
        achieved_win_rate: float | None,
        threshold: float,
        timesteps: int,
        history: list[dict[str, Any]] | None = None,
        final_model_path: str | None = None,
        metrics_callback: Any = None,
        best_model_path: str | None = None,
        patience: int | None = None,
        retries: int = 0,
        gate_record: dict[str, Any] | None = None,
        trained_timesteps: int | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.achieved_win_rate = achieved_win_rate
        self.threshold = threshold
        self.timesteps = timesteps
        self.patience = patience
        self.retries = retries
        self.gate_record = dict(gate_record) if gate_record else None
        self.trained_timesteps = trained_timesteps
        self.history = history or []
        self.final_model_path = final_model_path
        self.best_model_path = best_model_path
        self.metrics_callback = metrics_callback
        retried = f"after {retries} retr{'y' if retries == 1 else 'ies'}" if retries else ""
        if trained_timesteps is not None:
            spent = f"{trained_timesteps:,} timesteps trained, budget {timesteps:,}" + (" per attempt" if retries else "")
            where = f"stalled ({'; '.join(p for p in (retried, spent) if p)})"
        else:
            where = f"stalled at {timesteps:,} timesteps" + (f" ({retried})" if retried else "")
        verdict = stall_verdict(achieved_win_rate, threshold, patience, gate_record)
        super().__init__(f"Stage '{stage_name}' {where}: {verdict}")

    def partial_result(self) -> dict[str, Any]:
        """Return the same dict shape ``run_curriculum`` returns on success.

        Notebook diagnostics / video helpers consume that shape, so
        wrapping the partial state in a ``partial_result()`` lets the
        same plotting and replay code paths run after a stall.
        ``model`` is omitted (re-load from ``final_model_path`` if
        needed -- it isn't safe to keep a live SB3 model object alive
        on an exception path because env handles may be torn down).
        """
        return {
            "model": None,
            "history": list(self.history),
            "final_model_path": self.final_model_path,
            "best_model_path": self.best_model_path,
            "metrics_callback": self.metrics_callback,
            "stalled": True,
            "stalled_stage": self.stage_name,
        }


def gate_label(
    criterion: str,
    score: str = "win_rate",
    *,
    rolling_k: int = 3,
    confidence: float = 0.95,
    weaker_seat: bool = False,
) -> str:
    """What a promotion criterion compares with its threshold, in words ("Wilson 95% lower bound of win_rate").

    ``weaker_seat``: the gate takes the weaker of several seats
    (``eval.seat_aggregate: min``), not the pooled rate.
    """
    measured = "win+draw/2 score" if score == "win_plus_half_draw" else "win_rate"
    if weaker_seat:
        measured = f"weaker-seat {measured}"
    if criterion == "wilson":
        return f"Wilson {confidence:.0%} lower bound of {measured}"
    if criterion == "rolling":
        return f"rolling-{rolling_k} mean {measured}"
    return measured


def stall_verdict(
    achieved_win_rate: float | None,
    threshold: float,
    patience: int | None,
    gate_record: dict[str, Any] | None = None,
) -> str:
    """Why a stage did not promote, in the terms of the gate that judged it (review rltrain-7).

    Worded from the gate record (see :class:`CurriculumStalled`): the peak
    of the value the criterion actually compared, so a Wilson-bound, rolling
    or win+draw/2 stall is not reported as the win-only point estimate's.
    ``achieved_win_rate`` (the peak gate win rate) is added when the gate
    compared something else. When the gate took the weaker of several
    seats (``seat_aggregate: min``) that is said, and the peak pooled
    ``win_rate`` -- the eval rows' ``win_rate`` column -- is added too.
    """
    held = f"patience={patience}" if patience is not None else "the patience window"
    record = gate_record or {}
    if not record:
        if achieved_win_rate is None:
            return "no eval ran"
        if achieved_win_rate >= threshold:
            return (
                f"win_rate peaked at {achieved_win_rate:.1%} (>= threshold {threshold:.1%}) "
                f"but never held it for {held} consecutive evals"
            )
        return f"peak win_rate {achieved_win_rate:.1%} did not reach threshold {threshold:.1%}"

    criterion = str(record.get("criterion", "point"))
    score = str(record.get("score", "win_rate"))
    # The gate took the weaker of several seats: its peak (and the gate win
    # rate's, ``achieved_win_rate``) is that seat's, not the pooled win_rate
    # of the eval rows.
    weaker_seat = record.get("seat_aggregate") == "min" and len(record.get("seats") or []) > 1
    what = gate_label(
        criterion,
        score,
        rolling_k=int(record.get("rolling_k", 3)),
        confidence=float(record.get("confidence", 0.95)),
        weaker_seat=weaker_seat,
    )
    peak = record.get("peak_gate_value")
    passes = int(record.get("passes", 0) or 0)
    judged = int(record.get("evals_judged", 0) or 0)
    point_win_rate = criterion == "point" and score == "win_rate"
    steps = int(record.get("min_timesteps", 0) or 0)
    window = f" after the first {steps:,} stage steps (min_timesteps_before_promotion)" if steps else ""
    if peak is None:
        if achieved_win_rate is None and not int(record.get("prewindow_evals", 0) or 0):
            return "no eval ran"
        if criterion == "rolling":
            verdict = f"the {what} never had {int(record.get('rolling_k', 3))} evals{window} to average"
        else:
            verdict = f"no eval came{window}, so none counted"
    elif passes:
        verdict = (
            f"{what} peaked at {float(peak):.1%} (>= threshold {threshold:.1%}) but never held it for {held} "
            f"consecutive evals (passed on {passes} of {judged} evals, longest run {int(record.get('longest_streak', 0))})"
        )
    else:
        verdict = f"peak {what} {float(peak):.1%} did not reach threshold {threshold:.1%}"
    peaks = []
    if not point_win_rate and achieved_win_rate is not None:
        peaks.append(f"{'weaker-seat ' if weaker_seat else ''}win_rate peaked at {achieved_win_rate:.1%}")
    pooled = record.get("peak_pooled_win_rate")
    if weaker_seat and pooled is not None:
        peaks.append(f"pooled win_rate peaked at {float(pooled):.1%}")
    if peaks:
        verdict += f" ({'; '.join(peaks)})"
    prewindow_passes = int(record.get("prewindow_passes", 0) or 0)
    if prewindow_passes:
        verdict += (
            f"; {prewindow_passes} eval(s) inside the first {steps:,} stage steps (min_timesteps_before_promotion) "
            "passed but do not count"
        )
    return verdict


# ---------------------------------------------------------------------------
# Default builders. Tests / advanced callers can pass replacements through
# `run_curriculum(..., train_env_factory=..., model_factory=...)` to avoid
# importing sb3-contrib or constructing real environments.
# ---------------------------------------------------------------------------


def _default_train_env_factory(stage: CurriculumStage, cfg: TrainingConfig):
    from reinforcetactics.rl.masking import make_maskable_vec_env

    return make_maskable_vec_env(
        n_envs=cfg.env.n_envs,
        seed=cfg.seed,
        use_subprocess=cfg.env.use_subprocess,
        gamma=cfg.ppo.gamma,
        **_stage_env_kwargs(stage, cfg.env),
    )


def _default_eval_env_factory(stage: CurriculumStage, cfg: TrainingConfig):
    # Offset the eval env's construction seed away from the training envs'
    # range (``cfg.seed + rank``) so eval episodes don't deterministically
    # share map / opponent RNG state with concurrent training rollouts. Per
    # eval block ``PeriodicEvalCallback`` reseeds the env on every reset
    # via ``evaluate_model(seed=...)`` -- this offset matters mainly for
    # the initial ``env.reset(seed=...)`` inside ``make_maskable_env``.
    build = functools.partial(make_stage_env, stage, cfg.env, seed=cfg.seed + cfg.eval.seed_offset, gamma=cfg.ppo.gamma)
    if cfg.eval.n_eval_envs <= 1:
        return build()
    # K copies stepped together with one batched predict per step (review
    # rltrain-11); every episode is still reset with its own seed, so the
    # construction seed is irrelevant to the results.
    from reinforcetactics.rl.evaluation import EvalEnvPool

    return EvalEnvPool([build] * cfg.eval.n_eval_envs, use_subprocess=cfg.eval.eval_use_subprocess)


def _read_map_dims(map_file: str) -> tuple[int, int]:
    """Return ``(height, width)`` of the map at ``map_file``.

    Lightweight: parses the CSV with pandas and reads the resulting shape,
    matching :class:`FileIO.load_map`'s post-strip behaviour. Used to
    auto-compute the curriculum-wide max for ``pad_to_size``.
    """
    from reinforcetactics.utils.file_io import FileIO

    map_data = FileIO.load_map(map_file)
    if map_data is None:
        raise FileNotFoundError(f"Could not load map at {map_file!r} for pad-size detection")
    height, width = map_data.shape
    return int(height), int(width)


def _resolve_curriculum_pad_size(cfg: TrainingConfig) -> tuple[int, int] | None:
    """Resolve ``cfg.env.pad_to_size`` for cross-stage obs unification.

    Behaviour:
      - If the user set ``cfg.env.pad_to_size`` explicitly, validate it
        covers every stage's map and return the (possibly normalized) tuple.
      - Else, scan curriculum stages. If maps differ in size *and* the
        curriculum uses ``flat_discrete``, return the per-axis max so the
        env factories can pad observations to a fixed shape across stages.
      - Else (single map size or non-flat_discrete), return ``None``.
    """
    stages = cfg.curriculum.stages
    if not stages:
        return None
    map_files = [s.map_file for s in stages if s.map_file]
    if not map_files:
        return None

    # Stages share a handful of maps: read each file once.
    dims_by_file = {mf: _read_map_dims(mf) for mf in dict.fromkeys(map_files)}
    dims = [dims_by_file[mf] for mf in map_files]
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)

    user_set = cfg.env.pad_to_size
    if user_set is not None:
        ph, pw = int(user_set[0]), int(user_set[1])
        if ph < max_h or pw < max_w:
            raise ValueError(
                f"env.pad_to_size={(ph, pw)} is smaller than the curriculum's "
                f"largest map ({max_h}, {max_w}). Either bump pad_to_size or "
                f"drop the override to let the runner pick it automatically."
            )
        return (ph, pw)

    sizes_differ = any((h, w) != dims[0] for h, w in dims)
    if not sizes_differ:
        return None

    if cfg.env.action_space_type != "flat_discrete":
        # multi_discrete's action space is grid-sized; padding obs alone
        # leaves the action space mismatched. Surface the mismatch loudly
        # rather than silently mis-shaping observations.
        raise ValueError(
            "Curriculum mixes maps of different sizes "
            f"(found {sorted(set(dims))}) but env.action_space_type="
            f"{cfg.env.action_space_type!r}. Padding is only supported with "
            "'flat_discrete'; switch action_space_type or use a single map size."
        )
    return (max_h, max_w)


def resolve_config(cfg: TrainingConfig) -> TrainingConfig:
    """Return a copy of ``cfg`` with every value the runner derives filled in.

    Pure: ``cfg`` is not modified. Resolves the two env settings that the
    YAML alone does not determine, so a record written from the result
    (``resolved_config.yaml``) can rebuild the run's observation and
    action spaces (review rltrain-13):

    - ``env.pad_to_size``: the curriculum-wide observation padding
      (:func:`_resolve_curriculum_pad_size`; reads the stage maps).
    - ``env.flat_action_version``: ``None`` becomes the warm-start
      checkpoint's version (flat_discrete only), so a transplanted policy
      keeps the decode table it was trained on, else the latest version
      (:func:`reinforcetactics.rl.gym_env.resolve_flat_action_version`).
    - ``eval.eval_seats``: ``None`` becomes the seats ``env.agent_seat``
      trains in (:meth:`EvalConfig.resolve_eval_seats`).

    Idempotent: resolving a resolved config changes nothing.
    """
    from reinforcetactics.rl.gym_env import resolve_flat_action_version

    resolved = copy.deepcopy(cfg)
    resolved.env.pad_to_size = _resolve_curriculum_pad_size(cfg)
    resolved.env.flat_action_version = resolve_flat_action_version(
        cfg.env.flat_action_version,
        action_space_type=cfg.env.action_space_type,
        checkpoint_path=cfg.warm_start_path,
    )
    resolved.eval.eval_seats = cfg.eval.resolve_eval_seats(cfg.env)
    return resolved


# ---------------------------------------------------------------------------
# What the curriculum runner reads from a TrainingConfig (review rltrain-9).
# ``check_ignored_config_fields(cfg, CONSUMED_CONFIG_FIELDS, ...)`` reports
# every other field a config sets away from its default; train_bootstrap.py
# warns by default and errors under --strict.
# ---------------------------------------------------------------------------

CONSUMED_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "seed",
        "warm_start_path",
        # Every EnvConfig field _stage_env_kwargs forwards (map_file, opponent
        # and opponent_kwargs come from each stage instead), plus the
        # vec-env knobs of the default train-env factory.
        "env.max_steps",
        "env.max_turns",
        "env.fog_of_war",
        "env.enabled_units",
        "env.action_space_type",
        "env.max_flat_actions",
        "env.flat_action_version",
        "env.max_actions_per_turn",
        "env.reward_config",
        "env.engine_overrides",
        "env.pad_to_size",
        "env.gold_scale",
        "env.turn_scale",
        "env.unit_count_scale",
        "env.agent_seat",
        "env.n_envs",
        "env.use_subprocess",
        # MaskablePPO's constructor kwargs (PPOConfig.as_sb3_kwargs) and the
        # purchase-exploration hook.
        "ppo.learning_rate",
        "ppo.n_steps",
        "ppo.batch_size",
        "ppo.n_epochs",
        "ppo.gamma",
        "ppo.gae_lambda",
        "ppo.clip_range",
        "ppo.ent_coef",
        "ppo.vf_coef",
        "ppo.max_grad_norm",
        "ppo.device",
        "ppo.policy_kwargs",
        "ppo.purchase_explore_eps",
        # Annealed per stage by LRScheduleCallback (review rltrain-8).
        "ppo.lr_schedule",
        "curriculum.*",
        "eval.eval_freq",
        "eval.n_eval_episodes",
        # The rolling <stage>/latest.zip that --resume continues from.
        "eval.checkpoint_freq",
        "eval.seed_offset",
        "eval.resample_eval_seeds",
        "eval.best_eligible_after",
        "eval.eval_deterministic",
        "eval.eval_both_modes",
        "eval.eval_seats",
        "eval.seat_aggregate",
        "eval.n_eval_envs",
        "eval.eval_use_subprocess",
    }
)

# Algorithm labels consistent with what the runner trains.
CONSUMED_ALGORITHMS: tuple[str, ...] = ("maskable_ppo",)

IGNORED_FIELD_HINTS: dict[str, str] = {
    "total_timesteps": "run length is the sum of curriculum.stages[].max_timesteps",
    "algorithm": "the curriculum always trains MaskablePPO",
    "env.map_file": "each curriculum stage sets its own map_file",
    "env.opponent": "each curriculum stage sets its own opponent",
    "env.opponent_kwargs": "set opponent_kwargs on the curriculum stage",
    "ppo.use_action_masking": "the curriculum always trains MaskablePPO with masks",
    "logging.*": "outputs go under --output-dir, TensorBoard logs to <output-dir>/tensorboard",
    "feudal.*": "read only by train_feudal_rl.py",
    "self_play.*": "read only by train_self_play.py / train_feudal_rl.py",
    "alphazero.*": "read only by train_alphazero.py",
}


def _note_flat_version_change(checkpoint: Path, env_version: int | None) -> None:
    """Say so when a warm-started flat_discrete policy continues on a different decode table.

    Only possible when ``env.flat_action_version`` was set explicitly (the
    default follows the checkpoint). The model keeps the env's stamp, so
    checkpoints saved from here on record the table they now play.
    """
    from reinforcetactics.rl.gym_env import checkpoint_flat_action_version

    try:
        ckpt_version = checkpoint_flat_action_version(checkpoint)
    except (OSError, ValueError):
        return
    if env_version is not None and ckpt_version != env_version:
        print(
            f"  note: {checkpoint.name} was trained on flat_action_version {ckpt_version}; "
            f"this run continues it on version {env_version} (env.flat_action_version)"
        )


def _resolve_dotted(path: str) -> Any:
    """Import ``"pkg.module.Attr"`` and return the attribute.

    YAML configs can't carry Python class objects, so
    ``policy_kwargs.features_extractor_class`` is spelled as a dotted
    string and resolved here. Falls back to raising the underlying
    ``ImportError`` / ``AttributeError`` so misconfiguration surfaces
    immediately at model construction.
    """
    import importlib

    if "." not in path:
        raise ValueError(f"features_extractor_class string {path!r} is not dotted; expected 'pkg.module.Attr'")
    module_path, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _resolve_policy_kwargs(policy_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Resolve any string-valued class references inside ``policy_kwargs``.

    Currently handles ``features_extractor_class`` (the only key whose
    SB3 contract requires a class object rather than a primitive).
    Other entries pass through unchanged. ``None`` returns ``None``.
    """
    if not policy_kwargs:
        return policy_kwargs
    resolved = dict(policy_kwargs)
    fe_class = resolved.get("features_extractor_class")
    if isinstance(fe_class, str):
        resolved["features_extractor_class"] = _resolve_dotted(fe_class)
    return resolved


def _default_model_factory(vec_env, cfg: TrainingConfig, output_dir: Path):
    from sb3_contrib import MaskablePPO

    sb3_kwargs = cfg.ppo.as_sb3_kwargs()
    sb3_kwargs["policy_kwargs"] = _resolve_policy_kwargs(sb3_kwargs.get("policy_kwargs"))

    return MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        seed=cfg.seed,
        # SB3 verbose=1 prints rollout/train tables every iteration, which
        # drowns out the curriculum's per-eval WR line. TensorBoard logging
        # is independent of this setting so curves still land on disk.
        verbose=0,
        tensorboard_log=str(output_dir / "tensorboard"),
        **sb3_kwargs,
    )


def _default_model_loader(path: Path, vec_env: Any, cfg: TrainingConfig, output_dir: Path) -> Any:
    """Load a curriculum checkpoint to continue training it (``--resume``).

    ``MaskablePPO.load`` keeps what ``set_parameters`` would not:
    ``num_timesteps`` (so the timestep axis, TensorBoard and the eval
    timeline continue) along with the policy and optimizer state. Bound to
    ``vec_env``, whose spaces must match the checkpoint's.
    """
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(path), env=vec_env, device=cfg.ppo.device, tensorboard_log=str(output_dir / "tensorboard"))


class _MetadataFailures:
    """Best-effort write failures of a run (review rltrain-21).

    Metadata writes (config.json, eval_results.json, the CSVs, the eval
    JSONL, the rolling checkpoint, the manifest, ...) must not abort
    training, but they used to fail in silence. Each failure is now logged
    with its traceback and counted; run_status.json reports the count as
    ``metadata_write_failures``. The manifest carries the count and the
    descriptions, and a resumed run starts from them (:meth:`carry_over`),
    so the count covers every session of the run.
    """

    def __init__(self) -> None:
        self.count = 0
        self.what: list[str] = []

    def carry_over(self, manifest: dict[str, Any] | None) -> None:
        """Start from the failures an earlier session of the run recorded in its manifest."""
        manifest = manifest or {}
        self.count += int(manifest.get("metadata_write_failures", 0) or 0)
        self.what = [str(w) for w in manifest.get("metadata_write_failure_log") or []] + self.what

    def record(self, what: str) -> None:
        self.count += 1
        self.what.append(what)
        logger.warning("could not %s", what, exc_info=True)

    def attempt(self, what: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Run ``fn(*args, **kwargs)``; on an exception record it. Returns whether it succeeded."""
        try:
            fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - best effort by design; recorded and logged
            self.record(what)
            return False
        return True


def _print_policy_summary(model: Any, output_dir: Path | None = None, failures: _MetadataFailures | None = None) -> None:
    """One-time diagnostic: report what the SB3 ``MultiInputPolicy`` actually
    built so we can verify the post-#268 (6, 6, 12) observation isn't being
    flattened past its spatial structure.

    Surfaces:
      - The features-extractor class for each Dict obs key (does ``grid``
        and ``units`` go through a CNN, or just a flatten?). This determines
        whether the policy gets any spatial inductive bias.
      - Trainable parameter count broken down by extractor / mlp / heads,
        so we can compare against the bootstrap.yaml comment's ~150K target
        and notice if the head is the bottleneck vs the extractor.

    When ``output_dir`` is provided, also persists the summary to
    ``output_dir/policy_summary.json`` next to ``final_model.zip`` and
    ``bootstrap_results.csv``. Survives Colab disconnects and lets us
    diff architectures across runs without re-instantiating the model.

    Best-effort: a failure here must not block training, since the
    checkpoint is the load-bearing artifact and this is purely diagnostic.
    """
    try:
        policy = model.policy
        summary: dict[str, Any] = {
            "policy_class": type(policy).__name__,
            "extractors": {},
            "param_counts": {},
            "total_trainable_params": 0,
            "policy_repr": str(policy),
        }
        extractor = getattr(policy, "features_extractor", None)

        print("\n=== Policy network summary ===")
        if extractor is not None:
            summary["features_extractor_class"] = type(extractor).__name__
            extractors = getattr(extractor, "extractors", None)
            if extractors is not None:
                for key, sub in extractors.items():
                    cls_name = type(sub).__name__
                    summary["extractors"][str(key)] = cls_name
                    print(f"  obs[{key!r}] -> {cls_name}")
            else:
                print(f"  features_extractor: {type(extractor).__name__}")
            ext_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
            summary["param_counts"]["features_extractor"] = ext_params
            print(f"  features_extractor params: {ext_params:,}")

        def _count(module_attr: str) -> int | None:
            module = getattr(policy, module_attr, None)
            if module is None:
                return None
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        for name in ("mlp_extractor", "action_net", "value_net"):
            n = _count(name)
            if n is not None:
                summary["param_counts"][name] = n
                print(f"  {name} params: {n:,}")
        total = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        summary["total_trainable_params"] = total
        print(f"  total trainable params: {total:,}")
        print("===============================\n")

        if output_dir is not None:
            try:
                path = Path(output_dir) / "policy_summary.json"
                path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(f"  policy summary written to {path}")
            except Exception:  # noqa: BLE001
                # File write failure shouldn't mask the printed summary.
                if failures is not None:
                    failures.record("write policy_summary.json")
                else:
                    logger.warning("could not write policy_summary.json", exc_info=True)
    except Exception as exc:  # noqa: BLE001
        # A model without the attributes this walks (a test fake, an
        # unusual policy) gets no summary; training goes on.
        logger.warning("could not summarise the policy: %s", exc)


def _curriculum_hash(cfg: TrainingConfig) -> str:
    """Stable short hash of the ordered curriculum structure.

    Surfaces stage-list changes (e.g. the v19 consolidate-stage
    insertion) at a glance: two runs that "look like v18" but differ
    by one stage get different hashes. Keyed on the fields that define
    a stage's identity/difficulty, not cosmetic ones.
    """
    spine = [
        {
            "name": s.name,
            "map_file": s.map_file,
            "opponent": s.opponent,
            "opponent_kwargs": s.opponent_kwargs or {},
            "promotion_win_rate": s.promotion_win_rate,
            "patience": s.patience,
            "max_timesteps": s.max_timesteps,
        }
        for s in cfg.curriculum.stages
    ]
    canon = json.dumps(spine, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]


def _write_run_status(
    output_dir: Path,
    status: str,
    **fields: Any,
) -> None:
    """Write/overwrite ``<output_dir>/run_status.json`` -- the single
    run-level record of HOW a run ended.

    Status semantics (this is the resolution to the historical
    "ended-early vs stalled" ambiguity):

      * file present, status=="completed_curriculum"  -> ran to the end
      * file present, status=="curriculum_stalled"    -> genuine stall
        (a stage exhausted max_timesteps without promoting)
      * file ABSENT                                   -> the run never
        finished cleanly: Colab disconnect / OOM / kill. A killed
        process cannot self-report, so absence *is* the "aborted"
        signal. (This is exactly the Colab-disconnect case that made
        6/33-stage runs indistinguishable from stalls in runs_summary.)

    Per-stage config.json has ``extra.promoted`` but no run-level
    "why did this stop". An interrupted run's progress lives in
    ``run_manifest.json`` instead, so the "absent = aborted" signal above
    still holds. Best effort: never raises (a failure is logged).
    """
    try:
        payload = {
            "status": status,
            "written_at": datetime.now(UTC).isoformat(),
            **fields,
        }
        _write_json_atomically(Path(output_dir) / "run_status.json", payload)
    except Exception:  # noqa: BLE001
        logger.warning("could not write run_status.json", exc_info=True)


def _write_json_atomically(path: Path, payload: Any, *, default: Callable[[Any], Any] = str) -> None:
    """Write ``payload`` as JSON via a ``.partial`` sibling, so ``path`` is never half-written.

    ``--resume`` reads these files (config.json, eval_results.json, the
    manifest): a kill mid-write must leave the previous file or the new one,
    never a torn one it cannot parse.
    """
    from reinforcetactics.cloud.storage import PARTIAL_SUFFIX

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    try:
        partial.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _write_stage_config(
    *,
    stage: CurriculumStage,
    cfg: TrainingConfig,
    stage_dir: Path,
    output_dir: Path,
    promoted: bool,
    best_win_rate: float | None,
    best_checkpoint_timestep: int | None = None,
    best_checkpoint_stage_steps: int | None = None,
    warm_start_info: dict[str, Any] | None = None,
    peak_win_rate: float | None = None,
    stage_summary: dict[str, Any] | None = None,
) -> None:
    """Write ``stage_dir / config.json`` describing the stage's resolved settings.

    ``best_win_rate`` is ``None`` when no eval could claim best_model.zip
    (it used to be written as -1.0; review rltrain-6); ``peak_win_rate`` is
    the highest gate win rate of any eval. ``stage_summary`` (attempts,
    regression-guard restores, the last eval, resume provenance) is merged
    into ``extra``.

    Imported lazily so the bootstrap module stays importable even if the
    optional ``utils.run_config`` dependencies aren't on the path during
    a partial install (e.g. type-checking environments without torch).
    """
    from reinforcetactics.utils.run_config import build_run_config, write_run_config

    ppo_resolved = asdict(cfg.ppo)
    ppo_resolved["ent_coef"] = stage.resolve_ent_coef(cfg.ppo)
    ent_schedule = stage.resolve_ent_coef_schedule()
    if ent_schedule is not None:
        ppo_resolved["ent_coef_schedule"] = {**ent_schedule, "horizon": stage.resolve_horizon(ent_schedule)}
    ppo_resolved["purchase_explore_eps"] = stage.resolve_purchase_explore_eps(cfg.ppo)
    purchase_eps_schedule = stage.resolve_purchase_explore_eps_schedule()
    if purchase_eps_schedule is not None:
        ppo_resolved["purchase_explore_eps_schedule"] = {
            **purchase_eps_schedule,
            "horizon": stage.resolve_horizon(purchase_eps_schedule),
        }
    ppo_resolved["learning_rate"] = stage.resolve_learning_rate(cfg.ppo)
    ppo_resolved["learning_rate_schedule"] = stage.resolve_learning_rate_schedule(cfg.ppo)
    promotion = stage.resolve_promotion(cfg.curriculum)

    # Exactly the kwargs the stage's envs were built with, from the same
    # helper the env factories use. A hand-copied subset used to leave out
    # pad_to_size and the observation scales, so the record could not
    # rebuild the observation space (review rltrain-13).
    env_resolved: dict[str, Any] = copy.deepcopy(_stage_env_kwargs(stage, cfg.env))
    if env_resolved["pad_to_size"] is not None:
        env_resolved["pad_to_size"] = list(env_resolved["pad_to_size"])

    run_config = build_run_config(
        run_type="ppo_bootstrap",
        map_file=stage.map_file,
        opponent=stage.opponent,
        hyperparams=ppo_resolved,
        env_config=env_resolved,
        seed=cfg.seed,
        extra={
            "stage_name": stage.name,
            "promotion_win_rate": stage.promotion_win_rate,
            "patience": stage.patience,
            "max_timesteps": stage.max_timesteps,
            "n_eval_episodes": stage.resolve_n_eval_episodes(cfg.eval),
            "n_envs": cfg.env.n_envs,
            "eval_freq": cfg.eval.eval_freq,
            "promoted": promoted,
            "best_win_rate": best_win_rate,
            "peak_win_rate": peak_win_rate,
            # What the gate measured and how it decided (review rltrain-4 /
            # rltrain-12, critic-gaps-2).
            "eval_deterministic": cfg.eval.eval_deterministic,
            "eval_both_modes": cfg.eval.eval_both_modes,
            "eval_seats": cfg.eval.resolve_eval_seats(cfg.env),
            "seat_aggregate": cfg.eval.seat_aggregate,
            "n_eval_envs": cfg.eval.n_eval_envs,
            "promotion": promotion,
            "max_retries": stage.resolve_max_retries(cfg.curriculum),
            "regression_guard": stage.resolve_regression_guard(cfg.curriculum),
            "anneal_horizon": stage.resolve_horizon(),
            "checkpoint_freq": cfg.eval.checkpoint_freq,
            # Cumulative timestep at which best_model.zip was saved, and the
            # same relative to the stage start. A peak at ~0 stage steps means
            # the handed-forward checkpoint got little stage-specific training
            # (skip-ahead); read alongside the first-eval seize_available_rate
            # / captures_by_type to tell a robust carry-in from a lucky one.
            "best_checkpoint_timestep": best_checkpoint_timestep,
            "best_checkpoint_stage_steps": best_checkpoint_stage_steps,
            "output_dir": str(output_dir),
            "curriculum_hash": _curriculum_hash(cfg),
            "stage_count": len(cfg.curriculum.stages),
            # BC / warm-start provenance: stamped on every stage so an
            # aborted run still leaves a trail showing whether the policy
            # started from a BC checkpoint (and which one). ``used: False``
            # when warm_start_path was unset or the load was skipped.
            "warm_start": warm_start_info or {"used": False, "path": None, "sha256": None},
            **(stage_summary or {}),
        },
    )
    write_run_config(run_config, stage_dir / "config.json")


# Columns emitted to ``bootstrap_results.csv`` -- the run-level analogue of
# ppo_training.ipynb's ``benchmark_results.csv``. The first three identify the
# stage / map / bot the row was eval'd against (so a glob across runs is
# self-describing); the rest are the canonical eval-result fields documented
# in ``evaluate_model``'s return shape. Per-eval breakdown dicts
# (``outcome_reasons``, ``action_counts``, ``reward_components``,
# ``units_built``, ``combat_stats``) intentionally stay in the JSON sibling
# -- they don't flatten cleanly into a CSV row, and the diagnostics charts
# read them straight from the JSON.
#
# ``win_rate`` and the counts describe the eval's gate mode, which since the
# eval-gate change is the stochastic policy by default: ``deterministic``
# says which (empty in a pre-change CSV, whose rows were all greedy), and
# ``win_rate_stochastic`` / ``win_rate_greedy`` carry both modes when both
# were measured. ``gate_win_rate`` is the seat-aggregated rate best_model.zip
# and the peak compare; ``gate_value`` / ``gate_passed`` are what the
# promotion criterion compared with the threshold and whether it passed.
_RESULTS_CSV_COLUMNS = (
    "stage",
    "map_file",
    "opponent",
    "timesteps",
    "win_rate",
    "avg_reward",
    "std_reward",
    "avg_length",
    "std_length",
    "avg_turns",
    "std_turns",
    "wins",
    "losses",
    "draws",
    "episodes",
    "deterministic",
    "win_rate_stochastic",
    "win_rate_greedy",
    "draw_rate",
    "loss_rate",
    "gate_win_rate",
    "gate_criterion",
    "gate_statistic",
    "gate_value",
    "gate_passed",
    "attempt",
    "best_eligible",
)


def _write_results_csv(history: Sequence[dict[str, Any]], csv_path: Path) -> None:
    """Flatten ``history`` into ``bootstrap_results.csv``.

    One row per (stage, eval): the stage's ``map_file`` and ``opponent`` are
    repeated on every row so the CSV is self-describing without joining
    against ``config.json``. Missing fields are written as empty cells (rather
    than raising) so an older eval-result schema or a synthetic test entry
    still produces a parseable CSV.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_RESULTS_CSV_COLUMNS))
        writer.writeheader()
        for stage_entry in history:
            stage_name = stage_entry.get("stage", "")
            map_file = stage_entry.get("map_file", "")
            opponent = stage_entry.get("opponent", "")
            for r in stage_entry.get("results", []) or []:
                row = {col: r.get(col, "") for col in _RESULTS_CSV_COLUMNS}
                row["stage"] = stage_name
                row["map_file"] = map_file
                row["opponent"] = opponent
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Resume support (review rltrain-5 / prior-3)
#
# ``run_manifest.json`` records where an unfinished run is: the stage in
# progress, its attempt, the rolling ``<stage>/latest.zip`` and the gate /
# best-model bookkeeping at that checkpoint, and every stage that finished
# (with the summary its history entry needs). Finished stages are read from
# their ``config.json`` (``extra.promoted``), or from the manifest when that
# best-effort write failed. ``run_status.json`` keeps its meaning (written
# only when a run ends; absent = the run was killed).
# ---------------------------------------------------------------------------

MANIFEST_NAME = "run_manifest.json"
MANIFEST_VERSION = 1

# Config paths a resume may change without ``--force``: they change neither
# what is trained nor what is measured.
RESUME_NONMATERIAL_PATHS: tuple[str, ...] = ("ppo.device", "logging.", "total_timesteps", "algorithm")

# What a resolved_config.yaml written before the eval-gate change meant by
# the fields it does not have: the gate measured the greedy policy, measured
# only that mode, and a stall ended the run. New defaults would silently
# change all three for a resumed pre-change run.
LEGACY_RECORD_DEFAULTS: dict[tuple[str, str], Any] = {
    ("eval", "eval_deterministic"): True,
    ("eval", "eval_both_modes"): False,
    ("curriculum", "max_retries"): 0,
}

# Fields a pre-change record does have, but whose value this runner used to
# ignore: ``ppo.lr_schedule`` was reported as not implemented (the learning
# rate stayed constant), and today ``linear`` anneals every stage's LR to 0.
# A record that lacks any LEGACY_RECORD_DEFAULTS field is pre-change, and
# these get the value that was in effect.
LEGACY_RECORD_OVERRIDES: dict[tuple[str, str], Any] = {
    ("ppo", "lr_schedule"): "constant",
}

# How many failure descriptions the manifest / run_status.json keep (the
# count is exact).
_FAILURE_LOG_LIMIT = 50


class ResumeError(ValueError):
    """A run directory cannot be resumed as asked."""


def _read_json(path: Path) -> Any:
    """``path``'s JSON, or None when it does not exist (a torn file raises ``json.JSONDecodeError``)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _read_json_lenient(path: Path) -> Any:
    """:func:`_read_json`, with an unreadable file (cut short by a kill, say) treated as missing."""
    try:
        return _read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("%s is unreadable; ignoring it", path, exc_info=True)
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSON Lines file; a last line cut short by a kill is skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping an unreadable line of %s", path)
    return rows


def _diff_paths(recorded: Any, current: Any, prefix: str, out: list[str]) -> None:
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(set(recorded) | set(current), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in recorded or key not in current:
                out.append(path)
            else:
                _diff_paths(recorded[key], current[key], path, out)
    elif isinstance(recorded, list) and isinstance(current, list) and all(isinstance(x, dict) for x in recorded + current):
        if len(recorded) != len(current):
            out.append(f"{prefix} ({len(recorded)} -> {len(current)} entries)")
            return
        for i, (a, b) in enumerate(zip(recorded, current, strict=True)):
            _diff_paths(a, b, f"{prefix}[{i}]", out)
    elif recorded != current:
        out.append(prefix)


def load_recorded_config(path: ConfigPath) -> TrainingConfig:
    """Load a run's ``resolved_config.yaml`` with the meaning it had when it was written.

    A record written before the eval-gate change lacks the fields in
    :data:`LEGACY_RECORD_DEFAULTS`; loading it with today's defaults would
    resume that run with a stochastic gate, both eval modes and a stall
    retry it never had. Those absent fields get their historical values
    instead, and so do the fields in :data:`LEGACY_RECORD_OVERRIDES`, which
    such a record has but which had no effect then (a ``ppo.lr_schedule:
    linear`` left the learning rate constant). A record that has every
    field is taken as written.
    """
    import yaml

    from reinforcetactics.rl.config import load_config

    path = Path(path)
    cfg = load_config(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pre_change = False
    for (section, key), value in LEGACY_RECORD_DEFAULTS.items():
        written = raw.get(section) if isinstance(raw, dict) else None
        if not isinstance(written, dict) or key not in written:
            setattr(getattr(cfg, section), key, value)
            pre_change = True
    if pre_change:
        for (section, key), value in LEGACY_RECORD_OVERRIDES.items():
            setattr(getattr(cfg, section), key, value)
    return cfg


def resume_config_differences(cfg: TrainingConfig, run_dir: ConfigPath) -> list[str]:
    """Dotted paths at which ``cfg`` differs materially from the run's ``resolved_config.yaml``.

    Both sides are resolved (:func:`resolve_config`), so a value only one of
    them had filled in -- ``eval.eval_seats: null`` in an older record -- is
    not a difference; the record is read with :func:`load_recorded_config`,
    so a pre-change record that gated on the greedy policy differs from a
    config that gates on the stochastic one. Paths in
    :data:`RESUME_NONMATERIAL_PATHS` (the device, logging, the
    informational ``total_timesteps`` / ``algorithm``) are ignored.

    Raises:
        ResumeError: ``run_dir`` has no ``resolved_config.yaml``.
    """
    recorded_path = Path(run_dir) / "resolved_config.yaml"
    if not recorded_path.is_file():
        raise ResumeError(f"{run_dir} has no resolved_config.yaml: not a train_bootstrap.py run directory")
    recorded = resolve_config(load_recorded_config(recorded_path)).to_dict()
    current = resolve_config(cfg).to_dict()
    diffs: list[str] = []
    _diff_paths(recorded, current, "", diffs)
    return [d for d in diffs if not any(d == p or d.startswith(p) for p in RESUME_NONMATERIAL_PATHS)]


def resume_mismatches(cfg: TrainingConfig, run_dir: ConfigPath) -> list[str]:
    """Why ``cfg`` should not continue the run in ``run_dir`` (empty: it may).

    The :func:`resume_config_differences` against ``resolved_config.yaml``
    when the run has one, and a curriculum that differs from the one
    ``run_manifest.json`` recorded (its ``curriculum_hash``).
    """
    run_dir = Path(run_dir)
    problems: list[str] = []
    if (run_dir / "resolved_config.yaml").is_file():
        problems.extend(resume_config_differences(cfg, run_dir))
    manifest = _read_json_lenient(run_dir / MANIFEST_NAME)
    recorded_hash = manifest.get("curriculum_hash") if isinstance(manifest, dict) else None
    if recorded_hash is not None and recorded_hash != _curriculum_hash(cfg):
        problems.append(f"curriculum (run_manifest.json's curriculum_hash {recorded_hash} != {_curriculum_hash(cfg)})")
    return problems


def _history_entry_from_disk(stage: CurriculumStage, stage_dir: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """A finished stage's ``history`` entry, rebuilt from its config.json (or manifest record) and eval results."""
    results = _read_json_lenient(stage_dir / "eval_results.json")
    if not isinstance(results, list):
        results = _read_jsonl(stage_dir / "eval_results.jsonl")
    return {
        "stage": stage.name,
        "map_file": stage.map_file,
        "opponent": stage.opponent,
        "promoted": bool(extra.get("promoted")),
        "best_win_rate": extra.get("best_win_rate"),
        "peak_win_rate": extra.get("peak_win_rate"),
        "best_checkpoint_timestep": extra.get("best_checkpoint_timestep"),
        "best_checkpoint_stage_steps": extra.get("best_checkpoint_stage_steps"),
        "retries": int(extra.get("retries_used", 0) or 0),
        "attempts": extra.get("attempts") or [],
        "regression_restores": extra.get("regression_restores") or [],
        "gate": extra.get("gate"),
        "results": list(results or []),
        "stage_final_path": str(stage_dir / "stage_final.zip"),
        "from_previous_session": True,
    }


class _ResumePlan:
    """Where :func:`run_curriculum` continues a run (see :func:`_plan_resume`)."""

    def __init__(self, *, start_index: int, history: list[dict[str, Any]], manifest: dict[str, Any] | None) -> None:
        self.start_index = start_index
        self.history = history
        self.manifest = manifest
        self.completed = False
        # Checkpoint the continued model is loaded from (None: build a fresh
        # model, i.e. stage 1 starts over).
        self.model_path: Path | None = None
        # The step count to give the loaded model (a retry restarted from the
        # checkpoint it began from, whose own count is older).
        self.num_timesteps: int | None = None
        # A promoted stage's best_model.zip to load into it afterwards (the
        # between-stage restore the interrupted run had not reached yet).
        self.restore_best_path: Path | None = None
        # The manifest's in-progress entry, for a stage resumed mid-way.
        self.stage_state: dict[str, Any] | None = None
        self.kept_rows: list[dict[str, Any]] = []
        self.prior_attempt_rows: list[dict[str, Any]] = []
        self.warm_start_info: dict[str, Any] | None = None


def _stage_has_output(stage_dir: Path) -> bool:
    return stage_dir.is_dir() and any(p.is_file() for p in stage_dir.rglob("*"))


def _plan_resume(cfg: TrainingConfig, output_dir: Path, *, force: bool = False) -> _ResumePlan:
    """Work out where to continue the run in ``output_dir``.

    Finished stages are the leading stages whose ``config.json`` says
    ``promoted: true`` -- or, when that best-effort write failed or was cut
    short, whose promotion ``run_manifest.json`` recorded (with their
    ``stage_final.zip`` on disk). The first unfinished stage resumes from
    its rolling ``latest.zip`` when the manifest names it (a retry killed
    before its first checkpoint: from the checkpoint the retry started
    from), else from the previous stage's ``stage_final.zip`` (plus its
    ``best_model.zip`` when ``restore_best_checkpoint_between_stages``);
    with neither, stage 1 starts over.

    Raises:
        ResumeError: The run stalled (``run_status.json``, a stage's
            ``config.json`` or the manifest says so), a stage's
            ``config.json`` is unreadable and the manifest does not vouch
            for it, a checkpoint it needs is missing, or -- unless
            ``force`` -- later stages already have output, which resuming
            here would overwrite.
    """
    stages = cfg.curriculum.stages
    status = _read_json_lenient(output_dir / "run_status.json") or {}
    if status.get("status") == "curriculum_stalled":
        raise ResumeError(
            f"{output_dir} stalled at stage '{status.get('stalled_stage')}' (run_status.json); --resume continues "
            "interrupted runs. To try that stage again, start a new run with warm_start_path set to its best_model.zip."
        )
    manifest = _read_json_lenient(output_dir / MANIFEST_NAME)
    if manifest is not None and not isinstance(manifest, dict):
        manifest = None
    finished_by_manifest: dict[str, dict[str, Any]] = {
        entry["stage"]: entry
        for entry in (manifest or {}).get("completed") or []
        if isinstance(entry, dict) and entry.get("stage")
    }
    history: list[dict[str, Any]] = []
    warm: dict[str, Any] | None = (manifest or {}).get("warm_start")
    start = 0
    for index, stage in enumerate(stages):
        stage_dir = output_dir / stage.name
        config_path = stage_dir / "config.json"
        unreadable: Exception | None = None
        try:
            record = _read_json(config_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            record, unreadable = None, exc
        entry = finished_by_manifest.get(stage.name)
        if isinstance(record, dict):
            extra = record.get("extra") or {}
            if not extra.get("promoted"):
                raise ResumeError(
                    f"stage '{stage.name}' of {output_dir} ended without promoting (its config.json says promoted: "
                    "false); --resume continues interrupted runs, not stalled ones"
                )
        elif entry is not None and not entry.get("promoted"):
            raise ResumeError(
                f"stage '{stage.name}' of {output_dir} ended without promoting ({MANIFEST_NAME}); --resume continues "
                "interrupted runs, not stalled ones"
            )
        elif entry is not None and (stage_dir / "stage_final.zip").is_file():
            # The stage finished and promoted, but its config.json was never
            # written (a best-effort write that failed) or was cut short: the
            # manifest's record of it stands in, rather than the run
            # restarting from this stage and overwriting everything after it.
            why = "is unreadable" if unreadable is not None else "is missing"
            print(f"  note: {config_path} {why}; using {MANIFEST_NAME}'s record of the promoted stage '{stage.name}'")
            extra = {**(entry.get("summary") or {}), "promoted": True}
        elif unreadable is not None:
            raise ResumeError(
                f"{config_path} is unreadable ({unreadable}) and {MANIFEST_NAME} has no record of stage "
                f"'{stage.name}' finishing; restore or remove that file to resume"
            )
        else:
            break
        history.append(_history_entry_from_disk(stage, stage_dir, extra))
        warm = warm or extra.get("warm_start")
        start = index + 1
    plan = _ResumePlan(start_index=start, history=history, manifest=manifest)
    plan.warm_start_info = warm
    if start == len(stages):
        plan.completed = True
        return plan

    stage = stages[start]
    stage_dir = output_dir / stage.name
    current = (manifest or {}).get("current") or {}
    # Resuming here retrains this stage and every one after it. Output from a
    # later stage means the run got further than the records above show
    # (a lost record the manifest cannot vouch for): refuse rather than
    # silently overwrite it.
    names = [s.name for s in stages]
    later = [s.name for s in stages[start + 1 :] if _stage_has_output(output_dir / s.name)]
    if current.get("stage") in names[start + 1 :] and current["stage"] not in later:
        later.append(str(current["stage"]))
    if later and not force:
        raise ResumeError(
            f"cannot resume {output_dir} at stage '{stage.name}': later stage(s) {later} already have output, so the "
            f"run got further than its records show (a lost config.json?). Resuming here would retrain and "
            "overwrite them; pass --force (run_curriculum(force=True)) to do that anyway."
        )
    latest = stage_dir / "latest.zip"
    if current.get("stage") == stage.name:
        attempt = int(current.get("attempt", 0))
        rows = _read_jsonl(stage_dir / "eval_results.jsonl")
        prior = [r for r in rows if int(r.get("attempt", 0)) < attempt]
        this_attempt = [r for r in rows if int(r.get("attempt", 0)) == attempt]
        if current.get("latest_timesteps") is not None and latest.is_file():
            latest_ts = int(current["latest_timesteps"])
            last_block = current.get("last_eval_block")
            eval_freq = int(cfg.eval.eval_freq)

            def committed(row: dict[str, Any]) -> bool:
                # An eval is part of the checkpoint when it ran at or before
                # the checkpoint's step and its eval block was committed
                # (a row written by an eval an interrupt cut short is not).
                ts = int(row.get("timesteps", 0))
                return ts <= latest_ts and (last_block is None or ts // eval_freq <= int(last_block))

            plan.prior_attempt_rows = prior
            plan.kept_rows = [r for r in this_attempt if committed(r)]
            dropped = [r for r in this_attempt if not committed(r)]
            eval_state = dict(current.get("eval_state") or {})
            # Evals after the checkpoint are replayed, but a best_model.zip one
            # of them saved is on disk: keep the bookkeeping true to the file.
            for row in dropped:
                gate = row.get("gate_win_rate", row.get("win_rate"))
                if row.get("saved_best") and int(row["timesteps"]) > int(eval_state.get("best_timestep", -1)):
                    eval_state.update(
                        best_win_rate=gate, best_reward=row.get("avg_reward"), best_timestep=int(row["timesteps"])
                    )
                peak = eval_state.get("peak_win_rate")
                if gate is not None and (peak is None or gate > peak):
                    eval_state.update(peak_win_rate=gate, peak_timestep=int(row["timesteps"]))
            plan.stage_state = {**current, "eval_state": eval_state}
            plan.model_path = latest
            return plan
        restored = current.get("restored_from")
        restart_from = stage_dir / Path(str(restored)).name if restored else None
        if attempt > 0 and restart_from is not None and restart_from.is_file():
            # Killed after a retry began but before its first checkpoint: the
            # retry restarts from the checkpoint it started from (keeping the
            # earlier attempts), not the stage from its beginning.
            attempt_start = int(current.get("attempt_start_timesteps", 0))
            plan.prior_attempt_rows = prior
            plan.kept_rows = []
            plan.stage_state = {**current, "latest_timesteps": None, "restart_attempt": True}
            plan.model_path = restart_from
            plan.num_timesteps = attempt_start
            return plan
    if start > 0:
        prev_dir = output_dir / stages[start - 1].name
        final = prev_dir / "stage_final.zip"
        if not final.is_file():
            raise ResumeError(f"cannot resume at stage '{stage.name}': {final} is missing")
        plan.model_path = final
        best = prev_dir / "best_model.zip"
        if cfg.curriculum.restore_best_checkpoint_between_stages and best.is_file():
            plan.restore_best_path = best
    return plan


class _RunManifest:
    """``run_manifest.json``: a run's progress, rewritten atomically at every checkpoint."""

    def __init__(
        self,
        output_dir: Path,
        cfg: TrainingConfig,
        failures: _MetadataFailures,
        previous: dict[str, Any] | None,
        resumed: bool,
    ) -> None:
        self.path = output_dir / MANIFEST_NAME
        self.failures = failures
        prev = previous or {}
        completed = [e for e in prev.get("completed", []) if isinstance(e, dict)] if resumed else []
        self.data: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "curriculum_hash": _curriculum_hash(cfg),
            "stages": [s.name for s in cfg.curriculum.stages],
            "completed": completed,
            "current": prev.get("current") if resumed else None,
            "warm_start": prev.get("warm_start"),
            "resume_count": int(prev.get("resume_count", 0)) + (1 if resumed else 0),
        }

    @property
    def resume_count(self) -> int:
        return int(self.data["resume_count"])

    def write(self) -> None:
        self.data["updated_at"] = datetime.now(UTC).isoformat()
        # Every session's failures, so a resumed run reports them all.
        self.data["metadata_write_failures"] = self.failures.count
        self.data["metadata_write_failure_log"] = self.failures.what[-_FAILURE_LOG_LIMIT:]
        self.failures.attempt(f"write {MANIFEST_NAME}", _write_json_atomically, self.path, self.data)

    def flush_failures(self) -> None:
        """Rewrite the manifest if write failures were recorded since its last write.

        Best effort, like every manifest write. The manifest is how a
        resumed run learns an earlier session's failures
        (:meth:`_MetadataFailures.carry_over`); on an interrupt this is the
        last chance to record the ones since the last checkpoint.
        """
        if self.data.get("metadata_write_failures") != self.failures.count:
            self.write()

    def begin_attempt(self, **fields: Any) -> None:
        self.data["current"] = {**fields, "latest_checkpoint": None, "latest_timesteps": None}
        self.write()

    def checkpoint(self, **fields: Any) -> None:
        if self.data.get("current") is None:
            return
        self.data["current"].update(fields)
        self.write()

    def complete_stage(self, name: str, promoted: bool, summary: dict[str, Any] | None = None) -> None:
        """Record a finished stage (once: a stage finished again replaces its entry)."""
        self.data["completed"] = [e for e in self.data["completed"] if e.get("stage") != name]
        self.data["completed"].append({"stage": name, "promoted": promoted, "summary": summary or {}})
        self.data["current"] = None
        self.write()


def _eval_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """The headline numbers of one eval row (for config.json / run_status.json)."""
    if row is None:
        return None
    keys = (
        "timesteps",
        "attempt",
        "deterministic",
        "win_rate",
        "gate_win_rate",
        "gate_statistic",
        "win_rate_stochastic",
        "win_rate_greedy",
        "wins",
        "losses",
        "draws",
        "episodes",
        "draw_rate",
        "loss_rate",
        "win_rate_by_seat",
    )
    return {k: row[k] for k in keys if k in row}


def _clear_episode_buffers(model: Any) -> None:
    """Empty SB3's rollout episode buffers (review rltrain-18).

    ``learn(reset_num_timesteps=False)`` keeps ``ep_info_buffer`` /
    ``ep_success_buffer``, so a stage's first rollout/ep_rew_mean averaged
    the previous stage's episodes.
    """
    for name in ("ep_info_buffer", "ep_success_buffer"):
        buffer = getattr(model, name, None)
        if buffer is not None and hasattr(buffer, "clear"):
            buffer.clear()


def _copy_file_atomically(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` via a ``.partial`` sibling, so ``target`` is never half-written."""
    from reinforcetactics.cloud.storage import PARTIAL_SUFFIX

    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    try:
        shutil.copyfile(source, partial)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _finish_completed_run(
    cfg: TrainingConfig,
    output_dir: Path,
    plan: _ResumePlan,
    history: list[dict[str, Any]],
    failures: _MetadataFailures,
    warm_start_info: dict[str, Any],
) -> str:
    """Write up a run whose every stage promoted but which stopped before its last records.

    A kill after the last stage's ``config.json`` but before the run's
    ``run_status.json`` (closing SubprocVecEnv workers alone takes seconds)
    left every stage promoted and no ``final_model.zip`` / ``run_status.json``:
    the run read as aborted, had no final model for a self-play warm start,
    and a --resume said there was nothing to do. Nothing is trained here; the
    missing records are rebuilt:

    * ``final_model.zip`` from the last stage's ``best_model.zip`` when
      ``restore_best_checkpoint_between_stages`` is set and it exists (the
      policy the run would have saved; the zip's step counter is that
      eval's), else from its ``stage_final.zip`` (the file the run would
      have saved);
    * the last stage's ``eval_results.json`` if missing, ``bootstrap_results.csv``,
      the manifest's record of the finished stages, and any leftover
      ``latest.zip`` / ``stage_start.zip``;
    * then ``run_status.json`` as ``completed_curriculum``, counting this
      session as a resume (``finished_on_resume``), with the write failures
      of every session.

    A run whose ``run_status.json`` already says ``completed_curriculum``
    only gets a missing ``final_model.zip`` back.

    Returns:
        The path of ``final_model.zip``.

    Raises:
        ResumeError: ``final_model.zip`` is missing and the last stage has no
            checkpoint to rebuild it from.
    """
    final = output_dir / "final_model.zip"
    status = _read_json_lenient(output_dir / "run_status.json")
    finished = isinstance(status, dict) and status.get("status") == "completed_curriculum"
    if finished and final.is_file():
        print(f"  {output_dir}: every stage already promoted; nothing to resume")
        return str(final)
    missing = [name for name, there in (("final_model.zip", final.is_file()), ("run_status.json", finished)) if not there]
    print(f"  {output_dir}: every stage already promoted, but the run stopped before writing {' and '.join(missing)}")
    stages = cfg.curriculum.stages
    last_dir = output_dir / stages[-1].name
    if not final.is_file():
        candidates = [last_dir / "stage_final.zip"]
        if cfg.curriculum.restore_best_checkpoint_between_stages:
            candidates.insert(0, last_dir / "best_model.zip")
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ResumeError(f"cannot write {final}: none of {[str(p) for p in candidates]} exists to rebuild it from")
        _copy_file_atomically(source, final)
        print(f"  wrote {final} from {source}")
    if finished:
        return str(final)

    last = history[-1] if history else None
    if last is not None and last.get("results") and not (last_dir / "eval_results.json").is_file():
        failures.attempt(
            f"write {stages[-1].name}/eval_results.json",
            _write_json_atomically,
            last_dir / "eval_results.json",
            last["results"],
            default=float,
        )
    failures.attempt("write bootstrap_results.csv", _write_results_csv, history, output_dir / "bootstrap_results.csv")
    for stage in stages:
        for leftover in ("latest.zip", "stage_start.zip"):
            try:
                (output_dir / stage.name / leftover).unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove %s", output_dir / stage.name / leftover, exc_info=True)
    manifest = _RunManifest(output_dir, cfg, failures, plan.manifest, resumed=True)
    recorded = {e.get("stage") for e in manifest.data["completed"]}
    for entry in history:
        if entry["stage"] in recorded:
            continue
        summary = {
            key: entry.get(key)
            for key in (
                "best_win_rate",
                "peak_win_rate",
                "best_checkpoint_timestep",
                "best_checkpoint_stage_steps",
                "attempts",
                "regression_restores",
                "gate",
            )
        }
        summary.update(retries_used=int(entry.get("retries", 0) or 0), warm_start=warm_start_info)
        manifest.data["completed"].append({"stage": entry["stage"], "promoted": True, "summary": summary})
    manifest.data["current"] = None
    manifest.data["warm_start"] = warm_start_info
    manifest.write()
    _write_run_status(
        output_dir,
        "completed_curriculum",
        stages_completed=len(stages),
        stage_count=len(stages),
        curriculum_hash=_curriculum_hash(cfg),
        warm_start=warm_start_info,
        retries={h["stage"]: int(h.get("retries", 0) or 0) for h in history},
        resume_count=manifest.resume_count,
        metadata_write_failures=failures.count,
        metadata_write_failure_log=failures.what[-_FAILURE_LOG_LIMIT:],
        finished_on_resume=True,
    )
    return str(final)


def run_curriculum(
    cfg: TrainingConfig,
    output_dir: ConfigPath,
    *,
    train_env_factory: Callable[[CurriculumStage, TrainingConfig], Any] | None = None,
    eval_env_factory: Callable[[CurriculumStage, TrainingConfig], Any] | None = None,
    model_factory: Callable[..., Any] | None = None,
    model_loader: Callable[..., Any] | None = None,
    progress_bar: bool = False,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Train through every stage in ``cfg.curriculum.stages``.

    Args:
        cfg: Validated :class:`TrainingConfig` with a non-empty
            ``cfg.curriculum.stages``.
        output_dir: Root directory for stage subfolders, tensorboard logs,
            and the final checkpoint.
        train_env_factory: ``(stage, cfg) -> vec_env``. Defaults to
            ``make_maskable_vec_env`` with the resolved per-stage env.
        eval_env_factory: ``(stage, cfg) -> env``. Defaults to
            ``make_maskable_env`` with the resolved per-stage env (an
            :class:`~reinforcetactics.rl.evaluation.EvalEnvPool` of
            ``eval.n_eval_envs`` of them when that is > 1).
        model_factory: ``(vec_env, cfg, output_dir) -> model``. Called once
            for the first stage; later stages reuse the model via
            ``model.set_env(...)``. Defaults to MaskablePPO.
        model_loader: ``(path, vec_env, cfg, output_dir) -> model``, used
            by ``resume`` to load the checkpoint it continues from.
            Defaults to ``MaskablePPO.load`` (keeps ``num_timesteps``).
        progress_bar: Forwarded to ``model.learn()``.
        resume: Continue the run already in ``output_dir``: skip the stages
            whose ``config.json`` says they promoted and continue the
            interrupted one from its rolling ``latest.zip`` with the rest of
            its budget (see :func:`_plan_resume`). Restored: the weights and
            optimizer state, ``num_timesteps`` (TensorBoard continues its
            curve), the stage's eval timeline up to the checkpoint, the
            promotion streak / rolling window, the best-model and peak
            bookkeeping, the attempt number, regression-guard restores, and
            the position of the stage's entropy / LR / purchase-ε
            schedules and min-timesteps window. Not restored: the rollout
            buffer and the envs' in-flight episodes (a new rollout starts),
            the RNG streams, and the in-memory train_metrics records (the
            CSV on disk keeps its rows). A run whose every stage promoted
            returns its history without training; if it was killed before
            its ``final_model.zip`` / ``run_status.json``, they are written
            now (:func:`_finish_completed_run`). ``cfg`` must match the
            run's: a material difference from its ``resolved_config.yaml``
            or its manifest's curriculum (:func:`resume_mismatches`) is
            refused.
        force: With ``resume``, continue despite such a difference, or
            although later stages already have output (see
            :func:`_plan_resume`). The CLI's ``--force``.

    Returns:
        Dict with keys ``model``, ``history`` (list of per-stage dicts),
        ``final_model_path``, ``metrics_callback``.

    Raises:
        CurriculumStalled: if a stage's last attempt hits its
            ``max_timesteps`` without the promotion criterion.
        ResumeError: ``resume`` and the run cannot be continued (or, without
            ``force``, not with this ``cfg``).
    """
    from reinforcetactics.rl.callbacks import (
        EntropyScheduleCallback,
        LRScheduleCallback,
        PeriodicEvalCallback,
        PromotionCallback,
        RegressionGuardCallback,
        RollingCheckpointCallback,
        TrainingMetricsCallback,
        save_model_atomically,
        set_learning_rate,
    )
    from reinforcetactics.rl.purchase_exploration import (
        PurchaseExploreScheduleCallback,
        install_purchase_explore_hook,
    )

    # check_files: a missing warm_start_path fails here, before any env or
    # model is built, rather than after the first stage's envs are up.
    cfg.validate(check_files=True)
    if not cfg.curriculum.stages:
        raise ValueError("cfg.curriculum.stages is empty; nothing to run")
    # Say which config fields this runner does not read (a warning; the
    # train_bootstrap.py CLI reports them itself, before any output, and
    # makes them an error under --strict). Notebooks call run_curriculum
    # directly and used to get no report at all (review rltrain-9).
    check_ignored_config_fields(
        cfg,
        CONSUMED_CONFIG_FIELDS,
        entry_point="run_curriculum",
        algorithms=CONSUMED_ALGORITHMS,
        hints=IGNORED_FIELD_HINTS,
        strict_hint="train_bootstrap.py --strict makes this an error.",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume and not force:
        # Mixing two curricula (or two gates) in one stage's timeline and
        # promotion streak is never what a resume means.
        mismatches = resume_mismatches(cfg, output_dir)
        if mismatches:
            listed = "\n".join(f"  {m}" for m in mismatches[:40])
            raise ResumeError(
                f"cannot resume {output_dir} with this config; it differs from the run's at:\n{listed}\n"
                "Resume with the run's own config, or pass force=True (--force) to continue with this one."
            )

    # Resolve cross-stage pad_to_size (required for the policy to be
    # reusable across stages when the curriculum mixes map sizes: set_env in
    # bootstrap rejects shape mismatches), the flat_discrete table version
    # and the eval seats before instantiating any env. Written back into
    # ``cfg`` so callers that build stage envs after the run (sanity eval,
    # replays) get the same values training used.
    resolved = resolve_config(cfg)
    if resolved.env.pad_to_size is not None and cfg.env.pad_to_size is None:
        print(
            f"  curriculum spans multiple map sizes; auto-padding observations to {resolved.env.pad_to_size} (height, width)"
        )
    cfg.env.pad_to_size = resolved.env.pad_to_size
    cfg.env.flat_action_version = resolved.env.flat_action_version
    cfg.eval.eval_seats = resolved.eval.eval_seats

    train_env_factory = train_env_factory or _default_train_env_factory
    eval_env_factory = eval_env_factory or _default_eval_env_factory
    model_factory = model_factory or _default_model_factory
    model_loader = model_loader or _default_model_loader

    failures = _MetadataFailures()
    # ``csv_path`` makes the optimizer diagnostics (approx_kl, clip_fraction,
    # explained_variance, value_loss, ent_coef, ...) survive the process:
    # each committed record is appended to ``train_metrics.csv`` immediately,
    # so a Colab disconnect no longer erases the only evidence of *how* a
    # stage's policy collapsed. One continuous file across all stages; rows
    # carry the stage name via ``metrics_callback.context`` (set below).
    metrics_callback = TrainingMetricsCallback(csv_path=output_dir / "train_metrics.csv")
    metrics_callback.on_write_failure = failures.record
    history: list[dict[str, Any]] = []
    model = None
    # Provenance of the warm-start checkpoint, filled in the first time
    # the model is built. Captured here (rather than re-derived at each
    # _write_stage_config call) so the sha256 is computed once and the
    # same dict is stamped onto every stage's config.json + the final
    # run_status.json. ``used`` distinguishes "no warm_start_path set"
    # from "warm_start_path set but load skipped" -- if either grows a
    # branch, the metadata stays accurate without code changes here.
    warm_start_info: dict[str, Any] = {"used": False, "path": None, "sha256": None}

    plan = _plan_resume(cfg, output_dir, force=force) if resume else None
    if plan is not None:
        failures.carry_over(plan.manifest)
        history = list(plan.history)
        if plan.warm_start_info:
            warm_start_info = dict(plan.warm_start_info)
        if plan.completed:
            final_model_path = _finish_completed_run(cfg, output_dir, plan, history, failures, warm_start_info)
            return {
                "model": None,
                "history": history,
                "final_model_path": final_model_path,
                "metrics_callback": metrics_callback,
                "resumed": True,
            }
    manifest = _RunManifest(output_dir, cfg, failures, plan.manifest if plan else None, resumed=plan is not None)
    if plan is not None:
        where = cfg.curriculum.stages[plan.start_index].name
        how = f"from {plan.model_path}" if plan.model_path is not None else "from scratch"
        print(f"  resuming {output_dir} at stage '{where}' ({plan.start_index} stage(s) already promoted) {how}")
    # True once any stage has set the learning rate: later stages then set
    # their own (the base LR when they configure none) instead of
    # inheriting an earlier stage's schedule. A model loaded to resume
    # carries whatever LR its checkpoint was saved with, so it counts.
    lr_touched = plan is not None and plan.model_path is not None
    retries_by_stage: dict[str, int] = {h["stage"]: int(h.get("retries", 0) or 0) for h in history}

    for stage_index, stage in enumerate(cfg.curriculum.stages):
        if plan is not None and stage_index < plan.start_index:
            continue
        stage_dir = output_dir / stage.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        vec_env = train_env_factory(stage, cfg)
        eval_env = eval_env_factory(stage, cfg)
        resume_state = plan.stage_state if plan is not None and stage_index == plan.start_index else None

        if model is None:
            if plan is not None and plan.model_path is not None:
                model = model_loader(plan.model_path, vec_env, cfg, output_dir)
                if plan.num_timesteps is not None:
                    # A retry restarted from the checkpoint it began from: the
                    # step count is the retry's start, not that checkpoint's.
                    model.num_timesteps = plan.num_timesteps
                if plan.restore_best_path is not None:
                    print(f"  restoring {plan.restore_best_path} before '{stage.name}'")
                    model.set_parameters(str(plan.restore_best_path), exact_match=True)
            else:
                model = model_factory(vec_env, cfg, output_dir)
                # Warm-start from a checkpoint before any training. Loaded
                # after the model is built (so the env / spaces are already
                # bound) but before the policy-summary print and the
                # exploration hook, so the summary reflects the loaded
                # weights and the hook wraps the warm-started policy.
                # ``set_parameters`` keeps the freshly-constructed model's
                # env, schedules, and callbacks intact and only overwrites
                # the policy + optimizer tensors -- exactly the semantics we
                # want for transplanting a prior run's policy into a new
                # curriculum. A space mismatch surfaces here with an
                # actionable SB3 error rather than a silent mis-load.
                if cfg.warm_start_path:
                    # Existence was checked by validate(check_files=True) above.
                    warm_path = Path(cfg.warm_start_path)
                    # sha256 of the checkpoint bytes -- so per-stage config.json
                    # and run_status.json can distinguish two runs that loaded
                    # *different* BC builds from the same path (e.g. timestamped
                    # checkpoints rotated in place between runs).
                    warm_sha = hashlib.sha256(warm_path.read_bytes()).hexdigest()
                    print(f"  warm-starting policy from {warm_path}  sha256={warm_sha[:12]}")
                    model.set_parameters(str(warm_path), exact_match=True)
                    if cfg.env.action_space_type == "flat_discrete":
                        _note_flat_version_change(warm_path, cfg.env.flat_action_version)
                    warm_start_info = {
                        "used": True,
                        "path": str(warm_path),
                        "sha256": warm_sha,
                    }
            manifest.data["warm_start"] = warm_start_info
            _print_policy_summary(model, output_dir, failures)
            # Install the purchase-exploration hook once on the model.
            # Wrapping is idempotent and the hook short-circuits when
            # ``purchase_explore_eps <= 0``, so it's safe to install
            # unconditionally; per-attempt code below sets the live ε
            # attribute (and, optionally, attaches a schedule).
            install_purchase_explore_hook(
                model,
                eps=cfg.ppo.purchase_explore_eps,
                seed=cfg.seed,
            )
        else:
            # Reusing the model across stages requires matching spaces. SB3's
            # ``set_env`` will accept the swap and only fail later at rollout
            # time with an opaque shape error; check up front so a curriculum
            # that mixes maps of different sizes / unit sets fails on the
            # offending stage with an actionable message. The hasattr guards
            # let test fakes without space attributes skip the check.
            model_obs_space = getattr(model, "observation_space", None)
            env_obs_space = getattr(vec_env, "observation_space", None)
            if model_obs_space is not None and env_obs_space is not None and env_obs_space != model_obs_space:
                raise ValueError(
                    f"Stage '{stage.name}' observation space {env_obs_space} "
                    f"does not match the model's {model_obs_space}. "
                    "Curriculum stages must share observation shapes (same map size, "
                    "same enabled_units, same action_space_type)."
                )
            model_action_space = getattr(model, "action_space", None)
            env_action_space = getattr(vec_env, "action_space", None)
            if model_action_space is not None and env_action_space is not None and env_action_space != model_action_space:
                raise ValueError(
                    f"Stage '{stage.name}' action space {env_action_space} "
                    f"does not match the model's {model_action_space}. "
                    "Curriculum stages must share action shapes."
                )
            model.set_env(vec_env)

        ent_coef = stage.resolve_ent_coef(cfg.ppo)
        ent_schedule = stage.resolve_ent_coef_schedule()
        purchase_eps = stage.resolve_purchase_explore_eps(cfg.ppo)
        purchase_eps_schedule = stage.resolve_purchase_explore_eps_schedule()
        lr_schedule = stage.resolve_learning_rate_schedule(cfg.ppo)
        promotion = stage.resolve_promotion(cfg.curriculum)
        max_retries = stage.resolve_max_retries(cfg.curriculum)
        guard = stage.resolve_regression_guard(cfg.curriculum)
        eval_seats = cfg.eval.resolve_eval_seats(cfg.env)
        _print_stage_banner(stage, cfg, ent_coef, ent_schedule, purchase_eps, purchase_eps_schedule, lr_schedule, promotion)

        # Cumulative timestep entering the stage. ``num_timesteps`` is global
        # (reset_num_timesteps=False), so subtracting this from the best
        # checkpoint's timestep gives stage-relative "how much did this stage
        # actually train before its peak" -- the skip-ahead diagnostic.
        if resume_state is not None:
            stage_start_timesteps = int(resume_state.get("stage_start_timesteps", model.num_timesteps))
            attempt = int(resume_state.get("attempt", 0))
            attempts: list[dict[str, Any]] = list(resume_state.get("attempts") or [])
            stage_results: list[dict[str, Any]] = list(plan.prior_attempt_rows) if plan is not None else []
            best_state: dict[str, Any] | None = resume_state.get("eval_state")
            guard_restores: list[dict[str, Any]] = list(resume_state.get("regression_restores") or [])
            restored_from: str | None = resume_state.get("restored_from")
            # The gate record the attempt began with; a checkpoint's
            # promotion_state carries the newer one.
            gate_record: dict[str, Any] | None = resume_state.get("gate_record")
            # Only this stage's own rows are rewritten below; the timeline
            # after the checkpoint is replayed.
            kept = [*stage_results, *(plan.kept_rows if plan is not None else [])]
            failures.attempt("rewrite eval_results.jsonl", _rewrite_jsonl, stage_dir / "eval_results.jsonl", kept)
        else:
            stage_start_timesteps = int(model.num_timesteps)
            attempt = 0
            attempts = []
            stage_results = []
            best_state = None
            guard_restores = []
            restored_from = None
            gate_record = None
            # A stage starting afresh begins an empty timeline (a resumed run
            # restarting a stage it never checkpointed would otherwise
            # append to the aborted attempt's rows), and without the aborted
            # attempt's best_model.zip: the between-stages restore and a
            # retry load that file whenever it exists, though no eval of
            # this attempt chose it.
            (stage_dir / "eval_results.jsonl").unlink(missing_ok=True)
            (stage_dir / "best_model.zip").unlink(missing_ok=True)
            if max_retries > 0:
                # The weights a retry falls back to when the stage never
                # saves a best_model.zip.
                failures.attempt("save stage_start.zip", save_model_atomically, model, stage_dir / "stage_start.zip")

        # Stamp the stage name onto every train-metrics record committed
        # during this learn() so train_metrics.csv rows are joinable
        # against bootstrap_results.csv without reconstructing stage
        # boundaries from timestep ranges.
        metrics_callback.context = stage.name
        promoted = False
        eval_cb: Any = None
        resume_pending = resume_state is not None
        while True:
            # A retry killed before its first checkpoint restarts from the
            # weights it began with: a fresh attempt in all but its number.
            restart = resume_pending and bool((resume_state or {}).get("restart_attempt"))
            if resume_pending:
                assert resume_state is not None and plan is not None
                attempt_start = int(resume_state.get("attempt_start_timesteps", stage_start_timesteps))
                kept_rows = [] if restart else list(plan.kept_rows)
                promotion_state = None if restart else resume_state.get("promotion_state")
                last_eval_block = None if restart else resume_state.get("last_eval_block")
                resumed_at = resume_state.get("latest_timesteps")
                latest_timesteps: int | None = int(resumed_at) if resumed_at is not None else attempt_start
                pinned_start: int | None = attempt_start
            else:
                attempt_start = int(model.num_timesteps)
                kept_rows = []
                promotion_state = None
                last_eval_block = None
                latest_timesteps = None
                pinned_start = None
            budget = max(0, int(stage.max_timesteps) - (int(model.num_timesteps) - attempt_start))

            # Per-attempt knobs, re-applied on a retry (the entropy start
            # re-warms exploration). SB3 reads ``ent_coef`` fresh inside
            # every ``train()`` step, so live mutation works without
            # rebuilding the model; a schedule callback takes over from the
            # start value.
            if hasattr(model, "ent_coef"):
                model.ent_coef = ent_coef
            # Per-stage purchase-exploration ε. Hook installation happened
            # once, so this just updates the value the wrapper reads.
            setattr(model, "purchase_explore_eps", purchase_eps)
            if lr_schedule is not None:
                set_learning_rate(model, lr_schedule["start"])
                lr_touched = True
            elif lr_touched:
                set_learning_rate(model, cfg.ppo.learning_rate)

            eval_cb = PeriodicEvalCallback(
                eval_env=eval_env,
                eval_freq=cfg.eval.eval_freq,
                # The stage's override, else eval.n_eval_episodes (which this
                # runner used to ignore in favour of a hidden stage default).
                n_eval_episodes=stage.resolve_n_eval_episodes(cfg.eval),
                # Seed eval episodes far above the training envs' range so a
                # given (eval_block, episode_idx) reproduces across runs and
                # never collides with a training rollout's seed.
                eval_seed_base=cfg.seed + cfg.eval.seed_offset,
                save_dir=stage_dir,
                # Dump per-step JSONL traces for any eval episode that hits
                # the env step limit without ending (the stalling signature
                # we added ``env.max_actions_per_turn`` to defend against).
                # Healthy episodes leave no files on disk; truncated ones
                # land under ``<stage_dir>/traces/eval_<timesteps>/`` so it's
                # easy to grep for which actions the policy spammed.
                trace_dir=stage_dir / "traces",
                # Append each eval result as it happens so a mid-stage kill
                # (Colab disconnect) leaves the stage's eval timeline on disk.
                # ``eval_results.json`` below is still written at stage end.
                results_jsonl_path=stage_dir / "eval_results.jsonl",
                # Replay one fixed problem set for every eval in this stage, so
                # PromotionCallback's consecutive-crossings gate and the best-model
                # argmax compare like with like.
                resample_eval_seeds=cfg.eval.resample_eval_seeds,
                # The first _on_step of a stage always fires an eval (block index
                # is derived from the cumulative counter), which measures the
                # carry-in policy. Keep it out of the best-checkpoint race so
                # restore_best_checkpoint_between_stages can't rewind this stage's
                # own training. Defaults to one eval interval.
                best_eligible_after=(
                    cfg.eval.eval_freq if cfg.eval.best_eligible_after is None else cfg.eval.best_eligible_after
                ),
                # What the gate measures (review rltrain-4, critic-gaps-2).
                deterministic=cfg.eval.eval_deterministic,
                eval_both_modes=cfg.eval.eval_both_modes,
                seats=eval_seats,
                seat_aggregate=cfg.eval.seat_aggregate,
                start_step=pinned_start,
                last_eval_block=last_eval_block,
                # A retry keeps the stage's best so far: best_model.zip is only
                # replaced by a better eval.
                best_state=best_state,
                row_extra={"attempt": attempt},
                on_write_failure=failures.record,
            )
            eval_cb.results.extend(kept_rows)
            promote_cb = PromotionCallback(
                eval_callback=eval_cb,
                threshold=promotion["threshold"],
                patience=promotion["patience"],
                min_timesteps=stage.min_timesteps_before_promotion,
                criterion=promotion["criterion"],
                rolling_k=promotion["rolling_k"],
                confidence=promotion["confidence"],
                score=promotion["score"],
                seat_aggregate=cfg.eval.seat_aggregate,
                start_step=pinned_start,
                # A resume restores the streak / window / promotion; a retry
                # carries only the gate record (what a stall reports).
                initial_state=promotion_state if resume_pending else None,
                record=gate_record,
            )
            callbacks: list[Any] = [metrics_callback, eval_cb, promote_cb]
            guard_cb = None
            if guard["evals"] > 0:
                guard_cb = RegressionGuardCallback(eval_cb, guard["evals"], guard["drop"], stage_dir / "best_model.zip")
                callbacks.append(guard_cb)

            def _on_checkpoint(
                timesteps: int,
                *,
                _eval_cb: Any = eval_cb,
                _promote_cb: Any = promote_cb,
                _guard_cb: Any = guard_cb,
                _prior_restores: list[dict[str, Any]] = list(guard_restores),
            ) -> None:
                manifest.checkpoint(
                    latest_checkpoint=f"{stage.name}/latest.zip",
                    latest_timesteps=int(timesteps),
                    last_eval_block=int(_eval_cb._last_eval_block),
                    eval_state=_eval_cb.best_state(),
                    promotion_state=_promote_cb.state_dict(),
                    regression_restores=_prior_restores + (list(_guard_cb.restores) if _guard_cb is not None else []),
                    evals_at_checkpoint=len(_eval_cb.results),
                )

            # Last in the list: a checkpoint lands after any eval / promotion
            # decision of the same step, so the manifest snapshot matches.
            rolling_cb = RollingCheckpointCallback(
                stage_dir / "latest.zip",
                cfg.eval.checkpoint_freq,
                on_save=_on_checkpoint,
                on_error=failures.record,
                start_step=latest_timesteps,
            )
            if ent_schedule is not None:
                # Anneals over the stage budget (or the stage's
                # anneal_horizon); if the stage promotes early ``learn()``
                # returns before the callback hits ``end``, which is fine --
                # we wanted the cooling to have *been available* during the
                # run-up.
                callbacks.append(
                    EntropyScheduleCallback(
                        start=ent_schedule["start"],
                        end=ent_schedule["end"],
                        total_timesteps=stage.resolve_horizon(ent_schedule),
                        schedule=ent_schedule["schedule"],
                        start_step=pinned_start,
                    )
                )
            if purchase_eps_schedule is not None:
                callbacks.append(
                    PurchaseExploreScheduleCallback(
                        start=purchase_eps_schedule["start"],
                        end=purchase_eps_schedule["end"],
                        total_timesteps=stage.resolve_horizon(purchase_eps_schedule),
                        schedule=purchase_eps_schedule["schedule"],
                        start_step=pinned_start,
                    )
                )
            if lr_schedule is not None and lr_schedule["schedule"] != "constant":
                callbacks.append(
                    LRScheduleCallback(
                        start=lr_schedule["start"],
                        end=lr_schedule["end"],
                        total_timesteps=lr_schedule["horizon"],
                        schedule=lr_schedule["schedule"],
                        start_step=pinned_start,
                    )
                )
            callbacks.append(rolling_cb)

            if not resume_pending or restart:
                # Everything a resume needs to restart this attempt from the
                # weights it begins with (``restored_from`` for a retry) if it
                # is killed before its first checkpoint: the best / peak and
                # gate record carried from earlier attempts included.
                manifest.begin_attempt(
                    stage=stage.name,
                    index=stage_index,
                    stage_start_timesteps=stage_start_timesteps,
                    attempt=attempt,
                    attempt_start_timesteps=attempt_start,
                    attempts=list(attempts),
                    restored_from=restored_from,
                    eval_state=best_state,
                    gate_record=gate_record,
                    regression_restores=list(guard_restores),
                )
                # Checkpoint the attempt's starting weights, so a kill before
                # the first rolling save still resumes this attempt.
                if failures.attempt("save latest.zip", save_model_atomically, model, stage_dir / "latest.zip"):
                    _on_checkpoint(attempt_start)
            if resume_pending and promote_cb.promoted:
                # The checkpoint was taken on the promoting eval, and the run
                # died before the stage was written up: finish it, don't
                # train it again.
                print(f"  resuming '{stage.name}': it promoted at {int(model.num_timesteps):,} steps; finishing the stage")
            elif resume_pending and (attempt > 0 or budget < stage.max_timesteps):
                print(
                    f"  resuming '{stage.name}' attempt {attempt + 1} at {int(model.num_timesteps):,} "
                    f"({budget:,} of {stage.max_timesteps:,} steps left)"
                )
            _clear_episode_buffers(model)
            # Whether this attempt trained (a resumed stage that had already
            # promoted, or had no budget left, does not).
            learned = budget > 0 and not promote_cb.promoted
            try:
                if learned:
                    model.learn(
                        total_timesteps=budget,
                        callback=callbacks,
                        reset_num_timesteps=False,
                        progress_bar=progress_bar,
                    )
            except (KeyboardInterrupt, SystemExit):
                # SIGTERM reaches here as SystemExit (train_bootstrap.py). A
                # last checkpoint on the way out, so --resume loses nothing.
                try:
                    failures.attempt("save latest.zip on interrupt", rolling_cb.save_now)
                finally:
                    # A failure since the manifest's last write (that save's,
                    # which skips the manifest update a successful save
                    # makes) reaches the resumed run only through the manifest.
                    manifest.flush_failures()
                raise
            resume_pending = False
            if guard_cb is not None:
                guard_restores.extend(guard_cb.restores)
            best_state = eval_cb.best_state()
            gate_record = promote_cb.record()
            attempt_rows = list(eval_cb.results)
            stage_results.extend(attempt_rows)
            gates = [r.get("gate_win_rate", r.get("win_rate")) for r in attempt_rows]
            gates = [g for g in gates if g is not None]
            attempts.append(
                {
                    "attempt": attempt,
                    "start_timesteps": attempt_start,
                    "end_timesteps": int(model.num_timesteps),
                    "promoted": bool(promote_cb.promoted),
                    "evals": len(attempt_rows),
                    "peak_win_rate": max(gates) if gates else None,
                    "restored_from": restored_from,
                }
            )
            if promote_cb.promoted:
                promoted = True
                break
            if attempt >= max_retries:
                break
            # Stall with retries left (review rltrain-7): restore the stage's
            # best checkpoint (else its starting weights) and give it its
            # budget again with fresh callbacks.
            attempt += 1
            restore = next((p for p in (stage_dir / "best_model.zip", stage_dir / "stage_start.zip") if p.is_file()), None)
            if restore is not None:
                model.set_parameters(str(restore), exact_match=True)
            restored_from = str(restore) if restore is not None else None
            why = stall_verdict(eval_cb.peak_win_rate, float(promotion["threshold"]), int(promotion["patience"]), gate_record)
            print(
                f"  [retry] '{stage.name}' stalled ({why}); "
                f"attempt {attempt + 1}/{max_retries + 1} from {restore.name if restore is not None else 'the current weights'}"
            )

        if promoted and learned and rolling_cb._last_save != int(model.num_timesteps):
            # The promotion is on disk before the stage-end writes start: a
            # kill during them resumes into finishing the stage (the
            # manifest's promotion_state says promoted) rather than
            # retraining it from an earlier checkpoint.
            failures.attempt("save latest.zip at promotion", rolling_cb.save_now)

        final_state = best_state or {}
        best_win_rate = final_state.get("best_win_rate")
        peak_win_rate = final_state.get("peak_win_rate")
        # How far into the stage the saved peak was. ``best_timestep`` is
        # cumulative; ``best_stage_steps`` is stage-relative (None if no best
        # was ever saved this stage). A peak at ~0 stage-relative steps means
        # the carry-in policy was already at the bar and the stage did little
        # of its own learning before promoting (skip-ahead).
        raw_best_ts = int(final_state.get("best_timestep", -1) or -1)
        best_timestep = raw_best_ts if raw_best_ts >= 0 else None
        best_stage_steps = (best_timestep - stage_start_timesteps) if best_timestep is not None else None

        stage_final = stage_dir / "stage_final.zip"
        save_model_atomically(model, stage_final)

        stage_summary = {
            "retries_used": attempt,
            "attempts": attempts,
            "regression_restores": guard_restores,
            "stage_start_timesteps": stage_start_timesteps,
            "stage_end_timesteps": int(model.num_timesteps),
            "last_eval": _eval_summary(stage_results[-1] if stage_results else None),
            # What the promotion criterion compared, over every attempt: its
            # peak (``peak_gate_value``), passes and longest run.
            "gate": gate_record,
            "resumed": resume_state is not None,
        }
        # Per-stage run config -- written next to ``best_model.zip`` and
        # ``stage_final.zip`` immediately after the save, so that even if
        # the runtime dies (Colab disconnect, OOM kill, raised
        # CurriculumStalled on a subsequent stage) the completed stages
        # already have a self-describing config.json on disk. Captures
        # the *resolved* env + reward (after per-stage override merge),
        # the PPO hyperparams used for this stage (with the resolved
        # ent_coef / LR / schedules), the gate, the seed, and run metadata
        # (git commit + dirty flag, key library versions, hardware) --
        # enough to reproduce the stage even if configs/ppo/bootstrap.yaml
        # drifts. --resume reads its ``promoted`` flag.
        failures.attempt(
            f"write {stage.name}/config.json",
            _write_stage_config,
            stage=stage,
            cfg=cfg,
            stage_dir=stage_dir,
            output_dir=output_dir,
            promoted=promoted,
            best_win_rate=best_win_rate,
            best_checkpoint_timestep=best_timestep,
            best_checkpoint_stage_steps=best_stage_steps,
            warm_start_info=warm_start_info,
            peak_win_rate=peak_win_rate,
            stage_summary=stage_summary,
        )

        # Per-eval timeseries (win_rate, reward, length, W/L/D, end_reasons,
        # action_counts, reward_components) lives only in the eval callbacks
        # while the process is up. Persist it next to ``config.json`` so the
        # post-hoc charts in viz.py can be regenerated from disk after a run
        # finishes (or after a Colab disconnect drops the in-memory
        # ``history``). Every attempt's evals, in order.
        failures.attempt(
            f"write {stage.name}/eval_results.json",
            _write_json_atomically,
            stage_dir / "eval_results.json",
            stage_results,
            default=float,
        )

        history.append(
            {
                "stage": stage.name,
                "map_file": stage.map_file,
                "opponent": stage.opponent,
                "promoted": promoted,
                "best_win_rate": best_win_rate,
                "peak_win_rate": peak_win_rate,
                "best_checkpoint_timestep": best_timestep,
                "best_checkpoint_stage_steps": best_stage_steps,
                "retries": attempt,
                "attempts": attempts,
                "regression_restores": guard_restores,
                "gate": gate_record,
                "results": stage_results,
                "stage_final_path": str(stage_final),
            }
        )
        retries_by_stage[stage.name] = attempt

        # Refresh the run-level ``bootstrap_results.csv`` after every stage
        # so the CSV always reflects the latest completed stage even if the
        # run dies mid-curriculum (Colab disconnect, OOM kill, or a stall on
        # a later stage). Mirrors ``benchmark_results.csv`` from
        # ppo_training.ipynb but adds ``stage`` / ``map_file`` / ``opponent``
        # columns so a single CSV across the whole curriculum is grokkable.
        failures.attempt("write bootstrap_results.csv", _write_results_csv, history, output_dir / "bootstrap_results.csv")

        # Best-effort cleanup; vec envs hold subprocess handles when
        # use_subprocess=True. Don't let close-time errors mask training
        # outcomes.
        for env_obj in (vec_env, eval_env):
            close = getattr(env_obj, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.warning("could not close an env of stage '%s'", stage.name, exc_info=True)

        # The stage is over: its resume point and retry fallback go.
        # With the summary its history entry needs, so a resume can skip this
        # stage even if its config.json write failed.
        manifest.complete_stage(
            stage.name,
            promoted,
            {
                "best_win_rate": best_win_rate,
                "peak_win_rate": peak_win_rate,
                "best_checkpoint_timestep": best_timestep,
                "best_checkpoint_stage_steps": best_stage_steps,
                "warm_start": warm_start_info,
                **stage_summary,
            },
        )
        for leftover in ("latest.zip", "stage_start.zip"):
            try:
                (stage_dir / leftover).unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove %s", stage_dir / leftover, exc_info=True)

        if not promoted:
            # Save the in-progress policy at the moment of stall so
            # callers can still load it for replay videos / sanity
            # eval / hand-off, the same way they'd load a finished
            # ``final_model.zip``. Best-effort: a save failure here
            # shouldn't mask the underlying stall reason.
            stalled_final_path: str | None = None
            final_path = output_dir / "final_model.zip"
            if failures.attempt("save final_model.zip", save_model_atomically, model, final_path):
                stalled_final_path = str(final_path)
            # Surface the stage's peak-WR checkpoint alongside the
            # collapsed end-of-stage policy. Most stalls peak at or
            # above the promotion threshold before crashing (28/41 in
            # the May-June run history), so best_model.zip -- not the
            # collapsed final_model.zip -- is usually the checkpoint
            # worth warm-starting a retry from or replaying.
            stalled_best_ckpt = stage_dir / "best_model.zip"
            stalled_best_path = str(stalled_best_ckpt) if stalled_best_ckpt.exists() else None
            _write_run_status(
                output_dir,
                "curriculum_stalled",
                stalled_stage=stage.name,
                stages_completed=len(history),
                achieved_win_rate=peak_win_rate,
                best_win_rate=best_win_rate,
                peak_win_rate=peak_win_rate,
                threshold=stage.promotion_win_rate,
                patience=stage.patience,
                promotion=promotion,
                # What the criterion compared with the threshold (its peak,
                # passes and longest run), next to the win-rate peak.
                peak_gate_value=(gate_record or {}).get("peak_gate_value"),
                gate=gate_record,
                trained_timesteps=int(model.num_timesteps) - stage_start_timesteps,
                retries_used=attempt,
                retries=retries_by_stage,
                best_model_path=stalled_best_path,
                best_checkpoint_timestep=best_timestep,
                best_checkpoint_stage_steps=best_stage_steps,
                last_eval=_eval_summary(stage_results[-1] if stage_results else None),
                curriculum_hash=_curriculum_hash(cfg),
                stage_count=len(cfg.curriculum.stages),
                warm_start=warm_start_info,
                resume_count=manifest.resume_count,
                metadata_write_failures=failures.count,
                metadata_write_failure_log=failures.what[-_FAILURE_LOG_LIMIT:],
            )
            raise CurriculumStalled(
                stage_name=stage.name,
                achieved_win_rate=peak_win_rate,
                threshold=stage.promotion_win_rate,
                timesteps=stage.max_timesteps,
                history=list(history),
                final_model_path=stalled_final_path,
                best_model_path=stalled_best_path,
                metrics_callback=metrics_callback,
                patience=stage.patience,
                retries=attempt,
                gate_record=gate_record,
                trained_timesteps=int(model.num_timesteps) - stage_start_timesteps,
            )

        # Stage promoted (the stall branch above raises). PPO drifts
        # off the winning attractor *within* a stage after it first
        # clears the bar, so the in-memory model is the possibly-
        # drifted end-of-stage policy. Restore this stage's best-by-WR
        # checkpoint before the next stage inherits it -- this is the
        # fix for the ``*_random_N`` drift-propagation wall (v29
        # stalled carrying the drifted post-random_10 policy into
        # random_15; v30 warm-started random_15 from the peak
        # random_10 snapshot and cleared it trivially).
        # ``set_parameters`` swaps only policy + optimizer tensors,
        # leaving the env / schedules / callbacks intact -- identical
        # semantics to ``warm_start_path``. Also improves the final
        # stage: ``final_model.zip`` becomes the best policy, not the
        # drifted end-of-run one.
        if cfg.curriculum.restore_best_checkpoint_between_stages:
            best_ckpt = stage_dir / "best_model.zip"
            if best_ckpt.exists():
                # Surface how far into the stage the peak was: a peak at a few
                # k stage-steps is a skip-ahead carry-in (little stage-specific
                # training) -- a flag to sanity-check the next stage's first
                # eval rather than trust the transfer.
                if best_stage_steps is not None:
                    when = f", peak @ {best_stage_steps:,}/{stage.max_timesteps:,} stage steps"
                else:
                    when = ""
                shown = "n/a" if best_win_rate is None else f"{best_win_rate:.1%}"
                print(f"  restoring best checkpoint of '{stage.name}' (WR={shown}{when}) before next stage")
                model.set_parameters(str(best_ckpt), exact_match=True)
            else:
                print(f"  [warn] no best_model.zip for '{stage.name}'; carrying end-of-stage policy forward")

    final_path = output_dir / "final_model.zip"
    assert model is not None  # validate() guarantees stages is non-empty
    save_model_atomically(model, final_path)

    _write_run_status(
        output_dir,
        "completed_curriculum",
        stages_completed=len(cfg.curriculum.stages),
        stage_count=len(cfg.curriculum.stages),
        curriculum_hash=_curriculum_hash(cfg),
        warm_start=warm_start_info,
        retries=retries_by_stage,
        resume_count=manifest.resume_count,
        metadata_write_failures=failures.count,
        metadata_write_failure_log=failures.what[-_FAILURE_LOG_LIMIT:],
    )

    return {
        "model": model,
        "history": history,
        "final_model_path": str(final_path),
        "metrics_callback": metrics_callback,
    }


def _rewrite_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Replace ``path`` with ``rows`` as JSON Lines (atomically)."""
    from reinforcetactics.cloud.storage import PARTIAL_SUFFIX

    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    try:
        with partial.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=float) + "\n")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _print_stage_banner(
    stage: CurriculumStage,
    cfg: TrainingConfig,
    ent_coef: float,
    ent_schedule: dict[str, Any] | None,
    purchase_eps: float,
    purchase_eps_schedule: dict[str, Any] | None,
    lr_schedule: dict[str, Any] | None,
    promotion: dict[str, Any],
) -> None:
    max_steps = stage.resolve_max_steps(cfg.env)
    max_turns = stage.resolve_max_turns(cfg.env)
    reward_overrides = stage.reward_config or {}
    reward_note = f", reward overrides={sorted(reward_overrides.keys())}" if reward_overrides else ""
    if ent_schedule is not None:
        ent_note = f"ent_coef={ent_schedule['start']:.3f}->{ent_schedule['end']:.3f} ({ent_schedule['schedule']})"
    else:
        ent_note = f"ent_coef={ent_coef:.3f}"
    if purchase_eps_schedule is not None:
        purchase_note = (
            f", purchase_eps={purchase_eps_schedule['start']:.3f}->"
            f"{purchase_eps_schedule['end']:.3f} ({purchase_eps_schedule['schedule']})"
        )
    elif purchase_eps > 0:
        purchase_note = f", purchase_eps={purchase_eps:.3f}"
    else:
        purchase_note = ""
    if lr_schedule is None:
        lr_note = ""
    elif lr_schedule["schedule"] == "constant":
        lr_note = f", lr={lr_schedule['start']:.2e}"
    else:
        lr_note = f", lr={lr_schedule['start']:.2e}->{lr_schedule['end']:.2e} ({lr_schedule['schedule']})"
    gate = promotion["criterion"] if promotion["criterion"] != "rolling" else f"rolling-{promotion['rolling_k']}"
    if promotion["score"] != "win_rate":
        gate += ", win+0.5*draw"
    mode = "greedy" if cfg.eval.eval_deterministic else "stochastic"
    seats = cfg.eval.resolve_eval_seats(cfg.env)
    seat_note = f", seats={seats}" if seats != [1] else ""
    print(
        f"\n=== Stage '{stage.name}' :: map={stage.map_file}, "
        f"opp={stage.opponent}, target WR >= "
        f"{stage.promotion_win_rate:.0%} (patience={stage.patience}, {gate}, {mode} eval{seat_note}), "
        f"budget={stage.max_timesteps:,} steps, "
        f"max_steps={max_steps}, max_turns={max_turns}, "
        f"{ent_note}{purchase_note}{lr_note}{reward_note} ==="
    )


# ---------------------------------------------------------------------------
# Stage-faithful env construction + post-training evaluators
#
# These exist to kill a class of subtle bugs we hit during the skirmish BC
# probe (run 20260524_225835): ad-hoc env construction outside
# ``run_curriculum`` (sanity evals, replay videos, debugging probes) silently
# omitting kwargs that the production env uses -- ``reward_config``
# (env default is 10-1000x off from any tuned YAML), ``max_actions_per_turn``
# (defaults to None = disables the never-end-turn safety net), ``pad_to_size``,
# scale factors, ``opponent_kwargs``. The eval numbers then aren't comparable
# to PPO's in-training evals and the omission of ``max_actions_per_turn`` is
# actively dangerous (changes behaviour, not just measurement).
#
# ``make_stage_env`` is the single forwarding point. Other helpers that
# need to build "a faithful copy of the stage's env" should use it rather
# than calling ``make_maskable_env`` directly.
# ---------------------------------------------------------------------------


def _stage_env_kwargs(
    stage: CurriculumStage,
    env_cfg: Any,
    *,
    opponent: str | None = None,
    opponent_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The shared env-parameter set for a stage, spelled out exactly once.

    Feeds both :func:`make_stage_env` (single eval/replay envs) and the
    default vec-env training factory, so a new env parameter cannot be
    forwarded in one path and silently dropped in the other — the drift
    class ``make_stage_env``'s docstring describes. ``seed`` and ``gamma``
    stay with the callers: they are the two knobs that legitimately differ
    between train / eval / replay envs.
    """
    if opponent is None:
        opponent = stage.opponent
        if opponent_kwargs is None:
            opponent_kwargs = stage.opponent_kwargs

    return {
        "map_file": stage.map_file,
        "opponent": opponent,
        "max_steps": stage.resolve_max_steps(env_cfg),
        "max_turns": stage.resolve_max_turns(env_cfg),
        "reward_config": stage.resolve_reward_config(env_cfg),
        "enabled_units": env_cfg.enabled_units,
        "action_space_type": env_cfg.action_space_type,
        # Must match training: a config with a non-default cap (e.g.
        # max_flat_actions: 1024) trains a Discrete(1024) policy, and a
        # replay/sanity env silently built at the 512 default would
        # reject the checkpoint (or worse, evaluate a truncated action
        # space). This was the drift bug this helper exists to prevent.
        "max_flat_actions": env_cfg.max_flat_actions,
        # ``None`` only before resolve_config (run_curriculum resolves it
        # first); the env then uses FLAT_ACTION_VERSION_LATEST.
        "flat_action_version": env_cfg.flat_action_version,
        "max_actions_per_turn": env_cfg.max_actions_per_turn,
        "pad_to_size": env_cfg.pad_to_size,
        "opponent_kwargs": opponent_kwargs,
        "gold_scale": env_cfg.gold_scale,
        "turn_scale": env_cfg.turn_scale,
        "unit_count_scale": env_cfg.unit_count_scale,
        "engine_overrides": env_cfg.engine_overrides,
        # Was dropped here, so a curriculum with ``env.fog_of_war: true``
        # trained and evaluated with full information (review rltrain-9).
        "fog_of_war": env_cfg.fog_of_war,
        # 1 (default), 2 or "random" (critic-gaps-2). The eval callback sets
        # each eval episode's seat itself (``eval.eval_seats``).
        "agent_seat": env_cfg.agent_seat,
    }


def make_stage_env(
    stage: CurriculumStage,
    env_cfg: Any,
    *,
    seed: int,
    opponent: str | None = None,
    opponent_kwargs: dict[str, Any] | None = None,
    gamma: float | None = None,
) -> Any:
    """Build a ``make_maskable_env`` matching the production env for a stage.

    Forwards every kwarg that affects either behaviour or measurement so
    eval / replay envs report numbers comparable to in-training PPO evals.
    Pass this to every code path that needs "the same env the stage was
    trained against" -- sanity evals, replay video recording, BC bot-ladder
    eval, debugging probes.

    Args:
        stage: The :class:`CurriculumStage` to mirror. Provides the map,
            opponent (unless overridden), max_steps / max_turns /
            reward_config (resolved against ``env_cfg`` for defaults),
            and opponent_kwargs.
        env_cfg: The run-wide :class:`EnvConfig` providing fallbacks for
            ``stage.resolve_*`` calls plus the scale factors,
            ``enabled_units``, ``action_space_type``,
            ``max_actions_per_turn``, ``pad_to_size``, ``fog_of_war``,
            ``flat_action_version`` and ``engine_overrides`` that the stage
            doesn't override. Pass a :func:`resolve_config` result (or the
            cfg a finished :func:`run_curriculum` resolved in place) so
            ``pad_to_size`` and ``flat_action_version`` match training.
        seed: Per-env seed. Forwarded into ``env.reset(seed=...)`` so the
            episode RNG is deterministic. Required keyword to push
            callers to choose a fresh offset (e.g. ``cfg.seed + 9999``
            for a different seed than the training loop's eval used).
        opponent: Optional override for ``stage.opponent``. When set,
            ``stage.opponent_kwargs`` is dropped (it likely doesn't apply
            to the new opponent type) and the caller should pass any
            substitute kwargs via ``opponent_kwargs``.
        opponent_kwargs: Optional override for ``stage.opponent_kwargs``.
            Used when ``opponent`` is overridden and the new opponent
            type takes different constructor kwargs.
        gamma: The trainer's discount. The env uses it for the
            potential-based shaping delta (``gamma * Phi(s') - Phi(s)``),
            so an env built without it reports a different
            ``reward_breakdown['shaping_delta']`` -- and hence a different
            total reward -- than training measured. ``None`` falls back to
            the env's own default, which is only correct while
            ``ppo.gamma`` is left at 0.99. Pass ``cfg.ppo.gamma``.

    Returns:
        ``ActionMaskedEnv`` ready to feed into ``MaskablePPO.predict``
        or ``record_evaluation_to_video``.
    """
    from reinforcetactics.rl.masking import make_maskable_env

    return make_maskable_env(
        seed=seed,
        # Matches ``make_maskable_env`` / ``StrategyGameEnv``'s own default so
        # omitting the argument is behaviour-preserving, but stated here rather
        # than left implicit -- the shaping delta depends on it.
        gamma=_DEFAULT_SHAPING_GAMMA if gamma is None else float(gamma),
        **_stage_env_kwargs(stage, env_cfg, opponent=opponent, opponent_kwargs=opponent_kwargs),
    )


def record_curriculum_replays(
    cfg: TrainingConfig,
    stage_checkpoints: dict[str, dict[str, Any]],
    videos_dir: str | Path,
    *,
    seed_offset: int = 12345,
    fps: int = 4,
    scale: int = 4,
    use_pixel_art: bool = True,
    deterministic: bool = True,
    prefer_best: bool = True,
) -> list[dict[str, Any]]:
    """Record one replay video per stage checkpoint on the stage's env.

    For each entry in ``stage_checkpoints`` (the dict produced by the
    notebook's section 4b mirror), loads the checkpoint (preferring
    ``best_model.zip`` when available), builds a stage-faithful env via
    :func:`make_stage_env`, and records one episode against the stage's
    opponent on the stage's map via
    :func:`reinforcetactics.utils.video.record_evaluation_to_video`.

    Failures on a single stage's video (missing imageio_ffmpeg, pygame
    issues, etc.) are caught and logged so the rest of the loop still
    runs -- a renderer hiccup shouldn't lose the videos for the other
    stages.

    Args:
        cfg: The :class:`TrainingConfig` whose ``curriculum.stages`` and
            ``env`` provide the per-stage env templates.
        stage_checkpoints: Mapping ``stage_name -> {map_file, opponent,
            stage_final, best_model, promoted, best_win_rate}`` as
            produced by the notebook's section 4b. Stages whose
            checkpoint paths are both ``None`` are skipped.
        videos_dir: Output directory for ``<stage_name>.mp4`` files.
            Created if missing.
        seed_offset: Added to ``cfg.seed`` to derive the per-env seed.
            Use a value distinct from the training loop's eval seed
            (typical: cfg.seed + 9999 for sanity eval, + 12345 for
            replays) so the replay isn't a deterministic copy of an
            in-training eval episode.
        fps: Replay video frames-per-second.
        scale: Render scale multiplier for the video frames.
        use_pixel_art: Use the pixel-art renderer instead of the
            vector renderer.
        deterministic: Use deterministic argmax actions during replay
            recording (the default): one greedy game shows the policy's
            modal behaviour, where a single sampled game shows one draw of
            it. This is not the curriculum gate's mode by default
            (``eval.eval_deterministic`` is False: the gate measures the
            stochastic policy, and records the greedy win rate beside
            it); pass ``cfg.eval.eval_deterministic`` to replay the mode
            the gate judged.
        prefer_best: When True, prefer ``best_model.zip`` over
            ``stage_final.zip`` for the replay. Mirrors the
            in-notebook default.

    Returns:
        List of summary dicts, one per recorded video. Each contains
        ``stage``, ``map_file``, ``opponent``, ``video_path``,
        ``winner``, ``end_reason``, ``steps``, ``total_reward``, and
        ``step_stats`` (per-step trace consumed by
        ``plot_individual_game_stats``). Stages whose recording
        failed contribute no entry.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("sb3-contrib is required for record_curriculum_replays. Install: pip install sb3-contrib") from exc

    from reinforcetactics.utils.video import record_evaluation_to_video

    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    stages_by_name = {s.name: s for s in cfg.curriculum.stages}
    summaries: list[dict[str, Any]] = []

    for stage_name, meta in stage_checkpoints.items():
        # Prefer the best-by-WR snapshot when available; the final
        # post-promotion checkpoint is the fallback. A stage with no
        # checkpoint path (rare -- write failure mid-run) is skipped.
        ckpt_path = (meta.get("best_model") if prefer_best else None) or meta.get("stage_final")
        if ckpt_path is None:
            print(f"[{stage_name}] no checkpoint, skipping video")
            continue

        stage = stages_by_name.get(stage_name)
        if stage is None:
            print(f"[{stage_name}] no matching CurriculumStage in cfg, skipping video")
            continue

        replay_env = make_stage_env(stage, cfg.env, seed=cfg.seed + seed_offset, gamma=cfg.ppo.gamma)
        try:
            model = MaskablePPO.load(ckpt_path)
            video_path = videos_dir / f"{stage_name}.mp4"
            info = record_evaluation_to_video(
                replay_env,
                model,
                output_path=str(video_path),
                fps=fps,
                max_steps=stage.resolve_max_steps(cfg.env),
                deterministic=deterministic,
                scale=scale,
                use_pixel_art=use_pixel_art,
            )
            summaries.append(
                {
                    "stage": stage_name,
                    "map_file": meta.get("map_file"),
                    "opponent": meta.get("opponent"),
                    "video_path": info.get("video_path"),
                    "winner": info.get("winner"),
                    "agent_player": info.get("agent_player", 1),
                    "end_reason": info.get("end_reason"),
                    "steps": info.get("steps"),
                    "total_reward": info.get("total_reward"),
                    # Per-step trace required by section 9 (individual
                    # game stats 2x3). Each entry has unit/gold/structure
                    # counts plus per-step action_type / reward_breakdown.
                    "step_stats": info.get("step_stats", []),
                }
            )
            print(
                f"[{stage_name}] map={meta.get('map_file')}  opp={meta.get('opponent')}  "
                f"winner={info.get('winner')}  end={info.get('end_reason')}  "
                f"steps={info.get('steps')}  -> {info.get('video_path')}"
            )
        except Exception as exc:
            # Don't let a single rendering hiccup take out the rest of
            # the loop -- ffmpeg / pygame failures are usually
            # environment-specific.
            print(f"[{stage_name}] video failed: {exc}")
        finally:
            replay_env.close()

    return summaries
