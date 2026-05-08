"""
Curriculum-based PPO bootstrapping.

Trains a single MaskablePPO policy through a sequence of stages
(map x opponent combinations) before handing the resulting checkpoint
off to self-play. Each stage runs until ``PromotionCallback`` reports
that the win rate has held above the stage's threshold for
``patience`` evaluations, at which point ``model.learn()`` returns
early and the runner moves to the next stage.

If a stage exhausts its ``max_timesteps`` budget without promoting,
the runner raises :class:`CurriculumStalled`. Bumping the budget alone
usually masks a real issue (reward shaping, hyperparams), so failing
loud is the default.

Configuration lives in :class:`reinforcetactics.rl.config.TrainingConfig`:
``cfg.curriculum.stages`` defines the curriculum, ``cfg.env`` / ``cfg.ppo``
are shared across stages, and each stage may override ``max_steps``,
``max_turns``, ``ent_coef``, ``reward_config``, or ``opponent_kwargs``
on a per-stage basis. ``cfg.eval.eval_freq`` and ``cfg.env.n_envs``
drive eval cadence and parallelism respectively.

Usage:

    from reinforcetactics.rl.bootstrap import run_curriculum
    from reinforcetactics.rl.config import load_config

    cfg = load_config("configs/bootstrap.yaml")
    result = run_curriculum(cfg, output_dir="benchmarks/bootstrap")
    # result["final_model_path"] -> ready for self-play warm start
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from reinforcetactics.rl.config import CurriculumStage, TrainingConfig

ConfigPath = Union[str, Path]


class CurriculumStalled(RuntimeError):
    """Raised when a stage exhausts its budget without promoting.

    Attributes:
        stage_name: Name of the failing stage.
        achieved_win_rate: Best win rate observed during the stage.
        threshold: Promotion threshold the stage was required to clear.
        timesteps: Stage timestep budget that was exhausted.
        history: Per-stage results gathered up to (and including) the
            stalled stage. Same shape as ``run_curriculum``'s
            ``result["history"]`` so callers can recover diagnostics
            from a partial run by reading
            ``CurriculumStalled.partial_result()``.
        final_model_path: Path to ``final_model.zip`` saved at the
            point of stall (the in-progress policy at the moment the
            stalled stage gave up). ``None`` for older callers that
            didn't pass it.
        metrics_callback: ``TrainingMetricsCallback`` accumulated over
            the partial run, exposed for the same reason as
            ``history``.
    """

    def __init__(
        self,
        stage_name: str,
        achieved_win_rate: float,
        threshold: float,
        timesteps: int,
        history: Optional[List[Dict[str, Any]]] = None,
        final_model_path: Optional[str] = None,
        metrics_callback: Any = None,
    ) -> None:
        self.stage_name = stage_name
        self.achieved_win_rate = achieved_win_rate
        self.threshold = threshold
        self.timesteps = timesteps
        self.history = history or []
        self.final_model_path = final_model_path
        self.metrics_callback = metrics_callback
        super().__init__(
            f"Stage '{stage_name}' stalled at {timesteps:,} timesteps: "
            f"best win_rate {achieved_win_rate:.1%} did not reach "
            f"threshold {threshold:.1%}"
        )

    def partial_result(self) -> Dict[str, Any]:
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
            "metrics_callback": self.metrics_callback,
            "stalled": True,
            "stalled_stage": self.stage_name,
        }


# ---------------------------------------------------------------------------
# Default builders. Tests / advanced callers can pass replacements through
# `run_curriculum(..., train_env_factory=..., model_factory=...)` to avoid
# importing sb3-contrib or constructing real environments.
# ---------------------------------------------------------------------------


def _default_train_env_factory(stage: CurriculumStage, cfg: TrainingConfig):
    from reinforcetactics.rl.masking import make_maskable_vec_env

    return make_maskable_vec_env(
        n_envs=cfg.env.n_envs,
        map_file=stage.map_file,
        opponent=stage.opponent,
        max_steps=stage.resolve_max_steps(cfg.env),
        max_turns=stage.resolve_max_turns(cfg.env),
        reward_config=stage.resolve_reward_config(cfg.env),
        enabled_units=cfg.env.enabled_units,
        action_space_type=cfg.env.action_space_type,
        max_flat_actions=cfg.env.max_flat_actions,
        seed=cfg.seed,
        use_subprocess=False,
        opponent_kwargs=stage.opponent_kwargs,
    )


def _default_eval_env_factory(stage: CurriculumStage, cfg: TrainingConfig):
    from reinforcetactics.rl.masking import make_maskable_env

    return make_maskable_env(
        map_file=stage.map_file,
        opponent=stage.opponent,
        max_steps=stage.resolve_max_steps(cfg.env),
        max_turns=stage.resolve_max_turns(cfg.env),
        reward_config=stage.resolve_reward_config(cfg.env),
        enabled_units=cfg.env.enabled_units,
        action_space_type=cfg.env.action_space_type,
        max_flat_actions=cfg.env.max_flat_actions,
        seed=cfg.seed,
        opponent_kwargs=stage.opponent_kwargs,
    )


def _default_model_factory(vec_env, cfg: TrainingConfig, output_dir: Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        seed=cfg.seed,
        # SB3 verbose=1 prints rollout/train tables every iteration, which
        # drowns out the curriculum's per-eval WR line. TensorBoard logging
        # is independent of this setting so curves still land on disk.
        verbose=0,
        tensorboard_log=str(output_dir / "tensorboard"),
        **cfg.ppo.as_sb3_kwargs(),
    )


def _write_stage_config(
    *,
    stage: CurriculumStage,
    cfg: TrainingConfig,
    stage_dir: Path,
    output_dir: Path,
    promoted: bool,
    best_win_rate: float,
) -> None:
    """Write ``stage_dir / config.json`` describing the stage's resolved settings.

    Imported lazily so the bootstrap module stays importable even if the
    optional ``utils.run_config`` dependencies aren't on the path during
    a partial install (e.g. type-checking environments without torch).
    """
    from reinforcetactics.utils.run_config import build_run_config, write_run_config

    ppo_resolved = asdict(cfg.ppo)
    ppo_resolved["ent_coef"] = stage.resolve_ent_coef(cfg.ppo)
    ent_schedule = stage.resolve_ent_coef_schedule()
    if ent_schedule is not None:
        ppo_resolved["ent_coef_schedule"] = ent_schedule

    env_resolved: Dict[str, Any] = {
        "map_file": stage.map_file,
        "max_steps": stage.resolve_max_steps(cfg.env),
        "max_turns": stage.resolve_max_turns(cfg.env),
        "enabled_units": list(cfg.env.enabled_units) if cfg.env.enabled_units else None,
        "action_space_type": cfg.env.action_space_type,
        "max_flat_actions": cfg.env.max_flat_actions,
        "reward_config": stage.resolve_reward_config(cfg.env),
        "opponent_kwargs": dict(stage.opponent_kwargs) if stage.opponent_kwargs else None,
    }

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
            "n_eval_episodes": stage.n_eval_episodes,
            "n_envs": cfg.env.n_envs,
            "eval_freq": cfg.eval.eval_freq,
            "promoted": promoted,
            "best_win_rate": best_win_rate,
            "output_dir": str(output_dir),
        },
    )
    write_run_config(run_config, stage_dir / "config.json")


def run_curriculum(
    cfg: TrainingConfig,
    output_dir: ConfigPath,
    *,
    train_env_factory: Optional[Callable[[CurriculumStage, TrainingConfig], Any]] = None,
    eval_env_factory: Optional[Callable[[CurriculumStage, TrainingConfig], Any]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
    progress_bar: bool = False,
) -> Dict[str, Any]:
    """Train through every stage in ``cfg.curriculum.stages``.

    Args:
        cfg: Validated :class:`TrainingConfig` with a non-empty
            ``cfg.curriculum.stages``.
        output_dir: Root directory for stage subfolders, tensorboard logs,
            and the final checkpoint.
        train_env_factory: ``(stage, cfg) -> vec_env``. Defaults to
            ``make_maskable_vec_env`` with the resolved per-stage env.
        eval_env_factory: ``(stage, cfg) -> env``. Defaults to
            ``make_maskable_env`` with the resolved per-stage env.
        model_factory: ``(vec_env, cfg, output_dir) -> model``. Called once
            for the first stage; later stages reuse the model via
            ``model.set_env(...)``. Defaults to MaskablePPO.
        progress_bar: Forwarded to ``model.learn()``.

    Returns:
        Dict with keys ``model``, ``history`` (list of per-stage dicts),
        ``final_model_path``, ``metrics_callback``.

    Raises:
        CurriculumStalled: if any stage hits its ``max_timesteps`` without
            the promotion criterion.
    """
    from reinforcetactics.rl.callbacks import (
        EntropyScheduleCallback,
        PeriodicEvalCallback,
        PromotionCallback,
        TrainingMetricsCallback,
    )

    cfg.validate()
    if not cfg.curriculum.stages:
        raise ValueError("cfg.curriculum.stages is empty; nothing to run")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env_factory = train_env_factory or _default_train_env_factory
    eval_env_factory = eval_env_factory or _default_eval_env_factory
    model_factory = model_factory or _default_model_factory

    metrics_callback = TrainingMetricsCallback()
    history: List[Dict[str, Any]] = []
    model = None

    for stage in cfg.curriculum.stages:
        stage_dir = output_dir / stage.name
        stage_dir.mkdir(parents=True, exist_ok=True)

        vec_env = train_env_factory(stage, cfg)
        eval_env = eval_env_factory(stage, cfg)

        if model is None:
            model = model_factory(vec_env, cfg, output_dir)
        else:
            model.set_env(vec_env)

        # Apply per-stage entropy override. SB3 reads ``model.ent_coef`` fresh
        # inside every ``train()`` step, so live mutation works without
        # rebuilding the model. The starter -> beginner transition is the
        # primary use case: bumping entropy lets the policy break out of the
        # previous map's deterministic groove and re-explore.
        ent_coef = stage.resolve_ent_coef(cfg.ppo)
        if hasattr(model, "ent_coef"):
            model.ent_coef = ent_coef
        ent_schedule = stage.resolve_ent_coef_schedule()

        eval_cb = PeriodicEvalCallback(
            eval_env=eval_env,
            eval_freq=cfg.eval.eval_freq,
            n_eval_episodes=stage.n_eval_episodes,
            save_dir=stage_dir,
        )
        promote_cb = PromotionCallback(
            eval_callback=eval_cb,
            threshold=stage.promotion_win_rate,
            patience=stage.patience,
        )
        callbacks: List[Any] = [metrics_callback, eval_cb, promote_cb]
        if ent_schedule is not None:
            # Schedule annealing covers the full stage budget; if the
            # stage promotes early ``learn()`` returns before the
            # callback hits ``end``, which is fine -- we wanted the
            # cooling to have *been available* during the run-up.
            callbacks.append(
                EntropyScheduleCallback(
                    start=ent_schedule["start"],
                    end=ent_schedule["end"],
                    total_timesteps=stage.max_timesteps,
                    schedule=ent_schedule["schedule"],
                )
            )

        max_steps = stage.resolve_max_steps(cfg.env)
        max_turns = stage.resolve_max_turns(cfg.env)
        reward_overrides = stage.reward_config or {}
        reward_note = f", reward overrides={sorted(reward_overrides.keys())}" if reward_overrides else ""
        if ent_schedule is not None:
            ent_note = f"ent_coef={ent_schedule['start']:.3f}->{ent_schedule['end']:.3f} ({ent_schedule['schedule']})"
        else:
            ent_note = f"ent_coef={ent_coef:.3f}"
        print(
            f"\n=== Stage '{stage.name}' :: map={stage.map_file}, "
            f"opp={stage.opponent}, target WR >= "
            f"{stage.promotion_win_rate:.0%} (patience={stage.patience}), "
            f"budget={stage.max_timesteps:,} steps, "
            f"max_steps={max_steps}, max_turns={max_turns}, "
            f"{ent_note}{reward_note} ==="
        )

        model.learn(
            total_timesteps=stage.max_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=progress_bar,
        )

        stage_final = stage_dir / "stage_final.zip"
        model.save(str(stage_final))

        # Per-stage run config -- written next to ``best_model.zip`` and
        # ``stage_final.zip`` immediately after the save, so that even if
        # the runtime dies (Colab disconnect, OOM kill, raised
        # CurriculumStalled on a subsequent stage) the completed stages
        # already have a self-describing config.json on disk. Captures
        # the *resolved* env + reward (after per-stage override merge),
        # the PPO hyperparams used for this stage (with the resolved
        # ent_coef / schedule), the seed, and run metadata (git commit
        # + dirty flag, key library versions, hardware) -- enough to
        # reproduce the stage even if configs/bootstrap.yaml drifts.
        try:
            _write_stage_config(
                stage=stage,
                cfg=cfg,
                stage_dir=stage_dir,
                output_dir=output_dir,
                promoted=promote_cb.promoted,
                best_win_rate=eval_cb.best_win_rate,
            )
        except Exception:  # noqa: BLE001
            # A metadata write failure must not mask the training
            # outcome; the checkpoint is the load-bearing artifact.
            pass

        history.append(
            {
                "stage": stage.name,
                "map_file": stage.map_file,
                "opponent": stage.opponent,
                "promoted": promote_cb.promoted,
                "best_win_rate": eval_cb.best_win_rate,
                "results": list(eval_cb.results),
                "stage_final_path": str(stage_final),
            }
        )

        # Best-effort cleanup; vec envs hold subprocess handles when
        # use_subprocess=True. Don't let close-time errors mask training
        # outcomes.
        for env_obj in (vec_env, eval_env):
            close = getattr(env_obj, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

        if not promote_cb.promoted:
            # Save the in-progress policy at the moment of stall so
            # callers can still load it for replay videos / sanity
            # eval / hand-off, the same way they'd load a finished
            # ``final_model.zip``. Best-effort: a save failure here
            # shouldn't mask the underlying stall reason.
            stalled_final_path: Optional[str] = None
            try:
                final_path = output_dir / "final_model.zip"
                assert model is not None
                model.save(str(final_path))
                stalled_final_path = str(final_path)
            except Exception:  # noqa: BLE001
                stalled_final_path = None
            raise CurriculumStalled(
                stage_name=stage.name,
                achieved_win_rate=eval_cb.best_win_rate,
                threshold=stage.promotion_win_rate,
                timesteps=stage.max_timesteps,
                history=list(history),
                final_model_path=stalled_final_path,
                metrics_callback=metrics_callback,
            )

    final_path = output_dir / "final_model.zip"
    assert model is not None  # validate() guarantees stages is non-empty
    model.save(str(final_path))

    return {
        "model": model,
        "history": history,
        "final_model_path": str(final_path),
        "metrics_callback": metrics_callback,
    }
