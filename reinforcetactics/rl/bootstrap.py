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

Usage:

    from reinforcetactics.rl.bootstrap import (
        load_bootstrap_config,
        run_curriculum,
    )

    cfg = load_bootstrap_config("configs/bootstrap.yaml")
    result = run_curriculum(cfg, output_dir="benchmarks/bootstrap")
    # result["final_model_path"] -> ready for self-play warm start
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from reinforcetactics.rl.config import PPOConfig

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


@dataclass
class CurriculumStage:
    """One curriculum step: a (map, opponent) pair with a promotion criterion.

    The ``max_steps``, ``max_turns``, ``ent_coef``, and ``reward_config``
    fields are optional overrides; when ``None`` the runner falls back to
    ``BootstrapEnvDefaults`` / ``PPOConfig``. Typical use cases:

    - Bump ``max_turns`` and ``max_steps`` on a larger map (units take more
      turns to traverse).
    - Raise ``ent_coef`` on the first stage of a new map to crack the
      previous stage's policy out of a deterministic groove.
    - Override ``reward_config`` keys (merged into the env defaults) when a
      new map's geometry changes which win condition is achievable -- e.g.
      a sprawling map where HQ-capture is impractical and elimination is
      the natural endpoint, so you flip ``win_by_hq_capture`` /
      ``win_by_elimination`` weights for that stage only.
    """

    name: str
    map_file: str
    opponent: str
    promotion_win_rate: float = 0.9
    patience: int = 2
    max_timesteps: int = 1_000_000
    n_eval_episodes: int = 30
    # Optional per-stage overrides. None = inherit from cfg.env / cfg.ppo.
    max_steps: Optional[int] = None
    max_turns: Optional[int] = None
    # Either a constant float (held throughout the stage) or a mapping
    # ``{start, end, schedule}`` describing a per-stage anneal. Mapping
    # form drives :class:`EntropyScheduleCallback` so exploration can be
    # cooled as the policy approaches its promotion threshold; sustained
    # high entropy was producing ±15% WR oscillation in adjacent evals
    # on the random-opponent stages and preventing convergence past
    # ~60% WR even though the threshold was 75%.
    ent_coef: Optional[Union[float, Dict[str, Any]]] = None
    # Reward-config override is *merged* over the env default (not replaced)
    # so per-stage entries only need to spell out the keys that change.
    reward_config: Optional[Dict[str, float]] = None
    # Extra kwargs forwarded to the opponent constructor (e.g.
    # ``{max_actions: 10}`` for ``RandomBot``). None / empty = use bot defaults.
    opponent_kwargs: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("stage.name must be non-empty")
        if not self.map_file:
            raise ValueError(f"stage '{self.name}': map_file must be set")
        if self.opponent not in (
            "random",
            "balanced_random",
            "simple",
            "bot",
            "medium",
            "advanced",
            "noop",
        ):
            raise ValueError(
                f"stage '{self.name}': unknown opponent '{self.opponent}'. "
                "Expected one of: random, balanced_random, simple, bot, medium, advanced, noop"
            )
        if not 0.0 <= self.promotion_win_rate <= 1.0:
            raise ValueError(f"stage '{self.name}': promotion_win_rate must be in [0, 1]")
        if self.patience < 1:
            raise ValueError(f"stage '{self.name}': patience must be >= 1")
        if self.max_timesteps <= 0:
            raise ValueError(f"stage '{self.name}': max_timesteps must be > 0")
        if self.n_eval_episodes <= 0:
            raise ValueError(f"stage '{self.name}': n_eval_episodes must be > 0")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"stage '{self.name}': max_steps override must be > 0")
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError(f"stage '{self.name}': max_turns override must be > 0")
        if self.ent_coef is not None:
            if isinstance(self.ent_coef, Mapping):
                unknown = set(self.ent_coef.keys()) - {"start", "end", "schedule"}
                if unknown:
                    raise ValueError(
                        f"stage '{self.name}': ent_coef schedule has unknown keys {sorted(unknown)}. "
                        "Valid keys: start, end, schedule"
                    )
                for required in ("start", "end"):
                    if required not in self.ent_coef:
                        raise ValueError(
                            f"stage '{self.name}': ent_coef schedule missing required key '{required}'"
                        )
                    val = self.ent_coef[required]
                    if not isinstance(val, (int, float)) or val < 0:
                        raise ValueError(
                            f"stage '{self.name}': ent_coef.{required} must be a non-negative number, got {val!r}"
                        )
                schedule_kind = self.ent_coef.get("schedule", "linear")
                if schedule_kind not in ("linear", "cosine"):
                    raise ValueError(
                        f"stage '{self.name}': ent_coef.schedule must be 'linear' or 'cosine', got {schedule_kind!r}"
                    )
            elif isinstance(self.ent_coef, (int, float)):
                if self.ent_coef < 0:
                    raise ValueError(f"stage '{self.name}': ent_coef override must be >= 0")
            else:
                raise TypeError(
                    f"stage '{self.name}': ent_coef must be a number or a "
                    f"{{start, end, schedule}} mapping, got {type(self.ent_coef).__name__}"
                )
        if self.reward_config is not None and not isinstance(self.reward_config, Mapping):
            raise TypeError(
                f"stage '{self.name}': reward_config override must be a mapping, got {type(self.reward_config).__name__}"
            )
        if self.opponent_kwargs is not None and not isinstance(self.opponent_kwargs, Mapping):
            raise TypeError(
                f"stage '{self.name}': opponent_kwargs override must be a mapping, got {type(self.opponent_kwargs).__name__}"
            )

    def resolve_max_steps(self, defaults: "BootstrapEnvDefaults") -> int:
        return self.max_steps if self.max_steps is not None else defaults.max_steps

    def resolve_max_turns(self, defaults: "BootstrapEnvDefaults") -> Optional[int]:
        return self.max_turns if self.max_turns is not None else defaults.max_turns

    def resolve_ent_coef(self, ppo: PPOConfig) -> float:
        """Return the *initial* entropy coefficient for the stage.

        For a constant override this is the value itself; for a schedule
        mapping it's ``schedule['start']`` so ``model.ent_coef`` is
        seeded correctly before the schedule callback takes over.
        """
        if self.ent_coef is None:
            return ppo.ent_coef
        if isinstance(self.ent_coef, Mapping):
            return float(self.ent_coef["start"])
        return float(self.ent_coef)

    def resolve_ent_coef_schedule(self) -> Optional[Dict[str, Any]]:
        """Return ``{start, end, schedule}`` if ``ent_coef`` is a mapping, else ``None``.

        ``None`` means a constant coefficient (no callback installed); a
        dict means the runner should attach :class:`EntropyScheduleCallback`
        for this stage with ``total_timesteps=stage.max_timesteps``.
        """
        if isinstance(self.ent_coef, Mapping):
            return {
                "start": float(self.ent_coef["start"]),
                "end": float(self.ent_coef["end"]),
                "schedule": str(self.ent_coef.get("schedule", "linear")),
            }
        return None

    def resolve_reward_config(self, defaults: "BootstrapEnvDefaults") -> Optional[Dict[str, float]]:
        """Return the reward config to use for this stage.

        Per-stage overrides are merged on top of ``defaults.reward_config``,
        so a stage only needs to spell out the keys it changes. Returns
        ``None`` when neither side has anything (env will fall back to its
        own built-in defaults).
        """
        base = dict(defaults.reward_config) if defaults.reward_config else {}
        if self.reward_config:
            base.update(self.reward_config)
        return base if base else None


@dataclass
class BootstrapEnvDefaults:
    """Env settings shared across all stages (map_file/opponent come from stages)."""

    max_steps: int = 400
    max_turns: Optional[int] = 20
    enabled_units: Optional[List[str]] = None
    action_space_type: str = "flat_discrete"
    max_flat_actions: int = 512
    reward_config: Optional[Dict[str, float]] = None


@dataclass
class BootstrapConfig:
    """Full bootstrap-curriculum configuration."""

    stages: List[CurriculumStage]
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: BootstrapEnvDefaults = field(default_factory=BootstrapEnvDefaults)
    eval_freq: int = 50_000
    n_envs: int = 4
    seed: int = 0

    def validate(self) -> None:
        if not self.stages:
            raise ValueError("BootstrapConfig.stages must be non-empty")
        seen = set()
        for stage in self.stages:
            stage.validate()
            if stage.name in seen:
                raise ValueError(f"duplicate stage name: '{stage.name}'")
            seen.add(stage.name)
        if self.eval_freq <= 0:
            raise ValueError("eval_freq must be > 0")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be > 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "n_envs": self.n_envs,
            "eval_freq": self.eval_freq,
            "env": asdict(self.env),
            "ppo": asdict(self.ppo),
            "stages": [asdict(s) for s in self.stages],
        }


def _build_dataclass_from_mapping(cls, raw: Mapping[str, Any]):
    valid = {f.name for f in fields(cls)}
    unknown = set(raw.keys()) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}. Valid keys: {sorted(valid)}")
    return cls(**{k: v for k, v in raw.items() if k in valid})


def config_from_dict(data: Mapping[str, Any]) -> BootstrapConfig:
    """Construct a :class:`BootstrapConfig` from a plain dict."""
    if not isinstance(data, Mapping):
        raise TypeError(f"Bootstrap config must be a mapping, got {type(data).__name__}")

    valid_keys = {"seed", "n_envs", "eval_freq", "env", "ppo", "stages"}
    unknown = set(data.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown)}. Valid keys: {sorted(valid_keys)}")

    raw_stages = data.get("stages") or []
    if not isinstance(raw_stages, list):
        raise TypeError(f"'stages' must be a list, got {type(raw_stages).__name__}")
    stages = [_build_dataclass_from_mapping(CurriculumStage, s) for s in raw_stages]

    env_raw = data.get("env") or {}
    if not isinstance(env_raw, Mapping):
        raise TypeError(f"'env' must be a mapping, got {type(env_raw).__name__}")
    env = _build_dataclass_from_mapping(BootstrapEnvDefaults, env_raw)

    ppo_raw = data.get("ppo") or {}
    if not isinstance(ppo_raw, Mapping):
        raise TypeError(f"'ppo' must be a mapping, got {type(ppo_raw).__name__}")
    ppo = _build_dataclass_from_mapping(PPOConfig, ppo_raw)

    cfg = BootstrapConfig(
        stages=stages,
        ppo=ppo,
        env=env,
        eval_freq=int(data.get("eval_freq", 50_000)),
        n_envs=int(data.get("n_envs", 4)),
        seed=int(data.get("seed", 0)),
    )
    cfg.validate()
    return cfg


def load_bootstrap_config(path: ConfigPath) -> BootstrapConfig:
    """Load and validate a bootstrap config from YAML or JSON."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Bootstrap config not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError(
                f"Cannot load '{p}': PyYAML is not installed. Install with `pip install PyYAML` or use a .json config."
            )
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise ValueError(f"Unsupported config extension '{suffix}' for {p}. Use .yaml, .yml, or .json.")
    if not isinstance(data, Mapping):
        raise TypeError(f"Config file {p} must contain a mapping at the top level.")
    return config_from_dict(dict(data))


# ---------------------------------------------------------------------------
# Default builders. Tests / advanced callers can pass replacements through
# `run_curriculum(..., train_env_factory=..., model_factory=...)` to avoid
# importing sb3-contrib or constructing real environments.
# ---------------------------------------------------------------------------


def _default_train_env_factory(stage: CurriculumStage, cfg: BootstrapConfig):
    from reinforcetactics.rl.masking import make_maskable_vec_env

    return make_maskable_vec_env(
        n_envs=cfg.n_envs,
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


def _default_eval_env_factory(stage: CurriculumStage, cfg: BootstrapConfig):
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


def _default_model_factory(vec_env, cfg: BootstrapConfig, output_dir: Path):
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


def run_curriculum(
    cfg: BootstrapConfig,
    output_dir: ConfigPath,
    *,
    train_env_factory: Optional[Callable[[CurriculumStage, BootstrapConfig], Any]] = None,
    eval_env_factory: Optional[Callable[[CurriculumStage, BootstrapConfig], Any]] = None,
    model_factory: Optional[Callable[..., Any]] = None,
    progress_bar: bool = False,
) -> Dict[str, Any]:
    """Train through every stage in ``cfg.stages``, advancing on promotion.

    Args:
        cfg: Validated :class:`BootstrapConfig`.
        output_dir: Root directory for stage subfolders, tensorboard logs,
            and the final checkpoint.
        train_env_factory: ``(stage, cfg) -> vec_env``. Defaults to
            ``make_maskable_vec_env`` with the cfg's env defaults.
        eval_env_factory: ``(stage, cfg) -> env``. Defaults to
            ``make_maskable_env``.
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env_factory = train_env_factory or _default_train_env_factory
    eval_env_factory = eval_env_factory or _default_eval_env_factory
    model_factory = model_factory or _default_model_factory

    metrics_callback = TrainingMetricsCallback()
    history: List[Dict[str, Any]] = []
    model = None

    for stage in cfg.stages:
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
            eval_freq=cfg.eval_freq,
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
            ent_note = (
                f"ent_coef={ent_schedule['start']:.3f}->{ent_schedule['end']:.3f} "
                f"({ent_schedule['schedule']})"
            )
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
