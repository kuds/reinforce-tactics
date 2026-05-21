"""
Training configuration loader for Reinforce Tactics RL.

Centralizes hyperparameters in YAML/JSON files so training runs are
reproducible without editing source. Supports:

- Loading from ``.yaml``, ``.yml``, or ``.json`` files
- Hierarchical sections: ``env``, ``ppo``, ``feudal``, ``self_play``,
  ``alphazero``, ``curriculum``, ``eval``, ``logging``
- CLI overrides: values passed via ``--key value`` beat file values
- Dotted override keys (``ppo.learning_rate=1e-4``) for nested updates
- Dataclass validation with typed sections

Usage:
    from reinforcetactics.rl.config import load_config, apply_overrides

    cfg = load_config("configs/maskable_ppo.yaml")
    cfg = apply_overrides(cfg, {"ppo.learning_rate": 1e-4})
    model = MaskablePPO(**cfg.ppo.as_sb3_kwargs(), env=env)
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml is a required dev dep
    yaml = None


ConfigPath = Union[str, Path]


@dataclass
class EnvConfig:
    """Environment construction parameters."""

    map_file: Optional[str] = None
    opponent: str = "bot"
    max_steps: int = 200
    max_turns: Optional[int] = None
    fog_of_war: bool = False
    enabled_units: Optional[List[str]] = None
    action_space_type: str = "multi_discrete"
    max_flat_actions: int = 512
    # Optional hard cap on agent actions per game-turn. When set, the
    # action mask narrows to end_turn-only once the agent has executed
    # this many actions in the current game-turn. Defends against the
    # "never end the turn" stall mode where the policy cycles through
    # legal-but-unproductive actions until ``max_steps`` truncates the
    # episode. ``None`` (default) disables the cap. See
    # :class:`reinforcetactics.rl.gym_env.StrategyGameEnv` for details.
    max_actions_per_turn: Optional[int] = None
    reward_config: Optional[Dict[str, float]] = None
    # Optional sparse overlay over the non-YAML engine constants
    # (``constants.py``): ``starting_gold``, ``headquarters_income``,
    # ``building_income``, ``tower_income``, and ``unit_data``
    # (``{CODE: {field: value}}`` per-unit, per-field deltas). Absent /
    # ``None`` = use the module constants (today's behaviour). Makes
    # balance a first-class, swept, auto-recorded config axis instead of
    # an invisible engine constant that only a git checkout could change.
    # Resolved by ``GameState`` (its tables are the per-game source of
    # truth) and snapshotted into ``config.json``.
    engine_overrides: Optional[Dict[str, Any]] = None
    n_envs: int = 4
    use_subprocess: bool = True
    # Optional ``(pad_h, pad_w)`` for cross-stage observation-shape unification.
    # When the curriculum mixes maps of different sizes, the bootstrap runner
    # auto-fills this with the curriculum-wide max so a single PPO policy can
    # train across all stages without an observation-space mismatch. Set
    # explicitly to override the auto-computed value (e.g. to leave headroom
    # for a future larger map). Only honoured by ``flat_discrete``.
    pad_to_size: Optional[Tuple[int, int]] = None
    # ``global_features`` tanh normalization divisors. Defaults match the
    # module-level constants in ``reinforcetactics.rl.observation`` and
    # are tuned for the current curriculum's gold / turn / army-size
    # ranges. Override on a per-run basis when shipping a map / economy
    # whose typical values differ enough that the linear regime of
    # ``tanh`` no longer covers the relevant operating point.
    gold_scale: float = 1000.0
    turn_scale: float = 60.0
    unit_count_scale: float = 20.0


@dataclass
class PPOConfig:
    """Hyperparameters for PPO / MaskablePPO."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_action_masking: bool = True
    device: str = "auto"
    # LR schedule applied across the total budget. ``constant`` keeps the base
    # LR; ``linear`` anneals to zero. Consumed by feudal training; SB3's PPO
    # uses its own scheduler API so this field is ignored on the SB3 path.
    lr_schedule: str = "constant"
    # Forwarded to MaskablePPO/PPO as ``policy_kwargs``. Use to set
    # ``net_arch`` (e.g. ``{"net_arch": {"pi": [256, 256], "vf": [256, 256]}}``)
    # or wire in a custom features extractor. ``None`` keeps SB3's defaults
    # (MlpPolicy: ``[64, 64]``; CombinedExtractor for Dict obs spaces).
    policy_kwargs: Optional[Dict[str, Any]] = None
    # Probability with which a sampled ``create_unit`` action has its
    # ``unit_type`` sub-action resampled uniformly over the env's
    # currently-legal (enabled + affordable) unit types. Pure exploration
    # knob: ε=0 disables, ε=1 always randomizes purchases. The substituted
    # action is what gets executed *and* what gets stored in the rollout
    # buffer (with log-prob recomputed under the masked policy at the
    # substituted action), so PPO's ratio stays internally consistent.
    # Per-stage overrides live on :class:`CurriculumStage`.
    purchase_explore_eps: float = 0.0

    def as_sb3_kwargs(self) -> Dict[str, Any]:
        """Return the subset of fields accepted by PPO/MaskablePPO __init__."""
        # Both feudal-only (``lr_schedule``) and PPO-bootstrap-only
        # (``purchase_explore_eps``) fields need to be filtered before
        # forwarding to SB3, which doesn't recognize either kwarg.
        skip = {"use_action_masking", "lr_schedule", "purchase_explore_eps"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}


@dataclass
class FeudalConfig:
    """Feudal RL specific parameters."""

    manager_horizon: int = 10
    worker_reward_alpha: float = 0.5
    manager_lr_scale: float = 1.0
    worker_lr_scale: float = 1.0
    # AlphaStar-style autoregressive worker head with stage-conditional masking.
    autoregressive_worker: bool = False
    # Multiplier on extrinsic reward inside collect_rollout. Default 1.0 keeps
    # behavior unchanged; set << 1 (e.g. 0.001 against ±5000 terminals) to
    # keep value-target magnitudes in a sane range.
    reward_scale: float = 1.0


@dataclass
class SelfPlayConfig:
    """Self-play training parameters."""

    swap_players: bool = True
    opponent_update_freq: int = 10000
    use_opponent_pool: bool = False
    pool_size: int = 10
    pool_strategy: str = "uniform"
    add_to_pool_freq: int = 50000
    min_win_rate_for_pool: float = 0.55
    mixed_training: bool = False
    bot_ratio: float = 0.3
    # Feudal-specific self-play knobs (consumed by train_feudal_rl.py).
    # Snapshot the training agent every N env steps; sample opponents from
    # the rolling pool of the most-recent ``pool_size`` snapshots; evaluate
    # against a fixed opponent so eval scores don't drift with training.
    snapshot_freq: int = 10000
    eval_opponent: str = "random"


@dataclass
class AlphaZeroConfig:
    """AlphaZero-specific parameters."""

    res_blocks: int = 6
    channels: int = 128
    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    iterations: int = 100
    games_per_iter: int = 25
    epochs_per_iter: int = 10
    batch_size: int = 256
    buffer_size: int = 100_000
    max_game_steps: int = 400
    temperature_threshold: int = 30
    eval_games: int = 20
    eval_threshold: float = 0.55
    lr: float = 1e-3
    weight_decay: float = 1e-4


_CURRICULUM_OPPONENTS = (
    "random",
    "balanced_random",
    "simple",
    "bot",
    "medium",
    "mixed",
    "advanced",
    "noop",
)


@dataclass
class CurriculumStage:
    """One curriculum step: a (map, opponent) pair with a promotion criterion.

    The ``max_steps``, ``max_turns``, ``ent_coef``, ``reward_config``, and
    ``opponent_kwargs`` fields are optional per-stage overrides; when ``None``
    the runner falls back to ``cfg.env`` / ``cfg.ppo``. Typical use cases:

    - Bump ``max_turns`` and ``max_steps`` on a larger map (units take more
      turns to traverse).
    - Raise ``ent_coef`` on the first stage of a new map to crack the
      previous stage's policy out of a deterministic groove. Either a
      constant float or a ``{start, end, schedule}`` mapping describing
      a per-stage anneal driven by ``EntropyScheduleCallback``.
    - Override ``reward_config`` keys (merged into the env defaults) when
      a new map's geometry changes which win condition is achievable.
    - Forward extra constructor kwargs to the opponent bot via
      ``opponent_kwargs`` (e.g. ``{max_actions: 10}`` for ``RandomBot``).
    """

    name: str = ""
    map_file: str = ""
    opponent: str = ""
    promotion_win_rate: float = 0.9
    patience: int = 2
    max_timesteps: int = 1_000_000
    n_eval_episodes: int = 30
    # Optional per-stage overrides. None = inherit from cfg.env / cfg.ppo.
    max_steps: Optional[int] = None
    max_turns: Optional[int] = None
    ent_coef: Optional[Union[float, Dict[str, Any]]] = None
    reward_config: Optional[Dict[str, float]] = None
    opponent_kwargs: Optional[Dict[str, Any]] = None
    # Per-stage override for ``ppo.purchase_explore_eps``. Constant float
    # or a ``{start, end, schedule}`` mapping (same layout as ``ent_coef``)
    # that drives :class:`PurchaseExploreScheduleCallback`.
    purchase_explore_eps: Optional[Union[float, Dict[str, Any]]] = None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("stage.name must be non-empty")
        if not self.map_file:
            raise ValueError(f"stage '{self.name}': map_file must be set")
        if not self.opponent:
            raise ValueError(f"stage '{self.name}': opponent must be set")
        if self.opponent not in _CURRICULUM_OPPONENTS:
            raise ValueError(
                f"stage '{self.name}': unknown opponent '{self.opponent}'. Expected one of: {', '.join(_CURRICULUM_OPPONENTS)}"
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
                        raise ValueError(f"stage '{self.name}': ent_coef schedule missing required key '{required}'")
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
        if self.purchase_explore_eps is not None:
            if isinstance(self.purchase_explore_eps, Mapping):
                unknown = set(self.purchase_explore_eps.keys()) - {"start", "end", "schedule"}
                if unknown:
                    raise ValueError(
                        f"stage '{self.name}': purchase_explore_eps schedule has unknown keys {sorted(unknown)}. "
                        "Valid keys: start, end, schedule"
                    )
                for required in ("start", "end"):
                    if required not in self.purchase_explore_eps:
                        raise ValueError(
                            f"stage '{self.name}': purchase_explore_eps schedule missing required key '{required}'"
                        )
                    val = self.purchase_explore_eps[required]
                    if not isinstance(val, (int, float)) or not 0.0 <= float(val) <= 1.0:
                        raise ValueError(
                            f"stage '{self.name}': purchase_explore_eps.{required} must be in [0, 1], got {val!r}"
                        )
                schedule_kind = self.purchase_explore_eps.get("schedule", "linear")
                if schedule_kind not in ("linear", "cosine"):
                    raise ValueError(
                        f"stage '{self.name}': purchase_explore_eps.schedule must be 'linear' or 'cosine', "
                        f"got {schedule_kind!r}"
                    )
            elif isinstance(self.purchase_explore_eps, (int, float)):
                if not 0.0 <= float(self.purchase_explore_eps) <= 1.0:
                    raise ValueError(f"stage '{self.name}': purchase_explore_eps override must be in [0, 1]")
            else:
                raise TypeError(
                    f"stage '{self.name}': purchase_explore_eps must be a number or a "
                    f"{{start, end, schedule}} mapping, got {type(self.purchase_explore_eps).__name__}"
                )
        if self.reward_config is not None and not isinstance(self.reward_config, Mapping):
            raise TypeError(
                f"stage '{self.name}': reward_config override must be a mapping, got {type(self.reward_config).__name__}"
            )
        if self.opponent_kwargs is not None and not isinstance(self.opponent_kwargs, Mapping):
            raise TypeError(
                f"stage '{self.name}': opponent_kwargs override must be a mapping, got {type(self.opponent_kwargs).__name__}"
            )

    def resolve_max_steps(self, env: "EnvConfig") -> int:
        return self.max_steps if self.max_steps is not None else env.max_steps

    def resolve_max_turns(self, env: "EnvConfig") -> Optional[int]:
        return self.max_turns if self.max_turns is not None else env.max_turns

    def resolve_ent_coef(self, ppo: "PPOConfig") -> float:
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

    def resolve_purchase_explore_eps(self, ppo: "PPOConfig") -> float:
        """Return the *initial* purchase-exploration ε for the stage.

        Mirrors :meth:`resolve_ent_coef`: a constant override returns its
        own value; a ``{start, end, schedule}`` mapping returns ``start``
        so the model attribute is seeded before the schedule callback
        takes over; ``None`` falls back to ``ppo.purchase_explore_eps``.
        """
        if self.purchase_explore_eps is None:
            return ppo.purchase_explore_eps
        if isinstance(self.purchase_explore_eps, Mapping):
            return float(self.purchase_explore_eps["start"])
        return float(self.purchase_explore_eps)

    def resolve_purchase_explore_eps_schedule(self) -> Optional[Dict[str, Any]]:
        """Return ``{start, end, schedule}`` if the override is a mapping, else ``None``."""
        if isinstance(self.purchase_explore_eps, Mapping):
            return {
                "start": float(self.purchase_explore_eps["start"]),
                "end": float(self.purchase_explore_eps["end"]),
                "schedule": str(self.purchase_explore_eps.get("schedule", "linear")),
            }
        return None

    def resolve_reward_config(self, env: "EnvConfig") -> Optional[Dict[str, float]]:
        """Return the reward config to use for this stage.

        Per-stage overrides are merged on top of ``env.reward_config``,
        so a stage only needs to spell out the keys it changes. Returns
        ``None`` when neither side has anything (env will fall back to its
        own built-in defaults).
        """
        base = dict(env.reward_config) if env.reward_config else {}
        if self.reward_config:
            base.update(self.reward_config)
        return base if base else None


@dataclass
class CurriculumConfig:
    """Curriculum-bootstrap configuration: an ordered list of stages."""

    stages: List[CurriculumStage] = field(default_factory=list)
    # When True (default), after a stage promotes its best-by-WR
    # checkpoint (``<stage>/best_model.zip``) is reloaded into the
    # in-memory model before the next stage starts, instead of
    # carrying the possibly-drifted end-of-stage policy forward. PPO
    # drifts off the winning attractor *within* a stage after it first
    # clears the bar (the documented draw-with-shaping policy drift);
    # propagating that drifted policy is what made later ``*_random_N``
    # stages unrecoverable (v29 entered random_15 from a drifted
    # post-random_10 policy and stalled; warm-starting random_15 from
    # the peak random_10 snapshot cleared it trivially -- v30). Set
    # False to reproduce the legacy carry-end-of-stage behaviour.
    restore_best_checkpoint_between_stages: bool = True

    def validate(self) -> None:
        seen: set = set()
        for stage in self.stages:
            stage.validate()
            if stage.name in seen:
                raise ValueError(f"duplicate stage name: '{stage.name}'")
            seen.add(stage.name)


@dataclass
class EvalConfig:
    """Evaluation / checkpointing cadence."""

    eval_freq: int = 10000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 50000
    # Offset added to ``cfg.seed`` when constructing the eval env and when
    # seeding per-episode resets inside ``PeriodicEvalCallback``. Keeps eval
    # episodes from sharing seeds with the parallel training envs (which use
    # ``cfg.seed + rank`` for ``rank in range(n_envs)``) and from colliding
    # with the per-episode eval seeds emitted as
    # ``eval_seed_base + 1000 * eval_block + episode_idx``. The default leaves
    # ample headroom for any reasonable n_envs and total_timesteps.
    seed_offset: int = 1_000_000


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""

    log_dir: str = "./logs"
    wandb: bool = False
    wandb_project: str = "reinforcetactics"
    wandb_entity: Optional[str] = None
    tensorboard: bool = True


@dataclass
class TrainingConfig:
    """Root configuration for a training run."""

    algorithm: str = "maskable_ppo"
    total_timesteps: int = 1_000_000
    seed: int = 0
    # Optional path to a saved SB3 model (.zip) whose policy + optimizer
    # parameters are loaded into the freshly-built model before stage-1
    # training. Used for warm-starting a curriculum from a checkpoint of
    # an earlier run (e.g. transplanting a policy that already cleared
    # early stages directly into a later stage). The checkpoint's
    # observation/action spaces must match the curriculum's resolved
    # spaces (same pad_to_size, same enabled_units, same
    # action_space_type). None = cold start from random init.
    warm_start_path: Optional[str] = None
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    feudal: FeudalConfig = field(default_factory=FeudalConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    alphazero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    KNOWN_ALGORITHMS = ("ppo", "maskable_ppo", "feudal", "self_play", "mixed", "alphazero")

    def validate(self) -> None:
        """Raise ``ValueError`` if config is internally inconsistent."""
        if self.algorithm not in self.KNOWN_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. Must be one of {self.KNOWN_ALGORITHMS}")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.env.n_envs <= 0:
            raise ValueError("env.n_envs must be positive")
        if self.env.max_steps <= 0:
            raise ValueError("env.max_steps must be positive")
        if self.env.max_actions_per_turn is not None and self.env.max_actions_per_turn <= 0:
            raise ValueError("env.max_actions_per_turn must be positive (or None to disable)")
        if not 0.0 <= self.ppo.gamma <= 1.0:
            raise ValueError("ppo.gamma must be in [0, 1]")
        if not 0.0 <= self.ppo.gae_lambda <= 1.0:
            raise ValueError("ppo.gae_lambda must be in [0, 1]")
        if self.ppo.batch_size <= 0:
            raise ValueError("ppo.batch_size must be positive")
        if self.ppo.n_steps <= 0:
            raise ValueError("ppo.n_steps must be positive")
        if self.env.action_space_type not in ("multi_discrete", "flat_discrete"):
            raise ValueError(
                f"env.action_space_type must be 'multi_discrete' or 'flat_discrete', got '{self.env.action_space_type}'"
            )
        if self.self_play.pool_strategy not in ("uniform", "recent", "prioritized"):
            raise ValueError("self_play.pool_strategy must be 'uniform', 'recent', or 'prioritized'")
        if not 0.0 <= self.self_play.min_win_rate_for_pool <= 1.0:
            raise ValueError("self_play.min_win_rate_for_pool must be in [0, 1]")
        self.curriculum.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (suitable for JSON/YAML dumping)."""
        return asdict(self)


_SECTION_TYPES = {
    "env": EnvConfig,
    "ppo": PPOConfig,
    "feudal": FeudalConfig,
    "self_play": SelfPlayConfig,
    "alphazero": AlphaZeroConfig,
    "curriculum": CurriculumConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
}


def _build_curriculum(raw: Any) -> CurriculumConfig:
    """Build CurriculumConfig from a raw mapping, deserializing nested stages."""
    if raw is None:
        return CurriculumConfig()
    if not isinstance(raw, Mapping):
        raise TypeError(f"Section 'curriculum' must be a mapping, got {type(raw).__name__}")
    valid_fields = {f.name for f in fields(CurriculumConfig)}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        raise ValueError(f"Unknown keys in section 'curriculum': {sorted(unknown)}. Valid keys: {sorted(valid_fields)}")
    raw_stages = raw.get("stages") or []
    if not isinstance(raw_stages, list):
        raise TypeError(f"'curriculum.stages' must be a list, got {type(raw_stages).__name__}")
    stage_fields = {f.name for f in fields(CurriculumStage)}
    stages: List[CurriculumStage] = []
    for i, s in enumerate(raw_stages):
        if not isinstance(s, Mapping):
            raise TypeError(f"curriculum.stages[{i}] must be a mapping, got {type(s).__name__}")
        unknown_stage = set(s.keys()) - stage_fields
        if unknown_stage:
            raise ValueError(
                f"Unknown keys for CurriculumStage at index {i}: {sorted(unknown_stage)}. Valid keys: {sorted(stage_fields)}"
            )
        stages.append(CurriculumStage(**{k: v for k, v in s.items() if k in stage_fields}))
    kwargs: Dict[str, Any] = {"stages": stages}
    if "restore_best_checkpoint_between_stages" in raw:
        kwargs["restore_best_checkpoint_between_stages"] = bool(raw["restore_best_checkpoint_between_stages"])
    return CurriculumConfig(**kwargs)


def _build_section(section_name: str, raw: Any):
    """Instantiate a typed section from raw data, validating unknown fields."""
    if section_name == "curriculum":
        return _build_curriculum(raw)
    if raw is None:
        return _SECTION_TYPES[section_name]()
    if not isinstance(raw, Mapping):
        raise TypeError(f"Section '{section_name}' must be a mapping, got {type(raw).__name__}")
    cls = _SECTION_TYPES[section_name]
    valid_fields = {f.name for f in fields(cls)}
    unknown = set(raw.keys()) - valid_fields
    if unknown:
        raise ValueError(f"Unknown keys in section '{section_name}': {sorted(unknown)}. Valid keys: {sorted(valid_fields)}")
    return cls(**{k: v for k, v in raw.items() if k in valid_fields})


def config_from_dict(data: Mapping[str, Any]) -> TrainingConfig:
    """Construct a :class:`TrainingConfig` from a plain dict."""
    if not isinstance(data, Mapping):
        raise TypeError(f"Config data must be a mapping, got {type(data).__name__}")

    top_level_scalars = {"algorithm", "total_timesteps", "seed", "warm_start_path"}
    valid_keys = top_level_scalars | set(_SECTION_TYPES)
    unknown = set(data.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown)}. Valid keys: {sorted(valid_keys)}")

    kwargs: Dict[str, Any] = {}
    for key in top_level_scalars:
        if key in data:
            kwargs[key] = data[key]
    for section_name in _SECTION_TYPES:
        kwargs[section_name] = _build_section(section_name, data.get(section_name))

    cfg = TrainingConfig(**kwargs)
    cfg.validate()
    return cfg


def _read_config_file(path: Path) -> Dict[str, Any]:
    """Read a YAML or JSON config file into a dict."""
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError(
                f"Cannot load '{path}': PyYAML is not installed. Install with `pip install PyYAML` or use a .json config."
            )
        data = yaml.safe_load(text) or {}
    elif suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise ValueError(f"Unsupported config extension '{suffix}' for {path}. Use .yaml, .yml, or .json.")
    if not isinstance(data, Mapping):
        raise TypeError(f"Config file {path} must contain a mapping at the top level.")
    return dict(data)


def load_config(path: ConfigPath) -> TrainingConfig:
    """Load and validate a training config from a YAML or JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    return config_from_dict(_read_config_file(p))


def save_config(cfg: TrainingConfig, path: ConfigPath) -> None:
    """Dump a config to YAML (``.yaml``/``.yml``) or JSON (``.json``)."""
    p = Path(path)
    data = cfg.to_dict()
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is not installed; save as .json instead.")
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    elif suffix == ".json":
        p.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported config extension '{suffix}' for {p}")


def _coerce_value(current: Any, new: str) -> Any:
    """Best-effort coercion of a string override to the type of ``current``."""
    if isinstance(new, str):
        if isinstance(current, bool):
            lowered = new.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"Cannot parse '{new}' as bool")
        if isinstance(current, int) and not isinstance(current, bool):
            return int(new)
        if isinstance(current, float):
            return float(new)
        if current is None:
            # Try int, then float, else leave as string
            for conv in (int, float):
                try:
                    return conv(new)
                except (TypeError, ValueError):
                    continue
            return new
    return new


def _set_nested(cfg: TrainingConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target: Any = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"Unknown config key segment: '{part}' in '{dotted_key}'")
        target = getattr(target, part)
        if not is_dataclass(target):
            raise KeyError(f"'{part}' in '{dotted_key}' does not point to a config section")
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise KeyError(f"Unknown config key: '{dotted_key}'")
    current = getattr(target, leaf)
    setattr(target, leaf, _coerce_value(current, value))


def config_to_argparse_defaults(
    cfg: TrainingConfig,
    mapping: Mapping[str, str],
) -> Dict[str, Any]:
    """Flatten a config to a dict suitable for ``parser.set_defaults(**d)``.

    Args:
        cfg: Loaded training config.
        mapping: Maps argparse ``dest`` names to dotted config paths, e.g.
            ``{"learning_rate": "ppo.learning_rate", "seed": "seed"}``.

    Missing paths are silently skipped so scripts can share a mapping but
    declare only a subset of fields.
    """
    defaults: Dict[str, Any] = {}
    for arg_name, dotted_path in mapping.items():
        parts = dotted_path.split(".")
        try:
            val: Any = cfg
            for p in parts:
                val = getattr(val, p)
        except AttributeError:
            continue
        defaults[arg_name] = val
    return defaults


def apply_overrides(
    cfg: TrainingConfig,
    overrides: Optional[Mapping[str, Any]] = None,
) -> TrainingConfig:
    """Return a copy of ``cfg`` with dotted-key overrides applied.

    ``None`` values in ``overrides`` are ignored so that ``argparse`` defaults
    don't clobber file-provided values. Use the sentinel string ``"null"`` to
    force a field to ``None`` when needed.
    """
    new_cfg = copy.deepcopy(cfg)
    if not overrides:
        return new_cfg
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "null":
            value = None
        _set_nested(new_cfg, key, value)
    new_cfg.validate()
    return new_cfg
