"""
Training configuration loader for Reinforce Tactics RL.

Centralizes hyperparameters in YAML/JSON files so training runs are
reproducible without editing source. Supports:

- Loading from ``.yaml``, ``.yml``, or ``.json`` files
- Hierarchical sections: ``env``, ``ppo``, ``feudal``, ``self_play``,
  ``alphazero``, ``eval``, ``logging``
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
from typing import Any, Dict, List, Mapping, Optional, Union

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
    fog_of_war: bool = False
    enabled_units: Optional[List[str]] = None
    action_space_type: str = "multi_discrete"
    max_flat_actions: int = 512
    reward_config: Optional[Dict[str, float]] = None
    n_envs: int = 4
    use_subprocess: bool = True


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

    def as_sb3_kwargs(self) -> Dict[str, Any]:
        """Return the subset of fields accepted by PPO/MaskablePPO __init__."""
        skip = {"use_action_masking"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}


@dataclass
class FeudalConfig:
    """Feudal RL specific parameters."""

    manager_horizon: int = 10
    worker_reward_alpha: float = 0.5
    manager_lr_scale: float = 1.0
    worker_lr_scale: float = 1.0


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


@dataclass
class EvalConfig:
    """Evaluation / checkpointing cadence."""

    eval_freq: int = 10000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 50000


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
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    feudal: FeudalConfig = field(default_factory=FeudalConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    alphazero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (suitable for JSON/YAML dumping)."""
        return asdict(self)


_SECTION_TYPES = {
    "env": EnvConfig,
    "ppo": PPOConfig,
    "feudal": FeudalConfig,
    "self_play": SelfPlayConfig,
    "alphazero": AlphaZeroConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
}


def _build_section(section_name: str, raw: Any):
    """Instantiate a typed section from raw data, validating unknown fields."""
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

    top_level_scalars = {"algorithm", "total_timesteps", "seed"}
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
