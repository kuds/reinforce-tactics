"""
Training configuration loader for Reinforce Tactics RL.

Centralizes hyperparameters in YAML/JSON files so training runs are
reproducible without editing source. Supports:

- Loading from ``.yaml``, ``.yml``, or ``.json`` files
- Hierarchical sections: ``env``, ``ppo``, ``feudal``, ``self_play``,
  ``alphazero``, ``curriculum``, ``eval``, ``logging``
- CLI overrides: values passed via ``--key value`` beat file values
- Dotted override keys (``ppo.learning_rate=1e-4``) for nested updates
- Dataclass validation with typed sections: every value is coerced to its
  field's annotated type (so YAML's ``3e-4``, which PyYAML reads as a
  string, becomes a float) and range-checked, reward_config keys are checked
  against the env's :data:`KNOWN_REWARD_KEYS`, and opponents against the bot
  registry (review rltrain-10, rltrain-22)
- :func:`check_ignored_config_fields`: an entry point declares the fields it
  reads, and a config that sets any other field away from its default gets
  a warning (an error under ``--strict``) instead of silently doing nothing
  (review rltrain-9)

Usage:
    from reinforcetactics.rl.config import load_config, apply_overrides

    cfg = load_config("configs/ppo/maskable_ppo.yaml")
    cfg = apply_overrides(cfg, {"ppo.learning_rate": 1e-4})
    model = MaskablePPO(**cfg.ppo.as_sb3_kwargs(), env=env)
"""

from __future__ import annotations

import copy
import json
import math
import numbers
import types
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from reinforcetactics.game.bot_registry import accepted_names as accepted_bot_names
from reinforcetactics.game.bot_registry import is_scripted_name
from reinforcetactics.rl.env_schema import (
    resolve_opponent,
    validate_opponent_kwargs,
    validate_reward_config,
)
from reinforcetactics.rl.gym_env import FLAT_ACTION_VERSIONS

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a required dev dep
    yaml = None


ConfigPath = str | Path


@dataclass
class EnvConfig:
    """Environment construction parameters."""

    map_file: str | None = None
    # ``None``, ``'self'`` or a bot-registry name
    # (``bot_registry.accepted_names()``); validated in
    # :meth:`TrainingConfig.validate`. Curriculum runs ignore it: each stage
    # names its own opponent.
    opponent: str = "bot"
    max_steps: int = 200
    max_turns: int | None = None
    fog_of_war: bool = False
    enabled_units: list[str] | None = None
    action_space_type: str = "multi_discrete"
    max_flat_actions: int = 512
    # flat_discrete decode-table layout (``gym_env.FLAT_ACTION_VERSIONS``;
    # it only matters once the legal set exceeds ``max_flat_actions``).
    # ``None`` (default) means: the version of the checkpoint a run warm
    # starts or resumes from, so its policy keeps the table it was trained
    # on, else ``gym_env.FLAT_ACTION_VERSION_LATEST``. The resolved value is
    # what ``resolved_config.yaml`` and each stage's ``config.json`` record.
    flat_action_version: int | None = None
    # Optional hard cap on agent actions per game-turn. When set, the
    # action mask narrows to end_turn-only once the agent has executed
    # this many actions in the current game-turn. Defends against the
    # "never end the turn" stall mode where the policy cycles through
    # legal-but-unproductive actions until ``max_steps`` truncates the
    # episode. ``None`` (default) disables the cap. See
    # :class:`reinforcetactics.rl.gym_env.StrategyGameEnv` for details.
    max_actions_per_turn: int | None = None
    reward_config: dict[str, float] | None = None
    # Optional sparse overlay over the non-YAML engine constants
    # (``rules.py``): ``starting_gold``, ``headquarters_income``,
    # ``building_income``, ``tower_income``, and ``unit_data``
    # (``{CODE: {field: value}}`` per-unit, per-field deltas). Absent /
    # ``None`` = use the module constants (today's behaviour). Makes
    # balance a first-class, swept, auto-recorded config axis instead of
    # an invisible engine constant that only a git checkout could change.
    # Resolved by ``GameState`` (its tables are the per-game source of
    # truth) and snapshotted into ``config.json``.
    engine_overrides: dict[str, Any] | None = None
    n_envs: int = 4
    use_subprocess: bool = True
    # Optional ``(pad_h, pad_w)`` for cross-stage observation-shape unification.
    # When the curriculum mixes maps of different sizes, the bootstrap runner
    # auto-fills this with the curriculum-wide max so a single PPO policy can
    # train across all stages without an observation-space mismatch. Set
    # explicitly to override the auto-computed value (e.g. to leave headroom
    # for a future larger map). Only honoured by ``flat_discrete``.
    pad_to_size: tuple[int, int] | None = None
    # ``global_features`` tanh normalization divisors. Defaults match the
    # module-level constants in ``reinforcetactics.rl.observation`` and
    # are tuned for the current curriculum's gold / turn / army-size
    # ranges. Override on a per-run basis when shipping a map / economy
    # whose typical values differ enough that the linear regime of
    # ``tanh`` no longer covers the relevant operating point.
    gold_scale: float = 1000.0
    turn_scale: float = 60.0
    unit_count_scale: float = 20.0
    # Extra kwargs forwarded to the opponent bot's constructor (e.g.
    # ``{max_actions: 10}`` for ``RandomBot``, ``{easy: simple, hard: medium,
    # p_hard: 0.5}`` for ``MixedBot``). For the curriculum bootstrap path the
    # per-stage :attr:`CurriculumStage.opponent_kwargs` takes precedence; for
    # non-curriculum runs (feudal, flat baseline) this is the only knob.
    opponent_kwargs: dict[str, Any] | None = None


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
    policy_kwargs: dict[str, Any] | None = None
    # Probability with which a sampled ``create_unit`` action has its
    # ``unit_type`` sub-action resampled uniformly over the env's
    # currently-legal (enabled + affordable) unit types. Pure exploration
    # knob: ε=0 disables, ε=1 always randomizes purchases. The substituted
    # action is what gets executed *and* what gets stored in the rollout
    # buffer (with log-prob recomputed under the masked policy at the
    # substituted action), so PPO's ratio stays internally consistent.
    # Per-stage overrides live on :class:`CurriculumStage`.
    purchase_explore_eps: float = 0.0

    def as_sb3_kwargs(self) -> dict[str, Any]:
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
    # Once the opponent pool holds any snapshot, each self-play episode's
    # opponent is drawn at reset: the latest snapshot (the one
    # ``opponent_update_freq`` pushes) with this probability, otherwise a
    # pool sample. The default 0.0 is the long-standing behaviour: with a
    # non-empty pool every episode plays a pool sample, and the latest
    # snapshot only plays out the episodes already running when it is
    # pushed. 1.0 always plays the latest snapshot (the pool then only
    # records history). Consumed by train_self_play.py (``SelfPlayEnv``).
    latest_opponent_prob: float = 0.0
    mixed_training: bool = False
    bot_ratio: float = 0.3
    # Feudal-specific self-play knobs (consumed by train_feudal_rl.py).
    # Snapshot the training agent every N env steps; sample opponents from
    # the rolling pool of the most-recent ``pool_size`` snapshots; evaluate
    # against a fixed opponent so eval scores don't drift with training.
    snapshot_freq: int = 10000
    # A scripted bot (``bot_registry.accepted_names()``): an eval opponent
    # that moves with the learner says nothing about progress.
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


# The scripted opponents a curriculum stage may name: derived from the bot
# registry (every SCRIPTED_BOTS entry plus the "bot" alias), so a new bot is
# accepted here as soon as it is registered. This used to be a hand-written
# copy that had already drifted: it rejected "master" (review rltrain-22).
# "self" is not a curriculum opponent: the runner has no self-play wrapper,
# and a bare 'self' env plays no opponent at all.
_CURRICULUM_OPPONENTS: tuple[str, ...] = accepted_bot_names()


# ---------------------------------------------------------------------------
# Type coercion (review rltrain-10)
#
# YAML, JSON and ``--set`` hand the loader raw values: PyYAML reads ``3e-4``
# and even ``1.0e6`` as *strings* (YAML 1.1 wants ``3.0e-4`` / ``1.0e+6``),
# and nothing stopped ``n_envs: 4.5`` or ``fog_of_war: "false"`` (a truthy
# string) from reaching the trainer. Every field is coerced to its annotated
# type instead: numeric strings become numbers, integral floats become ints,
# ``"true"``/``"false"`` become bools, sequences become lists / tuples, and
# anything that does not fit raises with the field's dotted path.
# ---------------------------------------------------------------------------

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


def _type_label(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _coerce_to(value: Any, tp: Any, where: str) -> Any:
    """Coerce a raw config value to the annotated type ``tp``.

    Raises:
        TypeError: The value has the wrong shape (a list for a mapping, a
            bool for a number, ``None`` for a required field, ...).
        ValueError: A string that does not parse, a non-finite or
            non-integral number, or a tuple of the wrong length.
    """
    if tp is Any:
        return value
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        members = get_args(tp)
        if value is None:
            if type(None) in members:
                return None
            raise TypeError(f"{where} must not be null")
        candidates = [m for m in members if m is not type(None)]
        # A mapping can only mean the dict member; anything else never does.
        wants_dict = isinstance(value, Mapping)
        candidates = [m for m in candidates if (get_origin(m) is dict) == wants_dict] or candidates
        errors: list[Exception] = []
        for member in candidates:
            try:
                return _coerce_to(value, member, where)
            except (TypeError, ValueError) as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        raise TypeError(f"{where} must be {_type_label(tp)}, got {value!r} ({type(value).__name__})")
    if value is None:
        raise TypeError(f"{where} must be {_type_label(tp)}, got null")
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUE_STRINGS:
                return True
            if lowered in _FALSE_STRINGS:
                return False
            raise ValueError(f"Cannot parse {value!r} as bool for {where}")
        raise TypeError(f"{where} must be a bool, got {value!r} ({type(value).__name__})")
    if tp is int:
        if isinstance(value, bool):
            raise TypeError(f"{where} must be an integer, got {value!r} (bool)")
        if isinstance(value, numbers.Integral):
            return int(value)
        number = value
        if isinstance(value, str):
            text = value.strip()
            try:
                return int(text)
            except ValueError:
                try:
                    number = float(text)
                except ValueError:
                    raise ValueError(f"{where} must be an integer, got {value!r}") from None
        if isinstance(number, numbers.Real) and math.isfinite(float(number)) and float(number).is_integer():
            return int(float(number))
        raise ValueError(f"{where} must be an integer, got {value!r}")
    if tp is float:
        if isinstance(value, bool):
            raise TypeError(f"{where} must be a number, got {value!r} (bool)")
        if isinstance(value, numbers.Real):
            number = float(value)
        elif isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                raise ValueError(f"{where} must be a number, got {value!r}") from None
        else:
            raise TypeError(f"{where} must be a number, got {value!r} ({type(value).__name__})")
        if not math.isfinite(number):
            raise ValueError(f"{where} must be finite, got {value!r}")
        return number
    if tp is str:
        if isinstance(value, str):
            return value
        raise TypeError(f"{where} must be a string, got {value!r} ({type(value).__name__})")
    if origin in (list, tuple):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{where} must be a list, got {value!r} ({type(value).__name__})")
        args = get_args(tp)
        if origin is list:
            elem = args[0] if args else Any
            return [_coerce_to(v, elem, f"{where}[{i}]") for i, v in enumerate(value)]
        if args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError(f"{where} must have {len(args)} entries, got {len(value)}: {list(value)!r}")
            return tuple(_coerce_to(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args, strict=True)))
        elem = args[0] if args else Any
        return tuple(_coerce_to(v, elem, f"{where}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{where} must be a mapping, got {value!r} ({type(value).__name__})")
        key_t, val_t = get_args(tp) or (Any, Any)
        return {_coerce_to(k, key_t, f"{where} key {k!r}"): _coerce_to(v, val_t, f"{where}[{k!r}]") for k, v in value.items()}
    return value


_FIELD_TYPES: dict[type, dict[str, Any]] = {}


def _field_types(cls: type) -> dict[str, Any]:
    """``get_type_hints(cls)``, cached per dataclass."""
    hints = _FIELD_TYPES.get(cls)
    if hints is None:
        hints = _FIELD_TYPES[cls] = get_type_hints(cls)
    return hints


def _normalize_fields(obj: Any, prefix: str) -> None:
    """Coerce every non-section field of dataclass ``obj`` in place (see :func:`_coerce_to`)."""
    hints = _field_types(type(obj))
    for f in fields(obj):
        tp = hints[f.name]
        if isinstance(tp, type) and is_dataclass(tp):
            continue  # a nested section, normalized by its own validate()
        if get_origin(tp) is list and any(isinstance(a, type) and is_dataclass(a) for a in get_args(tp)):
            continue  # curriculum.stages: each stage normalizes itself
        value = getattr(obj, f.name)
        coerced = _coerce_to(value, tp, f"{prefix}{f.name}")
        if coerced is not value:
            setattr(obj, f.name, coerced)


def _normalize_schedule(value: Any, where: str) -> Any:
    """Coerce the numeric ``start`` / ``end`` of a ``{start, end, schedule}`` mapping."""
    if not isinstance(value, Mapping):
        return value
    out = dict(value)
    for key in ("start", "end"):
        if key in out:
            try:
                out[key] = _coerce_to(out[key], float, f"{where}.{key}")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{where}.{key} must be a non-negative number, got {out[key]!r} ({exc})") from None
    return out


def _check_reward_config(reward_config: Any, where: str) -> None:
    try:
        validate_reward_config(reward_config)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{where}: {exc}") from None


def _check_opponent(opponent: Any, opponent_kwargs: Any, where: str, *, scripted_only: bool = False) -> None:
    """``opponent`` must be one the env plays, and ``opponent_kwargs`` must suit it."""
    if scripted_only and not (isinstance(opponent, str) and is_scripted_name(opponent)):
        raise ValueError(
            f"{where}: unknown opponent {opponent!r}. Expected one of: {', '.join(accepted_bot_names())} "
            "(see reinforcetactics.game.bot_registry)"
        )
    try:
        resolve_opponent(opponent)
        validate_opponent_kwargs(opponent, opponent_kwargs)
    except (TypeError, ValueError, KeyError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise type(exc)(f"{where}: {message}") from None


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


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
    # Minimum env-steps the stage MUST train before promotion can fire,
    # measured *within the stage* (from the start of the stage's
    # ``learn()`` call — ``num_timesteps`` itself is cumulative across
    # stages under ``reset_num_timesteps=False``). Defends against the
    # "skip-ahead" failure where a strong carry-in policy (e.g. from the
    # prior stage's best-checkpoint handoff) passes the WR threshold on
    # the very first eval -- so the stage promotes immediately,
    # contributing ~0 stage-specific learning, and the next harder stage
    # inherits an under-trained policy that collapses (the v28
    # 20260522_163958 random_15 stall: random_10 promoted from a @250k
    # snapshot that was essentially balanced_random's best with no
    # random_10 refinement). When > 0, the promotion callback resets its
    # streak counter and ignores eval results until the stage has trained
    # ``min_timesteps_before_promotion`` env steps. Default 0 preserves
    # legacy behaviour; set on the noisy ``*_random_N`` stages where it
    # matters most.
    min_timesteps_before_promotion: int = 0
    # Eval episodes per eval for this stage. ``None`` (default) inherits
    # ``cfg.eval.n_eval_episodes`` (:meth:`resolve_n_eval_episodes`). It used
    # to default to a hidden 30 that ignored ``eval.n_eval_episodes``
    # entirely (review rltrain-9).
    n_eval_episodes: int | None = None
    # Optional per-stage overrides. None = inherit from cfg.env / cfg.ppo.
    max_steps: int | None = None
    max_turns: int | None = None
    ent_coef: float | dict[str, Any] | None = None
    reward_config: dict[str, float] | None = None
    opponent_kwargs: dict[str, Any] | None = None
    # Per-stage override for ``ppo.purchase_explore_eps``. Constant float
    # or a ``{start, end, schedule}`` mapping (same layout as ``ent_coef``)
    # that drives :class:`PurchaseExploreScheduleCallback`.
    purchase_explore_eps: float | dict[str, Any] | None = None

    def validate(self) -> None:
        """Coerce every field to its annotated type, then check values.

        Checks the opponent against the bot registry, ``opponent_kwargs``
        against that bot's constructor (MixedBot's values in depth), and
        ``reward_config`` keys against the env's ``KNOWN_REWARD_KEYS``.
        """
        _normalize_fields(self, f"stage '{self.name}': ")
        self.ent_coef = _normalize_schedule(self.ent_coef, f"stage '{self.name}': ent_coef")
        self.purchase_explore_eps = _normalize_schedule(
            self.purchase_explore_eps, f"stage '{self.name}': purchase_explore_eps"
        )
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
        if self.min_timesteps_before_promotion < 0:
            raise ValueError(f"stage '{self.name}': min_timesteps_before_promotion must be >= 0")
        if self.min_timesteps_before_promotion > self.max_timesteps:
            raise ValueError(
                f"stage '{self.name}': min_timesteps_before_promotion "
                f"({self.min_timesteps_before_promotion}) must be <= max_timesteps "
                f"({self.max_timesteps}); otherwise the stage can never promote."
            )
        if self.n_eval_episodes is not None and self.n_eval_episodes <= 0:
            raise ValueError(f"stage '{self.name}': n_eval_episodes must be > 0 (or null to inherit eval.n_eval_episodes)")
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
        _check_reward_config(self.reward_config, f"stage '{self.name}': reward_config")
        # Kwargs the stage's bot does not take (anything given for the
        # deterministic ladder, say) used to be dropped by the env without a
        # word; MixedBot's inner names, p_hard and nested kwargs are checked
        # here rather than at the reset whose coin flip first picks them.
        _check_opponent(self.opponent, self.opponent_kwargs, f"stage '{self.name}'", scripted_only=True)

    def resolve_n_eval_episodes(self, eval_cfg: EvalConfig) -> int:
        """Eval episodes for this stage: its own override, else ``eval.n_eval_episodes``."""
        return self.n_eval_episodes if self.n_eval_episodes is not None else eval_cfg.n_eval_episodes

    def resolve_max_steps(self, env: EnvConfig) -> int:
        return self.max_steps if self.max_steps is not None else env.max_steps

    def resolve_max_turns(self, env: EnvConfig) -> int | None:
        return self.max_turns if self.max_turns is not None else env.max_turns

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

    def resolve_ent_coef_schedule(self) -> dict[str, Any] | None:
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

    def resolve_purchase_explore_eps(self, ppo: PPOConfig) -> float:
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

    def resolve_purchase_explore_eps_schedule(self) -> dict[str, Any] | None:
        """Return ``{start, end, schedule}`` if the override is a mapping, else ``None``."""
        if isinstance(self.purchase_explore_eps, Mapping):
            return {
                "start": float(self.purchase_explore_eps["start"]),
                "end": float(self.purchase_explore_eps["end"]),
                "schedule": str(self.purchase_explore_eps.get("schedule", "linear")),
            }
        return None

    def resolve_reward_config(self, env: EnvConfig) -> dict[str, float] | None:
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

    stages: list[CurriculumStage] = field(default_factory=list)
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
        _normalize_fields(self, "curriculum.")
        seen: set = set()
        for stage in self.stages:
            if not isinstance(stage, CurriculumStage):
                raise TypeError(f"curriculum.stages entries must be CurriculumStage, got {type(stage).__name__}")
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
    # Redraw the eval problem set on every eval block instead of replaying a
    # fixed one. The fixed set (default) is what makes ``patience`` consecutive
    # crossings and the ``best_model.zip`` argmax comparable across evals;
    # resampling turns both into measurements of benchmark noise. Kept as an
    # escape hatch for anyone who wants the pre-2026-07-24 behaviour or is
    # worried about overfitting to a fixed eval set.
    resample_eval_seeds: bool = False
    # Stage-relative env steps before an eval may claim ``best_model.zip``.
    # ``None`` resolves to ``eval_freq`` (skip the stage-entry warm eval and
    # any block-boundary eval immediately after it). Set 0 to let the carry-in
    # policy compete for the stage's best checkpoint, as it used to.
    best_eligible_after: int | None = None


@dataclass
class LoggingConfig:
    """Logging and experiment tracking."""

    log_dir: str = "./logs"
    wandb: bool = False
    wandb_project: str = "reinforcetactics"
    wandb_entity: str | None = None
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
    warm_start_path: str | None = None
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    feudal: FeudalConfig = field(default_factory=FeudalConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    alphazero: AlphaZeroConfig = field(default_factory=AlphaZeroConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    KNOWN_ALGORITHMS = ("ppo", "maskable_ppo", "feudal", "self_play", "mixed", "alphazero")

    def validate(self, *, check_files: bool = False) -> None:
        """Coerce every field to its annotated type, then check ranges and cross-field rules.

        Raises ``ValueError`` (``TypeError`` for a value of the wrong shape)
        if the config is internally inconsistent. Values are normalized in
        place: ``"3e-4"`` becomes ``0.0003``, ``pad_to_size: [10, 12]``
        becomes ``(10, 12)``.

        Args:
            check_files: Also require ``warm_start_path`` (when set) to
                exist. Off by default because a config may legitimately be
                loaded before its checkpoint exists (``build_bc_warmstart.py``
                reads the curriculum config that will later warm start from
                the checkpoint it builds, and the shipped BC configs carry a
                placeholder path); :func:`run_curriculum` and the entry points
                turn it on before building any env.
        """
        self._normalize()
        env, ppo, ev, sp, fd, az = self.env, self.ppo, self.eval, self.self_play, self.feudal, self.alphazero

        _require(
            self.algorithm in self.KNOWN_ALGORITHMS,
            f"Unknown algorithm '{self.algorithm}'. Must be one of {self.KNOWN_ALGORITHMS}",
        )
        _require(self.total_timesteps > 0, "total_timesteps must be positive")
        _require(self.seed >= 0, f"seed must be >= 0, got {self.seed}")
        if check_files and self.warm_start_path and not Path(self.warm_start_path).is_file():
            raise FileNotFoundError(
                f"warm_start_path '{self.warm_start_path}' does not exist. "
                "Provide a valid SB3 .zip checkpoint or unset warm_start_path for a cold start."
            )

        # -- env ------------------------------------------------------------
        _require(env.n_envs > 0, "env.n_envs must be positive")
        _require(env.max_steps > 0, "env.max_steps must be positive")
        _require(env.max_turns is None or env.max_turns > 0, "env.max_turns must be positive (or null for no limit)")
        _require(
            env.max_actions_per_turn is None or env.max_actions_per_turn > 0,
            "env.max_actions_per_turn must be positive (or None to disable)",
        )
        _require(
            env.action_space_type in ("multi_discrete", "flat_discrete"),
            f"env.action_space_type must be 'multi_discrete' or 'flat_discrete', got '{env.action_space_type}'",
        )
        _require(env.max_flat_actions >= 1, f"env.max_flat_actions must be >= 1, got {env.max_flat_actions}")
        _require(
            env.flat_action_version is None or env.flat_action_version in FLAT_ACTION_VERSIONS,
            f"env.flat_action_version must be one of {FLAT_ACTION_VERSIONS} (or null), got {env.flat_action_version}",
        )
        _require(
            env.pad_to_size is None or all(v > 0 for v in env.pad_to_size),
            f"env.pad_to_size must be two positive integers (height, width), got {env.pad_to_size}",
        )
        for name in ("gold_scale", "turn_scale", "unit_count_scale"):
            _require(getattr(env, name) > 0, f"env.{name} must be > 0 (it divides a tanh input), got {getattr(env, name)}")
        _check_opponent(env.opponent, env.opponent_kwargs, "env.opponent")
        _check_reward_config(env.reward_config, "env.reward_config")

        # -- ppo ------------------------------------------------------------
        _require(ppo.learning_rate > 0, f"ppo.learning_rate must be > 0, got {ppo.learning_rate}")
        _require(ppo.n_steps > 0, "ppo.n_steps must be positive")
        _require(ppo.batch_size > 0, "ppo.batch_size must be positive")
        _require(ppo.n_epochs > 0, f"ppo.n_epochs must be positive, got {ppo.n_epochs}")
        _require(0.0 <= ppo.gamma <= 1.0, "ppo.gamma must be in [0, 1]")
        _require(0.0 <= ppo.gae_lambda <= 1.0, "ppo.gae_lambda must be in [0, 1]")
        _require(ppo.clip_range > 0, f"ppo.clip_range must be > 0, got {ppo.clip_range}")
        _require(ppo.ent_coef >= 0, f"ppo.ent_coef must be >= 0, got {ppo.ent_coef}")
        _require(ppo.vf_coef >= 0, f"ppo.vf_coef must be >= 0, got {ppo.vf_coef}")
        _require(ppo.max_grad_norm > 0, f"ppo.max_grad_norm must be > 0, got {ppo.max_grad_norm}")
        _require(
            ppo.lr_schedule in ("constant", "linear"),
            f"ppo.lr_schedule must be 'constant' or 'linear', got {ppo.lr_schedule!r}",
        )
        _require(
            0.0 <= ppo.purchase_explore_eps <= 1.0,
            f"ppo.purchase_explore_eps must be in [0, 1], got {ppo.purchase_explore_eps}",
        )

        # -- eval -----------------------------------------------------------
        _require(ev.eval_freq > 0, f"eval.eval_freq must be > 0, got {ev.eval_freq}")
        _require(ev.n_eval_episodes > 0, f"eval.n_eval_episodes must be > 0, got {ev.n_eval_episodes}")
        _require(ev.checkpoint_freq > 0, f"eval.checkpoint_freq must be > 0, got {ev.checkpoint_freq}")
        _require(ev.seed_offset >= 0, f"eval.seed_offset must be >= 0, got {ev.seed_offset}")
        _require(
            ev.best_eligible_after is None or ev.best_eligible_after >= 0,
            f"eval.best_eligible_after must be >= 0 (or null), got {ev.best_eligible_after}",
        )

        # -- self_play ------------------------------------------------------
        _require(
            sp.pool_strategy in ("uniform", "recent", "prioritized"),
            "self_play.pool_strategy must be 'uniform', 'recent', or 'prioritized'",
        )
        _require(0.0 <= sp.min_win_rate_for_pool <= 1.0, "self_play.min_win_rate_for_pool must be in [0, 1]")
        _require(
            0.0 <= sp.latest_opponent_prob <= 1.0,
            f"self_play.latest_opponent_prob must be in [0, 1], got {sp.latest_opponent_prob}",
        )
        _require(0.0 <= sp.bot_ratio < 1.0, f"self_play.bot_ratio must be in [0, 1), got {sp.bot_ratio}")
        for name in ("opponent_update_freq", "pool_size", "add_to_pool_freq", "snapshot_freq"):
            _require(getattr(sp, name) > 0, f"self_play.{name} must be positive, got {getattr(sp, name)}")
        _check_opponent(sp.eval_opponent, None, "self_play.eval_opponent", scripted_only=True)

        # -- feudal ---------------------------------------------------------
        _require(fd.manager_horizon > 0, f"feudal.manager_horizon must be positive, got {fd.manager_horizon}")
        _require(
            0.0 <= fd.worker_reward_alpha <= 1.0, f"feudal.worker_reward_alpha must be in [0, 1], got {fd.worker_reward_alpha}"
        )
        for name in ("manager_lr_scale", "worker_lr_scale", "reward_scale"):
            _require(getattr(fd, name) > 0, f"feudal.{name} must be > 0, got {getattr(fd, name)}")

        # -- alphazero ------------------------------------------------------
        for name in (
            "res_blocks",
            "channels",
            "num_simulations",
            "iterations",
            "games_per_iter",
            "epochs_per_iter",
            "batch_size",
            "buffer_size",
            "max_game_steps",
            "eval_games",
        ):
            _require(getattr(az, name) > 0, f"alphazero.{name} must be positive, got {getattr(az, name)}")
        _require(
            az.temperature_threshold >= 0, f"alphazero.temperature_threshold must be >= 0, got {az.temperature_threshold}"
        )
        _require(az.c_puct > 0, f"alphazero.c_puct must be > 0, got {az.c_puct}")
        _require(az.dirichlet_alpha > 0, f"alphazero.dirichlet_alpha must be > 0, got {az.dirichlet_alpha}")
        _require(0.0 <= az.eval_threshold <= 1.0, f"alphazero.eval_threshold must be in [0, 1], got {az.eval_threshold}")
        _require(az.lr > 0, f"alphazero.lr must be > 0, got {az.lr}")
        _require(az.weight_decay >= 0, f"alphazero.weight_decay must be >= 0, got {az.weight_decay}")

        self.curriculum.validate()

        # Purchase exploration resamples the ``unit_type`` sub-action of a
        # multi_discrete create_unit; a flat_discrete action has none, so the
        # hook would silently do nothing (review rltrain-10).
        if env.action_space_type == "flat_discrete":
            offenders = ["ppo"] if ppo.purchase_explore_eps > 0 else []
            for stage in self.curriculum.stages:
                eps = stage.purchase_explore_eps
                values = [eps.get("start", 0.0), eps.get("end", 0.0)] if isinstance(eps, Mapping) else [eps or 0.0]
                if any(float(v) > 0 for v in values):
                    offenders.append(f"stage '{stage.name}'")
            if offenders:
                raise ValueError(
                    f"purchase_explore_eps > 0 ({', '.join(offenders)}) needs action_space_type='multi_discrete': "
                    "it resamples the unit_type sub-action, which flat_discrete does not have"
                )

    def _normalize(self) -> None:
        _normalize_fields(self, "")
        for section in _SECTION_TYPES:
            obj = getattr(self, section)
            expected = _SECTION_TYPES[section]
            if not isinstance(obj, expected):
                raise TypeError(f"Section '{section}' must be a {expected.__name__}, got {type(obj).__name__}")
            if section != "curriculum":
                _normalize_fields(obj, f"{section}.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (suitable for JSON/YAML dumping).

        Tuples (``env.pad_to_size``) become lists, so the dict dumps as
        plain YAML and loads back to the same config.
        """
        return _plain(asdict(self))


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
    stages: list[CurriculumStage] = []
    for i, s in enumerate(raw_stages):
        if not isinstance(s, Mapping):
            raise TypeError(f"curriculum.stages[{i}] must be a mapping, got {type(s).__name__}")
        unknown_stage = set(s.keys()) - stage_fields
        if unknown_stage:
            raise ValueError(
                f"Unknown keys for CurriculumStage at index {i}: {sorted(unknown_stage)}. Valid keys: {sorted(stage_fields)}"
            )
        stages.append(CurriculumStage(**{k: v for k, v in s.items() if k in stage_fields}))
    kwargs: dict[str, Any] = {"stages": stages}
    if "restore_best_checkpoint_between_stages" in raw:
        # Coerced in validate() (``bool("false")`` used to be True here).
        kwargs["restore_best_checkpoint_between_stages"] = raw["restore_best_checkpoint_between_stages"]
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

    kwargs: dict[str, Any] = {}
    for key in top_level_scalars:
        if key in data:
            kwargs[key] = data[key]
    for section_name in _SECTION_TYPES:
        kwargs[section_name] = _build_section(section_name, data.get(section_name))

    cfg = TrainingConfig(**kwargs)
    cfg.validate()
    return cfg


def _read_config_file(path: Path) -> dict[str, Any]:
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
    """Load and validate a training config from a YAML or JSON file.

    Validation is :meth:`TrainingConfig.validate` without ``check_files``:
    a config can be loaded before the checkpoint it names exists.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    return config_from_dict(_read_config_file(p))


def _plain(value: Any) -> Any:
    """``value`` with tuples turned into lists, recursively (for YAML/JSON dumps)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


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
    if leaf not in {f.name for f in fields(target)}:
        raise KeyError(f"Unknown config key: '{dotted_key}'")
    # Coerced to the field's annotated type, as a file value would be:
    # ``--set env.pad_to_size=[10,12]`` becomes a tuple, ``"1e-5"`` a float.
    setattr(target, leaf, _coerce_to(value, _field_types(type(target))[leaf], dotted_key))


def config_to_argparse_defaults(
    cfg: TrainingConfig,
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    """Flatten a config to a dict suitable for ``parser.set_defaults(**d)``.

    Args:
        cfg: Loaded training config.
        mapping: Maps argparse ``dest`` names to dotted config paths, e.g.
            ``{"learning_rate": "ppo.learning_rate", "seed": "seed"}``.

    Missing paths are silently skipped so scripts can share a mapping but
    declare only a subset of fields.
    """
    defaults: dict[str, Any] = {}
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
    overrides: Mapping[str, Any] | None = None,
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


# ---------------------------------------------------------------------------
# Fields an entry point does not read (review rltrain-9)
#
# One TrainingConfig schema serves every trainer, but no trainer reads every
# field: the curriculum runner never looked at ``eval.checkpoint_freq``,
# ``logging.*`` or ``ppo.lr_schedule``, and a stage without
# ``n_eval_episodes`` silently ran 30 episodes whatever ``eval`` said. Each
# entry point now declares the fields it consumes; a loaded config that sets
# any other field away from its default is reported, as a warning by default
# and as an error under the entry points' ``--strict`` flag.
# ---------------------------------------------------------------------------

_TOP_LEVEL_FIELDS = ("algorithm", "total_timesteps", "seed", "warm_start_path")


class IgnoredConfigFieldWarning(UserWarning):
    """A loaded config sets a field that the running entry point does not read."""


class IgnoredConfigFieldError(ValueError):
    """Raised instead of :class:`IgnoredConfigFieldWarning` in strict mode."""


def config_field_values(cfg: TrainingConfig) -> dict[str, Any]:
    """Every leaf field of ``cfg`` by dotted path (``"env.max_steps"``, ``"curriculum.stages"``, ...)."""
    out: dict[str, Any] = {name: getattr(cfg, name) for name in _TOP_LEVEL_FIELDS}
    for section in _SECTION_TYPES:
        obj = getattr(cfg, section)
        for f in fields(obj):
            out[f"{section}.{f.name}"] = getattr(obj, f.name)
    return out


def _matches(path: str, patterns: Iterable[str]) -> bool:
    """``path`` equals a pattern, or a pattern is ``"<section>.*"`` for its section."""
    section = path.split(".", 1)[0]
    return any(p == path or p == f"{section}.*" for p in patterns)


def ignored_config_fields(
    cfg: TrainingConfig,
    consumed: Iterable[str],
    *,
    algorithms: Iterable[str] | None = None,
) -> list[str]:
    """Dotted paths of the fields ``cfg`` sets away from their default that the caller does not read.

    Args:
        cfg: The loaded config.
        consumed: Paths the entry point reads: exact (``"env.max_steps"``)
            or a whole section (``"curriculum.*"``).
        algorithms: The ``algorithm`` values the entry point accepts as a
            description of what it trains. ``algorithm`` is then reported
            only when set to something else (it is a label, which no trainer
            dispatches on).
    """
    consumed = tuple(consumed)
    defaults = config_field_values(TrainingConfig())
    ignored = []
    for path, value in config_field_values(cfg).items():
        if value == defaults[path] or _matches(path, consumed):
            continue
        if path == "algorithm" and algorithms is not None and value in tuple(algorithms):
            continue
        ignored.append(path)
    return ignored


def check_ignored_config_fields(
    cfg: TrainingConfig,
    consumed: Iterable[str],
    *,
    entry_point: str,
    strict: bool = False,
    algorithms: Iterable[str] | None = None,
    hints: Mapping[str, str] | None = None,
) -> list[str]:
    """Warn about (or, with ``strict``, reject) set fields the entry point ignores.

    See :func:`ignored_config_fields` for ``consumed`` and ``algorithms``.
    ``hints`` maps a path (or ``"<section>.*"``) to a short explanation shown
    next to it.

    Returns:
        The ignored paths (empty when there is nothing to report).

    Raises:
        IgnoredConfigFieldError: ``strict`` and at least one field is ignored.
    """
    ignored = ignored_config_fields(cfg, consumed, algorithms=algorithms)
    if not ignored:
        return ignored
    hints = hints or {}
    values = config_field_values(cfg)
    lines = []
    for path in ignored:
        hint = hints.get(path) or hints.get(path.split(".", 1)[0] + ".*")
        shown = values[path]
        if path == "curriculum.stages":
            shown = f"[{len(shown)} stages]"
        lines.append(f"  {path} = {shown!r}" + (f"  ({hint})" if hint else ""))
    message = (
        f"{entry_point} does not read {len(ignored)} field(s) this config sets away from their defaults, "
        "so they have no effect on this run:\n" + "\n".join(lines)
    )
    if strict:
        raise IgnoredConfigFieldError(message + "\nRemove them from the config, or run without --strict to only warn.")
    warnings.warn(message + "\nPass --strict to make this an error.", IgnoredConfigFieldWarning, stacklevel=2)
    return ignored
