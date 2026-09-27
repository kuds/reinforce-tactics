"""
Gymnasium environment for Reinforce Tactics
Supports both flat and hierarchical RL training
"""

import logging
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reinforcetactics.core.actions import ACTOR_KEYS
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot_registry import build_scripted as build_scripted_bot

# The opponent and reward_config contract lives in ``env_schema`` (which the
# config layer checks configs against); re-exported here.
from reinforcetactics.rl.env_schema import DEFAULT_REWARD_CONFIG as DEFAULT_REWARD_CONFIG
from reinforcetactics.rl.env_schema import KNOWN_REWARD_KEYS as KNOWN_REWARD_KEYS
from reinforcetactics.rl.env_schema import OPTIONAL_REWARD_KEYS as OPTIONAL_REWARD_KEYS
from reinforcetactics.rl.env_schema import SELF_PLAY_OPPONENT as SELF_PLAY_OPPONENT
from reinforcetactics.rl.env_schema import accepted_opponents as accepted_opponents
from reinforcetactics.rl.env_schema import resolve_opponent as resolve_opponent
from reinforcetactics.rl.env_schema import validate_opponent_kwargs as validate_opponent_kwargs
from reinforcetactics.rl.env_schema import validate_reward_config as validate_reward_config
from reinforcetactics.rl.observation import (
    GLOBAL_FEATURES_DIM,
    GOLD_SCALE,
    GRID_CHANNELS,
    TURN_SCALE,
    UNIT_CHANNELS,
    UNIT_COUNT_SCALE,
    build_observation,
)
from reinforcetactics.rules import ALL_UNIT_TYPES, UNIT_TYPE_TO_IDX
from reinforcetactics.utils.file_io import FileIO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuredActionMasks:
    """
    Decision-tree-aligned masks for autoregressive policies.

    Sampling order: atype -> source -> (unit_type if create_unit) -> target.
    Spatial arrays are indexed (y, x) = (row, col), matching the flat-mask
    convention ``flat_idx = atype * H * W + y * W + x``.

    Fields:
        atype:     (A,) bool             — legal action types
        source:    (A, H, W) bool        — legal source cells per atype
        target:    {(atype, sx, sy): (H, W) bool} — legal target cells given (atype, source)
        unit_type: {(sx, sy): (U,) bool} — legal unit types per building cell (create_unit)
    """

    atype: np.ndarray
    source: np.ndarray
    target: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    unit_type: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


# Mapping from action key → (action_type_idx, source_key, target_key).
# The single canonical layout, shared by StrategyGameEnv, the free mask
# builders below, and external consumers (imitation recorder, MCTS).
ACTION_KEY_MAP = {
    "create_unit": (0, None, ("x", "y")),
    "move": (1, ("from_x", "from_y"), ("to_x", "to_y")),
    "attack": (2, "attacker", "target"),
    "seize": (3, "unit", "tile"),
    "heal": (4, "healer", "target"),
    "cure": (4, "curer", "target"),
    "paralyze": (6, "paralyzer", "target"),
    "haste": (7, "sorcerer", "target"),
    "defence_buff": (8, "sorcerer", "target"),
    "attack_buff": (9, "sorcerer", "target"),
}


# Encoded action types that name one targeted engine action (type 4 picks
# heal or cure), with the check the env makes on (unit, target, player)
# before the engine sees the action. The engine validates every action
# itself; these checks are older and stay so that what counts as an invalid
# action (and pays the invalid-action penalty) does not change.
_TARGETED_ACTION_TYPES: dict[int, tuple[str, Callable[[Any, Any, int], bool]]] = {
    2: ("attack", lambda unit, target, player: unit.player == player and target.player != player),
    6: ("paralyze", lambda unit, target, player: unit.type == "M" and target.player != player),
    7: ("haste", lambda unit, target, player: unit.type == "S" and target.player == player),
    8: ("defence_buff", lambda unit, target, player: unit.type == "S" and target.player == player),
    9: ("attack_buff", lambda unit, target, player: unit.type == "S" and target.player == player),
}


def _action_pos(obj_or_dict, fields):
    """Extract (x, y) from an action object (.x/.y) or a dict (fields tuple)."""
    if isinstance(fields, str):
        o = obj_or_dict[fields]
        return o.x, o.y
    return obj_or_dict[fields[0]], obj_or_dict[fields[1]]


def build_per_dim_masks(
    game_state: "GameState",
    grid_width: int,
    grid_height: int,
    enabled_units: list[str] | None = None,
    flat_action_size: int | None = None,
    player: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The single implementation behind ``StrategyGameEnv._build_masks``.

    Also used directly by ``ModelBot``, the imitation recorder, and MCTS so
    a checkpoint can be played against an arbitrary live ``game_state``
    without constructing a full ``StrategyGameEnv`` around it.

    Args:
        player: Whose legal actions to mask. Defaults to
            ``game_state.current_player``; the env passes its
            ``agent_player`` explicitly (see ``_build_masks``).

    Returns ``(flat_mask, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask)``.
    """
    if player is None:
        player = game_state.current_player
    legal_actions = game_state.get_legal_actions(player=player)
    area = grid_width * grid_height
    flat_size = flat_action_size if flat_action_size is not None else 10 * area

    flat_mask = np.zeros(flat_size, dtype=np.float32)
    at_mask = np.zeros(10, dtype=bool)
    ut_mask = np.zeros(8, dtype=bool)
    fx_mask = np.zeros(grid_width, dtype=bool)
    fy_mask = np.zeros(grid_height, dtype=bool)
    tx_mask = np.zeros(grid_width, dtype=bool)
    ty_mask = np.zeros(grid_height, dtype=bool)

    for key, (at_idx, src_fields, tgt_fields) in ACTION_KEY_MAP.items():
        for action in legal_actions.get(key, []):
            at_mask[at_idx] = True
            tx, ty = _action_pos(action, tgt_fields)
            tx_mask[tx] = True
            ty_mask[ty] = True
            flat_idx = at_idx * area + ty * grid_width + tx
            if 0 <= flat_idx < flat_mask.size:
                flat_mask[flat_idx] = 1.0
            if src_fields is not None:
                sx, sy = _action_pos(action, src_fields)
                fx_mask[sx] = True
                fy_mask[sy] = True
            else:
                fx_mask[tx] = True
                fy_mask[ty] = True
            if key == "create_unit":
                ut_mask[UNIT_TYPE_TO_IDX.get(action["unit_type"], 0)] = True

    at_mask[5] = True
    flat_mask[5 * area] = 1.0
    fx_mask[0] = True
    fy_mask[0] = True
    tx_mask[0] = True
    ty_mask[0] = True

    if not ut_mask.any():
        if enabled_units:
            ut_mask[UNIT_TYPE_TO_IDX.get(enabled_units[0], 0)] = True
        else:
            ut_mask[0] = True

    return flat_mask, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask


def build_structured_masks(
    game_state: "GameState",
    grid_width: int,
    grid_height: int,
    player: int | None = None,
) -> "StructuredActionMasks":
    """The single implementation behind ``StrategyGameEnv._build_structured_masks``.

    Same purpose as :func:`build_per_dim_masks` — a free function so external
    callers (e.g. ``ModelBot`` for AR worker inference) can build masks
    against an arbitrary live ``game_state`` without owning an env.

    Args:
        player: Whose legal actions to mask. Defaults to
            ``game_state.current_player``; the env passes its
            ``agent_player`` explicitly.
    """
    if player is None:
        player = game_state.current_player
    legal_actions = game_state.get_legal_actions(player=player)
    H, W = grid_height, grid_width
    num_action_types = 10
    num_unit_types = 8

    atype = np.zeros(num_action_types, dtype=bool)
    source = np.zeros((num_action_types, H, W), dtype=bool)
    target: dict[tuple[int, int, int], np.ndarray] = {}
    unit_type: dict[tuple[int, int], np.ndarray] = {}

    def _mark_target(at_idx: int, sx: int, sy: int, tx: int, ty: int) -> None:
        key = (at_idx, sx, sy)
        m = target.get(key)
        if m is None:
            m = np.zeros((H, W), dtype=bool)
            target[key] = m
        m[ty, tx] = True

    for key, (at_idx, src_fields, tgt_fields) in ACTION_KEY_MAP.items():
        for action in legal_actions.get(key, []):
            tx, ty = _action_pos(action, tgt_fields)
            if src_fields is not None:
                sx, sy = _action_pos(action, src_fields)
            else:
                sx, sy = tx, ty
            atype[at_idx] = True
            source[at_idx, sy, sx] = True
            _mark_target(at_idx, sx, sy, tx, ty)
            if key == "create_unit":
                ut_idx = UNIT_TYPE_TO_IDX.get(action["unit_type"], 0)
                ukey = (sx, sy)
                m = unit_type.get(ukey)
                if m is None:
                    m = np.zeros(num_unit_types, dtype=bool)
                    unit_type[ukey] = m
                m[ut_idx] = True

    atype[5] = True
    source[5, 0, 0] = True
    end_t = np.zeros((H, W), dtype=bool)
    end_t[0, 0] = True
    target[(5, 0, 0)] = end_t

    return StructuredActionMasks(atype=atype, source=source, target=target, unit_type=unit_type)


# ---------------------------------------------------------------------------
# flat_discrete decode tables
# ---------------------------------------------------------------------------

# The flat_discrete decode table is part of a checkpoint's contract: a
# Discrete index means "the i-th entry of the table", so a policy trained on
# one table layout misreads another. The layout is therefore versioned, and
# a checkpoint is decoded with the version it was trained on.
#
#   1 (legacy): when the legal set exceeds ``max_flat_actions``, keep seize
#     and end_turn and fill the rest of the budget with the other actions in
#     ACTION_KEY_MAP order -- so casts, heals and then attacks (which all come
#     after moves) are dropped first (review rlenv-11 / prior-13). Every
#     checkpoint trained before versioning existed uses this layout.
#   2: the same table whenever nothing is truncated. Truncation drops moves
#     first, round-robin across units so each keeps a spread of its
#     destinations; then purchases; then attacks, heals and casts; then
#     seizes. end_turn is always kept, and kept entries stay in their
#     canonical order.
#
# ``StrategyGameEnv`` defaults to FLAT_ACTION_VERSION_LATEST and stamps the
# version on its Discrete action space (``flat_action_version``), which SB3
# saves with the model, so ``flat_action_version_of(model)`` recovers it; an
# unstamped (older) checkpoint reads as FLAT_ACTION_VERSION_LEGACY. The free
# functions default to the legacy layout so every existing caller keeps its
# exact behaviour until it passes a version.
FLAT_ACTION_VERSION_LEGACY = 1
FLAT_ACTION_VERSION_LATEST = 2
FLAT_ACTION_VERSIONS: tuple[int, ...] = (FLAT_ACTION_VERSION_LEGACY, FLAT_ACTION_VERSION_LATEST)

# Version 2's truncation order: the action-type groups dropped first come
# first. end_turn (5) is never dropped.
_V2_DROP_ORDER: tuple[frozenset[int], ...] = (
    frozenset({1}),  # move
    frozenset({0}),  # create_unit
    frozenset({2, 4, 6, 7, 8, 9}),  # attack, heal/cure, paralyze, haste, the buffs
    frozenset({3}),  # seize
)

_END_TURN_ACTION = (5, 0, 0, 0, 0, 0)


def _check_flat_action_version(version: int) -> int:
    if isinstance(version, bool) or not isinstance(version, (int, np.integer)) or int(version) not in FLAT_ACTION_VERSIONS:
        raise ValueError(f"flat_action_version must be one of {FLAT_ACTION_VERSIONS}, got {version!r}")
    return int(version)


def flat_action_version_of(obj: Any) -> int:
    """The flat_discrete table version recorded on a model, env or action space.

    Reads the ``flat_action_version`` attribute ``StrategyGameEnv`` stamps on
    its Discrete action space (for a model or env, on ``obj.action_space``).
    Checkpoints saved before versioning existed carry no stamp and read as
    :data:`FLAT_ACTION_VERSION_LEGACY`, which is how they were trained.
    """
    space = obj if isinstance(obj, spaces.Space) else getattr(obj, "action_space", None)
    version = getattr(space, "flat_action_version", FLAT_ACTION_VERSION_LEGACY)
    return _check_flat_action_version(version)


def stamp_flat_action_version(obj: Any, version: int) -> None:
    """Record ``version`` on a model's, env's or Discrete space's action space.

    For code that loads a checkpoint and keeps training it on envs of a
    different version (a warm start from an unstamped checkpoint, say):
    ``flat_action_version_of`` then reports what the policy now plays.
    """
    version = _check_flat_action_version(version)
    space = obj if isinstance(obj, spaces.Space) else getattr(obj, "action_space", None)
    if not isinstance(space, spaces.Discrete):
        raise TypeError(f"flat_action_version applies to a Discrete action space, got {space!r}")
    # A plain instance attribute: gymnasium spaces pickle their __dict__, so
    # it survives SubprocVecEnv transport and SB3 save/load.
    setattr(space, "flat_action_version", version)


def checkpoint_flat_action_version(path: str | Any) -> int:
    """The flat_discrete table version an SB3 ``.zip`` checkpoint was trained with.

    Reads only the checkpoint's metadata (the pickled action space in its
    ``data`` entry), not its weights, so the version can decide how the
    envs are built *before* the model is. Unstamped checkpoints read as
    :data:`FLAT_ACTION_VERSION_LEGACY` (see :func:`flat_action_version_of`).

    ``path`` may leave out the ``.zip`` suffix, as ``MaskablePPO.load``
    allows (SB3's ``open_path`` appends it when the bare path is missing).

    Raises:
        FileNotFoundError: Neither ``path`` nor ``path + '.zip'`` exists.
        ValueError: ``path`` is not an SB3 checkpoint with an action space.
    """
    import json
    import os
    import zipfile

    from stable_baselines3.common.save_util import json_to_data

    path = os.fspath(path)
    if not os.path.isfile(path) and os.path.isfile(f"{path}.zip"):
        path = f"{path}.zip"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint '{path}' does not exist")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = json.loads(archive.read("data").decode())
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"'{path}' is not a Stable-Baselines3 checkpoint: {exc}") from exc
    # Deserialize the action space alone: the other entries (schedules,
    # policy classes) need not even be importable here.
    space = json_to_data(json.dumps({"action_space": entries.get("action_space")})).get("action_space")
    if not isinstance(space, spaces.Space):
        raise ValueError(f"checkpoint '{path}' records no action space")
    return flat_action_version_of(space)


def resolve_flat_action_version(
    configured: int | None,
    *,
    action_space_type: str,
    checkpoint_path: str | None = None,
) -> int:
    """The flat_discrete table version a run's envs should use.

    ``configured`` (a config's ``env.flat_action_version``) wins when set.
    ``None`` means: the version of the checkpoint the run continues
    (``checkpoint_path``: a warm start or resume of a flat_discrete policy),
    so the policy keeps decoding with the table it was trained on; else
    :data:`FLAT_ACTION_VERSION_LATEST`. multi_discrete runs never read the
    checkpoint (the version only shapes flat_discrete tables).
    """
    if configured is not None:
        return _check_flat_action_version(configured)
    if action_space_type == "flat_discrete" and checkpoint_path:
        return checkpoint_flat_action_version(checkpoint_path)
    return FLAT_ACTION_VERSION_LATEST


class FlatActionVersionMismatch(ValueError):
    """A flat_discrete policy is paired with an env that decodes its indices with another table."""


def env_flat_action_version(env: Any) -> int | None:
    """The flat_discrete table version ``env`` decodes action indices with.

    ``env`` is a :class:`StrategyGameEnv`, any gymnasium wrapper over one
    (``ActionMaskedEnv``, ``SelfPlayEnv``, ...), or an SB3 VecEnv of them.
    Read from the envs' ``flat_action_version`` attribute, which is what
    their ``step()`` decodes with.

    Returns:
        The version, or ``None`` when ``env`` is not a flat_discrete
        StrategyGameEnv (a multi_discrete env, or anything else).

    Raises:
        FlatActionVersionMismatch: A VecEnv whose workers disagree.
    """
    if hasattr(env, "num_envs") and hasattr(env, "get_attr"):  # an SB3 VecEnv
        # has_attr first: a get_attr of a missing attribute kills a
        # SubprocVecEnv worker.
        if not (env.has_attr("action_space_type") and env.has_attr("flat_action_version")):
            return None
        kinds = env.get_attr("action_space_type")
        versions = {int(v) for k, v in zip(kinds, env.get_attr("flat_action_version"), strict=True) if k == "flat_discrete"}
        if len(versions) > 1:
            raise FlatActionVersionMismatch(
                f"The vec env's workers decode with different flat_action_versions {sorted(versions)}"
            )
        return versions.pop() if versions else None
    base = getattr(env, "unwrapped", env)
    if getattr(base, "action_space_type", None) != "flat_discrete":
        return None
    version = getattr(base, "flat_action_version", None)
    return None if version is None else _check_flat_action_version(version)


def check_flat_action_version(model: Any, env: Any, *, what: str = "this env") -> None:
    """Refuse to pair a flat_discrete ``model`` with an env that decodes another table.

    A flat_discrete policy outputs an index into the env's per-state action
    table, and the table's layout depends on ``flat_action_version`` once
    the legal set exceeds ``max_flat_actions``. SB3's space check compares
    only ``Discrete.n``, so an old (version-1, unstamped) checkpoint loaded
    into, or evaluated on, a default (version-2) env ran without a word,
    playing different actions than it was trained to at every truncated
    decision point -- and a continued checkpoint kept the old stamp, so
    ModelBot then decoded it with the wrong table too.

    Nothing to check (returns) when the env is not a flat_discrete
    StrategyGameEnv or the model has no Discrete action space.

    Raises:
        FlatActionVersionMismatch: The model's version
            (:func:`flat_action_version_of`) differs from the env's
            (:func:`env_flat_action_version`). Build the env with
            ``flat_action_version=flat_action_version_of(model)`` (or
            :func:`checkpoint_flat_action_version` of its file), or, to
            deliberately continue training on the env's table, call
            ``stamp_flat_action_version(model, <env version>)`` first.
    """
    env_version = env_flat_action_version(env)
    if env_version is None:
        return
    space = getattr(model, "action_space", None)
    if not isinstance(space, spaces.Discrete):
        return
    model_version = flat_action_version_of(space)
    if model_version != env_version:
        raise FlatActionVersionMismatch(
            f"The model decodes flat_discrete actions with flat_action_version {model_version}, but {what} "
            f"uses version {env_version}: every decision point with more legal actions than max_flat_actions "
            f"would play a different action than the policy chose. Build the env with "
            f"flat_action_version={model_version} (flat_action_version_of(model), or "
            "checkpoint_flat_action_version(path) for a saved checkpoint), or, to continue training the "
            f"policy on version {env_version}, call stamp_flat_action_version(model, {env_version}) first."
        )


class _RateLimitedWarning:
    """Log a warning at most once per ``interval_s``, counting the ones held back.

    The flat-action truncation warning fires on every decision point of a
    large army's turn -- hundreds per episode per worker -- which used to
    flood the training logs (review rlenv-11). Per process: each
    SubprocVecEnv worker keeps its own clock.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = float(interval_s)
        self._last: float | None = None
        self.suppressed = 0

    def __call__(self, msg: str, *args: Any) -> bool:
        now = time.monotonic()
        if self._last is not None and now - self._last < self.interval_s:
            self.suppressed += 1
            return False
        if self.suppressed:
            msg += " (%d similar warnings suppressed in the last %.0fs)"
            args = (*args, self.suppressed, self.interval_s)
        logger.warning(msg, *args)
        self._last = now
        self.suppressed = 0
        return True


_truncation_warning = _RateLimitedWarning(interval_s=300.0)


@dataclass(frozen=True)
class FlatActionTable:
    """A flat_discrete decode table with its truncation diagnostics.

    Fields:
        actions: The decode table (what :func:`build_flat_actions` returns).
        n_legal: Deduplicated legal actions (end_turn included) before any
            truncation; ``len(actions)`` when nothing was dropped.
        truncated: Whether entries were dropped to fit ``max_flat_actions``.
    """

    actions: list[np.ndarray]
    n_legal: int
    truncated: bool


def _spread_indices(n: int, k: int) -> list[int]:
    """``k`` indices into ``range(n)`` spread evenly, both ends included (``k == 1``: the last)."""
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n - 1]
    return [(i * (n - 1)) // (k - 1) for i in range(k)]


def _round_robin_keep(actions: list[np.ndarray], indices: list[int], budget: int) -> set[int]:
    """Keep ``budget`` of ``indices``, shared round-robin across their source cells.

    Each source (a unit, or a building for purchases) gets one entry per
    round until the budget runs out, so every unit keeps some options
    instead of the units listed last losing all of theirs. Within a source
    the kept entries are spread across its list (which runs from the
    nearest destinations the move search reaches to the farthest).
    """
    if budget <= 0:
        return set()
    if budget >= len(indices):
        return set(indices)
    groups: dict[tuple[int, int], list[int]] = {}
    for i in indices:
        a = actions[i]
        groups.setdefault((int(a[2]), int(a[3])), []).append(i)
    members = list(groups.values())
    quota = [0] * len(members)
    remaining = budget
    while remaining > 0:
        for g, m in enumerate(members):
            if remaining == 0:
                break
            if quota[g] < len(m):
                quota[g] += 1
                remaining -= 1
    kept: set[int] = set()
    for m, q in zip(members, quota):
        kept.update(m[j] for j in _spread_indices(len(m), q))
    return kept


def _truncate_legacy(actions: list[np.ndarray], max_flat_actions: int) -> list[np.ndarray]:
    """Version 1 truncation, unchanged: every pre-versioning checkpoint was trained on it."""
    # Naive head-truncation would silently drop the tail of the list
    # -- which is exactly end_turn (appended last) and seize
    # (action_type 3, built after create/move/attack). Losing
    # end_turn can strand the agent for the rest of the game-turn
    # when ``max_actions_per_turn`` is disabled; losing seize drops
    # the rare, high-value capture action. So always keep those two
    # action types and fill the remaining budget with the rest in
    # their original order.
    protected = [a for a in actions if int(a[0]) in (3, 5)]
    others = [a for a in actions if int(a[0]) not in (3, 5)]
    budget = max(0, max_flat_actions - len(protected))
    actions = others[:budget] + protected
    # Pathological fallback: if the protected set alone exceeds the
    # cap, hard-truncate but keep end_turn (last protected entry) by
    # trimming from the front of the seize block.
    if len(actions) > max_flat_actions:
        actions = actions[-max_flat_actions:]
    return actions


def _truncate_v2(actions: list[np.ndarray], max_flat_actions: int) -> list[np.ndarray]:
    """Version 2 truncation: drop moves first (round-robin per unit), keep combat and support."""
    keep = [True] * len(actions)
    total = len(actions)
    for group in _V2_DROP_ORDER:
        if total <= max_flat_actions:
            break
        indices = [i for i, a in enumerate(actions) if int(a[0]) in group]
        budget = max(0, len(indices) - (total - max_flat_actions))
        kept = _round_robin_keep(actions, indices, budget)
        for i in indices:
            if i not in kept:
                keep[i] = False
        total -= len(indices) - len(kept)
    return [a for a, k in zip(actions, keep) if k]


def flat_action_table(
    game_state: "GameState",
    player: int,
    max_flat_actions: int,
    *,
    version: int = FLAT_ACTION_VERSION_LEGACY,
) -> FlatActionTable:
    """Build a flat_discrete decode table and report whether it was truncated.

    See :func:`build_flat_actions` for the table itself; this also returns
    the pre-truncation count and a truncated flag, which the env surfaces
    in ``info`` / ``episode_stats``.
    """
    version = _check_flat_action_version(version)
    if max_flat_actions < 1:
        raise ValueError(f"max_flat_actions must be >= 1, got {max_flat_actions}")
    legal_actions = game_state.get_legal_actions(player=player)

    actions: list[np.ndarray] = []
    seen = set()

    for key, (at_idx, src_fields, tgt_fields) in ACTION_KEY_MAP.items():
        for action in legal_actions.get(key, []):
            tx, ty = _action_pos(action, tgt_fields)

            if src_fields is not None:
                fx, fy = _action_pos(action, src_fields)
            else:
                fx, fy = tx, ty  # create_unit: from = building position

            ut_idx = 0
            if key == "create_unit":
                ut_idx = UNIT_TYPE_TO_IDX.get(action["unit_type"], 0)

            action_key = (at_idx, ut_idx, fx, fy, tx, ty)
            if action_key not in seen:
                seen.add(action_key)
                actions.append(np.array(action_key, dtype=np.int32))

    # End turn is always valid (last entry)
    end_turn_key = _END_TURN_ACTION
    if end_turn_key not in seen:
        seen.add(end_turn_key)
        actions.append(np.array(end_turn_key, dtype=np.int32))

    n_legal = len(actions)
    if n_legal <= max_flat_actions:
        return FlatActionTable(actions=actions, n_legal=n_legal, truncated=False)

    if version == FLAT_ACTION_VERSION_LEGACY:
        _truncation_warning(
            "Legal actions (%d) exceed max_flat_actions (%d); truncating (flat_action_version 1: "
            "seize and end_turn kept, other actions dropped from the end of the list). "
            "Consider increasing max_flat_actions.",
            n_legal,
            max_flat_actions,
        )
        actions = _truncate_legacy(actions, max_flat_actions)
    else:
        _truncation_warning(
            "Legal actions (%d) exceed max_flat_actions (%d); truncating (flat_action_version %d: "
            "moves dropped first). Consider increasing max_flat_actions.",
            n_legal,
            max_flat_actions,
            version,
        )
        actions = _truncate_v2(actions, max_flat_actions)
    return FlatActionTable(actions=actions, n_legal=n_legal, truncated=True)


def build_flat_actions(
    game_state: "GameState",
    player: int,
    max_flat_actions: int,
    *,
    version: int = FLAT_ACTION_VERSION_LEGACY,
) -> list[np.ndarray]:
    """Build the ordered flat legal-action list for a ``Discrete`` policy.

    Pure-function port of ``StrategyGameEnv._build_flat_actions`` (minus the
    per-game-turn action-budget gate, which is env state). The returned list
    is the decode table for ``flat_discrete`` checkpoints: a sampled
    ``Discrete`` index ``i`` means "execute ``actions[i]``", where each entry
    is a 6-element int array ``[action_type, unit_type, from_x, from_y,
    to_x, to_y]`` — the same layout as a MultiDiscrete action.

    Shared by ``StrategyGameEnv`` and ``ModelBot`` so a flat_discrete
    checkpoint can be played against an arbitrary live ``game_state``
    (GUI / tournament) without constructing an env around it. The exact
    per-index legality mask is "first ``len(actions)`` entries True".

    Entries follow ``ACTION_KEY_MAP`` order, each kind in
    ``get_legal_actions`` order, deduplicated; end turn is always present
    as the last action. When the legal set exceeds ``max_flat_actions`` the
    list is truncated as ``version`` prescribes (see
    :data:`FLAT_ACTION_VERSION_LATEST`); the two versions agree whenever
    nothing is truncated.

    Args:
        version: The table layout the checkpoint was trained on. Defaults
            to the legacy layout (version 1), which is what every caller got
            before versioning; pass ``flat_action_version_of(model)`` when
            decoding a checkpoint, or the env's ``flat_action_version``.
    """
    return flat_action_table(game_state, player, max_flat_actions, version=version).actions


class StrategyGameEnv(gym.Env):
    """
    Gymnasium environment for turn-based strategy game.

    Observation Space (agent-relative; see ``rl.observation`` for the full
    contract):
        Dict with:
        - 'grid':  (H, W, GRID_CHANNELS) - one-hot terrain + agent-relative
                   owner one-hot (self/opp/neutral) + structure HP fraction.
        - 'units': (H, W, UNIT_CHANNELS) - one-hot unit type + agent-relative
                   owner one-hot + own_acted + HP fraction + 4 status channels
                   (paralyze, haste, defence_buff, attack_buff).
        - 'global_features': (5,) - own_gold, opp_gold, turn, own_units, opp_units.
        Action masks are pulled separately via ``env.action_masks()`` for
        MaskablePPO and are not part of the policy observation.

    Action Space:
        MultiDiscrete with 6 dimensions:
        - action_type: [0=create_unit, 1=move, 2=attack, 3=seize, 4=heal, 5=end_turn,
                        6=paralyze, 7=haste, 8=defence_buff, 9=attack_buff]
        - unit_type: [0=W, 1=M, 2=C, 3=A, 4=K, 5=R, 6=S, 7=B] (for create_unit)
        - from_x: [0, grid_width)
        - from_y: [0, grid_height)
        - to_x: [0, grid_width)
        - to_y: [0, grid_height)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # ``info["action_type"]`` for a flat_discrete index that names no entry of
    # the decode table: nothing was executed (see ``step``).
    INVALID_FLAT_INDEX_ACTION_TYPE = -1

    ALL_UNIT_TYPES = ALL_UNIT_TYPES

    def __init__(
        self,
        map_file: str | None = None,
        opponent: str | None = "bot",  # a bot_registry name ('bot', 'master', 'noop', ...), 'self', or None
        render_mode: str | None = None,
        max_steps: int = 200,
        max_turns: int | None = None,
        reward_config: dict[str, float] | None = None,
        hierarchical: bool = False,  # Enable for HRL
        goal_space_size: int = 64,  # For HRL goal space
        enabled_units: list[str] | None = None,  # List of enabled unit types
        fog_of_war: bool = False,  # Enable fog of war
        action_space_type: str = "multi_discrete",  # 'multi_discrete' or 'flat_discrete'
        max_flat_actions: int = 512,  # Max actions for flat_discrete mode
        max_actions_per_turn: int | None = None,  # Hard cap on agent actions per game-turn (None = unlimited)
        opponent_kwargs: dict[str, Any] | None = None,  # Extra kwargs forwarded to the opponent constructor
        gamma: float = 0.99,  # Discount used by potential-based shaping; should match the trainer's gamma
        pad_to_size: tuple[int, int] | None = None,  # (pad_h, pad_w) for cross-stage obs-shape unification
        gold_scale: float = GOLD_SCALE,  # tanh divisor for own_gold/opp_gold in global_features
        turn_scale: float = TURN_SCALE,  # tanh divisor for turn_number in global_features
        unit_count_scale: float = UNIT_COUNT_SCALE,  # tanh divisor for own_units/opp_units
        engine_overrides: dict[str, Any] | None = None,  # sparse overlay over rules.py (balance sweeps)
        flat_action_version: int = FLAT_ACTION_VERSION_LATEST,  # flat_discrete decode-table layout
    ):
        """
        Initialize environment.

        Args:
            map_file: Path to map CSV. If None, generates random map
            opponent: ``None`` (manual: the caller plays the other seat),
                ``'self'`` (self-play; see ``set_self_play_opponent_factory``)
                or a scripted bot from the bot registry
                (:func:`accepted_opponents`: 'bot' (= 'simple'), 'medium',
                'advanced', 'master', 'mixed', 'random', 'balanced_random',
                'noop'). 'noop' is a stationary opponent that only ends its
                turn — useful as a curriculum stage-0 / sanity check.
                Anything else raises ValueError.
            render_mode: 'human' or 'rgb_array' or None
            max_steps: Maximum steps per episode
            max_turns: Maximum game turns before auto-draw (None = unlimited).
                Setting this lets games end via game rules (terminated=True)
                rather than only via env step truncation, which avoids the
                value-bootstrapping mismatch in PPO at episode end.
            reward_config: Reward weights overlaid on
                :data:`DEFAULT_REWARD_CONFIG`. Keys outside
                :data:`KNOWN_REWARD_KEYS` and non-numeric values raise (see
                :func:`validate_reward_config`).
            hierarchical: Whether to use hierarchical action space
            goal_space_size: Size of goal space for HRL
            enabled_units: List of enabled unit types (default all)
            fog_of_war: Enable fog of war for partial observability (default False)
            action_space_type: 'multi_discrete' (per-dimension masks) or
                'flat_discrete' (exact per-action masks, eliminates invalid actions)
            max_flat_actions: Upper bound on legal actions per step for flat_discrete
                (the Discrete action-space size, >= 1).
            flat_action_version: The flat_discrete decode-table layout
                (:data:`FLAT_ACTION_VERSIONS`); decides which actions are
                dropped when the legal set exceeds ``max_flat_actions``.
                Default :data:`FLAT_ACTION_VERSION_LATEST`; pass
                :data:`FLAT_ACTION_VERSION_LEGACY` (or
                ``flat_action_version_of(model)``) to keep training or
                evaluating a checkpoint on the table it was trained on.
                Stamped on the Discrete action space, so SB3 saves it with
                the model.
            opponent_kwargs: Extra constructor kwargs for a scripted
                opponent; validated against that bot's constructor (see
                :func:`validate_opponent_kwargs`).
            max_actions_per_turn: Optional hard cap on the number of agent
                actions taken within a single game-turn before the action
                mask is narrowed to end_turn only. Defends against the
                "never end the turn" failure mode where the policy spins
                through legal-but-unproductive actions (idle moves, etc.)
                until ``max_steps`` truncates the episode. Resets on every
                end_turn the agent executes. ``None`` (default) disables
                the cap.
            gold_scale: ``tanh`` divisor applied to ``own_gold`` /
                ``opp_gold`` before they enter ``global_features``. Default
                :data:`~reinforcetactics.rl.observation.GOLD_SCALE` (1000)
                is tuned for the current curriculum's gold range; override
                for maps with very different income economies.
            turn_scale: ``tanh`` divisor applied to ``turn_number``.
                Default :data:`~reinforcetactics.rl.observation.TURN_SCALE`
                (60) sits in the linear regime for the early-curriculum
                ``max_turns`` range in ``configs/ppo/bootstrap.yaml`` (20-75)
                and saturates gracefully on the longer late stages
                (120-200).
            unit_count_scale: ``tanh`` divisor applied to per-side unit
                counts. Default
                :data:`~reinforcetactics.rl.observation.UNIT_COUNT_SCALE`
                (20) targets typical per-side army sizes.
            pad_to_size: Optional ``(pad_h, pad_w)`` to zero-pad the spatial
                observation tensors to a fixed shape independent of the live
                map size. Used by the curriculum runner so a single PPO
                policy can train across stages with different map sizes
                without an observation-space mismatch. Only supported with
                ``action_space_type='flat_discrete'`` because flat-discrete's
                action space is sized to ``max_flat_actions`` and is
                therefore already grid-independent; multi_discrete's action
                space depends on grid dims and would need separate padding.
        """
        super().__init__()

        # Validate the cheap arguments before any map I/O, so a typo fails
        # at construction rather than as a silently different MDP.
        validate_opponent_kwargs(opponent, opponent_kwargs)
        validate_reward_config(reward_config)
        self.flat_action_version = _check_flat_action_version(flat_action_version)
        if isinstance(max_flat_actions, bool) or not isinstance(max_flat_actions, (int, np.integer)) or max_flat_actions < 1:
            raise ValueError(f"max_flat_actions must be an integer >= 1, got {max_flat_actions!r}")

        # Load or generate map
        if map_file:
            map_data = FileIO.load_map(map_file)
        else:
            map_data = FileIO.generate_random_map(20, 20, num_players=2)

        self.initial_map_data = map_data
        # Store enabled units (default to all if not specified). An unknown
        # code used to pass here and raise a bare KeyError from the first
        # action mask (a broken pipe inside a SubprocVecEnv worker).
        if enabled_units is not None:
            unknown_units = [u for u in enabled_units if u not in self.ALL_UNIT_TYPES]
            if unknown_units:
                raise ValueError(f"Unknown enabled_units {unknown_units}; unit codes are {', '.join(self.ALL_UNIT_TYPES)}")
        self.enabled_units = enabled_units if enabled_units is not None else self.ALL_UNIT_TYPES.copy()
        # Fog of war setting
        self.fog_of_war = fog_of_war
        self.max_turns = max_turns
        # Sparse engine-constant overlay (balance sweeps). Resolved by
        # GameState; forwarded again on every reset() so re-created games
        # keep the same overlay.
        self.engine_overrides = engine_overrides
        # 1v1 only. The PPO/MaskablePPO observation contract (self/opp owner
        # channels, ``opp = 3 - perspective_player``, single opp_gold scalar)
        # is hard-coded to two players. Pygame 1v1v1-vs-bots still works
        # because it doesn't go through this env — it only uses the core
        # GameState and non-RL bot opponents.
        self.game_state = GameState(
            map_data,
            num_players=2,
            max_turns=max_turns,
            enabled_units=self.enabled_units,
            fog_of_war=fog_of_war,
            engine_overrides=self.engine_overrides,
        )
        if self.game_state.num_players != 2:
            raise ValueError(
                "StrategyGameEnv only supports 1v1 games (num_players=2). "
                f"Got num_players={self.game_state.num_players}. Multi-player "
                "(FFA / team) training requires a separate team-relative "
                "observation encoding."
            )

        # As given ("bot" stays "bot"); reset() resolves it through the bot
        # registry, so it may be reassigned before a reset (SelfPlayEnv sets
        # "self") and is re-validated there.
        self.opponent_type = opponent
        self.opponent_kwargs: dict[str, Any] = dict(opponent_kwargs) if opponent_kwargs else {}
        self.opponent: Any | None = None
        # Self-play hook: when ``opponent_type == "self"``, reset() rebinds the
        # opponent by calling ``factory(game_state, opponent_player) -> Bot``.
        # The training script provides it via ``set_self_play_opponent_factory``.
        # ``None`` keeps the slot empty (no-op opponent turn).
        self._self_play_opponent_factory: Any | None = None
        self.max_steps = max_steps
        self.current_step = 0
        # Hard per-game-turn action budget. When the agent has executed
        # ``max_actions_per_turn`` actions without ending the turn, the
        # mask is narrowed to end_turn only on subsequent steps. The
        # counter is bumped in ``step`` for every agent action and reset
        # whenever the agent actually ends its turn.
        if max_actions_per_turn is not None and max_actions_per_turn <= 0:
            raise ValueError("max_actions_per_turn must be > 0 (or None to disable)")
        self.max_actions_per_turn = max_actions_per_turn
        self._actions_this_turn = 0
        self.hierarchical = hierarchical
        self.goal_space_size = goal_space_size

        # Which player the RL agent controls (1 or 2). Change it with
        # ``set_agent_seat`` (or by assigning ``env.unwrapped.agent_player``)
        # *before* reset(): reset() binds the opponent to the other seat and,
        # when the agent is player 2, plays player 1's opening turn so that
        # the first observation, mask and shaping potential belong to the
        # agent's own turn. SelfPlayEnv uses the "random" seat mode.
        self.agent_player = 1
        # When True, reset() draws ``agent_player`` from np_random every
        # episode (see ``set_agent_seat``).
        self._random_agent_seat = False

        # Reward weights: the defaults overlaid with ``reward_config``
        # (validated above, so every key is one the env reads). Indexed
        # directly everywhere -- DEFAULT_REWARD_CONFIG is the only default.
        self.reward_config: dict[str, float] = {**DEFAULT_REWARD_CONFIG, **(reward_config or {})}

        # ``global_features`` normalization scales — stored so ``_get_obs``
        # can forward them to ``build_observation`` without re-reading
        # config on each call.
        self.gold_scale = float(gold_scale)
        self.turn_scale = float(turn_scale)
        self.unit_count_scale = float(unit_count_scale)

        # Previous potential for potential-based reward shaping (Phi(s) tracking).
        # ``gamma`` is the trainer's discount; the shaping delta uses
        # gamma * Phi(s') - Phi(s) per Ng et al. (1999) so the shaping is
        # policy-invariant for the chosen gamma. Mismatch with the trainer's
        # gamma reintroduces bias.
        self._prev_potential = 0.0
        self.gamma = float(gamma)

        # Action space configuration
        self.action_space_type = action_space_type
        self.max_flat_actions = int(max_flat_actions)
        # Legal action list for flat_discrete mode. Each entry is a 6-element
        # int array [action_type, unit_type, from_x, from_y, to_x, to_y] --
        # the same layout as a MultiDiscrete action -- built by
        # ``_build_flat_actions`` (not a dict, despite the per-action dict
        # used elsewhere). ``_flat_actions_key`` records the decision point
        # it was built for (see ``_flat_state_key``); step() rebuilds it when
        # it describes another one (review rlenv-6). ``_flat_n_legal`` /
        # ``_flat_truncated`` are its truncation diagnostics.
        self._current_actions: list[np.ndarray] = []
        self._flat_actions_key: tuple[int, int, int] | None = None
        self._flat_n_legal = 0
        self._flat_truncated = False

        # Grid dimensions
        self.grid_height = self.game_state.grid.height
        self.grid_width = self.game_state.grid.width

        # Padded observation shape (defaults to the live map's dims when
        # ``pad_to_size`` is None). Stored on the env so ``_get_obs`` can
        # forward it to ``build_observation`` without re-checking each step.
        if pad_to_size is not None:
            pad_h, pad_w = int(pad_to_size[0]), int(pad_to_size[1])
            if pad_h < self.grid_height or pad_w < self.grid_width:
                raise ValueError(
                    f"pad_to_size={pad_to_size} is smaller than the live map "
                    f"({self.grid_height}, {self.grid_width}). pad_to_size must "
                    f"be >= the largest map in the curriculum."
                )
            if action_space_type != "flat_discrete":
                # multi_discrete's action_space is MultiDiscrete([10, 8, W, H, W, H]) — sized
                # to the live grid. Padding obs alone would still leave the action space
                # mismatched across stages. flat_discrete's action_space is Discrete(max_flat_actions)
                # so padding obs is sufficient there.
                raise NotImplementedError(
                    "pad_to_size is currently only supported with "
                    "action_space_type='flat_discrete'. multi_discrete's "
                    "action space depends on grid dims; padding it would "
                    "also need a per-dim mask rewrite."
                )
            self.pad_height = pad_h
            self.pad_width = pad_w
        else:
            self.pad_height = self.grid_height
            self.pad_width = self.grid_width

        # Define observation space.
        # ``action_mask`` is intentionally NOT part of the policy observation
        # — MaskablePPO consumes masks via ``action_masks()``, and including
        # the (10*W*H,)-sized mask in the obs dict just bloats the features
        # extractor input without adding any state information.
        obs_dict: dict[str, spaces.Space] = {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(self.pad_height, self.pad_width, GRID_CHANNELS), dtype=np.float32),
            "units": spaces.Box(low=0.0, high=1.0, shape=(self.pad_height, self.pad_width, UNIT_CHANNELS), dtype=np.float32),
            # ``global_features`` are tanh-squashed in ``build_observation``
            # (gold / turn / unit-count divided by their respective scales
            # before tanh), so all five entries land in [0, 1). The Box
            # bounds here advertise that contract to SB3 and to any obs
            # validators downstream.
            "global_features": spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_FEATURES_DIM,), dtype=np.float32),
        }

        # Add visibility layer when fog of war is enabled
        if fog_of_war:
            obs_dict["visibility"] = spaces.Box(
                low=0,
                high=2,  # 0=unexplored, 1=shrouded, 2=visible
                shape=(self.pad_height, self.pad_width),
                dtype=np.uint8,
            )

        self.observation_space = spaces.Dict(obs_dict)

        # Define action space
        if hierarchical:
            # HRL: Manager outputs goals, worker outputs primitive actions
            self.action_space = spaces.Dict(
                {
                    "goal": spaces.Discrete(goal_space_size),  # Manager action
                    "primitive": spaces.MultiDiscrete(
                        [
                            10,  # action_type (0-9)
                            8,  # unit_type (for create): W, M, C, A, K, R, S, B
                            self.grid_width,  # from_x
                            self.grid_height,  # from_y
                            self.grid_width,  # to_x
                            self.grid_height,  # to_y
                        ]
                    ),
                }
            )
        elif action_space_type == "flat_discrete":
            # Flat Discrete: each index maps to a specific legal action.
            # Exact masking eliminates invalid actions entirely. The table
            # layout version rides on the space so it is saved with a model.
            self.action_space = spaces.Discrete(self.max_flat_actions)
            stamp_flat_action_version(self.action_space, self.flat_action_version)
        else:
            # MultiDiscrete: per-dimension masks (over-approximation)
            self.action_space = spaces.MultiDiscrete(
                [
                    10,  # action_type (0-9)
                    8,  # unit_type (for create): W, M, C, A, K, R, S, B
                    self.grid_width,  # from_x
                    self.grid_height,  # from_y
                    self.grid_width,  # to_x
                    self.grid_height,  # to_y
                ]
            )

        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            from reinforcetactics.ui.renderer import Renderer

            self.renderer = Renderer(self.game_state)

        # Episode statistics
        self.episode_stats: dict[str, Any] = self._new_episode_stats()

    def _new_episode_stats(self) -> dict[str, Any]:
        return {
            "reward": 0.0,
            "length": 0,
            "winner": None,
            "invalid_actions": 0,
            # Combat / progression counters surfaced for diagnostics. Keys
            # match what evaluation.evaluate_model aggregates per stage.
            "units_built": {ut: 0 for ut in self.ALL_UNIT_TYPES},
            "captures": 0,
            # Per-structure capture breakdown so eval_results.json can
            # distinguish tower / building / HQ progression. Tile codes
            # come from rules.TileType ("h"=HQ, "b"=Building, "t"=Tower).
            "captures_by_type": {"tower": 0, "building": 0, "hq": 0},
            "kills": 0,
            "attacks": 0,
            "seize_attempts": 0,
            # Damage the agent's units dealt: their own attacks, plus the
            # counter-attacks they made when attacked during the opponent's
            # turn (also broken out in ``counter_damage_dealt``). Nominal
            # damage, as the engine reports it, like ``damage_scale`` pays.
            "damage_dealt": 0.0,
            "counter_damage_dealt": 0.0,
            # HP the agent's units lost to combat: the opponent's attacks
            # during its turn (not netted against the agent's turn-start
            # structure healing, which lands in the same window) plus the
            # counter-attacks the agent's own attacks drew, which are also
            # broken out in ``counter_damage_taken``. HP actually lost: a
            # unit killed by a hit bigger than its HP loses only what it had.
            "damage_taken": 0.0,
            "counter_damage_taken": 0.0,
            # Enemy units the agent killed (``kills``, which includes these):
            # ones that died to its counter-attack during the opponent's turn.
            "counter_kills": 0,
            # Agent units that died: to a counter-attack on its own attack,
            # or to the opponent during its turn.
            "units_lost": 0,
            "structures_lost_neutral": 0,
            "structures_lost_owned": 0,
            # Action-space diagnostics (see step()), all taken at the decision
            # point the action was chosen at:
            #   ``seize_available_steps`` counts steps where a seize action
            #     was legal -- divided by episode length downstream it gives
            #     the "could the agent have captured" rate, which separates
            #     "never reaches a capturable tile" (navigation) from
            #     "reaches one but doesn't seize" (reward/exploration).
            #   ``max_legal_actions`` is the peak legal-action-set size seen
            #     this episode, counted *before* flat_discrete truncation (so
            #     it can exceed max_flat_actions) -- a guardrail for that
            #     truncation and a proxy for army bloat.
            #   ``truncated_steps`` counts decision points whose flat_discrete
            #     table was truncated to max_flat_actions (always 0 for
            #     multi_discrete).
            "seize_available_steps": 0,
            "max_legal_actions": 0,
            "truncated_steps": 0,
            # Army-economy diagnostics (see step()): these separate "the agent
            # wins by massing a big slow army" from "the agent wins with a
            # small precise force". ``peak_own_units`` / ``own_units_sum``
            # (divided by length downstream -> mean) track the agent's army
            # size over the episode; ``peak_gold_banked`` / ``gold_banked_sum``
            # track unspent gold. A monotonically growing army with near-zero
            # banked gold is the "convert-all-gold-to-permanent-free-units"
            # signature -- i.e. the economy (uncapped income, no upkeep) is
            # funding mass, not the reward. ``units_built`` (above) gives the
            # composition.
            "peak_own_units": 0,
            "own_units_sum": 0,
            "peak_gold_banked": 0.0,
            "gold_banked_sum": 0.0,
            # Structure auto-heal economics (see step()): mirrors of
            # ``GameState.healing_totals``, agent-relative. Wounded units
            # parked on owned structures heal 1-2 HP at turn start for
            # (heal/max_hp) * unit_cost gold -- mandatory, no opt-out, and
            # otherwise invisible in every log (the engine's per-turn stats
            # are discarded by callers). ``own_heal_gold`` quantifies the
            # silent drain on the agent's economy; ``opp_heal_hp`` measures
            # how much free durability the opponent's rebuild economy gets
            # -- a direct probe of the random-bot meat-wall / draw machine.
            "own_heal_hp": 0,
            "own_heal_gold": 0,
            "opp_heal_hp": 0,
            "opp_heal_gold": 0,
        }

    def _get_action_space_size(self) -> int:
        """Calculate total action space size for masking."""
        if self.action_space_type == "flat_discrete":
            return self.max_flat_actions
        return 10 * self.grid_width * self.grid_height

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Get current observation from the agent's perspective.

        The action mask is intentionally not included in the returned dict;
        callers needing it should use :meth:`action_masks` (MaskablePPO) or
        :meth:`get_action_mask_flat` (for diagnostics).
        """
        pad_to: tuple[int, int] | None = None
        if self.pad_height != self.grid_height or self.pad_width != self.grid_width:
            pad_to = (self.pad_height, self.pad_width)
        return build_observation(
            self.game_state,
            perspective_player=self.agent_player,
            action_mask=None,
            fog_of_war=self.fog_of_war,
            pad_to=pad_to,
            gold_scale=self.gold_scale,
            turn_scale=self.turn_scale,
            unit_count_scale=self.unit_count_scale,
        )

    def _build_masks(
        self,
    ) -> tuple[
        np.ndarray,  # flat mask  (10*W*H,)
        np.ndarray,
        np.ndarray,  # action_type (10,), unit_type (8,)
        np.ndarray,
        np.ndarray,  # from_x (W,), from_y (H,)
        np.ndarray,
        np.ndarray,  # to_x (W,), to_y (H,)
    ]:
        """
        Compute both the flat target-based mask and per-dimension masks from
        a single ``get_legal_actions`` call.

        Returns:
            (flat_mask, action_type_mask, unit_type_mask,
             from_x_mask, from_y_mask, to_x_mask, to_y_mask)
        """
        # Per-turn action budget gate (see ``_budget_exhausted``). When
        # the agent has used up its budget, advertise only end_turn at
        # the canonical (0, 0) coordinates so the multi_discrete policy
        # has exactly one legal action.
        if self._budget_exhausted():
            width = self.grid_width
            height = self.grid_height
            area = width * height
            flat_mask = np.zeros(self._get_action_space_size(), dtype=np.float32)
            at_mask = np.zeros(10, dtype=bool)
            ut_mask = np.zeros(8, dtype=bool)
            fx_mask = np.zeros(width, dtype=bool)
            fy_mask = np.zeros(height, dtype=bool)
            tx_mask = np.zeros(width, dtype=bool)
            ty_mask = np.zeros(height, dtype=bool)
            at_mask[5] = True
            flat_mask[5 * area] = 1.0
            fx_mask[0] = True
            fy_mask[0] = True
            tx_mask[0] = True
            ty_mask[0] = True
            ut_mask[0] = True
            return flat_mask, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask

        # Masks always describe the *agent's* legal actions (matching
        # ``_build_flat_actions`` and the obs contract). Using
        # ``current_player`` here would silently mask for the opponent
        # whenever a caller queries between the agent's end_turn and the
        # opponent's turn completing.
        return build_per_dim_masks(
            self.game_state,
            self.grid_width,
            self.grid_height,
            enabled_units=self.enabled_units,
            flat_action_size=self._get_action_space_size(),
            player=self.agent_player,
        )

    def _build_structured_masks(self) -> StructuredActionMasks:
        """
        Build dependent masks aligned with an autoregressive policy:
        p(atype) * p(source | atype) * p(unit_type | atype, source) * p(target | atype, source).

        For action_types where the source position is implicit (create_unit, end_turn)
        the source cell is the building / canonical (0, 0). For end_turn (atype=5) a
        single trivial (sx, sy, tx, ty) = (0, 0, 0, 0) entry is recorded so callers
        do not need a special case.

        Masks describe the *agent's* legal actions (see ``_build_masks``).
        Once the per-turn action budget is spent they offer end_turn alone,
        exactly like the other two mask builders (review rlenv-8: the
        feudal autoregressive worker samples from these masks, so without
        the gate its "never end the turn" safety net was off).
        """
        if self._budget_exhausted():
            H, W = self.grid_height, self.grid_width
            atype = np.zeros(10, dtype=bool)
            atype[5] = True
            source = np.zeros((10, H, W), dtype=bool)
            source[5, 0, 0] = True
            end_t = np.zeros((H, W), dtype=bool)
            end_t[0, 0] = True
            return StructuredActionMasks(atype=atype, source=source, target={(5, 0, 0): end_t}, unit_type={})
        return build_structured_masks(
            self.game_state,
            self.grid_width,
            self.grid_height,
            player=self.agent_player,
        )

    def structured_action_masks(self) -> StructuredActionMasks:
        """
        Public accessor for autoregressive-policy masks.

        Returns dependent masks aligned with the factorization
        p(atype) * p(source | atype) * p(unit_type | atype, source) * p(target | ...).
        Use ``encode_structured_action`` to convert a sampled tuple into the
        existing 6-vector action format.
        """
        return self._build_structured_masks()

    def _budget_exhausted(self) -> bool:
        """Whether the agent has spent its per-game-turn action budget.

        Once it has taken ``max_actions_per_turn`` actions without ending
        its turn, every mask builder (flat, per-dimension, structured)
        offers end_turn alone, at the canonical (0, 0) coordinates. Without
        the cap a policy that finds a legal-but-unproductive cycle (idle
        moves, etc.) can spin until ``max_steps`` truncates the episode --
        which surfaces in eval as len near max_steps with turns very low
        (the "never end the turn" attractor). The counter is reset by every
        end_turn the agent executes (see ``step``).
        """
        return self.max_actions_per_turn is not None and self._actions_this_turn >= self.max_actions_per_turn

    def set_self_play_opponent_factory(self, factory) -> None:
        """Register a callable used to (re)build the opponent in self-play.

        ``factory(game_state, opponent_player) -> Bot`` is invoked from
        :meth:`reset` whenever ``opponent_type == "self"``. Pass ``None`` to
        clear. The bot returned must implement ``take_turn()`` against the
        given ``game_state`` (i.e. SimpleBot / ModelBot / etc.).
        """
        self._self_play_opponent_factory = factory

    def set_agent_seat(self, seat: int | str) -> None:
        """Choose which player the agent controls, starting with the next reset().

        Args:
            seat: ``1`` or ``2`` for a fixed seat, or ``"random"`` to draw the
                seat from ``np_random`` on every reset (SelfPlayEnv's
                ``swap_players``). A fixed seat is written to
                ``agent_player`` immediately, but the game only becomes
                consistent with it after ``reset()``: that is where the
                opponent is rebuilt for the other seat and, for seat 2,
                player 1's opening turn is played.
        """
        if isinstance(seat, str):
            if seat != "random":
                raise ValueError(f"agent seat must be 1, 2 or 'random'; got {seat!r}")
            self._random_agent_seat = True
            return
        if seat not in (1, 2):
            raise ValueError(f"agent seat must be 1, 2 or 'random'; got {seat!r}")
        self._random_agent_seat = False
        self.agent_player = int(seat)

    @staticmethod
    def encode_structured_action(
        atype: int,
        sx: int,
        sy: int,
        tx: int,
        ty: int,
        unit_type_idx: int = 0,
    ) -> np.ndarray:
        """Pack a sampled autoregressive tuple into the env's 6-vector action."""
        return np.array([atype, unit_type_idx, sx, sy, tx, ty], dtype=np.int32)

    def _flat_state_key(self) -> tuple[int, int, int]:
        """Identifies the decision point a flat_discrete table was built for.

        The env step count (the action budget gate depends only on it), the
        live game and how many actions that game has recorded: an action
        the caller applied to ``game_state`` directly moves the key too.
        """
        return (self.current_step, id(self.game_state), len(self.game_state.action_history))

    def _build_flat_actions(self):
        """
        Build flat list of all legal actions for Discrete action space mode.

        Each action is stored as a numpy array [action_type, unit_type, from_x,
        from_y, to_x, to_y] — the same format as MultiDiscrete actions — so that
        ``_encode_action`` and ``_execute_action`` work unchanged. The list
        itself comes from the shared :func:`flat_action_table` in this env's
        ``flat_action_version`` (``ModelBot`` decodes flat_discrete
        checkpoints with the same function); only the per-turn action-budget
        gate (``_budget_exhausted``) is env-specific.
        """
        if self._budget_exhausted():
            self._current_actions = [np.array(_END_TURN_ACTION, dtype=np.int32)]
            self._flat_n_legal = 1
            self._flat_truncated = False
        else:
            table = flat_action_table(
                self.game_state, self.agent_player, self.max_flat_actions, version=self.flat_action_version
            )
            self._current_actions = table.actions
            self._flat_n_legal = table.n_legal
            self._flat_truncated = table.truncated
        self._flat_actions_key = self._flat_state_key()

    def _get_action_mask(self) -> np.ndarray:
        """
        Get binary mask of valid actions for the current player.

        For flat_discrete mode, returns mask of shape (max_flat_actions,) where
        each True index maps to a legal game action.

        For multi_discrete mode, returns the flat target-based mask of size
        (10 * W * H,).
        """
        if self.action_space_type == "flat_discrete":
            self._build_flat_actions()
            mask = np.zeros(self.max_flat_actions, dtype=np.float32)
            mask[: len(self._current_actions)] = 1.0
            return mask
        flat_mask, *_ = self._build_masks()
        return flat_mask

    def action_masks(self) -> tuple[np.ndarray, ...]:
        """
        Get action masks for MaskablePPO (sb3-contrib).

        For flat_discrete mode, returns a single exact boolean mask where each
        True index corresponds to a specific legal game action. This eliminates
        invalid actions entirely.

        For multi_discrete mode, returns per-dimension boolean masks (union
        over-approximation).

        Returns:
            Tuple of boolean numpy arrays
        """
        if self.action_space_type == "flat_discrete":
            self._build_flat_actions()
            mask = np.zeros(self.max_flat_actions, dtype=bool)
            mask[: len(self._current_actions)] = True
            return (mask,)
        _, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask = self._build_masks()
        return (at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask)

    def get_action_mask_flat(self) -> np.ndarray:
        """
        Get flattened action mask for compatibility with some algorithms.

        Returns the original target-based mask of size (10 * W * H,).
        """
        return self._get_action_mask()

    def _encode_action(self, action: np.ndarray) -> dict[str, Any]:
        """
        Encode action array into game action.

        Args:
            action: [action_type, unit_type, from_x, from_y, to_x, to_y]

        Returns:
            Dict with action details
        """
        action_type = int(action[0])
        unit_type_idx = int(action[1])
        from_x, from_y = int(action[2]), int(action[3])
        to_x, to_y = int(action[4]), int(action[5])

        unit_types = ["W", "M", "C", "A", "K", "R", "S", "B"]
        unit_type = unit_types[unit_type_idx % 8]

        return {"action_type": action_type, "unit_type": unit_type, "from_pos": (from_x, from_y), "to_pos": (to_x, to_y)}

    def execute_game_action(self, action_dict: dict[str, Any], player: int) -> tuple[dict[str, Any], bool]:
        """
        Execute an encoded action for the given player.

        This is the single dispatch point for all game actions. Both agent
        and opponent action execution flow through here.

        Args:
            action_dict: Encoded action with keys action_type, unit_type,
                from_pos, to_pos.
            player: The player number (1 or 2) performing the action.

        Returns:
            (result_info, is_valid) where result_info contains action-specific
            data (e.g. damage dealt, heal amount).
        """
        action_type = action_dict["action_type"]
        from_pos = action_dict["from_pos"]
        to_pos = action_dict["to_pos"]
        result_info: dict[str, Any] = {"action_type": action_type}
        is_valid = True
        gs = self.game_state

        try:
            if action_type == 0:  # Create unit
                create = {"unit_type": action_dict["unit_type"], "x": to_pos[0], "y": to_pos[1], "player": player}
                is_valid = gs.apply_action("create_unit", create).accepted

            elif action_type == 1:  # Move
                unit = gs.get_unit_at_position(*from_pos)
                if unit and unit.player == player and unit.can_move:
                    is_valid = gs.apply_action("move", {"unit": unit, "to_x": to_pos[0], "to_y": to_pos[1]}).accepted
                else:
                    is_valid = False

            elif action_type == 3:  # Seize
                unit = gs.get_unit_at_position(*from_pos)
                if unit and unit.player == player:
                    seized = gs.apply_action("seize", {"unit": unit})
                    result_info["seize_damage"] = seized.result.get("damage", 0)
                    result_info["captured"] = seized.result.get("captured", False)
                    # Forward structure_type so the reward path can break
                    # captures down by tile type (tower / building / HQ).
                    result_info["structure_type"] = seized.result.get("structure_type")
                    is_valid = seized.accepted
                else:
                    is_valid = False

            elif action_type == 4:  # Heal/Cure (Cleric): cure a paralyzed target, else heal it
                unit = gs.get_unit_at_position(*from_pos)
                target = gs.get_unit_at_position(*to_pos)
                if unit and target and unit.type == "C" and unit.player == player:
                    if target.is_paralyzed() and gs.apply_action("cure", {"curer": unit, "target": target}).accepted:
                        result_info["cured"] = True
                    else:
                        healed = gs.apply_action("heal", {"healer": unit, "target": target})
                        if healed.accepted:
                            result_info["heal_amount"] = healed.result
                        else:
                            is_valid = False
                else:
                    is_valid = False

            elif action_type == 5:  # End turn
                gs.apply_action("end_turn", {})

            elif action_type in _TARGETED_ACTION_TYPES:  # Attack, paralyze, haste, the buffs
                kind, env_check = _TARGETED_ACTION_TYPES[action_type]
                unit = gs.get_unit_at_position(*from_pos)
                target = gs.get_unit_at_position(*to_pos)
                if unit and target and env_check(unit, target, player):
                    attacker_hp_before = unit.health
                    outcome = gs.apply_action(kind, {ACTOR_KEYS[kind]: unit, "target": target})
                    if kind == "attack":
                        result_info["damage"] = outcome.result["damage"]
                        result_info["target_alive"] = outcome.result["target_alive"]
                        # The counter-attack's cost to the attacker (review
                        # rlenv-9): combat shaping charges it on this step.
                        result_info["counter_damage"] = outcome.result["counter_damage"]
                        result_info["attacker_alive"] = outcome.result["attacker_alive"]
                        # The HP the counter actually took. ``counter_damage``
                        # is the nominal hit, overkill included: a 1-HP
                        # attacker killed by a 5-damage counter loses 1 HP,
                        # which is what the opponent-turn measurement (an HP
                        # delta) charges for the same loss. Unit.take_damage
                        # clamps health at 0, so this is exact either way.
                        result_info["attacker_hp_lost"] = attacker_hp_before - unit.health
                    # The engine refuses an illegal action (spent or paralyzed
                    # unit, out of range, hidden by fog, wrong turn) and
                    # changes nothing. multi_discrete per-dimension masks
                    # over-approximate the legal set, so such combinations are
                    # sampled and must be penalised, not rewarded (review
                    # rlenv-2).
                    is_valid = outcome.accepted
                else:
                    is_valid = False

        except (ValueError, KeyError, IndexError) as e:
            logger.debug("Game action failed (type=%s): %s", action_type, e)
            is_valid = False
        except (TypeError, AttributeError):
            # Programming errors should propagate so they are not silently ignored
            raise
        except Exception as e:
            logger.error("Unexpected error executing action (type=%s): %s\n%s", action_type, e, traceback.format_exc())
            is_valid = False

        return result_info, is_valid

    def _execute_action(self, action_dict: dict[str, Any]) -> tuple[float, bool]:
        """
        Execute encoded action for the agent and compute reward.

        Returns:
            (reward, is_valid)
        """
        rc = self.reward_config
        ap = self.agent_player
        action_type = action_dict["action_type"]

        result_info, is_valid = self.execute_game_action(action_dict, ap)

        reward = 0.0
        if is_valid:
            if action_type == 0:
                reward += rc["create_unit"]
                ut_letter = action_dict.get("unit_type")
                if ut_letter in self.episode_stats["units_built"]:
                    self.episode_stats["units_built"][ut_letter] += 1
            elif action_type == 1:
                reward += rc["move"]
            elif action_type == 2:
                damage = result_info["damage"]
                reward += damage * rc["damage_scale"]
                self.episode_stats["attacks"] += 1
                self.episode_stats["damage_dealt"] += float(damage)
                if not result_info["target_alive"]:
                    reward += rc["kill"]
                    self.episode_stats["kills"] += 1
                # The exchange's other half (review rlenv-9): the counter-attack
                # this attack drew is damage taken, charged here rather than
                # left to the unit_diff potential -- otherwise a trade that
                # loses more HP to the counter than it deals still pays, and a
                # suicide attack costs nothing. Charged as the HP the attacker
                # actually lost, as the opponent-turn window measures it: the
                # nominal counter overcharged a unit that had less HP left.
                hp_lost = result_info["attacker_hp_lost"]
                if hp_lost:
                    reward += hp_lost * rc["damage_taken_scale"]
                    self.episode_stats["damage_taken"] += float(hp_lost)
                    self.episode_stats["counter_damage_taken"] += float(hp_lost)
                if not result_info["attacker_alive"]:
                    reward += rc["unit_lost"]
                    self.episode_stats["units_lost"] += 1
            elif action_type == 3:
                reward += rc["seize_progress"]
                self.episode_stats["seize_attempts"] += 1
                if result_info.get("captured", False):
                    self.episode_stats["captures"] += 1
                    # Bucket the capture by structure type. Tile codes:
                    # "t" = tower, "b" = building, "h" = headquarters.
                    structure_code = result_info.get("structure_type")
                    type_key = (
                        {"t": "tower", "b": "building", "h": "hq"}.get(structure_code)
                        if isinstance(structure_code, str)
                        else None
                    )
                    if type_key is not None:
                        self.episode_stats["captures_by_type"][type_key] += 1
                    # Per-type capture reward overrides the global ``capture``
                    # weight when present. Falls back to ``capture`` so existing
                    # configs that only set the global key keep current behavior.
                    type_reward_key = f"{type_key}_capture" if type_key else None
                    if type_reward_key and type_reward_key in rc:
                        reward += rc[type_reward_key]
                    else:
                        reward += rc["capture"]
            elif action_type == 4:
                if result_info.get("cured"):
                    reward += rc["cure"]
                elif result_info.get("heal_amount", 0) > 0:
                    reward += result_info["heal_amount"] * rc["heal_scale"]
            elif action_type == 5:
                reward += rc["turn_penalty"]
                # Opponent plays (dispatch on opponent_type, not opponent object).
                # ``"self"`` is dispatched like any scripted bot: reset() bound a
                # factory-built opponent (ModelBot / snapshot) to the live
                # game_state, and ``_opponent_turn`` runs its take_turn() here.
                # With no factory bound, ``self.opponent`` is None,
                # ``_opponent_turn`` no-ops, and the safety net below hands the
                # turn straight back to the agent. The SelfPlayEnv wrapper
                # plays through this same path (it registers the factory), so
                # opponent-turn penalties and opponent-turn terminals are
                # scored identically in self-play and bot training.
                if self.opponent_type:
                    if not self.game_state.game_over:
                        # Snapshot agent unit HP by id() so we can attribute
                        # the opponent turn's damage to the agent. Tracks
                        # both wounded survivors (HP delta) and units that
                        # died entirely (full remaining HP counted as taken).
                        pre_hp = {id(u): u.health for u in self.game_state.units if u.player == ap}
                        # The agent's turn-start structure healing runs inside
                        # the opponent's end_turn, i.e. inside this window, and
                        # used to net the damage down (review rlenv-9). It is
                        # the only other HP change the agent's units see here,
                        # so adding it back makes the sum exact.
                        pre_healed = self._healed_hp(ap)
                        # Snapshot structure ownership before the opponent
                        # moves so any captures the opponent makes during
                        # their turn can be attributed back as a penalty
                        # on the agent's reward (the action_type==3 branch
                        # only credits the agent's own captures).
                        pre_owners = {id(t): t.player for t in self.game_state.grid.get_capturable_tiles()}
                        # Where the opponent's records start, to credit the
                        # agent's counter-attacks below.
                        history_start = len(self.game_state.action_history)
                        self._opponent_turn()
                        # Safety net: if the opponent's take_turn() did not end its
                        # turn for some reason, end it here so play returns to the agent.
                        if not self.game_state.game_over:
                            if self.game_state.current_player != self.agent_player:
                                self.game_state.end_turn()
                        post_hp = {id(u): u.health for u in self.game_state.units if u.player == ap}
                        healed = self._healed_hp(ap) - pre_healed
                        hp_lost = sum(hp - post_hp.get(uid, 0) for uid, hp in pre_hp.items())
                        damage_taken = max(0, hp_lost + healed)
                        units_lost = sum(1 for uid in pre_hp if uid not in post_hp)
                        self.episode_stats["damage_taken"] += float(damage_taken)
                        # Symmetric-combat penalty: charge for damage taken so
                        # mutual trading nets ~0 and only decisive combat pays
                        # (see ``damage_taken_scale`` in the default config).
                        # Attributed to the end_turn step, mirroring how the
                        # opponent-capture penalties below are attributed.
                        reward += damage_taken * rc["damage_taken_scale"]
                        if units_lost:
                            reward += units_lost * rc["unit_lost"]
                            self.episode_stats["units_lost"] += units_lost
                        # The other half of the exchanges the opponent
                        # started: the agent's counter-attacks. Credited like
                        # the agent's own attacks (nominal damage *
                        # damage_scale, ``kill`` per enemy killed), so combat
                        # shaping scores an exchange the same whichever side
                        # swings first. Only the damage taken used to count
                        # here: a mutual trade the opponent started always
                        # netted negative, and a counter-kill paid nothing.
                        reward += self._credit_counter_attacks(history_start)
                        # Tiered opponent-capture penalty. ``neutral_lost``
                        # fires when the opponent seized an unowned tile
                        # (we lost a race); ``owned_lost`` fires when the
                        # opponent flipped one of our tiles (we lost
                        # ground). HQ flips end the game, so the loss
                        # terminal handles them -- skip them here.
                        opp = 3 - ap
                        neutral_lost = 0
                        owned_lost = 0
                        for t in self.game_state.grid.get_capturable_tiles():
                            if t.player != opp:
                                continue
                            prev = pre_owners.get(id(t))
                            if prev == opp:
                                continue
                            if t.type == "h":
                                continue
                            if prev is None or prev == 0:
                                neutral_lost += 1
                            elif prev == ap:
                                owned_lost += 1
                        # Penalty keys are signed (negative magnitudes in the
                        # reward_config), added directly -- matches the
                        # convention used by turn_penalty / invalid_action.
                        if neutral_lost:
                            reward += neutral_lost * rc["enemy_neutral_capture"]
                            self.episode_stats["structures_lost_neutral"] += neutral_lost
                        if owned_lost:
                            reward += owned_lost * rc["enemy_owned_capture"]
                            self.episode_stats["structures_lost_owned"] += owned_lost
            elif action_type == 6:
                reward += rc["paralyze"]
            elif action_type == 7:
                reward += rc["haste"]
            elif action_type == 8:
                reward += rc["defence_buff"]
            elif action_type == 9:
                reward += rc["attack_buff"]

        return reward, is_valid

    def _credit_counter_attacks(self, history_start: int) -> float:
        """Reward the agent's counter-attacks on the opponent's attacks since ``history_start``.

        Reads the engine's ``attack`` records (``GameState.action_history``)
        made by the opponent: each carries the counter the defending agent
        unit dealt (``counter_damage``) and whether it killed the attacker
        (``attacker_killed``). Counter-attacks are the only damage the agent
        deals during the opponent's turn.
        """
        rc = self.reward_config
        opp = 3 - self.agent_player
        reward = 0.0
        for record in self.game_state.action_history[history_start:]:
            if record.get("type") != "attack" or record.get("player") != opp:
                continue
            counter = record.get("counter_damage") or 0
            if counter:
                reward += counter * rc["damage_scale"]
                self.episode_stats["damage_dealt"] += float(counter)
                self.episode_stats["counter_damage_dealt"] += float(counter)
            if record.get("attacker_killed"):
                reward += rc["kill"]
                self.episode_stats["kills"] += 1
                self.episode_stats["counter_kills"] += 1
        return reward

    def _healed_hp(self, player: int) -> int:
        """HP structure auto-heal has restored to ``player``'s units this game (``GameState.healing_totals``)."""
        healing_totals = getattr(self.game_state, "healing_totals", None) or {}
        return int((healing_totals.get(player) or {}).get("hp", 0))

    def _opponent_turn(self):
        """Execute opponent's turn."""
        # reset() binds ``self.opponent`` for every opponent type that plays
        # (a scripted bot, or the self-play factory's snapshot); ``None``
        # (manual mode, or 'self' with no factory) no-ops, and the caller's
        # safety net hands the turn back to the agent.
        if self.opponent is not None:
            self.opponent.take_turn()

    def _compute_potential(self) -> float:
        """
        Compute potential function Phi(s) for potential-based reward shaping.

        Using potential-based shaping (Ng et al., 1999) preserves optimal policy:
        shaping = gamma * Phi(s') - Phi(s).
        """
        potential = 0.0
        ap = self.agent_player
        opp = 3 - ap

        # Gates use ``!= 0`` (not ``> 0``): a zero weight skips the term's
        # computation entirely (the income calc in particular is the
        # expensive one), while a *negative* weight is a legitimate config
        # choice and must not be silently dropped.
        if self.reward_config["income_diff"] != 0:
            income_agent = self.game_state.mechanics.calculate_income(ap, self.game_state.grid, self.game_state.income_rates)
            income_opp = self.game_state.mechanics.calculate_income(opp, self.game_state.grid, self.game_state.income_rates)
            potential += (income_agent["total"] - income_opp["total"]) * self.reward_config["income_diff"]

        if self.reward_config["unit_diff"] != 0:
            units_agent = sum(1 for u in self.game_state.units if u.player == ap)
            units_opp = sum(1 for u in self.game_state.units if u.player == opp)
            potential += (units_agent - units_opp) * self.reward_config["unit_diff"]

        if self.reward_config["structure_control"] != 0:
            structures_agent = len(self.game_state.grid.get_capturable_tiles(player=ap))
            structures_opp = len(self.game_state.grid.get_capturable_tiles(player=opp))
            potential += (structures_agent - structures_opp) * self.reward_config["structure_control"]

        return potential

    def _calculate_reward(
        self, action_reward: float, is_valid: bool, terminated: bool = False
    ) -> tuple[float, dict[str, float]]:
        """Calculate total reward including potential-based shaping terms.

        Args:
            action_reward: Reward from the executed action.
            is_valid: Whether the action was a valid game action.
            terminated: True if the *game* ended on this step (a win, a loss,
                or a max-turns draw). Potential-based shaping (Ng et al.,
                1999) requires Phi(terminal) = 0, which means charging
                ``F = gamma*0 - Phi(s_prev)`` here -- not skipping the term.
                Pass False for a step-limit truncation: the state is real and
                the learner bootstraps its value, so the ordinary
                ``gamma*Phi(s') - Phi(s)`` delta is the correct one there.

        Returns:
            (total_reward, breakdown) where breakdown has keys
            ``action``, ``invalid_penalty``, ``shaping_delta``. Terminal
            (win/loss/draw) is summed in by the caller and stored under
            ``terminal``.
        """
        breakdown = {
            "action": float(action_reward),
            "invalid_penalty": 0.0,
            "shaping_delta": 0.0,
            "terminal": 0.0,
        }
        reward = action_reward

        if not is_valid:
            penalty = self.reward_config["invalid_action"]
            reward += penalty
            breakdown["invalid_penalty"] = float(penalty)
            self.episode_stats["invalid_actions"] += 1

        # Potential-based reward shaping: reward += gamma * Phi(s') - Phi(s).
        # Using the trainer's discount preserves the policy-invariance
        # guarantee from Ng et al. (1999); a gamma=1 approximation biases
        # the policy toward states where Phi is sustained at a high value
        # over long horizons.
        #
        # Ng et al. requires Phi(terminal) = 0. *Skipping* the shaping on
        # terminal steps is not the same thing: it leaves the telescoping sum
        # with a dangling ``+gamma^(T-1) * Phi(s_(T-1))`` instead of collapsing
        # to the constant ``-Phi(s_0)``, so the guarantee formally does not
        # hold. Charging ``F = gamma*0 - Phi(s_prev)`` on the terminal step
        # restores it.
        #
        # Only a real game termination gets that treatment. A step-limit
        # truncation is an artificial cutoff whose successor state is real and
        # whose value the learner bootstraps, so it takes the ordinary delta.
        #
        # NOTE on magnitude: the per-step delta for a quiet board is
        # ``(gamma - 1) * Phi``, i.e. a drain proportional to *step count*.
        # At gamma=0.99 over ~1900 micro-actions that is ~-190 per episode for
        # a 10-structure lead. This terminal charge makes the *discounted*
        # shaping return exactly -Phi(s_0); it does not shrink the per-step
        # drain. That is governed by ``(1 - gamma)`` and by how many env steps
        # an episode takes -- see docs/REVIEW_rl_pipeline_2026-07-24.md 2.1.
        if terminated:
            delta = -self._prev_potential
        else:
            current_potential = self._compute_potential()
            delta = self.gamma * current_potential - self._prev_potential
            self._prev_potential = current_potential
        reward += delta
        breakdown["shaping_delta"] = float(delta)

        return reward, breakdown

    def _decision_point_diagnostics(self) -> tuple[int, int, bool, bool]:
        """Legal-action diagnostics for the decision point the agent acts at.

        Taken *before* the action runs, in both action-space modes (the
        multi_discrete path used to count the post-step state's actions
        while flat_discrete counted the pre-step table). Once the action
        budget is spent the agent is offered end_turn alone, and that is
        what is reported.

        Returns:
            ``(n_offered, n_legal, truncated, seize_available)``: actions
            offered to the policy, legal actions before flat_discrete
            truncation, whether the table was truncated, and whether a seize
            was on offer.
        """
        if self.action_space_type == "flat_discrete":
            n_offered = len(self._current_actions)
            seize_available = any(int(a[0]) == 3 for a in self._current_actions)
            return n_offered, max(self._flat_n_legal, n_offered), self._flat_truncated, seize_available
        if self._budget_exhausted():
            return 1, 1, False, False
        legal_actions = self.game_state.get_legal_actions(player=self.agent_player)
        # get_legal_actions returns lists of action dicts plus a boolean
        # "end_turn" flag — count list lengths, then +1 for end_turn.
        n_legal = sum(len(v) for v in legal_actions.values() if isinstance(v, list)) + 1
        return n_legal, n_legal, False, bool(legal_actions.get("seize"))

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step.

        For ``flat_discrete``, ``action`` is an index into the legal-action
        table for the current decision point -- the table ``action_masks()``
        built, or a fresh one if the state has moved on since (or it was
        never asked for). An index outside the table is an invalid action:
        nothing is executed and the ``invalid_action`` penalty applies
        (it used to execute end_turn for free).

        Returns:
            observation, reward, terminated, truncated, info
        """
        # In hierarchical mode, extract the primitive action from the Dict
        if self.hierarchical and isinstance(action, dict):
            action = action["primitive"]

        # For flat_discrete, map the integer index to the actual action array.
        # The table must describe *this* decision point (review rlenv-6):
        # rebuild it when the last one was built for another.
        flat_index_invalid = False
        if self.action_space_type == "flat_discrete":
            if self._flat_actions_key != self._flat_state_key():
                self._build_flat_actions()
            action_idx = int(action)
            if 0 <= action_idx < len(self._current_actions):
                action = self._current_actions[action_idx]
            else:
                flat_index_invalid = True

        # Diagnostics describe the decision point the action was chosen at,
        # so they are taken before it runs.
        n_legal_actions, n_legal_pre_truncation, flat_truncated, seize_available = self._decision_point_diagnostics()

        self.current_step += 1
        self.episode_stats["length"] = self.current_step

        if flat_index_invalid:
            action_dict: dict[str, Any] = {
                "action_type": self.INVALID_FLAT_INDEX_ACTION_TYPE,
                "unit_type": None,
                "from_pos": None,
                "to_pos": None,
            }
            # A spent step like any other invalid action: it counts against
            # the per-turn budget and pays the invalid-action penalty.
            self._actions_this_turn += 1
            action_reward, is_valid = 0.0, False
        else:
            # Decode and execute action
            action_dict = self._encode_action(action)

            # Bump / reset the per-game-turn action counter that drives the
            # mask-narrowing safety net (see ``_budget_exhausted``). The
            # update happens *before* dispatch so that an end_turn step
            # clears the counter for the next agent turn, while non-end_turn
            # steps see the post-increment value reflected on the next
            # mask query. ``_execute_action`` handles the opponent's turn
            # internally for action_type=5, so the counter doesn't need to
            # be touched again afterward.
            if action_dict["action_type"] == 5:
                self._actions_this_turn = 0
            else:
                self._actions_this_turn += 1

            action_reward, is_valid = self._execute_action(action_dict)

        # Determine terminal status BEFORE shaping so potential-based shaping
        # can charge ``-Phi(s_prev)`` on a real termination (where
        # Phi(terminal) must be 0 for the shaping to be policy-invariant).
        terminated = self.game_state.game_over
        truncated = self.current_step >= self.max_steps

        # Calculate total reward
        reward, breakdown = self._calculate_reward(action_reward, is_valid, terminated=terminated)
        self.episode_stats["reward"] += reward

        # How the episode ended, for eval/diagnostics (win/loss/draw split by
        # the actual game-over condition) and for the terminal bonus
        # (HQ-capture wins vs elimination wins). A game that ended says why
        # itself: ``GameState.end_reason`` (hq_capture / elimination /
        # max_turns_draw / resign), recorded by ``_set_game_over``. It used
        # to be guessed from the loser's unit count, which labels an HQ
        # capture against a side with no units left (a noop opponent) as an
        # elimination and pays it the wrong terminal (review rlenv-5).
        # ``max_steps_truncate``: the env step counter hit max_steps before
        # the game produced a terminal state.
        end_reason: str | None = None
        if terminated:
            end_reason = self.game_state.end_reason
        elif truncated:
            end_reason = "max_steps_truncate"

        rc = self.reward_config
        terminal_bonus = 0.0
        if terminated:
            if self.game_state.winner == self.agent_player:
                # Differentiate win-by-HQ-capture (intended goal) from
                # win-by-elimination (alternative path that doesn't transfer
                # to bigger maps). Falls back to the unified "win" key when
                # the per-reason keys aren't configured, preserving
                # backwards-compatibility with older reward_config dicts.
                if end_reason == "hq_capture" and "win_by_hq_capture" in rc:
                    terminal_bonus = rc["win_by_hq_capture"]
                elif end_reason == "elimination" and "win_by_elimination" in rc:
                    terminal_bonus = rc["win_by_elimination"]
                else:
                    terminal_bonus = rc["win"]
                # Speed bonus: linearly rewards winning early. Needs both
                # a configured magnitude and a finite max_turns to scale
                # against; with max_turns=None we'd have no horizon and
                # the bonus is skipped. Capped at 0 below if the win
                # somehow lands past max_turns (defensive — game would
                # have terminated as max_turns_draw before then).
                speed_bonus = rc["win_speed_bonus"]
                if speed_bonus > 0 and self.max_turns:
                    remaining = max(0, self.max_turns - self.game_state.turn_number)
                    terminal_bonus += speed_bonus * remaining / self.max_turns
                self.episode_stats["winner"] = self.agent_player
            elif self.game_state.winner is None:
                # Draw (e.g. max_turns reached)
                terminal_bonus = rc["draw"]
                self.episode_stats["winner"] = None
            else:
                terminal_bonus = rc["loss"]
                self.episode_stats["winner"] = self.game_state.winner
        elif truncated:
            # Step-limit truncation. ``truncated=True`` is the correct
            # Gymnasium signal, and SB3's on-policy rollout collector responds
            # to it by adding ``gamma * V(terminal_obs)`` to this step's reward
            # (the ``TimeLimit.truncated`` path in ``collect_rollouts``).
            #
            # Charging the ``draw`` terminal here as well double-counts: the
            # transition would carry both "this is a terminal draw, take the
            # penalty" and "this is an artificial cutoff, keep your future
            # value". Measured on the deepest archived run, truncation fires on
            # 27-42 of every 80 eval episodes, so this is not a corner case.
            #
            # The bootstrap value already prices "the game was unfinished", so
            # the default charge is 0. Set ``reward_config['truncation']`` to
            # reinstate an explicit penalty -- but note it stacks with the
            # bootstrap rather than replacing it.
            terminal_bonus = rc["truncation"]
            self.episode_stats["winner"] = None

        reward += terminal_bonus
        breakdown["terminal"] = float(terminal_bonus)

        # Accumulate the action-space diagnostics into episode_stats so the
        # eval pipeline (which only reads the terminal info dict) can surface
        # per-stage seize-availability, peak legal-action-set size and how
        # often the flat_discrete table was truncated.
        self.episode_stats["max_legal_actions"] = max(self.episode_stats["max_legal_actions"], int(n_legal_pre_truncation))
        if seize_available:
            self.episode_stats["seize_available_steps"] += 1
        if flat_truncated:
            self.episode_stats["truncated_steps"] += 1

        # Army-economy diagnostics: sample the agent's army size and unspent
        # gold at every decision point. Sampling per-action (not per game-turn)
        # weights the mean toward busy turns, which is fine -- peak is the
        # unambiguous signal and mean is a coarse trend proxy.
        own_units = sum(1 for u in self.game_state.units if u.player == self.agent_player)
        own_gold = float(self.game_state.player_gold.get(self.agent_player, 0))
        self.episode_stats["peak_own_units"] = max(self.episode_stats["peak_own_units"], int(own_units))
        self.episode_stats["own_units_sum"] += int(own_units)
        self.episode_stats["peak_gold_banked"] = max(self.episode_stats["peak_gold_banked"], own_gold)
        self.episode_stats["gold_banked_sum"] += own_gold

        # Structure auto-heal economics. ``GameState.healing_totals`` is
        # game-cumulative, so overwriting each step is idempotent and the
        # terminal snapshot carries the whole episode. Read via getattr so
        # a stale pickled GameState (pre-accumulator schema) degrades to
        # zeros instead of crashing the step loop.
        healing_totals = getattr(self.game_state, "healing_totals", None)
        if healing_totals:
            own_heal = healing_totals.get(self.agent_player) or {}
            opp_heal = healing_totals.get(3 - self.agent_player) or {}
            self.episode_stats["own_heal_hp"] = int(own_heal.get("hp", 0))
            self.episode_stats["own_heal_gold"] = int(own_heal.get("gold", 0))
            self.episode_stats["opp_heal_hp"] = int(opp_heal.get("hp", 0))
            self.episode_stats["opp_heal_gold"] = int(opp_heal.get("gold", 0))

        # Get observation
        obs = self._get_obs()

        # Info dict
        info = {
            "episode_stats": self.episode_stats.copy() if terminated or truncated else {},
            "game_over": terminated,
            "winner": self.game_state.winner if terminated else None,
            "end_reason": end_reason,
            "turn": self.game_state.turn_number,
            "valid_action": is_valid,
            # INVALID_FLAT_INDEX_ACTION_TYPE (-1) when a flat_discrete index
            # named no table entry and nothing ran.
            "action_type": action_dict["action_type"],
            # Surface unit_type only for create_unit (action_type=0) actions —
            # so per-game diagnostics can break the create_unit bar down by
            # which unit was actually spawned (W/M/A/etc.). None for other
            # actions to keep readers from accidentally treating it as
            # meaningful (e.g. for moves the action_dict has a unit_type
            # field describing the moving unit, but that's a different thing).
            "unit_type": action_dict.get("unit_type") if action_dict["action_type"] == 0 and is_valid else None,
            "reward_breakdown": breakdown,
            # Mask coverage at the decision point the action was chosen at:
            # the actions offered to the policy (flat_discrete: the table
            # length; multi_discrete: the engine's legal set), the legal set
            # before flat_discrete truncation, and whether it was truncated.
            "n_legal_actions": int(n_legal_actions),
            "n_legal_actions_pre_truncation": int(n_legal_pre_truncation),
            "flat_actions_truncated": bool(flat_truncated),
            "seize_available": bool(seize_available),
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset environment."""
        # Resolve the opponent first: ``opponent_type`` may have been
        # reassigned since __init__ (SelfPlayEnv sets "self"), and an unknown
        # name must fail here rather than quietly play with no opponent.
        opponent_name = resolve_opponent(self.opponent_type)
        validate_opponent_kwargs(opponent_name, self.opponent_kwargs)

        super().reset(seed=seed)

        # Seat draw for the "random" seat mode (self-play swaps). Drawn first
        # and only in that mode, so reset(seed=...) fixes the seat and
        # fixed-seat episode streams stay byte-identical with the historic
        # behaviour.
        if self._random_agent_seat:
            self.agent_player = int(self.np_random.integers(1, 3))

        # Engine-side combat RNG (currently only the Rogue evade roll in
        # mechanics.attack_unit). Derived from np_random so reset(seed=...)
        # controls combat stochasticity the same way it controls bot
        # tiebreaks below — without this the evade roll reads the module-
        # global ``random``, which the episode seed never touches (and which
        # forked SubprocVecEnv workers inherit in identical states).
        # Passed as a seed (GameState builds the same random.Random from it)
        # so the value is recorded in the game's saves and replays.
        engine_seed = int(self.np_random.integers(0, 2**31 - 1))

        # Reset game state (preserving enabled_units, fog_of_war, max_turns
        # and the engine-constant overlay)
        self.game_state = GameState(
            self.initial_map_data,
            num_players=2,
            max_turns=self.max_turns,
            enabled_units=self.enabled_units,
            fog_of_war=self.fog_of_war,
            engine_overrides=self.engine_overrides,
            seed=engine_seed,
        )
        self.current_step = 0
        self._actions_this_turn = 0
        # The flat_discrete table described the previous game; it is rebuilt
        # for this one on the next action_masks() / step() (review rlenv-6).
        self._current_actions = []
        self._flat_actions_key = None
        self._flat_n_legal = 0
        self._flat_truncated = False

        # Reset opponent.
        #
        # All bot types -- including the scripted ones (SimpleBot / MediumBot
        # / AdvancedBot) -- are seeded with a per-episode rng derived from
        # the env's np_random. Without this the scripted bots play
        # deterministically given the starting state, which collapses
        # cross-episode return variance to zero on the stochastic-tiebreak
        # axis and drives the draw-with-shaping policy-drift attractor
        # documented in docs/bootstrap_lessons_learned.md. Seeding from
        # np_random keeps reset(seed=...) reproducible while injecting
        # genuine per-episode opponent variance.
        opponent_player = 3 - self.agent_player
        if opponent_name == "noop":
            # NoopBot never chooses anything — no rng, and deliberately no
            # np_random draw so seeded episode streams stay byte-identical
            # with the historic behavior.
            self.opponent = build_scripted_bot("noop", self.game_state, player=opponent_player)
        elif opponent_name is not None and opponent_name != SELF_PLAY_OPPONENT:
            # Any other scripted opponent, built through the bot registry
            # ("bot" is "simple", and "master" plays too). ``opponent_kwargs``
            # were validated against the bot's constructor above: only the
            # stochastic bots take any.
            bot_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.opponent = build_scripted_bot(
                opponent_name,
                self.game_state,
                player=opponent_player,
                rng=random.Random(bot_seed),
                **self.opponent_kwargs,
            )
        elif opponent_name == SELF_PLAY_OPPONENT:
            # Self-play: the training script supplies a callable that builds
            # an opponent bot bound to the freshly-reset game_state. Without
            # one we leave the slot empty — _opponent_turn safely no-ops if
            # ``self.opponent is None``.
            if self._self_play_opponent_factory is not None:
                try:
                    self.opponent = self._self_play_opponent_factory(self.game_state, opponent_player)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.warning("self-play opponent factory raised %s; running with no opponent", exc)
                    self.opponent = None
            else:
                self.opponent = None
        else:
            self.opponent = None

        # Player 1 always moves first. With the agent in seat 2, play the
        # opponent's opening turn here so the episode starts on the agent's
        # own turn. Without this the agent would be offered (and execute)
        # moves during player 1's turn, and player 1 would never get its
        # first turn. Gated on ``opponent_type`` exactly like the end_turn
        # branch of ``_execute_action``: with no opponent (manual mode) the
        # caller drives player 1 itself. The trailing end_turn is the same
        # safety net as there, for an opponent whose take_turn() (or an
        # empty self-play slot) did not hand the turn over.
        if self.opponent_type and self.game_state.current_player != self.agent_player:
            self._opponent_turn()
            if not self.game_state.game_over and self.game_state.current_player != self.agent_player:
                self.game_state.end_turn()

        # Initialize prev potential to Phi(s_0) so the first step's shaping
        # delta is Phi(s_1) - Phi(s_0), preserving the policy-invariance
        # property of potential-based reward shaping (Ng et al., 1999).
        # Computed after the opening turn above: s_0 is the first state the
        # agent observes, not the pre-opening board.
        self._prev_potential = self._compute_potential()

        # Reset renderer
        if self.render_mode == "human" and self.renderer:
            from reinforcetactics.ui.renderer import Renderer

            self.renderer = Renderer(self.game_state)

        # Reset episode stats
        self.episode_stats = self._new_episode_stats()

        obs = self._get_obs()
        info: dict[str, Any] = {}

        return obs, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.renderer:
                self.renderer.render()
        elif self.render_mode == "rgb_array":
            if self.renderer:
                return self.renderer.get_rgb_array()

    def close(self):
        """Clean up."""
        if self.renderer:
            self.renderer.close()
