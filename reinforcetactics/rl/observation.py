"""
Shared observation builder for RL agents.

Single source of truth for turning a GameState into the observation dict
consumed by the gym environment, MCTS, and ModelBot. The observation is
*agent-relative*: ownership channels and global_features always treat
``perspective_player`` as "self" so a policy trained as player 1 sees the
same encoding when it later plays as player 2 (e.g. under self-play role
swaps).

Observation contract (1v1 only, agent-relative):
    grid:            (H, W, C_GRID=11) float32
        channels 0-7  one-hot tile type in TILE_TYPE_ORDER
        channel  8    owner == self
        channel  9    owner == opp
        channel  10   structure HP fraction in [0, 1]
        Neutral ownership is encoded implicitly as ``(own=0, opp=0)`` — the
        explicit neutral channel was dropped as a linear combination of the
        other two owner channels.
    units:           (H, W, C_UNITS=16) float32
        channels 0-7  one-hot unit type in ALL_UNIT_TYPES
                      ("empty cell" = all-zero across these eight channels)
        channel  8    owner == self
        channel  9    owner == opp
        channel  10   own_exhausted: 1.0 iff this is an own unit with no
                      actions left this turn (``not (can_move or
                      can_attack)``); always 0.0 for opponent units (they
                      don't act on our turn). Captures every way a unit
                      spends its turn -- move, attack, seize, heal, cast --
                      not just movement, so a unit that attacked in place
                      reads 1.0 while a unit that moved but can still attack
                      reads 0.0. Lets the policy / value head see which own
                      units are spent without inferring it from the action
                      mask.
        channel  11   unit HP fraction in [0, 1]
        channel  12   paralyzed_turns / PARALYZE_DURATION (Mage debuff;
                      remaining turns the unit cannot act, normalised
                      to [0, 1]).
        channel  13   is_hasted (Sorcerer haste buff; 1.0 iff the unit has
                      an extra action queued this turn, 0.0 otherwise).
        channel  14   defence_buff_turns / SORCERER_BUFF_DURATION
                      (Sorcerer defence buff; remaining turns of -50%
                      damage taken, normalised to [0, 1]).
        channel  15   attack_buff_turns / SORCERER_BUFF_DURATION
                      (Sorcerer attack buff; remaining turns of +50%
                      damage dealt, normalised to [0, 1]).
    global_features: (5,) float32, each in [0, 1)
        [own_gold, opp_gold, turn, own_units, opp_units], each squashed
        through ``tanh(x / scale)`` so all dims share a comparable
        magnitude regardless of map size or game length. Raw inputs are
        monotonically non-decreasing in expectation (gold compounds via
        income, ``turn_number`` is strictly monotonic, unit counts trend
        up over a game), so without normalization a flat MLP head sees
        gold values ~10^3 alongside unit counts ~10^1 — the dominant
        feature swamps the gradients. ``tanh`` keeps the typical regime
        roughly linear and saturates gracefully on the tails (e.g.
        a runaway 10k-gold game). Scale defaults live in module-level
        constants (``GOLD_SCALE`` / ``TURN_SCALE`` / ``UNIT_COUNT_SCALE``)
        and can be overridden per-call (e.g. by ``StrategyGameEnv`` from
        ``EnvConfig``).
        ``current_player`` is intentionally omitted: the agent only acts on
        its own turn, so this slot is always equal to ``perspective_player``
        and only adds spurious side information.
    visibility:      (H, W) uint8, present only when fog_of_war is True.

Action masks are not part of this dict — they're a constraint, not state,
and MaskablePPO already pulls them via ``env.action_masks()``. The
``action_mask`` argument is retained for back-compat with non-PPO consumers
(MCTS / AlphaZero, which want a packaged tuple of (obs, mask)) and is
included in the returned dict only when explicitly provided.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from reinforcetactics.constants import (
    ALL_UNIT_TYPES,
    PARALYZE_DURATION,
    SORCERER_BUFF_DURATION,
)

# Canonical tile-type ordering for the one-hot encoding. Must match the
# integer codes produced by ``TileGrid.to_numpy`` (see core/grid.py).
# Note: ocean ("o") is not a separate entry — ``TileGrid.to_numpy`` maps it
# onto the water ("w") code (index 1) since the two are mechanically
# identical (both impassable). Keep these two definitions in sync.
TILE_TYPE_ORDER = ("p", "w", "m", "f", "r", "b", "h", "t")
NUM_TILE_TYPES = len(TILE_TYPE_ORDER)
NUM_UNIT_TYPES = len(ALL_UNIT_TYPES)

# Channel layout constants — exposed so model factories / tests can size
# input layers without duplicating the encoding logic.
GRID_CHANNELS = NUM_TILE_TYPES + 2 + 1  # 11: 8 tile + 2 owner (self/opp) + 1 hp
# 16: 8 unit type + 2 owner + own_acted + hp + 4 status (paralyze,
# haste, defence_buff, attack_buff). The four status channels carry
# debuff / buff timers so the policy can value attacking paralyzed
# targets, predicting which own units have an extra action queued
# (haste), and pricing the +50% damage / -50% damage-taken modifiers
# from Sorcerer buffs. Without these channels the action mask tells
# the policy *what it can do*, but the policy can't observe the
# *consequences* of an opponent's buff/debuff on units it already sees.
UNIT_CHANNELS = NUM_UNIT_TYPES + 2 + 1 + 1 + 4
GLOBAL_FEATURES_DIM = 5

# Status-effect channel indices into the per-cell unit feature vector.
# Kept as named constants so the slicing in build_observation and any
# downstream consumer (e.g. status-aware reward shaping) stays in sync
# if the channel layout is ever reshuffled.
UNIT_CH_PARALYZE = NUM_UNIT_TYPES + 4  # 12
UNIT_CH_HASTE = NUM_UNIT_TYPES + 5  # 13
UNIT_CH_DEFENCE_BUFF = NUM_UNIT_TYPES + 6  # 14
UNIT_CH_ATTACK_BUFF = NUM_UNIT_TYPES + 7  # 15

# Only 1v1 games are supported. The agent-relative encoding uses
# ``opp = 3 - perspective_player`` to identify the single opponent; any
# ``num_players != 2`` setting would silently misrepresent the state.
SUPPORTED_NUM_PLAYERS = 2

# Default ``tanh`` scale factors for ``global_features`` normalization.
# Each chosen so the typical mid-game value lands in the linear regime of
# ``tanh`` (input ~ 1.0) and tails saturate gracefully:
#   GOLD_SCALE        — early-mid game gold sits ~250-3000 (STARTING_GOLD=250
#                       plus a few turns of compounding income at 50-150 per
#                       owned structure); 1000 centers the linear regime.
#   TURN_SCALE        — tuned for the early-curriculum cap (starter=20,
#                       beginner=75, intermediate/skirmish=60-120,
#                       corner_points=200 in configs/ppo/bootstrap.yaml).
#                       At turn=60 the feature is ~tanh(1.0)=0.76 (linear
#                       regime); at turn=200 it saturates near 1.0 — still
#                       useful when ``max_turns`` is unbounded.
#   UNIT_COUNT_SCALE  — per-side army sizes realistically peak around 20-30
#                       on current maps; 20 lands the linear regime there.
# Override per-call via ``build_observation(..., gold_scale=..., ...)`` or
# per-env via the corresponding ``StrategyGameEnv``/``EnvConfig`` knobs.
GOLD_SCALE = 1000.0
TURN_SCALE = 60.0
UNIT_COUNT_SCALE = 20.0


def build_observation(
    game_state: Any,
    perspective_player: int,
    action_mask: Optional[np.ndarray] = None,
    fog_of_war: Optional[bool] = None,
    pad_to: Optional[Tuple[int, int]] = None,
    gold_scale: float = GOLD_SCALE,
    turn_scale: float = TURN_SCALE,
    unit_count_scale: float = UNIT_COUNT_SCALE,
) -> Dict[str, np.ndarray]:
    """Build an RL observation from ``perspective_player``'s viewpoint.

    Args:
        game_state: The GameState to observe.
        perspective_player: Player whose gold / unit count go first in
            ``global_features`` and whose units / structures populate the
            "self" owner channel. Under fog of war, only this player's
            visibility is applied to grid and units.
        action_mask: Optional precomputed legal-action mask. When provided,
            it is included verbatim under the ``"action_mask"`` key for
            back-compat with MCTS / AlphaZero callers. The PPO observation
            space does not include this key — gym_env passes ``None``.
        fog_of_war: Override for the FOW flag. When ``None``, falls back to
            ``game_state.fog_of_war``.
        gold_scale: Divisor applied before ``tanh`` to ``own_gold`` /
            ``opp_gold``. Defaults to :data:`GOLD_SCALE`.
        turn_scale: Divisor applied before ``tanh`` to ``turn_number``.
            Defaults to :data:`TURN_SCALE`.
        unit_count_scale: Divisor applied before ``tanh`` to
            ``own_units`` / ``opp_units``. Defaults to
            :data:`UNIT_COUNT_SCALE`.
        pad_to: Optional ``(pad_h, pad_w)`` target shape for the spatial
            tensors. When set, ``grid`` / ``units`` (and ``visibility`` if
            present) are zero-padded so the real map sits at the top-left
            and rows ``[h:pad_h]`` / cols ``[w:pad_w]`` are pad cells.
            Padded cells have all-zero tile-type one-hot, all-zero owner
            channels, and zero HP — a unique signature distinct from any
            real cell, so a downstream MLP can learn to ignore them.
            Each of ``pad_h`` / ``pad_w`` must be ``>=`` the live map's
            corresponding dim. Used by the curriculum runner to keep
            ``observation_space`` shape constant across stages with
            different map sizes.

    Returns:
        Observation dict with keys ``grid``, ``units``, ``global_features``,
        optionally ``action_mask`` and (when FOW is enabled) ``visibility``.
    """
    if fog_of_war is None:
        fog_of_war = bool(getattr(game_state, "fog_of_war", False))

    # 1v1 enforcement. The encoding (self/opp owner channels, single
    # opp_gold scalar, ``opp = 3 - perspective_player``) is hard-coded to
    # two players. Multi-player (FFA / team) games need a permutation-
    # invariant rewrite (self/ally/enemy channels, aggregated globals)
    # which is intentionally out of scope right now — fail loudly instead
    # of silently producing a malformed observation.
    num_players = int(getattr(game_state, "num_players", SUPPORTED_NUM_PLAYERS))
    if num_players != SUPPORTED_NUM_PLAYERS:
        raise ValueError(
            f"build_observation only supports 1v1 games (num_players=2); got num_players={num_players}. "
            "Multi-player observations require a separate team-relative encoding (own/ally/enemy)."
        )
    if perspective_player not in (1, 2):
        raise ValueError(f"perspective_player must be 1 or 2 in a 1v1 game; got {perspective_player}.")

    opp = 3 - perspective_player

    if fog_of_war:
        state_arrays = game_state.to_numpy(for_player=perspective_player)
    else:
        state_arrays = game_state.to_numpy()

    own_gold = game_state.player_gold.get(perspective_player, 0)
    own_units = sum(1 for u in game_state.units if u.player == perspective_player)

    if fog_of_war:
        # Enemy gold is hidden; only visible enemy units are counted.
        opp_gold: float = 0
        opp_units = sum(
            1 for u in game_state.units if u.player == opp and game_state.is_position_visible(u.x, u.y, perspective_player)
        )
    else:
        opp_gold = game_state.player_gold.get(opp, 0)
        opp_units = sum(1 for u in game_state.units if u.player == opp)

    # tanh-squash to keep features comparable in magnitude. All five raw
    # inputs are non-negative, so the output sits in [0, 1). Under fog of
    # war, ``opp_gold`` is forced to 0 above, which maps to tanh(0)=0 —
    # preserving the "hidden info" sentinel.
    global_features = np.tanh(
        np.array(
            [
                own_gold / gold_scale,
                opp_gold / gold_scale,
                game_state.turn_number / turn_scale,
                own_units / unit_count_scale,
                opp_units / unit_count_scale,
            ],
            dtype=np.float32,
        )
    )

    raw_grid = state_arrays["grid"]
    raw_units = state_arrays["units"]
    h, w = raw_grid.shape[:2]

    # ---- Grid one-hot --------------------------------------------------
    # Raw layout (from TileGrid.to_numpy):
    #   [..., 0] = tile_type int in [0, NUM_TILE_TYPES)
    #   [..., 1] = absolute owner (0 = neutral, otherwise player number)
    #   [..., 2] = structure HP percentage in [0, 100]
    grid = np.zeros((h, w, GRID_CHANNELS), dtype=np.float32)
    tile_type_idx = raw_grid[..., 0].astype(np.int64)
    np.clip(tile_type_idx, 0, NUM_TILE_TYPES - 1, out=tile_type_idx)
    yy, xx = np.indices((h, w))
    grid[yy, xx, tile_type_idx] = 1.0
    raw_owner = raw_grid[..., 1]
    grid[..., NUM_TILE_TYPES + 0] = (raw_owner == perspective_player).astype(np.float32)
    grid[..., NUM_TILE_TYPES + 1] = (raw_owner == opp).astype(np.float32)
    # Neutral ownership is implicit: (own=0, opp=0). The explicit neutral
    # channel that used to live at NUM_TILE_TYPES + 2 was dropped because
    # it's exactly ``1 - own - opp`` — a linear combination of the other
    # two owner channels and pure dead weight in every conv kernel.
    grid[..., NUM_TILE_TYPES + 2] = raw_grid[..., 2].astype(np.float32) / 100.0

    # ---- Units one-hot -------------------------------------------------
    # Raw layout (from GameState.to_numpy):
    #   [..., 0] = unit_type int (0 = empty, 1..8 = ALL_UNIT_TYPES)
    #   [..., 1] = absolute owner (0 = empty cell, else player number)
    #   [..., 2] = unit HP percentage in [0, 100]
    #   [..., 3] = exhausted flag in {0.0, 1.0} (not (can_move or can_attack))
    #   [..., 4] = paralyzed_turns (raw int, 0..PARALYZE_DURATION)
    #   [..., 5] = is_hasted (0.0 / 1.0)
    #   [..., 6] = defence_buff_turns (raw int, 0..SORCERER_BUFF_DURATION)
    #   [..., 7] = attack_buff_turns (raw int, 0..SORCERER_BUFF_DURATION)
    units = np.zeros((h, w, UNIT_CHANNELS), dtype=np.float32)
    unit_type_idx = raw_units[..., 0].astype(np.int64)
    has_unit = unit_type_idx > 0
    # Subtract 1 so the type index lines up with ALL_UNIT_TYPES; clipped
    # below for safety against any out-of-range values.
    one_hot_idx = np.clip(unit_type_idx - 1, 0, NUM_UNIT_TYPES - 1)
    units[yy[has_unit], xx[has_unit], one_hot_idx[has_unit]] = 1.0
    raw_unit_owner = raw_units[..., 1]
    own_mask = raw_unit_owner == perspective_player
    units[..., NUM_UNIT_TYPES + 0] = own_mask.astype(np.float32)
    units[..., NUM_UNIT_TYPES + 1] = (raw_unit_owner == opp).astype(np.float32)
    # own_exhausted: gated by own_mask because (a) opponent move/attack
    # flags are stale on our turn (they're reset at *their* turn start) and
    # (b) the policy only cares which of *its own* units are spent.
    if raw_units.shape[-1] > 3:
        units[..., NUM_UNIT_TYPES + 2] = (raw_units[..., 3] * own_mask).astype(np.float32)
    units[..., NUM_UNIT_TYPES + 3] = raw_units[..., 2].astype(np.float32) / 100.0
    # Status-effect channels. Guarded against the older 4-channel
    # raw_units layout (pre-status-effect-channels schema) so a stale
    # GameState binary doesn't crash observation construction -- the
    # extra channels stay zero, matching "no status active". Visibility
    # filtering is handled by GameState.to_numpy (hidden enemy cells
    # have zero across every raw_units slot), so the status channels
    # are correctly zeroed under fog of war.
    if raw_units.shape[-1] > 4:
        units[..., UNIT_CH_PARALYZE] = raw_units[..., 4].astype(np.float32) / float(PARALYZE_DURATION)
    if raw_units.shape[-1] > 5:
        units[..., UNIT_CH_HASTE] = raw_units[..., 5].astype(np.float32)
    if raw_units.shape[-1] > 6:
        units[..., UNIT_CH_DEFENCE_BUFF] = raw_units[..., 6].astype(np.float32) / float(SORCERER_BUFF_DURATION)
    if raw_units.shape[-1] > 7:
        units[..., UNIT_CH_ATTACK_BUFF] = raw_units[..., 7].astype(np.float32) / float(SORCERER_BUFF_DURATION)

    if pad_to is not None:
        pad_h, pad_w = int(pad_to[0]), int(pad_to[1])
        if pad_h < h or pad_w < w:
            raise ValueError(
                f"pad_to={(pad_h, pad_w)} is smaller than the live map's ({h}, {w}). pad_to must be >= the live map's dims."
            )
        if pad_h != h or pad_w != w:
            padded_grid = np.zeros((pad_h, pad_w, GRID_CHANNELS), dtype=np.float32)
            padded_grid[:h, :w, :] = grid
            grid = padded_grid

            padded_units = np.zeros((pad_h, pad_w, UNIT_CHANNELS), dtype=np.float32)
            padded_units[:h, :w, :] = units
            units = padded_units

    obs: Dict[str, np.ndarray] = {
        "grid": grid,
        "units": units,
        "global_features": global_features,
    }

    if action_mask is not None:
        obs["action_mask"] = action_mask

    if fog_of_war and "visibility" in state_arrays:
        visibility = state_arrays["visibility"]
        if pad_to is not None:
            pad_h, pad_w = int(pad_to[0]), int(pad_to[1])
            if pad_h != h or pad_w != w:
                padded_vis = np.zeros((pad_h, pad_w), dtype=visibility.dtype)
                padded_vis[:h, :w] = visibility
                visibility = padded_vis
        obs["visibility"] = visibility

    return obs
