"""Apply recorded action outcomes directly to a GameState during replay.

Replay schema v2 (see ``replay_schema_version`` in saved game_info)
carries enough information per action -- HP after, killed flags,
counter damage, etc. -- that the replay player can mutate state
directly instead of re-calling the engine. That sidesteps two
divergence sources:

  * RNG inside ``mechanics.attack_unit`` (Rogue evade roll)
  * Cascading state drift once any single action's recomputation
    disagrees with the recorded outcome

Schema v3 keeps the v2 outcome fields and additionally tags every
action with the stable ``actor_unit_id`` / ``target_unit_id`` of
the units involved. The v3 helpers look up units by id so position
drift (e.g. a buggy bot iteration that calls ``move_unit`` on a
unit whose ``unit.x, unit.y`` are stale) can't silently mis-route
the action in the replay. When the recorded id isn't in
``self.units`` (real divergence), the v3 helpers emit a warning
and fall back to position-based recovery so the cascade stops
loudly instead of quietly.

These helpers are shared by :mod:`reinforcetactics.utils.replay_player`
and :mod:`reinforcetactics.utils.video` so the two playback paths
stay in sync.
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def get_schema_version(game_info: dict[str, Any]) -> int:
    """Return the replay schema version, defaulting to 1 for older replays."""
    return int(game_info.get("replay_schema_version", 1))


def find_unit_by_id(game_state, unit_id: int | None):
    """Locate a unit in ``game_state.units`` by its stable id.

    Returns ``None`` for missing or null ids -- the v3 dispatchers
    log a warning in that case so any future bot bug that produces
    a stale unit reference (the original "Knight ghost" symptom)
    surfaces in tests instead of cascading silently.
    """
    if unit_id is None:
        return None
    for unit in game_state.units:
        if unit.unit_id == unit_id:
            return unit
    return None


# ---------------------------------------------------------------------------
# Inner helpers (take Unit objects, not action dicts) shared by v2 + v3.
# Position lookup vs id lookup is the only thing that differs between the
# two schema generations; everything below is identical.
# ---------------------------------------------------------------------------


def _apply_attack_outcome(
    game_state,
    action: dict[str, Any],
    attacker,
    target,
    attacker_player: int | None,
) -> None:
    """Apply the recorded attack outcome to (optionally-found) units."""
    attacker_killed = bool(action.get("attacker_killed", False))
    target_killed = bool(action.get("target_killed", False))
    # Action records ``player`` for the attacker only; the target is
    # the other side in a 2-player game.
    target_player = 2 if attacker_player == 1 else 1

    # Apply HP-after for survivors. Dead units get removed outright;
    # the live engine doesn't bother zeroing HP before removal.
    if target is not None and not target_killed and "target_hp_after" in action:
        target.health = action["target_hp_after"]
    if attacker is not None and not attacker_killed and "attacker_hp_after" in action:
        attacker.health = action["attacker_hp_after"]

    # Death side-effects. Order (target first, then attacker) and
    # tile-regen behaviour mirror GameState.attack so capturable
    # tiles vacated by a dying defender start regenerating exactly
    # as they did in the original game.
    if target_killed:
        if target is not None:
            target_tile = game_state.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            if target in game_state.units:
                game_state.units.remove(target)
            game_state._invalidate_cache()
        # Fire the elimination check even if the target is missing --
        # in pre-fix replays this is the only way game_over can land
        # on the recorded winning_action_index.
        game_state._check_player_eliminated(target_player)

    if attacker_killed:
        if attacker is not None:
            attacker_tile = game_state.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            if attacker in game_state.units:
                game_state.units.remove(attacker)
            game_state._invalidate_cache()
        if attacker_player is not None:
            game_state._check_player_eliminated(attacker_player)

    # Lock out attacker for the rest of the turn (only if still alive
    # and present).
    if attacker is not None and not attacker_killed:
        attacker.can_move = False
        attacker.can_attack = False
    game_state._invalidate_cache()


def _apply_seize_outcome(game_state, action: dict[str, Any], position, unit) -> None:
    """Apply the recorded seize outcome at ``position`` with (optional) ``unit``."""
    tile = game_state.grid.get_tile(*position)

    # Record-driven tile state -- applied even if the seizer is gone.
    if "tile_hp_after" in action:
        tile.health = action["tile_hp_after"]
    if "tile_owner_after" in action:
        tile.player = action["tile_owner_after"]
    tile.regenerating = False

    # HQ capture end-game can be derived from the recorded captured
    # flag + action player; doesn't need the seizer unit.
    if action.get("captured") and action.get("structure_type") == "h":
        game_state._set_game_over(winner=action.get("player"), end_reason="hq_capture")

    if unit is not None:
        unit.can_move = False
        unit.can_attack = False
    game_state._invalidate_cache()


def _apply_heal_outcome(game_state, action: dict[str, Any], healer, target) -> None:
    if healer is None or target is None:
        return
    if "target_hp_after" in action:
        target.health = action["target_hp_after"]
    healer.can_move = False
    healer.can_attack = False
    game_state._invalidate_cache()


def _apply_move_outcome(game_state, action: dict[str, Any], unit, to_x: int, to_y: int) -> None:
    """Apply a recorded move directly to ``unit``.

    Unlike v1/v2 (which call ``game_state.move_unit``), the v3 path
    sets ``unit.x``/``unit.y`` from the recorded destination so the
    engine's ``can_move`` / reachable / fog-of-war / occupancy gates
    can't reject a legitimately-recorded second move (e.g. after a
    Sorcerer Haste resets ``can_move``). ``distance_moved`` is
    computed from the recorded ``from`` -> ``to`` to keep Knight
    Charge bookkeeping faithful to the original.
    """
    if unit is None:
        return
    from_x = action.get("from_x")
    from_y = action.get("from_y")
    if from_x is not None and from_y is not None:
        # Translate the recorded from-coords through the caller's
        # padding so we measure distance in the same grid the move
        # was originally executed on.
        delta_x = abs(to_x - unit.x) if unit.x != from_x else abs(to_x - from_x)
        delta_y = abs(to_y - unit.y) if unit.y != from_y else abs(to_y - from_y)
        distance = delta_x + delta_y
    else:
        distance = abs(to_x - unit.x) + abs(to_y - unit.y)
    unit.x = to_x
    unit.y = to_y
    unit.distance_moved += distance
    unit.has_moved = True
    unit.can_move = False
    game_state._invalidate_cache()


# ---------------------------------------------------------------------------
# v2 helpers (position-based lookup). Kept for backward compatibility with
# replays written before the unit_id schema bump.
# ---------------------------------------------------------------------------


def apply_recorded_attack(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v2: locate attacker/target by recorded position, then apply outcome."""
    attacker_pos = translate_fn(*action["attacker_pos"])
    target_pos = translate_fn(*action["target_pos"])
    attacker = game_state.get_unit_at_position(*attacker_pos)
    target = game_state.get_unit_at_position(*target_pos)
    _apply_attack_outcome(game_state, action, attacker, target, action.get("player"))


def apply_recorded_seize(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v2: tile state is positional regardless of schema."""
    position = translate_fn(*action["position"])
    unit = game_state.get_unit_at_position(*position)
    _apply_seize_outcome(game_state, action, position, unit)


def apply_recorded_heal(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v2: locate healer/target by recorded position, then apply outcome."""
    healer_pos = translate_fn(*action["healer_pos"])
    target_pos = translate_fn(*action["target_pos"])
    healer = game_state.get_unit_at_position(*healer_pos)
    target = game_state.get_unit_at_position(*target_pos)
    _apply_heal_outcome(game_state, action, healer, target)


# ---------------------------------------------------------------------------
# v3 helpers (id-based lookup). Used when ``replay_schema_version >= 3`` AND
# the action carries the relevant ``*_unit_id`` field. Missing-id lookups log
# a warning and fall through to v2 position-based recovery -- this is the
# "loud failure" the schema bump is meant to provide.
# ---------------------------------------------------------------------------


def _resolve_or_warn(game_state, unit_id: int | None, position, label: str):
    """Find unit by id; warn and fall back to position lookup on miss."""
    unit = find_unit_by_id(game_state, unit_id)
    if unit is not None:
        return unit
    if unit_id is not None:
        logger.warning(
            "v3 replay: %s unit_id=%s not in game_state.units; "
            "falling back to position %s. This indicates state divergence "
            "from the recorded game -- check for bot bugs that pass stale "
            "unit references to engine APIs.",
            label,
            unit_id,
            position,
        )
    if position is not None:
        return game_state.get_unit_at_position(*position)
    return None


def apply_recorded_attack_v3(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v3: locate attacker/target by ``attacker_unit_id``/``target_unit_id``."""
    attacker_pos = translate_fn(*action["attacker_pos"])
    target_pos = translate_fn(*action["target_pos"])
    attacker = _resolve_or_warn(game_state, action.get("attacker_unit_id"), attacker_pos, "attack attacker")
    target = _resolve_or_warn(game_state, action.get("target_unit_id"), target_pos, "attack target")
    _apply_attack_outcome(game_state, action, attacker, target, action.get("player"))


def apply_recorded_seize_v3(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v3: tile state is positional but the seizer lockout is id-based."""
    position = translate_fn(*action["position"])
    unit = _resolve_or_warn(game_state, action.get("actor_unit_id"), position, "seize actor")
    _apply_seize_outcome(game_state, action, position, unit)


def apply_recorded_heal_v3(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v3: locate healer/target by id."""
    healer_pos = translate_fn(*action["healer_pos"])
    target_pos = translate_fn(*action["target_pos"])
    healer = _resolve_or_warn(game_state, action.get("actor_unit_id"), healer_pos, "heal healer")
    target = _resolve_or_warn(game_state, action.get("target_unit_id"), target_pos, "heal target")
    _apply_heal_outcome(game_state, action, healer, target)


def apply_recorded_move_v3(game_state, action: dict[str, Any], translate_fn: Callable) -> None:
    """v3: locate mover by id, set position directly.

    Sidesteps the engine's ``can_move`` / reachable / occupancy
    checks that legitimately rejected recorded second-moves under
    Sorcerer Haste in earlier schemas. The recorded ``to_x, to_y``
    is the authoritative destination -- no validation, just apply.
    """
    from_x, from_y = translate_fn(action["from_x"], action["from_y"])
    to_x, to_y = translate_fn(action["to_x"], action["to_y"])
    unit = _resolve_or_warn(game_state, action.get("actor_unit_id"), (from_x, from_y), "move actor")
    _apply_move_outcome(game_state, action, unit, to_x, to_y)
