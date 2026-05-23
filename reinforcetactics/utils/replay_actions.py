"""Apply recorded action outcomes directly to a GameState during replay.

Replay schema v2 (see ``replay_schema_version`` in saved game_info)
carries enough information per action -- HP after, killed flags,
counter damage, etc. -- that the replay player can mutate state
directly instead of re-calling the engine. That sidesteps two
divergence sources:

  * RNG inside ``mechanics.attack_unit`` (Rogue evade roll)
  * Cascading state drift once any single action's recomputation
    disagrees with the recorded outcome

These helpers are shared by :mod:`reinforcetactics.utils.replay_player`
and :mod:`reinforcetactics.utils.video` so the two playback paths
stay in sync.
"""

from typing import Any, Callable, Dict


def get_schema_version(game_info: Dict[str, Any]) -> int:
    """Return the replay schema version, defaulting to 1 for older replays."""
    return int(game_info.get("replay_schema_version", 1))


def apply_recorded_attack(game_state, action: Dict[str, Any], translate_fn: Callable) -> None:
    """Apply a v2 ``attack`` action by setting recorded outcomes directly.

    Mirrors the post-mechanics work in ``GameState.attack`` (HP
    assignment, unit removal, tile-regeneration flag, elimination
    check, attacker action lockout) but never calls
    ``mechanics.attack_unit`` -- so no RNG is re-rolled and no
    damage is recomputed against potentially-diverged state.
    """
    attacker_pos = translate_fn(*action["attacker_pos"])
    target_pos = translate_fn(*action["target_pos"])
    attacker = game_state.get_unit_at_position(*attacker_pos)
    target = game_state.get_unit_at_position(*target_pos)
    if attacker is None or target is None:
        # State diverged before this action -- nothing useful to do.
        # Phase 3's determinism test catches this case.
        return

    attacker_killed = bool(action.get("attacker_killed", False))
    target_killed = bool(action.get("target_killed", False))

    # Apply HP-after for survivors. Dead units get removed outright;
    # the live engine doesn't bother zeroing HP before removal.
    if not target_killed and "target_hp_after" in action:
        target.health = action["target_hp_after"]
    if not attacker_killed and "attacker_hp_after" in action:
        attacker.health = action["attacker_hp_after"]

    # Death side-effects. Order (target first, then attacker) and
    # tile-regen behaviour mirror GameState.attack so capturable
    # tiles vacated by a dying defender start regenerating exactly
    # as they did in the original game.
    if target_killed:
        target_tile = game_state.grid.get_tile(target.x, target.y)
        if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
            target_tile.regenerating = True
        defeated_player = target.player
        if target in game_state.units:
            game_state.units.remove(target)
        game_state._invalidate_cache()
        game_state._check_player_eliminated(defeated_player)

    if attacker_killed:
        attacker_tile = game_state.grid.get_tile(attacker.x, attacker.y)
        if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
            attacker_tile.regenerating = True
        defeated_player = attacker.player
        if attacker in game_state.units:
            game_state.units.remove(attacker)
        game_state._invalidate_cache()
        game_state._check_player_eliminated(defeated_player)

    # Lock out attacker for the rest of the turn (only if still alive).
    if not attacker_killed:
        attacker.can_move = False
        attacker.can_attack = False
    game_state._invalidate_cache()


def apply_recorded_seize(game_state, action: Dict[str, Any], translate_fn: Callable) -> None:
    """Apply a v2 ``seize`` action by mutating tile state directly.

    Mirrors the post-mechanics work in ``GameState.seize``: set tile
    health and owner from the record, clear ``regenerating`` (which
    ``mechanics.seize_structure`` always does on a successful call),
    fire ``hq_capture`` end-game on a captured HQ, and lock out the
    seizer for the rest of the turn.
    """
    position = translate_fn(*action["position"])
    unit = game_state.get_unit_at_position(*position)
    if unit is None:
        return

    tile = game_state.grid.get_tile(*position)

    # Record-driven tile state. Fall back to the live tile values if
    # the record was written by a pre-v2 engine (the dispatcher only
    # routes us here when schema >= 2, but be defensive).
    if "tile_hp_after" in action:
        tile.health = action["tile_hp_after"]
    if "tile_owner_after" in action:
        tile.player = action["tile_owner_after"]
    tile.regenerating = False

    if action.get("captured") and action.get("structure_type") == "h":
        game_state._set_game_over(winner=unit.player, end_reason="hq_capture")

    unit.can_move = False
    unit.can_attack = False
    game_state._invalidate_cache()


def apply_recorded_heal(game_state, action: Dict[str, Any], translate_fn: Callable) -> None:
    """Apply a v2 ``heal`` action by setting target HP directly.

    Heal has no RNG today, but applying ``target_hp_after`` rather
    than re-calling ``mechanics.heal_unit`` makes the replay
    immune to future HEAL_AMOUNT drift between save and playback.
    """
    healer_pos = translate_fn(*action["healer_pos"])
    target_pos = translate_fn(*action["target_pos"])
    healer = game_state.get_unit_at_position(*healer_pos)
    target = game_state.get_unit_at_position(*target_pos)
    if healer is None or target is None:
        return

    if "target_hp_after" in action:
        target.health = action["target_hp_after"]

    healer.can_move = False
    healer.can_attack = False
    game_state._invalidate_cache()
