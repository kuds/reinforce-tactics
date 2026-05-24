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

    Survives one specific kind of state divergence: pre-fix replays
    where the action log contains "ghost actions" by units the
    engine had already removed via counter-attack. When the recorded
    kill can't find the named unit at the named position, the
    elimination check still fires for the player who *would have
    been* killed, so the replay can correctly terminate on the
    eliminating action even if the unit itself is missing.
    """
    attacker_pos = translate_fn(*action["attacker_pos"])
    target_pos = translate_fn(*action["target_pos"])
    attacker = game_state.get_unit_at_position(*attacker_pos)
    target = game_state.get_unit_at_position(*target_pos)

    attacker_killed = bool(action.get("attacker_killed", False))
    target_killed = bool(action.get("target_killed", False))
    attacker_player = action.get("player")
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


def apply_recorded_seize(game_state, action: Dict[str, Any], translate_fn: Callable) -> None:
    """Apply a v2 ``seize`` action by mutating tile state directly.

    Mirrors the post-mechanics work in ``GameState.seize``: set tile
    health and owner from the record, clear ``regenerating`` (which
    ``mechanics.seize_structure`` always does on a successful call),
    fire ``hq_capture`` end-game on a captured HQ, and lock out the
    seizer for the rest of the turn.

    Tile state is applied unconditionally -- the seize is fully
    described by its tile-after fields, so a missing seizer (which
    happens in pre-fix replays where the bot held a stale reference
    to a counter-killed unit) doesn't block the tile mutation. The
    seizer-side lockout is best-effort.
    """
    position = translate_fn(*action["position"])
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

    # Lock out the seizer for the rest of the turn -- best-effort.
    unit = game_state.get_unit_at_position(*position)
    if unit is not None:
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
