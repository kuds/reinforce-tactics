"""What a player may do: the legality rules and the legal-action enumeration (review core-16).

Each rule is written once and used twice: ``enumerate_legal_actions`` lists
what a player may do with it (``GameState.get_legal_actions``, which the RL
action masks, the bots, MCTS and the LLM prompts read), and ``GameState``'s
action methods reject anything else with it, through the validators
``is_legal`` asks too (``_may_create``, ``_move_steps``, ``_may_target``,
``_may_seize``). The action methods used to trust their callers, so any
caller that did not pre-filter against the legal list (multi_discrete
policies, LLM bots, the rule bots' knight charge, the GUI) could spawn units
anywhere, attack across the map, act out of turn or seize an HQ several
times in one turn (review core-2). One definition per rule is what keeps the
mask and the engine agreeing: an offered action the engine rejects traps a
deterministic policy, and an accepted action the mask never offers is an
exploit.

Two gates apply only on execution, not in enumeration, so they live in the
validators: the game must not be over, and it must be the acting player's
turn. ``enumerate_legal_actions(state, player)`` answers for any player at
any time (masks and prompts are built for a player's own turn in practice).

Moved out of ``game_state.py`` so the rules can be read, tested and changed
without the rest of the engine around them. ``GameState`` keeps
``get_legal_actions`` (which caches this enumeration), ``get_move_destinations``
and ``get_reachable_positions``, which call these.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from operator import methodcaller
from typing import TYPE_CHECKING, Any

from reinforcetactics.core.actions import ACTION_KINDS, ACTOR_KEYS
from reinforcetactics.rules import TileType

if TYPE_CHECKING:
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.core.unit import Unit

# ----------------------------------------------------------------------
# Actors and creation
# ----------------------------------------------------------------------


def is_ready_unit(unit: Unit, player: int) -> bool:
    """``unit`` belongs to ``player``, is alive and is not paralyzed."""
    return unit.player == player and unit.health > 0 and not unit.is_paralyzed()


def under_unit_cap(state: GameState, player: int) -> bool:
    """``player`` has fewer units than the game's cap (``max_units_per_player``)."""
    return sum(1 for u in state.units if u.player == player) < state.max_units_per_player


def is_free_spawn_tile(state: GameState, player: int, x: int, y: int) -> bool:
    """An in-bounds, empty Building owned by ``player`` (HQs and towers never spawn)."""
    tile = state.grid.get_tile(x, y)
    return (
        tile is not None
        and tile.type == TileType.BUILDING.value
        and tile.player == player
        and state.get_unit_at_position(x, y) is None
    )


def can_afford(state: GameState, player: int, unit_type: str) -> bool:
    """``player`` has the gold for a ``unit_type`` at this game's price."""
    return state.player_gold[player] >= state.unit_data[unit_type]["cost"]


# ----------------------------------------------------------------------
# Movement
# ----------------------------------------------------------------------


def find_paths(
    state: GameState,
    unit: Unit,
    blocked: set[tuple[int, int]] | None = None,
    came_from: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    """Every tile ``unit`` can reach this turn -> tiles stepped, in search order.

    Walkable tiles within its movement (under the game's terrain move
    costs), passing through friendly units but never enemies; tiles
    holding a friendly unit are included, as a path may cross them. The
    blockers are collected once per search (or once per enumeration, for
    all of a player's units: pass ``blocked``), so each tile the search
    examines costs a set lookup rather than a scan of every unit (review
    core-20). They come from the units the player knows of
    (``FogOfWar.pathing_units``: all of them without fog of war), so a
    hidden enemy neither blocks a path nor reveals itself through the move
    mask (review core-5). ``came_from`` receives the search's path tree
    (the ambush rule walks it).
    """
    if blocked is None:
        blocked = state.mechanics.movement_blockers(state.fog.pathing_units(unit.player), unit, state.teams)
    return unit.find_paths(
        state.grid.width,
        state.grid.height,
        state.mechanics.passability(state.grid, blocked),
        state.terrain_rules.move_cost_fn(state.grid),
        came_from,
    )


def move_paths(
    state: GameState,
    unit: Unit,
    occupied: set[tuple[int, int]] | None = None,
    blocked: set[tuple[int, int]] | None = None,
    came_from: dict[tuple[int, int], tuple[int, int]] | None = None,
) -> dict[tuple[int, int], int]:
    """``find_paths`` restricted to the tiles ``unit`` may end a move on (no known unit there).

    The move rule's destinations: what the enumeration offers and what
    ``GameState._move_steps`` accepts.
    """
    if occupied is None:
        occupied = {(u.x, u.y) for u in state.fog.pathing_units(unit.player)}
    return {pos: steps for pos, steps in find_paths(state, unit, blocked, came_from).items() if pos not in occupied}


# ----------------------------------------------------------------------
# Targets and seizing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TargetRule:
    """The rule of one targeted action: what it asks of the actor alone, and of each target.

    Calling it asks both (``rule(state, unit, target)``), which is what the
    validators do. The enumeration asks ``actor`` (the unit's type and
    cooldown) once per unit, and ``target`` only for a unit that passes:
    asking every rule of every unit for every target made
    ``get_legal_actions`` a third slower.
    """

    actor: Callable[[Unit], bool]
    target: Callable[[GameState, Unit, Unit], bool]

    def __call__(self, state: GameState, unit: Unit, target: Unit) -> bool:
        return self.actor(unit) and self.target(state, unit, target)


def _any_unit(unit: Unit) -> bool:
    """Any unit may attack (being in play, ready and on its turn are the callers' gates)."""
    return True


def _is_cleric(unit: Unit) -> bool:
    return unit.type == "C"


def _attackable(state: GameState, unit: Unit, target: Unit) -> bool:
    """A living enemy within ``unit``'s reach that fog of war lets it attack.

    Under fog of war the target must have been visible when the unit
    started its action (``FogOfWar.is_enemy_attackable``), so moving to
    discover an enemy does not also let the unit hit it.
    """
    return (
        state.are_enemies(target.player, unit.player)
        and target.health > 0
        and state.mechanics.can_reach(unit, target.x, target.y, state.grid)
        and state.fog.is_enemy_attackable(unit, target)
    )


def _paralyzable(state: GameState, unit: Unit, target: Unit) -> bool:
    """An enemy ``unit`` may attack, in paralyze range and not already paralyzed.

    Re-casting on a paralyzed target would only refresh the status (a
    near no-op) and inflate the action space, the same reason heal,
    cure and the buffs skip an ally that already has the effect.
    """
    return (
        not target.is_paralyzed()
        and state.mechanics.in_ability_range("paralyze", unit, target)
        and _attackable(state, unit, target)
    )


def _healable(state: GameState, unit: Unit, target: Unit) -> bool:
    return state.mechanics.is_healable_ally(unit, target, state.teams)


def _curable(state: GameState, unit: Unit, target: Unit) -> bool:
    return state.mechanics.is_curable_ally(unit, target, state.teams)


def _hasteable(state: GameState, unit: Unit, target: Unit) -> bool:
    return state.mechanics.is_hasteable_ally(unit, target)


def _defence_buffable(state: GameState, unit: Unit, target: Unit) -> bool:
    return state.mechanics.is_defence_buffable_ally(unit, target, state.teams)


def _attack_buffable(state: GameState, unit: Unit, target: Unit) -> bool:
    return state.mechanics.is_attack_buffable_ally(unit, target, state.teams)


# The rule of each targeted action (``attack`` and the abilities), by its
# ``get_legal_actions`` key: what the enumeration offers targets by and
# ``GameState._may_target`` validates with. The ally rules' target parts are
# GameMechanics' ``is_*_ally`` predicates.
TARGET_RULES: dict[str, TargetRule] = {
    # Any unit: a living enemy in reach it could see when its action began.
    "attack": TargetRule(_any_unit, _attackable),
    # A Mage off cooldown: an enemy it may attack, in range, not paralyzed.
    "paralyze": TargetRule(methodcaller("can_use_paralyze"), _paralyzable),
    # A Cleric: a damaged (heal) or paralyzed (cure) ally in range.
    "heal": TargetRule(_is_cleric, _healable),
    "cure": TargetRule(_is_cleric, _curable),
    # A Sorcerer, each ability off its own cooldown: one of its own units
    # for haste, an ally without the buff for the buffs.
    "haste": TargetRule(methodcaller("can_use_haste"), _hasteable),
    "defence_buff": TargetRule(methodcaller("can_use_defence_buff"), _defence_buffable),
    "attack_buff": TargetRule(methodcaller("can_use_attack_buff"), _attack_buffable),
}


def can_seize(state: GameState, unit: Unit) -> bool:
    """``unit`` stands on a structure neither its player nor a teammate owns."""
    tile = state.grid.get_tile(unit.x, unit.y)
    return tile is not None and tile.is_capturable() and not state.are_allies(tile.player, unit.player)


# The abilities that target allies, in ``ACTION_KINDS`` order, each with its
# rule's two parts and the payload key of its actor (unpacked once, for the
# hot loop). Attack and paralyze are enumerated together (see below).
_ALLY_TARGETED = tuple(
    (kind, TARGET_RULES[kind].actor, TARGET_RULES[kind].target, ACTOR_KEYS[kind])
    for kind in ACTION_KINDS
    if kind in TARGET_RULES and kind not in ("attack", "paralyze")
)


# ----------------------------------------------------------------------
# Enumeration
# ----------------------------------------------------------------------


def enumerate_create_actions(state: GameState, player: int) -> list[dict[str, Any]]:
    """The ``create_unit`` list of ``enumerate_legal_actions``, the same entries in the same order, alone.

    For callers that only buy, such as the scripted bots' purchase loops:
    the full enumeration also searches every unit's moves, which made
    listing the purchases cost as much as listing every action.
    """
    creates: list[dict[str, Any]] = []
    # Building units (only at Buildings, not HQ)
    # Only include enabled unit types. Suppressed entirely once the player
    # is at the unit cap so the action mask matches create_unit's own
    # enforcement (no offered-then-rejected create actions).
    if under_unit_cap(state, player):
        for tile in state.grid.get_capturable_tiles(player):
            if is_free_spawn_tile(state, player, tile.x, tile.y):
                for unit_type in state.enabled_units:
                    if can_afford(state, player, unit_type):
                        creates.append({"unit_type": unit_type, "x": tile.x, "y": tile.y})
    return creates


def enumerate_legal_actions(state: GameState, player: int) -> dict[str, Any]:
    """Every action ``player`` may take in ``state`` now, uncached (``GameState.get_legal_actions`` caches it).

    Returns:
        ``{kind: [payload, ...]}`` for every kind in ``ACTION_KINDS`` order,
        and ``"end_turn": True``. Payloads: ``{"unit_type", "x", "y"}`` for
        ``create_unit``; ``{"unit", "from_x", "from_y", "to_x", "to_y"}``
        for ``move``; ``{"unit", "tile"}`` for ``seize``; the actor (under
        ``ACTOR_KEYS[kind]``) and ``"target"`` for the targeted kinds.
        Lists are ordered by unit (``state.units`` order), then by
        destination (search order) or target (``state.units`` order).
        That order is part of the flat_discrete action encoding, and bots
        pick from these lists with seeded RNGs, so it must not change.
    """
    legal_actions: dict[str, Any] = {kind: [] for kind in ACTION_KINDS}
    legal_actions["end_turn"] = True
    legal_actions["create_unit"] = enumerate_create_actions(state, player)

    # Unit actions. Every move search shares one occupancy set, and one
    # blocker set: who blocks a unit depends only on its player. Both come
    # from the units this player knows of (see ``FogOfWar.pathing_units``).
    units = state.units
    attack, paralyze = TARGET_RULES["attack"], TARGET_RULES["paralyze"]
    attacks, paralyzes = legal_actions["attack"], legal_actions["paralyze"]
    known_units = state.fog.pathing_units(player)
    occupied = {(u.x, u.y) for u in known_units}
    blocked: set[tuple[int, int]] | None = None
    for unit in units:
        # Guard on health: dead units are normally removed synchronously
        # by ``attack`` (see state.units.remove), but the rules above all
        # filter on ``health > 0`` defensively -- mirror that here so a
        # corpse left in ``state.units`` by any future deferred-removal path
        # (AoE, end-of-turn DoT, status damage) can't emit phantom actions.
        if not is_ready_unit(unit, player):
            continue

        # Movement: reachable tiles that are also free to end on
        if unit.can_move:
            if blocked is None:
                blocked = state.mechanics.movement_blockers(known_units, unit, state.teams)
            for pos in move_paths(state, unit, occupied, blocked):
                legal_actions["move"].append(
                    {"unit": unit, "from_x": unit.x, "from_y": unit.y, "to_x": pos[0], "to_y": pos[1]}
                )

        if not unit.can_attack:
            continue

        # Combat: every enemy the unit may attack and, for a Mage off
        # cooldown, may paralyze. The paralyze rule's target part includes
        # the attack rule's (_paralyzable), so only attack targets are asked.
        if attack.actor(unit):
            may_paralyze = paralyze.actor(unit)
            for target in units:
                if attack.target(state, unit, target):
                    attacks.append({"attacker": unit, "target": target})
                    if may_paralyze and paralyze.target(state, unit, target):
                        paralyzes.append({"paralyzer": unit, "target": target})

        # The other abilities: for each one the unit may use (its rule's
        # actor part: a Cleric, a Sorcerer off that ability's cooldown),
        # every ally the rule lets it target.
        for kind, may_act, may_target, actor_key in _ALLY_TARGETED:
            if may_act(unit):
                entries = legal_actions[kind]
                for target in units:
                    if may_target(state, unit, target):
                        entries.append({actor_key: unit, "target": target})

        # Seizing
        if can_seize(state, unit):
            legal_actions["seize"].append({"unit": unit, "tile": state.grid.get_tile(unit.x, unit.y)})

    return legal_actions
