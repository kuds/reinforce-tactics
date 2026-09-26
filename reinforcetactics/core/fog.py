"""Fog of war for one game: what each player sees and knows, and the rules that follow (review core-16).

Moved out of ``game_state.py`` so the fog-of-war rules can be read and
changed in one place. ``GameState`` holds one ``FogOfWar`` (``game.fog``)
and keeps the names everyone calls (``update_visibility``,
``is_position_visible``, ``known_structure``, ``pathing_units``, ...,
``fog_of_war`` and ``visibility_maps``), which call these.

The component owns each player's ``VisibilityMap`` (``maps``) and the
refresh that recomputes them; the knowledge queries the observation, the
renderer, the LLM prompts and the legal actions read; the attack snapshot
that stops a unit attacking what its own move uncovered; the view
``cancel_move`` restores; and the ambush rule. The per-unit parts of that
state live on the units (``visible_enemies_at_action_start``,
``pre_move_visibility``, ``ambushed``), so they are saved and copied with
them. Without fog of war every query answers as if everything were in
sight and every hook does nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from reinforcetactics.core.visibility import StructureSnapshot, VisibilityMap, get_visible_units
from reinforcetactics.rules import TileType

if TYPE_CHECKING:
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.core.unit import Unit

# Turns an enemy unit stays in a player's last-seen memory after it leaves sight.
UNIT_MEMORY_TURNS = 10


class FogOfWar:
    """The fog-of-war state and rules of one game (``GameState.fog``).

    Args:
        game: The game it belongs to. Visibility is computed from its board,
            units and turn, and a refresh invalidates its legal-action cache.
        enabled: Whether the game is played under fog of war. Without it
            ``maps`` stays empty.
    """

    def __init__(self, game: GameState, enabled: bool) -> None:
        self.game = game
        self.enabled = enabled
        # FOW method for future compatibility when different algorithms are added
        # Current options: 'simple_radius' (Option A from proposal)
        # Future options: 'line_of_sight', 'hybrid'
        self.method = "simple_radius" if enabled else "none"
        # One map per player, built (and first computed) by reset(), which
        # GameState.__init__ calls once the whole state exists.
        self.maps: dict[int, VisibilityMap] = {}

    def copy_for(self, game: GameState) -> FogOfWar:
        """An independent deep copy of this component for ``game``, a copy of the game it belongs to.

        ``GameState.clone_for_search`` uses it: deep-copying the component
        itself would copy its game too.
        """
        clone = FogOfWar.__new__(FogOfWar)
        for name, value in vars(self).items():
            setattr(clone, name, game if name == "game" else copy.deepcopy(value))
        return clone

    # ------------------------------------------------------------------
    # Computing visibility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Build every player's fog-of-war map from scratch and compute it.

        Each player starts knowing where every HQ is and who owns it: the
        HQs are recorded in the last-seen memory at the current turn (and
        their tiles count as explored), so observations, the renderer and
        LLM prompts all show them, as the rules say ("enemy HQ is always
        known"). Their later HP and owner are only learnt by seeing them.
        """
        self.maps = {}
        if not self.enabled:
            return
        grid = self.game.grid
        hq_tiles = [
            grid.tiles[y][x] for x, y in grid.structure_positions if grid.tiles[y][x].type == TileType.HEADQUARTERS.value
        ]
        for player in range(1, self.game.num_players + 1):
            vis_map = VisibilityMap(grid.width, grid.height, player)
            for tile in hq_tiles:
                vis_map.remember_structure(tile, self.game.turn_number)
            self.maps[player] = vis_map
        self.update()

    def update(self, player: int | None = None) -> None:
        """Recompute ``player``'s view (every player's when None) from the board as it stands.

        The engine calls this itself whenever a player's vision can change
        (construction and load, moves, unit creation and placement, captures,
        deaths, turn changes), so callers never need to.
        """
        if not self.enabled:
            return

        # Legality under fog of war reads visibility (attackable targets, and
        # the units a player's pathfinding may treat as obstacles), so a
        # visibility change is a legality change.
        self.game._invalidate_cache()

        turn = self.game.turn_number
        if player is not None:
            if player in self.maps:
                self.maps[player].update(self.game)
                self.maps[player].clear_stale_unit_memory(max_turns=UNIT_MEMORY_TURNS, current_turn=turn)
        else:
            for vis_map in self.maps.values():
                vis_map.update(self.game)
                vis_map.clear_stale_unit_memory(max_turns=UNIT_MEMORY_TURNS, current_turn=turn)

    # ------------------------------------------------------------------
    # What a player sees and knows
    # ------------------------------------------------------------------

    def is_visible(self, x: int, y: int, player: int) -> bool:
        """Whether ``(x, y)`` is in ``player``'s sight now (always, without fog of war or a map for ``player``)."""
        if not self.enabled:
            return True
        vis_map = self.maps.get(player)
        if vis_map is None:
            return True
        return vis_map.is_visible(x, y)

    def is_explored(self, x: int, y: int, player: int) -> bool:
        """Whether ``player`` has ever explored ``(x, y)`` (always, without fog of war or a map for ``player``)."""
        if not self.enabled:
            return True
        vis_map = self.maps.get(player)
        if vis_map is None:
            return True
        return vis_map.is_explored(x, y)

    def visible_units(self, player: int, include_own: bool = True) -> list[Unit]:
        """The units ``player`` can see (``visibility.get_visible_units``); its own unless ``include_own`` is False."""
        return get_visible_units(self.game, player, include_own)

    def known_structure(self, player: int, x: int, y: int) -> StructureSnapshot | None:
        """What ``player`` knows about the structure at ``(x, y)``.

        The one view of structures under fog of war: ``to_numpy(for_player)``,
        the renderer and the LLM prompt all read it, so none of them can show
        a player more than it knows (review core-5, critic-integration-3,
        pygame-12).

        Returns:
            None when there is no structure at ``(x, y)`` or ``player`` has
            never seen it. Otherwise a snapshot with ``owner``, ``health``
            and ``turn_seen``: the live state (``turn_seen`` = this turn)
            without fog of war or while ``player`` can see the tile, else
            the state when ``player`` last saw it. Every HQ is known from
            the start of the game (see ``reset``).
        """
        tile = self.game.grid.get_tile(x, y)
        if tile is None or not tile.is_capturable():
            return None
        vis_map = self.maps.get(player) if self.enabled else None
        if vis_map is None or vis_map.is_visible(x, y):
            return StructureSnapshot(
                tile_type=tile.type, owner=tile.player, health=tile.health, position=(x, y), turn_seen=self.game.turn_number
            )
        return vis_map.get_last_seen_structure(x, y)

    def pathing_units(self, player: int) -> list[Unit]:
        """The units ``player``'s pathfinding treats as present: its blocking view.

        Without fog of war that is every unit. Under fog of war it is the
        player's own units plus the units on tiles it can see. Letting a
        hidden enemy block paths and destinations would reveal it through
        the move mask (review core-5), so pathfinding plans around the units
        the player knows of and ``move_unit`` resolves a collision with a
        hidden unit when the move is carried out (the ambush rule, see
        ``resolve_ambush``).
        """
        game = self.game
        if not self.enabled:
            return game.units
        # Teammates' units count as known wherever they stand: teams don't
        # share vision, but a hidden teammate treated as absent would let a
        # unit end its move on the teammate's tile.
        return [u for u in game.units if game.are_allies(u.player, player) or self.is_visible(u.x, u.y, player)]

    # ------------------------------------------------------------------
    # Attacks: only what was in sight when the action began
    # ------------------------------------------------------------------

    def capture_visible_enemies(self, unit: Unit) -> None:
        """Snapshot which enemy units ``unit``'s owner can see, as the enemies it may attack this action.

        This prevents "move to discover, then attack": a unit may only
        attack an enemy that was in sight when its action began. The GUI
        calls it when a unit is selected; ``move_unit`` calls it lazily
        (``before_move``) for callers that move a unit directly.
        """
        if not self.enabled:
            self._set_attack_snapshot(unit, None)
            return

        # The snapshot is taken once, when the action begins. A unit that has
        # already moved this action keeps it: re-selecting an ambushed unit in
        # the GUI (its move can't be cancelled) must not add the ambusher, or
        # any enemy the move revealed, to its attack targets.
        if unit.has_moved and unit.visible_enemies_at_action_start is not None:
            return

        game = self.game
        visible_positions = set()
        for enemy in game.units:
            if game.are_enemies(enemy.player, unit.player):
                if self.is_visible(enemy.x, enemy.y, unit.player):
                    visible_positions.add((enemy.x, enemy.y))

        self._set_attack_snapshot(unit, visible_positions)

    def _set_attack_snapshot(self, unit: Unit, snapshot: set[tuple[int, int]] | None) -> None:
        """Write ``unit``'s attack snapshot, dropping the legal-action cache if that changes it.

        The snapshot decides which enemies the unit may attack, so a new one
        is a legality change. The GUI re-takes it every time a unit that has
        not moved is selected: without the invalidation its action menu,
        built from the cached legal actions, kept an attack the engine now
        refused after the unit's side lost sight of the enemy, and missed
        one it now accepted after another unit revealed an enemy. Only a
        change invalidates, so re-selecting a unit while nothing changed
        keeps the cache.
        """
        changed = unit.visible_enemies_at_action_start != snapshot
        unit.visible_enemies_at_action_start = snapshot
        if changed:
            self.game._invalidate_cache()

    def is_enemy_attackable(self, unit: Unit, enemy: Unit) -> bool:
        """Whether fog of war lets ``unit`` attack ``enemy`` (always, without fog of war).

        Under fog of war the enemy must stand where ``unit``'s snapshot
        (``capture_visible_enemies``) saw one; a unit without a snapshot
        may attack what its owner sees now.
        """
        if not self.enabled:
            return True  # No FOW, all visible enemies are attackable

        # If no snapshot was captured, fall back to current visibility
        if unit.visible_enemies_at_action_start is None:
            return self.is_visible(enemy.x, enemy.y, unit.player)

        # Check if enemy's position was in the pre-move snapshot
        return (enemy.x, enemy.y) in unit.visible_enemies_at_action_start

    # ------------------------------------------------------------------
    # Moves: the snapshot before a move, the ambush, and cancelling
    # ------------------------------------------------------------------

    def before_move(self, unit: Unit) -> None:
        """Snapshot what a move by ``unit`` must not change: its attack targets and its side's view.

        Called by ``move_unit`` once the move is accepted, just before the
        unit moves. The attack snapshot is taken here if the action has none
        yet (the GUI takes it when the unit is selected; RL, LLM and bot
        code paths move units directly). The side's view is kept on the
        unit (``pre_move_visibility``) so ``cancel_move`` can take back
        what the move reveals (review core-9).
        """
        if not self.enabled:
            return
        if unit.visible_enemies_at_action_start is None:
            self.capture_visible_enemies(unit)
        if unit.player in self.maps:
            unit.pre_move_visibility = self.maps[unit.player].copy()

    def resolve_ambush(self, unit: Unit, path: list[tuple[int, int]]) -> tuple[tuple[int, int], Unit | None]:
        """Walk ``unit`` along its planned ``path`` and return where it really stops.

        The ambush rule (fog of war only). The path (start tile first) was
        planned around the units the player can see, so a hidden unit may
        stand on it. The first tile the unit cannot pass (a hidden enemy; a
        teammate's unit, like its own, is passed through) or end on (a
        hidden unit on the destination) stops it on the last tile before
        that one where no unit stands, at worst its start tile.

        Returns:
            ``(stop_tile, blocker)``; ``blocker`` is None when the path is clear.
        """
        game = self.game
        last = len(path) - 1
        for i in range(1, last + 1):
            x, y = path[i]
            blocker = game.get_unit_at_position(x, y)
            if blocker is None:
                continue
            if i < last and game.mechanics.can_move_to_position(
                x, y, game.grid, game.units, moving_unit=unit, teams=game.teams
            ):
                continue  # a unit it may pass through
            for j in range(i - 1, 0, -1):
                if game.get_unit_at_position(*path[j]) is None:
                    return path[j], blocker
            return path[0], blocker
        return path[last], None

    def undo_move(self, unit: Unit, snapshot: VisibilityMap | None) -> None:
        """Take back what ``unit``'s cancelled move revealed to its side (``cancel_move``).

        ``snapshot`` is the side's view from before the move
        (``before_move``); it replaces the current one.
        """
        if not self.enabled:
            return
        if snapshot is not None:
            self.maps[unit.player] = snapshot
        # Any attack snapshot taken since the move (by this unit or
        # another one yet to move, e.g. one that moved and was cancelled
        # in turn) may hold enemies only the move revealed; drop them so
        # they are taken again from the restored view. Units that moved
        # took theirs before this move, the latest action.
        for other in self.game.units:
            if other.player == unit.player and not other.has_moved:
                other.visible_enemies_at_action_start = None
        # Re-derive what the side sees from where its units stand now
        # (the unit back on its origin, any later mover where it went).
        # Without this the tiles it saw from where it moved to stayed
        # VISIBLE, so known_structure served their live state.
        self.update(unit.player)

    # ------------------------------------------------------------------
    # Saves
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """What each player has explored and remembers: a save's ``fog_of_war_state``.

        Keyed by ``str(player)`` so the dict is the same before and after a
        JSON round trip. Empty without fog of war.
        """
        return {str(p): vis_map.to_dict() for p, vis_map in self.maps.items()}

    def map_from_dict(self, data: dict[str, Any], player: int) -> VisibilityMap:
        """Rebuild ``player``'s map (or a unit's pre-move view) from ``VisibilityMap.to_dict`` for this board."""
        return VisibilityMap.from_dict(data, self.game.grid.width, self.game.grid.height, player)

    def restore(self, saved: Mapping[str, Any] | None) -> None:
        """Restore every player's map from ``to_dict``'s output (a save's ``fog_of_war_state``).

        A save that lacks a player's map (version 1 saves have none) gets
        fog rebuilt from the current board instead (``reset``): anything
        explored before the save is lost.
        """
        if not self.enabled:
            return
        saved = saved or {}
        players = range(1, self.game.num_players + 1)
        if all(str(p) in saved for p in players):
            self.maps = {p: self.map_from_dict(saved[str(p)], p) for p in players}
        else:
            self.reset()
