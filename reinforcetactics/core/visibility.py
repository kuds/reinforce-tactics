"""
Fog of War visibility system for Reinforce Tactics.

This module provides visibility tracking and calculation for each player,
implementing a simple radius-based visibility model (Option A).

What a player knows about a structure it cannot currently see is its
last-seen memory here (``VisibilityMap.last_seen_structures``), read through
``GameState.known_structure``. The memory is written when a structure leaves
the player's sight, so it holds exactly what the player last watched happen
there, and at game start every HQ is recorded as known (location and owner),
the documented "enemy HQ is always known" rule.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.core.unit import Unit


# Visibility state constants
UNEXPLORED = 0  # Never seen - terrain unknown
SHROUDED = 1  # Previously explored - terrain known, units/ownership hidden
VISIBLE = 2  # Currently visible - full information


# Default vision ranges for units (Chebyshev distance)
UNIT_VISION_RANGES: dict[str, int] = {
    "W": 3,  # Warrior - standard vision
    "M": 3,  # Mage - standard vision
    "C": 3,  # Cleric - standard vision
    "A": 4,  # Archer - extended vision (scout)
    "K": 3,  # Knight - standard vision
    "R": 4,  # Rogue - extended vision (scout)
    "S": 3,  # Sorcerer - standard vision
    "B": 2,  # Barbarian - limited vision
}

# Vision ranges for structures
STRUCTURE_VISION_RANGES: dict[str, int] = {
    "h": 4,  # Headquarters - large vision radius
    "b": 3,  # Building - standard vision
    "t": 5,  # Tower - best vision (elevated position)
}


@dataclass
class UnitSnapshot:
    """Snapshot of unit information when last visible.

    Used to remember enemy unit positions after they leave vision.
    """

    unit_type: str
    owner: int
    health: int
    max_health: int
    position: tuple[int, int]
    turn_seen: int


@dataclass(frozen=True)
class StructureSnapshot:
    """Snapshot of structure information when last visible.

    Frozen because ``GameState.known_structure`` hands the stored memory out
    to observers (observations, renderer, LLM prompts); none may edit it.
    """

    tile_type: str
    owner: int | None
    health: int
    position: tuple[int, int]
    turn_seen: int


class VisibilityMap:
    """Tracks visibility state for a single player.

    Maintains three layers of information:
    1. Visibility state (unexplored/shrouded/visible) for each tile
    2. Memory of last-seen enemy units
    3. Memory of last-seen structure states: for each structure the player
       has seen but cannot see now, its type, owner and HP at the moment it
       left sight (structures in sight are read live instead)

    Args:
        width: Grid width
        height: Grid height
        player: The player this visibility map belongs to
    """

    def __init__(self, width: int, height: int, player: int):
        self.width = width
        self.height = height
        self.player = player

        # Visibility state: 0=unexplored, 1=shrouded, 2=visible
        self.state = np.zeros((height, width), dtype=np.uint8)

        # Memory of last-seen enemy units (position -> UnitSnapshot)
        self.last_seen_units: dict[tuple[int, int], UnitSnapshot] = {}

        # Memory of last-seen structures (position -> StructureSnapshot)
        self.last_seen_structures: dict[tuple[int, int], StructureSnapshot] = {}

        # Current visibility mask (recomputed each update)
        self._current_visible = np.zeros((height, width), dtype=bool)

    def copy(self) -> "VisibilityMap":
        """An independent copy, much cheaper than ``copy.deepcopy``.

        The arrays and the memory dicts are copied. The snapshots in the
        dicts are shared: memory changes only by replacing a snapshot, never
        by editing one. ``GameState.move_unit`` takes one of these before
        every fog-of-war move so ``cancel_move`` can restore it.
        """
        clone = VisibilityMap.__new__(VisibilityMap)
        clone.width, clone.height, clone.player = self.width, self.height, self.player
        clone.state = self.state.copy()
        clone.last_seen_units = dict(self.last_seen_units)
        clone.last_seen_structures = dict(self.last_seen_structures)
        clone._current_visible = self._current_visible.copy()
        return clone

    def update(self, game_state: "GameState") -> None:
        """Recalculate visibility based on current unit/structure positions.

        This method:
        1. Marks previously visible tiles as shrouded
        2. Calculates new visibility from all owned units and structures
        3. Updates memory of seen enemy units and structures

        Args:
            game_state: Current game state
        """
        # What was in sight until now: structures that leave sight in this
        # update are remembered as they are at this moment (_update_memory).
        was_visible = self.state == VISIBLE

        # Step 1: Mark previously visible areas as shrouded (not unexplored)
        self.state[was_visible] = SHROUDED

        # Step 2: Calculate new visibility
        self._current_visible.fill(False)

        # Vision from units (includes terrain bonuses like mountain +1)
        grid = game_state.grid
        for unit in game_state.units:
            if unit.player == self.player:
                tile = grid.get_tile(unit.x, unit.y)
                tile_type = tile.type if tile else None
                vision_range = calculate_vision_radius(unit.type, tile_type=tile_type)
                self._add_vision_radius(unit.x, unit.y, vision_range)

        # Vision from structures. Tile types never change, so the grid's
        # cached structure positions replace a scan of every tile.
        for x, y in grid.structure_positions:
            tile = grid.tiles[y][x]
            if tile.player == self.player and tile.type in STRUCTURE_VISION_RANGES:
                vision_range = calculate_vision_radius(tile.type, is_structure=True)
                self._add_vision_radius(x, y, vision_range)

        # Optional terrain rules (forest concealment, HQ always known) from
        # engine_overrides; off by default. See core/terrain_rules.py.
        terrain_rules = getattr(game_state, "terrain_rules", None)
        if terrain_rules is not None:
            terrain_rules.adjust_vision(self, game_state)

        # Step 3: Update state array
        self.state[self._current_visible] = VISIBLE

        # Step 4: Update memory of what we can see
        self._update_memory(game_state, was_visible)

    def _add_vision_radius(self, cx: int, cy: int, radius: int) -> None:
        """Add circular vision around a point using Chebyshev distance.

        Chebyshev distance (king's movement) creates a square visibility area,
        which is simpler and faster than Euclidean distance circles: it is
        exactly the square slice around the point, clipped to the board.

        Args:
            cx: Center x coordinate
            cy: Center y coordinate
            radius: Vision radius in tiles
        """
        self._current_visible[max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1] = True

    def _update_memory(self, game_state: "GameState", was_visible: np.ndarray) -> None:
        """Update memory of seen units and structures.

        Args:
            game_state: Current game state
            was_visible: Mask of the tiles that were visible before this update
        """
        turn = game_state.turn_number

        # Clear memory for positions that are now visible (will re-add current info)
        positions_to_clear = [pos for pos in self.last_seen_units if self.is_visible(pos[0], pos[1])]
        for pos in positions_to_clear:
            del self.last_seen_units[pos]

        # Record enemy units we can see
        for unit in game_state.units:
            if unit.player != self.player and self.is_visible(unit.x, unit.y):
                self.last_seen_units[(unit.x, unit.y)] = UnitSnapshot(
                    unit_type=unit.type,
                    owner=unit.player,
                    health=unit.health,
                    max_health=unit.max_health,
                    position=(unit.x, unit.y),
                    turn_seen=turn,
                )

        # Structures. One in sight is known live, so it needs no memory. One
        # that leaves sight now is remembered as it is now: the player
        # watched it right up to this update, including any change made to it
        # since the previous one (an enemy's partial seize, say), which a
        # copy taken at the previous update would miss. Nothing else touches
        # the memory, so repeating an update changes nothing.
        grid = game_state.grid
        for x, y in grid.structure_positions:
            if self._current_visible[y, x]:
                self.last_seen_structures.pop((x, y), None)
            elif was_visible[y, x]:
                self.remember_structure(grid.tiles[y][x], turn)

    def remember_structure(self, tile: Any, turn: int) -> None:
        """Record ``tile``'s current type, owner and HP as last seen on ``turn``.

        Also marks the tile explored (its terrain is known from then on).
        ``GameState`` uses this directly to seed every HQ at game start.
        """
        self.last_seen_structures[(tile.x, tile.y)] = StructureSnapshot(
            tile_type=tile.type, owner=tile.player, health=tile.health, position=(tile.x, tile.y), turn_seen=turn
        )
        if self.state[tile.y, tile.x] == UNEXPLORED:
            self.state[tile.y, tile.x] = SHROUDED

    def to_dict(self) -> dict[str, Any]:
        """Serialise the explored/visible state and the last-seen memory.

        The state is one string of digits (0/1/2) per row, which keeps a save
        written with ``indent=2`` to one line per row instead of one per tile.
        Memory entries are sorted by position so equal maps serialise equally.
        """
        return {
            "state": ["".join(str(v) for v in row) for row in self.state.tolist()],
            "last_seen_structures": [
                {
                    "x": x,
                    "y": y,
                    "tile_type": snap.tile_type,
                    "owner": snap.owner,
                    "health": snap.health,
                    "turn_seen": snap.turn_seen,
                }
                for (x, y), snap in sorted(self.last_seen_structures.items())
            ],
            "last_seen_units": [
                {
                    "x": x,
                    "y": y,
                    "unit_type": snap.unit_type,
                    "owner": snap.owner,
                    "health": snap.health,
                    "max_health": snap.max_health,
                    "turn_seen": snap.turn_seen,
                }
                for (x, y), snap in sorted(self.last_seen_units.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], width: int, height: int, player: int) -> "VisibilityMap":
        """Rebuild a map written by :meth:`to_dict` for a ``width`` x ``height`` board.

        Raises:
            ValueError: if the saved state does not fit the board.
        """
        vis_map = cls(width, height, player)
        rows = data.get("state", [])
        if len(rows) != height or any(len(row) != width for row in rows):
            raise ValueError(f"Saved fog-of-war state for player {player} does not fit the {width}x{height} board")
        vis_map.state = np.array([[int(c) for c in row] for row in rows], dtype=np.uint8).reshape(height, width)
        vis_map._current_visible = vis_map.state == VISIBLE
        for entry in data.get("last_seen_structures", []):
            x, y = entry["x"], entry["y"]
            vis_map.last_seen_structures[(x, y)] = StructureSnapshot(
                tile_type=entry["tile_type"],
                owner=entry["owner"],
                health=entry["health"],
                position=(x, y),
                turn_seen=entry["turn_seen"],
            )
        for entry in data.get("last_seen_units", []):
            x, y = entry["x"], entry["y"]
            vis_map.last_seen_units[(x, y)] = UnitSnapshot(
                unit_type=entry["unit_type"],
                owner=entry["owner"],
                health=entry["health"],
                max_health=entry["max_health"],
                position=(x, y),
                turn_seen=entry["turn_seen"],
            )
        return vis_map

    def is_visible(self, x: int, y: int) -> bool:
        """Check if a tile is currently visible.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if tile is currently visible
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state[y, x] == VISIBLE
        return False

    def is_explored(self, x: int, y: int) -> bool:
        """Check if a tile has ever been explored.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if tile is explored (visible or shrouded)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.state[y, x] >= SHROUDED
        return False

    def get_visibility_state(self, x: int, y: int) -> int:
        """Get the visibility state of a tile.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            UNEXPLORED (0), SHROUDED (1), or VISIBLE (2)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.state[y, x])
        return UNEXPLORED

    def get_visible_mask(self) -> np.ndarray:
        """Get a boolean mask of currently visible tiles.

        Returns:
            2D numpy array where True = visible
        """
        return self.state == VISIBLE

    def get_explored_mask(self) -> np.ndarray:
        """Get a boolean mask of explored tiles.

        Returns:
            2D numpy array where True = explored (visible or shrouded)
        """
        return self.state >= SHROUDED

    def get_last_seen_structure(self, x: int, y: int) -> StructureSnapshot | None:
        """Get the last-seen structure info at a position.

        Most callers want ``GameState.known_structure``, which also answers
        for structures in sight (read live, so not kept here).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            StructureSnapshot if a structure was seen there and is out of
            sight now (or is an HQ known from the start), None otherwise
        """
        return self.last_seen_structures.get((x, y))

    def clear_stale_unit_memory(self, max_turns: int, current_turn: int) -> None:
        """Remove unit memories older than max_turns.

        Args:
            max_turns: Maximum number of turns to remember units
            current_turn: Current game turn
        """
        stale_positions = [
            pos for pos, snapshot in self.last_seen_units.items() if current_turn - snapshot.turn_seen > max_turns
        ]
        for pos in stale_positions:
            del self.last_seen_units[pos]

    def to_numpy(self) -> np.ndarray:
        """Convert visibility state to numpy array.

        Returns:
            2D numpy array with visibility state values (0, 1, or 2)
        """
        return self.state.copy()


def calculate_vision_radius(unit_or_structure_type: str, tile_type: str | None = None, is_structure: bool = False) -> int:
    """Calculate vision radius for a unit or structure.

    Args:
        unit_or_structure_type: Unit type code ('W', 'M', etc.) or structure type ('h', 'b', 't')
        tile_type: The terrain tile the unit is standing on (for bonuses)
        is_structure: True if this is a structure, False if unit

    Returns:
        Vision radius in tiles
    """
    if is_structure:
        base_range = STRUCTURE_VISION_RANGES.get(unit_or_structure_type, 3)
    else:
        base_range = UNIT_VISION_RANGES.get(unit_or_structure_type, 3)

    # Mountain bonus: +1 vision when standing on mountain (units only)
    if not is_structure and tile_type == "m":
        base_range += 1

    return base_range


def get_visible_units(game_state: "GameState", player: int, include_own: bool = True) -> list["Unit"]:
    """Get list of units visible to a player.

    Args:
        game_state: Current game state
        player: Player to get visible units for
        include_own: Whether to include the player's own units

    Returns:
        List of visible units
    """
    if not game_state.fog_of_war:
        # No fog of war - all units visible
        if include_own:
            return list(game_state.units)
        return [u for u in game_state.units if u.player != player]

    visibility_map = game_state.visibility_maps.get(player)
    if visibility_map is None:
        return list(game_state.units) if include_own else []

    visible = []
    for unit in game_state.units:
        if unit.player == player:
            if include_own:
                visible.append(unit)
        elif visibility_map.is_visible(unit.x, unit.y):
            visible.append(unit)

    return visible
