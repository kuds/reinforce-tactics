"""Optional terrain rules, switched on through ``engine_overrides`` (review core-25).

The README used to promise road speed, forest stealth and an always-visible
enemy HQ, none of which the engine implemented: movement cost 1 on every
walkable tile, forests only helped a Rogue's evade roll, and the fog-of-war
observation hid an unexplored enemy HQ. They are available here as opt-in
rules whose defaults reproduce the shipped game exactly, so turning one on
is a deliberate, recorded balance change (``engine_overrides`` is written
to saves and to training runs' config.json):

``terrain_move_cost``: ``{tile_code: cost}``
    Movement points to *enter* a tile of that type (default 1 for every
    walkable tile). A unit may enter a tile while the path's total stays
    within its movement stat, so ``{"r": 0.5}`` lets a unit travel twice
    as far along roads and ``{"f": 2, "m": 2}`` makes forests and
    mountains slow. Reachability is then a Dijkstra search; with every cost
    1 it is exactly the old breadth-first search.
``charge_distance``: ``"displacement"`` (default) | ``"path"``
    What the Knight's Charge counts: the Manhattan distance from where the
    Knight started (default) or the number of tiles along the path the
    engine found for the move (so going around a lake counts).
``forest_concealment``: bool (default False)
    Under fog of war, a unit standing in forest is seen only by an enemy
    with a unit on or orthogonally next to its tile. The forest tile itself
    is still explored.
``hq_always_visible``: bool (default False)
    Under fog of war, every HQ tile counts as explored for every player, so
    its position and owner are always known (units on it still need normal
    vision).

The two vision rules only matter when fog of war is on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reinforcetactics.core.visibility import SHROUDED, StructureSnapshot

if TYPE_CHECKING:
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.core.grid import TileGrid
    from reinforcetactics.core.visibility import VisibilityMap

# Walkable tile codes a movement cost may be set for. Water and ocean are
# impassable, so a cost for them would be meaningless and is rejected.
WALKABLE_TILE_CODES = ("p", "m", "f", "r", "b", "h", "t")
CHARGE_DISTANCE_MODES = ("displacement", "path")

# The engine_overrides keys this module resolves.
TERRAIN_RULE_KEYS = ("terrain_move_cost", "charge_distance", "forest_concealment", "hq_always_visible")


@dataclass(frozen=True)
class TerrainRules:
    """The resolved optional terrain rules of one game (all off by default)."""

    # Only the tile codes whose cost is not 1; empty means uniform cost.
    move_costs: Mapping[str, float] = field(default_factory=dict)
    charge_distance: str = "displacement"
    forest_concealment: bool = False
    hq_always_visible: bool = False

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any] | None) -> TerrainRules:
        """Resolve and validate the terrain keys of an ``engine_overrides`` overlay.

        Bad values fail loud, like every other override: a typo in a balance
        sweep must not silently train on the default rules.
        """
        overrides = overrides or {}
        costs: dict[str, float] = {}
        for code, cost in (overrides.get("terrain_move_cost") or {}).items():
            if code not in WALKABLE_TILE_CODES:
                raise KeyError(
                    f"engine_overrides.terrain_move_cost: '{code}' is not a walkable tile code "
                    f"(valid: {list(WALKABLE_TILE_CODES)})"
                )
            if isinstance(cost, bool) or not isinstance(cost, int | float) or not cost > 0:
                raise ValueError(f"engine_overrides.terrain_move_cost['{code}'] must be a positive number, got {cost!r}")
            if cost != 1:
                costs[code] = cost
        charge = overrides.get("charge_distance", "displacement")
        if charge not in CHARGE_DISTANCE_MODES:
            raise ValueError(f"engine_overrides.charge_distance must be one of {CHARGE_DISTANCE_MODES}, got {charge!r}")
        flags = {}
        for key in ("forest_concealment", "hq_always_visible"):
            value = overrides.get(key, False)
            if not isinstance(value, bool):
                raise ValueError(f"engine_overrides.{key} must be true or false, got {value!r}")
            flags[key] = value
        return cls(move_costs=costs, charge_distance=charge, **flags)

    def move_cost_fn(self, grid: TileGrid) -> Callable[[int, int], float] | None:
        """Entry cost of tile ``(x, y)`` for ``Unit.find_paths``; None for uniform cost."""
        if not self.move_costs:
            return None
        costs = self.move_costs
        tiles = grid.tiles
        return lambda x, y: costs.get(tiles[y][x].type, 1)

    def adjust_vision(self, vis_map: VisibilityMap, game_state: GameState) -> None:
        """Apply the vision rules to a visibility map mid-update.

        Called by ``VisibilityMap.update`` after it has computed this turn's
        raw vision (``vis_map._current_visible``) and before it writes it to
        ``vis_map.state`` and refreshes its memory, so a concealed unit is
        neither shown nor remembered. A no-op with both rules off.
        """
        if not (self.forest_concealment or self.hq_always_visible):
            return
        grid = game_state.grid
        visible = vis_map._current_visible  # noqa: SLF001 -- the documented hook contract
        if self.forest_concealment:
            spotters = {(u.x, u.y) for u in game_state.units if u.player == vis_map.player}
            for row in grid.tiles:
                for tile in row:
                    x, y = tile.x, tile.y
                    if tile.type != "f" or not visible[y, x]:
                        continue
                    if any((x + dx, y + dy) in spotters for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))):
                        continue
                    # In sight but concealed: the terrain is known, not what is in it.
                    visible[y, x] = False
                    vis_map.state[y, x] = max(int(vis_map.state[y, x]), SHROUDED)
        if self.hq_always_visible:
            for tile in grid.get_capturable_tiles():
                if tile.type != "h" or visible[tile.y, tile.x]:
                    continue
                vis_map.state[tile.y, tile.x] = max(int(vis_map.state[tile.y, tile.x]), SHROUDED)
                vis_map.last_seen_structures[(tile.x, tile.y)] = StructureSnapshot(
                    tile_type=tile.type,
                    owner=tile.player,
                    health=tile.health,
                    position=(tile.x, tile.y),
                    turn_seen=game_state.turn_number,
                )
