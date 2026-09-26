"""Pathfinding: one blocker set per search, and Dijkstra that equals the old BFS (core-20/25).

``can_move_to_position`` scanned every unit for each tile the search
examined, which made ``get_legal_actions`` O(units x reachable x units)
(review core-20). The engine, the rule bots and the GUI overlay now collect
the blocking tiles once per search. Reachability is also a Dijkstra search
when a terrain move-cost table is configured (core-25); with every cost 1 it
must give exactly the old breadth-first result -- same tiles, same order,
since the order fixes the move-action order that policies and bot
tiebreaks see -- which is checked on every shipped map.
"""

import glob
import random
from collections import deque

import pandas as pd
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.mechanics import GameMechanics
from reinforcetactics.core.unit import Unit
from reinforcetactics.game.bot import MasterBot, MediumBot
from reinforcetactics.utils.file_io import FileIO

SHIPPED_MAPS = sorted(glob.glob("maps/**/*.csv", recursive=True))


def _legacy_reachable(unit, grid, units):
    """The pre-core-20 search, verbatim: BFS calling can_move_to_position per tile."""
    reachable = []
    visited = {(unit.x, unit.y)}
    queue = deque([(unit.x, unit.y, 0)])
    while queue:
        x, y, distance = queue.popleft()
        if distance > 0:
            reachable.append((x, y))
        if distance < unit.movement_range:
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if GameMechanics.can_move_to_position(nx, ny, grid, units, moving_unit=unit):
                        visited.add((nx, ny))
                        queue.append((nx, ny, distance + 1))
    return reachable


def _populate(gs: GameState, seed: int, per_side: int = 10) -> None:
    """Scatter units of both players over walkable tiles (a reproducible crowd)."""
    rng = random.Random(seed)
    free = [(t.x, t.y) for row in gs.grid.tiles for t in row if t.is_walkable()]
    rng.shuffle(free)
    types = list(gs.unit_data)
    for i in range(min(2 * per_side, len(free))):
        x, y = free[i]
        gs.place_unit(types[i % len(types)], x, y, 1 + i % 2)


@pytest.mark.parametrize("map_path", SHIPPED_MAPS)
def test_unit_cost_dijkstra_equals_legacy_bfs_on_every_shipped_map(map_path):
    gs = GameState(FileIO.load_map(map_path), num_players=2)
    _populate(gs, seed=len(map_path))
    grid = gs.grid
    walkable = [(t.x, t.y) for row in grid.tiles for t in row if t.is_walkable()]
    occupied = {(u.x, u.y): u for u in gs.units}
    for i, (x, y) in enumerate(walkable):
        mover = occupied.get((x, y)) or Unit("W", x, y, 1 + i % 2)
        for movement in (1, 2, 3, 4, 5):
            mover.movement_range = movement
            legacy = _legacy_reachable(mover, grid, gs.units)
            blocked = GameMechanics.movement_blockers(gs.units, mover)
            can_enter = GameMechanics.passability(grid, blocked)
            dijkstra = mover.find_paths(grid.width, grid.height, can_enter, move_cost=lambda _x, _y: 1)
            assert list(dijkstra) == legacy, (map_path, (x, y), movement)
            # Step counts are the BFS distances.
            assert all(1 <= steps <= movement for steps in dijkstra.values())
            assert mover.find_paths(grid.width, grid.height, can_enter) == dijkstra
            assert gs.get_reachable_positions(mover) == legacy


@pytest.mark.parametrize("map_path", SHIPPED_MAPS)
def test_legal_moves_match_the_legacy_rule_on_every_shipped_map(map_path):
    gs = GameState(FileIO.load_map(map_path), num_players=2)
    _populate(gs, seed=7)
    occupied = {(u.x, u.y) for u in gs.units}
    for player in (1, 2):
        gs.current_player = player
        gs._invalidate_cache()
        expected = [
            (u.x, u.y, pos)
            for u in gs.units
            if u.player == player
            for pos in _legacy_reachable(u, gs.grid, gs.units)
            if pos not in occupied
        ]
        got = [(m["from_x"], m["from_y"], (m["to_x"], m["to_y"])) for m in gs.get_legal_actions(player)["move"]]
        assert got == expected


def _forbid_per_tile_scans(monkeypatch):
    def _scan(*_args, **_kwargs):
        raise AssertionError("per-tile can_move_to_position scan in a search")

    monkeypatch.setattr(GameMechanics, "can_move_to_position", staticmethod(_scan))


def test_searches_do_not_scan_every_unit_per_tile(monkeypatch):
    """get_legal_actions, move_unit and the bots' reachability build one blocker set."""
    gs = GameState(FileIO.load_map("maps/1v1/crossroads.csv"), num_players=2, seed=1)
    _populate(gs, seed=3, per_side=6)
    _forbid_per_tile_scans(monkeypatch)

    moves = gs.get_legal_actions(1)["move"]
    assert moves
    bot = MediumBot(gs, player=1, rng=random.Random(0))
    unit = moves[0]["unit"]
    assert bot.get_reachable(unit)
    assert MasterBot(gs, player=2, rng=random.Random(0))._enemy_reachable_positions(unit)
    assert gs.move_unit(unit, moves[0]["to_x"], moves[0]["to_y"])


def test_friends_are_passable_enemies_are_not():
    md = [["p"] * 5 for _ in range(3)]
    gs = GameState(pd.DataFrame(md), num_players=2)
    walker = gs.place_unit("W", 0, 1, 1)  # movement 3
    gs.place_unit("W", 1, 1, 1)  # friend: pass through, never end on
    gs.place_unit("W", 1, 0, 2)  # enemies seal the other routes
    gs.place_unit("W", 1, 2, 2)
    # BFS order: up, down, left, right from each tile in turn.
    assert gs.get_reachable_positions(walker) == [(0, 0), (0, 2), (1, 1), (2, 1), (2, 0), (2, 2), (3, 1)]
    assert gs.get_move_destinations(walker) == [(0, 0), (0, 2), (2, 1), (2, 0), (2, 2), (3, 1)]
    assert GameMechanics.movement_blockers(gs.units, walker) == {(1, 0), (1, 2)}
    assert GameMechanics.movement_blockers(gs.units) == {(u.x, u.y) for u in gs.units}
