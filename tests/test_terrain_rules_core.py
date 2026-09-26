"""Optional terrain rules from ``engine_overrides`` (review core-25).

The README promised road speed, forest stealth and an always-visible enemy
HQ; the engine had none of them. They now exist as opt-in rules whose
defaults are the shipped game (pinned here and by the shipped-map
equivalence tests in test_pathfinding_core.py): terrain move costs, a
path-based Knight Charge, forest concealment and HQ-always-known.
"""

import json

import numpy as np
import pandas as pd
import pytest

from reinforcetactics.core import legal_actions
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.terrain_rules import WALKABLE_TILE_CODES, TerrainRules
from reinforcetactics.core.visibility import SHROUDED, UNEXPLORED, VISIBLE
from reinforcetactics.utils.file_io import FileIO


def _board(rows: list[str], **kwargs) -> GameState:
    """A GameState from rows of space-separated tile codes."""
    return GameState(pd.DataFrame([row.split() for row in rows]), num_players=2, **kwargs)


class TestDefaultsAndValidation:
    def test_defaults_are_the_shipped_game(self):
        rules = _board(["p p", "p p"]).terrain_rules
        assert rules == TerrainRules()
        assert rules.move_costs == {}
        assert rules.charge_distance == "displacement"
        assert not rules.forest_concealment and not rules.hq_always_visible

    def test_all_one_cost_table_is_the_default_rule(self):
        overrides = {"terrain_move_cost": {code: 1 for code in WALKABLE_TILE_CODES}}
        assert TerrainRules.from_overrides(overrides) == TerrainRules()
        for map_path in ("maps/1v1/crossroads.csv", "maps/1v1/difficult_terrain.csv"):
            plain = GameState(FileIO.load_map(map_path), num_players=2)
            costed = GameState(FileIO.load_map(map_path), num_players=2, engine_overrides=overrides)
            for gs in (plain, costed):
                gs.place_unit("K", *next((t.x, t.y) for row in gs.grid.tiles for t in row if t.type == "r"), 1)
            assert [(m["to_x"], m["to_y"]) for m in plain.get_legal_actions(1)["move"]] == [
                (m["to_x"], m["to_y"]) for m in costed.get_legal_actions(1)["move"]
            ]

    @pytest.mark.parametrize(
        "overrides, error",
        [
            ({"terrain_move_cost": {"w": 2}}, KeyError),  # impassable
            ({"terrain_move_cost": {"x": 2}}, KeyError),  # not a tile code
            ({"terrain_move_cost": {"f": 0}}, ValueError),
            ({"terrain_move_cost": {"f": -1}}, ValueError),
            ({"terrain_move_cost": {"f": True}}, ValueError),
            ({"terrain_move_cost": {"f": "2"}}, ValueError),
            ({"charge_distance": "manhattan"}, ValueError),
            ({"forest_concealment": "yes"}, ValueError),
            ({"hq_always_visible": 1}, ValueError),
            ({"forest_concelment": True}, KeyError),  # misspelt key
        ],
    )
    def test_bad_overrides_fail_loud(self, overrides, error):
        with pytest.raises(error):
            _board(["p p", "p p"], engine_overrides=overrides)

    def test_rules_survive_save_and_load(self):
        overrides = {"terrain_move_cost": {"r": 0.5}, "charge_distance": "path", "forest_concealment": True}
        gs = _board(["p r", "r p"], engine_overrides=overrides)
        restored = GameState.from_dict(json.loads(json.dumps(gs.to_dict())))
        assert restored.terrain_rules == gs.terrain_rules


class TestMoveCosts:
    def test_roads_at_half_cost_double_the_range(self):
        default = _board(["r " * 12])
        fast = _board(["r " * 12], engine_overrides={"terrain_move_cost": {"r": 0.5}})
        for gs, reach in ((default, 3), (fast, 6)):
            warrior = gs.place_unit("W", 0, 0, 1)  # movement 3
            assert gs.get_move_destinations(warrior) == [(x, 0) for x in range(1, reach + 1)]
            assert [(m["to_x"], m["to_y"]) for m in gs.get_legal_actions(1)["move"]] == [(x, 0) for x in range(1, reach + 1)]
        # The engine validates moves with the same costs.
        assert not default.move_unit(default.units[0], 6, 0)
        assert fast.move_unit(fast.units[0], 6, 0)

    def test_slow_terrain_costs_more_and_the_cheapest_path_wins(self):
        gs = _board(["p f p p p p"], engine_overrides={"terrain_move_cost": {"f": 2}})
        warrior = gs.place_unit("W", 0, 0, 1)
        assert gs.get_move_destinations(warrior) == [(1, 0), (2, 0)]  # 2 into forest, 1 more

        # Straight through the forest costs 3 (2 + 1) in 2 steps; around it, 4.
        gs = _board(["p p p", "p f p", "p p p"], engine_overrides={"terrain_move_cost": {"f": 2}})
        warrior = gs.place_unit("W", 1, 0, 1)
        paths = legal_actions.find_paths(gs, warrior)
        assert paths[(1, 2)] == 2
        assert set(paths) == {(0, 0), (2, 0), (0, 1), (2, 1), (1, 1), (0, 2), (2, 2), (1, 2)}

    def test_units_still_pass_friends_and_stop_at_enemies_under_costs(self):
        overrides = {"terrain_move_cost": {"r": 0.5}}
        gs = _board(["r " * 8], engine_overrides=overrides)
        mover = gs.place_unit("W", 0, 0, 1)
        gs.place_unit("W", 2, 0, 1)
        gs.place_unit("W", 5, 0, 2)
        assert gs.get_reachable_positions(mover) == [(1, 0), (2, 0), (3, 0), (4, 0)]
        assert gs.get_move_destinations(mover) == [(1, 0), (3, 0), (4, 0)]


class TestKnightCharge:
    # The Knight at (0, 1) must go around the water at (1, 1) to reach
    # (2, 1): 4 tiles of path for 2 tiles of displacement.
    ROWS = ["p p p p p", "p w p p p", "p p p p p"]

    @pytest.mark.parametrize("mode, moved, charge", [(None, 2, False), ("path", 4, True)])
    def test_charge_counts_what_the_rule_says(self, mode, moved, charge):
        overrides = {"charge_distance": mode} if mode else None
        gs = _board(self.ROWS, engine_overrides=overrides)
        knight = gs.place_unit("K", 0, 1, 1)
        target = gs.place_unit("W", 3, 1, 2)
        assert gs.move_unit(knight, 2, 1)
        assert knight.distance_moved == moved
        assert gs.attack(knight, target)["charge_bonus"] is charge

    def test_path_mode_counts_a_straight_move_like_displacement(self):
        gs = _board(self.ROWS, engine_overrides={"charge_distance": "path"})
        knight = gs.place_unit("K", 0, 0, 1)
        assert gs.move_unit(knight, 3, 0)
        assert knight.distance_moved == 3


class TestForestConcealment:
    # P1's Archer (vision 4) at (2, 3); P2's Warrior hides in the forest at
    # (4, 3), two tiles away -- inside the Archer's vision and attack range.
    ROWS = [
        "h_1 p p p p p p",
        "p p p p p p p",
        "p p p p p p p",
        "p p p p f p p",
        "p p p p p p p",
        "p p p p p p p",
        "p p p p p p h_2",
    ]

    def _game(self, concealment: bool) -> GameState:
        gs = _board(self.ROWS, fog_of_war=True, engine_overrides={"forest_concealment": concealment})
        gs.update_visibility()
        gs.place_unit("A", 2, 3, 1)
        gs.place_unit("W", 4, 3, 2)
        return gs

    def test_without_the_rule_the_forest_unit_is_seen(self):
        gs = self._game(False)
        assert gs.is_position_visible(4, 3, 1)
        assert [a["target"].type for a in gs.get_legal_actions(1)["attack"]] == ["W"]

    def test_a_forest_unit_is_hidden_from_non_adjacent_enemies(self):
        gs = self._game(True)
        assert not gs.is_position_visible(4, 3, 1)
        # The forest itself is explored terrain, just not seen into.
        assert gs.visibility_maps[1].get_visibility_state(4, 3) == SHROUDED
        assert gs.to_numpy(for_player=1)["units"][3, 4, 0] == 0
        assert all(u.player == 1 for u in gs.get_visible_units_for_player(1))
        assert (4, 3) not in gs.visibility_maps[1].last_seen_units
        assert gs.get_legal_actions(1)["attack"] == []  # cannot shoot what it cannot see
        # Its owner still sees it, and forests elsewhere stay visible.
        assert gs.to_numpy(for_player=2)["units"][3, 4, 0] != 0

    def test_an_adjacent_unit_spots_it(self):
        gs = self._game(True)
        gs.place_unit("W", 3, 3, 1)
        assert gs.is_position_visible(4, 3, 1)
        assert gs.visibility_maps[1].get_visibility_state(4, 3) == VISIBLE


class TestHqAlwaysVisible:
    ROWS = ["h_1 " + "p " * 10 + "p"] + ["p " * 12] * 10 + ["p " * 11 + "h_2"]

    def _game(self, reveal: bool) -> GameState:
        gs = _board(self.ROWS, fog_of_war=True, engine_overrides={"hq_always_visible": reveal})
        gs.place_unit("W", 11, 11, 2)  # standing on its own HQ
        gs.update_visibility()
        return gs

    def test_without_the_rule_an_hq_out_of_sight_shows_its_last_seen_state(self):
        # Fog of war's default (review core-5): every HQ's location and
        # starting owner are known from the start, as a last-seen snapshot,
        # so a change made out of sight stays hidden.
        gs = self._game(False)
        assert gs.visibility_maps[1].get_visibility_state(11, 11) == SHROUDED
        gs.grid.get_tile(11, 11).health = 10  # damaged out of player 1's sight
        gs.update_visibility()
        obs = gs.to_numpy(for_player=1)
        assert obs["grid"][11, 11, 1] == 2 and obs["grid"][11, 11, 0] == 6  # owner, HQ type code
        assert gs.known_structure(1, 11, 11).health == 50  # as last seen, not the live 10

    def test_the_rule_shows_an_hqs_live_state(self):
        gs = self._game(True)
        gs.grid.get_tile(11, 11).health = 10
        gs.update_visibility()
        assert gs.known_structure(1, 11, 11).health == 10
        assert not gs.is_position_visible(11, 11, 1)

    def test_the_rule_makes_every_hq_known(self):
        gs = self._game(True)
        vis = gs.visibility_maps[1]
        assert vis.get_visibility_state(11, 11) == SHROUDED
        assert vis.get_last_seen_structure(11, 11).owner == 2
        obs = gs.to_numpy(for_player=1)
        assert obs["grid"][11, 11, 1] == 2 and obs["grid"][11, 11, 0] == 6  # owner, HQ type code
        # Only the tile: the unit on it still needs real vision.
        assert obs["units"][11, 11, 0] == 0
        assert not gs.is_position_visible(11, 11, 1)
        assert np.count_nonzero(vis.state == UNEXPLORED) > 0  # nothing else was revealed
