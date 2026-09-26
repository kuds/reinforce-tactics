"""Saves resume the exact game (review core-6, core-17, critic-integration-11).

``to_dict`` dropped fields that ``from_dict`` reads or that the rules depend
on (winning_action_index, healing_totals, per-unit has_moved, stats from
engine overrides, the fog-of-war state), and ``from_dict``
aliased its inputs. Saves now carry a format version; older saves (every file
in saves/) still load. The property test at the bottom plays random games
with and without fog of war, for 2 and 3 players, and checks that a JSON
round trip is exact and offers every player the same legal actions.
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.app.game_loop import restore_saved_game
from reinforcetactics.constants import BUILDING_MAX_HEALTH
from reinforcetactics.core.game_state import SAVE_FORMAT_VERSION, GameState
from reinforcetactics.core.tile import Tile
from reinforcetactics.core.unit import Unit
from reinforcetactics.game.bot import RandomBot, SimpleBot
from reinforcetactics.utils.file_io import FileIO

REPO_ROOT = Path(__file__).resolve().parents[1]


def _json(data):
    return json.loads(json.dumps(data))


def _reload(game):
    return GameState.from_dict(_json(game.to_dict()))


def _map(size=10):
    grid = np.array([["p"] * size for _ in range(size)], dtype=object)
    grid[0][0] = "h_1"
    grid[size - 1][size - 1] = "h_2"
    grid[3][3] = "b"
    return grid


def _actions(game):
    return [{k: v for k, v in a.items() if k != "timestamp"} for a in _json(game.action_history)]


def _canon(value):
    """Legal actions with units and tiles replaced by comparable identities."""
    if isinstance(value, Unit):
        return ("unit", value.unit_id, value.type, value.player, value.x, value.y)
    if isinstance(value, Tile):
        return ("tile", value.x, value.y)
    if isinstance(value, dict):
        return tuple(sorted((k, _canon(v)) for k, v in value.items()))
    if isinstance(value, list):
        return [_canon(v) for v in value]
    return value


class TestEveryFieldSurvivesAReload:
    def test_turn_limit_and_engine_override_stats(self):
        """The core-6 repro: max_turns and override stats were lost, leaving health above max."""
        game = GameState(_map(), max_turns=30, engine_overrides={"unit_data": {"W": {"health": 25, "attack": 20}}})
        warrior = game.place_unit("W", 5, 5, player=1)
        warrior.health = 22

        loaded = _reload(game)

        w = loaded.units[0]
        assert loaded.max_turns == 30
        assert (w.max_health, w.attack_data, w.health) == (25, 20, 22)

    def test_loaded_health_never_exceeds_max_health(self):
        data = _json(GameState(_map()).to_dict())
        data["units"] = [{"type": "W", "x": 5, "y": 5, "player": 1, "health": 99}]

        unit = GameState.from_dict(data).units[0]

        assert unit.health == unit.max_health == 15

    def test_end_of_game_and_healing_totals(self):
        game = GameState(_map())
        game.healing_totals[1] = {"hp": 7, "gold": 93}
        game.end_turn()
        game.resign(2)

        loaded = _reload(game)

        assert loaded.game_over_action_index == game.game_over_action_index == 1
        assert (loaded.end_reason, loaded.winner) == ("resign", 1)
        assert loaded.healing_totals == {1: {"hp": 7, "gold": 93}, 2: {"hp": 0, "gold": 0}}

    def test_a_mid_turn_move_off_a_seized_structure_still_resets_it(self):
        """has_moved was not saved, so end_turn skipped the vacated-structure reset after a load."""
        game = GameState(_map())
        warrior = game.place_unit("W", 3, 3, player=1)
        game.seize(warrior)
        assert game.grid.get_tile(3, 3).health == BUILDING_MAX_HEALTH - 15
        game.end_turn()
        game.end_turn()
        assert game.move_unit(warrior, 3, 5)

        loaded = _reload(game)
        loaded.end_turn()

        assert loaded.grid.get_tile(3, 3).health == BUILDING_MAX_HEALTH

    def test_knight_charge_distance_survives(self):
        game = GameState(_map())
        knight = game.place_unit("K", 5, 1, player=1)
        assert game.move_unit(knight, 5, 5)

        loaded_knight = _reload(game).units[0]

        assert (loaded_knight.has_moved, loaded_knight.distance_moved) == (True, 4)

    def test_an_early_version_2_save_with_padding_metadata_loads_unchanged(self, caplog):
        """Version 2 saves used to carry padding metadata; nothing ever set it."""
        game = GameState(_map(12))
        warrior = game.place_unit("W", 3, 3, player=1)
        assert game.move_unit(warrior, 3, 5)
        data = _json(game.to_dict())
        data.update(
            original_map_width=12,
            original_map_height=12,
            map_padding_offset_x=0,
            map_padding_offset_y=0,
            original_map_data=None,
        )

        with caplog.at_level(logging.WARNING, logger="reinforcetactics.core.game_state"):
            loaded = GameState.from_dict(data)

        assert _json(loaded.to_dict()) == _json(game.to_dict())
        assert not caplog.records

    def test_a_save_with_padding_offsets_warns(self, caplog):
        """Only a script calling the removed set_map_metadata could write one."""
        data = _json(GameState(_map()).to_dict())
        data.update(map_padding_offset_x=2, map_padding_offset_y=2)

        with caplog.at_level(logging.WARNING, logger="reinforcetactics.core.game_state"):
            GameState.from_dict(data)

        assert "padding offsets (2, 2)" in caplog.text

    def test_fog_of_war_state_survives(self):
        """critic-integration-11: a reloaded fog game re-fogged the map and forgot structures."""
        game = GameState(_map(), fog_of_war=True)
        archer = game.place_unit("A", 7, 3, player=1)
        assert game.move_unit(archer, 7, 6)
        explored = game.visibility_maps[1].get_explored_mask()

        loaded = _reload(game)

        np.testing.assert_array_equal(loaded.visibility_maps[1].state, game.visibility_maps[1].state)
        assert loaded.visibility_maps[1].get_explored_mask().sum() == explored.sum() > 0
        assert loaded.visibility_maps[1].last_seen_structures == game.visibility_maps[1].last_seen_structures
        for player in (1, 2):
            for key, value in game.to_numpy(for_player=player).items():
                np.testing.assert_array_equal(np.asarray(loaded.to_numpy(for_player=player)[key]), np.asarray(value))

    def test_the_fog_attack_snapshot_survives(self):
        """A unit may not attack an enemy it found by moving; a reload must not lift that."""
        game = GameState(_map(), fog_of_war=True)
        warrior = game.place_unit("W", 4, 5, player=1)
        enemy = game.place_unit("W", 8, 5, player=2)
        assert not game.is_position_visible(8, 5, player=1)
        assert game.move_unit(warrior, 7, 5)
        assert game.is_position_visible(8, 5, player=1)
        assert enemy not in [a["target"] for a in game.get_legal_actions(1)["attack"]]

        loaded = _reload(game)
        loaded.update_visibility()  # as the GUI's Load Game used to; must change nothing now

        assert loaded.is_position_visible(8, 5, player=1)
        assert loaded.get_legal_actions(1)["attack"] == []
        assert loaded.units[0].visible_enemies_at_action_start == set()

    def test_an_ambushed_move_stays_uncancellable_after_a_reload(self):
        game = GameState(_map(12), fog_of_war=True)
        barbarian = game.place_unit("B", 4, 7, player=1)
        game.place_unit("W", 7, 7, player=2)
        assert game.move_unit(barbarian, 9, 7) and barbarian.ambushed

        loaded = _reload(game)

        unit = loaded.get_unit_at_position(6, 7)
        assert unit.ambushed and unit.has_moved
        assert loaded.cancel_move(unit) is False
        assert (unit.x, unit.y) == (6, 7)

    def test_the_save_names_its_format_version(self):
        assert GameState(_map()).to_dict()["save_format_version"] == SAVE_FORMAT_VERSION == 2


class TestNoSharedContainers:
    def test_from_dict_copies_its_inputs(self):
        game = GameState(_map())
        game.end_turn()
        data = _json(game.to_dict())
        del data["enabled_units"]

        loaded = GameState.from_dict(data)
        loaded.action_history.append({"type": "x"})
        loaded.player_configs.append({"type": "human"})
        loaded.enabled_units.remove("W")
        loaded.engine_overrides["starting_gold"] = 1

        assert len(data["action_history"]) == 1 and data["player_configs"] == []
        assert "engine_overrides" in data and data["engine_overrides"] == {}
        assert "W" in GameState.ALL_UNIT_TYPES

    def test_to_dict_shares_nothing_with_the_game(self):
        game = GameState(_map(), fog_of_war=True)
        game.end_turn()
        data = game.to_dict()

        data["action_history"].clear()
        data["enabled_units"].clear()
        data["player_gold"][1] = 0
        data["healing_totals"][1]["hp"] = 99

        assert len(game.action_history) == 1 and game.enabled_units
        assert game.player_gold[1] > 0 and game.healing_totals[1]["hp"] == 0


SHIPPED_SAVES = sorted((REPO_ROOT / "saves").glob("*.json"))


class TestOlderSavesStillLoad:
    @pytest.mark.parametrize("path", SHIPPED_SAVES, ids=lambda p: p.name)
    def test_every_shipped_save_loads(self, path):
        data = json.loads(path.read_text())
        # Shipped before versioning (format 1); a maintainer may re-save one later.
        assert data.get("save_format_version", 1) <= SAVE_FORMAT_VERSION
        data["map_file"] = str(REPO_ROOT / data["map_file"])

        game = restore_saved_game(data)

        assert len(game.units) == len(data["units"])
        assert all(0 < u.health <= u.max_health for u in game.units)
        assert game.get_legal_actions(game.current_player)["end_turn"]
        # A loaded old save re-saves in the current format and round-trips exactly.
        resaved = _json(game.to_dict())
        assert resaved["save_format_version"] == SAVE_FORMAT_VERSION
        assert _json(GameState.from_dict(resaved).to_dict()) == resaved

    def test_the_shipped_saves_are_all_covered(self):
        assert len(SHIPPED_SAVES) >= 8

    @staticmethod
    def _as_version_1(data):
        """Strip what format 2 added, leaving what the base commit wrote."""
        for key in (
            "save_format_version",
            "winning_action_index",
            "healing_totals",
            "fog_of_war_state",
        ):
            data.pop(key)
        for unit in data["units"]:
            unit.pop("has_moved")
            unit.pop("visible_enemies_at_action_start")
            unit.pop("ambushed")
        return data

    def test_a_version_1_fog_save_loads_with_fog_rebuilt_from_the_board(self):
        game = GameState(_map(), fog_of_war=True)
        game.place_unit("A", 7, 3, player=1)
        data = self._as_version_1(_json(game.to_dict()))

        loaded = GameState.from_dict(data)

        np.testing.assert_array_equal(loaded.visibility_maps[1].get_visible_mask(), game.visibility_maps[1].get_visible_mask())
        assert loaded.known_structure(1, 9, 9).owner == 2  # HQs known, as in a new game
        assert loaded.healing_totals == {1: {"hp": 0, "gold": 0}, 2: {"hp": 0, "gold": 0}}

    def test_a_version_1_save_infers_has_moved(self):
        game = GameState(_map())
        warrior = game.place_unit("W", 3, 3, player=1)
        assert game.move_unit(warrior, 3, 5)
        data = self._as_version_1(_json(game.to_dict()))

        assert GameState.from_dict(data).units[0].has_moved

    def test_a_newer_version_warns_and_loads(self, caplog):
        data = _json(GameState(_map()).to_dict())
        data["save_format_version"] = SAVE_FORMAT_VERSION + 1

        with caplog.at_level(logging.WARNING, logger="reinforcetactics.core.game_state"):
            GameState.from_dict(data)

        assert "newer" in caplog.text

    def test_a_non_numeric_version_warns_and_loads(self, caplog):
        """A hand-edited version must not crash the load (it raised TypeError)."""
        game = GameState(_map())
        game.place_unit("W", 5, 5, player=1)
        data = _json(game.to_dict())
        data["save_format_version"] = "2b"

        with caplog.at_level(logging.WARNING, logger="reinforcetactics.core.game_state"):
            loaded = GameState.from_dict(data)

        assert "not a number" in caplog.text
        assert len(loaded.units) == 1


# ---------------------------------------------------------------------------
# Round-trip property test (core-17)
# ---------------------------------------------------------------------------


def _play_to_mid_game(game, seed, full_turns, mid_turn_actions):
    """Seeded bots play ``full_turns`` turns, then the player to move takes a few random actions."""
    rng = random.Random(seed)
    bots = {}
    for player in range(1, game.num_players + 1):
        if game.num_players == 2 and player == 1:
            bots[player] = SimpleBot(game, player=player, rng=random.Random(seed + player))
        else:
            bots[player] = RandomBot(game, player=player, rng=random.Random(seed + player))
    for _ in range(full_turns):
        if game.game_over:
            return
        bots[game.current_player].take_turn()
    mover = RandomBot(game, player=game.current_player, rng=rng)
    for _ in range(mid_turn_actions):
        if game.game_over:
            return
        legal = game.get_legal_actions(game.current_player)
        options = [(key, a) for key in RandomBot._SAMPLE_ACTION_KEYS for a in legal.get(key, [])]
        if not options:
            return
        mover._execute(*rng.choice(options))


ROUND_TRIP_CASES = [
    (map_file, players, fog, seed)
    for map_file, players in (
        ("maps/1v1/crossroads.csv", 2),
        ("maps/1v1/center_mountains.csv", 2),
        ("maps/1v1v1/triangle_arena.csv", 3),
    )
    for fog in (False, True)
    for seed in (1, 2)
]


@pytest.mark.parametrize(("map_file", "players", "fog", "seed"), ROUND_TRIP_CASES)
def test_json_round_trip_is_exact_and_keeps_the_legal_actions(map_file, players, fog, seed):
    """to_dict -> JSON -> from_dict -> to_dict is identical, with equal legal actions for every player.

    No field is volatile: the timestamp is stored to the second and restored,
    and the fields left out on purpose (the engine RNG, caches, the UI-only
    ``Unit.selected``) never appear in ``to_dict``.
    """
    game = GameState(FileIO.load_map(str(REPO_ROOT / map_file)), num_players=players, fog_of_war=fog, max_turns=60)
    game.player_gold = dict.fromkeys(range(1, players + 1), 2000)
    _play_to_mid_game(game, seed, full_turns=10 * players + seed, mid_turn_actions=3 + seed)
    assert len(game.action_history) > 10 and game.units

    saved = _json(game.to_dict())
    loaded = GameState.from_dict(saved)

    assert _json(loaded.to_dict()) == saved
    for player in range(1, players + 1):
        assert _canon(loaded.get_legal_actions(player)) == _canon(game.get_legal_actions(player)), player

    # And the two games play on identically (action timestamps aside).
    for g in (game, loaded):
        g.rng = random.Random(seed)
        _play_to_mid_game(g, seed + 100, full_turns=players, mid_turn_actions=0)
    assert _actions(loaded) == _actions(game)
    assert len(loaded.action_history) > len(saved["action_history"])
