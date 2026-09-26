"""``EngineConfig``: a game's rules as one frozen object (review core-16).

The engine_overrides overlay used to be resolved by GameState's static
``_resolve_*`` helpers straight into a dozen GameState attributes. It is now
resolved once into an ``EngineConfig`` that can be built and tested without
a map; GameState keeps every old attribute name as a read-only view of it.
The refactor must not change what an overlay is accepted with (the error
types and messages below are the ones GameState raised before), what saves
and replays record, or that an in-place edit of ``rules.UNIT_DATA`` reaches
games created afterwards.
"""

import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics import rules
from reinforcetactics.core.engine_config import ENGINE_OVERRIDE_KEYS, EngineConfig
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.terrain_rules import TerrainRules
from reinforcetactics.utils.file_io import FileIO

# Every key the engine accepts, with a non-default value.
FULL_OVERLAY = {
    "starting_gold": 1800,
    "headquarters_income": 180,
    "building_income": 90,
    "tower_income": 60,
    "tower_health": 20,
    "building_health": 25,
    "headquarters_health": 30,
    "damage_model": "hp_scaled",
    "max_units_per_player": 6,
    "unit_data": {"W": {"attack": 8, "cost": 150}, "K": {"defence": 5}},
    "begin_first_turn": True,
    "legacy_end_rules": True,
    "terrain_move_cost": {"r": 0.5, "f": 2},
    "charge_distance": "path",
    "forest_concealment": True,
    "hq_always_visible": True,
}


def _map():
    grid = np.array([["p"] * 8 for _ in range(8)], dtype=object)
    grid[0][0], grid[7][7], grid[3][3], grid[4][4] = "h_1", "h_2", "t", "b"
    return grid


class TestResolution:
    def test_no_overrides_is_the_shipped_game(self):
        for overrides in (None, {}):
            config = EngineConfig.from_overrides(overrides)
            assert config.overrides == {}
            assert config.unit_data == rules.UNIT_DATA and config.unit_data is not rules.UNIT_DATA
            assert config.income_rates == {
                "headquarters": rules.HEADQUARTERS_INCOME,
                "building": rules.BUILDING_INCOME,
                "tower": rules.TOWER_INCOME,
            }
            assert config.starting_gold == rules.STARTING_GOLD
            assert config.damage_model == "flat"
            assert config.structure_health == {}
            assert config.max_units_per_player == rules.MAX_UNITS_PER_PLAYER
            assert config.terrain_rules == TerrainRules()
            assert config.begin_first_turn is False and config.legacy_end_rules is False

    def test_every_key_resolves(self):
        config = EngineConfig.from_overrides(FULL_OVERLAY)
        assert set(FULL_OVERLAY) == ENGINE_OVERRIDE_KEYS  # the overlay above covers every key
        assert config.unit_data["W"]["attack"] == 8 and config.unit_data["W"]["cost"] == 150
        assert config.unit_data["K"]["defence"] == 5
        assert config.unit_data["M"] == rules.UNIT_DATA["M"]
        assert config.income_rates == {"headquarters": 180, "building": 90, "tower": 60}
        assert config.starting_gold == 1800
        assert config.damage_model == "hp_scaled"
        assert config.structure_health == {"t": 20, "b": 25, "h": 30}
        assert config.max_units_per_player == 6
        assert config.terrain_rules == TerrainRules.from_overrides(FULL_OVERLAY)
        assert config.begin_first_turn is True and config.legacy_end_rules is True

    def test_the_overlay_is_copied_and_the_module_table_untouched(self):
        overlay = {"unit_data": {"W": {"attack": 1}}}
        config = EngineConfig.from_overrides(overlay)
        overlay["starting_gold"] = 5
        assert "starting_gold" not in config.overrides
        assert rules.UNIT_DATA["W"]["attack"] != 1

    def test_it_is_frozen(self):
        config = EngineConfig.from_overrides(None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.starting_gold = 1

    def test_structure_health_goes_onto_structures_only(self):
        gs = GameState(_map(), engine_overrides={"headquarters_health": 30, "tower_health": 20})
        tiles = {t.type: t for row in gs.grid.tiles for t in row}
        assert (tiles["h"].max_health, tiles["h"].health) == (30, 30)
        assert (tiles["t"].max_health, tiles["t"].health) == (20, 20)
        assert tiles["b"].max_health == GameState(_map()).grid.tiles[4][4].max_health


# The errors GameState raised before EngineConfig existed: same type, same
# message, and for an overlay with several problems the same one first.
_VALID = sorted(ENGINE_OVERRIDE_KEYS)
BAD_OVERLAYS = [
    ({"starting_gld": 1}, KeyError, f"engine_overrides: unknown key(s) ['starting_gld'] (valid: {_VALID})"),
    ({"starting_gold": 1, "zzz": 2, "aaa": 3}, KeyError, f"engine_overrides: unknown key(s) ['aaa', 'zzz'] (valid: {_VALID})"),
    ({"unit_data": {"ZZ": {"attack": 1}}}, KeyError, "engine_overrides.unit_data: unknown unit code 'ZZ'"),
    (
        {"unit_data": {"W": {"not_a_field": 1}}},
        ValueError,
        "engine_overrides.unit_data['W']: unknown stat field 'not_a_field' "
        "(valid: ['attack', 'cost', 'defence', 'health', 'movement', 'name'])",
    ),
    (
        {"damage_model": "quadratic"},
        ValueError,
        "engine_overrides.damage_model must be 'flat' or 'hp_scaled', got 'quadratic'",
    ),
    ({"max_units_per_player": 0}, ValueError, "engine_overrides.max_units_per_player must be a positive int, got 0"),
    ({"tower_health": 0}, ValueError, "engine_overrides.tower_health must be a positive int, got 0"),
    (
        {"headquarters_health": -1, "tower_health": 0},
        ValueError,
        "engine_overrides.tower_health must be a positive int, got 0",
    ),
    ({"begin_first_turn": "false"}, ValueError, "engine_overrides.begin_first_turn must be a bool, got 'false'"),
    ({"legacy_end_rules": 1}, ValueError, "engine_overrides.legacy_end_rules must be a bool, got 1"),
    (
        {"terrain_move_cost": {"w": 2}},
        KeyError,
        "engine_overrides.terrain_move_cost: 'w' is not a walkable tile code (valid: ['p', 'm', 'f', 'r', 'b', 'h', 't'])",
    ),
    (
        {"charge_distance": "teleport"},
        ValueError,
        "engine_overrides.charge_distance must be one of ('displacement', 'path'), got 'teleport'",
    ),
    ({"forest_concealment": "yes"}, ValueError, "engine_overrides.forest_concealment must be true or false, got 'yes'"),
    ({"starting_gold": "abc"}, ValueError, "invalid literal for int() with base 10: 'abc'"),
    (
        {"damage_model": "bad", "max_units_per_player": 0, "unit_data": {"ZZ": {}}},
        KeyError,
        "engine_overrides.unit_data: unknown unit code 'ZZ'",
    ),
    (
        {"max_units_per_player": 0, "charge_distance": "x", "begin_first_turn": 1},
        ValueError,
        "engine_overrides.max_units_per_player must be a positive int, got 0",
    ),
]


class TestValidation:
    @pytest.mark.parametrize("overlay, error, message", BAD_OVERLAYS)
    def test_bad_overlays_fail_as_they_always_did(self, overlay, error, message):
        with pytest.raises(error) as from_config:
            EngineConfig.from_overrides(overlay)
        assert from_config.type is error and from_config.value.args[0] == message
        # GameState resolves its overrides through EngineConfig
        with pytest.raises(error) as from_game:
            GameState(_map(), engine_overrides=overlay)
        assert from_game.value.args == from_config.value.args

    def test_the_game_keeps_the_accepted_key_set(self):
        assert GameState.ENGINE_OVERRIDE_KEYS is ENGINE_OVERRIDE_KEYS


class TestRoundTrip:
    @pytest.mark.parametrize(
        "overlay",
        [
            None,
            {},
            FULL_OVERLAY,
            # Recorded as given, not normalised: defaults, all-1 move costs
            # and a numeric string survive the trip unchanged.
            {"begin_first_turn": False, "terrain_move_cost": {"r": 1}, "starting_gold": "250"},
        ],
    )
    def test_to_overrides_gives_back_the_sparse_overlay(self, overlay):
        config = EngineConfig.from_overrides(overlay)
        sparse = config.to_overrides()
        assert sparse == (overlay or {})
        assert EngineConfig.from_overrides(sparse) == config
        assert EngineConfig.from_overrides(json.loads(json.dumps(sparse))) == config

    def test_to_overrides_shares_nothing(self):
        config = EngineConfig.from_overrides(FULL_OVERLAY)
        sparse = config.to_overrides()
        sparse["unit_data"]["W"]["attack"] = 99
        sparse["starting_gold"] = 1
        assert config.overrides == FULL_OVERLAY

    def test_saves_and_replays_record_the_overlay(self, tmp_path):
        overlay = {"starting_gold": 2500, "unit_data": {"K": {"defence": 5}}, "begin_first_turn": True}
        gs = GameState(_map(), engine_overrides=overlay)
        assert gs.to_dict()["engine_overrides"] == overlay == gs.engine_config.to_overrides()

        replay_path = gs.save_replay_to_file(str(tmp_path / "replay.json"))
        assert json.loads(Path(replay_path).read_text())["game_info"]["engine_overrides"] == overlay

        restored = GameState.from_dict(json.loads(json.dumps(gs.to_dict())))
        assert restored.engine_config == gs.engine_config


class TestModuleUnitData:
    def test_an_in_place_edit_reaches_a_new_default_game(self, monkeypatch):
        monkeypatch.setitem(rules.UNIT_DATA["W"], "attack", 42)
        gs = GameState(_map())
        assert gs.unit_data["W"]["attack"] == 42
        assert gs.place_unit("W", 2, 2, player=1).attack_data == 42

    def test_a_game_keeps_the_table_it_was_created_with(self, monkeypatch):
        gs = GameState(_map())
        before = copy.deepcopy(gs.unit_data)
        monkeypatch.setitem(rules.UNIT_DATA["W"], "attack", 42)
        assert gs.unit_data == before
        gs.unit_data["K"]["defence"] = 1  # a game's own table never reaches the module
        assert rules.UNIT_DATA["K"]["defence"] != 1


class TestGameStateViews:
    def test_old_names_read_the_config(self):
        gs = GameState(_map(), engine_overrides=FULL_OVERLAY)
        config = gs.engine_config
        assert gs.engine_overrides is config.overrides
        assert gs.unit_data is config.unit_data
        assert gs.income_rates is config.income_rates
        assert gs.structure_health is config.structure_health
        assert gs.terrain_rules is config.terrain_rules
        assert gs.starting_gold == config.starting_gold == 1800
        assert gs.damage_model == "hp_scaled"
        assert gs.max_units_per_player == 6
        assert gs.begin_first_turn is True and gs.legacy_end_rules is True

    @pytest.mark.parametrize(
        "name",
        [
            "engine_overrides",
            "unit_data",
            "income_rates",
            "starting_gold",
            "damage_model",
            "structure_health",
            "max_units_per_player",
            "terrain_rules",
            "begin_first_turn",
            "legacy_end_rules",
        ],
    )
    def test_old_names_are_read_only(self, name):
        gs = GameState(_map())
        with pytest.raises(AttributeError):
            setattr(gs, name, getattr(gs, name))

    def test_reset_keeps_the_rules(self):
        gs = GameState(_map(), engine_overrides=FULL_OVERLAY)
        before = gs.engine_config
        gs.reset(_map())
        assert gs.engine_config == before


class TestSearchClones:
    def test_clones_share_the_config(self):
        gs = GameState(FileIO.load_map("maps/1v1/crossroads.csv"), engine_overrides=FULL_OVERLAY, seed=3)
        clone = gs.clone_for_search()
        assert clone.engine_config is gs.engine_config
        assert clone.unit_data is gs.unit_data and clone.terrain_rules is gs.terrain_rules
        assert clone.clone_for_search().engine_config is gs.engine_config

    def test_a_deep_copy_gets_its_own_equal_config(self):
        gs = GameState(_map(), engine_overrides=FULL_OVERLAY)
        deep = copy.deepcopy(gs)
        assert deep.engine_config == gs.engine_config and deep.engine_config is not gs.engine_config
        assert deep.unit_data is not gs.unit_data
