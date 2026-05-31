"""Tests for the engine_overrides sparse overlay (balance-as-config).

Covers the Phase-1 injection points: GameState resolution, Unit stat
chokepoint, mechanics income, the EnvConfig field + round-trip, and the
run_config provenance snapshot.
"""

import numpy as np
import pytest

from reinforcetactics import constants as C
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.unit import Unit
from reinforcetactics.game.mechanics import GameMechanics


def _map():
    return np.array([["p" for _ in range(8)] for _ in range(8)], dtype=object)


# --- GameState resolution ------------------------------------------------


def test_no_overrides_matches_module_constants():
    gs = GameState(_map(), num_players=2)
    assert gs.starting_gold == C.STARTING_GOLD
    assert gs.player_gold[1] == C.STARTING_GOLD
    assert gs.income_rates["headquarters"] == C.HEADQUARTERS_INCOME
    assert gs.income_rates["building"] == C.BUILDING_INCOME
    assert gs.income_rates["tower"] == C.TOWER_INCOME
    assert gs.unit_data["K"]["defence"] == C.UNIT_DATA["K"]["defence"]


def test_overrides_applied():
    ov = {
        "starting_gold": 999,
        "headquarters_income": 7,
        "tower_income": 3,
        "unit_data": {"K": {"defence": 5}, "W": {"attack": 10}},
    }
    gs = GameState(_map(), num_players=2, engine_overrides=ov)
    assert gs.starting_gold == 999
    assert gs.player_gold[1] == 999 and gs.player_gold[2] == 999
    assert gs.income_rates["headquarters"] == 7
    assert gs.income_rates["tower"] == 3
    assert gs.income_rates["building"] == C.BUILDING_INCOME  # untouched
    assert gs.unit_data["K"]["defence"] == 5
    assert gs.unit_data["W"]["attack"] == 10


def test_module_constants_not_mutated():
    before_k = C.UNIT_DATA["K"]["defence"]
    before_w = C.UNIT_DATA["W"]["attack"]
    GameState(_map(), engine_overrides={"unit_data": {"K": {"defence": before_k + 13}}})
    assert C.UNIT_DATA["K"]["defence"] == before_k
    assert C.UNIT_DATA["W"]["attack"] == before_w
    # A second default game still sees pristine values (no cross-leak).
    gs = GameState(_map())
    assert gs.unit_data["K"]["defence"] == before_k


@pytest.mark.parametrize(
    "bad",
    [
        {"unit_data": {"ZZ": {"attack": 1}}},
        {"unit_data": {"W": {"not_a_field": 1}}},
    ],
)
def test_unknown_code_or_field_raises(bad):
    with pytest.raises((KeyError, ValueError)):
        GameState(_map(), engine_overrides=bad)


def test_create_unit_uses_overridden_cost_and_stats():
    gs = GameState(_map(), engine_overrides={"unit_data": {"W": {"cost": 5, "health": 99}}})
    gs.player_gold[1] = 10
    u = gs.create_unit("W", 1, 1, 1)
    assert u is not None
    assert u.max_health == 99
    assert u.health == 99
    assert gs.player_gold[1] == 5  # only the overridden cost deducted


# --- Unit chokepoint -----------------------------------------------------


def test_unit_default_stats_unchanged():
    u = Unit("A", 0, 0, 1)
    assert u.defence == C.UNIT_DATA["A"]["defence"]
    assert u.max_health == C.UNIT_DATA["A"]["health"]


def test_unit_explicit_stats_override():
    spec = dict(C.UNIT_DATA["A"])
    spec["defence"] = C.UNIT_DATA["A"]["defence"] + 5
    u = Unit("A", 0, 0, 1, stats=spec)
    assert u.defence == C.UNIT_DATA["A"]["defence"] + 5


# --- mechanics income ----------------------------------------------------


def test_calculate_income_default_vs_override():
    gs = GameState(_map())
    grid = gs.grid
    base = GameMechanics.calculate_income(1, grid)
    rated = GameMechanics.calculate_income(1, grid, {"headquarters": 1, "building": 1, "tower": 1})
    assert base["headquarters"] == rated["headquarters"]  # counts unchanged
    # total uses the supplied rates
    assert rated["total"] == (rated["headquarters"] + rated["buildings"] + rated["towers"])


# --- EnvConfig field + round-trip ---------------------------------------


def test_envconfig_has_engine_overrides_field():
    from reinforcetactics.rl.config import EnvConfig

    assert EnvConfig().engine_overrides is None
    ec = EnvConfig(engine_overrides={"starting_gold": 250})
    assert ec.engine_overrides == {"starting_gold": 250}


def test_config_round_trips_engine_overrides(tmp_path):
    import yaml

    from reinforcetactics.rl.config import load_config

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "map_file": "maps/1v1/beginner.csv",
                    "engine_overrides": {
                        "starting_gold": 250,
                        "unit_data": {"K": {"defence": 5}},
                    },
                }
            }
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.env.engine_overrides["starting_gold"] == 250
    assert cfg.env.engine_overrides["unit_data"]["K"]["defence"] == 5


# --- run_config provenance ----------------------------------------------


def test_run_config_snapshots_overrides_and_effective_economy():
    from reinforcetactics.utils.run_config import build_run_config

    ov = {"starting_gold": 777, "unit_data": {"W": {"attack": 42}}}
    rc = build_run_config(
        run_type="test",
        map_file=None,
        opponent="random",
        hyperparams={},
        env_config={"engine_overrides": ov},
    )
    meta = rc["meta"]
    assert meta["engine_overrides"] == ov
    eff = meta["effective_engine_economy"]
    assert eff["starting_gold"] == 777
    assert eff["unit_data"]["W"]["attack"] == 42
    # effective hash diverges from the defaults hash when an overlay is set
    assert meta["effective_balance_profile_hash"] != meta["balance_profile_hash"]
    assert meta["engine_constants_hash"] is not None


def test_run_config_no_overrides_effective_equals_defaults():
    from reinforcetactics.utils.run_config import build_run_config

    rc = build_run_config(
        run_type="test",
        map_file=None,
        opponent="random",
        hyperparams={},
        env_config={},
    )
    meta = rc["meta"]
    assert meta["engine_overrides"] is None
    assert meta["effective_engine_economy"] == meta["engine_economy"]
    assert meta["effective_balance_profile_hash"] == meta["balance_profile_hash"]


# --- gym_env end-to-end --------------------------------------------------


def test_gym_env_threads_overrides_into_game_state():
    from reinforcetactics.rl.gym_env import StrategyGameEnv

    ov = {"starting_gold": 250, "unit_data": {"K": {"defence": 5}}}
    env = StrategyGameEnv(
        map_file="maps/1v1/beginner.csv",
        opponent="random",
        action_space_type="flat_discrete",
        engine_overrides=ov,
    )
    assert env.game_state.starting_gold == 250
    assert env.game_state.unit_data["K"]["defence"] == 5
    # survives reset (overlay re-applied on the re-created GameState)
    env.reset(seed=0)
    assert env.game_state.starting_gold == 250
    assert env.game_state.unit_data["K"]["defence"] == 5


# --- structure-health overrides (capture-difficulty lever) ---------------


def _struct_map():
    """3x3 map with one HQ (p1), one building (p1), one neutral tower."""
    return np.array(
        [
            ["h_1", "b_1", "t"],
            ["p", "p", "p"],
            ["p", "p", "p"],
        ],
        dtype=object,
    )


def _find(gs, code):
    return [t for row in gs.grid.tiles for t in row if t.type == code]


def test_structure_health_default_unchanged():
    gs = GameState(_struct_map(), num_players=2)
    assert _find(gs, "h")[0].max_health == C.HEADQUARTERS_MAX_HEALTH
    assert _find(gs, "b")[0].max_health == C.BUILDING_MAX_HEALTH
    assert _find(gs, "t")[0].max_health == C.TOWER_MAX_HEALTH


def test_structure_health_override_applied():
    gs = GameState(
        _struct_map(),
        num_players=2,
        engine_overrides={"headquarters_health": 30, "tower_health": 20},
    )
    hq, twr, bld = _find(gs, "h")[0], _find(gs, "t")[0], _find(gs, "b")[0]
    assert (hq.health, hq.max_health) == (30, 30)
    assert (twr.health, twr.max_health) == (20, 20)
    # absent key keeps the constants.py default
    assert bld.max_health == C.BUILDING_MAX_HEALTH


@pytest.mark.parametrize("bad", [{"headquarters_health": 0}, {"tower_health": -5}, {"building_health": 0}])
def test_structure_health_rejects_non_positive(bad):
    with pytest.raises(ValueError):
        GameState(_struct_map(), num_players=2, engine_overrides=bad)


def test_structure_health_override_changes_capture_turns():
    # HQ@30 -> a Warrior (15 HP) captures in 2 seizes vs 4 at the default 50.
    gs = GameState(_struct_map(), num_players=2, engine_overrides={"headquarters_health": 30})
    hq = _find(gs, "h")[0]
    hq.player = 2
    u = Unit("W", hq.x, hq.y, 1)
    seizes = 0
    while seizes <= 9:
        result = GameMechanics.seize_structure(u, hq)
        seizes += 1
        if result["captured"]:
            break
    assert seizes == 2


def test_structure_health_regen_scales_off_override():
    # Regen is 50% of max_health; with HQ@30 a regen tick adds int(30*0.5)=15.
    gs = GameState(_struct_map(), num_players=2, engine_overrides={"headquarters_health": 30})
    hq = _find(gs, "h")[0]
    hq.health = 5
    hq.regenerating = True
    GameMechanics.regenerate_structures(gs.grid, [])
    assert hq.health == 20  # 5 + int(30 * 0.5)
