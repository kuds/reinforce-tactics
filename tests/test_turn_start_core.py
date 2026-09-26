"""Start-of-turn processing and Player 1's turn 0 (review core-13).

Income, structure healing, status/cooldown ticks and the fog update run in
``GameState._begin_turn`` for the player whose turn starts. ``end_turn`` runs
it for every turn except Player 1's first, which by default starts on the
starting gold alone (Player 2 collects income before its first move). That
choice is now the documented engine override ``begin_first_turn`` (default
False, the old schedule), recorded in saves and replay game_info.
"""

import json

import numpy as np
import pytest

from reinforcetactics.constants import BUILDING_INCOME, HEADQUARTERS_INCOME, UNIT_DATA
from reinforcetactics.core.game_state import GameState
from reinforcetactics.utils import replay_actions
from reinforcetactics.utils.replay_actions import execute_replay_action


def _grid():
    grid = np.full((8, 8), "p", dtype=object)
    grid[0, 0], grid[0, 1] = "h_1", "b_1"
    grid[7, 7], grid[7, 6] = "h_2", "b_2"
    return grid


INCOME = HEADQUARTERS_INCOME + BUILDING_INCOME


def test_default_player_1_starts_turn_0_without_income():
    game = GameState(_grid(), num_players=2)
    start = game.starting_gold

    assert game.begin_first_turn is False
    assert game.player_gold == {1: start, 2: start}
    game.end_turn()
    assert game.player_gold == {1: start, 2: start + INCOME}


def test_begin_first_turn_gives_player_1_turn_0_income():
    game = GameState(_grid(), num_players=2, engine_overrides={"begin_first_turn": True})
    start = game.starting_gold

    assert game.player_gold == {1: start + INCOME, 2: start}
    game.end_turn()
    assert game.player_gold == {1: start + INCOME, 2: start + INCOME}


def test_begin_first_turn_must_be_a_bool():
    with pytest.raises(ValueError, match="begin_first_turn"):
        GameState(_grid(), num_players=2, engine_overrides={"begin_first_turn": "false"})


def test_end_turn_runs_begin_turn_for_the_incoming_player():
    game = GameState(_grid(), num_players=2)
    unit = game.place_unit("W", 6, 7, 2)  # on player 2's building
    unit.health -= 4
    unit.haste_refreshed = True
    game.end_turn()

    assert unit.can_move and unit.can_attack and not unit.haste_refreshed
    assert unit.health == unit.max_health - 2  # healed 2 on its building
    assert game.healing_totals[2]["hp"] == 2


def test_save_round_trip_does_not_pay_turn_0_income_twice():
    game = GameState(_grid(), num_players=2, engine_overrides={"begin_first_turn": True})

    restored = GameState.from_dict(json.loads(json.dumps(game.to_dict())))

    assert restored.begin_first_turn is True
    assert restored.player_gold == game.player_gold


def test_replay_game_info_records_it_and_the_replay_reproduces_it(monkeypatch):
    """With no starting gold, Player 1's turn-0 build is paid by the turn-0 income."""
    from reinforcetactics.utils import file_io

    overrides = {"begin_first_turn": True, "starting_gold": 0}
    game = GameState(_grid(), num_players=2, engine_overrides=overrides)
    assert game.create_unit("W", 1, 0) is not None
    assert game.player_gold[1] == INCOME - UNIT_DATA["W"]["cost"]
    captured = {}
    monkeypatch.setattr(
        file_io.FileIO, "save_replay", staticmethod(lambda actions, info, path=None: captured.update(info) or "x")
    )
    game.save_replay_to_file()

    assert captured["begin_first_turn"] is True
    assert captured["engine_overrides"] == overrides
    replay = GameState(_grid(), **replay_actions.replay_game_state_kwargs(captured))
    for action in game.action_history:
        execute_replay_action(replay, action, lambda x, y: (x, y), schema_version=3)
    assert [(u.type, u.x, u.y) for u in replay.units] == [("W", 1, 0)]
    assert replay.player_gold == game.player_gold
