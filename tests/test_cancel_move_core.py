"""cancel_move takes the move back completely (review core-9).

It used to reset only the unit's position: the recorded ``move`` stayed in the
action history (replays moved the unit the live game had put back) and, under
fog of war, everything the move revealed stayed visible, so a player could
scout for free by moving and cancelling.
"""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.utils.replay_actions import execute_replay_action


def _grid(size=20):
    grid = np.full((size, size), "p", dtype=object)
    grid[0, 0], grid[size - 1, size - 1] = "h_1", "h_2"
    return grid


@pytest.fixture
def game():
    return GameState(_grid(10), num_players=2)


class TestHistory:
    def test_cancelling_the_latest_move_removes_its_record(self, game):
        unit = game.place_unit("W", 2, 2, 1)
        game.move_unit(unit, 2, 4)

        assert game.cancel_move(unit)

        assert (unit.x, unit.y) == (2, 2) and unit.can_move and unit.can_attack
        assert game.action_history == []
        assert any(a["unit"] is unit and (a["to_x"], a["to_y"]) == (2, 4) for a in game.get_legal_actions(1)["move"])

    def test_cancelling_an_earlier_move_is_recorded_and_replayed(self, game):
        first = game.place_unit("W", 2, 2, 1)
        second = game.place_unit("W", 5, 5, 1)
        game.move_unit(first, 2, 4)
        game.move_unit(second, 6, 6)

        assert game.cancel_move(first)

        assert [a["type"] for a in game.action_history] == ["move", "move", "cancel_move"]
        replay = GameState(_grid(10), num_players=2)
        replay.place_unit("W", 2, 2, 1)
        replay.place_unit("W", 5, 5, 1)
        for action in game.action_history:
            execute_replay_action(replay, action, lambda x, y: (x, y), schema_version=3)
        assert sorted((u.x, u.y) for u in replay.units) == [(2, 2), (6, 6)]


class TestWhenACancelIsRefused:
    def test_not_after_the_unit_acted(self, game):
        unit = game.place_unit("W", 2, 2, 1)
        enemy = game.place_unit("W", 2, 5, 2)
        enemy.health = 100
        game.move_unit(unit, 2, 4)
        game.attack(unit, enemy)

        assert not game.cancel_move(unit)
        assert (unit.x, unit.y) == (2, 4)

    def test_not_onto_a_tile_someone_else_took(self, game):
        unit = game.place_unit("W", 2, 2, 1)
        other = game.place_unit("W", 3, 2, 1)
        game.move_unit(unit, 2, 4)
        game.move_unit(other, 2, 2)

        assert not game.cancel_move(unit)
        assert (unit.x, unit.y) == (2, 4)

    def test_not_on_another_players_turn(self, game):
        unit = game.place_unit("W", 2, 2, 1)
        game.move_unit(unit, 2, 4)
        game.current_player = 2

        assert not game.cancel_move(unit)


class TestFogOfWar:
    def test_a_cancelled_move_reveals_nothing(self):
        game = GameState(_grid(20), num_players=2, fog_of_war=True)
        scout = game.place_unit("W", 2, 2, 1)
        enemy = game.place_unit("W", 2, 8, 2)
        game.update_visibility()
        vis = game.visibility_maps[1]
        assert not game.is_position_visible(2, 8, 1)
        explored_before = vis.state.copy()

        game.move_unit(scout, 2, 5)
        assert game.is_position_visible(enemy.x, enemy.y, 1)

        assert game.cancel_move(scout)

        assert not game.is_position_visible(enemy.x, enemy.y, 1)
        assert not game.is_position_explored(enemy.x, enemy.y, 1)
        assert (game.visibility_maps[1].state == explored_before).all()
        assert (enemy.x, enemy.y) not in game.visibility_maps[1].last_seen_units
        # Selecting the unit again (the GUI's FOW snapshot) must not see it either.
        game.capture_visible_enemies_for_unit(scout)
        assert scout.visible_enemies_at_action_start == set()

    def test_only_the_latest_move_can_be_cancelled_under_fog(self):
        """Scout with one unit, strike with another, cancel the scout: refused."""
        game = GameState(_grid(20), num_players=2, fog_of_war=True)
        scout = game.place_unit("W", 2, 2, 1)
        striker = game.place_unit("W", 4, 2, 1)
        game.update_visibility()
        game.move_unit(scout, 2, 5)
        game.move_unit(striker, 4, 4)

        assert not game.cancel_move(scout)
        assert (scout.x, scout.y) == (2, 5)
        assert game.cancel_move(striker)  # the latest move still can be
