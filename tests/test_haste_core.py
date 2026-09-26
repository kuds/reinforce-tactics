"""Haste is an engine rule: one extra full action, the same on every path (review core-8).

The extra action used to be granted only by ``end_unit_turn``, which the GUI
and the rule bots call after an action and the RL env, MCTS and LLM bots never
do. So a hasted unit acted twice for a bot, once for an RL agent, and three
times in the GUI when it was hasted after acting. Now the engine grants it as
the hasted unit's action is spent (``GameState._consume_action``).
"""

import numpy as np
import pytest

from reinforcetactics.app.action_executor import execute_unit_action
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.replay_actions import execute_replay_action


@pytest.fixture
def game():
    grid = np.full((10, 10), "p", dtype=object)
    grid[0, 0], grid[9, 9] = "h_1", "h_2"
    return GameState(grid, num_players=2)


def _setup(game):
    """A Sorcerer, an ally Warrior next to it and a sturdy enemy two tiles east of the Warrior."""
    sorcerer = game.place_unit("S", 2, 4, 1)
    warrior = game.place_unit("W", 3, 4, 1)
    enemy = game.place_unit("B", 5, 4, 2)
    enemy.health = enemy.max_health = 100  # survives several hits
    return sorcerer, warrior, enemy


def _unit_actions(game, unit):
    """Recorded actions ``unit`` performed (as actor)."""
    return [
        a["type"]
        for a in game.action_history
        if a.get("actor_unit_id") == unit.unit_id or a.get("attacker_unit_id") == unit.unit_id
    ]


class TestEnginePath:
    def test_a_fresh_hasted_unit_gets_exactly_one_extra_action(self, game):
        sorcerer, warrior, enemy = _setup(game)
        assert game.haste(sorcerer, warrior)

        assert game.move_unit(warrior, 4, 4)
        assert game.attack(warrior, enemy)["damage"] > 0
        # The extra action: a fresh move and a fresh attack, without any
        # end_unit_turn from the caller (the RL env / MCTS / LLM path).
        assert warrior.can_move and warrior.can_attack and not warrior.is_hasted
        assert game.move_unit(warrior, 5, 3)
        assert game.attack(warrior, enemy)["damage"] > 0

        assert not warrior.can_move and not warrior.can_attack
        assert game.attack(warrior, enemy)["damage"] == 0  # no third action
        legal = game.get_legal_actions(1)
        assert not any(a["attacker"] is warrior for a in legal["attack"])
        assert _unit_actions(game, warrior) == ["move", "attack", "move", "attack"]

    def test_hasting_a_unit_that_already_acted_refreshes_it_once(self, game):
        sorcerer, warrior, enemy = _setup(game)
        game.move_unit(warrior, 4, 4)
        game.attack(warrior, enemy)
        assert not (warrior.can_move or warrior.can_attack)

        assert game.haste(sorcerer, warrior)
        assert warrior.can_move and warrior.can_attack and not warrior.is_hasted

        game.attack(warrior, enemy)
        # The GUI's end_unit_turn after the action used to find is_hasted
        # still set and hand out a third action.
        assert game.end_unit_turn(warrior) is False
        assert not (warrior.can_move or warrior.can_attack)
        assert _unit_actions(game, warrior) == ["move", "attack", "attack"]

    def test_waiting_with_a_unit_hasted_after_it_acted_ends_its_extra_action(self, game):
        """The GUI's Wait on the refreshed target used to be swallowed as if it had just acted."""
        sorcerer, warrior, enemy = _setup(game)
        game.move_unit(warrior, 4, 4)
        game.attack(warrior, enemy)
        assert game.haste(sorcerer, warrior)

        assert game.end_unit_turn(warrior) is False
        assert not (warrior.can_move or warrior.can_attack)

    def test_end_unit_turn_right_after_a_refreshed_action_keeps_the_extra_action(self, game):
        sorcerer, warrior, enemy = _setup(game)
        game.haste(sorcerer, warrior)
        game.move_unit(warrior, 4, 4)
        game.attack(warrior, enemy)

        assert game.end_unit_turn(warrior) is True
        assert warrior.can_move and warrior.can_attack

        # Called again without acting, it is a Wait: the extra action ends.
        assert game.end_unit_turn(warrior) is False
        assert not (warrior.can_move or warrior.can_attack)

    def test_waiting_spends_the_first_action(self, game):
        """GUI Wait after a move: the haste refreshes the unit for a second move."""
        sorcerer, warrior, _enemy = _setup(game)
        game.haste(sorcerer, warrior)
        game.move_unit(warrior, 3, 5)

        assert game.end_unit_turn(warrior) is True
        assert game.move_unit(warrior, 3, 7)
        assert game.end_unit_turn(warrior) is False

    def test_a_unit_without_haste_is_done_after_acting(self, game):
        _sorcerer, warrior, enemy = _setup(game)
        game.move_unit(warrior, 4, 4)
        game.attack(warrior, enemy)

        assert not (warrior.can_move or warrior.can_attack)
        assert game.end_unit_turn(warrior) is False

    def test_paralyzed_allies_cannot_be_hasted(self, game):
        sorcerer, warrior, _enemy = _setup(game)
        warrior.paralyzed_turns = 2

        assert not any(a["target"] is warrior for a in game.get_legal_actions(1)["haste"])
        assert not game.haste(sorcerer, warrior)
        assert not warrior.is_hasted
        assert not game.move_unit(warrior, 3, 5)


class TestCallerPaths:
    def test_gui_menu_attack_keeps_a_hasted_unit_selected_once(self, game):
        sorcerer, warrior, enemy = _setup(game)
        game.haste(sorcerer, warrior)
        game.move_unit(warrior, 4, 4)
        selected = [None]

        _mode, _action, still = execute_unit_action(game, {"type": "attack", "targets": [enemy]}, warrior, selected)
        assert still is warrior and selected[0] is warrior

        _mode, _action, still = execute_unit_action(game, {"type": "attack", "targets": [enemy]}, warrior, selected)
        assert still is None and selected[0] is None
        assert _unit_actions(game, warrior) == ["move", "attack", "attack"]

    def test_scripted_bot_uses_the_extra_action(self, game):
        sorcerer, warrior, enemy = _setup(game)
        game.haste(sorcerer, warrior)
        bot = SimpleBot(game, player=1)

        bot.act_with_unit(warrior)

        assert _unit_actions(game, warrior).count("attack") == 2
        assert not (warrior.can_move or warrior.can_attack)

    def test_gym_env_path_gets_the_extra_action(self, game):
        """The env dispatches through the same engine calls; it never calls end_unit_turn."""
        from reinforcetactics.rl.gym_env import StrategyGameEnv

        env = StrategyGameEnv.__new__(StrategyGameEnv)  # only execute_game_action is exercised
        env.game_state = game
        sorcerer, warrior, enemy = _setup(game)
        game.haste(sorcerer, warrior)

        def step(action_type, to_pos):
            action = {"action_type": action_type, "unit_type": "W", "from_pos": (warrior.x, warrior.y), "to_pos": to_pos}
            return env.execute_game_action(action, 1)[1]

        assert step(1, (4, 4)) and step(2, (enemy.x, enemy.y))
        assert step(1, (5, 3)) and step(2, (enemy.x, enemy.y))
        assert not step(2, (enemy.x, enemy.y))
        assert _unit_actions(game, warrior) == ["move", "attack", "move", "attack"]


def test_replay_reproduces_haste(game):
    sorcerer, warrior, enemy = _setup(game)
    game.haste(sorcerer, warrior)
    game.move_unit(warrior, 4, 4)
    game.attack(warrior, enemy)
    game.move_unit(warrior, 5, 3)
    game.attack(warrior, enemy)
    game.end_turn()

    grid = np.full((10, 10), "p", dtype=object)
    grid[0, 0], grid[9, 9] = "h_1", "h_2"
    replay = GameState(grid, num_players=2)
    _setup(replay)  # same units, same ids
    for action in game.action_history:
        execute_replay_action(replay, action, lambda x, y: (x, y), schema_version=3)

    assert sorted((u.type, u.x, u.y, u.health) for u in replay.units) == sorted(
        (u.type, u.x, u.y, u.health) for u in game.units
    )
    assert (warrior.x, warrior.y) == (5, 3)
