"""
Tests for NoopBot and the ``opponent="noop"`` env wiring.

NoopBot is a stationary opponent that takes no actions and immediately
ends its turn. It's used as a curriculum stage-0 / sanity check for RL
agents — if the agent can't beat a do-nothing opponent, the issue is in
the policy or reward signal, not opponent strength.
"""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import NoopBot
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.utils.file_io import FileIO

END_TURN = np.array([5, 0, 0, 0, 0, 0])


@pytest.fixture
def map_data():
    np.random.seed(42)
    md = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()
    return md


class TestNoopBotBasics:
    def test_take_turn_ends_turn(self, map_data):
        """NoopBot.take_turn() must hand control back to the other player."""
        gs = GameState(map_data, num_players=2)
        gs.current_player = 2
        bot = NoopBot(gs, player=2)

        bot.take_turn()

        assert gs.current_player != 2 or gs.game_over

    def test_take_turn_creates_no_units(self, map_data):
        """NoopBot must never spawn units — that's the whole point."""
        gs = GameState(map_data, num_players=2)
        gs.current_player = 2
        before = len([u for u in gs.units if u.player == 2])

        NoopBot(gs, player=2).take_turn()

        after = len([u for u in gs.units if u.player == 2])
        assert after == before

    def test_take_turn_does_not_move_existing_units(self, map_data):
        """Any existing player-2 units must be in the same position after the turn."""
        gs = GameState(map_data, num_players=2)
        # Manufacture a unit for player 2 if none exist on the random map.
        # We use a corner tile so it doesn't conflict with terrain.
        if not any(u.player == 2 for u in gs.units):
            for tile in gs.grid.get_capturable_tiles(player=2):
                if gs.create_unit("W", tile.x, tile.y, player=2):
                    break
        gs.current_player = 2
        before = [(u.x, u.y, u.health) for u in gs.units if u.player == 2]

        NoopBot(gs, player=2).take_turn()

        after = [(u.x, u.y, u.health) for u in gs.units if u.player == 2]
        assert before == after

    def test_take_turn_idempotent_when_game_already_over(self, map_data):
        """If the game is already over, take_turn() must be a no-op."""
        gs = GameState(map_data, num_players=2)
        gs.game_over = True
        gs.winner = 1
        gs.current_player = 2

        # Should not raise and should not change winner/game_over.
        NoopBot(gs, player=2).take_turn()

        assert gs.game_over is True
        assert gs.winner == 1


class TestNoopOpponentEnvWiring:
    def test_env_accepts_noop_opponent(self):
        env = StrategyGameEnv(map_file=None, opponent="noop", render_mode=None)
        env.reset()
        assert env.opponent_type == "noop"
        assert isinstance(env.opponent, NoopBot)
        env.close()

    def test_noop_opponent_does_not_grow_army(self):
        """After many agent end_turns against a noop opponent, the
        opponent's unit count must not increase. This is the property
        that makes noop useful: no respawning fodder for the agent to
        farm or be blocked by.
        """
        env = StrategyGameEnv(
            map_file="maps/1v1/beginner.csv",
            opponent="noop",
            render_mode=None,
            max_steps=200,
            max_turns=10,
        )
        env.reset(seed=0)
        agent_player = env.agent_player
        opp_player = 3 - agent_player
        initial_opp_units = sum(1 for u in env.game_state.units if u.player == opp_player)

        for _ in range(150):
            _, _, terminated, truncated, _ = env.step(END_TURN)
            if terminated or truncated:
                break

        final_opp_units = sum(1 for u in env.game_state.units if u.player == opp_player)
        assert final_opp_units == initial_opp_units
        env.close()

    def test_noop_opponent_never_wins(self):
        """With max_turns set, the noop opponent can only ever draw or lose
        (it can't win because it never seizes the agent's HQ).
        """
        env = StrategyGameEnv(
            map_file="maps/1v1/beginner.csv",
            opponent="noop",
            render_mode=None,
            max_steps=2_000,
            max_turns=5,
        )
        env.reset(seed=0)
        winner = "running"
        for _ in range(2_000):
            _, _, terminated, truncated, info = env.step(END_TURN)
            if terminated or truncated:
                winner = info.get("winner")
                break
        # winner is either None (draw) or the agent — never the opponent.
        assert winner != (3 - env.agent_player)
        env.close()
