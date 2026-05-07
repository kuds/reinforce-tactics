"""Tests for BalancedRandomBot.

The BalancedRandomBot is a curriculum stepping stone between NoopBot and
RandomBot: action throughput scales with the bot's army size (one build
attempt + one random action per owned unit per turn) so the bot stays
"alive" even after the agent kills its units, instead of stalling out
the way RandomBot(max_actions=1) would.

These tests cover:
- The bot terminates each turn (no infinite loops).
- Same RNG seed produces identical state evolution (reproducibility).
- Each owned unit acts at most once per turn.
- The bot still ends its turn when it has no units (degenerate case).
"""

import random

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import BalancedRandomBot
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def map_data():
    np.random.seed(42)
    md = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()
    return md


def _new_game_state(map_data):
    return GameState(map_data, num_players=2)


def _state_signature(gs: GameState):
    units = tuple(sorted((u.player, u.x, u.y, u.type, u.health) for u in gs.units))
    gold = tuple(sorted(gs.player_gold.items()))
    return (units, gold, gs.current_player, gs.game_over, gs.winner)


class TestBalancedRandomBotBasics:
    def test_take_turn_terminates(self, map_data):
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = BalancedRandomBot(gs, player=2, rng=random.Random(0))

        bot.take_turn()

        # Either the game finished or the bot handed the turn back.
        assert gs.game_over or gs.current_player != 2

    def test_no_units_still_ends_turn(self, map_data):
        # Strip all units owned by player 2 so the per-unit loop has
        # nothing to iterate over. The bot may still try to build and
        # must end the turn cleanly either way.
        gs = _new_game_state(map_data)
        gs.units = [u for u in gs.units if u.player != 2]
        gs.current_player = 2
        bot = BalancedRandomBot(gs, player=2, rng=random.Random(0))

        bot.take_turn()

        assert gs.game_over or gs.current_player != 2

    def test_same_seed_same_outcome(self, map_data):
        signatures = []
        for _ in range(2):
            gs = _new_game_state(map_data)
            gs.current_player = 2
            bot = BalancedRandomBot(gs, player=2, rng=random.Random(123))
            bot.take_turn()
            signatures.append(_state_signature(gs))
        assert signatures[0] == signatures[1]

    def test_each_unit_acts_at_most_once(self, map_data):
        # The defining property: "each available unit takes one random
        # action per turn." We instrument _execute to record the actor unit
        # per non-create action and assert no unit is recorded twice.
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = BalancedRandomBot(gs, player=2, rng=random.Random(7))

        seen_actors: list = []
        original_execute = bot._execute

        def _record(action_key, action):
            actor_field = bot._ACTOR_FIELDS.get(action_key)
            if actor_field is not None:
                actor = action.get(actor_field)
                if actor is not None:
                    seen_actors.append(id(actor))
            return original_execute(action_key, action)

        bot._execute = _record
        bot.take_turn()

        # No actor id should appear twice in the per-unit phase.
        assert len(seen_actors) == len(set(seen_actors))
