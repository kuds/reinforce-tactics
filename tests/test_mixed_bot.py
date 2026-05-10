"""Tests for MixedBot.

MixedBot is a per-episode bridge between SimpleBot and MediumBot for the
``simple`` -> ``medium`` curriculum jump in configs/bootstrap.yaml. On
construction it samples one of the two using ``p_medium`` and delegates
``take_turn()`` to that instance for its lifetime. The env reconstructs
its opponent on every ``reset()``, so the choice is effectively
resampled per episode without a mid-episode switch.

These tests cover:
- Boundary p_medium values pick the expected inner bot deterministically.
- Same RNG seed produces the same Simple/Medium choice (reproducibility).
- ``take_turn`` delegates to the inner bot and ends the bot's turn.
- Sampling frequency over many constructions is close to ``p_medium``.
- The env dispatch wires MixedBot in correctly when ``opponent="mixed"``.
"""

import random

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import MediumBot, MixedBot, SimpleBot
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def map_data():
    np.random.seed(42)
    md = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()
    return md


def _new_game_state(map_data):
    return GameState(map_data, num_players=2)


class TestMixedBotChoice:
    def test_p_zero_always_simple(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, p_medium=0.0, rng=random.Random(0))
        assert bot.use_medium is False
        assert isinstance(bot._inner, SimpleBot)

    def test_p_one_always_medium(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, p_medium=1.0, rng=random.Random(0))
        assert bot.use_medium is True
        assert isinstance(bot._inner, MediumBot)

    def test_same_seed_same_choice(self, map_data):
        choices = []
        for _ in range(2):
            gs = _new_game_state(map_data)
            bot = MixedBot(gs, player=2, p_medium=0.5, rng=random.Random(123))
            choices.append(bot.use_medium)
        assert choices[0] == choices[1]

    def test_inner_bot_bound_to_same_game_state(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, p_medium=0.5, rng=random.Random(0))
        assert bot._inner.game_state is gs
        assert bot._inner.bot_player == 2


class TestMixedBotTakeTurn:
    def test_take_turn_terminates_simple_branch(self, map_data):
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, p_medium=0.0, rng=random.Random(0))
        bot.take_turn()
        # SimpleBot ends the turn unconditionally (bot.py:411).
        assert gs.game_over or gs.current_player != 2

    def test_take_turn_terminates_medium_branch(self, map_data):
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, p_medium=1.0, rng=random.Random(0))
        bot.take_turn()
        assert gs.game_over or gs.current_player != 2

    def test_take_turn_delegates_to_inner(self, map_data):
        # Replace the inner bot's take_turn to verify delegation, then
        # call MixedBot.take_turn() and confirm the inner method ran.
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, p_medium=0.0, rng=random.Random(0))

        called = []
        original = bot._inner.take_turn

        def _spy():
            called.append(True)
            original()

        bot._inner.take_turn = _spy
        bot.take_turn()
        assert called == [True]


class TestMixedBotSamplingDistribution:
    def test_p_medium_matches_empirical_frequency(self, map_data):
        # Single shared rng so successive constructions see fresh draws,
        # mirroring how the env feeds a freshly seeded random.Random per
        # reset() but here exercising the sampler directly.
        rng = random.Random(2024)
        n = 2000
        p = 0.7
        med_count = 0
        for _ in range(n):
            gs = _new_game_state(map_data)
            bot = MixedBot(gs, player=2, p_medium=p, rng=rng)
            if bot.use_medium:
                med_count += 1
        # Wide tolerance (+/- 5pp) keeps the test stable across CI seeds.
        assert abs(med_count / n - p) < 0.05


class TestMixedBotEnvIntegration:
    def test_env_dispatches_mixed_opponent(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_medium": 0.5},
        )
        env.reset(seed=0)
        assert isinstance(env.opponent, MixedBot)
        assert isinstance(env.opponent._inner, (SimpleBot, MediumBot))

    def test_env_reset_resamples_choice(self, map_data):
        # Across many resets with varying seeds, both branches should
        # appear -- proving the env's per-episode reconstruction (gym_env.py
        # reset()) is what produces the per-episode resampling we want.
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_medium": 0.5},
        )
        seen = set()
        for seed in range(40):
            env.reset(seed=seed)
            seen.add(env.opponent.use_medium)
            if seen == {True, False}:
                break
        assert seen == {True, False}

    def test_env_p_medium_zero_always_simple(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_medium": 0.0},
        )
        for seed in range(5):
            env.reset(seed=seed)
            assert env.opponent.use_medium is False
            assert isinstance(env.opponent._inner, SimpleBot)

    def test_env_p_medium_one_always_medium(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_medium": 1.0},
        )
        for seed in range(5):
            env.reset(seed=seed)
            assert env.opponent.use_medium is True
            assert isinstance(env.opponent._inner, MediumBot)
