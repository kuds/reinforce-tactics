"""Tests for MixedBot.

MixedBot is a per-episode bridge between two scripted bots for the
``simple`` -> ``medium`` and ``medium`` -> ``advanced`` curriculum jumps in
configs/bootstrap.yaml. On construction it samples one of (``easy``,
``hard``) using ``p_hard`` and delegates ``take_turn()`` to that
instance for its lifetime. The env reconstructs its opponent on every
``reset()``, so the choice is effectively resampled per episode without
a mid-episode switch.

These tests cover:
- Boundary p_hard values pick the expected inner bot deterministically.
- Same RNG seed produces the same choice (reproducibility).
- ``take_turn`` delegates to the inner bot and ends the bot's turn.
- Sampling frequency over many constructions is close to ``p_hard``.
- Both default (simple/medium) and explicit (medium/advanced) bot pairs work.
- Unknown bot type names raise ValueError.
- The env dispatch wires MixedBot in correctly when ``opponent="mixed"``.
"""

import random

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MediumBot, MixedBot, SimpleBot
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


class TestMixedBotChoiceSimpleMedium:
    def test_p_zero_always_easy(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, p_hard=0.0, rng=random.Random(0))
        assert bot.use_hard is False
        assert isinstance(bot._inner, SimpleBot)

    def test_p_one_always_hard(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, p_hard=1.0, rng=random.Random(0))
        assert bot.use_hard is True
        assert isinstance(bot._inner, MediumBot)

    def test_same_seed_same_choice(self, map_data):
        choices = []
        for _ in range(2):
            gs = _new_game_state(map_data)
            bot = MixedBot(gs, player=2, p_hard=0.5, rng=random.Random(123))
            choices.append(bot.use_hard)
        assert choices[0] == choices[1]

    def test_inner_bot_bound_to_same_game_state(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, p_hard=0.5, rng=random.Random(0))
        assert bot._inner.game_state is gs
        assert bot._inner.bot_player == 2


class TestMixedBotChoiceMediumAdvanced:
    def test_p_zero_picks_easy_medium(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, easy="medium", hard="advanced", p_hard=0.0, rng=random.Random(0))
        assert bot.use_hard is False
        # AdvancedBot subclasses MediumBot, so check the exact class to
        # distinguish "medium" from "advanced" inner selection.
        assert type(bot._inner) is MediumBot

    def test_p_one_picks_hard_advanced(self, map_data):
        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, easy="medium", hard="advanced", p_hard=1.0, rng=random.Random(0))
        assert bot.use_hard is True
        assert isinstance(bot._inner, AdvancedBot)


class TestMixedBotInvalidName:
    def test_unknown_easy_raises(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="unknown bot type"):
            MixedBot(gs, player=2, easy="random", hard="medium", p_hard=0.0, rng=random.Random(0))

    def test_unknown_hard_raises(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="unknown bot type"):
            MixedBot(gs, player=2, easy="simple", hard="bogus", p_hard=1.0, rng=random.Random(0))


class TestMixedBotTakeTurn:
    def test_take_turn_terminates_easy_branch(self, map_data):
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, p_hard=0.0, rng=random.Random(0))
        bot.take_turn()
        # SimpleBot ends the turn unconditionally (bot.py:411).
        assert gs.game_over or gs.current_player != 2

    def test_take_turn_terminates_hard_branch(self, map_data):
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, p_hard=1.0, rng=random.Random(0))
        bot.take_turn()
        assert gs.game_over or gs.current_player != 2

    def test_take_turn_terminates_advanced_branch(self, map_data):
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, easy="medium", hard="advanced", p_hard=1.0, rng=random.Random(0))
        bot.take_turn()
        assert gs.game_over or gs.current_player != 2

    def test_take_turn_delegates_to_inner(self, map_data):
        # Replace the inner bot's take_turn to verify delegation, then
        # call MixedBot.take_turn() and confirm the inner method ran.
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = MixedBot(gs, player=2, p_hard=0.0, rng=random.Random(0))

        called = []
        original = bot._inner.take_turn

        def _spy():
            called.append(True)
            original()

        bot._inner.take_turn = _spy
        bot.take_turn()
        assert called == [True]


class TestMixedBotSamplingDistribution:
    def test_p_hard_matches_empirical_frequency(self, map_data):
        # Single shared rng so successive constructions see fresh draws,
        # mirroring how the env feeds a freshly seeded random.Random per
        # reset() but here exercising the sampler directly.
        rng = random.Random(2024)
        n = 2000
        p = 0.7
        hard_count = 0
        for _ in range(n):
            gs = _new_game_state(map_data)
            bot = MixedBot(gs, player=2, p_hard=p, rng=rng)
            if bot.use_hard:
                hard_count += 1
        # Wide tolerance (+/- 5pp) keeps the test stable across CI seeds.
        assert abs(hard_count / n - p) < 0.05


class TestMixedBotEnvIntegration:
    def test_env_dispatches_mixed_opponent(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_hard": 0.5},
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
            opponent_kwargs={"p_hard": 0.5},
        )
        seen = set()
        for seed in range(40):
            env.reset(seed=seed)
            seen.add(env.opponent.use_hard)
            if seen == {True, False}:
                break
        assert seen == {True, False}

    def test_env_p_hard_zero_always_easy(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_hard": 0.0},
        )
        for seed in range(5):
            env.reset(seed=seed)
            assert env.opponent.use_hard is False
            assert isinstance(env.opponent._inner, SimpleBot)

    def test_env_p_hard_one_always_hard(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"p_hard": 1.0},
        )
        for seed in range(5):
            env.reset(seed=seed)
            assert env.opponent.use_hard is True
            assert isinstance(env.opponent._inner, MediumBot)

    def test_env_medium_advanced_pair(self, map_data):
        env = StrategyGameEnv(
            map_file=None,
            opponent="mixed",
            opponent_kwargs={"easy": "medium", "hard": "advanced", "p_hard": 1.0},
        )
        env.reset(seed=0)
        assert isinstance(env.opponent._inner, AdvancedBot)
