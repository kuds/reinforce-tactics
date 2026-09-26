"""Tests for MixedBot.

MixedBot is a per-episode bridge between two scripted bots for the
``simple`` -> ``medium`` and ``medium`` -> ``advanced`` curriculum jumps in
configs/ppo/bootstrap.yaml. On construction it samples one of (``easy``,
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
            MixedBot(gs, player=2, easy="bogus", hard="medium", p_hard=0.0, rng=random.Random(0))

    def test_unknown_hard_raises(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="unknown bot type"):
            MixedBot(gs, player=2, easy="simple", hard="bogus", p_hard=1.0, rng=random.Random(0))


class TestMixedBotRandomBranches:
    """``random`` and ``balanced_random`` are valid inner bot names so the
    curriculum's intermediate/skirmish/corner_points mixed-bridge stages
    (which use ``easy=random`` or ``easy=balanced_random``) can construct
    without crashing the env at reset.

    Identity checks use ``type(...) is X`` rather than ``isinstance`` so a
    future refactor that accidentally routes ``"random"`` to
    ``BalancedRandomBot`` (its subclass) would fail the test loudly --
    ``isinstance(BalancedRandomBot_instance, RandomBot)`` is True and
    would mask the bug.
    """

    def test_easy_random_constructs(self, map_data):
        from reinforcetactics.game.bot import RandomBot

        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, easy="random", hard="simple", p_hard=0.0, rng=random.Random(0))
        assert type(bot._inner) is RandomBot

    def test_easy_balanced_random_constructs(self, map_data):
        from reinforcetactics.game.bot import BalancedRandomBot

        gs = _new_game_state(map_data)
        bot = MixedBot(gs, player=2, easy="balanced_random", hard="simple", p_hard=0.0, rng=random.Random(0))
        assert type(bot._inner) is BalancedRandomBot

    def test_easy_random_with_max_actions_kwarg(self, map_data):
        from reinforcetactics.game.bot import RandomBot

        gs = _new_game_state(map_data)
        bot = MixedBot(
            gs,
            player=2,
            easy="random",
            hard="simple",
            p_hard=0.0,
            rng=random.Random(0),
            easy_kwargs={"max_actions": 10},
        )
        assert type(bot._inner) is RandomBot
        assert bot._inner.max_actions == 10

    def test_hard_random_with_max_actions_kwarg(self, map_data):
        """Symmetric coverage with the easy_kwargs test: forces p_hard=1.0 so
        the selector picks ``hard_kwargs``. Without this the easy/hard
        ternary selector (MixedBot.__init__ ``chosen_kwargs = (hard_kwargs
        if self.use_hard else easy_kwargs)``) is untested in the hard
        branch -- a swapped selector regression would slip through.
        """
        from reinforcetactics.game.bot import RandomBot

        gs = _new_game_state(map_data)
        bot = MixedBot(
            gs,
            player=2,
            easy="simple",
            hard="random",
            p_hard=1.0,
            rng=random.Random(0),
            hard_kwargs={"max_actions": 7},
        )
        assert type(bot._inner) is RandomBot
        assert bot._inner.max_actions == 7


class TestMixedBotRejectsReservedKwargs:
    """``easy_kwargs`` / ``hard_kwargs`` must not carry keys MixedBot already
    supplies to ``_build_inner`` (``rng`` / ``player`` / ``game_state``).
    Pre-PR these would crash mid-curriculum with the opaque ``TypeError:
    got multiple values for keyword argument`` only on the episodes whose
    coin flip picked the side carrying the collision.
    """

    @pytest.mark.parametrize("reserved", ["rng", "player", "game_state"])
    def test_easy_kwargs_rejects_reserved_key(self, map_data, reserved):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="reserved keys"):
            MixedBot(
                gs,
                player=2,
                easy="random",
                hard="simple",
                p_hard=0.0,
                rng=random.Random(0),
                easy_kwargs={reserved: "anything"},
            )

    @pytest.mark.parametrize("reserved", ["rng", "player", "game_state"])
    def test_hard_kwargs_rejects_reserved_key(self, map_data, reserved):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="reserved keys"):
            MixedBot(
                gs,
                player=2,
                easy="simple",
                hard="random",
                p_hard=1.0,
                rng=random.Random(0),
                hard_kwargs={reserved: "anything"},
            )

    def test_validation_fires_regardless_of_coin_flip(self, map_data):
        """Even when the coin flip *won't* select the side carrying the bad
        kwarg, the validation still fires -- catches misconfig at construction
        rather than on a probabilistic later episode."""
        gs = _new_game_state(map_data)
        # use_hard=False (p_hard=0.0) -- only easy_kwargs would normally be
        # consumed, but the validation reads both regardless.
        with pytest.raises(ValueError, match="reserved keys"):
            MixedBot(
                gs,
                player=2,
                easy="simple",
                hard="random",
                p_hard=0.0,
                rng=random.Random(0),
                hard_kwargs={"rng": random.Random(99)},
            )


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

    def test_capabilities_fired_are_the_inner_bots(self, map_data):
        """The tournament snapshots get_capabilities_fired; MixedBot's own were always empty (review rulebots-19)."""
        gs = _new_game_state(map_data)
        gs.current_player = 2
        gs.player_gold[2] = 1000
        bot = MixedBot(gs, player=2, p_hard=0.0, rng=random.Random(0))

        bot.take_turn()

        fired = bot.get_capabilities_fired()
        assert fired == bot._inner.get_capabilities_fired()
        assert any(name.startswith("buy_") for name in fired)


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


class _NoDrawRng:
    """An rng that fails the test if MixedBot flips its coin."""

    def random(self):
        raise AssertionError("MixedBot drew from its rng before validating")


class TestMixedBotValidatesBeforeTheCoinFlip:
    """Review rulebots-11: both sides and ``p_hard`` are checked up front.

    Only the side the coin picked used to be validated, so a typo on the
    other side passed construction and crashed a curriculum run at the
    random later reset that first picked it; ``p_hard`` was never checked.
    """

    def test_unknown_hard_raises_even_when_the_coin_would_pick_easy(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="unknown bot type 'bogus'"):
            MixedBot(gs, player=2, easy="simple", hard="bogus", p_hard=0.0, rng=random.Random(0))

    def test_unknown_easy_raises_even_when_the_coin_would_pick_hard(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="unknown bot type 'bogus'"):
            MixedBot(gs, player=2, easy="bogus", hard="medium", p_hard=1.0, rng=random.Random(0))

    @pytest.mark.parametrize("p_hard", [-0.01, 1.5, float("nan"), "0.5", True])
    def test_p_hard_must_be_a_probability(self, map_data, p_hard):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="p_hard"):
            MixedBot(gs, player=2, p_hard=p_hard, rng=random.Random(0))

    def test_bad_kwargs_for_the_side_not_picked_raise(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError, match="hard_kwargs"):
            MixedBot(gs, player=2, easy="random", hard="medium", p_hard=0.0, hard_kwargs={"max_actions": 5})

    def test_validation_happens_before_any_rng_draw(self, map_data):
        gs = _new_game_state(map_data)
        with pytest.raises(ValueError):
            MixedBot(gs, player=2, easy="simple", hard="bogus", p_hard=0.5, rng=_NoDrawRng())
        with pytest.raises(ValueError):
            MixedBot(gs, player=2, p_hard=2.0, rng=_NoDrawRng())

    def test_boundaries_are_valid(self, map_data):
        gs = _new_game_state(map_data)
        assert not MixedBot(gs, player=2, p_hard=0, rng=random.Random(0)).use_hard
        assert MixedBot(gs, player=2, p_hard=1, rng=random.Random(0)).use_hard

    def test_validate_config_needs_no_game(self):
        MixedBot.validate_config(easy="balanced_random", hard="master", p_hard=0.3)
        with pytest.raises(ValueError):
            MixedBot.validate_config(easy="noop")
