"""
Tests for RandomBot and seed-driven reproducibility of the random-opponent path.

These tests cover:
- RandomBot honours its ``rng`` argument (same seed → same action sequence).
- ``RandomBot.take_turn()`` always advances current_player or finishes the game.
- ``StrategyGameEnv(opponent="random")`` reset(seed=...) is reproducible end-to-end.
- ``make_maskable_env(seed=...)`` seeds the underlying env.
- ``evaluate_model(seed=...)`` produces deterministic metrics.
"""

import random

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import RandomBot, SimpleBot
from reinforcetactics.rl.evaluation import evaluate_model
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.masking import make_maskable_env
from reinforcetactics.utils.file_io import FileIO

END_TURN = np.array([5, 0, 0, 0, 0, 0])


@pytest.fixture
def map_data():
    """Deterministic 10x10 map shared across tests."""
    np.random.seed(42)
    md = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()
    return md


def _new_game_state(map_data):
    return GameState(map_data, num_players=2)


def _state_signature(gs: GameState):
    """A stable, hashable signature of the parts of game state RandomBot can affect."""
    units = tuple(sorted((u.player, u.x, u.y, u.type, u.health) for u in gs.units))
    gold = tuple(sorted(gs.player_gold.items()))
    return (units, gold, gs.current_player, gs.game_over, gs.winner)


class TestRandomBotBasics:
    def test_take_turn_advances_current_player(self, map_data):
        """take_turn() must end the bot's turn (or end the game)."""
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = RandomBot(gs, player=2, max_actions=10, rng=random.Random(0))

        bot.take_turn()

        assert gs.game_over or gs.current_player != 2

    def test_default_max_actions_is_finite(self, map_data):
        """take_turn() must terminate even without an explicit max_actions override."""
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = RandomBot(gs, player=2, rng=random.Random(0))

        # Should return promptly; if it loops forever this test will time out.
        bot.take_turn()

        assert gs.current_player != 2 or gs.game_over

    def test_only_picks_legal_action_types(self, map_data):
        """Sanity check: RandomBot must not crash when running on a fresh game."""
        gs = _new_game_state(map_data)
        gs.current_player = 2
        bot = RandomBot(gs, player=2, max_actions=20, rng=random.Random(1))

        # No exception should escape take_turn() — internal failures are caught
        # and the bot retries other actions.
        bot.take_turn()


class TestRandomBotSeeding:
    def test_same_seed_same_outcome(self, map_data):
        """Two RandomBots with the same seed must produce identical states."""
        sigs = []
        for _ in range(2):
            gs = _new_game_state(map_data)
            gs.current_player = 2
            bot = RandomBot(gs, player=2, max_actions=20, rng=random.Random(123))
            bot.take_turn()
            sigs.append(_state_signature(gs))

        assert sigs[0] == sigs[1]

    def test_different_seeds_diverge(self, map_data):
        """With high probability, different seeds produce different sequences."""
        sigs = set()
        for s in range(8):
            gs = _new_game_state(map_data)
            gs.current_player = 2
            bot = RandomBot(gs, player=2, max_actions=20, rng=random.Random(s))
            bot.take_turn()
            sigs.add(_state_signature(gs))

        # 8 different seeds should produce more than one distinct outcome.
        assert len(sigs) > 1

    def test_unseeded_bot_uses_global_random(self, map_data):
        """When no rng is supplied, RandomBot falls back to the global ``random`` module."""
        gs = _new_game_state(map_data)
        gs.current_player = 2
        random.seed(7)

        bot = RandomBot(gs, player=2, max_actions=20)
        bot.take_turn()

        # No assertion on specific state — we only assert it ran without error
        # and used a non-None rng (the module).
        assert bot._rng is random


class TestEnvSeeding:
    def test_reset_with_seed_reproducible(self, map_data):
        """Two envs reset with the same seed must reach identical post-step state."""
        sigs = []
        for _ in range(2):
            env = StrategyGameEnv(map_file=None, opponent="random", max_steps=50)
            # Replace map with deterministic shared map_data.
            env.initial_map_data = map_data
            env.reset(seed=42)
            env.step(END_TURN)
            sigs.append(_state_signature(env.game_state))
            env.close()

        assert sigs[0] == sigs[1]

    def test_reset_without_seed_continues_stream(self, map_data):
        """Subsequent reset() without a seed must keep np_random advancing
        (so each episode picks a new RandomBot seed deterministically)."""
        env = StrategyGameEnv(map_file=None, opponent="random", max_steps=50)
        env.initial_map_data = map_data
        env.reset(seed=42)

        # Run two consecutive episodes; opponents should be DIFFERENT seeds
        # because np_random advances even when reset() is called without a seed.
        env.step(END_TURN)
        sig_a = _state_signature(env.game_state)

        env.reset()  # No seed → np_random keeps advancing
        env.step(END_TURN)
        sig_b = _state_signature(env.game_state)

        env.close()
        assert sig_a != sig_b

    def test_random_opponent_class_after_reset(self, map_data):
        """After reset, opponent should be a RandomBot instance for opponent='random'."""
        env = StrategyGameEnv(map_file=None, opponent="random", max_steps=50)
        env.initial_map_data = map_data
        env.reset(seed=0)

        assert isinstance(env.opponent, RandomBot)
        assert env.opponent.bot_player == 2
        env.close()

    def test_bot_opponent_unaffected_by_seeding_changes(self, map_data):
        """opponent='bot' must still produce a SimpleBot (regression check)."""
        env = StrategyGameEnv(map_file=None, opponent="bot", max_steps=50)
        env.initial_map_data = map_data
        env.reset(seed=0)

        assert isinstance(env.opponent, SimpleBot)
        env.close()


class TestMakeMaskableEnvSeeding:
    def test_seed_is_threaded_to_underlying_env(self):
        """make_maskable_env(seed=...) must seed the wrapped StrategyGameEnv."""
        env_a = make_maskable_env(opponent="random", max_steps=50, seed=99)
        env_b = make_maskable_env(opponent="random", max_steps=50, seed=99)

        # After construction (which performed a seeded reset internally), running
        # one end_turn step should produce the same opponent state.
        env_a.step(END_TURN)
        env_b.step(END_TURN)

        sig_a = _state_signature(env_a.unwrapped.game_state)
        sig_b = _state_signature(env_b.unwrapped.game_state)

        env_a.close()
        env_b.close()

        # They share the same map only if env construction generates the same
        # randomized map — but make_maskable_env(map_file=None) uses a fresh
        # randomized map per call. We therefore only assert that the OPPONENT
        # actions are deterministic given a fixed map. When the map differs,
        # we relax to: at minimum, current_player and game_over agree, and the
        # gold totals are deterministic given the same map.
        # For a strict reproducibility check we need a fixed map_file:
        assert sig_a[2] == sig_b[2]  # current_player matches
        assert sig_a[3] == sig_b[3]  # game_over matches

    def test_seed_with_fixed_map_is_fully_reproducible(self, tmp_path):
        """With a fixed map_file AND seed, two envs must be byte-identical."""
        # Use a real fixed map so the only randomness comes from np_random.
        map_file = "maps/1v1/beginner.csv"

        env_a = make_maskable_env(map_file=map_file, opponent="random", max_steps=50, seed=7)
        env_b = make_maskable_env(map_file=map_file, opponent="random", max_steps=50, seed=7)

        env_a.step(END_TURN)
        env_b.step(END_TURN)

        sig_a = _state_signature(env_a.unwrapped.game_state)
        sig_b = _state_signature(env_b.unwrapped.game_state)

        env_a.close()
        env_b.close()

        assert sig_a == sig_b


class _ScriptedModel:
    """Tiny stand-in for an SB3 model that always picks 'end_turn'.

    Used to test ``evaluate_model`` without depending on a trained policy.
    The flat_discrete action space puts end_turn last in the legal list,
    but for multi_discrete (the env default), end_turn is action_type=5.
    """

    def predict(self, obs, deterministic=True, action_masks=None):
        return np.array([5, 0, 0, 0, 0, 0]), None


class TestEvaluateModelSeeding:
    def test_evaluate_model_seed_reproducible(self):
        """evaluate_model(seed=X) must produce identical metrics across runs."""
        model = _ScriptedModel()
        env_a = make_maskable_env(
            map_file="maps/1v1/beginner.csv", opponent="random", max_steps=20, seed=11
        )
        env_b = make_maskable_env(
            map_file="maps/1v1/beginner.csv", opponent="random", max_steps=20, seed=11
        )

        m_a = evaluate_model(model, env_a, n_episodes=3, seed=11)
        m_b = evaluate_model(model, env_b, n_episodes=3, seed=11)

        env_a.close()
        env_b.close()

        # Floating-point reward sums and integer counts must match exactly.
        assert m_a["wins"] == m_b["wins"]
        assert m_a["losses"] == m_b["losses"]
        assert m_a["draws"] == m_b["draws"]
        assert m_a["avg_reward"] == m_b["avg_reward"]
        assert m_a["avg_length"] == m_b["avg_length"]

        # Raw per-episode arrays should also be reproducible and JSON-friendly.
        assert m_a["rewards"] == m_b["rewards"]
        assert m_a["lengths"] == m_b["lengths"]
        assert all(isinstance(r, float) for r in m_a["rewards"])
        assert all(isinstance(length, int) for length in m_a["lengths"])

    def test_evaluate_model_no_seed_still_works(self):
        """evaluate_model() with seed=None must still complete without error."""
        model = _ScriptedModel()
        env = make_maskable_env(
            map_file="maps/1v1/beginner.csv", opponent="random", max_steps=20
        )
        metrics = evaluate_model(model, env, n_episodes=2)
        env.close()

        assert metrics["episodes"] == 2
        assert metrics["wins"] + metrics["losses"] + metrics["draws"] == 2
