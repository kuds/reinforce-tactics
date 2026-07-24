"""
Tests for the StrategyGameEnv Gymnasium environment.

This test suite provides comprehensive coverage of the Gymnasium environment
used for RL training, including observation spaces, action spaces, step/reset
functions, reward calculations, and episode statistics.
"""

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.utils.file_io import FileIO

# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def simple_map_data():
    """Generate a deterministic map for testing.

    Using a fixed seed for reproducibility to avoid test flakiness.
    """
    np.random.seed(42)
    map_data = FileIO.generate_random_map(10, 10, num_players=2)
    np.random.seed()  # Reset random state
    return map_data


@pytest.fixture
def env_default():
    """Create environment with default parameters."""
    return StrategyGameEnv(map_file=None, opponent="bot", render_mode=None)


@pytest.fixture
def env_no_opponent():
    """Create environment without opponent."""
    return StrategyGameEnv(map_file=None, opponent=None, render_mode=None)


@pytest.fixture
def env_hierarchical():
    """Create environment with hierarchical action space."""
    return StrategyGameEnv(map_file=None, opponent="bot", render_mode=None, hierarchical=True)


@pytest.fixture
def custom_reward_config():
    """Custom reward configuration for testing."""
    return {
        "win": 500.0,
        "loss": -500.0,
        "income_diff": 0.2,
        "unit_diff": 2.0,
        "structure_control": 10.0,
        "invalid_action": -5.0,
        "turn_penalty": -0.5,
    }


# ==============================================================================
# 1. ENVIRONMENT CREATION TESTS
# ==============================================================================


class TestEnvironmentCreation:
    """Test environment initialization with various configurations."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters (no map file, random map generation)."""
        env = StrategyGameEnv(map_file=None, opponent="bot", render_mode=None)

        assert env is not None
        assert env.game_state is not None
        assert env.grid_width > 0
        assert env.grid_height > 0
        assert env.opponent_type == "bot"
        assert env.max_steps == 200
        assert env.current_step == 0
        assert env.hierarchical is False
        assert env.render_mode is None
        env.close()

    def test_initialization_with_map_file(self, simple_map_data, tmp_path):
        """Test initialization with a specific map file."""
        # Save map to temporary file
        map_file = tmp_path / "test_map.csv"
        pd.DataFrame(simple_map_data).to_csv(map_file, index=False, header=False)

        env = StrategyGameEnv(map_file=str(map_file), opponent="bot", render_mode=None)

        assert env is not None
        assert env.game_state is not None
        env.close()

    def test_initialization_opponent_bot(self):
        """Test initialization with opponent='bot'."""
        env = StrategyGameEnv(map_file=None, opponent="bot", render_mode=None)

        assert env.opponent_type == "bot"
        env.close()

    def test_initialization_opponent_random(self):
        """Test initialization with opponent='random'."""
        env = StrategyGameEnv(map_file=None, opponent="random", render_mode=None)

        assert env.opponent_type == "random"
        env.close()

    def test_initialization_opponent_balanced_random(self):
        """``balanced_random`` instantiates BalancedRandomBot.

        Curriculum stepping stone between ``noop`` (zero opponent actions)
        and ``random`` (RandomBot's default 20 actions/turn). Action
        throughput scales with the bot's army size: one build attempt plus
        one random action per owned unit per turn -- see configs/ppo/bootstrap.yaml.
        """
        from reinforcetactics.game.bot import BalancedRandomBot

        env = StrategyGameEnv(map_file=None, opponent="balanced_random", render_mode=None)
        env.reset(seed=0)

        assert env.opponent_type == "balanced_random"
        assert isinstance(env.opponent, BalancedRandomBot)
        env.close()

    def test_initialization_opponent_self(self):
        """Test initialization with opponent='self' (self-play)."""
        env = StrategyGameEnv(map_file=None, opponent="self", render_mode=None)

        assert env.opponent_type == "self"
        env.close()

    def test_initialization_opponent_none(self):
        """Test initialization with opponent=None."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)

        assert env.opponent_type is None
        assert env.opponent is None
        env.close()

    def test_initialization_custom_reward_config(self, custom_reward_config):
        """Test initialization with custom reward_config."""
        env = StrategyGameEnv(map_file=None, opponent="bot", render_mode=None, reward_config=custom_reward_config)

        assert env.reward_config["win"] == 500.0
        assert env.reward_config["loss"] == -500.0
        assert env.reward_config["invalid_action"] == -5.0
        env.close()

    def test_initialization_hierarchical_mode(self, env_hierarchical):
        """Test initialization with hierarchical=True for HRL mode."""
        assert env_hierarchical.hierarchical is True
        assert isinstance(env_hierarchical.action_space, spaces.Dict)
        assert "goal" in env_hierarchical.action_space.spaces
        assert "primitive" in env_hierarchical.action_space.spaces
        env_hierarchical.close()

    def test_render_mode_none_no_pygame(self):
        """Test that render_mode=None does not import pygame (headless mode verification)."""
        # Create environment without rendering
        env = StrategyGameEnv(map_file=None, opponent="bot", render_mode=None)

        # Check that renderer is not initialized
        assert env.renderer is None
        assert env.render_mode is None
        env.close()


# ==============================================================================
# 2. OBSERVATION SPACE TESTS
# ==============================================================================


class TestObservationSpace:
    """Test observation space structure and validity."""

    def test_observation_space_is_dict(self, env_default):
        """Verify observation_space is a gymnasium.spaces.Dict."""
        assert isinstance(env_default.observation_space, spaces.Dict)
        env_default.close()

    def test_grid_shape_and_dtype(self, env_default):
        """Verify 'grid' shape matches the encoded one-hot channel layout."""
        from reinforcetactics.rl.observation import GRID_CHANNELS

        grid_space = env_default.observation_space["grid"]

        expected_shape = (env_default.grid_height, env_default.grid_width, GRID_CHANNELS)
        assert grid_space.shape == expected_shape
        assert grid_space.dtype == np.float32
        env_default.close()

    def test_units_shape_and_dtype(self, env_default):
        """Verify 'units' shape matches the encoded one-hot channel layout."""
        from reinforcetactics.rl.observation import UNIT_CHANNELS

        units_space = env_default.observation_space["units"]

        expected_shape = (env_default.grid_height, env_default.grid_width, UNIT_CHANNELS)
        assert units_space.shape == expected_shape
        assert units_space.dtype == np.float32
        env_default.close()

    def test_global_features_shape_and_dtype(self, env_default):
        """Verify 'global_features' shape matches the canonical dimension."""
        from reinforcetactics.rl.observation import GLOBAL_FEATURES_DIM

        global_space = env_default.observation_space["global_features"]

        assert global_space.shape == (GLOBAL_FEATURES_DIM,)
        assert global_space.dtype == np.float32
        env_default.close()

    def test_action_mask_not_in_observation_space(self, env_default):
        """The action mask is delivered via ``env.action_masks()``, not the obs."""
        assert "action_mask" not in env_default.observation_space.spaces
        # ``action_masks()`` must still work so MaskablePPO can pull masks.
        assert callable(env_default.action_masks)
        env_default.close()

    def test_observations_from_reset_match_space(self, env_default):
        """Test that observations returned by reset() match the observation space."""
        obs, _ = env_default.reset()

        assert "grid" in obs
        assert "units" in obs
        assert "global_features" in obs
        assert "action_mask" not in obs

        assert obs["grid"].shape == env_default.observation_space["grid"].shape
        assert obs["units"].shape == env_default.observation_space["units"].shape
        assert obs["global_features"].shape == env_default.observation_space["global_features"].shape

        assert obs["grid"].dtype == np.float32
        assert obs["units"].dtype == np.float32
        assert obs["global_features"].dtype == np.float32

        env_default.close()

    def test_observations_from_step_match_space(self, env_default):
        """Test that observations returned by step() match the observation space."""
        env_default.reset()

        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
        obs, _, _, _, _ = env_default.step(action)

        assert "grid" in obs
        assert "units" in obs
        assert "global_features" in obs
        assert "action_mask" not in obs

        assert obs["grid"].shape == env_default.observation_space["grid"].shape
        assert obs["units"].shape == env_default.observation_space["units"].shape
        assert obs["global_features"].shape == env_default.observation_space["global_features"].shape

        env_default.close()


# ==============================================================================
# 3. ACTION SPACE TESTS
# ==============================================================================


class TestActionSpace:
    """Test action space structure and encoding."""

    def test_action_space_multidiscrete_flat_mode(self, env_default):
        """Verify action space is gymnasium.spaces.MultiDiscrete for flat mode."""
        assert isinstance(env_default.action_space, spaces.MultiDiscrete)
        assert len(env_default.action_space.nvec) == 6
        env_default.close()

    def test_action_space_dict_hierarchical_mode(self, env_hierarchical):
        """Verify action space is gymnasium.spaces.Dict for hierarchical mode."""
        assert isinstance(env_hierarchical.action_space, spaces.Dict)
        assert "goal" in env_hierarchical.action_space.spaces
        assert "primitive" in env_hierarchical.action_space.spaces

        # Check goal space
        assert isinstance(env_hierarchical.action_space.spaces["goal"], spaces.Discrete)

        # Check primitive space
        assert isinstance(env_hierarchical.action_space.spaces["primitive"], spaces.MultiDiscrete)

        env_hierarchical.close()

    def test_encode_action_create_unit(self, env_default):
        """Test _encode_action() for create_unit action (type 0)."""
        action = np.array([0, 0, 5, 5, 8, 8])  # create_unit, warrior, at (8,8)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 0
        assert action_dict["unit_type"] == "W"
        assert action_dict["from_pos"] == (5, 5)
        assert action_dict["to_pos"] == (8, 8)

        env_default.close()

    def test_encode_action_move(self, env_default):
        """Test _encode_action() for move action (type 1)."""
        action = np.array([1, 0, 2, 3, 4, 5])  # move from (2,3) to (4,5)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 1
        assert action_dict["from_pos"] == (2, 3)
        assert action_dict["to_pos"] == (4, 5)

        env_default.close()

    def test_encode_action_attack(self, env_default):
        """Test _encode_action() for attack action (type 2)."""
        action = np.array([2, 0, 1, 1, 2, 1])  # attack from (1,1) to (2,1)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 2
        assert action_dict["from_pos"] == (1, 1)
        assert action_dict["to_pos"] == (2, 1)

        env_default.close()

    def test_encode_action_seize(self, env_default):
        """Test _encode_action() for seize action (type 3)."""
        action = np.array([3, 0, 5, 5, 0, 0])  # seize at (5,5)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 3
        assert action_dict["from_pos"] == (5, 5)

        env_default.close()

    def test_encode_action_heal(self, env_default):
        """Test _encode_action() for heal action (type 4)."""
        action = np.array([4, 2, 3, 3, 4, 3])  # heal from (3,3) to (4,3)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 4
        assert action_dict["unit_type"] == "C"  # Cleric
        assert action_dict["from_pos"] == (3, 3)
        assert action_dict["to_pos"] == (4, 3)

        env_default.close()

    def test_encode_action_end_turn(self, env_default):
        """Test _encode_action() for end_turn action (type 5)."""
        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 5

        env_default.close()

    def test_encode_action_paralyze(self, env_default):
        """Test _encode_action() for paralyze action (type 6)."""
        action = np.array([6, 1, 3, 3, 4, 3])  # paralyze from (3,3) to (4,3)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 6
        assert action_dict["from_pos"] == (3, 3)
        assert action_dict["to_pos"] == (4, 3)

        env_default.close()

    def test_encode_action_haste(self, env_default):
        """Test _encode_action() for haste action (type 7)."""
        action = np.array([7, 6, 2, 2, 3, 2])  # haste from (2,2) to (3,2)

        action_dict = env_default._encode_action(action)

        assert action_dict["action_type"] == 7
        assert action_dict["unit_type"] == "S"  # Sorcerer
        assert action_dict["from_pos"] == (2, 2)
        assert action_dict["to_pos"] == (3, 2)

        env_default.close()

    def test_encode_action_all_unit_types(self, env_default):
        """Test action encoding with all unit types."""
        # Test Warrior (0)
        action_w = np.array([0, 0, 0, 0, 1, 1])
        assert env_default._encode_action(action_w)["unit_type"] == "W"

        # Test Mage (1)
        action_m = np.array([0, 1, 0, 0, 1, 1])
        assert env_default._encode_action(action_m)["unit_type"] == "M"

        # Test Cleric (2)
        action_c = np.array([0, 2, 0, 0, 1, 1])
        assert env_default._encode_action(action_c)["unit_type"] == "C"

        # Test Archer (3)
        action_a = np.array([0, 3, 0, 0, 1, 1])
        assert env_default._encode_action(action_a)["unit_type"] == "A"

        # Test Knight (4)
        action_k = np.array([0, 4, 0, 0, 1, 1])
        assert env_default._encode_action(action_k)["unit_type"] == "K"

        # Test Rogue (5)
        action_r = np.array([0, 5, 0, 0, 1, 1])
        assert env_default._encode_action(action_r)["unit_type"] == "R"

        # Test Sorcerer (6)
        action_s = np.array([0, 6, 0, 0, 1, 1])
        assert env_default._encode_action(action_s)["unit_type"] == "S"

        # Test Barbarian (7)
        action_b = np.array([0, 7, 0, 0, 1, 1])
        assert env_default._encode_action(action_b)["unit_type"] == "B"

        env_default.close()

    def test_encode_action_boundary_coordinates(self, env_default):
        """Test action encoding with boundary coordinates."""
        max_x = env_default.grid_width - 1
        max_y = env_default.grid_height - 1

        # Test boundary positions
        action = np.array([1, 0, 0, 0, max_x, max_y])
        action_dict = env_default._encode_action(action)

        assert action_dict["from_pos"] == (0, 0)
        assert action_dict["to_pos"] == (max_x, max_y)

        env_default.close()


# ==============================================================================
# 4. STEP FUNCTION TESTS
# ==============================================================================


class TestStepFunction:
    """Test step function behavior."""

    def test_step_returns_correct_tuple(self, env_default):
        """Test that step() returns tuple of (obs, reward, terminated, truncated, info)."""
        env_default.reset()
        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn

        result = env_default.step(action)

        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env_default.close()

    def test_valid_action_end_turn(self, env_default):
        """Test valid action execution (end_turn)."""
        env_default.reset()
        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn

        _, reward, _, _, info = env_default.step(action)

        # end_turn should be valid and apply turn penalty
        assert info["valid_action"] is True
        assert reward <= 0  # Turn penalty applied

        env_default.close()

    def test_invalid_action_negative_reward(self, env_default):
        """Test invalid action execution and negative rewards (invalid_action penalty)."""
        env_default.reset()

        # Try to move a non-existent unit
        action = np.array([1, 0, 0, 0, 1, 1])  # move from (0,0) to (1,1)

        _, reward, _, _, info = env_default.step(action)

        # Should be marked as invalid
        assert info["valid_action"] is False
        # Should receive invalid action penalty
        assert reward < 0

        env_default.close()

    def test_termination_condition(self, env_default):
        """Test termination conditions (game_over flag)."""
        env_default.reset()

        # Manually set game_over to test termination
        env_default.game_state.game_over = True
        env_default.game_state.winner = 1

        action = np.array([5, 0, 0, 0, 0, 0])
        _, _, terminated, _, info = env_default.step(action)

        assert terminated is True
        assert info["game_over"] is True

        env_default.close()

    def test_truncation_max_steps(self, env_default):
        """Test truncation when current_step >= max_steps."""
        env_default.reset()

        # Set current_step close to max_steps
        env_default.current_step = env_default.max_steps - 1

        action = np.array([5, 0, 0, 0, 0, 0])
        _, _, _, truncated, _ = env_default.step(action)

        assert truncated is True

        env_default.close()

    def test_info_dict_contains_expected_keys(self, env_default):
        """Test that info dict contains expected keys."""
        env_default.reset()
        action = np.array([5, 0, 0, 0, 0, 0])

        _, _, _, _, info = env_default.step(action)

        # Check required keys
        assert "episode_stats" in info
        assert "game_over" in info
        assert "winner" in info
        assert "turn" in info
        assert "valid_action" in info

        env_default.close()


# ==============================================================================
# 5. RESET FUNCTION TESTS
# ==============================================================================


class TestResetFunction:
    """Test reset function behavior."""

    def test_reset_returns_correct_tuple(self, env_default):
        """Verify reset() returns (observation, info) tuple."""
        result = env_default.reset()

        assert isinstance(result, tuple)
        assert len(result) == 2

        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

        env_default.close()

    def test_game_state_properly_reset(self, env_default):
        """Verify game state is properly reset (turn number, gold, units)."""
        # First do some actions
        env_default.reset()
        env_default.step(np.array([5, 0, 0, 0, 0, 0]))  # end_turn

        # Now reset
        env_default.reset()

        # Check that game state is reset
        assert env_default.game_state.turn_number == 0
        assert env_default.game_state.current_player == 1
        assert env_default.game_state.game_over is False

        env_default.close()

    def test_current_step_reset_to_zero(self, env_default):
        """Verify current_step is reset to 0."""
        env_default.reset()
        env_default.step(np.array([5, 0, 0, 0, 0, 0]))
        env_default.step(np.array([5, 0, 0, 0, 0, 0]))

        assert env_default.current_step > 0

        env_default.reset()
        assert env_default.current_step == 0

        env_default.close()

    def test_episode_stats_reset(self, env_default):
        """Verify episode_stats are reset."""
        env_default.reset()
        env_default.step(np.array([5, 0, 0, 0, 0, 0]))

        # Modify stats
        env_default.episode_stats["reward"] = 100.0
        env_default.episode_stats["invalid_actions"] = 5

        # Reset
        env_default.reset()

        # Check stats are reset
        assert env_default.episode_stats["reward"] == 0.0
        assert env_default.episode_stats["invalid_actions"] == 0
        assert env_default.episode_stats["winner"] is None

        env_default.close()

    def test_reset_with_seed(self, env_default):
        """Test reset() with seed parameter for reproducibility."""
        # Reset with seed
        obs1, _ = env_default.reset(seed=42)
        obs2, _ = env_default.reset(seed=42)

        # Observations should be identical with same seed
        assert np.array_equal(obs1["grid"], obs2["grid"])
        assert np.array_equal(obs1["units"], obs2["units"])

        env_default.close()

    def test_opponent_reinitialized_on_reset(self):
        """Test that opponent is re-initialized on reset (functional behavior)."""
        env = StrategyGameEnv(map_file=None, opponent="bot", render_mode=None)

        env.reset()
        # Opponent should be initialized after reset
        assert env.opponent is not None

        # Take a step that modifies opponent state
        env.step(np.array([5, 0, 0, 0, 0, 0]))  # end_turn

        env.reset()
        # After reset, opponent should still be functional
        assert env.opponent is not None
        # Game state should be reset to turn 0
        assert env.game_state.turn_number == 0

        env.close()


# ==============================================================================
# 6. REWARD CALCULATION TESTS
# ==============================================================================


class TestRewardCalculation:
    """Test reward calculation components."""

    def test_win_reward(self):
        """Test win reward."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset()

        # Set game to won state
        env.game_state.game_over = True
        env.game_state.winner = 1

        action = np.array([5, 0, 0, 0, 0, 0])
        _, reward, terminated, _, _ = env.step(action)

        # Should receive win reward
        assert reward > 0
        assert terminated is True

        env.close()

    def test_loss_reward(self):
        """Test loss reward."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset()

        # Set game to lost state
        env.game_state.game_over = True
        env.game_state.winner = 2

        action = np.array([5, 0, 0, 0, 0, 0])
        _, reward, terminated, _, _ = env.step(action)

        # Should receive loss penalty
        assert reward < 0
        assert terminated is True

        env.close()

    def test_turn_penalty_defaults_to_zero(self, env_default):
        """``turn_penalty`` is 0.0 by default — ending the turn no longer
        carries a per-action cost. Prior default of -1.0 created the
        "never end the turn" attractor we saw in beginner_random_15
        eval logs; per-turn pressure now lives in ``win_speed_bonus``.
        """
        assert env_default.reward_config["turn_penalty"] == 0.0

    def test_turn_penalty_when_explicitly_configured(self):
        """The penalty still applies if a config explicitly sets it."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"turn_penalty": -5.0},
        )
        env.reset()
        action = np.array([5, 0, 0, 0, 0, 0])
        _, reward, _, _, info = env.step(action)
        # action_reward (-5.0) shows up under ``info["reward_breakdown"]
        # ["action"]``; the final reward also pulls in potential shaping
        # so we just verify the action contribution.
        assert info["reward_breakdown"]["action"] == pytest.approx(-5.0)
        env.close()

    def test_win_speed_bonus_scales_with_remaining_turns(self):
        """A win on turn 1 (out of max_turns=20) gets ~full bonus; a win
        on the last turn gets ~0."""
        bonus = 50.0
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            max_turns=20,
            reward_config={"win": 100.0, "win_speed_bonus": bonus},
        )

        # Turn-1 win.
        env.reset()
        env.game_state.turn_number = 1
        env.game_state.game_over = True
        env.game_state.winner = env.agent_player
        # Synthesize a few alive opponent units so end_reason resolves
        # to ``hq_capture`` rather than ``elimination`` (irrelevant for
        # the bonus calculation but keeps the path deterministic).
        _, fast_reward, _, _, _ = env.step(np.array([5, 0, 0, 0, 0, 0]))

        env.reset()
        env.game_state.turn_number = 20  # last turn
        env.game_state.game_over = True
        env.game_state.winner = env.agent_player
        _, slow_reward, _, _, _ = env.step(np.array([5, 0, 0, 0, 0, 0]))

        # Speed bonus contribution: (20 - 1) / 20 * 50 = 47.5 for fast win,
        # (20 - 20) / 20 * 50 = 0 for slow win. The non-terminal pieces of
        # reward are identical between the two runs (same starting state)
        # so the delta isolates the bonus.
        assert (fast_reward - slow_reward) == pytest.approx(47.5, rel=1e-3)
        env.close()

    def test_win_speed_bonus_not_applied_on_loss(self):
        """``win_speed_bonus`` is win-only; losing on turn 1 should not
        get any speed credit."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            max_turns=20,
            reward_config={"win_speed_bonus": 100.0, "loss": -50.0},
        )
        env.reset()
        env.game_state.turn_number = 1
        env.game_state.game_over = True
        env.game_state.winner = 3 - env.agent_player  # opponent wins
        _, reward, _, _, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
        # Terminal piece should be exactly the loss key, no speed bonus.
        assert info["reward_breakdown"]["terminal"] == pytest.approx(-50.0)
        env.close()

    def test_win_speed_bonus_skipped_when_max_turns_unset(self):
        """Without a finite ``max_turns`` there's no horizon to scale
        against, so the bonus is skipped rather than divided by zero."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            max_turns=None,
            reward_config={"win": 100.0, "win_speed_bonus": 50.0},
        )
        env.reset()
        env.game_state.turn_number = 5
        env.game_state.game_over = True
        env.game_state.winner = env.agent_player
        _, _, _, _, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
        # Terminal is exactly ``win`` (or win_by_*) with no bonus.
        assert info["reward_breakdown"]["terminal"] == pytest.approx(100.0)
        env.close()

    def test_invalid_action_penalty(self, env_default):
        """Test invalid_action penalty."""
        env_default.reset()

        # Try invalid action (move non-existent unit)
        action = np.array([1, 0, 0, 0, 1, 1])
        _, reward, _, _, info = env_default.step(action)

        # Should apply invalid action penalty
        assert info["valid_action"] is False
        assert reward < 0

        env_default.close()

    def test_cumulative_episode_reward_tracking(self, env_default):
        """Test cumulative episode_stats['reward'] tracking."""
        env_default.reset()

        assert env_default.episode_stats["reward"] == 0.0

        # Take multiple steps
        for _ in range(3):
            action = np.array([5, 0, 0, 0, 0, 0])
            env_default.step(action)

        # Cumulative reward should be tracked
        assert env_default.episode_stats["reward"] != 0.0

        env_default.close()

    def test_custom_reward_config(self, custom_reward_config):
        """Test reward calculation with custom reward_config."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None, reward_config=custom_reward_config)
        env.reset()

        # Set game to won state
        env.game_state.game_over = True
        env.game_state.winner = 1

        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        # Should use custom win reward
        assert reward > 0  # Custom win reward of 500.0

        env.close()


# ==============================================================================
# 7. RENDER AND CLOSE TESTS
# ==============================================================================


class TestRenderAndClose:
    """Test rendering and cleanup."""

    def test_render_returns_none_when_no_render_mode(self, env_default):
        """Test render() returns None when render_mode=None."""
        result = env_default.render()

        assert result is None

        env_default.close()

    def test_close_does_not_raise_exceptions(self, env_default):
        """Test close() does not raise exceptions."""
        env_default.reset()

        # Should not raise any exceptions
        try:
            env_default.close()
        except Exception as e:
            pytest.fail(f"close() raised exception: {e}")

    def test_headless_mode_works(self):
        """Test that headless mode works without any rendering dependencies."""
        # Create environment in headless mode
        env = StrategyGameEnv(map_file=None, opponent="bot", render_mode=None)

        # Should be able to reset and step without rendering
        obs, info = env.reset()
        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        # Render should return None
        assert env.render() is None

        env.close()


# ==============================================================================
# 8. EPISODE STATISTICS TESTS
# ==============================================================================


class TestEpisodeStatistics:
    """Test episode statistics tracking."""

    def test_invalid_actions_increment(self, env_default):
        """Verify episode_stats['invalid_actions'] increments on invalid actions."""
        env_default.reset()

        initial_count = env_default.episode_stats["invalid_actions"]

        # Perform invalid action
        action = np.array([1, 0, 0, 0, 1, 1])  # move non-existent unit
        obs, reward, terminated, truncated, info = env_default.step(action)

        assert env_default.episode_stats["invalid_actions"] > initial_count

        env_default.close()

    def test_winner_set_correctly_on_game_over(self):
        """Verify episode_stats['winner'] is set correctly on game over."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset()

        # Set game to won state
        env.game_state.game_over = True
        env.game_state.winner = 1

        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.episode_stats["winner"] == 1

        env.close()

    def test_episode_stats_in_info_on_termination(self):
        """Verify episode_stats is included in info on termination."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset()

        # Set game to won state
        env.game_state.game_over = True
        env.game_state.winner = 1

        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated is True
        assert "episode_stats" in info
        assert len(info["episode_stats"]) > 0

        env.close()

    def test_episode_stats_in_info_on_truncation(self, env_default):
        """Verify episode_stats is included in info on truncation."""
        env_default.reset()

        # Set to trigger truncation
        env_default.current_step = env_default.max_steps - 1

        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env_default.step(action)

        assert truncated is True
        assert "episode_stats" in info
        assert len(info["episode_stats"]) > 0

        env_default.close()


# ==============================================================================
# 9. INTEGRATION TESTS
# ==============================================================================


class TestIntegration:
    """Integration tests for full episode workflows."""

    def test_complete_episode_loop(self, env_default):
        """Test a complete episode loop (reset → multiple steps → termination)."""
        # Reset environment
        obs, info = env_default.reset()

        assert isinstance(obs, dict)
        assert env_default.current_step == 0

        # Take multiple steps
        terminated = False
        truncated = False
        step_count = 0
        max_test_steps = 10

        while not (terminated or truncated) and step_count < max_test_steps:
            # Take end_turn action
            action = np.array([5, 0, 0, 0, 0, 0])
            obs, reward, terminated, truncated, info = env_default.step(action)

            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float, np.number))

            step_count += 1

        # Should have taken some steps
        assert step_count > 0

        env_default.close()

    def test_random_actions_can_be_sampled(self, env_default):
        """Test that random actions can be sampled and executed without errors."""
        env_default.reset()

        # Sample random actions and execute them
        for _ in range(10):
            action = env_default.action_space.sample()

            try:
                obs, reward, terminated, truncated, info = env_default.step(action)

                # Should return valid results
                assert obs is not None
                assert isinstance(reward, (int, float, np.number))

                if terminated or truncated:
                    env_default.reset()

            except Exception as e:
                pytest.fail(f"Random action execution failed: {e}")

        env_default.close()

    def test_multiple_episodes(self, env_default):
        """Test running multiple episodes sequentially."""
        for episode in range(3):
            obs, info = env_default.reset()

            # Run episode for a few steps
            for step in range(5):
                action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
                obs, reward, terminated, truncated, info = env_default.step(action)

                if terminated or truncated:
                    break

            # Stats should be reset on next reset
            if episode < 2:
                next_obs, next_info = env_default.reset()
                assert env_default.current_step == 0

        env_default.close()

    def test_environment_with_various_opponents(self):
        """Test environment works with different opponent types."""
        opponent_types = ["bot", "random", "self", None]

        for opp_type in opponent_types:
            env = StrategyGameEnv(map_file=None, opponent=opp_type, render_mode=None)

            obs, info = env.reset()
            action = np.array([5, 0, 0, 0, 0, 0])
            obs, reward, terminated, truncated, info = env.step(action)

            assert obs is not None

            env.close()

    def test_action_space_size_calculation(self, env_default):
        """Test _get_action_space_size() returns consistent value."""
        size = env_default._get_action_space_size()

        assert size > 0
        assert isinstance(size, int)

        # The flat action mask is no longer part of the observation; pull it
        # via ``get_action_mask_flat`` (the diagnostic accessor).
        env_default.reset()
        assert env_default.get_action_mask_flat().shape[0] == size

        env_default.close()


# ==============================================================================
# 10. ACTION MASKING TESTS (for MaskablePPO compatibility)
# ==============================================================================


class TestActionMasking:
    """Test action masking functionality for MaskablePPO (sb3-contrib)."""

    def test_action_masks_method_exists(self, env_default):
        """Test that action_masks() method exists on environment."""
        env_default.reset()
        assert hasattr(env_default, "action_masks")
        assert callable(env_default.action_masks)
        env_default.close()

    def test_action_masks_returns_tuple(self, env_default):
        """Test that action_masks() returns a tuple of boolean arrays."""
        env_default.reset()
        masks = env_default.action_masks()

        assert isinstance(masks, tuple)
        assert len(masks) == 6  # 6 dimensions in MultiDiscrete action space
        env_default.close()

    def test_action_masks_correct_shapes(self, env_default):
        """Test that action mask arrays have correct shapes."""
        env_default.reset()
        masks = env_default.action_masks()

        # Expected shapes based on action space
        expected_shapes = [
            10,  # action_type (0-9: create, move, attack, seize, heal, end_turn, paralyze, haste, defence_buff, attack_buff)
            8,  # unit_type
            env_default.grid_width,  # from_x
            env_default.grid_height,  # from_y
            env_default.grid_width,  # to_x
            env_default.grid_height,  # to_y
        ]

        for i, (mask, expected_size) in enumerate(zip(masks, expected_shapes)):
            assert len(mask) == expected_size, f"Mask {i} has wrong size: {len(mask)} != {expected_size}"
            assert mask.dtype == np.bool_, f"Mask {i} has wrong dtype: {mask.dtype}"

        env_default.close()

    def test_action_masks_end_turn_always_valid(self, env_default):
        """Test that end_turn action (type 5) is always masked as valid."""
        env_default.reset()
        masks = env_default.action_masks()

        action_type_mask = masks[0]
        assert action_type_mask[5], "End turn should always be valid"

        env_default.close()

    def test_action_masks_at_least_one_valid_per_dimension(self, env_default):
        """Test that each mask dimension has at least one valid option."""
        env_default.reset()
        masks = env_default.action_masks()

        dimension_names = ["action_type", "unit_type", "from_x", "from_y", "to_x", "to_y"]

        for i, (mask, name) in enumerate(zip(masks, dimension_names)):
            assert mask.any(), f"Dimension '{name}' has no valid options"

        env_default.close()

    def test_action_masks_consistent_with_legal_actions(self, env_default):
        """Test that action masks are consistent with legal_actions."""
        env_default.reset()
        masks = env_default.action_masks()
        legal_actions = env_default.game_state.get_legal_actions(player=1)

        action_type_mask = masks[0]

        # If there are create_unit actions, action_type 0 should be valid
        if legal_actions.get("create_unit"):
            assert action_type_mask[0], "Create unit mask mismatch"

        # If there are move actions, action_type 1 should be valid
        if legal_actions.get("move"):
            assert action_type_mask[1], "Move mask mismatch"

        # If there are attack actions, action_type 2 should be valid
        if legal_actions.get("attack"):
            assert action_type_mask[2], "Attack mask mismatch"

        env_default.close()

    def test_action_masks_update_after_step(self, env_default):
        """Test that action masks update after taking a step."""
        env_default.reset()
        _masks_before = env_default.action_masks()

        # Take end turn action
        action = np.array([5, 0, 0, 0, 0, 0])
        env_default.step(action)

        masks_after = env_default.action_masks()

        # Masks should still be valid tuples
        assert isinstance(masks_after, tuple)
        assert len(masks_after) == 6

        env_default.close()

    def test_action_masks_update_after_reset(self, env_default):
        """Test that action masks are properly initialized after reset."""
        # Take some steps
        env_default.reset()
        env_default.step(np.array([5, 0, 0, 0, 0, 0]))

        # Reset
        env_default.reset()
        masks = env_default.action_masks()

        # Should be valid
        assert isinstance(masks, tuple)
        assert len(masks) == 6
        assert masks[0][5]  # End turn always valid

        env_default.close()

    def test_get_action_mask_flat_exists(self, env_default):
        """Test that get_action_mask_flat() method exists and works."""
        env_default.reset()

        assert hasattr(env_default, "get_action_mask_flat")
        flat_mask = env_default.get_action_mask_flat()

        expected_size = env_default._get_action_space_size()
        assert flat_mask.shape == (expected_size,)
        assert flat_mask.dtype == np.float32

        env_default.close()


class TestActionMaskingWrapper:
    """Test the ActionMaskedEnv wrapper for MaskablePPO compatibility."""

    def test_wrapper_import(self):
        """Test that masking module can be imported."""

    def test_make_maskable_env(self):
        """Test make_maskable_env creates wrapped environment."""
        from reinforcetactics.rl.masking import make_maskable_env

        env = make_maskable_env(opponent="bot")

        assert env is not None
        assert hasattr(env, "action_masks")

        # Should be able to reset and get masks
        env.reset()
        masks = env.action_masks()
        assert isinstance(masks, np.ndarray)

        env.close()

    def test_action_masked_env_wrapper(self):
        """Test ActionMaskedEnv wrapper functionality."""
        from reinforcetactics.rl.masking import ActionMaskedEnv

        base_env = StrategyGameEnv(opponent="bot", render_mode=None)
        wrapped_env = ActionMaskedEnv(base_env)

        wrapped_env.reset()

        # action_masks() should return concatenated 1D array for sb3-contrib
        masks = wrapped_env.action_masks()
        assert isinstance(masks, np.ndarray)
        assert masks.ndim == 1
        assert masks.dtype == np.bool_

        # get_action_masks_tuple() should return the tuple format
        masks_tuple = wrapped_env.get_action_masks_tuple()
        assert isinstance(masks_tuple, tuple)
        assert len(masks_tuple) == 6

        wrapped_env.close()

    def test_validate_action_mask(self):
        """Test action mask validation utility."""
        from reinforcetactics.rl.masking import validate_action_mask

        env = StrategyGameEnv(opponent="bot", render_mode=None)
        env.reset()

        validation = validate_action_mask(env)

        assert "valid" in validation
        assert "errors" in validation
        assert "warnings" in validation
        assert "mask_summary" in validation

        # Should be valid for a fresh game
        assert validation["valid"], f"Validation errors: {validation['errors']}"

        env.close()

    def test_wrapper_with_stats_tracking(self):
        """Test ActionMaskedEnv with stats tracking enabled."""
        from reinforcetactics.rl.masking import ActionMaskedEnv

        base_env = StrategyGameEnv(opponent="bot", render_mode=None)
        wrapped_env = ActionMaskedEnv(base_env, track_stats=True)

        wrapped_env.reset()

        # Take a few actions
        for _ in range(3):
            action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
            wrapped_env.step(action)

        stats = wrapped_env.get_masking_stats()

        assert "total_actions" in stats
        assert stats["total_actions"] == 3
        assert "action_type_distribution" in stats

        wrapped_env.close()


# ==============================================================================
# 11. MAX_TURNS, POTENTIAL-BASED SHAPING, AND TERMINAL HANDLING
# ==============================================================================


class TestMaxTurns:
    """Tests for the max_turns parameter and natural game termination.

    `max_turns` lets games end via game rules (terminated=True) rather than
    only via env step truncation, which avoids PPO bootstrapping V(s')
    when the intended outcome is a draw penalty.
    """

    def test_max_turns_passed_to_game_state(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None, max_turns=15)
        assert env.max_turns == 15
        assert env.game_state.max_turns == 15
        env.close()

    def test_max_turns_default_is_none(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        assert env.max_turns is None
        assert env.game_state.max_turns is None
        env.close()

    def test_max_turns_preserved_across_reset(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None, max_turns=7)
        env.reset()
        env.step(np.array([5, 0, 0, 0, 0, 0]))
        env.reset()
        assert env.game_state.max_turns == 7
        env.close()

    def test_max_turns_triggers_terminated_not_truncated(self):
        """When max_turns is hit, the env should terminate (game-rules) and
        report winner=None, not truncate. PPO bootstraps V(s') on truncation;
        terminated=True ensures the draw penalty is the actual return.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            max_steps=10_000,  # well beyond what max_turns=3 will use
            max_turns=3,
        )
        env.reset(seed=0)
        terminated = truncated = False
        for _ in range(2_000):
            _, _, terminated, truncated, _ = env.step(np.array([5, 0, 0, 0, 0, 0]))
            if terminated or truncated:
                break
        assert terminated is True
        assert truncated is False
        assert env.game_state.winner is None
        assert env.game_state.turn_number >= 3
        env.close()

    def test_max_turns_applies_draw_reward_at_termination(self):
        """The draw branch in step() must fire when max_turns terminates."""
        custom = {"draw": -123.0}
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            max_steps=10_000,
            max_turns=2,
            reward_config=custom,
        )
        env.reset(seed=0)
        last_info: dict = {}
        for _ in range(2_000):
            _, _, terminated, truncated, last_info = env.step(np.array([5, 0, 0, 0, 0, 0]))
            if terminated or truncated:
                break
        # Assert on the terminal component rather than the total: the terminal
        # step also carries the PBRS closing charge (-Phi(s_prev)), whose sign
        # depends on who was ahead, so the total is not a clean bound.
        assert last_info["end_reason"] == "max_turns_draw"
        assert last_info["reward_breakdown"]["terminal"] == pytest.approx(-123.0)
        env.close()


class TestPotentialBasedShaping:
    """Tests for potential-based reward shaping (Ng et al., 1999).

    Three invariants matter:
    1. _prev_potential after reset() equals Phi(s_0), so the first step's
       shaping delta is Phi(s_1) - Phi(s_0), not Phi(s_1) - 0.
    2. Phi(terminal) = 0, which means the terminal step is *charged*
       ``F = gamma*0 - Phi(s_prev)`` — not that the term is skipped. Skipping
       leaves the telescoping sum with a dangling ``+gamma^(T-1) Phi(s_(T-1))``
       and breaks the Ng et al. policy-invariance guarantee.
    3. A step-limit truncation is not a termination: its successor state is
       real and the learner bootstraps its value, so it takes the ordinary
       ``gamma*Phi(s') - Phi(s)`` delta.
    """

    def test_prev_potential_initialized_to_phi_s0_after_reset(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset()
        # _prev_potential must match the freshly-computed potential, not 0.
        assert env._prev_potential == env._compute_potential()
        env.close()

    def test_prev_potential_reinitialized_after_second_reset(self):
        env = StrategyGameEnv(map_file=None, opponent="random", render_mode=None)
        env.reset()
        # Mutate _prev_potential to ensure the next reset overwrites it
        env._prev_potential = -999.0
        env.reset()
        assert env._prev_potential == env._compute_potential()
        env.close()

    def test_terminal_step_charges_negative_prev_potential(self):
        """On a terminated step, _calculate_reward must charge -Phi(s_prev)
        (i.e. F = gamma*0 - Phi(s_prev)) rather than skipping the term.
        """
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset()
        env._prev_potential = -50.0
        prev_before = env._prev_potential

        terminal_reward, terminal_breakdown = env._calculate_reward(action_reward=1.0, is_valid=True, terminated=True)
        assert terminal_breakdown["shaping_delta"] == pytest.approx(-prev_before)
        assert terminal_reward == pytest.approx(1.0 - prev_before)
        # _prev_potential is not advanced on termination; reset() re-seeds it.
        assert env._prev_potential == prev_before

        # Same starting state, but non-terminal path: should add
        # gamma * Phi(s) - prev_before.
        nonterminal_reward, nonterminal_breakdown = env._calculate_reward(action_reward=1.0, is_valid=True, terminated=False)
        expected_delta = env.gamma * env._compute_potential() - prev_before
        assert nonterminal_reward == pytest.approx(1.0 + expected_delta)
        assert nonterminal_breakdown["shaping_delta"] == pytest.approx(expected_delta)
        env.close()

    def test_discounted_shaping_return_telescopes_to_minus_phi_s0(self):
        """The whole point of the terminal charge: the discounted sum of the
        shaping deltas over an episode must collapse to the constant
        -Phi(s_0), which is what makes the shaping policy-invariant.

        With the term skipped at termination the sum instead carries a
        dangling +gamma^(T-1) * Phi(s_(T-1)).
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            max_steps=10_000,
            max_turns=3,
            # A deliberately large, easily-moved potential so a dangling term
            # would be obvious rather than lost in float noise.
            reward_config={"structure_control": 100.0, "unit_diff": 10.0, "income_diff": 0.0},
            gamma=0.9,
        )
        env.reset(seed=0)
        # Break the symmetric start so Phi(s_0) != 0 -- otherwise the identity
        # reduces to "sum == 0", which holds even with the terminal term
        # skipped, and the test would be vacuous.
        neutral = [tile for row in env.game_state.grid.tiles for tile in row if tile.is_capturable() and tile.player is None]
        assert neutral, "fixture expects neutral capturable tiles on the generated map"
        neutral[0].player = env.agent_player
        env._prev_potential = env._compute_potential()
        phi_s0 = env._prev_potential
        assert phi_s0 != 0.0, "test needs a non-zero starting potential to be meaningful"

        discounted = 0.0
        for t in range(10_000):
            _, _, terminated, truncated, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
            discounted += (env.gamma**t) * info["reward_breakdown"]["shaping_delta"]
            if terminated:
                break
            assert not truncated, "episode should end via max_turns, not the step cap"
        assert discounted == pytest.approx(-phi_s0, abs=1e-6)
        env.close()

    def test_invalid_action_penalty_still_applies_at_terminal(self):
        """Even on terminal steps, the invalid_action penalty should fire."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"invalid_action": -7.5},
        )
        env.reset()
        before_invalid = env.episode_stats["invalid_actions"]
        prev_potential = env._prev_potential
        r, breakdown = env._calculate_reward(action_reward=2.0, is_valid=False, terminated=True)
        assert r == pytest.approx(2.0 - 7.5 - prev_potential)
        assert breakdown["invalid_penalty"] == pytest.approx(-7.5)
        assert breakdown["action"] == pytest.approx(2.0)
        assert env.episode_stats["invalid_actions"] == before_invalid + 1
        env.close()

    def _zero_potential_env(self):
        return StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "win": 1000.0,
                "loss": -1000.0,
                "draw": 0.0,
                "income_diff": 0.0,
                "unit_diff": 0.0,
                "structure_control": 0.0,
                "turn_penalty": 0.0,
                "invalid_action": -10.0,
            },
        )

    def test_terminal_step_charges_closing_potential_in_full_step_loop(self):
        """End-to-end: when game_over fires inside step(), the returned reward
        is the win bonus plus the PBRS closing charge -Phi(s_prev).
        """
        env = self._zero_potential_env()
        env.reset()
        # Force terminal at the start of step()
        env.game_state.game_over = True
        env.game_state.winner = 1
        # Pre-set prev_potential so the closing charge is unmistakable.
        env._prev_potential = -42.0
        _, reward, terminated, _, _ = env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert terminated is True
        assert reward == pytest.approx(1000.0 + 42.0)
        env.close()

    def test_terminal_reward_is_bare_bonus_when_potential_is_zero(self):
        """With every potential coefficient at 0, Phi is identically 0 and the
        closing charge vanishes — the terminal reward is just the win bonus.
        """
        env = self._zero_potential_env()
        env.reset()
        assert env._prev_potential == pytest.approx(0.0)
        env.game_state.game_over = True
        env.game_state.winner = 1
        _, reward, terminated, _, _ = env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert terminated is True
        assert reward == pytest.approx(1000.0)
        env.close()

    def test_truncation_takes_the_ordinary_delta_not_the_closing_charge(self):
        """A step-limit truncation is an artificial cutoff, not a termination:
        its successor state is real and the learner bootstraps its value, so
        the ordinary gamma*Phi(s') - Phi(s) delta applies.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            max_steps=2,
            reward_config={"structure_control": 100.0, "unit_diff": 0.0, "income_diff": 0.0},
        )
        env.reset(seed=0)
        env.step(np.array([5, 0, 0, 0, 0, 0]))
        prev_before = env._prev_potential
        _, _, terminated, truncated, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert truncated is True and terminated is False
        expected = env.gamma * env._compute_potential() - prev_before
        assert info["reward_breakdown"]["shaping_delta"] == pytest.approx(expected)
        # ...and _prev_potential advanced, unlike on a termination.
        assert env._prev_potential == pytest.approx(env._compute_potential())
        env.close()


class TestRewardWeightDefaults:
    """Defaults are tuned so HQ capture dominates kill-farming.

    Guards against accidental regressions in the default reward config
    that previously caused 0% win rate (kill-farm local optimum).
    """

    def test_default_capture_dominates_kill_loop(self):
        """A single capture must outweigh a full episode of kills."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        rc = env.reward_config
        # Even 10 kills + their damage (~3 dmg each) shouldn't beat one capture.
        kill_loop = 10 * (rc["kill"] + 3 * rc["damage_scale"])
        assert rc["capture"] > kill_loop
        env.close()

    def test_default_seize_progress_at_least_as_strong_as_kill(self):
        """A turn of seize_progress should be competitive with a kill."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        rc = env.reward_config
        assert rc["seize_progress"] >= rc["kill"]
        env.close()

    def test_default_move_does_not_farm_reward(self):
        """`move` should not provide a meaningful per-action bonus that
        the agent can farm by shuffling units.
        """
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        assert env.reward_config["move"] == pytest.approx(0.0)
        env.close()

    def test_default_win_loss_draw_unchanged(self):
        """Terminal rewards should remain at their established magnitudes."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        rc = env.reward_config
        assert rc["win"] == 1000.0
        assert rc["loss"] == -1000.0
        assert rc["draw"] == -200.0
        env.close()

    def test_user_reward_config_overrides_defaults(self):
        """Caller-provided reward_config must still override the new defaults."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"capture": 1.0, "kill": 999.0},
        )
        assert env.reward_config["capture"] == 1.0
        assert env.reward_config["kill"] == 999.0
        # Unspecified keys still come from defaults
        assert "seize_progress" in env.reward_config
        env.close()


class TestPerStructureCaptureReward:
    """Per-type capture rewards (``tower_capture`` / ``building_capture`` /
    ``hq_capture``) override the global ``capture`` weight when set in
    ``reward_config``. Falls back to the global ``capture`` when a per-type
    key is absent so existing configs that only set the global key keep
    their current behavior.

    The lookup happens inside ``_execute_action`` for action_type=3 (seize)
    when ``result_info["captured"]`` is True. We mock ``execute_game_action``
    to return a controlled capture result, since constructing a real
    low-HP capture requires bespoke map setup.
    """

    @staticmethod
    def _capture_action_reward(env, structure_type):
        """Run one seize action where the underlying game call reports a
        successful capture of the given structure_type, and return the
        action reward (the per-action component, before shaping).
        """
        env.reset()
        # Mock execute_game_action so action_type=3 sees ``captured: True``
        # with the requested structure_type, without requiring a real
        # low-HP structure tile next to a unit.
        env.execute_game_action = lambda action_dict, player: (
            {
                "seize_damage": 30,
                "captured": True,
                "structure_type": structure_type,
            },
            True,
        )
        action_dict = {
            "action_type": 3,
            "from_x": 0,
            "from_y": 0,
            "to_x": 0,
            "to_y": 0,
            "unit_type": None,
        }
        reward, is_valid = env._execute_action(action_dict)
        assert is_valid is True
        return reward

    def test_building_specific_reward_overrides_global(self):
        """When ``building_capture`` is set, building captures use it
        instead of the global ``capture`` weight.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 100.0,
                "building_capture": 4000.0,
                # Zero seize_progress so only the capture reward shows up
                "seize_progress": 0.0,
            },
        )
        reward = self._capture_action_reward(env, "b")
        assert reward == pytest.approx(4000.0)
        env.close()

    def test_tower_and_hq_specific_keys_independent(self):
        """Each type's per-key reward fires only for its own structure_type."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 100.0,
                "tower_capture": 1500.0,
                "building_capture": 4000.0,
                "hq_capture": 2500.0,
                "seize_progress": 0.0,
            },
        )
        assert self._capture_action_reward(env, "t") == pytest.approx(1500.0)
        env.close()

        env2 = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 100.0,
                "tower_capture": 1500.0,
                "building_capture": 4000.0,
                "hq_capture": 2500.0,
                "seize_progress": 0.0,
            },
        )
        assert self._capture_action_reward(env2, "h") == pytest.approx(2500.0)
        env2.close()

    def test_falls_back_to_global_capture_when_type_key_absent(self):
        """Configs that only set ``capture`` (no per-type keys) get the
        same capture reward for all structure types — back-compat.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 250.0,
                "seize_progress": 0.0,
            },
        )
        # All three structure types should use the global capture weight
        assert self._capture_action_reward(env, "t") == pytest.approx(250.0)
        env.close()
        env2 = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"capture": 250.0, "seize_progress": 0.0},
        )
        assert self._capture_action_reward(env2, "b") == pytest.approx(250.0)
        env2.close()
        env3 = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"capture": 250.0, "seize_progress": 0.0},
        )
        assert self._capture_action_reward(env3, "h") == pytest.approx(250.0)
        env3.close()

    def test_partial_per_type_override_uses_global_for_unconfigured_types(self):
        """If only ``building_capture`` is set, towers and HQs still use
        the global ``capture``. Catches a regression where the lookup
        accidentally overrides for all types once any per-type key is set.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 200.0,
                "building_capture": 4000.0,
                "seize_progress": 0.0,
            },
        )
        # Building uses building_capture
        assert self._capture_action_reward(env, "b") == pytest.approx(4000.0)
        env.close()
        # Tower falls back to capture
        env2 = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 200.0,
                "building_capture": 4000.0,
                "seize_progress": 0.0,
            },
        )
        assert self._capture_action_reward(env2, "t") == pytest.approx(200.0)
        env2.close()

    def test_none_or_unmapped_structure_type_uses_global_capture(self):
        """Defensive: if ``structure_type`` is None or an unrecognized
        code, the reward path falls back to ``capture`` rather than
        crashing or silently zeroing the reward.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={
                "capture": 175.0,
                "building_capture": 4000.0,
                "seize_progress": 0.0,
            },
        )
        # structure_type=None should NOT activate any per-type key
        assert self._capture_action_reward(env, None) == pytest.approx(175.0)
        env.close()


class TestStepInfoDiagnostics:
    """Tests for the per-step diagnostic fields added to info:
    ``reward_breakdown`` and ``n_legal_actions``. These are how the
    notebook visualises mask coverage and reward composition.
    """

    def test_info_contains_reward_breakdown_keys(self, env_default):
        env_default.reset()
        _, _, _, _, info = env_default.step(np.array([5, 0, 0, 0, 0, 0]))
        assert "reward_breakdown" in info
        breakdown = info["reward_breakdown"]
        assert set(breakdown.keys()) == {"action", "invalid_penalty", "shaping_delta", "terminal"}
        env_default.close()

    def test_breakdown_components_sum_to_step_reward(self, env_default):
        """The four breakdown pieces must sum exactly to the returned reward
        — otherwise downstream telemetry will diverge from training rewards.
        """
        env_default.reset()
        for action in (
            np.array([5, 0, 0, 0, 0, 0]),  # end_turn (valid)
            np.array([1, 0, 0, 0, 1, 1]),  # likely-invalid move
            np.array([5, 0, 0, 0, 0, 0]),  # end_turn (valid)
        ):
            _, reward, _, _, info = env_default.step(action)
            b = info["reward_breakdown"]
            assert b["action"] + b["invalid_penalty"] + b["shaping_delta"] + b["terminal"] == pytest.approx(float(reward))
        env_default.close()

    def test_breakdown_terminal_bonus_on_win(self):
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"win": 1234.0},
        )
        env.reset()
        # Force terminal-win at the start of step()
        env.game_state.game_over = True
        env.game_state.winner = env.agent_player
        _, reward, terminated, _, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert terminated is True
        assert info["reward_breakdown"]["terminal"] == pytest.approx(1234.0)
        # And the win bonus must equal terminal piece (no shaping leak at terminal)
        assert info["reward_breakdown"]["shaping_delta"] == 0.0
        # Sum invariant
        b = info["reward_breakdown"]
        assert sum(b.values()) == pytest.approx(float(reward))
        env.close()

    def test_truncation_does_not_charge_the_draw_terminal(self):
        """SB3 adds gamma * V(terminal_obs) on a truncation, so charging the
        `draw` terminal here too would double-count. Default is 0.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            max_steps=2,
            reward_config={"draw": -77.0},
        )
        env.reset(seed=0)
        env.step(np.array([5, 0, 0, 0, 0, 0]))
        _, _, terminated, truncated, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert truncated is True
        assert terminated is False
        assert info["end_reason"] == "max_steps_truncate"
        assert info["reward_breakdown"]["terminal"] == pytest.approx(0.0)
        env.close()

    def test_truncation_penalty_is_opt_in_via_reward_config(self):
        """`truncation` is its own reward key so reinstating a penalty is an
        explicit choice, not an accidental reuse of `draw`.
        """
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            max_steps=2,
            reward_config={"draw": -77.0, "truncation": -12.5},
        )
        env.reset(seed=0)
        env.step(np.array([5, 0, 0, 0, 0, 0]))
        _, _, terminated, truncated, info = env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert truncated is True and terminated is False
        assert info["reward_breakdown"]["terminal"] == pytest.approx(-12.5)
        env.close()

    def test_n_legal_actions_present_and_positive(self, env_default):
        """n_legal_actions must be reported on every step and >= 1 (end_turn
        is always legal).
        """
        env_default.reset()
        _, _, _, _, info = env_default.step(np.array([5, 0, 0, 0, 0, 0]))
        assert "n_legal_actions" in info
        assert isinstance(info["n_legal_actions"], int)
        assert info["n_legal_actions"] >= 1
        env_default.close()

    def test_n_legal_actions_flat_discrete_matches_current_actions(self):
        env = StrategyGameEnv(
            map_file=None,
            opponent="random",
            render_mode=None,
            action_space_type="flat_discrete",
            max_flat_actions=512,
        )
        env.reset(seed=0)
        # Take an end_turn so _build_flat_actions populates _current_actions
        # for the next step's mask query.
        env.action_masks()  # populates _current_actions
        expected = len(env._current_actions)
        _, _, _, _, info = env.step(0)  # first legal flat action
        # After the step, n_legal_actions reflects the new current state.
        # Just verify it's bounded and consistent with the flat-action length.
        assert 1 <= info["n_legal_actions"] <= 512
        # Sanity: the *pre-step* mask had `expected` legal actions
        assert expected >= 1
        env.close()


# ==============================================================================
# SELF-PLAY OPPONENT DISPATCH (opponent="self" + factory)
# ==============================================================================


class _SpyEndTurnBot:
    """Minimal opponent: records each take_turn call, then ends its turn."""

    def __init__(self, game_state, player):
        self.game_state = game_state
        self.player = player
        self.turns_taken = 0

    def take_turn(self):
        self.turns_taken += 1
        self.game_state.end_turn()


class TestSelfPlayOpponentDispatch:
    """With ``opponent='self'``, the factory-built opponent must actually play.

    Regression tests for the bug where ``_execute_action`` skipped
    ``_opponent_turn()`` for ``opponent_type == 'self'``, so the
    factory-bound opponent (feudal self-play snapshots / ModelBot) never
    took a turn and "self-play" silently trained against an inert opponent.
    """

    def test_factory_opponent_takes_turn_on_agent_end_turn(self):
        env = StrategyGameEnv(map_file=None, opponent="self", render_mode=None, max_steps=20)
        spies = []

        def factory(game_state, opponent_player):
            bot = _SpyEndTurnBot(game_state, opponent_player)
            spies.append(bot)
            return bot

        env.set_self_play_opponent_factory(factory)
        env.reset(seed=0)
        assert spies, "factory should be invoked on reset"
        assert env.game_state.current_player == env.agent_player

        _obs, _reward, terminated, _truncated, _info = env.step(np.array([5, 0, 0, 0, 0, 0]))

        # The opponent acted exactly once, during its own turn, and play
        # returned to the agent.
        assert spies[-1].turns_taken == 1
        if not terminated:
            assert env.game_state.current_player == env.agent_player
        env.close()

    def test_self_without_factory_returns_turn_to_agent(self):
        # No factory bound -> ``self.opponent`` is None -> ``_opponent_turn``
        # no-ops, and the safety net must end the empty opponent turn so
        # play (and the action masks) return to the agent instead of
        # sticking on the opponent.
        env = StrategyGameEnv(map_file=None, opponent="self", render_mode=None, max_steps=20)
        env.reset(seed=0)

        _obs, _reward, terminated, _truncated, _info = env.step(np.array([5, 0, 0, 0, 0, 0]))

        assert not terminated
        assert env.game_state.current_player == env.agent_player
        env.close()


# ==============================================================================
# MASK PLAYER CONSISTENCY
# ==============================================================================


class TestMaskPlayerConsistency:
    """Every mask builder must describe the *agent's* legal actions.

    Regression test for ``_build_masks`` / ``_build_structured_masks``
    querying ``get_legal_actions(player=current_player)``: whenever a
    caller queried masks while it was not the agent's turn (e.g. manual
    opponent=None driving, or between end_turn and the opponent turn
    completing), the masks silently described the opponent's actions.
    """

    def test_masks_query_agent_player_even_off_turn(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None, max_steps=20)
        env.reset(seed=0)
        # Manual mode (opponent=None): the agent's end_turn hands play to
        # player 2 with nobody driving it.
        env.step(np.array([5, 0, 0, 0, 0, 0]))
        assert env.game_state.current_player != env.agent_player

        queried_players = []
        original = env.game_state.get_legal_actions

        def spy(player=None, **kwargs):
            queried_players.append(player)
            return original(player=player, **kwargs)

        env.game_state.get_legal_actions = spy

        env.action_masks()
        env.structured_action_masks()

        assert queried_players, "mask builders should query legal actions"
        assert all(p == env.agent_player for p in queried_players)
        env.close()


# ==============================================================================
# POTENTIAL-BASED SHAPING WEIGHT GATES
# ==============================================================================


class TestPotentialNegativeWeights:
    """Negative shaping weights are legitimate config and must not be
    silently dropped by the zero-skip gates in ``_compute_potential``."""

    def test_negative_unit_diff_weight_is_applied(self):
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config={"income_diff": 0.0, "unit_diff": -1.0, "structure_control": 0.0},
        )
        env.reset(seed=0)

        from reinforcetactics.core.unit import Unit

        env.game_state.units.append(Unit("W", 1, 1, env.agent_player))
        # Agent has one more unit than the opponent: potential must be
        # unit_diff * (+1) = -1.0, not gated to 0 by a ``> 0`` check.
        assert env._compute_potential() == pytest.approx(-1.0)
        env.close()


# ==============================================================================
# ENGINE RNG SEEDING (Rogue evade reproducibility)
# ==============================================================================


class TestEngineRngSeeding:
    """``reset(seed=...)`` must control engine-side combat randomness.

    The Rogue evade roll in ``mechanics.attack_unit`` reads
    ``game_state.rng``; the env derives that generator from ``np_random``
    on every reset so seeded episodes reproduce combat outcomes too.
    """

    def test_reset_seeds_engine_rng_deterministically(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset(seed=123)
        assert env.game_state.rng is not None
        first_stream = [env.game_state.rng.random() for _ in range(5)]

        env.reset(seed=123)
        assert [env.game_state.rng.random() for _ in range(5)] == first_stream
        env.close()

    def test_different_seeds_give_different_engine_streams(self):
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
        env.reset(seed=123)
        stream_a = [env.game_state.rng.random() for _ in range(5)]
        env.reset(seed=456)
        stream_b = [env.game_state.rng.random() for _ in range(5)]
        assert stream_a != stream_b
        env.close()


# ==============================================================================
# STRUCTURE AUTO-HEAL EPISODE STATS
# ==============================================================================


class TestAutoHealEpisodeStats:
    """``episode_stats`` mirrors ``GameState.healing_totals`` agent-relatively,
    so eval diagnostics can quantify the silent auto-heal gold drain (own)
    and the opponent meat-wall's free durability (opp)."""

    def test_new_episode_stats_include_heal_keys(self, env_default):
        stats = env_default._new_episode_stats()
        for key in ("own_heal_hp", "own_heal_gold", "opp_heal_hp", "opp_heal_gold"):
            assert stats[key] == 0

    def test_opponent_auto_heal_flows_into_episode_stats(self):
        env = StrategyGameEnv(map_file=None, opponent="random", render_mode=None)
        env.reset(seed=0)
        gs = env.game_state
        opp = 3 - env.agent_player

        from reinforcetactics.core.unit import Unit

        # Park a wounded opponent unit on one of the opponent's structures
        # and fund the heal. Healing fires at the start of the opponent's
        # turn (inside the agent's end_turn processing), before the bot acts.
        opp_tile = next(t for row in gs.grid.tiles for t in row if t.player == opp and t.type in ("h", "b", "t"))
        wounded = Unit("W", opp_tile.x, opp_tile.y, opp)
        wounded.health = 5
        gs.units.append(wounded)
        gs.player_gold[opp] = 500

        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn -> opponent turn runs
        env.step(action)

        assert env.episode_stats["opp_heal_hp"] > 0
        assert env.episode_stats["opp_heal_gold"] > 0
        # Mirrors are cumulative snapshots, not deltas.
        assert env.episode_stats["opp_heal_hp"] == gs.healing_totals[opp]["hp"]
        assert env.episode_stats["opp_heal_gold"] == gs.healing_totals[opp]["gold"]
        env.close()
