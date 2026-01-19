"""
Tests for the StrategyGameEnv Gymnasium environment.

This test suite provides comprehensive coverage of the Gymnasium environment
used for RL training, including observation spaces, action spaces, step/reset
functions, reward calculations, and episode statistics.
"""
import sys
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
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
    return StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)


@pytest.fixture
def env_no_opponent():
    """Create environment without opponent."""
    return StrategyGameEnv(map_file=None, opponent=None, render_mode=None)


@pytest.fixture
def env_hierarchical():
    """Create environment with hierarchical action space."""
    return StrategyGameEnv(map_file=None, opponent='bot', render_mode=None, hierarchical=True)


@pytest.fixture
def custom_reward_config():
    """Custom reward configuration for testing."""
    return {
        'win': 500.0,
        'loss': -500.0,
        'income_diff': 0.2,
        'unit_diff': 2.0,
        'structure_control': 10.0,
        'invalid_action': -5.0,
        'turn_penalty': -0.5
    }


# ==============================================================================
# 1. ENVIRONMENT CREATION TESTS
# ==============================================================================

class TestEnvironmentCreation:
    """Test environment initialization with various configurations."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters (no map file, random map generation)."""
        env = StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)

        assert env is not None
        assert env.game_state is not None
        assert env.grid_width > 0
        assert env.grid_height > 0
        assert env.opponent_type == 'bot'
        assert env.max_steps == 500
        assert env.current_step == 0
        assert env.hierarchical is False
        assert env.render_mode is None
        env.close()

    def test_initialization_with_map_file(self, simple_map_data, tmp_path):
        """Test initialization with a specific map file."""
        # Save map to temporary file
        map_file = tmp_path / "test_map.csv"
        pd.DataFrame(simple_map_data).to_csv(map_file, index=False, header=False)

        env = StrategyGameEnv(map_file=str(map_file), opponent='bot', render_mode=None)

        assert env is not None
        assert env.game_state is not None
        env.close()

    def test_initialization_opponent_bot(self):
        """Test initialization with opponent='bot'."""
        env = StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)

        assert env.opponent_type == 'bot'
        env.close()

    def test_initialization_opponent_random(self):
        """Test initialization with opponent='random'."""
        env = StrategyGameEnv(map_file=None, opponent='random', render_mode=None)

        assert env.opponent_type == 'random'
        env.close()

    def test_initialization_opponent_self(self):
        """Test initialization with opponent='self' (self-play)."""
        env = StrategyGameEnv(map_file=None, opponent='self', render_mode=None)

        assert env.opponent_type == 'self'
        env.close()

    def test_initialization_opponent_none(self):
        """Test initialization with opponent=None."""
        env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)

        assert env.opponent_type is None
        assert env.opponent is None
        env.close()

    def test_initialization_custom_reward_config(self, custom_reward_config):
        """Test initialization with custom reward_config."""
        env = StrategyGameEnv(
            map_file=None,
            opponent='bot',
            render_mode=None,
            reward_config=custom_reward_config
        )

        assert env.reward_config['win'] == 500.0
        assert env.reward_config['loss'] == -500.0
        assert env.reward_config['invalid_action'] == -5.0
        env.close()

    def test_initialization_hierarchical_mode(self, env_hierarchical):
        """Test initialization with hierarchical=True for HRL mode."""
        assert env_hierarchical.hierarchical is True
        assert isinstance(env_hierarchical.action_space, spaces.Dict)
        assert 'goal' in env_hierarchical.action_space.spaces
        assert 'primitive' in env_hierarchical.action_space.spaces
        env_hierarchical.close()

    def test_render_mode_none_no_pygame(self):
        """Test that render_mode=None does not import pygame (headless mode verification)."""
        # Create environment without rendering
        env = StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)

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
        """Verify 'grid' shape is (grid_height, grid_width, 3) with dtype float32."""
        grid_space = env_default.observation_space['grid']

        expected_shape = (env_default.grid_height, env_default.grid_width, 3)
        assert grid_space.shape == expected_shape
        assert grid_space.dtype == np.float32
        env_default.close()

    def test_units_shape_and_dtype(self, env_default):
        """Verify 'units' shape is (grid_height, grid_width, 3) with dtype float32."""
        units_space = env_default.observation_space['units']

        expected_shape = (env_default.grid_height, env_default.grid_width, 3)
        assert units_space.shape == expected_shape
        assert units_space.dtype == np.float32
        env_default.close()

    def test_global_features_shape_and_dtype(self, env_default):
        """Verify 'global_features' shape is (6,) with dtype float32."""
        global_space = env_default.observation_space['global_features']

        assert global_space.shape == (6,)
        assert global_space.dtype == np.float32
        env_default.close()

    def test_action_mask_shape(self, env_default):
        """Verify 'action_mask' shape matches _get_action_space_size()."""
        action_mask_space = env_default.observation_space['action_mask']

        expected_size = env_default._get_action_space_size()
        assert action_mask_space.shape == (expected_size,)
        assert action_mask_space.dtype == np.float32
        env_default.close()

    def test_observations_from_reset_match_space(self, env_default):
        """Test that observations returned by reset() match the observation space."""
        obs, _ = env_default.reset()

        # Check that observation is in the observation space
        assert 'grid' in obs
        assert 'units' in obs
        assert 'global_features' in obs
        assert 'action_mask' in obs

        # Verify shapes
        assert obs['grid'].shape == env_default.observation_space['grid'].shape
        assert obs['units'].shape == env_default.observation_space['units'].shape
        assert obs['global_features'].shape == env_default.observation_space['global_features'].shape
        assert obs['action_mask'].shape == env_default.observation_space['action_mask'].shape

        # Verify dtypes
        assert obs['grid'].dtype == np.float32
        assert obs['units'].dtype == np.float32
        assert obs['global_features'].dtype == np.float32
        assert obs['action_mask'].dtype == np.float32

        env_default.close()

    def test_observations_from_step_match_space(self, env_default):
        """Test that observations returned by step() match the observation space."""
        env_default.reset()

        # Take a step with end_turn action
        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
        obs, _, _, _, _ = env_default.step(action)

        # Check that observation is in the observation space
        assert 'grid' in obs
        assert 'units' in obs
        assert 'global_features' in obs
        assert 'action_mask' in obs

        # Verify shapes match
        assert obs['grid'].shape == env_default.observation_space['grid'].shape
        assert obs['units'].shape == env_default.observation_space['units'].shape
        assert obs['global_features'].shape == env_default.observation_space['global_features'].shape
        assert obs['action_mask'].shape == env_default.observation_space['action_mask'].shape

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
        assert 'goal' in env_hierarchical.action_space.spaces
        assert 'primitive' in env_hierarchical.action_space.spaces

        # Check goal space
        assert isinstance(env_hierarchical.action_space.spaces['goal'], spaces.Discrete)

        # Check primitive space
        assert isinstance(env_hierarchical.action_space.spaces['primitive'], spaces.MultiDiscrete)

        env_hierarchical.close()

    def test_encode_action_create_unit(self, env_default):
        """Test _encode_action() for create_unit action (type 0)."""
        action = np.array([0, 0, 5, 5, 8, 8])  # create_unit, warrior, at (8,8)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 0
        assert action_dict['unit_type'] == 'W'
        assert action_dict['from_pos'] == (5, 5)
        assert action_dict['to_pos'] == (8, 8)

        env_default.close()

    def test_encode_action_move(self, env_default):
        """Test _encode_action() for move action (type 1)."""
        action = np.array([1, 0, 2, 3, 4, 5])  # move from (2,3) to (4,5)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 1
        assert action_dict['from_pos'] == (2, 3)
        assert action_dict['to_pos'] == (4, 5)

        env_default.close()

    def test_encode_action_attack(self, env_default):
        """Test _encode_action() for attack action (type 2)."""
        action = np.array([2, 0, 1, 1, 2, 1])  # attack from (1,1) to (2,1)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 2
        assert action_dict['from_pos'] == (1, 1)
        assert action_dict['to_pos'] == (2, 1)

        env_default.close()

    def test_encode_action_seize(self, env_default):
        """Test _encode_action() for seize action (type 3)."""
        action = np.array([3, 0, 5, 5, 0, 0])  # seize at (5,5)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 3
        assert action_dict['from_pos'] == (5, 5)

        env_default.close()

    def test_encode_action_heal(self, env_default):
        """Test _encode_action() for heal action (type 4)."""
        action = np.array([4, 2, 3, 3, 4, 3])  # heal from (3,3) to (4,3)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 4
        assert action_dict['unit_type'] == 'C'  # Cleric
        assert action_dict['from_pos'] == (3, 3)
        assert action_dict['to_pos'] == (4, 3)

        env_default.close()

    def test_encode_action_end_turn(self, env_default):
        """Test _encode_action() for end_turn action (type 5)."""
        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 5

        env_default.close()

    def test_encode_action_paralyze(self, env_default):
        """Test _encode_action() for paralyze action (type 6)."""
        action = np.array([6, 1, 3, 3, 4, 3])  # paralyze from (3,3) to (4,3)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 6
        assert action_dict['from_pos'] == (3, 3)
        assert action_dict['to_pos'] == (4, 3)

        env_default.close()

    def test_encode_action_haste(self, env_default):
        """Test _encode_action() for haste action (type 7)."""
        action = np.array([7, 6, 2, 2, 3, 2])  # haste from (2,2) to (3,2)

        action_dict = env_default._encode_action(action)

        assert action_dict['action_type'] == 7
        assert action_dict['unit_type'] == 'S'  # Sorcerer
        assert action_dict['from_pos'] == (2, 2)
        assert action_dict['to_pos'] == (3, 2)

        env_default.close()

    def test_encode_action_all_unit_types(self, env_default):
        """Test action encoding with all unit types."""
        # Test Warrior (0)
        action_w = np.array([0, 0, 0, 0, 1, 1])
        assert env_default._encode_action(action_w)['unit_type'] == 'W'

        # Test Mage (1)
        action_m = np.array([0, 1, 0, 0, 1, 1])
        assert env_default._encode_action(action_m)['unit_type'] == 'M'

        # Test Cleric (2)
        action_c = np.array([0, 2, 0, 0, 1, 1])
        assert env_default._encode_action(action_c)['unit_type'] == 'C'

        # Test Archer (3)
        action_a = np.array([0, 3, 0, 0, 1, 1])
        assert env_default._encode_action(action_a)['unit_type'] == 'A'

        # Test Knight (4)
        action_k = np.array([0, 4, 0, 0, 1, 1])
        assert env_default._encode_action(action_k)['unit_type'] == 'K'

        # Test Rogue (5)
        action_r = np.array([0, 5, 0, 0, 1, 1])
        assert env_default._encode_action(action_r)['unit_type'] == 'R'

        # Test Sorcerer (6)
        action_s = np.array([0, 6, 0, 0, 1, 1])
        assert env_default._encode_action(action_s)['unit_type'] == 'S'

        # Test Barbarian (7)
        action_b = np.array([0, 7, 0, 0, 1, 1])
        assert env_default._encode_action(action_b)['unit_type'] == 'B'

        env_default.close()

    def test_encode_action_boundary_coordinates(self, env_default):
        """Test action encoding with boundary coordinates."""
        max_x = env_default.grid_width - 1
        max_y = env_default.grid_height - 1

        # Test boundary positions
        action = np.array([1, 0, 0, 0, max_x, max_y])
        action_dict = env_default._encode_action(action)

        assert action_dict['from_pos'] == (0, 0)
        assert action_dict['to_pos'] == (max_x, max_y)

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
        assert info['valid_action'] is True
        assert reward <= 0  # Turn penalty applied

        env_default.close()

    def test_invalid_action_negative_reward(self, env_default):
        """Test invalid action execution and negative rewards (invalid_action penalty)."""
        env_default.reset()

        # Try to move a non-existent unit
        action = np.array([1, 0, 0, 0, 1, 1])  # move from (0,0) to (1,1)

        _, reward, _, _, info = env_default.step(action)

        # Should be marked as invalid
        assert info['valid_action'] is False
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
        assert info['game_over'] is True

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
        assert 'episode_stats' in info
        assert 'game_over' in info
        assert 'winner' in info
        assert 'turn' in info
        assert 'valid_action' in info

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
        env_default.episode_stats['reward'] = 100.0
        env_default.episode_stats['invalid_actions'] = 5

        # Reset
        env_default.reset()

        # Check stats are reset
        assert env_default.episode_stats['reward'] == 0.0
        assert env_default.episode_stats['invalid_actions'] == 0
        assert env_default.episode_stats['winner'] is None

        env_default.close()

    def test_reset_with_seed(self, env_default):
        """Test reset() with seed parameter for reproducibility."""
        # Reset with seed
        obs1, _ = env_default.reset(seed=42)
        obs2, _ = env_default.reset(seed=42)

        # Observations should be identical with same seed
        assert np.array_equal(obs1['grid'], obs2['grid'])
        assert np.array_equal(obs1['units'], obs2['units'])

        env_default.close()

    def test_opponent_reinitialized_on_reset(self):
        """Test that opponent is re-initialized on reset (functional behavior)."""
        env = StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)

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

    def test_turn_penalty_on_end_turn(self, env_default):
        """Test turn_penalty on end_turn action."""
        env_default.reset()

        # End turn action should apply turn penalty
        action = np.array([5, 0, 0, 0, 0, 0])
        _, reward, _, _, _ = env_default.step(action)

        # Reward should include turn penalty (negative)
        assert reward <= 0

        env_default.close()

    def test_invalid_action_penalty(self, env_default):
        """Test invalid_action penalty."""
        env_default.reset()

        # Try invalid action (move non-existent unit)
        action = np.array([1, 0, 0, 0, 1, 1])
        _, reward, _, _, info = env_default.step(action)

        # Should apply invalid action penalty
        assert info['valid_action'] is False
        assert reward < 0

        env_default.close()

    def test_cumulative_episode_reward_tracking(self, env_default):
        """Test cumulative episode_stats['reward'] tracking."""
        env_default.reset()

        assert env_default.episode_stats['reward'] == 0.0

        # Take multiple steps
        for _ in range(3):
            action = np.array([5, 0, 0, 0, 0, 0])
            env_default.step(action)

        # Cumulative reward should be tracked
        assert env_default.episode_stats['reward'] != 0.0

        env_default.close()

    def test_custom_reward_config(self, custom_reward_config):
        """Test reward calculation with custom reward_config."""
        env = StrategyGameEnv(
            map_file=None,
            opponent=None,
            render_mode=None,
            reward_config=custom_reward_config
        )
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
        env = StrategyGameEnv(map_file=None, opponent='bot', render_mode=None)

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

        initial_count = env_default.episode_stats['invalid_actions']

        # Perform invalid action
        action = np.array([1, 0, 0, 0, 1, 1])  # move non-existent unit
        obs, reward, terminated, truncated, info = env_default.step(action)

        assert env_default.episode_stats['invalid_actions'] > initial_count

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

        assert env.episode_stats['winner'] == 1

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
        assert 'episode_stats' in info
        assert len(info['episode_stats']) > 0

        env.close()

    def test_episode_stats_in_info_on_truncation(self, env_default):
        """Verify episode_stats is included in info on truncation."""
        env_default.reset()

        # Set to trigger truncation
        env_default.current_step = env_default.max_steps - 1

        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env_default.step(action)

        assert truncated is True
        assert 'episode_stats' in info
        assert len(info['episode_stats']) > 0

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
        opponent_types = ['bot', 'random', 'self', None]

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

        # Should be positive integer
        assert size > 0
        assert isinstance(size, int)

        # Should match action_mask shape
        obs, _ = env_default.reset()
        assert obs['action_mask'].shape[0] == size

        env_default.close()


# ==============================================================================
# 10. ACTION MASKING TESTS (for MaskablePPO compatibility)
# ==============================================================================

class TestActionMasking:
    """Test action masking functionality for MaskablePPO (sb3-contrib)."""

    def test_action_masks_method_exists(self, env_default):
        """Test that action_masks() method exists on environment."""
        env_default.reset()
        assert hasattr(env_default, 'action_masks')
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
            8,                          # action_type
            8,                          # unit_type
            env_default.grid_width,     # from_x
            env_default.grid_height,    # from_y
            env_default.grid_width,     # to_x
            env_default.grid_height     # to_y
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
        assert action_type_mask[5] == True, "End turn should always be valid"

        env_default.close()

    def test_action_masks_at_least_one_valid_per_dimension(self, env_default):
        """Test that each mask dimension has at least one valid option."""
        env_default.reset()
        masks = env_default.action_masks()

        dimension_names = ['action_type', 'unit_type', 'from_x', 'from_y', 'to_x', 'to_y']

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
        if legal_actions.get('create_unit'):
            assert action_type_mask[0] == True, "Create unit mask mismatch"

        # If there are move actions, action_type 1 should be valid
        if legal_actions.get('move'):
            assert action_type_mask[1] == True, "Move mask mismatch"

        # If there are attack actions, action_type 2 should be valid
        if legal_actions.get('attack'):
            assert action_type_mask[2] == True, "Attack mask mismatch"

        env_default.close()

    def test_action_masks_update_after_step(self, env_default):
        """Test that action masks update after taking a step."""
        env_default.reset()
        masks_before = env_default.action_masks()

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
        assert masks[0][5] == True  # End turn always valid

        env_default.close()

    def test_get_action_mask_flat_exists(self, env_default):
        """Test that get_action_mask_flat() method exists and works."""
        env_default.reset()

        assert hasattr(env_default, 'get_action_mask_flat')
        flat_mask = env_default.get_action_mask_flat()

        expected_size = env_default._get_action_space_size()
        assert flat_mask.shape == (expected_size,)
        assert flat_mask.dtype == np.float32

        env_default.close()


class TestActionMaskingWrapper:
    """Test the ActionMaskedEnv wrapper for MaskablePPO compatibility."""

    def test_wrapper_import(self):
        """Test that masking module can be imported."""
        from reinforcetactics.rl.masking import (
            ActionMaskedEnv,
            make_maskable_env,
            make_maskable_vec_env,
            validate_action_mask
        )

    def test_make_maskable_env(self):
        """Test make_maskable_env creates wrapped environment."""
        from reinforcetactics.rl.masking import make_maskable_env

        env = make_maskable_env(opponent='bot')

        assert env is not None
        assert hasattr(env, 'action_masks')

        # Should be able to reset and get masks
        env.reset()
        masks = env.action_masks()
        assert isinstance(masks, np.ndarray)

        env.close()

    def test_action_masked_env_wrapper(self):
        """Test ActionMaskedEnv wrapper functionality."""
        from reinforcetactics.rl.masking import ActionMaskedEnv

        base_env = StrategyGameEnv(opponent='bot', render_mode=None)
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

        env = StrategyGameEnv(opponent='bot', render_mode=None)
        env.reset()

        validation = validate_action_mask(env)

        assert 'valid' in validation
        assert 'errors' in validation
        assert 'warnings' in validation
        assert 'mask_summary' in validation

        # Should be valid for a fresh game
        assert validation['valid'] == True, f"Validation errors: {validation['errors']}"

        env.close()

    def test_make_curriculum_env(self):
        """Test curriculum environment creation."""
        from reinforcetactics.rl.masking import make_curriculum_env

        for difficulty in ['easy', 'medium', 'hard']:
            env = make_curriculum_env(difficulty=difficulty)

            assert env is not None
            env.reset()
            masks = env.action_masks()
            assert masks is not None

            env.close()

    def test_wrapper_with_stats_tracking(self):
        """Test ActionMaskedEnv with stats tracking enabled."""
        from reinforcetactics.rl.masking import ActionMaskedEnv

        base_env = StrategyGameEnv(opponent='bot', render_mode=None)
        wrapped_env = ActionMaskedEnv(base_env, track_stats=True)

        wrapped_env.reset()

        # Take a few actions
        for _ in range(3):
            action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
            wrapped_env.step(action)

        stats = wrapped_env.get_masking_stats()

        assert 'total_actions' in stats
        assert stats['total_actions'] == 3
        assert 'action_type_distribution' in stats

        wrapped_env.close()
