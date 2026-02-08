"""
Tests for the self-play RL training functionality.

This test suite provides coverage of the self-play environment, opponent pool,
and related utilities for training RL agents against themselves.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.self_play import (
    OpponentPool,
    SelfPlayEnv,
    SelfPlayCallback,
    make_self_play_env,
    make_self_play_vec_env,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def base_env():
    """Create a base StrategyGameEnv for testing."""
    env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)
    yield env
    env.close()


@pytest.fixture
def self_play_env():
    """Create a SelfPlayEnv for testing."""
    env = make_self_play_env(max_steps=100, swap_players=False)
    yield env
    env.close()


@pytest.fixture
def opponent_pool():
    """Create an OpponentPool for testing."""
    return OpponentPool(max_size=5, selection_strategy='uniform')


# ==============================================================================
# 1. OPPONENT POOL TESTS
# ==============================================================================

class TestOpponentPool:
    """Test OpponentPool functionality."""

    def test_initialization_default(self):
        """Test OpponentPool initialization with default parameters."""
        pool = OpponentPool()

        assert pool.max_size == 10
        assert pool.selection_strategy == 'uniform'
        assert pool.size == 0
        assert len(pool) == 0

    def test_initialization_custom(self):
        """Test OpponentPool initialization with custom parameters."""
        pool = OpponentPool(
            max_size=5,
            selection_strategy='recent',
            save_dir=None
        )

        assert pool.max_size == 5
        assert pool.selection_strategy == 'recent'

    def test_add_model_params(self, opponent_pool):
        """Test adding model parameters to pool."""
        # Create mock model params
        mock_params = {
            'layer1.weight': np.random.randn(10, 10),
            'layer1.bias': np.random.randn(10)
        }

        # Manually add to test (normally _copy_model_params handles this)
        opponent_pool.models.append(mock_params)
        opponent_pool.metadata.append({
            'timestep': 1000,
            'win_rate': 0.6,
            'index': 0
        })
        opponent_pool._update_selection_weights()

        assert opponent_pool.size == 1
        assert len(opponent_pool.metadata) == 1
        assert opponent_pool.metadata[0]['timestep'] == 1000

    def test_sample_opponent_empty_pool(self, opponent_pool):
        """Test sampling from empty pool returns None."""
        result = opponent_pool.sample_opponent()
        assert result is None

    def test_sample_opponent_with_models(self, opponent_pool):
        """Test sampling from pool with models."""
        # Add mock models
        for i in range(3):
            mock_params = {'layer': np.random.randn(5, 5)}
            opponent_pool.models.append(mock_params)
            opponent_pool.metadata.append({
                'timestep': i * 1000,
                'win_rate': 0.5,
                'index': i
            })
        opponent_pool._update_selection_weights()

        # Sample should return a model
        result = opponent_pool.sample_opponent()
        assert result is not None
        assert 'layer' in result

    def test_sample_opponent_with_metadata(self, opponent_pool):
        """Test sampling with metadata."""
        mock_params = {'layer': np.random.randn(5, 5)}
        opponent_pool.models.append(mock_params)
        opponent_pool.metadata.append({
            'timestep': 1000,
            'win_rate': 0.6,
            'index': 0
        })
        opponent_pool._update_selection_weights()

        result = opponent_pool.sample_opponent_with_metadata()
        assert result is not None
        params, metadata = result
        assert 'layer' in params
        assert metadata['timestep'] == 1000

    def test_selection_strategy_uniform(self):
        """Test uniform selection strategy."""
        pool = OpponentPool(max_size=5, selection_strategy='uniform')

        for i in range(3):
            pool.models.append({'id': i})
            pool.metadata.append({'timestep': i, 'win_rate': 0.5, 'index': i})
        pool._update_selection_weights()

        # All weights should be equal for uniform
        expected = 1.0 / 3
        for weight in pool._selection_weights:
            assert abs(weight - expected) < 0.001

    def test_selection_strategy_recent(self):
        """Test recent selection strategy favors newer models."""
        pool = OpponentPool(max_size=5, selection_strategy='recent')

        for i in range(3):
            pool.models.append({'id': i})
            pool.metadata.append({'timestep': i, 'win_rate': 0.5, 'index': i})
        pool._update_selection_weights()

        # Later models should have higher weights
        assert pool._selection_weights[2] > pool._selection_weights[1]
        assert pool._selection_weights[1] > pool._selection_weights[0]

    def test_selection_strategy_prioritized(self):
        """Test prioritized selection strategy favors higher win rates."""
        pool = OpponentPool(max_size=5, selection_strategy='prioritized')

        # Add models with different win rates
        win_rates = [0.3, 0.5, 0.8]
        for i, wr in enumerate(win_rates):
            pool.models.append({'id': i})
            pool.metadata.append({'timestep': i, 'win_rate': wr, 'index': i})
        pool._update_selection_weights()

        # Higher win rate should have higher weight
        assert pool._selection_weights[2] > pool._selection_weights[1]
        assert pool._selection_weights[1] > pool._selection_weights[0]

    def test_update_win_rate(self, opponent_pool):
        """Test updating a model's win rate."""
        opponent_pool.models.append({'id': 0})
        opponent_pool.metadata.append({
            'timestep': 0,
            'win_rate': 0.5,
            'index': 0
        })
        opponent_pool._update_selection_weights()

        opponent_pool.update_win_rate(0, 0.75)
        assert opponent_pool.metadata[0]['win_rate'] == 0.75

    def test_max_size_limit(self):
        """Test that pool respects max_size limit."""
        pool = OpponentPool(max_size=3)

        # Add more than max_size models
        for i in range(5):
            pool.models.append({'id': i})
            pool.metadata.append({'timestep': i, 'win_rate': 0.5, 'index': i})

        # Should only keep max_size models (deque behavior)
        assert pool.size == 3


# ==============================================================================
# 2. SELF-PLAY ENVIRONMENT TESTS
# ==============================================================================

class TestSelfPlayEnv:
    """Test SelfPlayEnv functionality."""

    def test_initialization(self, self_play_env):
        """Test SelfPlayEnv initialization."""
        assert self_play_env is not None
        assert self_play_env.agent_player == 1

    def test_reset_returns_correct_format(self, self_play_env):
        """Test reset returns observation and info."""
        obs, info = self_play_env.reset()

        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        assert 'grid' in obs
        assert 'units' in obs
        assert 'global_features' in obs

    def test_step_returns_correct_format(self, self_play_env):
        """Test step returns correct tuple format."""
        self_play_env.reset()
        action = np.array([5, 0, 0, 0, 0, 0])  # end_turn

        result = self_play_env.step(action)

        assert isinstance(result, tuple)
        assert len(result) == 5

        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_masks_method(self, self_play_env):
        """Test that action_masks method works."""
        self_play_env.reset()
        masks = self_play_env.action_masks()

        assert isinstance(masks, tuple)
        assert len(masks) == 6

    def test_stats_tracking(self, self_play_env):
        """Test that self-play stats are tracked."""
        self_play_env.reset()

        # Initial stats
        assert self_play_env.stats['total_games'] == 0
        assert self_play_env.stats['agent_wins'] == 0
        assert self_play_env.stats['opponent_wins'] == 0

    def test_get_win_rate(self, self_play_env):
        """Test win rate calculation."""
        # No games played
        assert self_play_env.get_win_rate() == 0.5

        # Simulate some games
        self_play_env.stats['total_games'] = 10
        self_play_env.stats['agent_wins'] = 7
        assert self_play_env.get_win_rate() == 0.7

    def test_set_opponent_model(self, self_play_env):
        """Test setting opponent model."""
        mock_model = MagicMock()
        self_play_env.set_opponent_model(mock_model)

        assert self_play_env.opponent_model is mock_model

    def test_flip_observation(self, self_play_env):
        """Test observation flipping for opponent perspective."""
        self_play_env.reset()

        # Get original observation
        obs = self_play_env.env._get_obs()

        # Flip for opponent
        flipped = self_play_env._flip_observation(obs)

        # Global features should be swapped
        # [gold_p1, gold_p2, turn, units_p1, units_p2, current_player]
        assert flipped['global_features'][0] == obs['global_features'][1]
        assert flipped['global_features'][1] == obs['global_features'][0]
        assert flipped['global_features'][3] == obs['global_features'][4]
        assert flipped['global_features'][4] == obs['global_features'][3]

    def test_get_random_valid_action(self, self_play_env):
        """Test fallback random action generation."""
        self_play_env.reset()
        action = self_play_env._get_random_valid_action()

        assert isinstance(action, np.ndarray)
        assert len(action) == 6

    def test_multiple_episodes(self, self_play_env):
        """Test running multiple episodes."""
        for _ in range(3):
            obs, info = self_play_env.reset()
            assert obs is not None

            # Take a few steps
            for _ in range(5):
                action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
                obs, reward, terminated, truncated, info = self_play_env.step(action)

                if terminated or truncated:
                    break

    def test_self_play_stats_in_info(self, self_play_env):
        """Test that self-play stats are included in info."""
        self_play_env.reset()
        action = np.array([5, 0, 0, 0, 0, 0])
        _, _, _, _, info = self_play_env.step(action)

        assert 'self_play_stats' in info


# ==============================================================================
# 3. SELF-PLAY ENVIRONMENT CREATION TESTS
# ==============================================================================

class TestSelfPlayEnvCreation:
    """Test self-play environment factory functions."""

    def test_make_self_play_env_default(self):
        """Test make_self_play_env with default parameters."""
        env = make_self_play_env()

        assert env is not None
        assert hasattr(env, 'action_masks')

        obs, info = env.reset()
        assert 'grid' in obs

        env.close()

    def test_make_self_play_env_custom_params(self):
        """Test make_self_play_env with custom parameters."""
        env = make_self_play_env(
            max_steps=200,
            swap_players=False
        )

        assert env is not None
        env.reset()
        env.close()

    def test_make_self_play_env_with_opponent_pool(self):
        """Test make_self_play_env with opponent pool."""
        pool = OpponentPool(max_size=5)
        env = make_self_play_env(opponent_pool=pool, swap_players=False)

        assert env is not None
        assert env.opponent_pool is pool

        env.close()

    def test_make_self_play_env_with_reward_config(self):
        """Test make_self_play_env with custom reward config."""
        reward_config = {
            'win': 500.0,
            'loss': -500.0,
            'income_diff': 0.2,
            'unit_diff': 2.0,
            'structure_control': 10.0,
            'invalid_action': -5.0,
            'turn_penalty': -0.05
        }

        env = make_self_play_env(reward_config=reward_config)
        assert env is not None
        env.close()

    def test_make_self_play_vec_env_dummy(self):
        """Test make_self_play_vec_env with DummyVecEnv."""
        vec_env = make_self_play_vec_env(
            n_envs=2,
            use_subprocess=False,
            swap_players=False
        )

        assert vec_env is not None

        # Should be able to reset
        obs = vec_env.reset()
        assert obs is not None

        vec_env.close()

    def test_make_self_play_vec_env_with_pool(self):
        """Test make_self_play_vec_env with shared opponent pool."""
        pool = OpponentPool(max_size=5)
        vec_env = make_self_play_vec_env(
            n_envs=2,
            use_subprocess=False,
            opponent_pool=pool,
            swap_players=False
        )

        assert vec_env is not None
        vec_env.close()


# ==============================================================================
# 4. SELF-PLAY CALLBACK TESTS
# ==============================================================================

class TestSelfPlayCallback:
    """Test SelfPlayCallback functionality."""

    def test_callback_initialization(self, self_play_env):
        """Test SelfPlayCallback initialization."""
        callback = SelfPlayCallback(
            env=self_play_env,
            update_freq=1000,
            add_to_pool_freq=5000,
            verbose=0
        )

        assert callback.update_freq == 1000
        assert callback.add_to_pool_freq == 5000
        assert callback.n_calls == 0

    def test_get_self_play_envs_single(self, self_play_env):
        """Test extracting self-play env from single environment."""
        callback = SelfPlayCallback(env=self_play_env, verbose=0)
        envs = callback._get_self_play_envs()

        assert len(envs) >= 1

    def test_callback_init_with_model(self, self_play_env):
        """Test callback initialization with model via init_callback (SB3 lifecycle)."""
        callback = SelfPlayCallback(env=self_play_env, verbose=0)
        mock_model = MagicMock()

        # SB3's BaseCallback.init_callback() sets self.model then calls _init_callback()
        callback.model = mock_model
        callback._init_callback()

        assert callback.model is mock_model


# ==============================================================================
# 5. INTEGRATION TESTS
# ==============================================================================

class TestSelfPlayIntegration:
    """Integration tests for self-play training workflow."""

    def test_complete_episode_with_self_play(self):
        """Test a complete episode with self-play environment."""
        env = make_self_play_env(max_steps=50, swap_players=False)

        obs, info = env.reset()
        assert obs is not None

        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 20:
            action = np.array([5, 0, 0, 0, 0, 0])  # end_turn
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert steps > 0
        env.close()

    def test_opponent_pool_integration(self):
        """Test self-play with opponent pool integration."""
        pool = OpponentPool(max_size=3, selection_strategy='uniform')

        # Add some mock opponents
        for i in range(2):
            pool.models.append({'layer': np.random.randn(5, 5)})
            pool.metadata.append({
                'timestep': i * 1000,
                'win_rate': 0.5,
                'index': i
            })
        pool._update_selection_weights()

        env = make_self_play_env(opponent_pool=pool, swap_players=False)

        # Reset should potentially update opponent from pool
        obs, info = env.reset()
        assert obs is not None

        env.close()

    def test_random_actions_self_play(self):
        """Test random actions in self-play environment."""
        env = make_self_play_env(max_steps=50, swap_players=False)
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()

            try:
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    env.reset()

            except Exception as e:
                pytest.fail(f"Random action failed: {e}")

        env.close()

    def test_vectorized_self_play(self):
        """Test vectorized self-play environments."""
        vec_env = make_self_play_vec_env(
            n_envs=2,
            max_steps=50,
            use_subprocess=False,
            swap_players=False
        )

        # Reset all environments
        obs = vec_env.reset()
        assert obs is not None

        # Take a step in all environments
        actions = np.array([[5, 0, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]])
        obs, rewards, dones, infos = vec_env.step(actions)

        assert obs is not None
        assert len(rewards) == 2
        assert len(dones) == 2

        vec_env.close()


# ==============================================================================
# 6. EDGE CASES AND ERROR HANDLING
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_opponent_pool_sampling(self):
        """Test behavior when sampling from empty opponent pool."""
        pool = OpponentPool(max_size=5)

        result = pool.sample_opponent()
        assert result is None

        result_with_meta = pool.sample_opponent_with_metadata()
        assert result_with_meta is None

    def test_self_play_without_opponent_model(self):
        """Test self-play works when opponent model is None."""
        env = make_self_play_env(swap_players=False)
        env.opponent_model = None

        obs, info = env.reset()
        action = np.array([5, 0, 0, 0, 0, 0])
        obs, reward, terminated, truncated, info = env.step(action)

        # Should still work with fallback random opponent
        assert obs is not None

        env.close()

    def test_pool_max_size_enforcement(self):
        """Test that pool enforces max size."""
        pool = OpponentPool(max_size=2)

        # Add more than max_size
        for i in range(5):
            pool.models.append({'id': i})
            pool.metadata.append({'timestep': i, 'win_rate': 0.5, 'index': i})

        # Should only have max_size items
        assert pool.size == 2
        # Should have most recent items (deque drops oldest)
        assert pool.models[0]['id'] == 3
        assert pool.models[1]['id'] == 4

    def test_update_win_rate_invalid_index(self, opponent_pool):
        """Test updating win rate with invalid index."""
        # Should not raise error for invalid index
        opponent_pool.update_win_rate(999, 0.75)  # No-op

    def test_self_play_env_close(self):
        """Test that environment closes properly."""
        env = make_self_play_env()
        env.reset()

        # Should not raise exception
        env.close()


# ==============================================================================
# 7. MODULE IMPORT TESTS
# ==============================================================================

class TestModuleImports:
    """Test that self-play module imports correctly."""

    def test_import_from_rl_module(self):
        """Test importing self-play components from rl module."""
        from reinforcetactics.rl import (
            SelfPlayEnv,
            OpponentPool,
            SelfPlayCallback,
            make_self_play_env,
            make_self_play_vec_env,
        )

        assert SelfPlayEnv is not None
        assert OpponentPool is not None
        assert SelfPlayCallback is not None
        assert make_self_play_env is not None
        assert make_self_play_vec_env is not None

    def test_import_directly(self):
        """Test direct import from self_play module."""
        from reinforcetactics.rl.self_play import (
            SelfPlayEnv,
            OpponentPool,
            SelfPlayCallback,
            make_self_play_env,
            make_self_play_vec_env,
        )

        assert SelfPlayEnv is not None
        assert OpponentPool is not None
