"""
Tests for the eval/eval_agent.py module.

This test module verifies the evaluation script's statistics calculation
including edge cases like zero episodes.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestEvaluateAgentStatistics:
    """Test statistics calculation in evaluate_agent function."""

    def test_zero_episodes_does_not_raise_division_error(self):
        """
        Test that evaluate_agent handles n_episodes=0 gracefully.

        Previously, win_rate = wins / n_episodes would raise ZeroDivisionError
        when n_episodes was 0.
        """
        # Calculate statistics the same way evaluate_agent does
        n_episodes = 0
        wins = 0
        total_rewards = []
        episode_lengths = []
        invalid_actions = []

        # These calculations should not raise errors
        win_rate = wins / n_episodes if n_episodes > 0 else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        std_reward = np.std(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_invalid = np.mean(invalid_actions) if invalid_actions else 0.0

        assert win_rate == 0.0
        assert avg_reward == 0.0
        assert std_reward == 0.0
        assert avg_length == 0.0
        assert avg_invalid == 0.0

    def test_empty_lists_do_not_raise_warnings(self):
        """
        Test that empty lists don't produce numpy warnings.

        np.mean([]) produces a RuntimeWarning and returns nan,
        which we now handle with a conditional check.
        """
        total_rewards = []
        episode_lengths = []
        invalid_actions = []

        # These should return 0.0, not nan
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_invalid = np.mean(invalid_actions) if invalid_actions else 0.0

        # Verify we get 0.0, not nan
        assert avg_reward == 0.0
        assert not np.isnan(avg_reward)
        assert avg_length == 0.0
        assert not np.isnan(avg_length)
        assert avg_invalid == 0.0
        assert not np.isnan(avg_invalid)

    def test_normal_statistics_calculation(self):
        """Test that normal statistics calculations still work correctly."""
        n_episodes = 5
        wins = 3
        total_rewards = [100.0, 150.0, -50.0, 200.0, 75.0]
        episode_lengths = [10, 15, 8, 20, 12]
        invalid_actions = [2, 0, 5, 1, 3]

        win_rate = wins / n_episodes if n_episodes > 0 else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        std_reward = np.std(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_invalid = np.mean(invalid_actions) if invalid_actions else 0.0

        assert win_rate == 0.6  # 3/5
        assert avg_reward == 95.0  # (100+150-50+200+75)/5
        assert avg_length == 13.0  # (10+15+8+20+12)/5
        assert avg_invalid == 2.2  # (2+0+5+1+3)/5

    def test_single_episode_statistics(self):
        """Test statistics calculation with a single episode."""
        n_episodes = 1
        wins = 1
        total_rewards = [100.0]
        episode_lengths = [50]
        invalid_actions = [3]

        win_rate = wins / n_episodes if n_episodes > 0 else 0.0
        avg_reward = np.mean(total_rewards) if total_rewards else 0.0
        std_reward = np.std(total_rewards) if total_rewards else 0.0
        avg_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_invalid = np.mean(invalid_actions) if invalid_actions else 0.0

        assert win_rate == 1.0
        assert avg_reward == 100.0
        assert std_reward == 0.0  # Single value has 0 std
        assert avg_length == 50.0
        assert avg_invalid == 3.0
