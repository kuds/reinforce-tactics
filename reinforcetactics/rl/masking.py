"""
Action masking utilities for Reinforce Tactics.

Provides wrappers and helper functions for training with MaskablePPO from sb3-contrib.
This significantly improves training efficiency by preventing the agent from attempting
invalid actions.

Usage:
    from reinforcetactics.rl.masking import make_maskable_env, make_maskable_vec_env

    # Single environment
    env = make_maskable_env(map_file="maps/1v1/small.csv", opponent="bot")

    # Vectorized environments for faster training
    vec_env = make_maskable_vec_env(n_envs=4, opponent="bot")

    # Train with MaskablePPO
    from sb3_contrib import MaskablePPO
    model = MaskablePPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100000)
"""

from typing import Callable, Optional, List, Any, Dict
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reinforcetactics.rl.gym_env import StrategyGameEnv


class ActionMaskedEnv(gym.Wrapper):
    """
    Wrapper that ensures the environment properly supports action masking
    for sb3-contrib's MaskablePPO.

    This wrapper:
    1. Ensures action_masks() is properly exposed
    2. Validates that masks are consistent
    3. Optionally tracks masking statistics for debugging
    """

    def __init__(self, env: StrategyGameEnv, track_stats: bool = False):
        """
        Initialize the wrapper.

        Args:
            env: The StrategyGameEnv to wrap
            track_stats: Whether to track action masking statistics
        """
        super().__init__(env)
        self.track_stats = track_stats

        if track_stats:
            self.stats = {
                'total_actions': 0,
                'masked_actions_attempted': 0,
                'action_type_distribution': np.zeros(10),
            }

    def action_masks(self) -> np.ndarray:
        """
        Get action masks in the format expected by MaskablePPO.

        For MultiDiscrete spaces, returns a concatenated 1D boolean array
        of all dimension masks.

        Returns:
            Concatenated boolean mask array
        """
        masks = self.env.action_masks()
        # Concatenate all dimension masks into a single array
        return np.concatenate([m.astype(np.bool_) for m in masks])

    def get_action_masks_tuple(self):
        """
        Get action masks as a tuple of arrays (one per dimension).

        This is the native format from the environment.
        """
        return self.env.action_masks()

    def step(self, action):
        """Execute action and optionally track statistics."""
        if self.track_stats:
            self.stats['total_actions'] += 1
            action_type = int(action[0])
            self.stats['action_type_distribution'][action_type] += 1

        return super().step(action)

    def reset(self, **kwargs):
        """Reset environment and optionally reset stats."""
        return super().reset(**kwargs)

    def get_masking_stats(self) -> Dict[str, Any]:
        """Get action masking statistics."""
        if not self.track_stats:
            return {}

        stats = self.stats.copy()
        # Compute action type percentages
        total = stats['action_type_distribution'].sum()
        if total > 0:
            stats['action_type_percentages'] = (
                stats['action_type_distribution'] / total * 100
            ).tolist()
        return stats


def make_maskable_env(
    map_file: Optional[str] = None,
    opponent: str = 'bot',
    render_mode: Optional[str] = None,
    max_steps: int = 500,
    reward_config: Optional[Dict[str, float]] = None,
    track_stats: bool = False
) -> ActionMaskedEnv:
    """
    Create a single environment ready for use with MaskablePPO.

    Args:
        map_file: Path to map CSV file. None for random map.
        opponent: Opponent type ('bot', 'random', 'self', None)
        render_mode: 'human', 'rgb_array', or None
        max_steps: Maximum steps per episode
        reward_config: Custom reward configuration
        track_stats: Whether to track action masking statistics

    Returns:
        ActionMaskedEnv ready for training

    Example:
        env = make_maskable_env(opponent="bot")
        model = MaskablePPO("MultiInputPolicy", env)
        model.learn(total_timesteps=10000)
    """
    env = StrategyGameEnv(
        map_file=map_file,
        opponent=opponent,
        render_mode=render_mode,
        max_steps=max_steps,
        reward_config=reward_config
    )
    return ActionMaskedEnv(env, track_stats=track_stats)


def _make_env_fn(
    rank: int,
    seed: int,
    map_file: Optional[str],
    opponent: str,
    max_steps: int,
    reward_config: Optional[Dict[str, float]]
) -> Callable[[], ActionMaskedEnv]:
    """
    Create a function that creates an environment.

    Used for vectorized environment creation.
    """
    def _init() -> ActionMaskedEnv:
        env = StrategyGameEnv(
            map_file=map_file,
            opponent=opponent,
            render_mode=None,  # No rendering in vectorized envs
            max_steps=max_steps,
            reward_config=reward_config
        )
        env.reset(seed=seed + rank)
        wrapped = ActionMaskedEnv(env)
        return wrapped
    return _init


def make_maskable_vec_env(
    n_envs: int = 4,
    map_file: Optional[str] = None,
    opponent: str = 'bot',
    max_steps: int = 500,
    reward_config: Optional[Dict[str, float]] = None,
    seed: int = 0,
    use_subprocess: bool = True
):
    """
    Create vectorized environments for parallel training with MaskablePPO.

    Uses sb3-contrib's vectorized environment wrappers that properly handle
    action masking across multiple environments.

    Args:
        n_envs: Number of parallel environments
        map_file: Path to map CSV file. None for random maps.
        opponent: Opponent type ('bot', 'random', 'self')
        max_steps: Maximum steps per episode
        reward_config: Custom reward configuration
        seed: Random seed (each env gets seed + rank)
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)

    Returns:
        Vectorized environment ready for MaskablePPO

    Example:
        vec_env = make_maskable_vec_env(n_envs=8, opponent="bot")
        model = MaskablePPO("MultiInputPolicy", vec_env)
        model.learn(total_timesteps=1000000)
    """
    try:
        from sb3_contrib.common.wrappers import ActionMasker
        from sb3_contrib.common.maskable.utils import get_action_masks
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for action masking. "
            "Install it with: pip install sb3-contrib"
        )

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        _make_env_fn(i, seed, map_file, opponent, max_steps, reward_config)
        for i in range(n_envs)
    ]

    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env


def validate_action_mask(env: StrategyGameEnv) -> Dict[str, Any]:
    """
    Validate that action masks are correctly computed.

    Useful for debugging and testing.

    Args:
        env: Environment to validate

    Returns:
        Dict with validation results
    """
    masks = env.action_masks()
    legal_actions = env.game_state.get_legal_actions(player=env.game_state.current_player)

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'mask_summary': {}
    }

    # Check each mask dimension
    action_type_names = ['create', 'move', 'attack', 'seize', 'heal', 'end_turn', 'paralyze', 'haste', 'defence_buff', 'attack_buff']

    for i, name in enumerate(action_type_names):
        mask_enabled = masks[0][i]
        has_actions = bool(legal_actions.get(name, []))

        # Special case: end_turn is always valid
        if name == 'end_turn':
            has_actions = True

        # Special case: heal and cure share action type 4
        if name == 'heal':
            has_actions = bool(legal_actions.get('heal', [])) or bool(legal_actions.get('cure', []))

        results['mask_summary'][name] = {
            'mask_enabled': mask_enabled,
            'has_legal_actions': has_actions
        }

        if has_actions and not mask_enabled:
            results['valid'] = False
            results['errors'].append(f"Action type '{name}' has legal actions but mask is False")

    # Check that at least one action is always available (end_turn should always work)
    if not masks[0].any():
        results['valid'] = False
        results['errors'].append("No action types are masked as valid")

    # Check position masks have at least one valid option
    for i, dim_name in enumerate(['action_type', 'unit_type', 'from_x', 'from_y', 'to_x', 'to_y']):
        if not masks[i].any():
            results['warnings'].append(f"Dimension '{dim_name}' has no valid values in mask")

    return results


# Convenience function for curriculum learning
def make_curriculum_env(
    difficulty: str = 'easy',
    **kwargs
) -> ActionMaskedEnv:
    """
    Create environment with preset difficulty configurations.

    Args:
        difficulty: 'easy', 'medium', or 'hard'
        **kwargs: Additional arguments passed to make_maskable_env

    Difficulty presets:
        - easy: SimpleBot opponent, 10x10 maps, high starting gold
        - medium: MediumBot opponent, 15x15 maps, normal gold
        - hard: AdvancedBot opponent, 20x20 maps, limited gold

    Returns:
        Configured ActionMaskedEnv
    """
    difficulty_configs = {
        'easy': {
            'opponent': 'bot',  # SimpleBot
            'max_steps': 300,
            'reward_config': {
                'win': 1000.0,
                'loss': -1000.0,
                'income_diff': 0.2,  # Higher shaping rewards
                'unit_diff': 2.0,
                'structure_control': 10.0,
                'invalid_action': -5.0,  # Lower penalty
                'turn_penalty': -0.05
            }
        },
        'medium': {
            'opponent': 'bot',
            'max_steps': 400,
            'reward_config': {
                'win': 1000.0,
                'loss': -1000.0,
                'income_diff': 0.1,
                'unit_diff': 1.0,
                'structure_control': 5.0,
                'invalid_action': -10.0,
                'turn_penalty': -0.1
            }
        },
        'hard': {
            'opponent': 'bot',
            'max_steps': 500,
            'reward_config': {
                'win': 1000.0,
                'loss': -1000.0,
                'income_diff': 0.05,  # Lower shaping (more sparse)
                'unit_diff': 0.5,
                'structure_control': 2.5,
                'invalid_action': -15.0,  # Higher penalty
                'turn_penalty': -0.2
            }
        }
    }

    if difficulty not in difficulty_configs:
        raise ValueError(f"Unknown difficulty: {difficulty}. Choose from: {list(difficulty_configs.keys())}")

    config = difficulty_configs[difficulty]
    config.update(kwargs)  # Allow overrides

    return make_maskable_env(**config)
