"""
Reinforcement Learning module for the strategy game.

This module provides:
- StrategyGameEnv: Gymnasium environment for the tactical strategy game
- Action masking utilities for efficient training with MaskablePPO
- Helper functions for creating single and vectorized environments
"""
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.masking import (
    ActionMaskedEnv,
    make_maskable_env,
    make_maskable_vec_env,
    make_curriculum_env,
    validate_action_mask,
)

__all__ = [
    'StrategyGameEnv',
    'ActionMaskedEnv',
    'make_maskable_env',
    'make_maskable_vec_env',
    'make_curriculum_env',
    'validate_action_mask',
]
