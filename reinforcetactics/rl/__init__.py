"""
Reinforcement Learning module for the strategy game.

This module provides:
- StrategyGameEnv: Gymnasium environment for the tactical strategy game
- Action masking utilities for efficient training with MaskablePPO
- Self-play utilities for training agents against themselves
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
from reinforcetactics.rl.self_play import (
    SelfPlayEnv,
    OpponentPool,
    SelfPlayCallback,
    make_self_play_env,
    make_self_play_vec_env,
)

__all__ = [
    # Core environment
    'StrategyGameEnv',
    # Action masking
    'ActionMaskedEnv',
    'make_maskable_env',
    'make_maskable_vec_env',
    'make_curriculum_env',
    'validate_action_mask',
    # Self-play
    'SelfPlayEnv',
    'OpponentPool',
    'SelfPlayCallback',
    'make_self_play_env',
    'make_self_play_vec_env',
]
