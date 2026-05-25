"""
Reinforcement Learning module for the strategy game.

This module provides:
- StrategyGameEnv: Gymnasium environment for the tactical strategy game
- Action masking utilities for efficient training with MaskablePPO
- Self-play utilities for training agents against themselves
- AlphaZero: dual-head network + MCTS for planning-based RL
- Helper functions for creating single and vectorized environments
"""

from reinforcetactics.rl.alphazero_net import AlphaZeroNet
from reinforcetactics.rl.alphazero_trainer import AlphaZeroTrainer, ReplayBuffer
from reinforcetactics.rl.bootstrap import (
    make_stage_env,
    record_curriculum_replays,
)
from reinforcetactics.rl.evaluation import evaluate_model
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.imitation import (
    BCStats,
    Demonstration,
    DemonstrationDataset,
    DemonstrationScenario,
    EpisodeOutcome,
    ScenarioStats,
    behavior_clone,
    collect_demonstrations,
    collect_demonstrations_multi,
    evaluate_bc_against_bot_ladder,
    format_scenario_stats_table,
    load_scenarios_from_yaml,
    make_warm_started_model,
    record_episode,
)
from reinforcetactics.rl.masking import (
    ActionMaskedEnv,
    make_maskable_env,
    make_maskable_vec_env,
    validate_action_mask,
)
from reinforcetactics.rl.mcts import MCTS
from reinforcetactics.rl.self_play import (
    OpponentPool,
    SelfPlayCallback,
    SelfPlayEnv,
    make_self_play_env,
    make_self_play_vec_env,
)

__all__ = [
    # Core environment
    "StrategyGameEnv",
    # Action masking
    "ActionMaskedEnv",
    "make_maskable_env",
    "make_maskable_vec_env",
    "validate_action_mask",
    # Evaluation
    "evaluate_model",
    # Self-play
    "SelfPlayEnv",
    "OpponentPool",
    "SelfPlayCallback",
    "make_self_play_env",
    "make_self_play_vec_env",
    # AlphaZero
    "AlphaZeroNet",
    "MCTS",
    "AlphaZeroTrainer",
    "ReplayBuffer",
    # Imitation / BC warm-start
    "BCStats",
    "Demonstration",
    "DemonstrationDataset",
    "DemonstrationScenario",
    "EpisodeOutcome",
    "ScenarioStats",
    "behavior_clone",
    "collect_demonstrations",
    "collect_demonstrations_multi",
    "evaluate_bc_against_bot_ladder",
    "format_scenario_stats_table",
    "load_scenarios_from_yaml",
    "make_warm_started_model",
    "record_episode",
    # Curriculum stage helpers
    "make_stage_env",
    "record_curriculum_replays",
]
