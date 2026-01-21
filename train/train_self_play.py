"""
Training script for self-play RL agents.

Self-play training allows the agent to learn by playing against copies of itself,
enabling continuous improvement without requiring hand-crafted opponents.

Features:
- Fictitious self-play with opponent pool
- Periodic opponent updates
- Mixed training (self-play + bots)
- Curriculum learning integration
- Win rate tracking and model selection

Usage:
    # Basic self-play training
    python train_self_play.py

    # With opponent pool
    python train_self_play.py --use-opponent-pool --pool-size 10

    # Mixed training (50% self-play, 50% bots)
    python train_self_play.py --mixed-training --bot-ratio 0.5

    # Resume from checkpoint
    python train_self_play.py --resume-from logs/self_play_xxx/checkpoints/model_100000.zip
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor

# Local imports
from reinforcetactics.rl.self_play import (
    OpponentPool,
    SelfPlayEnv,
    make_self_play_env,
    make_self_play_vec_env,
)
from reinforcetactics.rl.masking import make_maskable_vec_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfPlayTrainingCallback(BaseCallback):
    """
    Custom callback for self-play training that integrates with SB3.

    Handles:
    - Opponent model updates
    - Adding models to opponent pool
    - Win rate tracking
    - Logging
    """

    def __init__(
        self,
        envs: List[SelfPlayEnv],
        opponent_pool: Optional[OpponentPool] = None,
        update_freq: int = 10000,
        add_to_pool_freq: int = 50000,
        min_win_rate_for_pool: float = 0.55,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.envs = envs
        self.opponent_pool = opponent_pool
        self.update_freq = update_freq
        self.add_to_pool_freq = add_to_pool_freq
        self.min_win_rate_for_pool = min_win_rate_for_pool

        self.win_rate_history: List[float] = []
        self.pool_additions = 0

    def _on_training_start(self) -> None:
        """Initialize opponents with the current model."""
        logger.info("Initializing self-play opponents with current model...")
        self._update_all_opponents()

    def _on_step(self) -> bool:
        """Called after each step."""
        # Update opponents periodically
        if self.n_calls % self.update_freq == 0:
            self._update_all_opponents()
            self._log_stats()

        # Add to pool periodically
        if self.opponent_pool and self.n_calls % self.add_to_pool_freq == 0:
            self._try_add_to_pool()

        return True

    def _update_all_opponents(self) -> None:
        """Update all opponent models to current policy."""
        for env in self.envs:
            env.set_opponent_model(self.model)
            env.update_opponent_from_current()

    def _try_add_to_pool(self) -> None:
        """Add current model to pool if win rate is good enough."""
        if not self.opponent_pool:
            return

        avg_win_rate = self._get_average_win_rate()
        self.win_rate_history.append(avg_win_rate)

        if avg_win_rate >= self.min_win_rate_for_pool:
            self.opponent_pool.add_model(
                self.model,
                timestep=self.num_timesteps,
                win_rate=avg_win_rate
            )
            self.pool_additions += 1
            logger.info(
                "Step %d: Added model to pool (win rate: %.2f%%, pool size: %d)",
                self.num_timesteps, avg_win_rate * 100, self.opponent_pool.size
            )
        else:
            logger.info(
                "Step %d: Win rate %.2f%% below threshold %.2f%%, not adding to pool",
                self.num_timesteps, avg_win_rate * 100, self.min_win_rate_for_pool * 100
            )

    def _get_average_win_rate(self) -> float:
        """Get average win rate across all environments."""
        win_rates = []
        for env in self.envs:
            win_rates.append(env.get_win_rate())
        return np.mean(win_rates) if win_rates else 0.5

    def _log_stats(self) -> None:
        """Log training statistics."""
        avg_win_rate = self._get_average_win_rate()

        # Get total games and wins
        total_games = sum(env.stats['total_games'] for env in self.envs)
        total_wins = sum(env.stats['agent_wins'] for env in self.envs)

        logger.info(
            "Step %d: Win rate: %.2f%%, Total games: %d, Wins: %d",
            self.num_timesteps, avg_win_rate * 100, total_games, total_wins
        )

        # Log to tensorboard if available
        if self.logger:
            self.logger.record("self_play/win_rate", avg_win_rate)
            self.logger.record("self_play/total_games", total_games)
            if self.opponent_pool:
                self.logger.record("self_play/pool_size", self.opponent_pool.size)


class MixedTrainingCallback(BaseCallback):
    """
    Callback for mixed training that alternates between self-play and bot opponents.

    This helps maintain performance against fixed strategies while also
    improving through self-play.
    """

    def __init__(
        self,
        self_play_envs: List[SelfPlayEnv],
        bot_vec_env: Any,
        bot_ratio: float = 0.5,
        switch_freq: int = 5000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.self_play_envs = self_play_envs
        self.bot_vec_env = bot_vec_env
        self.bot_ratio = bot_ratio
        self.switch_freq = switch_freq
        self.using_bots = False

    def _on_step(self) -> bool:
        """Potentially switch training environments."""
        if self.n_calls % self.switch_freq == 0:
            self.using_bots = np.random.random() < self.bot_ratio
            if self.verbose >= 1:
                env_type = "bot" if self.using_bots else "self-play"
                logger.info("Step %d: Switched to %s training", self.n_calls, env_type)

        return True


def get_self_play_envs_from_vec(vec_env: Any) -> List[SelfPlayEnv]:
    """Extract SelfPlayEnv instances from a vectorized environment."""
    envs = []

    # Handle different vectorized env types
    if hasattr(vec_env, 'envs'):
        for env in vec_env.envs:
            # Unwrap to find SelfPlayEnv
            current = env
            while current is not None:
                if isinstance(current, SelfPlayEnv):
                    envs.append(current)
                    break
                current = getattr(current, 'env', None)

    return envs


def train_self_play(args) -> Path:
    """
    Main self-play training function.

    Args:
        args: Parsed command-line arguments

    Returns:
        Path to the log directory
    """
    logger.info("\n" + "=" * 60)
    logger.info("Self-Play Training")
    logger.info("=" * 60 + "\n")

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f"self_play_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    pool_dir = log_dir / "opponent_pool" if args.use_opponent_pool else None

    # Set random seed
    set_random_seed(args.seed)

    # Create opponent pool if enabled
    opponent_pool = None
    if args.use_opponent_pool:
        opponent_pool = OpponentPool(
            max_size=args.pool_size,
            selection_strategy=args.pool_strategy,
            save_dir=str(pool_dir)
        )
        logger.info(
            "Created opponent pool (max size: %d, strategy: %s)",
            args.pool_size, args.pool_strategy
        )

    # Create self-play vectorized environments
    logger.info("Creating %d self-play environments...", args.n_envs)
    vec_env = make_self_play_vec_env(
        n_envs=args.n_envs,
        max_steps=args.max_steps,
        seed=args.seed,
        use_subprocess=(args.n_envs > 1 and not args.no_subprocess),
        opponent_pool=opponent_pool,
        swap_players=args.swap_players,
        enabled_units=args.enabled_units.split(',') if args.enabled_units else None
    )

    # Wrap with monitor for logging
    vec_env = VecMonitor(vec_env)

    # Get self-play env references for callback
    self_play_envs = get_self_play_envs_from_vec(vec_env)
    logger.info("Found %d self-play environments", len(self_play_envs))

    # Create evaluation environment
    eval_env = make_self_play_env(
        max_steps=args.max_steps,
        swap_players=False  # Consistent evaluation
    )

    # Import MaskablePPO
    try:
        from sb3_contrib import MaskablePPO
        logger.info("Using MaskablePPO from sb3-contrib")
    except ImportError:
        raise ImportError(
            "sb3-contrib is required for self-play training. "
            "Install with: pip install sb3-contrib"
        )

    # Create or load model
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        model = MaskablePPO.load(
            args.resume_from,
            env=vec_env,
            device=args.device
        )
    else:
        model = MaskablePPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
            tensorboard_log=str(log_dir / "tensorboard"),
            device=args.device
        )

    # Initialize opponents with the model
    for env in self_play_envs:
        env.set_opponent_model(model)
        env.update_opponent_from_current()

    # Set opponent for eval env
    eval_env.set_opponent_model(model)
    eval_env.update_opponent_from_current()

    # Create callbacks
    callbacks = []

    # Self-play callback
    self_play_callback = SelfPlayTrainingCallback(
        envs=self_play_envs,
        opponent_pool=opponent_pool,
        update_freq=args.opponent_update_freq,
        add_to_pool_freq=args.add_to_pool_freq,
        min_win_rate_for_pool=args.min_win_rate_for_pool,
        verbose=1
    )
    callbacks.append(self_play_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="self_play"
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback (evaluate against self)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True
    )
    callbacks.append(eval_callback)

    # Save training config
    config = vars(args)
    config_path = log_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    logger.info("Saved config to %s", config_path)

    # Train
    logger.info("Starting training for %s timesteps...", f"{args.total_timesteps:,}")
    logger.info("Opponent update frequency: %d", args.opponent_update_freq)
    if opponent_pool:
        logger.info("Add to pool frequency: %d", args.add_to_pool_freq)
        logger.info("Min win rate for pool: %.2f%%", args.min_win_rate_for_pool * 100)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True
    )

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    logger.info("Training complete! Model saved to %s", final_path)

    # Save final statistics
    stats = {
        'total_timesteps': args.total_timesteps,
        'final_win_rate': self_play_callback._get_average_win_rate(),
        'win_rate_history': self_play_callback.win_rate_history,
        'pool_additions': self_play_callback.pool_additions,
        'pool_size': opponent_pool.size if opponent_pool else 0
    }
    stats_path = log_dir / "final_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    return log_dir


def train_mixed(args) -> Path:
    """
    Mixed training: alternating between self-play and bot opponents.

    This approach maintains performance against fixed strategies (bots)
    while also improving through self-play.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Mixed Training (Self-Play + Bots)")
    logger.info("=" * 60 + "\n")

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f"mixed_training_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Set random seed
    set_random_seed(args.seed)

    # Create opponent pool
    opponent_pool = None
    if args.use_opponent_pool:
        pool_dir = log_dir / "opponent_pool"
        opponent_pool = OpponentPool(
            max_size=args.pool_size,
            selection_strategy=args.pool_strategy,
            save_dir=str(pool_dir)
        )

    # Create self-play environments
    logger.info("Creating self-play environments...")
    self_play_vec_env = make_self_play_vec_env(
        n_envs=args.n_envs // 2,  # Half for self-play
        max_steps=args.max_steps,
        seed=args.seed,
        use_subprocess=False,  # Use DummyVecEnv for easier env switching
        opponent_pool=opponent_pool,
        swap_players=args.swap_players
    )

    # Create bot environments
    logger.info("Creating bot environments...")
    _ = make_maskable_vec_env(
        n_envs=args.n_envs // 2,  # Half for bots
        opponent='bot',
        max_steps=args.max_steps,
        seed=args.seed + 1000,
        use_subprocess=False
    )

    # Use self-play env as primary (will switch during training)
    vec_env = VecMonitor(self_play_vec_env)

    # Get self-play envs for callback
    self_play_envs = get_self_play_envs_from_vec(self_play_vec_env)

    # Create evaluation environment
    eval_env = make_self_play_env(max_steps=args.max_steps, swap_players=False)

    # Import and create model
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise ImportError("sb3-contrib is required. Install with: pip install sb3-contrib") from exc

    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        device=args.device
    )

    # Initialize opponents
    for env in self_play_envs:
        env.set_opponent_model(model)
        env.update_opponent_from_current()
    eval_env.set_opponent_model(model)

    # Create callbacks
    callbacks = []

    # Self-play callback
    self_play_callback = SelfPlayTrainingCallback(
        envs=self_play_envs,
        opponent_pool=opponent_pool,
        update_freq=args.opponent_update_freq,
        add_to_pool_freq=args.add_to_pool_freq,
        min_win_rate_for_pool=args.min_win_rate_for_pool
    )
    callbacks.append(self_play_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="mixed"
    )
    callbacks.append(checkpoint_callback)

    # Save config
    config = vars(args)
    with open(log_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # Train
    logger.info("Starting mixed training for %s timesteps...", f"{args.total_timesteps:,}")
    logger.info("Bot ratio: %.2f%%", args.bot_ratio * 100)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True
    )

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    logger.info("Training complete! Model saved to %s", final_path)

    return log_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train RL agents with self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training mode
    parser.add_argument('--mode', type=str, default='self-play',
                       choices=['self-play', 'mixed'],
                       help='Training mode')

    # Self-play settings
    parser.add_argument('--swap-players', action='store_true', default=True,
                       help='Randomly swap player order each episode')
    parser.add_argument('--opponent-update-freq', type=int, default=10000,
                       help='How often to update opponent model (steps)')

    # Opponent pool settings
    parser.add_argument('--use-opponent-pool', action='store_true',
                       help='Use pool of historical opponents')
    parser.add_argument('--pool-size', type=int, default=10,
                       help='Maximum size of opponent pool')
    parser.add_argument('--pool-strategy', type=str, default='uniform',
                       choices=['uniform', 'recent', 'prioritized'],
                       help='Opponent selection strategy')
    parser.add_argument('--add-to-pool-freq', type=int, default=50000,
                       help='How often to add model to pool (steps)')
    parser.add_argument('--min-win-rate-for-pool', type=float, default=0.55,
                       help='Minimum win rate to add model to pool')

    # Mixed training settings
    parser.add_argument('--bot-ratio', type=float, default=0.3,
                       help='Ratio of training against bots (mixed mode)')

    # Environment settings
    parser.add_argument('--n-envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--no-subprocess', action='store_true',
                       help='Use DummyVecEnv instead of SubprocVecEnv')
    parser.add_argument('--enabled-units', type=str, default=None,
                       help='Comma-separated list of enabled unit types')

    # Training settings
    parser.add_argument('--total-timesteps', type=int, default=5000000,
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cpu, cuda, or auto')

    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5,
                       help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='Max gradient norm')

    # Evaluation settings
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                       help='Checkpoint save frequency')

    # Logging settings
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Logging directory')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Weights & Biases
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='reinforcetactics-selfplay',
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='W&B entity name')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Print settings
    logger.info("Starting training on %s", args.device)
    logger.info("Mode: %s", args.mode)
    logger.info("Total timesteps: %s", f"{args.total_timesteps:,}")
    logger.info("Parallel envs: %d", args.n_envs)
    logger.info("Opponent pool: %s", 'enabled' if args.use_opponent_pool else 'disabled')

    # Initialize W&B if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            logger.info("Weights & Biases initialized")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    # Train
    if args.mode == 'self-play':
        log_dir = train_self_play(args)
    elif args.mode == 'mixed':
        log_dir = train_mixed(args)

    logger.info("Training complete! Logs saved to: %s", log_dir)

    if args.wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
