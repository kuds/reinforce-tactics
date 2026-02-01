"""
Training script for Feudal RL agent on GCP.
Supports distributed training with multiple seeds.

Supports both regular PPO and MaskablePPO (sb3-contrib) for action masking.
Action masking significantly improves training efficiency by preventing
the agent from wasting samples on invalid actions.

Usage:
    # Train with action masking (recommended)
    python train_feudal_rl.py --mode flat --use-action-masking

    # Train without action masking (baseline)
    python train_feudal_rl.py --mode flat
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Local imports
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.rl.masking import (
    make_maskable_env,
    make_maskable_vec_env,
    ActionMaskedEnv
)


def make_env(rank: int, seed: int = 0, opponent: str = 'bot', use_masking: bool = False):
    """
    Utility function for multiprocessed env.

    Args:
        rank: Index of the subprocess
        seed: Random seed
        opponent: Opponent type
        use_masking: Whether to wrap env for action masking
    """
    def _init():
        env = StrategyGameEnv(
            map_file=None,  # Random maps
            opponent=opponent,
            render_mode=None,
            max_steps=500
        )
        env.reset(seed=seed + rank)
        if use_masking:
            env = ActionMaskedEnv(env)
        return env
    set_random_seed(seed)
    return _init


def train_flat_baseline(args):
    """Train flat PPO baseline for comparison."""
    use_masking = getattr(args, 'use_action_masking', False)

    print("\n" + "="*60)
    if use_masking:
        print("Training Flat MaskablePPO (with Action Masking)")
    else:
        print("Training Flat PPO Baseline")
    print("="*60 + "\n")

    # Create output directories
    model_name = "maskable_ppo" if use_masking else "flat_ppo"
    log_dir = Path(args.log_dir) / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Create vectorized environments
    if use_masking:
        # Use masking-compatible vectorized environments
        env = make_maskable_vec_env(
            n_envs=args.n_envs,
            opponent=args.opponent,
            seed=args.seed,
            use_subprocess=(args.n_envs > 1)
        )
        # Create eval environment with masking
        eval_env = make_maskable_env(opponent=args.opponent, render_mode=None)
    else:
        # Standard environments without masking
        if args.n_envs > 1:
            env = SubprocVecEnv([make_env(i, args.seed, args.opponent, use_masking=False) for i in range(args.n_envs)])
        else:
            env = DummyVecEnv([make_env(0, args.seed, args.opponent, use_masking=False)])
        eval_env = StrategyGameEnv(opponent=args.opponent, render_mode=None)

    # Create model - use MaskablePPO if action masking is enabled
    if use_masking:
        try:
            from sb3_contrib import MaskablePPO
            print("Using MaskablePPO from sb3-contrib")
        except ImportError:
            raise ImportError(
                "sb3-contrib is required for action masking. "
                "Install with: pip install sb3-contrib"
            )

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
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
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
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

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="flat_ppo"
    )

    # Train
    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    final_path = log_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\n✅ Training complete! Model saved to {final_path}")

    # Save training config
    config = vars(args)
    config_path = log_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return log_dir


def train_feudal_rl(args):
    """Train Feudal RL agent with Manager-Worker hierarchy."""
    from torch.utils.tensorboard import SummaryWriter
    from reinforcetactics.rl.feudal_rl import FeudalRLAgent

    print("\n" + "="*60)
    print("Training Feudal RL Agent (Manager-Worker Hierarchy)")
    print("="*60 + "\n")

    # Create output directories
    log_dir = Path(args.log_dir) / f"feudal_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    (log_dir / "best_model").mkdir(exist_ok=True)

    # Create environments (single env for feudal — hierarchy is per-episode stateful)
    env = StrategyGameEnv(
        map_file=None, opponent=args.opponent,
        render_mode=None, max_steps=500
    )
    eval_env = StrategyGameEnv(
        map_file=None, opponent=args.opponent,
        render_mode=None, max_steps=500
    )

    # Create agent
    agent = FeudalRLAgent(
        observation_space=env.observation_space,
        grid_width=env.grid_width,
        grid_height=env.grid_height,
        device=args.device
    )
    agent.manager_horizon = args.manager_horizon

    # Setup training
    agent.setup_training(
        learning_rate=args.learning_rate,
        manager_lr_scale=args.manager_lr_scale,
        worker_lr_scale=args.worker_lr_scale,
    )

    # Initialize
    obs, _ = env.reset()
    agent._last_obs = obs  # pylint: disable=protected-access
    agent.reset_goal()

    writer = SummaryWriter(str(log_dir / "tensorboard"))

    num_updates = args.total_timesteps // args.n_steps
    total_timesteps = 0
    best_eval_reward = float('-inf')

    print(f"Manager horizon: {args.manager_horizon}")
    print(f"Worker reward alpha: {args.worker_reward_alpha}")
    print(f"Updates to run: {num_updates}")
    print(f"Steps per update: {args.n_steps}\n")

    for update_idx in range(num_updates):
        # Collect rollout
        buf = agent.collect_rollout(
            env, n_steps=args.n_steps,
            gamma=args.gamma, gae_lambda=args.gae_lambda,
            worker_reward_alpha=args.worker_reward_alpha,
        )
        total_timesteps += args.n_steps

        # PPO update
        losses = agent.update(
            buf, n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
        )

        # Log to TensorBoard
        for key, val in losses.items():
            writer.add_scalar(f"train/{key}", val, total_timesteps)
        writer.add_scalar("train/manager_segments", len(buf.m_rewards), total_timesteps)
        writer.add_scalar("train/worker_mean_reward", float(buf.w_rewards.mean()), total_timesteps)

        # Progress logging
        if (update_idx + 1) % 10 == 0:
            print(f"[{total_timesteps:,}] w_policy={losses.get('worker_policy_loss', 0):.3f} "
                  f"m_policy={losses.get('manager_policy_loss', 0):.3f} "
                  f"w_entropy={losses.get('worker_entropy', 0):.3f}")

        # Periodic evaluation
        if total_timesteps % args.eval_freq < args.n_steps:
            eval_results = agent.evaluate(eval_env, n_episodes=args.n_eval_episodes)
            writer.add_scalar("eval/mean_reward", eval_results['mean_reward'], total_timesteps)
            writer.add_scalar("eval/win_rate", eval_results['win_rate'], total_timesteps)
            print(f"  EVAL [{total_timesteps:,}] reward={eval_results['mean_reward']:.1f} "
                  f"win_rate={eval_results['win_rate']:.2f}")

            if eval_results['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_results['mean_reward']
                agent.save_checkpoint(str(log_dir / "best_model" / "best_feudal.pt"))

        # Periodic checkpoint
        if total_timesteps % args.checkpoint_freq < args.n_steps:
            agent.save_checkpoint(str(checkpoint_dir / f"feudal_{total_timesteps}.pt"))

    # Save final model and config
    agent.save_checkpoint(str(log_dir / "final_model.pt"))
    config = vars(args)
    config_path = log_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    writer.close()
    print(f"\nTraining complete! Model saved to {log_dir / 'final_model.pt'}")

    return log_dir


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train RL agents for Reinforce Tactics")

    # Training mode
    parser.add_argument('--mode', type=str, default='flat', choices=['flat', 'feudal'],
                       help='Training mode: flat baseline or feudal RL')

    # Environment args
    parser.add_argument('--opponent', type=str, default='bot', choices=['bot', 'random', 'self'],
                       help='Opponent type')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--use-action-masking', action='store_true',
                       help='Use MaskablePPO with action masking (recommended for faster training)')

    # Training args
    parser.add_argument('--total-timesteps', type=int, default=10000000,
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

    # Feudal RL hyperparameters
    parser.add_argument('--manager-horizon', type=int, default=10,
                       help='Worker steps between manager goal updates')
    parser.add_argument('--worker-reward-alpha', type=float, default=0.5,
                       help='Weight of extrinsic reward in worker reward (0=intrinsic only, 1=extrinsic only)')
    parser.add_argument('--manager-lr-scale', type=float, default=1.0,
                       help='Manager learning rate multiplier relative to base LR')
    parser.add_argument('--worker-lr-scale', type=float, default=1.0,
                       help='Worker learning rate multiplier relative to base LR')

    # Evaluation args
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--checkpoint-freq', type=int, default=50000,
                       help='Checkpoint save frequency')

    # Logging args
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Logging directory')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='reinforcetactics',
                       help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='W&B entity name')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n🚀 Starting training on {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Action masking: {'enabled (MaskablePPO)' if args.use_action_masking else 'disabled (standard PPO)'}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel envs: {args.n_envs}")

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
            print("✅ Weights & Biases initialized")
        except ImportError:
            print("⚠️  wandb not installed, skipping W&B logging")

    # Train
    if args.mode == 'flat':
        log_dir = train_flat_baseline(args)
    elif args.mode == 'feudal':
        log_dir = train_feudal_rl(args)

    print("\n✅ Training complete!")
    print(f"Logs saved to: {log_dir}")

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
