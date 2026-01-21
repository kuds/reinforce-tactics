#!/usr/bin/env python3
"""
Quickstart: Train an RL Agent with Action Masking

This example demonstrates how to train a reinforcement learning agent
for Reinforce Tactics using MaskablePPO from sb3-contrib.

Action masking prevents the agent from attempting invalid actions,
significantly improving training efficiency (typically 2-3x faster convergence).

Requirements:
    pip install stable-baselines3 sb3-contrib

Usage:
    python examples/train_with_action_masking.py

    # With custom settings
    python examples/train_with_action_masking.py --timesteps 500000 --difficulty medium
"""

import sys
import argparse
from pathlib import Path

# Check for required packages
try:
    from sb3_contrib import MaskablePPO
except ImportError:
    print("Error: sb3-contrib is required for action masking.")
    print("Install with: pip install sb3-contrib")
    sys.exit(1)

from reinforcetactics.rl import (
    make_maskable_env,
    make_maskable_vec_env,
    make_curriculum_env,
    validate_action_mask,
)


def train_basic(timesteps: int = 100000, save_path: str = "models/maskable_ppo_basic.zip"):
    """
    Basic training example with action masking.

    This is the simplest way to get started with RL training.
    """
    print("\n" + "=" * 60)
    print("Basic Training with Action Masking")
    print("=" * 60)

    # Create a single environment with action masking
    env = make_maskable_env(opponent="bot")

    # Validate that action masks are working correctly
    print("\nValidating action masks...")
    validation = validate_action_mask(env.env)
    if validation['valid']:
        print("Action masks are valid!")
    else:
        print("Warning: Action mask validation found issues:")
        for error in validation['errors']:
            print(f"  - {error}")

    # Create MaskablePPO model
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
    )

    # Train
    print(f"\nTraining for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return model


def train_parallel(
    timesteps: int = 500000,
    n_envs: int = 4,
    save_path: str = "models/maskable_ppo_parallel.zip"
):
    """
    Parallel training with multiple environments.

    Using multiple environments speeds up training by collecting
    experience in parallel.
    """
    print("\n" + "=" * 60)
    print(f"Parallel Training with {n_envs} Environments")
    print("=" * 60)

    # Create vectorized environments
    vec_env = make_maskable_vec_env(
        n_envs=n_envs,
        opponent="bot",
        use_subprocess=True  # Use separate processes for true parallelism
    )

    # Create MaskablePPO model
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
    )

    # Train
    print(f"\nTraining for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return model


def train_with_curriculum(
    timesteps_per_stage: int = 100000,
    save_path: str = "models/maskable_ppo_curriculum.zip"
):
    """
    Curriculum learning: progressively increase difficulty.

    This approach often produces better agents by starting with
    easier tasks and gradually increasing complexity.
    """
    print("\n" + "=" * 60)
    print("Curriculum Learning: Easy -> Medium -> Hard")
    print("=" * 60)

    difficulties = ['easy', 'medium', 'hard']
    model = None

    for i, difficulty in enumerate(difficulties):
        print(f"\n--- Stage {i+1}/3: {difficulty.upper()} ---")

        # Create environment for this difficulty level
        env = make_curriculum_env(difficulty=difficulty)

        if model is None:
            # Create new model for first stage
            model = MaskablePPO(
                "MultiInputPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                verbose=1,
            )
        else:
            # Transfer to new environment
            model.set_env(env)

        # Train at this difficulty
        print(f"Training for {timesteps_per_stage:,} timesteps at {difficulty} difficulty...")
        model.learn(
            total_timesteps=timesteps_per_stage, progress_bar=True, reset_num_timesteps=False
        )

        # Save checkpoint
        checkpoint_path = f"models/curriculum_stage_{i+1}_{difficulty}.zip"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\nFinal model saved to: {save_path}")

    return model


def evaluate_agent(model_path: str, num_episodes: int = 10):
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
    """
    print("\n" + "=" * 60)
    print(f"Evaluating Agent: {model_path}")
    print("=" * 60)

    # Load model
    model = MaskablePPO.load(model_path)

    # Create evaluation environment
    env = make_maskable_env(opponent="bot", render_mode=None)

    wins = 0
    total_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get action mask and predict
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        if info.get('winner') == 1:
            wins += 1
            result = "WIN"
        else:
            result = "LOSS"

        print(f"Episode {episode + 1}: {result} (reward: {episode_reward:.1f})")

    print("\n--- Results ---")
    print(f"Win rate: {wins}/{num_episodes} ({100*wins/num_episodes:.1f}%)")
    print(f"Average reward: {sum(total_rewards)/len(total_rewards):.1f}")

    return wins / num_episodes


def watch_agent(model_path: str):
    """
    Watch a trained agent play with visualization.

    Args:
        model_path: Path to saved model
    """
    print("\n" + "=" * 60)
    print(f"Watching Agent Play: {model_path}")
    print("=" * 60)
    print("(Close the game window to exit)")

    # Load model
    model = MaskablePPO.load(model_path)

    # Create environment with rendering
    env = make_maskable_env(opponent="bot", render_mode="human")

    obs, _ = env.reset()
    done = False

    while not done:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.render()

    winner = info.get('winner', 'Unknown')
    print(f"\nGame Over! Winner: Player {winner}")

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents with action masking for Reinforce Tactics"
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='basic',
        choices=['basic', 'parallel', 'curriculum', 'evaluate', 'watch'],
        help='Training mode'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--n-envs',
        type=int,
        default=4,
        help='Number of parallel environments (for parallel mode)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/maskable_ppo_basic.zip',
        help='Path to save/load model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )

    args = parser.parse_args()

    if args.mode == 'basic':
        train_basic(timesteps=args.timesteps, save_path=args.model_path)

    elif args.mode == 'parallel':
        train_parallel(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.model_path
        )

    elif args.mode == 'curriculum':
        train_with_curriculum(
            timesteps_per_stage=args.timesteps // 3,
            save_path=args.model_path
        )

    elif args.mode == 'evaluate':
        evaluate_agent(model_path=args.model_path, num_episodes=args.episodes)

    elif args.mode == 'watch':
        watch_agent(model_path=args.model_path)


if __name__ == '__main__':
    main()
