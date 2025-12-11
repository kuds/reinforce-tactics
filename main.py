"""
Reinforce Tactics - Main Entry Point

A turn-based strategy game with reinforcement learning capabilities.

Requirements:
    pip install pygame numpy gymnasium stable-baselines3 tensorboard pandas

Usage:
    # Train with Stable-Baselines3 (PPO)
    python main.py --mode train --algorithm ppo --timesteps 100000

    # Evaluate trained agent
    python main.py --mode evaluate --model models/ppo_model.zip --episodes 10

    # Play manually (GUI mode)
    python main.py --mode play

    # View training stats
    python main.py --mode stats
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.dependency_checker import check_dependencies
from cli.commands import train_mode, evaluate_mode, play_mode, stats_mode


def main():
    """Main entry point."""
    # Initialize settings
    from reinforcetactics.utils.settings import get_settings
    from reinforcetactics.utils.language import get_language

    settings = get_settings()
    _ = get_language()  # Initialize language system

    # Ensure all directories exist
    settings.ensure_directories()
    
    parser = argparse.ArgumentParser(
        description="Reinforce Tactics - Turn-Based Strategy with RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO agent
  python main.py --mode train --algorithm ppo --timesteps 100000
  
  # Evaluate trained model
  python main.py --mode evaluate --model models/ppo_final.zip --episodes 20
  
  # Play manually
  python main.py --mode play
  
  # View stats
  python main.py --mode stats
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="play",
        choices=["train", "evaluate", "play", "stats"],
        help="Mode: train, evaluate, play, or stats"
    )
    
    # Training arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "dqn"],
        help="RL algorithm for training"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="bot",
        choices=["bot", "random", "self"],
        help="Opponent type"
    )
    parser.add_argument(
        "--map-file",
        type=str,
        default=None,
        help="Path to map file (None for random)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for saved model"
    )
    
    # Reward shaping
    parser.add_argument(
        "--reward-income",
        type=float,
        default=0.0,
        help="Reward coefficient for income difference"
    )
    parser.add_argument(
        "--reward-units",
        type=float,
        default=0.0,
        help="Reward coefficient for unit advantage"
    )
    parser.add_argument(
        "--reward-structures",
        type=float,
        default=0.0,
        help="Reward coefficient for structure control"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model for evaluation"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Route to appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "play":
        play_mode(args)
    elif args.mode == "stats":
        stats_mode(args)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()
