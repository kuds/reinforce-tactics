"""
Training script for AlphaZero-based agents.

Runs the full AlphaZero training pipeline:
1. Self-play with MCTS to generate training data
2. Network training on self-play data (policy + value loss)
3. Periodic evaluation against previous best network

Usage:
    # Basic training with defaults
    python train/train_alphazero.py

    # Train on a specific map with more simulations
    python train/train_alphazero.py --map-file maps/1v1/beginner.csv --num-simulations 200

    # Resume from checkpoint
    python train/train_alphazero.py --resume checkpoints/alphazero/alphazero_iter_0050.pt

    # Train with GPU
    python train/train_alphazero.py --device cuda

    # Quick test run
    python train/train_alphazero.py --iterations 2 --games-per-iter 2 --num-simulations 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reinforcetactics.rl.alphazero_trainer import AlphaZeroTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='AlphaZero training for Reinforce Tactics',
    )

    # Map configuration
    parser.add_argument(
        '--map-file', type=str, default=None,
        help='Path to map CSV file. If not specified, random maps are generated.',
    )

    # Network architecture
    parser.add_argument(
        '--res-blocks', type=int, default=6,
        help='Number of residual blocks in the network (default: 6)',
    )
    parser.add_argument(
        '--channels', type=int, default=128,
        help='Number of channels in residual blocks (default: 128)',
    )

    # MCTS configuration
    parser.add_argument(
        '--num-simulations', type=int, default=100,
        help='Number of MCTS simulations per move (default: 100)',
    )
    parser.add_argument(
        '--c-puct', type=float, default=1.5,
        help='PUCT exploration constant (default: 1.5)',
    )
    parser.add_argument(
        '--dirichlet-alpha', type=float, default=0.3,
        help='Dirichlet noise alpha (default: 0.3)',
    )

    # Training configuration
    parser.add_argument(
        '--iterations', type=int, default=100,
        help='Number of training iterations (default: 100)',
    )
    parser.add_argument(
        '--games-per-iter', type=int, default=25,
        help='Self-play games per iteration (default: 25)',
    )
    parser.add_argument(
        '--epochs-per-iter', type=int, default=10,
        help='Training epochs per iteration (default: 10)',
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Training batch size (default: 256)',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 1e-3)',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='Weight decay (default: 1e-4)',
    )
    parser.add_argument(
        '--buffer-size', type=int, default=100_000,
        help='Replay buffer capacity (default: 100000)',
    )
    parser.add_argument(
        '--max-game-steps', type=int, default=400,
        help='Maximum steps per self-play game (default: 400)',
    )
    parser.add_argument(
        '--temperature-threshold', type=int, default=30,
        help='Actions before switching to greedy in self-play (default: 30)',
    )

    # Evaluation
    parser.add_argument(
        '--eval-games', type=int, default=20,
        help='Number of evaluation games against previous best (default: 20)',
    )
    parser.add_argument(
        '--eval-threshold', type=float, default=0.55,
        help='Win rate threshold to accept new network (default: 0.55)',
    )

    # Infrastructure
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints/alphazero',
        help='Directory for saving checkpoints',
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Torch device (default: cpu)',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume training from',
    )

    # Game configuration
    parser.add_argument(
        '--enabled-units', type=str, nargs='*', default=None,
        help='Enabled unit types (e.g., W M A). Default: all units.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("AlphaZero Training Configuration:")
    for key, value in vars(args).items():
        logger.info("  %s: %s", key, value)

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer = AlphaZeroTrainer.load_checkpoint(
            args.resume,
            device=args.device,
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            epochs_per_iteration=args.epochs_per_iter,
        )
    else:
        trainer = AlphaZeroTrainer(
            map_file=args.map_file,
            num_res_blocks=args.res_blocks,
            channels=args.channels,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            replay_buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            epochs_per_iteration=args.epochs_per_iter,
            max_game_steps=args.max_game_steps,
            temperature_threshold=args.temperature_threshold,
            eval_games=args.eval_games,
            eval_win_threshold=args.eval_threshold,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            enabled_units=args.enabled_units,
        )

    history = trainer.train()

    # Print summary
    if history['iteration']:
        final_idx = -1
        logger.info("\n=== Training Complete ===")
        logger.info("Final policy loss: %.4f", history['policy_loss'][final_idx])
        logger.info("Final value loss: %.4f", history['value_loss'][final_idx])
        logger.info("Total iterations: %d", len(history['iteration']))
        logger.info("Checkpoints saved in: %s", args.checkpoint_dir)


if __name__ == '__main__':
    main()
