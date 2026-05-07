"""
Training script for AlphaZero-based agents.

Runs the full AlphaZero training pipeline:
1. Self-play with MCTS to generate training data
2. Network training on self-play data (policy + value loss)
3. Periodic evaluation against previous best network

Usage:
    # Basic training with defaults
    python scripts/train/train_alphazero.py

    # Train on a specific map with more simulations
    python scripts/train/train_alphazero.py --map-file maps/1v1/beginner.csv --num-simulations 200

    # Resume from checkpoint
    python scripts/train/train_alphazero.py --resume checkpoints/alphazero/alphazero_iter_0050.pt

    # Train with GPU
    python scripts/train/train_alphazero.py --device cuda

    # Quick test run
    python scripts/train/train_alphazero.py --iterations 2 --games-per-iter 2 --num-simulations 10
"""

import argparse
import logging
import sys
from pathlib import Path

from stable_baselines3.common.utils import set_random_seed

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from reinforcetactics.rl.alphazero_trainer import AlphaZeroTrainer  # noqa: E402
from reinforcetactics.rl.config import TrainingConfig, config_to_argparse_defaults, load_config  # noqa: E402
from reinforcetactics.rl.training_utils import resolve_device  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_ARG_TO_CONFIG_PATH = {
    "seed": "seed",
    "map_file": "env.map_file",
    "res_blocks": "alphazero.res_blocks",
    "channels": "alphazero.channels",
    "num_simulations": "alphazero.num_simulations",
    "c_puct": "alphazero.c_puct",
    "dirichlet_alpha": "alphazero.dirichlet_alpha",
    "iterations": "alphazero.iterations",
    "games_per_iter": "alphazero.games_per_iter",
    "epochs_per_iter": "alphazero.epochs_per_iter",
    "batch_size": "alphazero.batch_size",
    "buffer_size": "alphazero.buffer_size",
    "max_game_steps": "alphazero.max_game_steps",
    "temperature_threshold": "alphazero.temperature_threshold",
    "eval_games": "alphazero.eval_games",
    "eval_threshold": "alphazero.eval_threshold",
    "lr": "alphazero.lr",
    "weight_decay": "alphazero.weight_decay",
    "checkpoint_dir": "logging.log_dir",
}


def _script_default_config() -> TrainingConfig:
    """AlphaZero preset on top of TrainingConfig() defaults."""
    cfg = TrainingConfig(algorithm="alphazero")
    cfg.logging.log_dir = "checkpoints/alphazero"
    return cfg


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON training config")
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_config(pre_args.config) if pre_args.config else _script_default_config()

    parser = argparse.ArgumentParser(
        description="AlphaZero training for Reinforce Tactics",
        parents=[pre_parser],
    )

    # Config-backed args (defaults filled in by parser.set_defaults below)
    parser.add_argument("--map-file", type=str, help="Path to map CSV file. If not specified, random maps are generated.")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Network architecture
    parser.add_argument("--res-blocks", type=int, help="Number of residual blocks in the network")
    parser.add_argument("--channels", type=int, help="Number of channels in residual blocks")

    # MCTS configuration
    parser.add_argument("--num-simulations", type=int, help="Number of MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, help="PUCT exploration constant")
    parser.add_argument("--dirichlet-alpha", type=float, help="Dirichlet noise alpha")

    # Training configuration
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, help="Self-play games per iteration")
    parser.add_argument("--epochs-per-iter", type=int, help="Training epochs per iteration")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--buffer-size", type=int, help="Replay buffer capacity")
    parser.add_argument("--max-game-steps", type=int, help="Maximum steps per self-play game")
    parser.add_argument("--temperature-threshold", type=int, help="Actions before switching to greedy in self-play")

    # Evaluation
    parser.add_argument("--eval-games", type=int, help="Number of evaluation games against previous best")
    parser.add_argument("--eval-threshold", type=float, help="Win rate threshold to accept new network")

    # Infrastructure (script-only flags below keep their argparse defaults)
    parser.add_argument("--checkpoint-dir", type=str, help="Directory for saving checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Torch device (default: cpu; 'auto' picks cuda if available)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    # Game configuration
    parser.add_argument(
        "--enabled-units",
        type=str,
        nargs="*",
        default=None,
        help="Enabled unit types (e.g., W M A). Default: all units.",
    )

    parser.set_defaults(**config_to_argparse_defaults(cfg, _ARG_TO_CONFIG_PATH))

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    return args


def main():
    args = parse_args()

    set_random_seed(args.seed)

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
    if history["iteration"]:
        final_idx = -1
        logger.info("\n=== Training Complete ===")
        logger.info("Final policy loss: %.4f", history["policy_loss"][final_idx])
        logger.info("Final value loss: %.4f", history["value_loss"][final_idx])
        logger.info("Total iterations: %d", len(history["iteration"]))
        logger.info("Checkpoints saved in: %s", args.checkpoint_dir)


if __name__ == "__main__":
    main()
