#!/usr/bin/env python3
# pylint: disable=logging-fstring-interpolation
"""
Round-robin tournament script for Reinforce Tactics bots.

Runs tournaments between all configured bots, including:
- Built-in SimpleBot, MediumBot, AdvancedBot
- LLM bots (if API keys configured and working)
- Trained model bots (from models/ directory)

Each matchup consists of multiple games with sides swapped to account
for first-move advantage.

This script is a thin CLI wrapper around the tournament library.
For programmatic use, import from reinforcetactics.tournament directly.

Example usage:
    # Run with default settings
    python scripts/tournament.py

    # Run with multiple maps
    python scripts/tournament.py --maps maps/1v1/6x6_beginner.csv maps/1v1/8x8_islands.csv

    # Run with all maps from a directory
    python scripts/tournament.py --map-dir maps/1v1/ --map-pool-mode all
"""
import argparse
import logging
import sys
from pathlib import Path

from reinforcetactics.tournament import (
    TournamentConfig,
    TournamentRunner,
    BotDescriptor,
)
from reinforcetactics.tournament.bots import (
    discover_all_bots,
    discover_builtin_bots,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for tournament script."""
    parser = argparse.ArgumentParser(
        description='Run a round-robin tournament between Reinforce Tactics bots'
    )
    parser.add_argument(
        '--map',
        help='Path to single map file (for backward compatibility)'
    )
    parser.add_argument(
        '--maps',
        nargs='+',
        help='List of map file paths to use in evaluation'
    )
    parser.add_argument(
        '--map-dir',
        help='Directory to load all maps from (alternative to listing individual maps)'
    )
    parser.add_argument(
        '--map-pool-mode',
        choices=['cycle', 'random', 'all'],
        default='cycle',
        help='How to select maps: cycle (default), random, or all'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Path to models directory (default: models/)'
    )
    parser.add_argument(
        '--output-dir',
        default='tournament_results',
        help='Path for results and replays (default: tournament_results/)'
    )
    parser.add_argument(
        '--games-per-side',
        type=int,
        default=2,
        help='Games per side in each matchup (default: 2, meaning 4 total per matchup)'
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=500,
        help='Maximum turns per game (default: 500)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: add duplicate SimpleBots for testing the tournament system'
    )
    parser.add_argument(
        '--log-conversations',
        action='store_true',
        help='Enable LLM conversation logging to JSON files'
    )
    parser.add_argument(
        '--conversation-log-dir',
        help='Directory for conversation logs (default: output_dir/llm_conversations/)'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=1,
        help='Number of concurrent games (default: 1, sequential)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM bot discovery'
    )
    parser.add_argument(
        '--no-models',
        action='store_true',
        help='Skip trained model bot discovery'
    )

    args = parser.parse_args()

    # Handle map arguments with clear precedence
    maps = []
    if args.maps:
        maps = args.maps
    elif args.map_dir:
        map_dir = Path(args.map_dir)
        if not map_dir.exists():
            logger.error(f"Map directory not found: {args.map_dir}")
            sys.exit(1)
        maps = sorted([str(f) for f in map_dir.glob('*.csv')])
        if not maps:
            logger.error(f"No .csv map files found in: {args.map_dir}")
            sys.exit(1)
    elif args.map:
        if not Path(args.map).exists():
            logger.error(f"Map file not found: {args.map}")
            sys.exit(1)
        maps = [args.map]
    else:
        maps = ['maps/1v1/6x6_beginner.csv']

    # Validate all map files exist
    for map_file in maps:
        if not Path(map_file).exists():
            logger.error(f"Map file not found: {map_file}")
            sys.exit(1)

    # Create tournament configuration
    config = TournamentConfig(
        name="CLI Tournament",
        maps=maps,
        map_pool_mode=args.map_pool_mode,
        games_per_side=args.games_per_side,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        save_replays=True,
        log_conversations=args.log_conversations,
        conversation_log_dir=args.conversation_log_dir,
        concurrent_games=args.concurrent,
    )

    # Discover bots
    models_dir = None if args.no_models else args.models_dir
    include_llm = not args.no_llm
    include_models = not args.no_models

    bots = discover_all_bots(
        models_dir=models_dir,
        test_keys=True,
        test_models=True,
        include_llm=include_llm,
        include_models=include_models,
    )

    # Add test bots if requested
    if args.test:
        bots.append(BotDescriptor.simple_bot("SimpleBot2"))
        logger.info("Added SimpleBot2 (test bot)")

    if len(bots) < 2:
        logger.error(
            "Need at least 2 bots for a tournament. "
            "Only built-in bots were found. Add LLM API keys or model files, "
            "or use --test to add duplicate SimpleBots for testing."
        )
        sys.exit(1)

    # Create runner and run tournament
    runner = TournamentRunner(config)

    try:
        results = runner.run(bots)

        # Export results
        paths = runner.export_results()
        logger.info(f"\nResults exported to: {config.output_dir}")
        for format_name, path in paths.items():
            logger.info(f"  {format_name}: {path}")

        logger.info("\nTournament completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Tournament failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
