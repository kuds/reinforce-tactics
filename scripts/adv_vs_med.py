#!/usr/bin/env python3
"""Run AdvancedBot vs MediumBot head-to-head."""
import argparse
import logging
import sys
from pathlib import Path

from reinforcetactics.tournament import (
    BotDescriptor,
    TournamentConfig,
    TournamentRunner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-side", type=int, default=25)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--map-dir", default="maps/1v1")
    parser.add_argument("--output-dir", default="tournament_results/adv_vs_med")
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--map-pool-mode", default="all", choices=["cycle", "random", "all"])
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    maps = sorted(str(f) for f in map_dir.glob("*.csv"))
    if not maps:
        print(f"No maps in {args.map_dir}", file=sys.stderr)
        sys.exit(1)

    config = TournamentConfig(
        name="AdvancedBot vs MediumBot",
        maps=maps,
        map_pool_mode=args.map_pool_mode,
        games_per_side=args.games_per_side,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        save_replays=False,
        concurrent_games=args.concurrent,
    )

    bots = [
        BotDescriptor.advanced_bot("AdvancedBot"),
        BotDescriptor.medium_bot("MediumBot"),
    ]

    runner = TournamentRunner(config)
    runner.run(bots)
    paths = runner.export_results()
    for fmt, path in paths.items():
        print(f"{fmt}: {path}")


if __name__ == "__main__":
    main()
