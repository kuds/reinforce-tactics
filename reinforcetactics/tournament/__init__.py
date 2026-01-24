"""
Reinforce Tactics Tournament Library.

This package provides a unified tournament system for running round-robin
tournaments between bots. It supports:
- Built-in bots (SimpleBot, MediumBot, AdvancedBot)
- LLM bots (OpenAI, Anthropic, Google)
- Trained model bots
- Elo rating tracking
- Concurrent game execution
- Resume interrupted tournaments
- Results export (JSON/CSV)

Example usage:
    from reinforcetactics.tournament import (
        TournamentRunner,
        BotDescriptor,
        TournamentConfig
    )

    # Create bot descriptors
    bots = [
        BotDescriptor.simple_bot("SimpleBot"),
        BotDescriptor.medium_bot("MediumBot"),
    ]

    # Configure and run tournament
    config = TournamentConfig(
        maps=["maps/1v1/6x6_beginner.csv"],
        games_per_side=2
    )
    runner = TournamentRunner(config)
    results = runner.run(bots)
"""

from .elo import EloRatingSystem
from .bots import BotDescriptor, BotType, create_bot_instance
from .schedule import (
    ScheduledGame,
    MapConfig,
    generate_round_robin_schedule,
)
from .results import (
    GameResult,
    TournamentResults,
    ResultsExporter,
)
from .config import TournamentConfig
from .runner import TournamentRunner

__all__ = [
    # Elo
    "EloRatingSystem",
    # Bots
    "BotDescriptor",
    "BotType",
    "create_bot_instance",
    # Scheduling
    "ScheduledGame",
    "MapConfig",
    "generate_round_robin_schedule",
    # Results
    "GameResult",
    "TournamentResults",
    "ResultsExporter",
    # Config
    "TournamentConfig",
    # Runner
    "TournamentRunner",
]
