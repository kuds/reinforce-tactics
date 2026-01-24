"""
Tournament results handling and export.

This module provides classes for tracking game results and exporting
tournament data to various formats (JSON, CSV).
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .elo import EloRatingSystem

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """
    Result of a completed tournament game.

    Attributes:
        game_id: Unique game identifier
        bot1_name: Name of bot who played as player 1
        bot2_name: Name of bot who played as player 2
        winner: Winner (0=draw, 1=bot1, 2=bot2)
        winner_name: Name of winning bot (or "Draw")
        turns: Number of turns played
        map_name: Name of the map
        replay_path: Path to replay file (if saved)
        error: Error message if game failed
    """

    game_id: int
    bot1_name: str
    bot2_name: str
    winner: int
    winner_name: str
    turns: int
    map_name: str
    replay_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "game_id": self.game_id,
            "bot1": self.bot1_name,
            "bot2": self.bot2_name,
            "winner": self.winner,
            "winner_name": self.winner_name,
            "turns": self.turns,
            "map": self.map_name,
        }
        if self.replay_path:
            result["replay_path"] = self.replay_path
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class BotStanding:
    """
    Tournament standing for a single bot.

    Attributes:
        bot_name: Name of the bot
        wins: Total wins
        losses: Total losses
        draws: Total draws
        elo: Current Elo rating
        elo_change: Elo change since start
        per_map_stats: Per-map win/loss/draw breakdown
    """

    bot_name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo: float = 1500.0
    elo_change: float = 0.0
    per_map_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def total_games(self) -> int:
        """Total games played."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bot": self.bot_name,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_games": self.total_games,
            "win_rate": self.win_rate,
            "elo": round(self.elo, 0),
            "elo_change": round(self.elo_change, 0),
            "per_map_stats": self.per_map_stats,
        }


@dataclass
class MatchupResult:
    """
    Head-to-head results between two bots.

    Attributes:
        bot1: Name of first bot (alphabetically first)
        bot2: Name of second bot
        bot1_wins: Number of wins for bot1
        bot2_wins: Number of wins for bot2
        draws: Number of draws
    """

    bot1: str
    bot2: str
    bot1_wins: int = 0
    bot2_wins: int = 0
    draws: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bot1": self.bot1,
            "bot2": self.bot2,
            "bot1_wins": self.bot1_wins,
            "bot2_wins": self.bot2_wins,
            "draws": self.draws,
        }


class TournamentResults:
    """
    Aggregates and manages tournament results.

    This class tracks all game results, updates statistics, and provides
    methods for generating standings and matchup data.
    """

    def __init__(self, elo_system: Optional[EloRatingSystem] = None):
        """
        Initialize tournament results tracker.

        Args:
            elo_system: EloRatingSystem to use (creates new one if None)
        """
        self.elo_system = elo_system or EloRatingSystem()
        self.game_results: List[GameResult] = []
        self.bot_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "losses": 0, "draws": 0}
        )
        self.per_map_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0})
        )
        self.matchup_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"bot1_wins": 0, "bot2_wins": 0, "draws": 0}
        )
        self.maps_used: List[str] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start(self) -> None:
        """Mark tournament start time."""
        self.start_time = datetime.now()

    def finish(self) -> None:
        """Mark tournament end time."""
        self.end_time = datetime.now()

    def add_game_result(self, result: GameResult) -> None:
        """
        Add a game result and update all statistics.

        Args:
            result: GameResult to add
        """
        self.game_results.append(result)

        bot1 = result.bot1_name
        bot2 = result.bot2_name
        map_name = result.map_name

        # Track maps used
        if map_name not in self.maps_used:
            self.maps_used.append(map_name)

        # Create sorted matchup key
        sorted_bots = tuple(sorted([bot1, bot2]))
        matchup_key = f"{sorted_bots[0]}|{sorted_bots[1]}"

        # Update statistics based on winner
        if result.winner == 1:  # bot1 wins
            self.bot_stats[bot1]["wins"] += 1
            self.bot_stats[bot2]["losses"] += 1
            self.per_map_stats[bot1][map_name]["wins"] += 1
            self.per_map_stats[bot2][map_name]["losses"] += 1

            if bot1 == sorted_bots[0]:
                self.matchup_stats[matchup_key]["bot1_wins"] += 1
            else:
                self.matchup_stats[matchup_key]["bot2_wins"] += 1

            self.elo_system.update_ratings(bot1, bot2, 1)

        elif result.winner == 2:  # bot2 wins
            self.bot_stats[bot2]["wins"] += 1
            self.bot_stats[bot1]["losses"] += 1
            self.per_map_stats[bot2][map_name]["wins"] += 1
            self.per_map_stats[bot1][map_name]["losses"] += 1

            if bot2 == sorted_bots[0]:
                self.matchup_stats[matchup_key]["bot1_wins"] += 1
            else:
                self.matchup_stats[matchup_key]["bot2_wins"] += 1

            self.elo_system.update_ratings(bot2, bot1, 1)

        else:  # Draw
            self.bot_stats[bot1]["draws"] += 1
            self.bot_stats[bot2]["draws"] += 1
            self.per_map_stats[bot1][map_name]["draws"] += 1
            self.per_map_stats[bot2][map_name]["draws"] += 1
            self.matchup_stats[matchup_key]["draws"] += 1
            self.elo_system.update_ratings(bot1, bot2, 0)

    def get_standings(self) -> List[BotStanding]:
        """
        Get tournament standings sorted by Elo rating.

        Returns:
            List of BotStanding objects, sorted by Elo descending
        """
        standings = []

        for bot_name, stats in self.bot_stats.items():
            standing = BotStanding(
                bot_name=bot_name,
                wins=stats["wins"],
                losses=stats["losses"],
                draws=stats["draws"],
                elo=self.elo_system.get_rating(bot_name),
                elo_change=self.elo_system.get_rating_change(bot_name),
                per_map_stats={
                    map_name: dict(map_stats)
                    for map_name, map_stats in self.per_map_stats[bot_name].items()
                },
            )
            standings.append(standing)

        # Sort by Elo descending
        standings.sort(key=lambda x: x.elo, reverse=True)
        return standings

    def get_matchups(self) -> List[MatchupResult]:
        """
        Get head-to-head matchup results.

        Returns:
            List of MatchupResult objects
        """
        matchups = []

        for matchup_key, stats in self.matchup_stats.items():
            bot1, bot2 = matchup_key.split("|")
            matchups.append(
                MatchupResult(
                    bot1=bot1,
                    bot2=bot2,
                    bot1_wins=stats["bot1_wins"],
                    bot2_wins=stats["bot2_wins"],
                    draws=stats["draws"],
                )
            )

        return matchups

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all results to dictionary for serialization.

        Returns:
            Complete results dictionary
        """
        standings = self.get_standings()
        matchups = self.get_matchups()

        return {
            "timestamp": self.end_time.isoformat() if self.end_time else datetime.now().isoformat(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "maps_used": self.maps_used,
            "total_games": len(self.game_results),
            "standings": [s.to_dict() for s in standings],
            "matchups": [m.to_dict() for m in matchups],
            "elo_history": {
                bot_name: [round(r, 0) for r in history]
                for bot_name, history in self.elo_system.rating_history.items()
            },
            "games": [g.to_dict() for g in self.game_results],
        }


class ResultsExporter:
    """
    Exports tournament results to various formats.
    """

    def __init__(self, output_dir: str):
        """
        Initialize exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        results: TournamentResults,
        config: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Export results to all formats.

        Args:
            results: TournamentResults to export
            config: Optional configuration to include
            timestamp: Optional timestamp for filenames

        Returns:
            Dictionary mapping format to file path
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        paths = {}

        # Export JSON
        paths["json"] = self.export_json(results, config, timestamp)

        # Export standings CSV
        paths["standings_csv"] = self.export_standings_csv(results, timestamp)

        # Export matchups CSV
        paths["matchups_csv"] = self.export_matchups_csv(results, timestamp)

        # Export matrix CSV
        paths["matrix_csv"] = self.export_matrix_csv(results, timestamp)

        return paths

    def export_json(
        self,
        results: TournamentResults,
        config: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> str:
        """Export results to JSON."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = results.to_dict()

        # Add config if provided (sanitized)
        if config:
            data["config"] = _sanitize_config(config)

        # Add metadata
        data["metadata"] = {
            "export_time": datetime.now().isoformat(),
            "timestamp": timestamp,
        }

        filepath = self.output_dir / f"tournament_results_{timestamp}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to: {filepath}")
        return str(filepath)

    def export_standings_csv(
        self, results: TournamentResults, timestamp: Optional[str] = None
    ) -> str:
        """Export standings to CSV."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        standings = results.get_standings()
        filepath = self.output_dir / f"tournament_standings_{timestamp}.csv"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Bot,Wins,Losses,Draws,Total Games,Win Rate,Elo,Elo Change\n")
            for s in standings:
                f.write(
                    f"{s.bot_name},{s.wins},{s.losses},{s.draws},"
                    f"{s.total_games},{s.win_rate:.3f},"
                    f"{s.elo:.0f},{s.elo_change:+.0f}\n"
                )

        logger.info(f"Standings saved to: {filepath}")
        return str(filepath)

    def export_matchups_csv(
        self, results: TournamentResults, timestamp: Optional[str] = None
    ) -> str:
        """Export matchups to CSV."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        matchups = results.get_matchups()
        filepath = self.output_dir / f"tournament_matchups_{timestamp}.csv"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Bot 1,Bot 2,Bot 1 Wins,Bot 2 Wins,Draws\n")
            for m in matchups:
                f.write(
                    f"{m.bot1},{m.bot2},{m.bot1_wins},{m.bot2_wins},{m.draws}\n"
                )

        logger.info(f"Matchups saved to: {filepath}")
        return str(filepath)

    def export_matrix_csv(
        self, results: TournamentResults, timestamp: Optional[str] = None
    ) -> str:
        """Export results as a matrix table."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        matchups = results.get_matchups()
        filepath = self.output_dir / f"tournament_matrix_{timestamp}.csv"

        # Get all bot names
        bots_set = set()
        for m in matchups:
            bots_set.add(m.bot1)
            bots_set.add(m.bot2)
        bots_list = sorted(list(bots_set))

        # Build matrix
        matrix = {b1: {b2: "0-0-0" for b2 in bots_list} for b1 in bots_list}
        for b in bots_list:
            matrix[b][b] = "X"

        for m in matchups:
            b1, b2 = m.bot1, m.bot2
            w1, w2, d = m.bot1_wins, m.bot2_wins, m.draws
            matrix[b1][b2] = f"{w1}-{w2}-{d}"
            matrix[b2][b1] = f"{w2}-{w1}-{d}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("," + ",".join(bots_list) + "\n")
            for b1 in bots_list:
                row = [matrix[b1][b2] for b2 in bots_list]
                f.write(f"{b1}," + ",".join(row) + "\n")

        logger.info(f"Matrix saved to: {filepath}")
        return str(filepath)

    def print_standings(self, results: TournamentResults) -> None:
        """Print standings to logger."""
        standings = results.get_standings()

        logger.info("=" * 84)
        logger.info("TOURNAMENT RESULTS")
        logger.info("=" * 84)
        logger.info(
            f"{'Rank':<6}{'Bot':<25}{'Wins':<8}{'Losses':<8}{'Draws':<8}"
            f"{'Win Rate':<10}{'Elo':<8}{'Change':<8}"
        )
        logger.info("-" * 84)

        for rank, s in enumerate(standings, 1):
            elo_change_str = f"{s.elo_change:+.0f}"
            logger.info(
                f"{rank:<6}{s.bot_name:<25}{s.wins:<8}{s.losses:<8}"
                f"{s.draws:<8}{s.win_rate:.1%}{'':2}{s.elo:<8.0f}{elo_change_str:<8}"
            )

        logger.info("=" * 84)


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove sensitive data from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Sanitized configuration
    """
    sensitive_patterns = [
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "password",
        "token",
        "credential",
        "auth_key",
        "private_key",
        "access_key",
    ]

    def is_sensitive(field_name: str) -> bool:
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in sensitive_patterns)

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: sanitize(v) for k, v in value.items() if not is_sensitive(k)
            }
        elif isinstance(value, list):
            return [sanitize(item) for item in value]
        else:
            return value

    return sanitize(config)
