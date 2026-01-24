"""
Tournament configuration.

This module provides a unified configuration class for tournament settings.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .schedule import MapConfig
from .bots import BotDescriptor, BotType


@dataclass
class TournamentConfig:
    """
    Configuration for a tournament.

    This class holds all settings needed to run a tournament, including
    maps, game settings, and output options.

    Attributes:
        name: Tournament name
        maps: List of map configurations
        games_per_side: Games per side per matchup
        max_turns: Default maximum turns per game
        map_pool_mode: How to select maps ('all', 'cycle', 'random')
        output_dir: Directory for results and replays
        save_replays: Whether to save game replays
        replay_dir: Directory for replays (default: output_dir/replays)
        log_conversations: Enable LLM conversation logging
        conversation_log_dir: Directory for conversation logs
        should_reason: Enable LLM reasoning output
        concurrent_games: Number of concurrent games
        llm_api_delay: Delay between LLM API calls (seconds)
    """

    # Tournament identification
    name: str = "Tournament"

    # Map settings
    maps: List[MapConfig] = field(default_factory=list)
    map_pool_mode: str = "all"

    # Game settings
    games_per_side: int = 2
    max_turns: int = 500

    # Output settings
    output_dir: str = "tournament_results"
    save_replays: bool = True
    replay_dir: Optional[str] = None

    # LLM settings
    log_conversations: bool = False
    conversation_log_dir: Optional[str] = None
    should_reason: bool = False
    llm_api_delay: float = 1.0

    # Execution settings
    concurrent_games: int = 1

    def __post_init__(self):
        """Process configuration after initialization."""
        # Convert string maps to MapConfig
        processed_maps = []
        for m in self.maps:
            if isinstance(m, str):
                processed_maps.append(MapConfig(path=m, max_turns=self.max_turns))
            elif isinstance(m, dict):
                processed_maps.append(MapConfig.from_config(m, self.max_turns))
            elif isinstance(m, MapConfig):
                processed_maps.append(m)
            else:
                raise ValueError(f"Invalid map config: {m}")
        self.maps = processed_maps

        # Set default directories
        if self.replay_dir is None:
            self.replay_dir = str(Path(self.output_dir) / "replays")

        if self.log_conversations and self.conversation_log_dir is None:
            self.conversation_log_dir = str(
                Path(self.output_dir) / "llm_conversations"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TournamentConfig":
        """
        Create TournamentConfig from dictionary.

        Supports both flat config and nested 'tournament' key format.

        Args:
            data: Configuration dictionary

        Returns:
            TournamentConfig instance
        """
        # Handle nested format (docker config style)
        if "tournament" in data:
            tournament_data = data["tournament"]
            output_data = data.get("output", {})
            maps_data = data.get("maps", [])

            return cls(
                name=tournament_data.get("name", "Tournament"),
                maps=[
                    MapConfig.from_config(m, tournament_data.get("max_turns", 500))
                    for m in maps_data
                ],
                map_pool_mode=tournament_data.get("map_pool_mode", "all"),
                games_per_side=tournament_data.get("games_per_matchup", 2),
                max_turns=tournament_data.get("max_turns", 500),
                output_dir=output_data.get("results_dir", "tournament_results"),
                save_replays=tournament_data.get("save_replays", True),
                replay_dir=output_data.get("replay_dir"),
                log_conversations=tournament_data.get("log_conversations", False),
                conversation_log_dir=output_data.get("conversation_log_dir"),
                should_reason=tournament_data.get("should_reason", False),
                llm_api_delay=tournament_data.get("llm_api_delay", 1.0),
                concurrent_games=tournament_data.get("concurrent_games", 1),
            )

        # Handle flat format
        maps = data.get("maps", [])
        if isinstance(maps, str):
            maps = [maps]

        return cls(
            name=data.get("name", "Tournament"),
            maps=maps,
            map_pool_mode=data.get("map_pool_mode", "all"),
            games_per_side=data.get("games_per_side", 2),
            max_turns=data.get("max_turns", 500),
            output_dir=data.get("output_dir", "tournament_results"),
            save_replays=data.get("save_replays", True),
            replay_dir=data.get("replay_dir"),
            log_conversations=data.get("log_conversations", False),
            conversation_log_dir=data.get("conversation_log_dir"),
            should_reason=data.get("should_reason", False),
            llm_api_delay=data.get("llm_api_delay", 1.0),
            concurrent_games=data.get("concurrent_games", 1),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "TournamentConfig":
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to JSON config file

        Returns:
            TournamentConfig instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "maps": [m.to_dict() for m in self.maps],
            "map_pool_mode": self.map_pool_mode,
            "games_per_side": self.games_per_side,
            "max_turns": self.max_turns,
            "output_dir": self.output_dir,
            "save_replays": self.save_replays,
            "replay_dir": self.replay_dir,
            "log_conversations": self.log_conversations,
            "conversation_log_dir": self.conversation_log_dir,
            "should_reason": self.should_reason,
            "llm_api_delay": self.llm_api_delay,
            "concurrent_games": self.concurrent_games,
        }

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.maps:
            errors.append("At least one map is required")

        for m in self.maps:
            if not Path(m.path).exists():
                errors.append(f"Map file not found: {m.path}")

        if self.games_per_side < 1:
            errors.append("games_per_side must be at least 1")

        if self.max_turns < 1:
            errors.append("max_turns must be at least 1")

        if self.map_pool_mode not in ("all", "cycle", "random"):
            errors.append(f"Invalid map_pool_mode: {self.map_pool_mode}")

        if self.concurrent_games < 1:
            errors.append("concurrent_games must be at least 1")

        return errors

    def add_map(
        self, path: str, max_turns: Optional[int] = None
    ) -> "TournamentConfig":
        """
        Add a map to the configuration.

        Args:
            path: Path to map file
            max_turns: Optional max turns override

        Returns:
            self for chaining
        """
        self.maps.append(
            MapConfig(path=path, max_turns=max_turns or self.max_turns)
        )
        return self

    def add_maps_from_directory(
        self, directory: str, max_turns: Optional[int] = None
    ) -> "TournamentConfig":
        """
        Add all CSV maps from a directory.

        Args:
            directory: Directory to scan for maps
            max_turns: Optional max turns override

        Returns:
            self for chaining
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        for map_file in sorted(dir_path.glob("*.csv")):
            self.add_map(str(map_file), max_turns)

        return self


def parse_bots_from_config(data: Dict[str, Any]) -> List[BotDescriptor]:
    """
    Parse bot configurations from config dictionary.

    Args:
        data: Configuration dictionary with 'bots' key

    Returns:
        List of BotDescriptor objects
    """
    bots = []
    bots_data = data.get("bots", [])

    for bot_config in bots_data:
        name = bot_config.get("name", "Bot")
        bot_type = bot_config.get("type", "simple")

        if bot_type in ("simple", "medium", "advanced"):
            bots.append(
                BotDescriptor(
                    name=name,
                    bot_type=BotType(bot_type),
                )
            )
        elif bot_type == "llm":
            bots.append(
                BotDescriptor(
                    name=name,
                    bot_type=BotType.LLM,
                    provider=bot_config.get("provider"),
                    model=bot_config.get("model"),
                    temperature=bot_config.get("temperature"),
                    max_tokens=bot_config.get("max_tokens", 8000),
                )
            )
        elif bot_type == "model":
            bots.append(
                BotDescriptor(
                    name=name,
                    bot_type=BotType.MODEL,
                    model_path=bot_config.get("model_path"),
                )
            )
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")

    return bots
