"""
Tournament scheduling utilities.

This module provides functions for generating round-robin tournament schedules
and managing map configurations.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .bots import BotDescriptor


@dataclass
class MapConfig:
    """
    Configuration for a tournament map.

    Attributes:
        path: Path to the map file
        max_turns: Maximum turns for games on this map (None = use default)
    """

    path: str
    max_turns: Optional[int] = None

    @classmethod
    def from_config(
        cls, config_entry: Any, default_max_turns: int = 500
    ) -> "MapConfig":
        """
        Create MapConfig from config entry.

        Args:
            config_entry: Either a string path or a dict with path and max_turns
            default_max_turns: Default max turns if not specified

        Returns:
            MapConfig instance
        """
        if isinstance(config_entry, str):
            return cls(path=config_entry, max_turns=default_max_turns)
        elif isinstance(config_entry, dict):
            return cls(
                path=config_entry["path"],
                max_turns=config_entry.get("max_turns", default_max_turns),
            )
        elif isinstance(config_entry, MapConfig):
            return config_entry
        else:
            raise ValueError(f"Invalid map config entry: {config_entry}")

    @property
    def name(self) -> str:
        """Get the map filename without path."""
        return Path(self.path).name

    @property
    def stem(self) -> str:
        """Get the map filename without path or extension."""
        return Path(self.path).stem

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"path": self.path, "max_turns": self.max_turns}

    def __repr__(self) -> str:
        return f"MapConfig(path={self.path!r}, max_turns={self.max_turns})"


@dataclass
class ScheduledGame:
    """
    Represents a scheduled game in the tournament.

    Attributes:
        game_id: Unique game identifier
        bot1: First bot (plays as player 1)
        bot2: Second bot (plays as player 2)
        map_config: Map configuration for this game
        round_index: Which round this game belongs to
        game_index: Index within the round
    """

    game_id: int
    bot1: "BotDescriptor"
    bot2: "BotDescriptor"
    map_config: MapConfig
    round_index: int = 0
    game_index: int = 0

    @property
    def map_name(self) -> str:
        """Get the map filename."""
        return self.map_config.name

    @property
    def max_turns(self) -> Optional[int]:
        """Get max turns for this game."""
        return self.map_config.max_turns

    def __repr__(self) -> str:
        return (
            f"ScheduledGame({self.game_id}: "
            f"{self.bot1.name} vs {self.bot2.name} on {self.map_name})"
        )


@dataclass
class CompletedMatchInfo:
    """
    Information about a completed match (for resume functionality).

    Attributes:
        bot1: Name of first bot
        bot2: Name of second bot
        map_name: Name of the map file
        player1_bot: Which bot was player 1
        winner: Winner (0=draw, 1=p1, 2=p2)
        turns: Number of turns played
    """

    bot1: str
    bot2: str
    map_name: str
    player1_bot: str
    winner: int
    turns: int

    def __repr__(self) -> str:
        return f"CompletedMatch({self.player1_bot} vs opponent on {self.map_name})"


def generate_round_robin_schedule(
    bots: List["BotDescriptor"],
    map_configs: List[MapConfig],
    games_per_side: int = 1,
    map_pool_mode: str = "all",
    completed_matches: Optional[Dict[str, List[CompletedMatchInfo]]] = None,
) -> Tuple[List[List[ScheduledGame]], int]:
    """
    Generate a complete round-robin tournament schedule.

    In a round-robin tournament, every bot plays against every other bot.
    Games are organized by map, and each matchup plays games_per_side games
    with each side taking turns as player 1.

    Args:
        bots: List of bot descriptors
        map_configs: List of map configurations
        games_per_side: Number of games per side per map
        map_pool_mode: How to select maps
            - "all": Play all maps for each matchup
            - "cycle": Cycle through maps
            - "random": Random map selection
        completed_matches: Dict of already completed matches (for resume)
            Key format: "bot1|bot2|map_name" (names sorted alphabetically)

    Returns:
        Tuple of (schedule_by_round, skipped_games)
        - schedule_by_round: List of lists, each inner list is games for one round
        - skipped_games: Number of games skipped (already completed)
    """
    # Generate all matchups (round-robin)
    matchups = []
    for i in range(len(bots)):
        for j in range(i + 1, len(bots)):
            matchups.append((bots[i], bots[j]))

    schedule_by_round: List[List[ScheduledGame]] = []
    game_id = 0
    skipped_games = 0

    if map_pool_mode == "all" and len(map_configs) > 1:
        # Play each map for all matchups before moving to next map
        for round_idx, map_config in enumerate(map_configs):
            round_games = []
            map_name = map_config.name

            for bot1, bot2 in matchups:
                # Check how many games still need to be played
                if completed_matches:
                    bot1_needed, bot2_needed = _get_pending_games(
                        bot1.name,
                        bot2.name,
                        map_name,
                        games_per_side,
                        completed_matches,
                    )
                else:
                    bot1_needed = games_per_side
                    bot2_needed = games_per_side

                # Track skipped games
                skipped_games += (games_per_side - bot1_needed)
                skipped_games += (games_per_side - bot2_needed)

                # Add games with bot1 as player 1
                for _ in range(bot1_needed):
                    round_games.append(
                        ScheduledGame(
                            game_id=game_id,
                            bot1=bot1,
                            bot2=bot2,
                            map_config=map_config,
                            round_index=round_idx,
                            game_index=len(round_games),
                        )
                    )
                    game_id += 1

                # Add games with bot2 as player 1
                for _ in range(bot2_needed):
                    round_games.append(
                        ScheduledGame(
                            game_id=game_id,
                            bot1=bot2,
                            bot2=bot1,
                            map_config=map_config,
                            round_index=round_idx,
                            game_index=len(round_games),
                        )
                    )
                    game_id += 1

            if round_games:
                schedule_by_round.append(round_games)

    else:
        # Cycle or random mode - all games in single round
        round_games = []
        map_idx = 0

        for bot1, bot2 in matchups:
            # Games with bot1 as player 1
            for game_num in range(games_per_side):
                map_config = _select_map(
                    map_configs, map_pool_mode, map_idx + game_num
                )
                round_games.append(
                    ScheduledGame(
                        game_id=game_id,
                        bot1=bot1,
                        bot2=bot2,
                        map_config=map_config,
                        round_index=0,
                        game_index=len(round_games),
                    )
                )
                game_id += 1

            map_idx += games_per_side

            # Games with bot2 as player 1
            for game_num in range(games_per_side):
                map_config = _select_map(
                    map_configs, map_pool_mode, map_idx + game_num
                )
                round_games.append(
                    ScheduledGame(
                        game_id=game_id,
                        bot1=bot2,
                        bot2=bot1,
                        map_config=map_config,
                        round_index=0,
                        game_index=len(round_games),
                    )
                )
                game_id += 1

            map_idx += games_per_side

        if round_games:
            schedule_by_round.append(round_games)

    return schedule_by_round, skipped_games


def _select_map(
    map_configs: List[MapConfig], mode: str, index: int
) -> MapConfig:
    """Select a map based on mode and index."""
    if len(map_configs) == 1:
        return map_configs[0]

    if mode == "cycle":
        return map_configs[index % len(map_configs)]
    elif mode == "random":
        return random.choice(map_configs)
    else:
        return map_configs[0]


def _get_pending_games(
    bot1_name: str,
    bot2_name: str,
    map_name: str,
    games_per_side: int,
    completed_matches: Dict[str, List[CompletedMatchInfo]],
) -> Tuple[int, int]:
    """
    Determine how many games still need to be played for a matchup.

    Args:
        bot1_name: Name of first bot
        bot2_name: Name of second bot
        map_name: Name of the map file
        games_per_side: Target number of games per side
        completed_matches: Dictionary of completed matches

    Returns:
        Tuple of (games_needed_bot1_as_p1, games_needed_bot2_as_p1)
    """
    # Create sorted key for consistent lookup
    sorted_bots = tuple(sorted([bot1_name, bot2_name]))
    key = f"{sorted_bots[0]}|{sorted_bots[1]}|{map_name}"

    matches = completed_matches.get(key, [])

    # Count completed games for each configuration
    bot1_as_p1_count = 0
    bot2_as_p1_count = 0

    for match in matches:
        if match.player1_bot == bot1_name:
            bot1_as_p1_count += 1
        elif match.player1_bot == bot2_name:
            bot2_as_p1_count += 1

    # Calculate remaining games needed
    bot1_needed = max(0, games_per_side - bot1_as_p1_count)
    bot2_needed = max(0, games_per_side - bot2_as_p1_count)

    return bot1_needed, bot2_needed


def calculate_total_games(
    num_bots: int,
    num_maps: int,
    games_per_side: int,
    map_pool_mode: str = "all",
) -> int:
    """
    Calculate total games in a tournament.

    Args:
        num_bots: Number of bots
        num_maps: Number of maps
        games_per_side: Games per side per matchup
        map_pool_mode: Map pool mode

    Returns:
        Total number of games
    """
    # Number of unique matchups in round-robin
    num_matchups = num_bots * (num_bots - 1) // 2

    # Games per matchup depends on mode
    if map_pool_mode == "all" and num_maps > 1:
        games_per_matchup = games_per_side * 2 * num_maps
    else:
        games_per_matchup = games_per_side * 2

    return num_matchups * games_per_matchup
