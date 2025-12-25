#!/usr/bin/env python3
"""
Docker Tournament Runner for Reinforce Tactics.

Reads tournament configuration from config.json and runs a round-robin tournament
between specified bots. Supports built-in bots (simple, medium, advanced) and
LLM bots (OpenAI, Anthropic, Google).

Features:
- Resume interrupted tournaments by passing --resume with a folder path
- Upload replays and conversation logs to Google Cloud Storage

Usage:
    python run_tournament.py [--config CONFIG_PATH] [--resume FOLDER_PATH]
"""
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, '/app')

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot, MediumBot, AdvancedBot
from reinforcetactics.utils.file_io import FileIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GCSUploader:
    """Handles uploading files to Google Cloud Storage."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        credentials_file: Optional[str] = None
    ):
        """
        Initialize GCS uploader.

        Args:
            bucket_name: Name of the GCS bucket
            prefix: Optional prefix/folder path within the bucket
            credentials_file: Optional path to service account credentials JSON
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ""
        self.credentials_file = credentials_file
        self._client = None
        self._bucket = None

    def _get_client(self):
        """Lazily initialize GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                if self.credentials_file and os.path.exists(self.credentials_file):
                    self._client = storage.Client.from_service_account_json(
                        self.credentials_file
                    )
                    logger.info(f"GCS client initialized with credentials from {self.credentials_file}")
                else:
                    # Use default credentials (GOOGLE_APPLICATION_CREDENTIALS env var or metadata server)
                    self._client = storage.Client()
                    logger.info("GCS client initialized with default credentials")
                self._bucket = self._client.bucket(self.bucket_name)
            except ImportError:
                logger.error("google-cloud-storage package not installed. Install with: pip install google-cloud-storage")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}")
                raise
        return self._client

    def upload_file(self, local_path: str, remote_path: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to GCS.

        Args:
            local_path: Local file path to upload
            remote_path: Remote path within the bucket (uses filename if not specified)

        Returns:
            GCS URI (gs://bucket/path) if successful, None otherwise
        """
        try:
            self._get_client()

            if remote_path is None:
                remote_path = os.path.basename(local_path)

            # Add prefix
            full_remote_path = f"{self.prefix}{remote_path}"

            blob = self._bucket.blob(full_remote_path)
            blob.upload_from_filename(local_path)

            gcs_uri = f"gs://{self.bucket_name}/{full_remote_path}"
            logger.debug(f"Uploaded {local_path} to {gcs_uri}")
            return gcs_uri

        except Exception as e:
            logger.warning(f"Failed to upload {local_path} to GCS: {e}")
            return None

    def upload_directory(self, local_dir: str, remote_prefix: Optional[str] = None) -> int:
        """
        Upload all files in a directory to GCS.

        Args:
            local_dir: Local directory path
            remote_prefix: Remote prefix for all files

        Returns:
            Number of files successfully uploaded
        """
        uploaded = 0
        local_path = Path(local_dir)

        if not local_path.exists():
            logger.warning(f"Directory does not exist: {local_dir}")
            return 0

        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                if remote_prefix:
                    remote_path = f"{remote_prefix}/{relative_path}"
                else:
                    remote_path = str(relative_path)

                if self.upload_file(str(file_path), remote_path):
                    uploaded += 1

        logger.info(f"Uploaded {uploaded} files from {local_dir} to GCS")
        return uploaded


class CompletedMatchInfo:
    """Information about a completed match extracted from replay files."""

    def __init__(
        self,
        bot1: str,
        bot2: str,
        map_name: str,
        player1_bot: str,
        winner: int,
        turns: int
    ):
        self.bot1 = bot1
        self.bot2 = bot2
        self.map_name = map_name
        self.player1_bot = player1_bot  # Which bot was player 1
        self.winner = winner
        self.turns = turns

    def __repr__(self):
        return f"CompletedMatch({self.player1_bot} vs opponent on {self.map_name})"


def scan_completed_matches(resume_folder: str) -> Dict[str, List[CompletedMatchInfo]]:
    """
    Scan a folder for completed match replay files.

    Args:
        resume_folder: Path to folder containing replays or tournament output

    Returns:
        Dictionary mapping matchup keys to list of completed matches
        Key format: "bot1_name|bot2_name|map_name" (names sorted alphabetically)
    """
    completed = defaultdict(list)
    resume_path = Path(resume_folder)

    if not resume_path.exists():
        logger.warning(f"Resume folder does not exist: {resume_folder}")
        return completed

    # Look for replay files in multiple possible locations
    search_paths = [
        resume_path,  # Direct folder
        resume_path / 'replays',  # replays subfolder
        resume_path / 'output' / 'replays',  # output/replays subfolder
    ]

    replay_files = []
    for search_path in search_paths:
        if search_path.exists():
            replay_files.extend(search_path.rglob('game_*.json'))

    logger.info(f"Found {len(replay_files)} replay files to scan")

    for replay_file in replay_files:
        try:
            with open(replay_file, 'r') as f:
                data = json.load(f)

            game_info = data.get('game_info', {})
            bot1 = game_info.get('bot1', '')
            bot2 = game_info.get('bot2', '')
            winner = game_info.get('winner', 0)
            turns = game_info.get('turns', 0)
            map_path = game_info.get('map', '')
            bot1_player = game_info.get('bot1_player', 1)

            # Extract map name from path
            map_name = Path(map_path).name if map_path else ''

            if not bot1 or not bot2 or not map_name:
                logger.warning(f"Incomplete game info in {replay_file}")
                continue

            # Determine which bot was player 1
            if bot1_player == 1:
                player1_bot = bot1
            else:
                player1_bot = bot2

            # Create sorted key for consistent lookup
            sorted_bots = tuple(sorted([bot1, bot2]))
            key = f"{sorted_bots[0]}|{sorted_bots[1]}|{map_name}"

            match_info = CompletedMatchInfo(
                bot1=bot1,
                bot2=bot2,
                map_name=map_name,
                player1_bot=player1_bot,
                winner=winner,
                turns=turns
            )
            completed[key].append(match_info)

        except Exception as e:
            logger.warning(f"Error reading replay file {replay_file}: {e}")
            continue

    # Log summary
    total_matches = sum(len(matches) for matches in completed.values())
    logger.info(f"Found {total_matches} completed matches across {len(completed)} matchup configurations")

    return completed


def get_pending_games(
    bot1_name: str,
    bot2_name: str,
    map_name: str,
    games_per_side: int,
    completed_matches: Dict[str, List[CompletedMatchInfo]]
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
    bot1_as_p1_needed = max(0, games_per_side - bot1_as_p1_count)
    bot2_as_p1_needed = max(0, games_per_side - bot2_as_p1_count)

    return bot1_as_p1_needed, bot2_as_p1_needed


def load_previous_results(resume_folder: str) -> Optional[Dict[str, Any]]:
    """
    Load previous tournament results from a resume folder.

    Args:
        resume_folder: Path to folder containing previous results

    Returns:
        Previous results dictionary if found, None otherwise
    """
    resume_path = Path(resume_folder)

    # Look for results files in multiple possible locations
    search_paths = [
        resume_path,
        resume_path / 'results',
        resume_path / 'output' / 'results',
    ]

    results_files = []
    for search_path in search_paths:
        if search_path.exists():
            results_files.extend(search_path.glob('tournament_results_*.json'))

    if not results_files:
        logger.info("No previous results found to merge")
        return None

    # Get the most recent results file
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_results = results_files[0]

    try:
        with open(latest_results, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded previous results from {latest_results}")
        return data
    except Exception as e:
        logger.warning(f"Error loading previous results: {e}")
        return None


class EloRatingSystem:
    """Manages Elo ratings for tournament participants."""

    def __init__(self, starting_elo: int = 1500, k_factor: int = 32):
        self.starting_elo = starting_elo
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.initial_ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[float]] = {}

    def initialize_bot(self, bot_name: str) -> None:
        if bot_name not in self.ratings:
            self.ratings[bot_name] = float(self.starting_elo)
            self.initial_ratings[bot_name] = float(self.starting_elo)
            self.rating_history[bot_name] = [float(self.starting_elo)]

    def calculate_expected_score(self, player_elo: float, opponent_elo: float) -> float:
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))

    def update_ratings(self, bot1_name: str, bot2_name: str, result: int) -> None:
        self.initialize_bot(bot1_name)
        self.initialize_bot(bot2_name)

        bot1_elo = self.ratings[bot1_name]
        bot2_elo = self.ratings[bot2_name]

        bot1_expected = self.calculate_expected_score(bot1_elo, bot2_elo)
        bot2_expected = self.calculate_expected_score(bot2_elo, bot1_elo)

        if result == 1:
            bot1_actual, bot2_actual = 1.0, 0.0
        elif result == 2:
            bot1_actual, bot2_actual = 0.0, 1.0
        else:
            bot1_actual, bot2_actual = 0.5, 0.5

        bot1_new = bot1_elo + self.k_factor * (bot1_actual - bot1_expected)
        bot2_new = bot2_elo + self.k_factor * (bot2_actual - bot2_expected)

        self.ratings[bot1_name] = bot1_new
        self.ratings[bot2_name] = bot2_new

        self.rating_history[bot1_name].append(bot1_new)
        self.rating_history[bot2_name].append(bot2_new)

    def get_rating(self, bot_name: str) -> float:
        return self.ratings.get(bot_name, float(self.starting_elo))

    def get_rating_change(self, bot_name: str) -> float:
        initial = self.initial_ratings.get(bot_name, float(self.starting_elo))
        current = self.ratings.get(bot_name, float(self.starting_elo))
        return current - initial


class TournamentBot:
    """Configuration for a bot participating in a tournament."""

    def __init__(self, name: str, bot_class: Union[str, type], model: Optional[str] = None,
                 temperature: Optional[float] = None, max_tokens: int = 8000):
        self.name = name
        self.bot_class = bot_class
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class MapConfig:
    """Configuration for a map with optional per-map settings."""

    def __init__(self, path: str, max_turns: Optional[int] = None):
        self.path = path
        self.max_turns = max_turns

    @classmethod
    def from_config(cls, config_entry: Union[str, Dict[str, Any]], default_max_turns: int) -> 'MapConfig':
        """Create MapConfig from config entry (string or dict)."""
        if isinstance(config_entry, str):
            return cls(path=config_entry, max_turns=default_max_turns)
        else:
            path = config_entry['path']
            max_turns = config_entry.get('max_turns', default_max_turns)
            return cls(path=path, max_turns=max_turns)

    def __repr__(self):
        return f"MapConfig(path={self.path}, max_turns={self.max_turns})"


@dataclass
class ScheduledGame:
    """Represents a scheduled game in the tournament."""
    game_id: int
    p1_bot: 'TournamentBot'
    p2_bot: 'TournamentBot'
    map_config: MapConfig
    round_index: int  # Which round (map) this game belongs to
    game_index_in_round: int  # Index within the round

    def __repr__(self):
        return f"Game({self.game_id}: {self.p1_bot.name} vs {self.p2_bot.name} on {Path(self.map_config.path).name})"


@dataclass
class GameResult:
    """Result of a completed game."""
    game_id: int
    p1_bot_name: str
    p2_bot_name: str
    winner: int  # 0=draw, 1=p1 wins, 2=p2 wins
    winner_name: str
    turns: int
    map_name: str
    replay_path: Optional[str] = None
    error: Optional[str] = None


def generate_round_robin_schedule(
    bots: List['TournamentBot'],
    map_configs: List[MapConfig],
    games_per_side: int,
    map_pool_mode: str = 'all',
    completed_matches: Optional[Dict[str, List[CompletedMatchInfo]]] = None
) -> Tuple[List[List[ScheduledGame]], int]:
    """
    Generate the complete round-robin schedule upfront, organized by map.

    Returns:
        Tuple of (schedule_by_map, skipped_games)
        schedule_by_map: List of lists, where each inner list contains games for one map
    """
    # Generate all matchups (round-robin)
    matchups = []
    for i in range(len(bots)):
        for j in range(i + 1, len(bots)):
            matchups.append((bots[i], bots[j]))

    schedule_by_map: List[List[ScheduledGame]] = []
    game_id = 0
    skipped_games = 0

    if map_pool_mode == 'all' and len(map_configs) > 1:
        # For 'all' mode, play each map for all matchups before moving to next map
        for round_idx, map_config in enumerate(map_configs):
            round_games = []
            map_name = Path(map_config.path).name

            for bot1, bot2 in matchups:
                # Check how many games still need to be played for this map
                if completed_matches:
                    bot1_as_p1_needed, bot2_as_p1_needed = get_pending_games(
                        bot1.name, bot2.name, map_name,
                        games_per_side, completed_matches
                    )
                else:
                    bot1_as_p1_needed = games_per_side
                    bot2_as_p1_needed = games_per_side

                # Track skipped games
                skipped_games += (games_per_side - bot1_as_p1_needed)
                skipped_games += (games_per_side - bot2_as_p1_needed)

                # Add remaining games with bot1 as player 1
                for game_in_round in range(bot1_as_p1_needed):
                    round_games.append(ScheduledGame(
                        game_id=game_id,
                        p1_bot=bot1,
                        p2_bot=bot2,
                        map_config=map_config,
                        round_index=round_idx,
                        game_index_in_round=len(round_games)
                    ))
                    game_id += 1

                # Add remaining games with bot2 as player 1
                for game_in_round in range(bot2_as_p1_needed):
                    round_games.append(ScheduledGame(
                        game_id=game_id,
                        p1_bot=bot2,
                        p2_bot=bot1,
                        map_config=map_config,
                        round_index=round_idx,
                        game_index_in_round=len(round_games)
                    ))
                    game_id += 1

            if round_games:
                schedule_by_map.append(round_games)
    else:
        # For 'cycle' or 'random' mode, or single map
        # Group all games into a single round
        round_games = []
        map_idx = 0

        for bot1, bot2 in matchups:
            for _ in range(games_per_side):
                if map_pool_mode == 'cycle':
                    map_config = map_configs[map_idx % len(map_configs)]
                    map_idx += 1
                elif map_pool_mode == 'random':
                    map_config = random.choice(map_configs)
                else:
                    map_config = map_configs[0]

                round_games.append(ScheduledGame(
                    game_id=game_id,
                    p1_bot=bot1,
                    p2_bot=bot2,
                    map_config=map_config,
                    round_index=0,
                    game_index_in_round=len(round_games)
                ))
                game_id += 1

            for _ in range(games_per_side):
                if map_pool_mode == 'cycle':
                    map_config = map_configs[map_idx % len(map_configs)]
                    map_idx += 1
                elif map_pool_mode == 'random':
                    map_config = random.choice(map_configs)
                else:
                    map_config = map_configs[0]

                round_games.append(ScheduledGame(
                    game_id=game_id,
                    p1_bot=bot2,
                    p2_bot=bot1,
                    map_config=map_config,
                    round_index=0,
                    game_index_in_round=len(round_games)
                ))
                game_id += 1

        if round_games:
            schedule_by_map.append(round_games)

    return schedule_by_map, skipped_games


def parse_maps_from_config(config: Dict[str, Any]) -> List[MapConfig]:
    """Parse maps from config, supporting both string and object format."""
    default_max_turns = config['tournament'].get('max_turns', 100)
    map_configs = []

    for entry in config['maps']:
        map_config = MapConfig.from_config(entry, default_max_turns)
        map_configs.append(map_config)

    return map_configs


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate tournament configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Validate required fields
    if 'tournament' not in config:
        raise ValueError("Config missing 'tournament' section")
    if 'maps' not in config or not config['maps']:
        raise ValueError("Config missing 'maps' or maps list is empty")
    if 'bots' not in config or len(config['bots']) < 2:
        raise ValueError("Config missing 'bots' or needs at least 2 bots")

    return config


def create_bots_from_config(config: Dict[str, Any]) -> List[TournamentBot]:
    """Create TournamentBot instances from config."""
    bots = []

    for bot_config in config['bots']:
        name = bot_config['name']
        bot_type = bot_config['type']

        if bot_type in ('simple', 'medium', 'advanced'):
            bots.append(TournamentBot(name, bot_type))
        elif bot_type == 'llm':
            provider = bot_config.get('provider')
            model = bot_config.get('model')
            temperature = bot_config.get('temperature')
            max_tokens = bot_config.get('max_tokens', 8000)

            # Get the appropriate bot class
            if provider == 'openai':
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    logger.warning(f"Skipping {name}: OPENAI_API_KEY not set")
                    continue
                try:
                    from reinforcetactics.game.llm_bot import OpenAIBot
                    bots.append(TournamentBot(name, OpenAIBot, model, temperature, max_tokens))
                except ImportError:
                    logger.warning(f"Skipping {name}: openai package not installed")
            elif provider == 'anthropic':
                api_key = os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.warning(f"Skipping {name}: ANTHROPIC_API_KEY not set")
                    continue
                try:
                    from reinforcetactics.game.llm_bot import ClaudeBot
                    bots.append(TournamentBot(name, ClaudeBot, model, temperature, max_tokens))
                except ImportError:
                    logger.warning(f"Skipping {name}: anthropic package not installed")
            elif provider == 'google':
                api_key = os.environ.get('GOOGLE_API_KEY')
                if not api_key:
                    logger.warning(f"Skipping {name}: GOOGLE_API_KEY not set")
                    continue
                try:
                    from reinforcetactics.game.llm_bot import GeminiBot
                    bots.append(TournamentBot(name, GeminiBot, model, temperature, max_tokens))
                except ImportError:
                    logger.warning(f"Skipping {name}: google-genai package not installed")
            else:
                logger.warning(f"Unknown LLM provider: {provider}")
        elif bot_type == 'model':
            model_path = bot_config.get('model_path')
            if model_path and Path(model_path).exists():
                bots.append(TournamentBot(name, 'model', model_path=model_path))
            else:
                logger.warning(f"Skipping {name}: model file not found at {model_path}")

    return bots


def run_tournament(
    bots: List[TournamentBot],
    map_file: str = 'maps/1v1/beginner.csv',
    maps: Optional[List[MapConfig]] = None,
    map_pool_mode: str = 'all',
    games_per_matchup: int = 2,
    max_turns: int = 500,
    should_reason: bool = False,
    log_conversations: bool = False,
    conversation_log_dir: Optional[str] = None,
    save_replays: bool = False,
    replay_dir: Optional[str] = None,
    completed_matches: Optional[Dict[str, List[CompletedMatchInfo]]] = None,
    gcs_uploader: Optional[GCSUploader] = None,
    concurrent_games: int = 1,
    llm_api_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Run a round-robin tournament between multiple bots with Elo ratings.

    The tournament schedule is determined upfront, with games organized by map.
    For each map, all matchups are executed in order. Games can run concurrently
    with configurable parallelism.

    Args:
        bots: List of TournamentBot configurations
        map_file: Default map file (used if maps is None)
        maps: List of MapConfig objects with per-map settings (including max_turns)
        map_pool_mode: How to select maps - 'all', 'cycle', or 'random'
        games_per_matchup: Games per side per map
        max_turns: Default max turns (used when MapConfig has no max_turns)
        should_reason: Enable LLM reasoning output
        log_conversations: Save LLM conversation logs
        conversation_log_dir: Directory for conversation logs
        save_replays: Save game replays
        replay_dir: Directory for replays
        completed_matches: Dictionary of already completed matches (for resume)
        gcs_uploader: Optional GCS uploader for cloud storage
        concurrent_games: Number of games to run concurrently (default 1 = sequential)
        llm_api_delay: Delay in seconds between LLM API calls to avoid rate limits
    """
    if len(bots) < 2:
        raise ValueError("Need at least 2 bots for a tournament")

    # Handle map list - convert to MapConfig if needed
    if maps:
        map_list = maps
    else:
        map_list = [MapConfig(path=map_file, max_turns=max_turns)]

    # Validate maps exist
    for m in map_list:
        if not Path(m.path).exists():
            raise FileNotFoundError(f"Map file not found: {m.path}")

    # Initialize Elo rating system
    elo_system = EloRatingSystem()
    for bot in bots:
        elo_system.initialize_bot(bot.name)

    # Create output directories
    abs_log_dir = None
    if log_conversations and conversation_log_dir:
        abs_log_dir = os.path.abspath(conversation_log_dir)
        os.makedirs(abs_log_dir, exist_ok=True)
        logger.info(f"Conversation log directory: {abs_log_dir}")

    abs_replay_dir = None
    if save_replays:
        if replay_dir is None:
            replay_dir = 'replays'
        abs_replay_dir = os.path.abspath(replay_dir)
        os.makedirs(abs_replay_dir, exist_ok=True)
        logger.info(f"Replay directory: {abs_replay_dir}")

    # Print tournament info
    logger.info("=" * 70)
    logger.info("TOURNAMENT START")
    logger.info("=" * 70)
    if len(map_list) == 1:
        m = map_list[0]
        logger.info(f"Map: {m.path} (max_turns: {m.max_turns})")
    else:
        logger.info(f"Maps: {len(map_list)} maps")
        for m in map_list:
            logger.info(f"  - {m.path} (max_turns: {m.max_turns})")
        logger.info(f"Map Pool Mode: {map_pool_mode}")
    logger.info(f"Participants: {len(bots)}")

    for bot in bots:
        model_str = f" ({bot.model})" if bot.model else ""
        temp_str = f" [temp={bot.temperature}]" if bot.temperature is not None else ""
        if bot.bot_class == 'simple':
            bot_type_str = "SimpleBot"
        elif bot.bot_class == 'medium':
            bot_type_str = "MediumBot"
        elif bot.bot_class == 'advanced':
            bot_type_str = "AdvancedBot"
        else:
            bot_type_str = bot.bot_class.__name__
        logger.info(f"  - {bot.name}: {bot_type_str}{model_str}{temp_str}")

    # Calculate total games
    if map_pool_mode == 'all' and len(map_list) > 1:
        games_per_matchup_total = games_per_matchup * 2 * len(map_list)
    else:
        games_per_matchup_total = games_per_matchup * 2

    logger.info(f"Games per matchup: {games_per_matchup_total}")
    if should_reason:
        logger.info("LLM Reasoning: ENABLED")
    if log_conversations:
        logger.info("LLM Conversation Logging: ENABLED")
    if save_replays:
        logger.info("Replay Saving: ENABLED")
    if completed_matches:
        total_completed = sum(len(m) for m in completed_matches.values())
        logger.info(f"Resume Mode: ENABLED ({total_completed} completed matches found)")
    if gcs_uploader:
        logger.info(f"GCS Upload: ENABLED (bucket: {gcs_uploader.bucket_name})")
    logger.info("=" * 70)

    # Generate complete schedule upfront
    logger.info("\nGenerating tournament schedule...")
    schedule_by_map, skipped_games = generate_round_robin_schedule(
        bots=bots,
        map_configs=map_list,
        games_per_side=games_per_matchup,
        map_pool_mode=map_pool_mode,
        completed_matches=completed_matches
    )

    total_games = sum(len(round_games) for round_games in schedule_by_map)
    num_rounds = len(schedule_by_map)

    logger.info(f"Schedule generated: {num_rounds} rounds, {total_games} games total")
    if skipped_games > 0:
        logger.info(f"  (Skipping {skipped_games} already completed games)")

    # Log schedule summary per round
    for round_idx, round_games in enumerate(schedule_by_map):
        if round_games:
            map_name = Path(round_games[0].map_config.path).name
            logger.info(f"  Round {round_idx + 1}: {len(round_games)} games on {map_name}")

    # Determine effective concurrency
    effective_concurrent = concurrent_games
    logger.info(f"Concurrent games: {effective_concurrent}")
    if llm_api_delay > 0:
        logger.info(f"LLM API delay: {llm_api_delay}s")
    logger.info("=" * 70)

    # Initialize results tracking
    results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
    per_map_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0}))
    matchup_details_map = defaultdict(lambda: defaultdict(lambda: {'bot1_wins': 0, 'bot2_wins': 0, 'draws': 0}))
    new_games_played = 0

    known_llm_bots = ['OpenAIBot', 'ClaudeBot', 'GeminiBot']

    # Lock for thread-safe result updates
    results_lock = Lock()

    def check_is_llm_bot(bot: TournamentBot) -> bool:
        """Check if a bot is an LLM bot."""
        return not isinstance(bot.bot_class, str) and bot.bot_class.__name__ in known_llm_bots

    def execute_game(scheduled_game: ScheduledGame) -> GameResult:
        """Execute a single scheduled game and return the result."""
        p1_bot = scheduled_game.p1_bot
        p2_bot = scheduled_game.p2_bot
        map_config = scheduled_game.map_config
        map_path = map_config.path
        map_name = Path(map_path).name

        # Set up conversation logging directory for this matchup
        matchup_log_dir = None
        if log_conversations and abs_log_dir:
            matchup_log_dir = os.path.join(
                abs_log_dir,
                map_name.replace('.csv', ''),
                f"{p1_bot.name}_vs_{p2_bot.name}"
            )
            os.makedirs(matchup_log_dir, exist_ok=True)

        # Set up replay directory for this matchup
        matchup_replay_dir = None
        if save_replays and abs_replay_dir:
            matchup_replay_dir = os.path.join(
                abs_replay_dir,
                map_name.replace('.csv', ''),
                f"{p1_bot.name}_vs_{p2_bot.name}"
            )
            os.makedirs(matchup_replay_dir, exist_ok=True)

        # Check if this game involves LLM bots (for delay)
        has_llm = check_is_llm_bot(p1_bot) or check_is_llm_bot(p2_bot)

        # Run the game
        result = run_single_game(
            p1_bot, p2_bot, 1, 2, map_path,
            max_turns=map_config.max_turns,
            should_reason=should_reason,
            log_conversations=log_conversations,
            conversation_log_dir=matchup_log_dir,
            save_replay=save_replays,
            replay_dir=matchup_replay_dir
        )

        # Apply delay for LLM games to avoid rate limits
        if has_llm and llm_api_delay > 0:
            time.sleep(llm_api_delay)

        return GameResult(
            game_id=scheduled_game.game_id,
            p1_bot_name=p1_bot.name,
            p2_bot_name=p2_bot.name,
            winner=result['winner'],
            winner_name=result['winner_name'],
            turns=result['turns'],
            map_name=map_name,
            replay_path=result.get('replay_path'),
            error=result.get('error')
        )

    def process_game_result(game_result: GameResult, scheduled_game: ScheduledGame):
        """Process a completed game result and update statistics."""
        nonlocal new_games_played

        p1_bot = scheduled_game.p1_bot
        p2_bot = scheduled_game.p2_bot
        map_name = game_result.map_name

        with results_lock:
            new_games_played += 1

            # Create sorted key for matchup tracking
            sorted_names = tuple(sorted([p1_bot.name, p2_bot.name]))
            matchup_key = f"{sorted_names[0]}|{sorted_names[1]}"

            # Update statistics based on winner
            if game_result.winner == 1:  # P1 wins
                results[p1_bot.name]['wins'] += 1
                results[p2_bot.name]['losses'] += 1
                per_map_stats[p1_bot.name][map_name]['wins'] += 1
                per_map_stats[p2_bot.name][map_name]['losses'] += 1
                if p1_bot.name == sorted_names[0]:
                    matchup_details_map[matchup_key][map_name]['bot1_wins'] += 1
                else:
                    matchup_details_map[matchup_key][map_name]['bot2_wins'] += 1
                elo_system.update_ratings(p1_bot.name, p2_bot.name, 1)
            elif game_result.winner == 2:  # P2 wins
                results[p2_bot.name]['wins'] += 1
                results[p1_bot.name]['losses'] += 1
                per_map_stats[p2_bot.name][map_name]['wins'] += 1
                per_map_stats[p1_bot.name][map_name]['losses'] += 1
                if p2_bot.name == sorted_names[0]:
                    matchup_details_map[matchup_key][map_name]['bot1_wins'] += 1
                else:
                    matchup_details_map[matchup_key][map_name]['bot2_wins'] += 1
                elo_system.update_ratings(p2_bot.name, p1_bot.name, 1)
            else:  # Draw
                results[p1_bot.name]['draws'] += 1
                results[p2_bot.name]['draws'] += 1
                per_map_stats[p1_bot.name][map_name]['draws'] += 1
                per_map_stats[p2_bot.name][map_name]['draws'] += 1
                matchup_details_map[matchup_key][map_name]['draws'] += 1
                elo_system.update_ratings(p1_bot.name, p2_bot.name, 0)

            # Upload replay to GCS if configured
            if gcs_uploader and game_result.replay_path:
                gcs_replay_path = f"replays/{map_name.replace('.csv', '')}/{p1_bot.name}_vs_{p2_bot.name}/{os.path.basename(game_result.replay_path)}"
                gcs_uploader.upload_file(game_result.replay_path, gcs_replay_path)

    # Execute games round by round (map by map)
    for round_idx, round_games in enumerate(schedule_by_map):
        if not round_games:
            continue

        map_name = Path(round_games[0].map_config.path).name
        logger.info(f"\n{'='*70}")
        logger.info(f"Round {round_idx + 1}/{num_rounds}: {map_name}")
        logger.info(f"Games in round: {len(round_games)}")
        logger.info("=" * 70)

        # Determine concurrency for this round (capped by round size)
        round_concurrency = min(effective_concurrent, len(round_games))

        if round_concurrency == 1:
            # Sequential execution
            for game_idx, scheduled_game in enumerate(round_games):
                logger.info(f"  Game {game_idx + 1}/{len(round_games)}: "
                           f"{scheduled_game.p1_bot.name} (P1) vs {scheduled_game.p2_bot.name} (P2)")
                game_result = execute_game(scheduled_game)
                logger.info(f"    Result: {game_result.winner_name} (turns: {game_result.turns})")
                process_game_result(game_result, scheduled_game)
        else:
            # Concurrent execution
            logger.info(f"  Running {len(round_games)} games with {round_concurrency} concurrent workers")

            # Create a queue of games to execute
            game_queue = list(enumerate(round_games))
            completed_count = 0

            with ThreadPoolExecutor(max_workers=round_concurrency) as executor:
                # Submit initial batch of games
                future_to_game = {}
                for game_idx, scheduled_game in game_queue:
                    future = executor.submit(execute_game, scheduled_game)
                    future_to_game[future] = (game_idx, scheduled_game)

                # Process results as they complete
                for future in as_completed(future_to_game):
                    game_idx, scheduled_game = future_to_game[future]
                    try:
                        game_result = future.result()
                        completed_count += 1
                        logger.info(f"  [{completed_count}/{len(round_games)}] "
                                   f"{scheduled_game.p1_bot.name} vs {scheduled_game.p2_bot.name}: "
                                   f"{game_result.winner_name} (turns: {game_result.turns})")
                        process_game_result(game_result, scheduled_game)
                    except Exception as e:
                        completed_count += 1
                        logger.error(f"  [{completed_count}/{len(round_games)}] "
                                    f"{scheduled_game.p1_bot.name} vs {scheduled_game.p2_bot.name}: "
                                    f"Error - {e}")

        logger.info(f"  Round {round_idx + 1} complete")

    # Aggregate matchup details from the map
    matchup_details = []
    seen_matchups = set()
    for matchup_key, map_results in matchup_details_map.items():
        if matchup_key in seen_matchups:
            continue
        seen_matchups.add(matchup_key)

        bot1_name, bot2_name = matchup_key.split('|')
        total_bot1_wins = sum(r['bot1_wins'] for r in map_results.values())
        total_bot2_wins = sum(r['bot2_wins'] for r in map_results.values())
        total_draws = sum(r['draws'] for r in map_results.values())

        matchup_details.append({
            'bot1': bot1_name,
            'bot2': bot2_name,
            'bot1_wins': total_bot1_wins,
            'bot2_wins': total_bot2_wins,
            'draws': total_draws
        })

    # Generate final results
    standings = []
    for bot_name, stats in results.items():
        total_games = stats['wins'] + stats['losses'] + stats['draws']
        win_rate = stats['wins'] / total_games if total_games > 0 else 0.0

        bot_per_map = {}
        if bot_name in per_map_stats:
            bot_per_map = {
                map_name: dict(map_stats)
                for map_name, map_stats in per_map_stats[bot_name].items()
            }

        standings.append({
            'bot': bot_name,
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'total_games': total_games,
            'win_rate': win_rate,
            'elo': round(elo_system.get_rating(bot_name), 0),
            'elo_change': round(elo_system.get_rating_change(bot_name), 0),
            'per_map_stats': bot_per_map
        })

    standings.sort(key=lambda x: x['elo'], reverse=True)

    # Print resume statistics if applicable
    if completed_matches:
        logger.info("\n" + "-" * 70)
        logger.info("RESUME STATISTICS")
        logger.info("-" * 70)
        logger.info(f"Games skipped (already completed): {skipped_games}")
        logger.info(f"New games played this session: {new_games_played}")
        logger.info("-" * 70)

    # Print final standings
    logger.info("\n" + "=" * 84)
    logger.info("TOURNAMENT RESULTS")
    logger.info("=" * 84)
    logger.info(f"{'Rank':<6}{'Bot':<25}{'Wins':<8}{'Losses':<8}{'Draws':<8}"
                f"{'Win Rate':<10}{'Elo':<8}{'Change':<8}")
    logger.info("-" * 84)
    for rank, s in enumerate(standings, 1):
        elo_change_str = f"{s['elo_change']:+.0f}"
        logger.info(f"{rank:<6}{s['bot']:<25}{s['wins']:<8}{s['losses']:<8}"
                   f"{s['draws']:<8}{s['win_rate']:.1%}{'':2}{s['elo']:<8.0f}{elo_change_str:<8}")
    logger.info("=" * 84)

    # Convert MapConfig to serializable format
    maps_info = [{'path': m.path, 'max_turns': m.max_turns} for m in map_list]

    result_data = {
        'timestamp': datetime.now().isoformat(),
        'maps_used': maps_info,
        'map_pool_mode': map_pool_mode,
        'games_per_matchup': games_per_matchup,
        'standings': standings,
        'matchups': matchup_details,
        'elo_history': {
            bot_name: [round(r, 0) for r in history]
            for bot_name, history in elo_system.rating_history.items()
        }
    }

    # Add resume statistics if applicable
    if completed_matches:
        result_data['resume_stats'] = {
            'skipped_games': skipped_games,
            'new_games_played': new_games_played,
            'resumed_from_existing': True
        }

    return result_data


def run_single_game(
    bot1: TournamentBot,
    bot2: TournamentBot,
    player1: int,
    player2: int,
    map_file: str,
    max_turns: int = 500,
    should_reason: bool = False,
    log_conversations: bool = False,
    conversation_log_dir: Optional[str] = None,
    save_replay: bool = False,
    replay_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run a single game between two bots."""

    # Load map and create game state
    map_data = FileIO.load_map(map_file)
    game_state = GameState(map_data, num_players=2)

    # Create unique session ID for conversation logging
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{bot1.name}_vs_{bot2.name}"

    # Create bot instances
    def create_bot_instance(bot: TournamentBot, player: int):
        if bot.bot_class == 'simple':
            return SimpleBot(game_state, player)
        elif bot.bot_class == 'medium':
            return MediumBot(game_state, player)
        elif bot.bot_class == 'advanced':
            return AdvancedBot(game_state, player)
        else:
            # LLM bot
            kwargs = {
                'game_state': game_state,
                'player': player,
                'model': bot.model,
                'max_tokens': bot.max_tokens,
                'should_reason': should_reason,
                'log_conversations': log_conversations,
                'conversation_log_dir': conversation_log_dir,
                'game_session_id': session_id
            }
            if bot.temperature is not None:
                kwargs['temperature'] = bot.temperature
            return bot.bot_class(**kwargs)

    bot1_instance = create_bot_instance(bot1, player1)
    bot2_instance = create_bot_instance(bot2, player2)
    bots = {player1: bot1_instance, player2: bot2_instance}

    # Play the game
    turn_count = 0
    try:
        while not game_state.game_over and turn_count < max_turns:
            current_player = game_state.current_player
            current_bot = bots[current_player]
            current_bot.take_turn()
            turn_count += 1

        # Determine winner
        if game_state.game_over and game_state.winner:
            winner = game_state.winner
            if winner == player1:
                winner_name = bot1.name
            else:
                winner_name = bot2.name
        else:
            winner = 0
            winner_name = "Draw"

        # Save replay if requested
        replay_path = None
        if save_replay and replay_dir:
            map_basename = Path(map_file).stem
            replay_filename = (
                f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{bot1.name}_vs_{bot2.name}_{map_basename}.json"
            )
            replay_path = os.path.join(replay_dir, replay_filename)

            game_info = {
                'bot1': bot1.name,
                'bot2': bot2.name,
                'bot1_player': player1,
                'bot2_player': player2,
                'winner': winner,
                'winner_name': winner_name,
                'turns': turn_count,
                'map': map_file
            }

            FileIO.save_replay(
                game_state.action_history,
                game_info,
                replay_path
            )

        return {
            'winner': winner,
            'winner_name': winner_name,
            'turns': turn_count,
            'replay_path': replay_path,
            'conversation_log_dir': conversation_log_dir
        }

    except Exception as e:
        logger.error(f"Error during game: {e}", exc_info=True)
        return {
            'winner': 0,
            'winner_name': "Error",
            'turns': turn_count,
            'error': str(e)
        }


def sanitize_config_for_saving(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a sanitized copy of the configuration with sensitive data removed.

    Removes or masks fields that may contain API keys, secrets, or other sensitive data.
    """
    # Fields that should be completely removed (case-insensitive matching)
    sensitive_field_patterns = [
        'api_key', 'apikey', 'api-key',
        'secret', 'password', 'token',
        'credential', 'auth_key', 'authkey',
        'private_key', 'privatekey', 'access_key', 'accesskey'
    ]

    def is_sensitive_field(field_name: str) -> bool:
        """Check if a field name indicates sensitive data."""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in sensitive_field_patterns)

    def sanitize_value(value: Any) -> Any:
        """Recursively sanitize a value."""
        if isinstance(value, dict):
            return sanitize_dict(value)
        elif isinstance(value, list):
            return [sanitize_value(item) for item in value]
        else:
            return value

    def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize a dictionary by removing sensitive fields."""
        result = {}
        for key, value in d.items():
            if is_sensitive_field(key):
                # Skip sensitive fields entirely
                continue
            result[key] = sanitize_value(value)
        return result

    return sanitize_dict(config)


def analyze_token_usage(log_dir: str) -> Optional[Dict[str, Any]]:
    """Analyze token usage from log files."""
    if not log_dir or not os.path.exists(log_dir):
        return None

    token_stats = defaultdict(lambda: {'input_tokens': 0, 'output_tokens': 0, 'count': 0})

    for root, _, files in os.walk(log_dir):
        for file in files:
            if not file.endswith('.json') or not file.startswith('game_'):
                continue

            try:
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                model = data.get('model', 'unknown')
                turns = data.get('turns', [])
                file_input_tokens = 0
                file_output_tokens = 0

                for turn in turns:
                    usage = turn.get('usage') or turn.get('token_usage') or turn.get('usage_metadata')
                    if usage and isinstance(usage, dict):
                        file_input_tokens += (usage.get('prompt_tokens') or
                                              usage.get('input_tokens') or
                                              usage.get('prompt_token_count') or 0)
                        file_output_tokens += (usage.get('completion_tokens') or
                                               usage.get('output_tokens') or
                                               usage.get('candidates_token_count') or 0)

                if file_input_tokens > 0 or file_output_tokens > 0:
                    token_stats[model]['input_tokens'] += file_input_tokens
                    token_stats[model]['output_tokens'] += file_output_tokens
                    token_stats[model]['count'] += 1

            except Exception as e:
                logger.warning(f"Error reading log file {file}: {e}")

    return dict(token_stats) if token_stats else None


def save_tournament_results(
    results_data: Dict[str, Any],
    output_dir: str,
    llm_log_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Save tournament results to JSON and CSV files.

    Args:
        results_data: Tournament results data
        output_dir: Directory to save results
        llm_log_dir: Directory containing LLM conversation logs for token analysis
        config: Original tournament configuration (will be sanitized before saving)
    """
    if not results_data:
        logger.warning("No results data to save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add metadata
    try:
        import reinforcetactics
        rt_version = getattr(reinforcetactics, '__version__', 'unknown')
    except ImportError:
        rt_version = 'unknown'

    data_to_save = results_data.copy()
    data_to_save['metadata'] = {
        'reinforcetactics_version': rt_version,
        'timestamp': timestamp,
        'export_time': datetime.now().isoformat()
    }

    # Analyze token usage
    if llm_log_dir:
        logger.info(f"Analyzing token usage from: {llm_log_dir}")
        token_stats = analyze_token_usage(llm_log_dir)
        if token_stats:
            data_to_save['token_stats'] = token_stats
            logger.info("Token usage analysis complete")

            # Print token summary
            logger.info("\nToken Usage Summary:")
            for model, stats in token_stats.items():
                total = stats['input_tokens'] + stats['output_tokens']
                logger.info(f"  {model}: {stats['input_tokens']} in, "
                           f"{stats['output_tokens']} out, {total} total ({stats['count']} games)")

            # Save token stats CSV
            token_csv_path = os.path.join(output_dir, f'token_usage_{timestamp}.csv')
            with open(token_csv_path, 'w') as f:
                f.write("Model,Input Tokens,Output Tokens,Total Tokens,Games Logged\n")
                for model, stats in token_stats.items():
                    total = stats['input_tokens'] + stats['output_tokens']
                    f.write(f"{model},{stats['input_tokens']},{stats['output_tokens']},"
                           f"{total},{stats['count']}\n")
            logger.info(f"Token stats saved to: {token_csv_path}")

    # Save sanitized configuration (API keys removed)
    if config:
        sanitized_config = sanitize_config_for_saving(config)
        config_path = os.path.join(output_dir, f'tournament_config_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(sanitized_config, f, indent=2)
        logger.info(f"Configuration saved to: {config_path}")

    # Save full results JSON
    json_path = os.path.join(output_dir, f'tournament_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    logger.info(f"Full results saved to: {json_path}")

    # Save standings CSV
    if 'standings' in results_data:
        csv_path = os.path.join(output_dir, f'tournament_standings_{timestamp}.csv')
        with open(csv_path, 'w') as f:
            f.write("Bot,Wins,Losses,Draws,Total Games,Win Rate,Elo,Elo Change\n")
            for s in results_data['standings']:
                f.write(f"{s['bot']},{s['wins']},{s['losses']},{s['draws']},"
                       f"{s['total_games']},{s['win_rate']:.3f},"
                       f"{s['elo']:.0f},{s['elo_change']:+.0f}\n")
        logger.info(f"Standings saved to: {csv_path}")

    # Save matchups CSV
    if 'matchups' in results_data and results_data['matchups']:
        matchups_csv_path = os.path.join(output_dir, f'tournament_matchups_{timestamp}.csv')
        with open(matchups_csv_path, 'w') as f:
            f.write("Bot 1,Bot 2,Bot 1 Wins,Bot 2 Wins,Draws\n")
            for m in results_data['matchups']:
                f.write(f"{m['bot1']},{m['bot2']},{m['bot1_wins']},"
                       f"{m['bot2_wins']},{m['draws']}\n")
        logger.info(f"Matchups saved to: {matchups_csv_path}")

        # Save matrix CSV
        matrix_csv_path = os.path.join(output_dir, f'tournament_matrix_{timestamp}.csv')
        bots_set = set()
        for m in results_data['matchups']:
            bots_set.add(m['bot1'])
            bots_set.add(m['bot2'])
        bots_list = sorted(list(bots_set))

        matrix = {b1: {b2: "0-0-0" for b2 in bots_list} for b1 in bots_list}
        for b in bots_list:
            matrix[b][b] = 'X'

        for m in results_data['matchups']:
            b1, b2 = m['bot1'], m['bot2']
            w1, w2, d = int(m['bot1_wins']), int(m['bot2_wins']), int(m['draws'])
            matrix[b1][b2] = f"{w1}-{w2}-{d}"
            matrix[b2][b1] = f"{w2}-{w1}-{d}"

        with open(matrix_csv_path, 'w') as f:
            f.write("," + ",".join(bots_list) + "\n")
            for b1 in bots_list:
                row = [matrix[b1][b2] for b2 in bots_list]
                f.write(f"{b1}," + ",".join(row) + "\n")
        logger.info(f"Matrix table saved to: {matrix_csv_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Reinforce Tactics tournament from config file'
    )
    parser.add_argument(
        '--config',
        default='/app/config/config.json',
        help='Path to tournament configuration file (default: /app/config/config.json)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to folder containing previous tournament output to resume from'
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Create bots
    bots = create_bots_from_config(config)
    if len(bots) < 2:
        logger.error("Need at least 2 bots for a tournament. Check API keys and config.")
        sys.exit(1)

    logger.info(f"Created {len(bots)} bots from config")

    # Parse maps from config (supports both string and object format with per-map max_turns)
    map_configs = parse_maps_from_config(config)
    logger.info(f"Loaded {len(map_configs)} maps from config")

    # Get tournament settings
    tournament_config = config['tournament']
    output_config = config.get('output', {})
    gcs_config = config.get('gcs', {})

    # Set up GCS uploader if configured
    gcs_uploader = None
    if gcs_config.get('enabled', False):
        bucket_name = gcs_config.get('bucket')
        if bucket_name:
            try:
                gcs_uploader = GCSUploader(
                    bucket_name=bucket_name,
                    prefix=gcs_config.get('prefix', ''),
                    credentials_file=gcs_config.get('credentials_file')
                )
                logger.info(f"GCS uploader configured for bucket: {bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS uploader: {e}")
                logger.warning("Continuing without GCS upload...")
        else:
            logger.warning("GCS enabled but no bucket specified")

    # Check for resume mode
    completed_matches = None
    if args.resume:
        logger.info(f"Resume mode enabled, scanning folder: {args.resume}")
        completed_matches = scan_completed_matches(args.resume)
        if not completed_matches:
            logger.info("No completed matches found, starting fresh tournament")
            completed_matches = None

    # Run tournament
    try:
        results = run_tournament(
            bots=bots,
            maps=map_configs,
            map_pool_mode=tournament_config.get('map_pool_mode', 'all'),
            games_per_matchup=tournament_config.get('games_per_matchup', 1),
            max_turns=tournament_config.get('max_turns', 100),
            should_reason=tournament_config.get('should_reason', False),
            log_conversations=tournament_config.get('log_conversations', False),
            conversation_log_dir=output_config.get('conversation_log_dir'),
            save_replays=tournament_config.get('save_replays', False),
            replay_dir=output_config.get('replay_dir'),
            completed_matches=completed_matches,
            gcs_uploader=gcs_uploader,
            concurrent_games=tournament_config.get('concurrent_games', 1),
            llm_api_delay=tournament_config.get('llm_api_delay', 1.0)
        )

        # Save results locally
        results_dir = output_config.get('results_dir', '/app/output/results')
        conversation_log_dir = output_config.get('conversation_log_dir')

        save_tournament_results(results, results_dir, conversation_log_dir, config)

        # Upload results and logs to GCS if configured
        if gcs_uploader:
            logger.info("Uploading tournament output to GCS...")

            # Upload results directory
            gcs_uploader.upload_directory(results_dir, 'results')

            # Upload conversation logs if they exist
            if conversation_log_dir and os.path.exists(conversation_log_dir):
                gcs_uploader.upload_directory(conversation_log_dir, 'conversations')

            # Upload replay directory if it exists
            replay_dir = output_config.get('replay_dir')
            if replay_dir and os.path.exists(replay_dir):
                gcs_uploader.upload_directory(replay_dir, 'replays')

            logger.info("GCS upload complete!")

        logger.info("\nTournament completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Tournament failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
