#!/usr/bin/env python3
"""
Docker Tournament Runner for Reinforce Tactics.

Reads tournament configuration from config.json and runs a round-robin tournament
between specified bots. Supports built-in bots (simple, medium, advanced) and
LLM bots (OpenAI, Anthropic, Google).

Features:
- Resume interrupted tournaments by passing --resume with a folder path
- Upload replays and conversation logs to Google Cloud Storage
- Concurrent game execution

This script is a thin wrapper around the tournament library with Docker-specific
features (GCS upload, resume support).

Usage:
    python run_tournament.py [--config CONFIG_PATH] [--resume FOLDER_PATH]
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, '/app')

from reinforcetactics.tournament import (
    TournamentConfig,
    TournamentRunner,
    BotDescriptor,
    BotType,
    MapConfig,
)
from reinforcetactics.tournament.config import parse_bots_from_config
from reinforcetactics.tournament.schedule import CompletedMatchInfo

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
                    self._client = storage.Client()
                    logger.info("GCS client initialized with default credentials")
                self._bucket = self._client.bucket(self.bucket_name)
            except ImportError:
                logger.error("google-cloud-storage package not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize GCS client: {e}")
                raise
        return self._client

    def upload_file(self, local_path: str, remote_path: Optional[str] = None) -> Optional[str]:
        """Upload a file to GCS."""
        try:
            self._get_client()

            if remote_path is None:
                remote_path = os.path.basename(local_path)

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
        """Upload all files in a directory to GCS."""
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


def scan_completed_matches(resume_folder: str) -> Dict[str, List[CompletedMatchInfo]]:
    """
    Scan a folder for completed match replay files.

    Args:
        resume_folder: Path to folder containing replays or tournament output

    Returns:
        Dictionary mapping matchup keys to list of completed matches
    """
    completed = defaultdict(list)
    resume_path = Path(resume_folder)

    if not resume_path.exists():
        logger.warning(f"Resume folder does not exist: {resume_folder}")
        return completed

    # Look for replay files in multiple possible locations
    search_paths = [
        resume_path,
        resume_path / 'replays',
        resume_path / 'output' / 'replays',
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

            map_name = Path(map_path).name if map_path else ''

            if not bot1 or not bot2 or not map_name:
                continue

            player1_bot = bot1 if bot1_player == 1 else bot2
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

    total_matches = sum(len(matches) for matches in completed.values())
    logger.info(f"Found {total_matches} completed matches")
    return completed


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate tournament configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if 'tournament' not in config:
        raise ValueError("Config missing 'tournament' section")
    if 'maps' not in config or not config['maps']:
        raise ValueError("Config missing 'maps' or maps list is empty")
    if 'bots' not in config or len(config['bots']) < 2:
        raise ValueError("Config missing 'bots' or needs at least 2 bots")

    return config


def create_bots_from_config(config: Dict[str, Any]) -> List[BotDescriptor]:
    """Create BotDescriptor instances from config."""
    bots = []

    for bot_config in config['bots']:
        name = bot_config['name']
        bot_type = bot_config['type']

        if bot_type in ('simple', 'medium', 'advanced'):
            bots.append(BotDescriptor(
                name=name,
                bot_type=BotType(bot_type)
            ))
        elif bot_type == 'llm':
            provider = bot_config.get('provider')
            model = bot_config.get('model')
            temperature = bot_config.get('temperature')
            max_tokens = bot_config.get('max_tokens', 8000)

            # Validate API keys
            if provider == 'openai':
                if not os.environ.get('OPENAI_API_KEY'):
                    logger.warning(f"Skipping {name}: OPENAI_API_KEY not set")
                    continue
            elif provider == 'anthropic':
                if not os.environ.get('ANTHROPIC_API_KEY'):
                    logger.warning(f"Skipping {name}: ANTHROPIC_API_KEY not set")
                    continue
            elif provider == 'google':
                if not os.environ.get('GOOGLE_API_KEY'):
                    logger.warning(f"Skipping {name}: GOOGLE_API_KEY not set")
                    continue

            bots.append(BotDescriptor(
                name=name,
                bot_type=BotType.LLM,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ))
        elif bot_type == 'model':
            model_path = bot_config.get('model_path')
            if model_path and Path(model_path).exists():
                bots.append(BotDescriptor(
                    name=name,
                    bot_type=BotType.MODEL,
                    model_path=model_path
                ))
            else:
                logger.warning(f"Skipping {name}: model file not found at {model_path}")

    return bots


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Reinforce Tactics tournament from config file'
    )
    parser.add_argument(
        '--config',
        default='/app/config/config.json',
        help='Path to tournament configuration file'
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
        raw_config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Create bots
    bots = create_bots_from_config(raw_config)
    if len(bots) < 2:
        logger.error("Need at least 2 bots for a tournament. Check API keys and config.")
        sys.exit(1)

    logger.info(f"Created {len(bots)} bots from config")

    # Create tournament config
    tournament_data = raw_config['tournament']
    output_data = raw_config.get('output', {})
    gcs_config = raw_config.get('gcs', {})

    config = TournamentConfig(
        name=tournament_data.get('name', 'Docker Tournament'),
        maps=[
            MapConfig.from_config(m, tournament_data.get('max_turns', 500))
            for m in raw_config['maps']
        ],
        map_pool_mode=tournament_data.get('map_pool_mode', 'all'),
        games_per_side=tournament_data.get('games_per_matchup', 1),
        max_turns=tournament_data.get('max_turns', 500),
        output_dir=output_data.get('results_dir', '/app/output/results'),
        save_replays=tournament_data.get('save_replays', True),
        replay_dir=output_data.get('replay_dir', '/app/output/replays'),
        log_conversations=tournament_data.get('log_conversations', False),
        conversation_log_dir=output_data.get('conversation_log_dir'),
        should_reason=tournament_data.get('should_reason', False),
        llm_api_delay=tournament_data.get('llm_api_delay', 1.0),
        concurrent_games=tournament_data.get('concurrent_games', 1),
    )

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

    # Check for resume mode
    completed_matches = None
    if args.resume:
        logger.info(f"Resume mode enabled, scanning folder: {args.resume}")
        completed_matches = scan_completed_matches(args.resume)
        if not completed_matches:
            logger.info("No completed matches found, starting fresh")
            completed_matches = None

    # Create runner
    runner = TournamentRunner(config)

    if completed_matches:
        runner.set_completed_matches(completed_matches)

    # Run tournament
    try:
        results = runner.run(bots)

        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = runner.export_results(timestamp)

        logger.info(f"\nResults exported to: {config.output_dir}")

        # Upload to GCS if configured
        if gcs_uploader:
            logger.info("Uploading tournament output to GCS...")

            # Upload results directory
            gcs_uploader.upload_directory(config.output_dir, 'results')

            # Upload conversation logs
            if config.conversation_log_dir and os.path.exists(config.conversation_log_dir):
                gcs_uploader.upload_directory(config.conversation_log_dir, 'conversations')

            # Upload replays
            if config.replay_dir and os.path.exists(config.replay_dir):
                gcs_uploader.upload_directory(config.replay_dir, 'replays')

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
