"""
Tournament runner - core execution engine.

This module provides the main TournamentRunner class that executes
tournament games and manages the overall tournament flow.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from reinforcetactics.core.game_state import GameState
from reinforcetactics.utils.file_io import FileIO

from .bots import BotDescriptor, BotType, create_bot_instance
from .config import TournamentConfig
from .elo import EloRatingSystem
from .results import GameResult, ResultsExporter, TournamentResults
from .schedule import (
    CompletedMatchInfo,
    MapConfig,
    ScheduledGame,
    generate_round_robin_schedule,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TournamentRunner:
    """
    Runs round-robin tournaments between bots.

    This class handles:
    - Game scheduling and execution
    - Concurrent game execution (optional)
    - Progress tracking and logging
    - Results aggregation
    - Replay saving

    Example usage:
        config = TournamentConfig(
            maps=["maps/1v1/6x6_beginner.csv"],
            games_per_side=2
        )
        runner = TournamentRunner(config)
        bots = [
            BotDescriptor.simple_bot("SimpleBot"),
            BotDescriptor.medium_bot("MediumBot"),
        ]
        results = runner.run(bots)
    """

    def __init__(
        self,
        config: TournamentConfig,
        elo_system: Optional[EloRatingSystem] = None,
    ):
        """
        Initialize tournament runner.

        Args:
            config: Tournament configuration
            elo_system: Optional EloRatingSystem (creates new one if None)
        """
        self.config = config
        self.elo_system = elo_system or EloRatingSystem()
        self.results = TournamentResults(self.elo_system)
        self.exporter = ResultsExporter(config.output_dir)

        # For concurrent execution
        self._results_lock = Lock()

        # Progress callback
        self._progress_callback: Optional[Callable[[int, int, GameResult], None]] = None

        # Completed matches for resume support
        self._completed_matches: Optional[Dict[str, List[CompletedMatchInfo]]] = None

        # Create output directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary output directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        if self.config.save_replays and self.config.replay_dir:
            os.makedirs(self.config.replay_dir, exist_ok=True)

        if self.config.log_conversations and self.config.conversation_log_dir:
            os.makedirs(self.config.conversation_log_dir, exist_ok=True)

    def set_progress_callback(
        self, callback: Callable[[int, int, GameResult], None]
    ) -> None:
        """
        Set a callback for progress updates.

        Args:
            callback: Function(completed, total, last_result) called after each game
        """
        self._progress_callback = callback

    def set_completed_matches(
        self, completed: Dict[str, List[CompletedMatchInfo]]
    ) -> None:
        """
        Set completed matches for resume support.

        Args:
            completed: Dictionary of completed matches
        """
        self._completed_matches = completed

    def run(self, bots: List[BotDescriptor]) -> TournamentResults:
        """
        Run the tournament.

        Args:
            bots: List of bot descriptors to compete

        Returns:
            TournamentResults with all game data

        Raises:
            ValueError: If less than 2 bots provided
        """
        if len(bots) < 2:
            raise ValueError("Need at least 2 bots for a tournament")

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {'; '.join(errors)}")

        # Initialize Elo ratings
        for bot in bots:
            self.elo_system.initialize_bot(bot.name)

        # Log tournament info
        self._log_tournament_start(bots)

        # Mark start time
        self.results.start()

        # Generate schedule
        schedule, skipped = generate_round_robin_schedule(
            bots=bots,
            map_configs=self.config.maps,
            games_per_side=self.config.games_per_side,
            map_pool_mode=self.config.map_pool_mode,
            completed_matches=self._completed_matches,
        )

        total_games = sum(len(round_games) for round_games in schedule)
        logger.info(f"Schedule: {len(schedule)} rounds, {total_games} games")
        if skipped > 0:
            logger.info(f"Skipping {skipped} already completed games")

        # Execute games
        completed_count = 0
        for round_idx, round_games in enumerate(schedule):
            if not round_games:
                continue

            map_name = round_games[0].map_config.name
            logger.info(f"\nRound {round_idx + 1}/{len(schedule)}: {map_name}")
            logger.info(f"Games in round: {len(round_games)}")

            if self.config.concurrent_games > 1:
                completed_count = self._run_round_concurrent(
                    round_games, completed_count, total_games
                )
            else:
                completed_count = self._run_round_sequential(
                    round_games, completed_count, total_games
                )

        # Mark end time
        self.results.finish()

        # Log final results
        self.exporter.print_standings(self.results)

        return self.results

    def _run_round_sequential(
        self,
        games: List[ScheduledGame],
        completed_count: int,
        total_games: int,
    ) -> int:
        """Run games sequentially."""
        for game in games:
            logger.info(
                f"  Game {completed_count + 1}/{total_games}: "
                f"{game.bot1.name} vs {game.bot2.name}"
            )

            result = self._execute_game(game)
            self.results.add_game_result(result)
            completed_count += 1

            logger.info(f"    Result: {result.winner_name} ({result.turns} turns)")

            if self._progress_callback:
                self._progress_callback(completed_count, total_games, result)

        return completed_count

    def _run_round_concurrent(
        self,
        games: List[ScheduledGame],
        completed_count: int,
        total_games: int,
    ) -> int:
        """Run games concurrently."""
        concurrency = min(self.config.concurrent_games, len(games))
        logger.info(f"  Running with {concurrency} concurrent workers")

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_game = {
                executor.submit(self._execute_game, game): game for game in games
            }

            for future in as_completed(future_to_game):
                game = future_to_game[future]
                try:
                    result = future.result()
                    with self._results_lock:
                        self.results.add_game_result(result)
                        completed_count += 1
                        logger.info(
                            f"  [{completed_count}/{total_games}] "
                            f"{game.bot1.name} vs {game.bot2.name}: "
                            f"{result.winner_name} ({result.turns} turns)"
                        )
                        if self._progress_callback:
                            self._progress_callback(
                                completed_count, total_games, result
                            )
                except Exception as e:
                    logger.error(f"  Error in game {game.game_id}: {e}")
                    completed_count += 1

        return completed_count

    def _execute_game(self, scheduled_game: ScheduledGame) -> GameResult:
        """
        Execute a single scheduled game.

        Args:
            scheduled_game: Game to execute

        Returns:
            GameResult with outcome
        """
        bot1_desc = scheduled_game.bot1
        bot2_desc = scheduled_game.bot2
        map_config = scheduled_game.map_config

        # Set up directories for this game
        replay_path = None
        matchup_log_dir = None

        if self.config.save_replays and self.config.replay_dir:
            replay_subdir = Path(self.config.replay_dir) / map_config.stem
            replay_subdir.mkdir(parents=True, exist_ok=True)

        if self.config.log_conversations and self.config.conversation_log_dir:
            matchup_log_dir = str(
                Path(self.config.conversation_log_dir)
                / map_config.stem
                / f"{bot1_desc.name}_vs_{bot2_desc.name}"
            )
            os.makedirs(matchup_log_dir, exist_ok=True)

        # Create session ID for logging
        session_id = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{bot1_desc.name}_vs_{bot2_desc.name}"
        )

        # Load map and create game state
        map_data = FileIO.load_map(map_config.path)
        game_state = GameState(map_data, num_players=2)
        max_turns = map_config.max_turns or self.config.max_turns
        game_state.max_turns = max_turns

        # Create bot instances
        bot1 = create_bot_instance(
            bot1_desc,
            game_state,
            player=1,
            log_conversations=self.config.log_conversations,
            conversation_log_dir=matchup_log_dir,
            game_session_id=session_id,
            should_reason=self.config.should_reason,
        )
        bot2 = create_bot_instance(
            bot2_desc,
            game_state,
            player=2,
            log_conversations=self.config.log_conversations,
            conversation_log_dir=matchup_log_dir,
            game_session_id=session_id,
            should_reason=self.config.should_reason,
        )
        bots = {1: bot1, 2: bot2}

        # Check if LLM bots are involved (for API delay)
        has_llm = bot1_desc.bot_type == BotType.LLM or bot2_desc.bot_type == BotType.LLM

        # Play the game
        try:
            while not game_state.game_over and game_state.turn_number < max_turns:
                current_player = game_state.current_player
                current_bot = bots[current_player]
                current_bot.take_turn()

            # Determine winner
            if game_state.game_over and game_state.winner:
                winner = game_state.winner
                winner_name = bot1_desc.name if winner == 1 else bot2_desc.name
            elif game_state.turn_number >= max_turns:
                game_state.game_over = True
                winner = 0
                winner_name = "Draw"
            else:
                winner = 0
                winner_name = "Draw"

            # Save replay
            if self.config.save_replays and self.config.replay_dir:
                replay_path = self._save_replay(
                    game_state,
                    bot1_desc,
                    bot2_desc,
                    winner,
                    winner_name,
                    map_config,
                    scheduled_game.game_id,
                )

            # Apply LLM API delay
            if has_llm and self.config.llm_api_delay > 0:
                time.sleep(self.config.llm_api_delay)

            return GameResult(
                game_id=scheduled_game.game_id,
                bot1_name=bot1_desc.name,
                bot2_name=bot2_desc.name,
                winner=winner,
                winner_name=winner_name,
                turns=game_state.turn_number,
                map_name=map_config.name,
                replay_path=replay_path,
            )

        except Exception as e:
            logger.error(f"Error in game: {e}", exc_info=True)
            return GameResult(
                game_id=scheduled_game.game_id,
                bot1_name=bot1_desc.name,
                bot2_name=bot2_desc.name,
                winner=0,
                winner_name="Error",
                turns=game_state.turn_number,
                map_name=map_config.name,
                error=str(e),
            )

    def _save_replay(
        self,
        game_state: GameState,
        bot1_desc: BotDescriptor,
        bot2_desc: BotDescriptor,
        winner: int,
        winner_name: str,
        map_config: MapConfig,
        game_id: int,
    ) -> str:
        """Save game replay and return path."""
        replay_filename = (
            f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{bot1_desc.name}_vs_{bot2_desc.name}_{map_config.stem}.json"
        )
        replay_path = str(
            Path(self.config.replay_dir) / map_config.stem / replay_filename
        )

        # Map bot_type to player_type
        type_mapping = {
            BotType.SIMPLE: "bot",
            BotType.MEDIUM: "bot",
            BotType.ADVANCED: "bot",
            BotType.LLM: "llm",
            BotType.MODEL: "rl",
        }

        game_info = {
            "bot1": bot1_desc.name,
            "bot2": bot2_desc.name,
            "bot1_player": 1,
            "bot2_player": 2,
            "winner": winner,
            "winner_name": winner_name,
            "turns": game_state.turn_number,
            "max_turns": map_config.max_turns or self.config.max_turns,
            "map": map_config.path,
            "player_configs": [
                GameState.build_player_config(
                    player_no=1,
                    name=bot1_desc.name,
                    player_type=type_mapping.get(bot1_desc.bot_type, "bot"),
                    temperature=bot1_desc.temperature,
                    max_tokens=bot1_desc.max_tokens,
                ),
                GameState.build_player_config(
                    player_no=2,
                    name=bot2_desc.name,
                    player_type=type_mapping.get(bot2_desc.bot_type, "bot"),
                    temperature=bot2_desc.temperature,
                    max_tokens=bot2_desc.max_tokens,
                ),
            ],
        }

        FileIO.save_replay(game_state.action_history, game_info, replay_path)
        return replay_path

    def _log_tournament_start(self, bots: List[BotDescriptor]) -> None:
        """Log tournament start information."""
        logger.info("=" * 70)
        logger.info(f"TOURNAMENT: {self.config.name}")
        logger.info("=" * 70)

        if len(self.config.maps) == 1:
            m = self.config.maps[0]
            logger.info(f"Map: {m.path} (max_turns: {m.max_turns})")
        else:
            logger.info(f"Maps: {len(self.config.maps)}")
            for m in self.config.maps:
                logger.info(f"  - {m.path} (max_turns: {m.max_turns})")
            logger.info(f"Map Pool Mode: {self.config.map_pool_mode}")

        logger.info(f"Participants: {len(bots)}")
        for bot in bots:
            logger.info(f"  - {bot.get_display_info()}")

        logger.info(f"Games per side: {self.config.games_per_side}")

        if self.config.concurrent_games > 1:
            logger.info(f"Concurrent games: {self.config.concurrent_games}")
        if self.config.should_reason:
            logger.info("LLM Reasoning: ENABLED")
        if self.config.log_conversations:
            logger.info("Conversation Logging: ENABLED")
        if self.config.save_replays:
            logger.info("Replay Saving: ENABLED")

        logger.info("=" * 70)

    def export_results(
        self, timestamp: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export results to all formats.

        Args:
            timestamp: Optional timestamp for filenames

        Returns:
            Dictionary mapping format to file path
        """
        return self.exporter.export_all(
            self.results,
            config=self.config.to_dict(),
            timestamp=timestamp,
        )
