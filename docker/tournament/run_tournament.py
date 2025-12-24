#!/usr/bin/env python3
"""
Docker Tournament Runner for Reinforce Tactics.

Reads tournament configuration from config.json and runs a round-robin tournament
between specified bots. Supports built-in bots (simple, medium, advanced) and
LLM bots (OpenAI, Anthropic, Google).

Usage:
    python run_tournament.py [--config CONFIG_PATH]
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    replay_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a round-robin tournament between multiple bots with Elo ratings.

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
    logger.info("=" * 70)

    # Initialize results tracking
    results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
    per_map_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0}))
    matchup_details = []
    current_map_index = 0

    def select_map(game_num):
        nonlocal current_map_index
        if len(map_list) == 1:
            return map_list[0]
        if map_pool_mode == 'cycle':
            selected = map_list[current_map_index % len(map_list)]
            current_map_index += 1
            return selected
        elif map_pool_mode == 'random':
            return random.choice(map_list)
        else:
            return map_list[(game_num - 1) % len(map_list)]

    # Generate all matchups (round-robin)
    matchups = []
    for i in range(len(bots)):
        for j in range(i + 1, len(bots)):
            matchups.append((i, j))

    total_games = len(matchups) * games_per_matchup_total
    logger.info(f"Total matchups: {len(matchups)}")
    logger.info(f"Total games: {total_games}")

    game_num = 0
    known_llm_bots = ['OpenAIBot', 'ClaudeBot', 'GeminiBot']

    # Run all matchups
    for matchup_idx, (i, j) in enumerate(matchups, 1):
        bot1 = bots[i]
        bot2 = bots[j]

        logger.info(f"\n{'='*70}")
        logger.info(f"Matchup {matchup_idx}/{len(matchups)}: {bot1.name} vs {bot2.name}")
        logger.info("=" * 70)

        has_llm_bot = False
        bot1_is_llm = not isinstance(bot1.bot_class, str) and bot1.bot_class.__name__ in known_llm_bots
        bot2_is_llm = not isinstance(bot2.bot_class, str) and bot2.bot_class.__name__ in known_llm_bots
        has_llm_bot = bot1_is_llm or bot2_is_llm

        matchup_results = {
            'bot1': bot1.name,
            'bot2': bot2.name,
            'bot1_wins': 0,
            'bot2_wins': 0,
            'draws': 0
        }

        # Determine game schedule
        if map_pool_mode == 'all' and len(map_list) > 1:
            game_schedule = []
            for m in map_list:
                for _ in range(games_per_matchup):
                    game_schedule.append((bot1, bot2, 1, 2, m))
                for _ in range(games_per_matchup):
                    game_schedule.append((bot2, bot1, 1, 2, m))
        else:
            game_schedule = []
            for g in range(games_per_matchup):
                m = select_map(game_num + g + 1)
                game_schedule.append((bot1, bot2, 1, 2, m))
            for g in range(games_per_matchup):
                m = select_map(game_num + games_per_matchup + g + 1)
                game_schedule.append((bot2, bot1, 1, 2, m))

        # Run games
        for game_idx, (p1_bot, p2_bot, p1_num, p2_num, map_config) in enumerate(game_schedule, 1):
            game_num += 1
            map_path = map_config.path
            map_max_turns = map_config.max_turns
            map_name = Path(map_path).name

            logger.info(f"  Game {game_idx}/{len(game_schedule)}: "
                       f"{p1_bot.name} (P1) vs {p2_bot.name} (P2) on {map_name} (max_turns: {map_max_turns})")

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

            # Run the game with per-map max_turns
            result = run_single_game(
                p1_bot, p2_bot, p1_num, p2_num, map_path,
                max_turns=map_max_turns,
                should_reason=should_reason,
                log_conversations=log_conversations,
                conversation_log_dir=matchup_log_dir,
                save_replay=save_replays,
                replay_dir=matchup_replay_dir
            )

            winner = result['winner']
            winner_name = result['winner_name']
            turns = result['turns']

            logger.info(f"    Result: {winner_name} (turns: {turns})")

            # Update statistics
            if winner == p1_num:
                if p1_bot.name == bot1.name:
                    results[bot1.name]['wins'] += 1
                    results[bot2.name]['losses'] += 1
                    per_map_stats[bot1.name][map_name]['wins'] += 1
                    per_map_stats[bot2.name][map_name]['losses'] += 1
                    matchup_results['bot1_wins'] += 1
                    elo_system.update_ratings(bot1.name, bot2.name, 1)
                else:
                    results[bot2.name]['wins'] += 1
                    results[bot1.name]['losses'] += 1
                    per_map_stats[bot2.name][map_name]['wins'] += 1
                    per_map_stats[bot1.name][map_name]['losses'] += 1
                    matchup_results['bot2_wins'] += 1
                    elo_system.update_ratings(bot1.name, bot2.name, 2)
            elif winner == p2_num:
                if p2_bot.name == bot1.name:
                    results[bot1.name]['wins'] += 1
                    results[bot2.name]['losses'] += 1
                    per_map_stats[bot1.name][map_name]['wins'] += 1
                    per_map_stats[bot2.name][map_name]['losses'] += 1
                    matchup_results['bot1_wins'] += 1
                    elo_system.update_ratings(bot1.name, bot2.name, 1)
                else:
                    results[bot2.name]['wins'] += 1
                    results[bot1.name]['losses'] += 1
                    per_map_stats[bot2.name][map_name]['wins'] += 1
                    per_map_stats[bot1.name][map_name]['losses'] += 1
                    matchup_results['bot2_wins'] += 1
                    elo_system.update_ratings(bot1.name, bot2.name, 2)
            else:
                results[bot1.name]['draws'] += 1
                results[bot2.name]['draws'] += 1
                per_map_stats[bot1.name][map_name]['draws'] += 1
                per_map_stats[bot2.name][map_name]['draws'] += 1
                matchup_results['draws'] += 1
                elo_system.update_ratings(bot1.name, bot2.name, 0)

            # Add delay between LLM games to avoid rate limits
            if has_llm_bot and game_idx < len(game_schedule):
                time.sleep(1)

        matchup_details.append(matchup_results)

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

    return {
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
            'turns': turn_count
        }

    except Exception as e:
        logger.error(f"Error during game: {e}", exc_info=True)
        return {
            'winner': 0,
            'winner_name': "Error",
            'turns': turn_count,
            'error': str(e)
        }


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
    llm_log_dir: Optional[str] = None
) -> None:
    """Save tournament results to JSON and CSV files."""
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
            replay_dir=output_config.get('replay_dir')
        )

        # Save results
        results_dir = output_config.get('results_dir', '/app/output/results')
        conversation_log_dir = output_config.get('conversation_log_dir')

        save_tournament_results(results, results_dir, conversation_log_dir)

        logger.info("\nTournament completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Tournament failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
