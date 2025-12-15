#!/usr/bin/env python3
# pylint: disable=logging-fstring-interpolation
"""
Round-robin tournament script for Reinforce Tactics bots.

Runs tournaments between all configured bots, including:
- Built-in SimpleBot
- LLM bots (if API keys configured and working)
- Trained model bots (from models/ directory)

Each matchup consists of multiple games with sides swapped to account
for first-move advantage.
"""
import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot, MediumBot, AdvancedBot
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EloRatingSystem:
    """Manages Elo ratings for tournament participants."""

    def __init__(self, starting_elo: int = 1500, k_factor: int = 32):
        """
        Initialize Elo rating system.

        Args:
            starting_elo: Initial Elo rating for all bots (default: 1500)
            k_factor: K-factor for rating changes (default: 32)
        """
        self.starting_elo = starting_elo
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.initial_ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, List[float]] = {}

    def initialize_bot(self, bot_name: str) -> None:
        """
        Initialize a bot with starting Elo rating.

        Args:
            bot_name: Name of the bot
        """
        if bot_name not in self.ratings:
            self.ratings[bot_name] = float(self.starting_elo)
            self.initial_ratings[bot_name] = float(self.starting_elo)
            self.rating_history[bot_name] = [float(self.starting_elo)]

    def calculate_expected_score(self, player_elo: float, opponent_elo: float) -> float:
        """
        Calculate expected score for a player.

        Args:
            player_elo: Player's current Elo rating
            opponent_elo: Opponent's current Elo rating

        Returns:
            Expected score (0.0 to 1.0)
        """
        return 1.0 / (1.0 + 10 ** ((opponent_elo - player_elo) / 400.0))

    def update_ratings(self, bot1_name: str, bot2_name: str, result: int) -> None:
        """
        Update Elo ratings after a game.

        Args:
            bot1_name: Name of first bot
            bot2_name: Name of second bot
            result: Game result (1=bot1 wins, 2=bot2 wins, 0=draw)
        """
        # Initialize bots if needed
        self.initialize_bot(bot1_name)
        self.initialize_bot(bot2_name)

        # Get current ratings
        bot1_elo = self.ratings[bot1_name]
        bot2_elo = self.ratings[bot2_name]

        # Calculate expected scores
        bot1_expected = self.calculate_expected_score(bot1_elo, bot2_elo)
        bot2_expected = self.calculate_expected_score(bot2_elo, bot1_elo)

        # Determine actual scores
        if result == 1:  # bot1 wins
            bot1_actual = 1.0
            bot2_actual = 0.0
        elif result == 2:  # bot2 wins
            bot1_actual = 0.0
            bot2_actual = 1.0
        else:  # draw
            bot1_actual = 0.5
            bot2_actual = 0.5

        # Update ratings
        bot1_new = bot1_elo + self.k_factor * (bot1_actual - bot1_expected)
        bot2_new = bot2_elo + self.k_factor * (bot2_actual - bot2_expected)

        self.ratings[bot1_name] = bot1_new
        self.ratings[bot2_name] = bot2_new

        # Record history
        self.rating_history[bot1_name].append(bot1_new)
        self.rating_history[bot2_name].append(bot2_new)

    def get_rating(self, bot_name: str) -> float:
        """
        Get current Elo rating for a bot.

        Args:
            bot_name: Name of the bot

        Returns:
            Current Elo rating
        """
        return self.ratings.get(bot_name, float(self.starting_elo))

    def get_rating_change(self, bot_name: str) -> float:
        """
        Get Elo rating change since tournament start.

        Args:
            bot_name: Name of the bot

        Returns:
            Rating change (positive or negative)
        """
        initial = self.initial_ratings.get(bot_name, float(self.starting_elo))
        current = self.ratings.get(bot_name, float(self.starting_elo))
        return current - initial

    def save_ratings(self, filepath: str) -> None:
        """
        Save ratings to a JSON file.

        Args:
            filepath: Path to save ratings
        """
        data = {
            'ratings': self.ratings,
            'rating_history': self.rating_history,
            'starting_elo': self.starting_elo,
            'k_factor': self.k_factor
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_ratings(self, filepath: str) -> None:
        """
        Load ratings from a JSON file.

        Args:
            filepath: Path to load ratings from
        
        Note:
            When loading, initial_ratings is set to current ratings so that
            rating changes track from this load point forward, not from the
            original tournament start.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ratings = data['ratings']
        self.rating_history = data['rating_history']
        self.starting_elo = data.get('starting_elo', 1500)
        self.k_factor = data.get('k_factor', 32)
        # Set initial ratings to current ratings when loading to track changes from this point
        self.initial_ratings = {k: v for k, v in self.ratings.items()}


class BotDescriptor:
    """Describes a bot that can participate in the tournament."""

    def __init__(self, name: str, bot_type: str, **kwargs):
        """
        Initialize bot descriptor.

        Args:
            name: Display name for the bot
            bot_type: Type of bot ('simple', 'llm', 'model')
            **kwargs: Additional arguments for bot initialization
        """
        self.name = name
        self.bot_type = bot_type
        self.kwargs = kwargs

    def create_bot(self, game_state: GameState, player: int):
        """
        Create an instance of this bot.

        Args:
            game_state: GameState instance
            player: Player number (1 or 2)

        Returns:
            Bot instance
        """
        if self.bot_type == 'simple':
            return SimpleBot(game_state, player)
        elif self.bot_type == 'medium':
            return MediumBot(game_state, player)
        elif self.bot_type == 'advanced':
            return AdvancedBot(game_state, player)
        elif self.bot_type == 'llm':
            bot_class = self.kwargs['bot_class']
            api_key = self.kwargs.get('api_key')
            log_conversations = self.kwargs.get('log_conversations', False)
            conversation_log_dir = self.kwargs.get('conversation_log_dir')
            return bot_class(
                game_state, player, api_key=api_key,
                log_conversations=log_conversations,
                conversation_log_dir=conversation_log_dir
            )
        elif self.bot_type == 'model':
            from reinforcetactics.game.model_bot import ModelBot
            model_path = self.kwargs['model_path']
            return ModelBot(game_state, player, model_path=model_path)
        else:
            raise ValueError(f"Unknown bot type: {self.bot_type}")

    def __repr__(self):
        return f"BotDescriptor(name={self.name}, type={self.bot_type})"


class TournamentRunner:
    """Runs a round-robin tournament between bots."""

    def __init__(self, map_file: Optional[str] = None, output_dir: str = 'tournament_results',
                 games_per_side: int = 2, log_conversations: bool = False,
                 conversation_log_dir: Optional[str] = None, maps: Optional[List[str]] = None,
                 map_pool_mode: str = 'cycle'):
        """
        Initialize tournament runner.

        Args:
            map_file: Path to single map file (deprecated, use maps parameter)
            output_dir: Directory for results and replays
            games_per_side: Number of games per side (total games = 2 * games_per_side)
            log_conversations: Enable conversation logging for LLM bots (default: False)
            conversation_log_dir: Directory for conversation logs (default: output_dir/llm_conversations/)
            maps: List of map file paths to use (takes precedence over map_file)
            map_pool_mode: Map selection mode ('cycle', 'random', 'all')
        """
        # Handle backward compatibility
        if maps:
            self.maps = maps
        elif map_file:
            self.maps = [map_file]
        else:
            self.maps = ['maps/1v1/6x6_beginner.csv']

        self.map_file = self.maps[0]  # Keep for backward compatibility
        self.map_pool_mode = map_pool_mode
        self.output_dir = Path(output_dir)
        self.games_per_side = games_per_side
        self.log_conversations = log_conversations
        # Default conversation log dir to output_dir/llm_conversations/ if logging is enabled
        if log_conversations and conversation_log_dir is None:
            self.conversation_log_dir = str(self.output_dir / 'llm_conversations')
        else:
            self.conversation_log_dir = conversation_log_dir
        self.results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        self.matchup_results = []
        self.per_map_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0}))
        self.map_game_lengths = defaultdict(list)
        self.current_map_index = 0  # For cycling through maps
        self.elo_system = EloRatingSystem()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir = self.output_dir / 'replays'
        self.replays_dir.mkdir(exist_ok=True)

    def discover_bots(self, models_dir: Optional[str] = None, include_test_bots: bool = False) -> List[BotDescriptor]:
        """
        Discover all available bots.

        Args:
            models_dir: Directory containing trained models (default: 'models/')
            include_test_bots: If True, add duplicate SimpleBots for testing (default: False)

        Returns:
            List of BotDescriptor objects
        """
        bots = []

        # Always include SimpleBot, MediumBot, and AdvancedBot
        bots.append(BotDescriptor('SimpleBot', 'simple'))
        logger.info("Added SimpleBot")
        
        bots.append(BotDescriptor('MediumBot', 'medium'))
        logger.info("Added MediumBot")
        
        bots.append(BotDescriptor('AdvancedBot', 'advanced'))
        logger.info("Added AdvancedBot")

        # For testing: add duplicate SimpleBots if requested
        if include_test_bots:
            bots.append(BotDescriptor('SimpleBot2', 'simple'))
            logger.info("Added SimpleBot2 (test bot)")

        # Try to add LLM bots if API keys are configured
        bots.extend(self._discover_llm_bots())

        # Try to add model bots from models directory
        if models_dir:
            bots.extend(self._discover_model_bots(models_dir))

        logger.info(f"Discovered {len(bots)} bots total")
        return bots

    def _discover_llm_bots(self) -> List[BotDescriptor]:
        """
        Discover LLM bots with valid API keys.

        Returns:
            List of LLM BotDescriptors
        """
        bots = []
        settings = get_settings()

        # Check OpenAI
        openai_key = settings.get_api_key('openai')
        if openai_key and self._test_openai_bot(openai_key):
            try:
                from reinforcetactics.game.llm_bot import OpenAIBot
                bots.append(BotDescriptor(
                    'OpenAIBot',
                    'llm',
                    bot_class=OpenAIBot,
                    api_key=openai_key,
                    log_conversations=self.log_conversations,
                    conversation_log_dir=self.conversation_log_dir
                ))
                logger.info("Added OpenAIBot")
            except ImportError:
                logger.warning("OpenAI API key found but openai package not installed")

        # Check Claude
        anthropic_key = settings.get_api_key('anthropic')
        if anthropic_key and self._test_claude_bot(anthropic_key):
            try:
                from reinforcetactics.game.llm_bot import ClaudeBot
                bots.append(BotDescriptor(
                    'ClaudeBot',
                    'llm',
                    bot_class=ClaudeBot,
                    api_key=anthropic_key,
                    log_conversations=self.log_conversations,
                    conversation_log_dir=self.conversation_log_dir
                ))
                logger.info("Added ClaudeBot")
            except ImportError:
                logger.warning("Anthropic API key found but anthropic package not installed")

        # Check Gemini
        google_key = settings.get_api_key('google')
        if google_key and self._test_gemini_bot(google_key):
            try:
                from reinforcetactics.game.llm_bot import GeminiBot
                bots.append(BotDescriptor(
                    'GeminiBot',
                    'llm',
                    bot_class=GeminiBot,
                    api_key=google_key,
                    log_conversations=self.log_conversations,
                    conversation_log_dir=self.conversation_log_dir
                ))
                logger.info("Added GeminiBot")
            except ImportError:
                logger.warning("Google API key found but google-generativeai package not installed")

        return bots

    def _test_openai_bot(self, api_key: str) -> bool:
        """Test if OpenAI API key is valid."""
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Try a minimal API call to test the key
            client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI API key test failed: {e}")
            return False

    def _test_claude_bot(self, api_key: str) -> bool:
        """Test if Anthropic API key is valid."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            # Try to create a client - if the key format is invalid, it will fail
            # We can't easily test without making a real API call
            return True
        except Exception as e:
            logger.warning(f"Anthropic API key test failed: {e}")
            return False

    def _test_gemini_bot(self, api_key: str) -> bool:
        """Test if Google API key is valid."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Try to list models to test the key
            list(genai.list_models())
            return True
        except Exception as e:
            logger.warning(f"Google API key test failed: {e}")
            return False

    def _discover_model_bots(self, models_dir: str) -> List[BotDescriptor]:
        """
        Discover trained model bots from a directory.

        Args:
            models_dir: Directory containing .zip model files

        Returns:
            List of model BotDescriptors
        """
        bots = []
        models_path = Path(models_dir)

        if not models_path.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return bots

        # Find all .zip files
        model_files = list(models_path.glob('*.zip'))

        for model_file in model_files:
            # Try to load the model to verify it's compatible
            if self._test_model_bot(model_file):
                bot_name = f"Model_{model_file.stem}"
                bots.append(BotDescriptor(
                    bot_name,
                    'model',
                    model_path=str(model_file)
                ))
                logger.info(f"Added model bot: {bot_name}")

        return bots

    def _test_model_bot(self, model_path: Path) -> bool:
        """
        Test if a model file is valid and compatible.

        Args:
            model_path: Path to model .zip file

        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            from reinforcetactics.game.model_bot import ModelBot
            # Create a dummy game state for testing
            map_data = FileIO.load_map(self.map_file)
            dummy_state = GameState(map_data, num_players=2)
            # Try to create the bot - this will load the model
            bot = ModelBot(dummy_state, player=2, model_path=str(model_path))
            return bot.model is not None
        except Exception as e:
            logger.warning(f"Failed to load model {model_path.name}: {e}")
            return False

    def _select_map(self, matchup_idx: int, game_num: int) -> str:
        """
        Select a map based on the configured map pool mode.

        Args:
            matchup_idx: Index of current matchup
            game_num: Game number within matchup

        Returns:
            Path to selected map file
        """
        if len(self.maps) == 1:
            return self.maps[0]

        if self.map_pool_mode == 'cycle':
            # Cycle through maps
            map_file = self.maps[self.current_map_index % len(self.maps)]
            self.current_map_index += 1
            return map_file
        elif self.map_pool_mode == 'random':
            # Random selection
            return random.choice(self.maps)
        elif self.map_pool_mode == 'all':
            # This is handled differently in _run_matchup
            # For now, cycle through (will be overridden)
            return self.maps[(game_num - 1) % len(self.maps)]
        else:
            raise ValueError(f"Unknown map_pool_mode: {self.map_pool_mode}")

    def run_tournament(self, bots: List[BotDescriptor]) -> Dict[str, Any]:
        """
        Run a round-robin tournament.

        Args:
            bots: List of BotDescriptors to compete

        Returns:
            Tournament results dictionary
        """
        if len(bots) < 2:
            logger.error("Need at least 2 bots for a tournament")
            return {}

        # Initialize Elo ratings for all bots
        for bot in bots:
            self.elo_system.initialize_bot(bot.name)

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting Tournament with {len(bots)} bots")
        if len(self.maps) == 1:
            logger.info(f"Map: {self.maps[0]}")
        else:
            logger.info(f"Maps: {len(self.maps)} maps")
            for map_file in self.maps:
                logger.info(f"  - {map_file}")
            logger.info(f"Map Pool Mode: {self.map_pool_mode}")
        
        # Calculate total games based on map pool mode
        if self.map_pool_mode == 'all' and len(self.maps) > 1:
            games_per_matchup = self.games_per_side * 2 * len(self.maps)
        else:
            games_per_matchup = self.games_per_side * 2
        
        logger.info(f"Games per matchup: {games_per_matchup}")
        if self.log_conversations:
            logger.info(f"LLM Conversation Logging: ENABLED")
            logger.info(f"Conversation Log Directory: {self.conversation_log_dir}")
        logger.info(f"{'='*80}\n")

        # Generate all matchups (round-robin)
        matchups = []
        for i in range(len(bots)):
            for j in range(i + 1, len(bots)):
                matchups.append((bots[i], bots[j]))

        logger.info(f"Total matchups: {len(matchups)}")
        logger.info(f"Total games: {len(matchups) * games_per_matchup}\n")

        # Run all matchups
        for matchup_idx, (bot1_desc, bot2_desc) in enumerate(matchups, 1):
            logger.info(f"\n--- Matchup {matchup_idx}/{len(matchups)}: {bot1_desc.name} vs {bot2_desc.name} ---")
            self._run_matchup(bot1_desc, bot2_desc, matchup_idx)

        # Generate results
        results = self._generate_results()
        self._save_results(results)

        return results

    def _run_matchup(self, bot1_desc: BotDescriptor, bot2_desc: BotDescriptor, matchup_idx: int) -> None:
        """
        Run a single matchup between two bots.

        Args:
            bot1_desc: First bot descriptor
            bot2_desc: Second bot descriptor
            matchup_idx: Index of this matchup
        """
        matchup_results = {
            'bot1': bot1_desc.name,
            'bot2': bot2_desc.name,
            'games': []
        }

        # Determine game structure based on map pool mode
        if self.map_pool_mode == 'all' and len(self.maps) > 1:
            # Play all maps for each side
            game_counter = 0
            for map_file in self.maps:
                # Play games_per_side games with bot1 as player 1
                for game_num in range(self.games_per_side):
                    game_counter += 1
                    map_name = Path(map_file).name
                    logger.info(f"  Game {game_counter}: {bot1_desc.name} (P1) vs {bot2_desc.name} (P2) on {map_name}")
                    result = self._run_game(bot1_desc, bot2_desc, 1, 2, matchup_idx, game_counter, map_file)
                    matchup_results['games'].append(result)
                    self._update_results(bot1_desc.name, bot2_desc.name, result['winner'], map_file)
                
                # Play games_per_side games with bot2 as player 1
                for game_num in range(self.games_per_side):
                    game_counter += 1
                    map_name = Path(map_file).name
                    logger.info(f"  Game {game_counter}: {bot2_desc.name} (P1) vs {bot1_desc.name} (P2) on {map_name}")
                    result = self._run_game(bot2_desc, bot1_desc, 1, 2, matchup_idx, game_counter, map_file)
                    matchup_results['games'].append(result)
                    # Swap perspective for results
                    winner = 2 if result['winner'] == 1 else (1 if result['winner'] == 2 else 0)
                    self._update_results(bot1_desc.name, bot2_desc.name, winner, map_file)
        else:
            # Standard mode: cycle or random
            # Play games_per_side games with bot1 as player 1
            for game_num in range(self.games_per_side):
                map_file = self._select_map(matchup_idx, game_num + 1)
                map_name = Path(map_file).name
                logger.info(f"  Game {game_num + 1}/{self.games_per_side * 2}: "
                           f"{bot1_desc.name} (P1) vs {bot2_desc.name} (P2) on {map_name}")
                result = self._run_game(bot1_desc, bot2_desc, 1, 2, matchup_idx, game_num + 1, map_file)
                matchup_results['games'].append(result)
                self._update_results(bot1_desc.name, bot2_desc.name, result['winner'], map_file)

            # Play games_per_side games with bot2 as player 1
            for game_num in range(self.games_per_side):
                map_file = self._select_map(matchup_idx, game_num + self.games_per_side + 1)
                map_name = Path(map_file).name
                logger.info(f"  Game {game_num + self.games_per_side + 1}/{self.games_per_side * 2}: "
                           f"{bot2_desc.name} (P1) vs {bot1_desc.name} (P2) on {map_name}")
                result = self._run_game(bot2_desc, bot1_desc, 1, 2, matchup_idx,
                                       game_num + self.games_per_side + 1, map_file)
                matchup_results['games'].append(result)
                # Swap perspective for results
                winner = 2 if result['winner'] == 1 else (1 if result['winner'] == 2 else 0)
                self._update_results(bot1_desc.name, bot2_desc.name, winner, map_file)

        self.matchup_results.append(matchup_results)

    def _run_game(self, bot1_desc: BotDescriptor, bot2_desc: BotDescriptor,
                  player1: int, player2: int, matchup_idx: int, game_num: int,
                  map_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a single game between two bots.

        Args:
            bot1_desc: First bot (plays as player1)
            bot2_desc: Second bot (plays as player2)
            player1: Player number for bot1 (1 or 2)
            player2: Player number for bot2 (1 or 2)
            matchup_idx: Matchup index
            game_num: Game number within matchup
            map_file: Map file to use (if None, uses self.map_file)

        Returns:
            Game result dictionary
        """
        # Use provided map or default
        if map_file is None:
            map_file = self.map_file
        
        # Load map and create game state
        map_data = FileIO.load_map(map_file)
        game_state = GameState(map_data, num_players=2)

        # Create bot instances
        bot1 = bot1_desc.create_bot(game_state, player1)
        bot2 = bot2_desc.create_bot(game_state, player2)
        bots = {player1: bot1, player2: bot2}

        # Play the game
        max_turns = 500  # Safety limit
        turn_count = 0

        try:
            while not game_state.game_over and turn_count < max_turns:
                current_player = game_state.current_player
                current_bot = bots[current_player]

                # Bot takes turn
                current_bot.take_turn()

                turn_count += 1

                # Check for game over conditions
                if game_state.game_over:
                    break

            # Determine winner
            if game_state.game_over and game_state.winner:
                winner = game_state.winner
                winner_name = bot1_desc.name if winner == player1 else bot2_desc.name
            elif turn_count >= max_turns:
                # Draw due to turn limit
                winner = 0
                winner_name = "Draw"
            else:
                winner = 0
                winner_name = "Draw"

            logger.info(f"    Result: {winner_name} (turns: {turn_count})")

            # Track game length for map statistics
            map_name = Path(map_file).name
            self.map_game_lengths[map_name].append(turn_count)

            # Save replay with map name in filename
            map_basename = Path(map_file).stem
            replay_filename = f"matchup{matchup_idx:03d}_game{game_num:02d}_{map_basename}_{bot1_desc.name}_vs_{bot2_desc.name}.json"
            replay_path = self.replays_dir / replay_filename

            game_info = {
                'bot1': bot1_desc.name,
                'bot2': bot2_desc.name,
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
                str(replay_path)
            )

            return {
                'winner': winner,
                'winner_name': winner_name,
                'turns': turn_count,
                'map': map_file,
                'replay': str(replay_path)
            }

        except Exception as e:
            logger.error(f"Error during game: {e}", exc_info=True)
            return {
                'winner': 0,
                'winner_name': "Error",
                'turns': turn_count,
                'error': str(e)
            }

    def _update_results(self, bot1_name: str, bot2_name: str, winner: int, map_file: str) -> None:
        """
        Update tournament results after a game.

        Args:
            bot1_name: Name of first bot
            bot2_name: Name of second bot
            winner: Winner (1=bot1, 2=bot2, 0=draw)
            map_file: Map file used for the game
        """
        # Update overall results
        if winner == 1:
            self.results[bot1_name]['wins'] += 1
            self.results[bot2_name]['losses'] += 1
        elif winner == 2:
            self.results[bot1_name]['losses'] += 1
            self.results[bot2_name]['wins'] += 1
        else:  # Draw
            self.results[bot1_name]['draws'] += 1
            self.results[bot2_name]['draws'] += 1

        # Update per-map statistics
        map_name = Path(map_file).name
        if winner == 1:
            self.per_map_stats[bot1_name][map_name]['wins'] += 1
            self.per_map_stats[bot2_name][map_name]['losses'] += 1
        elif winner == 2:
            self.per_map_stats[bot1_name][map_name]['losses'] += 1
            self.per_map_stats[bot2_name][map_name]['wins'] += 1
        else:  # Draw
            self.per_map_stats[bot1_name][map_name]['draws'] += 1
            self.per_map_stats[bot2_name][map_name]['draws'] += 1

        # Update Elo ratings
        self.elo_system.update_ratings(bot1_name, bot2_name, winner)

    def _generate_results(self) -> Dict[str, Any]:
        """
        Generate final tournament results.

        Returns:
            Results dictionary with rankings and statistics
        """
        # Calculate win rates and create rankings
        rankings = []
        for bot_name, stats in self.results.items():
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = stats['wins'] / total_games if total_games > 0 else 0.0

            # Add Elo ratings and per-map stats
            elo = self.elo_system.get_rating(bot_name)
            elo_change = self.elo_system.get_rating_change(bot_name)
            
            # Convert per-map stats to regular dict
            per_map_stats = {}
            if bot_name in self.per_map_stats:
                per_map_stats = {map_name: dict(map_stats) 
                                for map_name, map_stats in self.per_map_stats[bot_name].items()}

            rankings.append({
                'bot': bot_name,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'draws': stats['draws'],
                'total_games': total_games,
                'win_rate': win_rate,
                'elo': round(elo, 0),
                'elo_change': round(elo_change, 0),
                'per_map_stats': per_map_stats
            })

        # Sort by Elo rating (descending)
        rankings.sort(key=lambda x: x['elo'], reverse=True)

        # Calculate per-map performance summary
        per_map_summary = {}
        for map_file in self.maps:
            map_name = Path(map_file).name
            map_winners = defaultdict(int)
            
            # Count wins per bot on this map
            for bot_name in self.per_map_stats:
                if map_name in self.per_map_stats[bot_name]:
                    wins = self.per_map_stats[bot_name][map_name]['wins']
                    map_winners[bot_name] = wins
            
            # Find best performer
            best_bot = max(map_winners.items(), key=lambda x: x[1])[0] if map_winners else 'N/A'
            
            # Calculate average game length
            avg_length = sum(self.map_game_lengths[map_name]) / len(self.map_game_lengths[map_name]) if self.map_game_lengths[map_name] else 0
            
            per_map_summary[map_name] = {
                'best_performer': best_bot,
                'avg_game_length': round(avg_length, 1)
            }

        return {
            'timestamp': datetime.now().isoformat(),
            'maps_used': self.maps,
            'map_pool_mode': self.map_pool_mode,
            'games_per_side': self.games_per_side,
            'rankings': rankings,
            'elo_history': {bot_name: [round(r, 0) for r in history] 
                          for bot_name, history in self.elo_system.rating_history.items()},
            'per_map_summary': per_map_summary,
            'matchups': self.matchup_results
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save tournament results to files.

        Args:
            results: Results dictionary
        """
        # Save as JSON
        json_path = self.output_dir / 'tournament_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✅ Results saved to: {json_path}")

        # Save as CSV
        csv_path = self.output_dir / 'tournament_results.csv'
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Bot,Wins,Losses,Draws,Total Games,Win Rate,Elo,Elo Change\n")
            for ranking in results['rankings']:
                f.write(f"{ranking['bot']},{ranking['wins']},{ranking['losses']},"
                       f"{ranking['draws']},{ranking['total_games']},{ranking['win_rate']:.3f},"
                       f"{ranking['elo']:.0f},{ranking['elo_change']:+.0f}\n")
        logger.info(f"✅ Results saved to: {csv_path}")

        # Print summary
        logger.info(f"\n{'='*84}")
        logger.info("TOURNAMENT RESULTS")
        logger.info(f"{'='*84}")
        logger.info(f"{'Rank':<6}{'Bot':<20}{'Wins':<8}{'Losses':<8}{'Draws':<8}{'Win Rate':<10}{'Elo':<8}{'Δ Elo':<8}")
        logger.info(f"{'-'*84}")
        for rank, ranking in enumerate(results['rankings'], 1):
            elo_change_str = f"{ranking['elo_change']:+.0f}"
            logger.info(f"{rank:<6}{ranking['bot']:<20}{ranking['wins']:<8}"
                       f"{ranking['losses']:<8}{ranking['draws']:<8}"
                       f"{ranking['win_rate']:.1%}{'':2}{ranking['elo']:<8.0f}{elo_change_str:<8}")
        logger.info(f"{'='*84}")

        # Print per-map performance if multiple maps were used
        if len(self.maps) > 1 and 'per_map_summary' in results:
            logger.info(f"\nPer-Map Performance:")
            logger.info(f"{'-'*84}")
            logger.info(f"{'Map':<30}{'Best Performer':<30}{'Avg Game Length':<24}")
            logger.info(f"{'-'*84}")
            for map_name, summary in results['per_map_summary'].items():
                logger.info(f"{map_name:<30}{summary['best_performer']:<30}{summary['avg_game_length']:.0f} turns")
            logger.info(f"{'='*84}\n")
        else:
            logger.info("")


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
        help='How to select maps: cycle (default), random, or all (play every map for each matchup)'
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

    args = parser.parse_args()

    # Handle map arguments with clear precedence
    maps = []
    if args.maps:
        # --maps takes precedence; use explicitly provided maps
        maps = args.maps
    elif args.map_dir:
        # Load all maps from directory
        map_dir = Path(args.map_dir)
        if not map_dir.exists():
            logger.error(f"Map directory not found: {args.map_dir}")
            sys.exit(1)
        maps = sorted([str(f) for f in map_dir.glob('*.csv')])
        if not maps:
            logger.error(f"No .csv map files found in: {args.map_dir}")
            sys.exit(1)
    elif args.map:
        # Use single map (backward compatibility)
        if not Path(args.map).exists():
            logger.error(f"Map file not found: {args.map}")
            sys.exit(1)
        maps = [args.map]
    else:
        # Use default map
        maps = ['maps/1v1/6x6_beginner.csv']

    # Validate all map files exist
    for map_file in maps:
        if not Path(map_file).exists():
            logger.error(f"Map file not found: {map_file}")
            sys.exit(1)

    # Create tournament runner
    runner = TournamentRunner(
        maps=maps,
        output_dir=args.output_dir,
        games_per_side=args.games_per_side,
        log_conversations=args.log_conversations,
        conversation_log_dir=args.conversation_log_dir,
        map_pool_mode=args.map_pool_mode
    )

    # Discover bots
    bots = runner.discover_bots(models_dir=args.models_dir, include_test_bots=args.test)

    if len(bots) < 2:
        logger.error("Need at least 2 bots for a tournament. "
                    "Only SimpleBot was found. Add LLM API keys or model files.")
        sys.exit(1)

    # Run tournament
    try:
        runner.run_tournament(bots)
        logger.info("Tournament completed successfully!")
    except KeyboardInterrupt:
        logger.info("\nTournament interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Tournament failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
