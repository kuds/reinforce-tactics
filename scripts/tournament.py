#!/usr/bin/env python3
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
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        elif self.bot_type == 'llm':
            bot_class = self.kwargs['bot_class']
            api_key = self.kwargs.get('api_key')
            return bot_class(game_state, player, api_key=api_key)
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

    def __init__(self, map_file: str, output_dir: str, games_per_side: int = 2):
        """
        Initialize tournament runner.

        Args:
            map_file: Path to map file
            output_dir: Directory for results and replays
            games_per_side: Number of games per side (total games = 2 * games_per_side)
        """
        self.map_file = map_file
        self.output_dir = Path(output_dir)
        self.games_per_side = games_per_side
        self.results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        self.matchup_results = []

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

        # Always include SimpleBot
        bots.append(BotDescriptor('SimpleBot', 'simple'))
        logger.info("Added SimpleBot")

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
                    api_key=openai_key
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
                    api_key=anthropic_key
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
                    api_key=google_key
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

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Tournament with {len(bots)} bots")
        logger.info(f"Map: {self.map_file}")
        logger.info(f"Games per matchup: {self.games_per_side * 2}")
        logger.info(f"{'='*60}\n")

        # Generate all matchups (round-robin)
        matchups = []
        for i in range(len(bots)):
            for j in range(i + 1, len(bots)):
                matchups.append((bots[i], bots[j]))

        logger.info(f"Total matchups: {len(matchups)}")
        logger.info(f"Total games: {len(matchups) * self.games_per_side * 2}\n")

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

        # Play games_per_side games with bot1 as player 1
        for game_num in range(self.games_per_side):
            logger.info(f"  Game {game_num + 1}/{self.games_per_side * 2}: "
                       f"{bot1_desc.name} (P1) vs {bot2_desc.name} (P2)")
            result = self._run_game(bot1_desc, bot2_desc, 1, 2, matchup_idx, game_num + 1)
            matchup_results['games'].append(result)
            self._update_results(bot1_desc.name, bot2_desc.name, result['winner'])

        # Play games_per_side games with bot2 as player 1
        for game_num in range(self.games_per_side):
            logger.info(f"  Game {game_num + self.games_per_side + 1}/{self.games_per_side * 2}: "
                       f"{bot2_desc.name} (P1) vs {bot1_desc.name} (P2)")
            result = self._run_game(bot2_desc, bot1_desc, 1, 2, matchup_idx, 
                                   game_num + self.games_per_side + 1)
            matchup_results['games'].append(result)
            # Swap perspective for results
            self._update_results(bot1_desc.name, bot2_desc.name, 
                               2 if result['winner'] == 1 else (1 if result['winner'] == 2 else 0))

        self.matchup_results.append(matchup_results)

    def _run_game(self, bot1_desc: BotDescriptor, bot2_desc: BotDescriptor,
                  player1: int, player2: int, matchup_idx: int, game_num: int) -> Dict[str, Any]:
        """
        Run a single game between two bots.

        Args:
            bot1_desc: First bot (plays as player1)
            bot2_desc: Second bot (plays as player2)
            player1: Player number for bot1 (1 or 2)
            player2: Player number for bot2 (1 or 2)
            matchup_idx: Matchup index
            game_num: Game number within matchup

        Returns:
            Game result dictionary
        """
        # Load map and create game state
        map_data = FileIO.load_map(self.map_file)
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

            # Save replay
            replay_filename = f"matchup{matchup_idx:03d}_game{game_num:02d}_{bot1_desc.name}_vs_{bot2_desc.name}.json"
            replay_path = self.replays_dir / replay_filename

            game_info = {
                'bot1': bot1_desc.name,
                'bot2': bot2_desc.name,
                'bot1_player': player1,
                'bot2_player': player2,
                'winner': winner,
                'winner_name': winner_name,
                'turns': turn_count,
                'map': self.map_file
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

    def _update_results(self, bot1_name: str, bot2_name: str, winner: int) -> None:
        """
        Update tournament results after a game.

        Args:
            bot1_name: Name of first bot
            bot2_name: Name of second bot
            winner: Winner (1=bot1, 2=bot2, 0=draw)
        """
        if winner == 1:
            self.results[bot1_name]['wins'] += 1
            self.results[bot2_name]['losses'] += 1
        elif winner == 2:
            self.results[bot1_name]['losses'] += 1
            self.results[bot2_name]['wins'] += 1
        else:  # Draw
            self.results[bot1_name]['draws'] += 1
            self.results[bot2_name]['draws'] += 1

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

            rankings.append({
                'bot': bot_name,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'draws': stats['draws'],
                'total_games': total_games,
                'win_rate': win_rate
            })

        # Sort by wins (descending), then win_rate
        rankings.sort(key=lambda x: (x['wins'], x['win_rate']), reverse=True)

        return {
            'timestamp': datetime.now().isoformat(),
            'map': self.map_file,
            'games_per_side': self.games_per_side,
            'rankings': rankings,
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
            f.write("Bot,Wins,Losses,Draws,Total Games,Win Rate\n")
            for ranking in results['rankings']:
                f.write(f"{ranking['bot']},{ranking['wins']},{ranking['losses']},"
                       f"{ranking['draws']},{ranking['total_games']},{ranking['win_rate']:.3f}\n")
        logger.info(f"✅ Results saved to: {csv_path}")

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("TOURNAMENT RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"{'Rank':<6}{'Bot':<20}{'Wins':<8}{'Losses':<8}{'Draws':<8}{'Win Rate':<10}")
        logger.info(f"{'-'*60}")
        for rank, ranking in enumerate(results['rankings'], 1):
            logger.info(f"{rank:<6}{ranking['bot']:<20}{ranking['wins']:<8}"
                       f"{ranking['losses']:<8}{ranking['draws']:<8}"
                       f"{ranking['win_rate']:.3f}")
        logger.info(f"{'='*60}\n")


def main():
    """Main entry point for tournament script."""
    parser = argparse.ArgumentParser(
        description='Run a round-robin tournament between Reinforce Tactics bots'
    )
    parser.add_argument(
        '--map',
        default='maps/1v1/6x6_beginner.csv',
        help='Path to map file (default: maps/1v1/6x6_beginner.csv)'
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

    args = parser.parse_args()

    # Validate map file exists
    if not Path(args.map).exists():
        logger.error(f"Map file not found: {args.map}")
        sys.exit(1)

    # Create tournament runner
    runner = TournamentRunner(
        map_file=args.map,
        output_dir=args.output_dir,
        games_per_side=args.games_per_side
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
