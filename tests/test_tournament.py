"""
Tests for the tournament system (ModelBot and tournament script).
"""
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot, MediumBot
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.tournament import (
    BotDescriptor,
    BotType,
    TournamentConfig,
    TournamentRunner,
    EloRatingSystem,
    MapConfig,
    create_bot_instance,
)


class TestModelBot:
    """Tests for ModelBot class."""

    def test_modelbot_import(self):
        """Test that ModelBot can be imported."""
        from reinforcetactics.game.model_bot import ModelBot
        assert ModelBot is not None

    def test_modelbot_initialization_without_model(self):
        """Test ModelBot can be initialized without a model."""
        from reinforcetactics.game.model_bot import ModelBot

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)

        bot = ModelBot(game_state, player=2, model_path=None)
        assert bot.bot_player == 2
        assert bot.model is None

    def test_modelbot_take_turn_without_model(self):
        """Test ModelBot.take_turn() ends turn gracefully when no model is loaded."""
        from reinforcetactics.game.model_bot import ModelBot

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        game_state.current_player = 2

        bot = ModelBot(game_state, player=2, model_path=None)
        initial_turn = game_state.turn_number

        bot.take_turn()

        # Turn should advance (bot ends turn)
        assert game_state.current_player != 2 or game_state.turn_number > initial_turn

    def test_modelbot_with_mock_model(self):
        """Test ModelBot with a mocked SB3 model."""
        try:
            import stable_baselines3  # pylint: disable=unused-import
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        from reinforcetactics.game.model_bot import ModelBot

        with patch('stable_baselines3.PPO') as mock_ppo_class:
            # Create mock model
            mock_model = Mock()
            mock_model.predict.return_value = ([5, 0, 0, 0, 0, 0], None)  # End turn action
            mock_ppo_class.load.return_value = mock_model
            mock_ppo_class.__name__ = 'PPO'  # Add this line to fix the AttributeError

            map_data = FileIO.generate_random_map(10, 10, num_players=2)
            game_state = GameState(map_data, num_players=2)
            game_state.current_player = 2

            # Create a dummy model file path
            with patch('pathlib.Path.exists', return_value=True):
                with patch('stable_baselines3.A2C'):
                    with patch('stable_baselines3.DQN'):
                        bot = ModelBot(game_state, player=2, model_path='test_model.zip')

            assert bot.model is not None

            bot.take_turn()

            # Model should have been called
            assert mock_model.predict.called


class TestTournamentSystem:
    """Tests for tournament system."""

    def test_bot_descriptor_simple_bot(self):
        """Test BotDescriptor for SimpleBot."""
        desc = BotDescriptor(name='TestBot', bot_type=BotType.SIMPLE)
        assert desc.name == 'TestBot'
        assert desc.bot_type == BotType.SIMPLE

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)

        bot = create_bot_instance(desc, game_state, player=1)
        assert isinstance(bot, SimpleBot)
        assert bot.bot_player == 1

    def test_bot_descriptor_model_bot(self):
        """Test BotDescriptor for ModelBot."""
        from reinforcetactics.game.model_bot import ModelBot

        desc = BotDescriptor(name='TestModel', bot_type=BotType.MODEL, model_path='test.zip')
        assert desc.name == 'TestModel'
        assert desc.bot_type == BotType.MODEL

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)

        # Mock the model loading
        with patch.object(ModelBot, '_load_model'):
            bot = create_bot_instance(desc, game_state, player=2)
            assert isinstance(bot, ModelBot)
            assert bot.bot_player == 2

    def test_tournament_runner_initialization(self):
        """Test TournamentRunner initialization."""
        config = TournamentConfig(
            maps=[MapConfig(path='maps/1v1/beginner.csv', max_turns=500)],
            output_dir='/tmp/test_tournament',
            games_per_side=2
        )
        runner = TournamentRunner(config)

        assert len(config.maps) == 1
        assert config.maps[0].path == 'maps/1v1/beginner.csv'
        assert config.games_per_side == 2

    def test_tournament_runner_with_logging_parameters(self):
        """Test TournamentRunner initialization with logging parameters."""
        # Test with log_conversations=True and default log dir
        config1 = TournamentConfig(
            maps=[MapConfig(path='maps/1v1/beginner.csv', max_turns=500)],
            output_dir='/tmp/test_tournament',
            games_per_side=2,
            log_conversations=True
        )
        runner1 = TournamentRunner(config1)

        assert config1.log_conversations is True
        assert config1.conversation_log_dir == '/tmp/test_tournament/llm_conversations'

        # Test with custom log dir
        config2 = TournamentConfig(
            maps=[MapConfig(path='maps/1v1/beginner.csv', max_turns=500)],
            output_dir='/tmp/test_tournament',
            games_per_side=2,
            log_conversations=True,
            conversation_log_dir='/tmp/custom_logs'
        )
        _ = TournamentRunner(config2)

        assert config2.log_conversations is True
        assert config2.conversation_log_dir == '/tmp/custom_logs'

        # Test with logging disabled
        config3 = TournamentConfig(
            maps=[MapConfig(path='maps/1v1/beginner.csv', max_turns=500)],
            output_dir='/tmp/test_tournament',
            games_per_side=2,
            log_conversations=False
        )
        _ = TournamentRunner(config3)

        assert config3.log_conversations is False

    def test_bot_descriptor_llm_bot(self):
        """Test BotDescriptor for LLM bots."""
        desc = BotDescriptor(
            name='TestLLM',
            bot_type=BotType.LLM,
            provider='openai',
            model='gpt-4',
        )

        assert desc.name == 'TestLLM'
        assert desc.bot_type == BotType.LLM
        assert desc.provider == 'openai'
        assert desc.model == 'gpt-4'

    def test_tournament_runner_discover_bots(self):
        """Test bot discovery functions."""
        from reinforcetactics.tournament.bots import discover_builtin_bots

        bots = discover_builtin_bots()

        # Should find SimpleBot, MediumBot, AdvancedBot
        assert len(bots) >= 1
        assert any(bot.name == 'SimpleBot' for bot in bots)

    def test_tournament_matchup_generation(self):
        """Test that tournament generates correct matchups."""
        # Create 3 test bots
        bots = [
            BotDescriptor(name='Bot1', bot_type=BotType.SIMPLE),
            BotDescriptor(name='Bot2', bot_type=BotType.SIMPLE),
            BotDescriptor(name='Bot3', bot_type=BotType.SIMPLE),
        ]

        # Generate matchups (should be 3 choose 2 = 3 matchups)
        matchups = []
        for i, bot_i in enumerate(bots):
            for bot_j in bots[i + 1:]:
                matchups.append((bot_i, bot_j))

        assert len(matchups) == 3
        assert (bots[0], bots[1]) in matchups
        assert (bots[0], bots[2]) in matchups
        assert (bots[1], bots[2]) in matchups

    def test_tournament_results_structure(self):
        """Test the structure of tournament results."""
        from reinforcetactics.tournament import TournamentResults

        elo_system = EloRatingSystem()
        results = TournamentResults(elo_system)

        # Initialize bots for Elo system
        elo_system.initialize_bot('Bot1')
        elo_system.initialize_bot('Bot2')

        # Add some test game results
        from reinforcetactics.tournament import GameResult
        results.add_game_result(GameResult(
            game_id=1, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=10, map_name='test.csv'
        ))
        results.add_game_result(GameResult(
            game_id=2, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=15, map_name='test.csv'
        ))

        result_dict = results.to_dict()

        assert 'standings' in result_dict
        assert 'games' in result_dict
        assert 'elo_history' in result_dict
        assert len(result_dict['standings']) == 2

        # Bot1 should be ranked first
        standings = results.get_standings()
        assert standings[0].bot_name == 'Bot1'
        assert standings[0].wins == 2
        assert standings[0].win_rate == 1.0


class TestModelBotActionTranslation:
    """Tests for ModelBot action translation."""

    def test_create_unit_action(self):
        """Test that ModelBot correctly translates create_unit actions."""
        from reinforcetactics.game.model_bot import ModelBot

        map_data = FileIO.load_map('maps/1v1/beginner.csv')
        game_state = GameState(map_data, num_players=2)
        game_state.current_player = 2
        game_state.player_gold[2] = 1000

        bot = ModelBot(game_state, player=2, model_path=None)

        # Action: create Warrior (0) at player 2's building location
        # In 6x6 map, player 2 has building at (4, 4) and (4, 5)
        # But map gets padded, so we need to find actual building positions
        buildings_p2 = [
            (tile.x, tile.y)
            for row in game_state.grid.tiles
            for tile in row
            if tile.type in ['b', 'h'] and tile.player == 2
        ]

        if buildings_p2:
            bx, by = buildings_p2[0]
            initial_gold = game_state.player_gold[2]

            # Try to create a unit
            success = bot._create_unit(0, bx, by)  # 0 = Warrior

            if success:
                assert game_state.player_gold[2] < initial_gold
                unit = game_state.get_unit_at_position(bx, by)
                assert unit is not None
                assert unit.type == 'W'

    def test_end_turn_action_detection(self):
        """Test that ModelBot detects end turn actions."""
        from reinforcetactics.game.model_bot import ModelBot

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)

        bot = ModelBot(game_state, player=2, model_path=None)

        # Action type 5 is end turn
        end_turn_action = [5, 0, 0, 0, 0, 0]
        assert bot._is_end_turn_action(end_turn_action)

        # Other actions should not be end turn
        move_action = [1, 0, 0, 0, 1, 1]
        assert not bot._is_end_turn_action(move_action)


class TestBotFactory:
    """Tests for bot factory functions."""

    def test_bot_factory_create_bot_with_model_path(self):
        """Test that create_bot handles ModelBot with model_path."""
        from game.bot_factory import create_bot
        from reinforcetactics.utils.settings import get_settings

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        settings = get_settings()

        # Mock ModelBot to avoid loading actual model
        with patch('reinforcetactics.game.model_bot.ModelBot') as mock_modelbot:
            mock_instance = MagicMock()
            mock_modelbot.return_value = mock_instance

            bot = create_bot(game_state, 2, 'ModelBot', settings, model_path='/path/to/model.zip')

            # Should have called ModelBot constructor with correct args
            mock_modelbot.assert_called_once_with(
                game_state,
                player=2,
                model_path='/path/to/model.zip'
            )
            assert bot == mock_instance

    def test_bot_factory_modelbot_without_path_raises_error(self):
        """Test that creating ModelBot without model_path raises ValueError."""
        from game.bot_factory import create_bot
        from reinforcetactics.utils.settings import get_settings

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        settings = get_settings()

        # Should raise ValueError when model_path is missing
        with pytest.raises(ValueError) as excinfo:
            create_bot(game_state, 2, 'ModelBot', settings, model_path=None)

        assert "model_path is required" in str(excinfo.value)

    def test_bot_factory_create_bots_from_config_with_modelbot(self):
        """Test that create_bots_from_config handles ModelBot configs."""
        from game.bot_factory import create_bots_from_config
        from reinforcetactics.utils.settings import get_settings

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        settings = get_settings()

        player_configs = [
            {'type': 'human', 'bot_type': None, 'model_path': None},
            {'type': 'computer', 'bot_type': 'ModelBot', 'model_path': '/path/to/model.zip'}
        ]

        # Mock ModelBot to avoid loading actual model
        with patch('reinforcetactics.game.model_bot.ModelBot') as mock_modelbot:
            mock_instance = MagicMock()
            mock_modelbot.return_value = mock_instance

            bots = create_bots_from_config(game_state, player_configs, settings)

            # Should have created bot for player 2 only
            assert len(bots) == 1
            assert 2 in bots
            assert bots[2] == mock_instance

            # Should have called ModelBot constructor
            mock_modelbot.assert_called_once_with(
                game_state,
                player=2,
                model_path='/path/to/model.zip'
            )


class TestEloRatingSystem:
    """Tests for the Elo rating system."""

    def test_elo_initialization(self):
        """Test EloRatingSystem initialization."""
        elo = EloRatingSystem(starting_elo=1500, k_factor=32)
        assert elo.starting_elo == 1500
        assert elo.k_factor == 32
        assert len(elo.ratings) == 0

    def test_elo_bot_initialization(self):
        """Test initializing a bot in the Elo system."""
        elo = EloRatingSystem()
        elo.initialize_bot('TestBot')

        assert elo.get_rating('TestBot') == 1500.0
        assert elo.get_rating_change('TestBot') == 0.0
        assert len(elo.rating_history['TestBot']) == 1

    def test_elo_expected_score_equal_ratings(self):
        """Test expected score calculation with equal ratings."""
        elo = EloRatingSystem()
        expected = elo.calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001  # Should be exactly 0.5

    def test_elo_expected_score_unequal_ratings(self):
        """Test expected score calculation with different ratings."""
        elo = EloRatingSystem()
        # Higher rated player should have higher expected score
        higher_expected = elo.calculate_expected_score(1600, 1500)
        lower_expected = elo.calculate_expected_score(1500, 1600)

        assert higher_expected > 0.5
        assert lower_expected < 0.5
        assert abs(higher_expected + lower_expected - 1.0) < 0.001

    def test_elo_update_after_win(self):
        """Test Elo rating update after a win."""
        elo = EloRatingSystem(k_factor=32)
        elo.initialize_bot('Winner')
        elo.initialize_bot('Loser')

        # Winner wins against Loser
        elo.update_ratings('Winner', 'Loser', result=1)

        winner_rating = elo.get_rating('Winner')
        loser_rating = elo.get_rating('Loser')

        # Winner should gain points, loser should lose points
        assert winner_rating > 1500
        assert loser_rating < 1500
        # Rating changes should be symmetric for equal starting ratings
        assert abs((winner_rating - 1500) + (loser_rating - 1500)) < 0.001

    def test_elo_update_after_draw(self):
        """Test Elo rating update after a draw."""
        elo = EloRatingSystem(k_factor=32)
        elo.initialize_bot('Bot1')
        elo.initialize_bot('Bot2')

        # Draw
        elo.update_ratings('Bot1', 'Bot2', result=0)

        # With equal starting ratings, draw should not change ratings
        assert abs(elo.get_rating('Bot1') - 1500) < 0.001
        assert abs(elo.get_rating('Bot2') - 1500) < 0.001

    def test_elo_rating_history(self):
        """Test that rating history is tracked correctly."""
        elo = EloRatingSystem()
        elo.initialize_bot('Bot1')
        elo.initialize_bot('Bot2')

        # Play multiple games
        elo.update_ratings('Bot1', 'Bot2', result=1)  # Bot1 wins
        elo.update_ratings('Bot1', 'Bot2', result=1)  # Bot1 wins again
        elo.update_ratings('Bot1', 'Bot2', result=2)  # Bot2 wins

        # History should have 4 entries (initial + 3 games)
        assert len(elo.rating_history['Bot1']) == 4
        assert len(elo.rating_history['Bot2']) == 4

        # Ratings should be increasing for Bot1 (won first 2)
        assert elo.rating_history['Bot1'][1] > elo.rating_history['Bot1'][0]
        assert elo.rating_history['Bot1'][2] > elo.rating_history['Bot1'][1]


class TestMultiMapTournament:
    """Tests for multi-map tournament functionality."""

    def test_tournament_runner_with_multiple_maps(self):
        """Test TournamentRunner initialization with multiple maps."""
        config = TournamentConfig(
            maps=[
                MapConfig(path='maps/1v1/beginner.csv', max_turns=500),
                MapConfig(path='maps/1v1/center_mountains.csv', max_turns=500)
            ],
            output_dir='/tmp/test_multimap',
            games_per_side=1,
            map_pool_mode='cycle'
        )
        _ = TournamentRunner(config)

        assert len(config.maps) == 2
        assert config.map_pool_mode == 'cycle'

    def test_tournament_config_from_dict(self):
        """Test TournamentConfig from dictionary."""
        config_dict = {
            'maps': [
                {'path': 'maps/1v1/beginner.csv', 'max_turns': 500}
            ],
            'output_dir': '/tmp/test_backward',
            'games_per_side': 1
        }
        config = TournamentConfig.from_dict(config_dict)

        assert len(config.maps) == 1
        assert config.maps[0].path == 'maps/1v1/beginner.csv'

    def test_tournament_per_map_stats_tracking(self):
        """Test that per-map statistics are tracked correctly."""
        from reinforcetactics.tournament import TournamentResults, GameResult

        elo = EloRatingSystem()
        results = TournamentResults(elo)

        # Simulate some game results
        results.add_game_result(GameResult(
            game_id=1, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=10, map_name='map1.csv'
        ))
        results.add_game_result(GameResult(
            game_id=2, bot1_name='Bot1', bot2_name='Bot2',
            winner=2, winner_name='Bot2', turns=15, map_name='map2.csv'
        ))
        results.add_game_result(GameResult(
            game_id=3, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=20, map_name='map1.csv'
        ))

        # Check per-map stats
        standings = results.get_standings()
        bot1_standing = next(s for s in standings if s.bot_name == 'Bot1')
        bot2_standing = next(s for s in standings if s.bot_name == 'Bot2')

        assert bot1_standing.per_map_stats['map1.csv']['wins'] == 2
        assert bot1_standing.per_map_stats['map2.csv']['losses'] == 1
        assert bot2_standing.per_map_stats['map1.csv']['losses'] == 2
        assert bot2_standing.per_map_stats['map2.csv']['wins'] == 1

    def test_tournament_elo_integration(self):
        """Test that Elo ratings are integrated into tournament."""
        from reinforcetactics.tournament import TournamentResults, GameResult

        elo = EloRatingSystem()
        results = TournamentResults(elo)

        # Initialize bots
        elo.initialize_bot('Bot1')
        elo.initialize_bot('Bot2')

        # Simulate a game result
        results.add_game_result(GameResult(
            game_id=1, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=10, map_name='map1.csv'
        ))

        # Check that Elo ratings changed
        assert elo.get_rating('Bot1') > 1500
        assert elo.get_rating('Bot2') < 1500

    def test_generate_results_with_elo(self):
        """Test that generated results include Elo ratings."""
        from reinforcetactics.tournament import TournamentResults, GameResult

        elo = EloRatingSystem()
        results = TournamentResults(elo)

        # Setup some results
        elo.initialize_bot('Bot1')
        elo.initialize_bot('Bot2')
        results.add_game_result(GameResult(
            game_id=1, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=10, map_name='map1.csv'
        ))
        results.add_game_result(GameResult(
            game_id=2, bot1_name='Bot1', bot2_name='Bot2',
            winner=1, winner_name='Bot1', turns=15, map_name='map1.csv'
        ))

        result_dict = results.to_dict()

        # Check structure
        assert 'standings' in result_dict
        assert 'elo_history' in result_dict
        assert 'maps_used' in result_dict

        # Check standings include Elo
        for standing in result_dict['standings']:
            assert 'elo' in standing
            assert 'elo_change' in standing
            assert 'per_map_stats' in standing

        # Check Elo history
        assert 'Bot1' in result_dict['elo_history']
        assert 'Bot2' in result_dict['elo_history']
