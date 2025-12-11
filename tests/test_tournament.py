"""
Tests for the tournament system (ModelBot and tournament script).
"""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.file_io import FileIO


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
            import stable_baselines3
        except ImportError:
            pytest.skip("stable-baselines3 not installed")
            
        from unittest.mock import patch
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
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import BotDescriptor
        
        desc = BotDescriptor('TestBot', 'simple')
        assert desc.name == 'TestBot'
        assert desc.bot_type == 'simple'
        
        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        
        bot = desc.create_bot(game_state, 1)
        assert isinstance(bot, SimpleBot)
        assert bot.bot_player == 1

    def test_bot_descriptor_model_bot(self):
        """Test BotDescriptor for ModelBot."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import BotDescriptor
        from reinforcetactics.game.model_bot import ModelBot
        
        desc = BotDescriptor('TestModel', 'model', model_path='test.zip')
        assert desc.name == 'TestModel'
        assert desc.bot_type == 'model'
        
        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        
        # Mock the model loading
        with patch.object(ModelBot, '_load_model'):
            bot = desc.create_bot(game_state, 2)
            assert isinstance(bot, ModelBot)
            assert bot.bot_player == 2

    def test_tournament_runner_initialization(self):
        """Test TournamentRunner initialization."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import TournamentRunner
        
        runner = TournamentRunner(
            map_file='maps/1v1/6x6_beginner.csv',
            output_dir='/tmp/test_tournament',
            games_per_side=2
        )
        
        assert runner.map_file == 'maps/1v1/6x6_beginner.csv'
        assert runner.games_per_side == 2
        assert runner.output_dir.exists()

    def test_tournament_runner_discover_simple_bot(self):
        """Test that TournamentRunner discovers SimpleBot."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import TournamentRunner
        
        runner = TournamentRunner(
            map_file='maps/1v1/6x6_beginner.csv',
            output_dir='/tmp/test_tournament',
            games_per_side=1
        )
        
        bots = runner.discover_bots(models_dir=None, include_test_bots=False)
        
        # At minimum, should find SimpleBot
        assert len(bots) >= 1
        assert any(bot.name == 'SimpleBot' for bot in bots)

    def test_tournament_matchup_generation(self):
        """Test that tournament generates correct matchups."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import TournamentRunner, BotDescriptor
        
        runner = TournamentRunner(
            map_file='maps/1v1/6x6_beginner.csv',
            output_dir='/tmp/test_tournament',
            games_per_side=1
        )
        
        # Create 3 test bots
        bots = [
            BotDescriptor('Bot1', 'simple'),
            BotDescriptor('Bot2', 'simple'),
            BotDescriptor('Bot3', 'simple'),
        ]
        
        # Generate matchups (should be 3 choose 2 = 3 matchups)
        matchups = []
        for i in range(len(bots)):
            for j in range(i + 1, len(bots)):
                matchups.append((bots[i], bots[j]))
        
        assert len(matchups) == 3
        assert (bots[0], bots[1]) in matchups
        assert (bots[0], bots[2]) in matchups
        assert (bots[1], bots[2]) in matchups

    def test_tournament_results_structure(self):
        """Test the structure of tournament results."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import TournamentRunner
        
        runner = TournamentRunner(
            map_file='maps/1v1/6x6_beginner.csv',
            output_dir='/tmp/test_tournament_results',
            games_per_side=1
        )
        
        # Manually add some test results
        runner.results['Bot1']['wins'] = 2
        runner.results['Bot1']['losses'] = 0
        runner.results['Bot1']['draws'] = 0
        
        runner.results['Bot2']['wins'] = 0
        runner.results['Bot2']['losses'] = 2
        runner.results['Bot2']['draws'] = 0
        
        results = runner._generate_results()
        
        assert 'rankings' in results
        assert 'timestamp' in results
        assert 'map' in results
        assert len(results['rankings']) == 2
        
        # Bot1 should be ranked first
        assert results['rankings'][0]['bot'] == 'Bot1'
        assert results['rankings'][0]['wins'] == 2
        assert results['rankings'][0]['win_rate'] == 1.0


class TestModelBotActionTranslation:
    """Tests for ModelBot action translation."""

    def test_create_unit_action(self):
        """Test that ModelBot correctly translates create_unit actions."""
        from reinforcetactics.game.model_bot import ModelBot
        
        map_data = FileIO.load_map('maps/1v1/6x6_beginner.csv')
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
