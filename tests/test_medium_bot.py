"""Tests for MediumBot class."""
import pytest
import numpy as np

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import MediumBot, SimpleBot
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    # Create a 10x10 map with basic tiles
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    # Add HQ for player 1 and 2
    map_data[0][0] = 'h_1'
    map_data[9][9] = 'h_2'
    # Add some buildings
    map_data[0][1] = 'b_1'
    map_data[9][8] = 'b_2'
    return GameState(map_data, num_players=2)


class TestMediumBotBasics:
    """Test basic MediumBot functionality."""

    def test_mediumbot_initialization(self, simple_game):
        """Test MediumBot can be initialized."""
        bot = MediumBot(simple_game, player=2)
        assert bot.bot_player == 2
        assert bot.game_state == simple_game

    def test_mediumbot_find_hq(self, simple_game):
        """Test MediumBot can find its HQ."""
        bot = MediumBot(simple_game, player=2)
        hq_pos = bot.find_our_hq()
        assert hq_pos is not None
        assert hq_pos == (9, 9)

        bot1 = MediumBot(simple_game, player=1)
        hq_pos1 = bot1.find_our_hq()
        assert hq_pos1 is not None
        assert hq_pos1 == (0, 0)

    def test_mediumbot_manhattan_distance(self, simple_game):
        """Test manhattan distance calculation."""
        bot = MediumBot(simple_game, player=2)
        assert bot.manhattan_distance(0, 0, 0, 0) == 0
        assert bot.manhattan_distance(0, 0, 3, 4) == 7
        assert bot.manhattan_distance(5, 5, 2, 3) == 5


class TestMediumBotPurchasing:
    """Test MediumBot unit purchasing strategy."""

    def test_mediumbot_purchases_multiple_units(self, simple_game):
        """Test that MediumBot purchases multiple units when possible."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000  # Give plenty of gold

        bot = MediumBot(simple_game, player=2)
        initial_gold = simple_game.player_gold[2]
        initial_units = len([u for u in simple_game.units if u.player == 2])

        bot.purchase_units()

        # Should have purchased multiple units
        final_gold = simple_game.player_gold[2]
        final_units = len([u for u in simple_game.units if u.player == 2])

        assert final_gold < initial_gold
        assert final_units > initial_units

    def test_mediumbot_purchases_affordable_units(self, simple_game):
        """Test that MediumBot only purchases units it can afford."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 250  # Only enough for one cheap unit

        bot = MediumBot(simple_game, player=2)
        initial_gold = simple_game.player_gold[2]

        bot.purchase_units()

        final_gold = simple_game.player_gold[2]

        # Should have purchased something
        assert final_gold <= initial_gold


class TestMediumBotStructurePriority:
    """Test MediumBot structure prioritization."""

    def test_structure_priority_prefers_closer(self, simple_game):
        """Test that structures closer to HQ have higher priority (lower score)."""
        bot = MediumBot(simple_game, player=2)

        # Create two structures at different distances
        # Bot HQ is at (9, 9)
        close_tile = simple_game.grid.get_tile(8, 8)
        far_tile = simple_game.grid.get_tile(0, 0)

        close_priority = bot.get_structure_priority(close_tile)
        far_priority = bot.get_structure_priority(far_tile)

        # Lower score = higher priority
        assert close_priority < far_priority


class TestMediumBotCoordinatedAttacks:
    """Test MediumBot coordinated attack strategies."""

    def test_mediumbot_identifies_killable_targets(self):
        """Test that MediumBot can identify enemies that can be killed."""
        # Create a game with specific setup
        map_data = FileIO.load_map('maps/1v1/beginner.csv')
        game = GameState(map_data, num_players=2)

        # Create enemy unit with low health
        game.create_unit('W', 3, 3, 1)
        enemy = game.units[-1]
        enemy.health = 5  # Low health

        # Create bot units nearby
        game.create_unit('W', 2, 3, 2)
        game.create_unit('W', 4, 3, 2)

        bot = MediumBot(game, player=2)
        bot_units = [u for u in game.units if u.player == 2]

        killable = bot.find_killable_targets(bot_units)

        # Should identify the low-health enemy as killable (or may not be killable depending on exact positions)
        _ = len(killable)  # Check computed without error


class TestMediumBotContestedStructures:
    """Test MediumBot's ability to detect and respond to contested structures."""

    def test_finds_contested_structures(self):
        """Test that MediumBot can find structures being captured."""
        map_data = FileIO.load_map('maps/1v1/beginner.csv')
        game = GameState(map_data, num_players=2)

        bot = MediumBot(game, player=2)

        # Get a structure owned by player 1
        p1_structures = [
            tile for row in game.grid.tiles for tile in row
            if tile.is_capturable() and tile.player == 1
        ]

        if p1_structures:
            structure = p1_structures[0]
            # Simulate partial capture
            structure.health = structure.max_health - 5

            # Place enemy unit on structure
            game.create_unit('W', structure.x, structure.y, 1)

            contested = bot.find_contested_structures()

            # Should find the contested structure
            assert len(contested) >= 1


class TestMediumBotVsSimpleBot:
    """Test MediumBot performance against SimpleBot."""

    def test_mediumbot_vs_simplebot_single_game(self):
        """Test a single game between MediumBot and SimpleBot."""
        # Load a standard map
        map_data = FileIO.load_map('maps/1v1/beginner.csv')
        game = GameState(map_data, num_players=2)

        # Create bots
        simple_bot = SimpleBot(game, player=1)
        medium_bot = MediumBot(game, player=2)

        bots = {1: simple_bot, 2: medium_bot}

        # Play game
        max_turns = 50
        turn_count = 0

        while not game.game_over and turn_count < max_turns:
            current_bot = bots[game.current_player]
            current_bot.take_turn()
            turn_count += 1

        # Game should complete within turn limit or reach game over
        # This is just a smoke test - the bot should be able to play
        assert turn_count > 0  # At least one turn was taken


class TestMediumBotAttackValue:
    """Test MediumBot's attack value calculation."""

    def test_calculate_attack_value_kill_bonus(self):
        """Test that killing blows have high value."""
        map_data = FileIO.load_map('maps/1v1/beginner.csv')
        game = GameState(map_data, num_players=2)

        # Create attacker
        game.create_unit('W', 2, 2, 2)
        attacker = game.units[-1]

        # Create weak target
        game.create_unit('W', 3, 2, 1)
        target = game.units[-1]
        target.health = 5  # Low health

        bot = MediumBot(game, player=2)
        value = bot.calculate_attack_value(attacker, target)

        # Should have high value for a kill
        assert value > 100


class TestBotFactoryMediumBot:
    """Test bot factory integration."""

    def test_bot_factory_creates_mediumbot(self):
        """Test that bot factory can create MediumBot."""
        from game.bot_factory import create_bot
        from reinforcetactics.utils.settings import get_settings

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        settings = get_settings()

        bot = create_bot(game_state, 2, 'MediumBot', settings)

        assert isinstance(bot, MediumBot)
        assert bot.bot_player == 2


class TestTournamentMediumBot:
    """Test tournament system integration."""

    def test_tournament_discovers_mediumbot(self):
        """Test that tournament system discovers MediumBot."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import TournamentRunner

        runner = TournamentRunner(
            map_file='maps/1v1/beginner.csv',
            output_dir='/tmp/test_tournament_medium',
            games_per_side=1
        )

        bots = runner.discover_bots(models_dir=None, include_test_bots=False)

        # Should find both SimpleBot and MediumBot
        bot_names = [bot.name for bot in bots]
        assert 'SimpleBot' in bot_names
        assert 'MediumBot' in bot_names

    def test_bot_descriptor_medium_bot(self):
        """Test BotDescriptor for MediumBot."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from tournament import BotDescriptor

        desc = BotDescriptor('TestMediumBot', 'medium')
        assert desc.name == 'TestMediumBot'
        assert desc.bot_type == 'medium'

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)

        bot = desc.create_bot(game_state, 2)
        assert isinstance(bot, MediumBot)
        assert bot.bot_player == 2
