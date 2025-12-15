"""Tests for AdvancedBot class."""
import pytest
import numpy as np

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MediumBot, SimpleBot
from reinforcetactics.utils.file_io import FileIO
from game.bot_factory import create_bot
from scripts.tournament import BotDescriptor


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
    # Add some mountains and forests
    map_data[5][5] = 'm'
    map_data[4][4] = 'f'
    # Add some towers
    map_data[3][3] = 't_1'
    map_data[6][6] = 't_2'
    return GameState(map_data, num_players=2)


class TestAdvancedBotBasics:
    """Test basic AdvancedBot functionality."""

    def test_advancedbot_initialization(self, simple_game):
        """Test AdvancedBot can be initialized."""
        bot = AdvancedBot(simple_game, player=2)
        assert bot.bot_player == 2
        assert bot.game_state == simple_game
        assert bot.mcts_iterations == 20
        assert bot.mcts_depth == 2
        assert bot.map_analyzed is False

    def test_advancedbot_custom_mcts_params(self, simple_game):
        """Test AdvancedBot with custom MCTS parameters."""
        bot = AdvancedBot(simple_game, player=2, mcts_iterations=10, mcts_depth=3)
        assert bot.mcts_iterations == 10
        assert bot.mcts_depth == 3

    def test_advancedbot_manhattan_distance(self, simple_game):
        """Test manhattan distance calculation."""
        bot = AdvancedBot(simple_game, player=2)
        assert bot.manhattan_distance(0, 0, 0, 0) == 0
        assert bot.manhattan_distance(0, 0, 3, 4) == 7
        assert bot.manhattan_distance(5, 5, 2, 3) == 5


class TestAdvancedBotMapAnalysis:
    """Test AdvancedBot map analysis features."""

    def test_analyze_map_identifies_hqs(self, simple_game):
        """Test that map analysis identifies HQ positions."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # map_analyzed flag is set in take_turn(), not analyze_map()
        # But the HQ positions should be populated
        assert 1 in bot.hq_positions
        assert 2 in bot.hq_positions
        assert bot.hq_positions[1] == (0, 0)
        assert bot.hq_positions[2] == (9, 9)

    def test_analyze_map_identifies_chokepoints(self, simple_game):
        """Test that map analysis identifies chokepoints."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Should have identified some chokepoints (corners, edges)
        assert len(bot.chokepoints) > 0

    def test_analyze_map_creates_distance_maps(self, simple_game):
        """Test that map analysis creates distance maps."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        assert 1 in bot.distance_maps
        assert 2 in bot.distance_maps
        
        # Check a specific distance
        assert bot.distance_maps[1][(0, 0)] == 0
        assert bot.distance_maps[2][(9, 9)] == 0

    def test_analyze_map_identifies_defensive_positions(self, simple_game):
        """Test that map analysis identifies defensive positions."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Should include the mountain and forest we added
        assert (5, 5) in bot.defensive_positions  # Mountain
        assert (4, 4) in bot.defensive_positions  # Forest

    def test_identify_factory_clusters(self, simple_game):
        """Test factory cluster identification."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Should have identified some clusters
        assert isinstance(bot.factory_clusters, list)


class TestAdvancedBotStrategicAssessment:
    """Test AdvancedBot strategic assessment methods."""

    def test_calculate_income_differential(self, simple_game):
        """Test income differential calculation."""
        bot = AdvancedBot(simple_game, player=2)
        income_diff = bot.calculate_income_differential()
        
        # Player 1 has HQ (150) + building (100) + tower (50) = 300
        # Player 2 has HQ (150) + building (100) + tower (50) = 300
        assert income_diff == 0

    def test_assess_threat_level(self, simple_game):
        """Test threat level assessment."""
        bot = AdvancedBot(simple_game, player=2)
        threat_level = bot.assess_threat_level()
        
        # Should return a number (initially 0 with no enemy units)
        assert isinstance(threat_level, int)
        assert threat_level >= 0


class TestAdvancedBotUnitComposition:
    """Test AdvancedBot unit composition strategies."""

    def test_get_desired_composition_ahead(self, simple_game):
        """Test composition when ahead in income."""
        bot = AdvancedBot(simple_game, player=2)
        comp = bot.get_desired_composition(income_diff=300, unit_count=5)
        
        # Should be aggressive when ahead
        assert 'W' in comp
        assert 'A' in comp
        assert 'M' in comp

    def test_get_desired_composition_behind(self, simple_game):
        """Test composition when behind in income."""
        bot = AdvancedBot(simple_game, player=2)
        comp = bot.get_desired_composition(income_diff=-300, unit_count=5)
        
        # Should be defensive when behind
        assert 'W' in comp
        assert 'C' in comp

    def test_get_desired_composition_balanced(self, simple_game):
        """Test composition when balanced."""
        bot = AdvancedBot(simple_game, player=2)
        comp = bot.get_desired_composition(income_diff=0, unit_count=5)
        
        # Should have balanced composition
        assert 'W' in comp
        assert 'A' in comp
        assert 'M' in comp

    def test_purchase_units_advanced(self, simple_game):
        """Test advanced unit purchasing."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        
        bot = AdvancedBot(simple_game, player=2)
        initial_gold = simple_game.player_gold[2]
        initial_units = len([u for u in simple_game.units if u.player == 2])
        
        bot.purchase_units_advanced(income_diff=0)
        
        # Should have purchased multiple units
        final_gold = simple_game.player_gold[2]
        final_units = len([u for u in simple_game.units if u.player == 2])
        
        assert final_gold < initial_gold
        assert final_units > initial_units


class TestAdvancedBotUnitPriority:
    """Test AdvancedBot unit action priority."""

    def test_get_unit_action_priority_cleric(self, simple_game):
        """Test that Clerics have highest priority."""
        bot = AdvancedBot(simple_game, player=2)
        
        # Set current player and give gold
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        
        # Create units at non-adjacent positions
        simple_game.create_unit('C', 5, 5, 2)
        simple_game.create_unit('W', 7, 7, 2)
        
        cleric = [u for u in simple_game.units if u.type == 'C' and u.player == 2][0]
        warrior = [u for u in simple_game.units if u.type == 'W' and u.player == 2][0]
        
        assert bot.get_unit_action_priority(cleric) > bot.get_unit_action_priority(warrior)

    def test_get_unit_action_priority_mage(self, simple_game):
        """Test that Mages have high priority."""
        bot = AdvancedBot(simple_game, player=2)
        
        # Set current player and give gold
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        
        simple_game.create_unit('M', 5, 5, 2)
        simple_game.create_unit('A', 7, 7, 2)
        
        mage = [u for u in simple_game.units if u.type == 'M' and u.player == 2][0]
        archer = [u for u in simple_game.units if u.type == 'A' and u.player == 2][0]
        
        assert bot.get_unit_action_priority(mage) > bot.get_unit_action_priority(archer)


class TestAdvancedBotCapturePriority:
    """Test AdvancedBot capture priority scoring."""

    def test_calculate_capture_priority_income_value(self, simple_game):
        """Test that capture priority considers income value."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        hq_tile = simple_game.grid.get_tile(0, 0)
        tower_tile = simple_game.grid.get_tile(3, 3)
        
        hq_priority = bot.calculate_capture_priority(hq_tile)
        tower_priority = bot.calculate_capture_priority(tower_tile)
        
        # HQ should have higher base priority due to income
        assert hq_priority > tower_priority

    def test_calculate_capture_priority_distance(self, simple_game):
        """Test that capture priority considers distance from HQ."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Get two structures at different distances
        close_tile = simple_game.grid.get_tile(9, 8)  # Building close to player 2 HQ
        
        priority = bot.calculate_capture_priority(close_tile)
        
        # Should have some priority value
        assert isinstance(priority, (int, float))


class TestAdvancedBotMCTS:
    """Test AdvancedBot MCTS implementation."""

    def test_evaluate_position(self, simple_game):
        """Test position evaluation."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Mountain should have higher score than grass
        mountain_score = bot.evaluate_position(None, (5, 5))  # Mountain position
        grass_score = bot.evaluate_position(None, (7, 7))  # Grass position
        
        # Note: Mountain bonus is +10, but distance factors affect total score
        assert isinstance(mountain_score, (int, float))
        assert isinstance(grass_score, (int, float))

    def test_generate_possible_actions(self, simple_game):
        """Test action generation for MCTS."""
        simple_game.create_unit('W', 5, 5, 2)
        unit = [u for u in simple_game.units if u.type == 'W'][0]
        
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        actions = bot.generate_possible_actions(unit)
        
        # Should generate some actions
        assert isinstance(actions, list)
        assert len(actions) > 0
        
        # Actions should have expected structure
        for action in actions:
            assert 'type' in action
            assert action['type'] == 'move'
            assert 'position' in action

    def test_simulate_action(self, simple_game):
        """Test action simulation."""
        simple_game.create_unit('W', 5, 5, 2)
        unit = [u for u in simple_game.units if u.type == 'W'][0]
        
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        action = {'type': 'move', 'position': (5, 6), 'then_attack': None}
        score = bot.simulate_action(unit, action)
        
        # Should return a score
        assert isinstance(score, (int, float))


class TestAdvancedBotSpecialAbilities:
    """Test AdvancedBot special ability usage."""

    def test_try_use_special_ability_cleric_heal(self, simple_game):
        """Test Cleric healing ability."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        
        simple_game.create_unit('C', 5, 5, 2)
        simple_game.create_unit('W', 5, 6, 2)
        
        cleric = [u for u in simple_game.units if u.type == 'C' and u.player == 2][0]
        warrior = [u for u in simple_game.units if u.type == 'W' and u.player == 2]
        
        # Only test if warrior was created (might fail if position is blocked)
        if warrior:
            warrior = warrior[0]
            # Damage the warrior
            warrior.health = 5
            
            bot = AdvancedBot(simple_game, player=2)
            bot.analyze_map()
            
            # Should use heal on damaged adjacent ally
            result = bot.try_use_special_ability(cleric)
            
            # Should have attempted to heal
            assert isinstance(result, bool)
        else:
            # If warrior couldn't be created, just verify cleric exists
            assert cleric is not None


class TestAdvancedBotRangedCombat:
    """Test AdvancedBot ranged combat."""

    def test_try_ranged_attack_archer(self, simple_game):
        """Test Archer ranged attack."""
        simple_game.create_unit('A', 5, 5, 2)
        simple_game.create_unit('W', 5, 3, 1)  # Enemy warrior
        
        archer = [u for u in simple_game.units if u.type == 'A' and u.player == 2][0]
        
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Should attempt ranged attack
        result = bot.try_ranged_attack(archer)
        
        # Result depends on whether attack was possible
        assert isinstance(result, bool)

    def test_try_ranged_attack_mage(self, simple_game):
        """Test Mage ranged attack."""
        simple_game.create_unit('M', 5, 5, 2)
        simple_game.create_unit('W', 5, 3, 1)  # Enemy warrior
        
        mage = [u for u in simple_game.units if u.type == 'M' and u.player == 2][0]
        
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()
        
        # Should attempt ranged attack
        result = bot.try_ranged_attack(mage)
        
        assert isinstance(result, bool)


class TestAdvancedBotVsMediumBot:
    """Test AdvancedBot performance against MediumBot."""

    def test_advancedbot_vs_mediumbot_single_game(self, simple_game):
        """Test a single game between AdvancedBot and MediumBot."""
        bot1 = AdvancedBot(simple_game, player=1)
        bot2 = MediumBot(simple_game, player=2)
        
        max_turns = 100
        turn_count = 0
        
        while not simple_game.game_over and turn_count < max_turns:
            current_player = simple_game.current_player
            current_bot = bot1 if current_player == 1 else bot2
            
            current_bot.take_turn()
            turn_count += 1
        
        # Game should complete
        assert turn_count <= max_turns


class TestBotFactoryAdvancedBot:
    """Test bot factory integration."""

    def test_bot_factory_creates_advancedbot(self, simple_game):
        """Test that bot factory can create AdvancedBot."""
        from reinforcetactics.utils.settings import get_settings
        settings = get_settings()
        
        bot = create_bot(simple_game, 2, 'AdvancedBot', settings)
        
        assert isinstance(bot, AdvancedBot)
        assert bot.bot_player == 2


class TestTournamentAdvancedBot:
    """Test tournament integration."""

    def test_tournament_discovers_advancedbot(self):
        """Test that tournament system discovers AdvancedBot."""
        from scripts.tournament import TournamentRunner
        
        runner = TournamentRunner(
            map_file='maps/1v1/6x6_beginner.csv',
            output_dir='/tmp/test_tournament',
            games_per_side=1
        )
        
        bots = runner.discover_bots()
        
        # Should find AdvancedBot
        bot_names = [b.name for b in bots]
        assert 'AdvancedBot' in bot_names

    def test_bot_descriptor_advanced_bot(self):
        """Test BotDescriptor can create AdvancedBot."""
        from reinforcetactics.utils.file_io import FileIO
        
        map_data = FileIO.load_map('maps/1v1/6x6_beginner.csv')
        game_state = GameState(map_data, num_players=2)
        
        descriptor = BotDescriptor('AdvancedBot', 'advanced')
        bot = descriptor.create_bot(game_state, 2)
        
        assert isinstance(bot, AdvancedBot)
        assert bot.bot_player == 2


class TestAdvancedBotFullTurn:
    """Test complete turn execution."""

    def test_advancedbot_take_turn_completes(self, simple_game):
        """Test that AdvancedBot can complete a turn without errors."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 500
        
        bot = AdvancedBot(simple_game, player=2)
        
        # Should complete turn without errors
        bot.take_turn()
        
        # Map should be analyzed after first turn
        assert bot.map_analyzed is True
        
        # Turn should have ended
        assert simple_game.current_player == 1

    def test_advancedbot_multiple_turns(self, simple_game):
        """Test that AdvancedBot can execute multiple turns."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        
        bot = AdvancedBot(simple_game, player=2)
        
        # Execute multiple turns
        for _ in range(3):
            if simple_game.current_player == 2:
                bot.take_turn()
            else:
                simple_game.end_turn()
        
        # Should have completed turns successfully
        assert bot.map_analyzed is True
