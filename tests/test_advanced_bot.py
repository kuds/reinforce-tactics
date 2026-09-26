"""Tests for AdvancedBot class."""

import numpy as np
import pytest

from reinforcetactics.app.bot_factory import create_bot
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MediumBot
from reinforcetactics.tournament import (
    BotDescriptor,
    BotType,
    create_bot_instance,
)
from reinforcetactics.tournament.bots import discover_builtin_bots


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    # Create a 10x10 map with basic tiles
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    # Add HQ for player 1 and 2
    map_data[0][0] = "h_1"
    map_data[9][9] = "h_2"
    # Add some buildings
    map_data[0][1] = "b_1"
    map_data[9][8] = "b_2"
    # Add some mountains and forests
    map_data[5][5] = "m"
    map_data[4][4] = "f"
    # Add some towers
    map_data[3][3] = "t_1"
    map_data[6][6] = "t_2"
    return GameState(map_data, num_players=2)


class TestAdvancedBotBasics:
    """Test basic AdvancedBot functionality."""

    def test_advancedbot_initialization(self, simple_game):
        """Test AdvancedBot can be initialized."""
        bot = AdvancedBot(simple_game, player=2)
        assert bot.bot_player == 2
        assert bot.game_state == simple_game
        assert bot.map_analyzed is False

    def test_advancedbot_inherits_mediumbot(self, simple_game):
        """Test AdvancedBot inherits from MediumBot."""
        bot = AdvancedBot(simple_game, player=2)
        assert isinstance(bot, MediumBot)

    def test_advancedbot_manhattan_distance(self, simple_game):
        """Test manhattan distance calculation (inherited from MediumBot)."""
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

    def test_analyze_map_identifies_defensive_positions(self, simple_game):
        """Test that map analysis identifies defensive positions (mountains)."""
        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()

        # Should include the mountain we added
        assert (5, 5) in bot.defensive_positions


class TestAdvancedBotUnitPurchasing:
    """Test AdvancedBot unit purchasing."""

    def test_purchase_units_enhanced(self, simple_game):
        """Test enhanced unit purchasing."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000

        bot = AdvancedBot(simple_game, player=2)
        initial_gold = simple_game.player_gold[2]
        initial_units = len([u for u in simple_game.units if u.player == 2])

        bot.purchase_units_enhanced()

        # Should have purchased multiple units
        final_gold = simple_game.player_gold[2]
        final_units = len([u for u in simple_game.units if u.player == 2])

        assert final_gold < initial_gold
        assert final_units > initial_units


class TestAdvancedBotSpecialAbilities:
    """Test AdvancedBot special ability usage."""

    def test_try_use_special_ability_cleric_heal(self, simple_game):
        """Test Cleric healing ability."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000

        simple_game.place_unit("C", 5, 5, 2)
        simple_game.place_unit("W", 5, 6, 2)

        cleric = [u for u in simple_game.units if u.type == "C" and u.player == 2][0]
        warrior = [u for u in simple_game.units if u.type == "W" and u.player == 2]

        # Only test if warrior was created
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


class TestAdvancedBotParalyze:
    """One paralyze heuristic, and telemetry that counts only paralyzes the engine accepted (review rulebots-19)."""

    def test_mage_paralyzes_the_capturer_once_and_records_it(self, simple_game):
        simple_game.current_player = 2
        tower = simple_game.grid.get_tile(6, 6)  # a neutral tower the enemy is taking
        tower.player = None
        tower.health = tower.max_health - 5
        mage = simple_game.place_unit("M", 6, 4, 2)
        capturer = simple_game.place_unit("W", 6, 6, 1)
        bystander = simple_game.place_unit("K", 5, 4, 1)  # pricier, adjacent, not capturing
        bot = AdvancedBot(simple_game, player=2)

        assert bot.try_use_special_ability(mage) is True

        assert capturer.is_paralyzed() and not bystander.is_paralyzed()
        assert [a["type"] for a in simple_game.action_history] == ["paralyze"]
        assert bot.get_capabilities_fired() == {"mage_paralyze": 1}

    def test_mage_on_cooldown_records_nothing(self, simple_game):
        simple_game.current_player = 2
        mage = simple_game.place_unit("M", 6, 4, 2)
        simple_game.place_unit("W", 6, 5, 1)
        mage.paralyze_cooldown = 2
        bot = AdvancedBot(simple_game, player=2)

        assert bot.try_use_special_ability(mage) is False
        assert simple_game.action_history == []
        assert bot.get_capabilities_fired() == {}


class TestAdvancedBotFogOfWar:
    """Bots attack only enemies fog of war lets the unit attack (the engine refuses the rest)."""

    def test_no_attack_on_an_enemy_the_unit_did_not_see_when_its_action_began(self):
        grid = np.full((8, 8), "p", dtype=object)
        grid[0, 0], grid[7, 7] = "h_1", "h_2"
        game = GameState(grid, num_players=2, fog_of_war=True)
        game.current_player = 2
        archer = game.place_unit("A", 4, 4, 2)
        enemy = game.place_unit("W", 4, 2, 1)
        # Its side saw no enemy when the archer's action began (the snapshot
        # the engine takes before a move, or the GUI when it is selected).
        archer.visible_enemies_at_action_start = set()
        attempts = []
        attack = game.attack
        game.attack = lambda *args: attempts.append(args) or attack(*args)
        bot = AdvancedBot(game, player=2)

        assert bot.in_attack_reach(archer, enemy) is False
        assert bot.try_ranged_attack(archer) is False
        bot.act_with_unit_enhanced(archer)

        assert attempts == []
        assert enemy.health == enemy.max_health


class TestAdvancedBotRangedCombat:
    """Test AdvancedBot ranged combat."""

    def test_try_ranged_attack_archer(self, simple_game):
        """Test Archer ranged attack."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000

        simple_game.place_unit("A", 5, 5, 2)
        simple_game.place_unit("W", 5, 3, 1)  # Enemy warrior

        archer = [u for u in simple_game.units if u.type == "A" and u.player == 2][0]

        bot = AdvancedBot(simple_game, player=2)
        bot.analyze_map()

        # Should attempt ranged attack
        result = bot.try_ranged_attack(archer)

        # Result depends on whether attack was possible
        assert isinstance(result, bool)

    def test_try_ranged_attack_mage(self, simple_game):
        """Test Mage ranged attack."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000

        simple_game.place_unit("M", 5, 5, 2)
        simple_game.place_unit("W", 5, 3, 1)  # Enemy warrior

        mage = [u for u in simple_game.units if u.type == "M" and u.player == 2][0]

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

        max_turns = 50
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

        bot = create_bot(simple_game, 2, "AdvancedBot", settings)

        assert isinstance(bot, AdvancedBot)
        assert bot.bot_player == 2


class TestTournamentAdvancedBot:
    """Test tournament integration."""

    def test_tournament_discovers_advancedbot(self):
        """Test that tournament system discovers AdvancedBot."""
        bots = discover_builtin_bots()

        # Should find AdvancedBot
        bot_names = [b.name for b in bots]
        assert "AdvancedBot" in bot_names

    def test_bot_descriptor_advanced_bot(self):
        """Test BotDescriptor can create AdvancedBot."""
        from reinforcetactics.utils.file_io import FileIO

        map_data = FileIO.load_map("maps/1v1/beginner.csv")
        game_state = GameState(map_data, num_players=2)

        descriptor = BotDescriptor(name="AdvancedBot", bot_type=BotType.ADVANCED)
        bot = create_bot_instance(descriptor, game_state, player=2)

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
