"""Tests for StrategySwitchingBot class."""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MediumBot, SimpleBot, StrategySwitchingBot
from reinforcetactics.tournament import (
    BotDescriptor,
    BotType,
    create_bot_instance,
)
from reinforcetactics.tournament.bots import discover_builtin_bots


@pytest.fixture
def simple_game():
    """Create a simple game state for testing."""
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = "h_1"
    map_data[9][9] = "h_2"
    map_data[0][1] = "b_1"
    map_data[9][8] = "b_2"
    # Add some neutral buildings for expansion targets
    map_data[5][5] = "b"
    map_data[3][3] = "b"
    map_data[7][7] = "b"
    return GameState(map_data, num_players=2)


@pytest.fixture
def game_with_towers():
    """Create a game with more structures for expansion testing."""
    map_data = np.array([["p" for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = "h_1"
    map_data[9][9] = "h_2"
    map_data[0][1] = "b_1"
    map_data[9][8] = "b_2"
    map_data[2][2] = "t_1"
    map_data[7][7] = "t_2"
    map_data[5][5] = "b"
    map_data[4][6] = "b"
    return GameState(map_data, num_players=2)


class TestStrategySwitchingBotBasics:
    """Test basic StrategySwitchingBot functionality."""

    def test_initialization(self, simple_game):
        """Test StrategySwitchingBot can be initialized."""
        bot = StrategySwitchingBot(simple_game, player=2)
        assert bot.bot_player == 2
        assert bot.game_state == simple_game
        assert bot.current_phase == StrategySwitchingBot.EXPAND

    def test_custom_phase_thresholds(self, simple_game):
        """Test custom phase transition thresholds."""
        bot = StrategySwitchingBot(simple_game, player=2, expand_until=5, buildup_until=10)
        assert bot.expand_until == 5
        assert bot.buildup_until == 10

    def test_default_phase_thresholds(self, simple_game):
        """Test default phase transition thresholds."""
        bot = StrategySwitchingBot(simple_game, player=2)
        assert bot.expand_until == StrategySwitchingBot.DEFAULT_EXPAND_UNTIL
        assert bot.buildup_until == StrategySwitchingBot.DEFAULT_BUILDUP_UNTIL

    def test_has_delegate_bots(self, simple_game):
        """Test that delegate bot instances are created."""
        bot = StrategySwitchingBot(simple_game, player=2)
        assert isinstance(bot._simple, SimpleBot)
        assert isinstance(bot._medium, MediumBot)
        assert isinstance(bot._advanced, AdvancedBot)

    def test_delegates_share_game_state(self, simple_game):
        """Test that delegate bots share the same game state."""
        bot = StrategySwitchingBot(simple_game, player=2)
        assert bot._simple.game_state is simple_game
        assert bot._medium.game_state is simple_game
        assert bot._advanced.game_state is simple_game

    def test_mixin_methods_available(self, simple_game):
        """Test that BotUnitMixin methods work."""
        bot = StrategySwitchingBot(simple_game, player=2)
        assert bot.manhattan_distance(0, 0, 3, 4) == 7
        enabled = bot.get_enabled_units()
        assert isinstance(enabled, list)


class TestPhaseTransitions:
    """Test strategy phase transition logic."""

    def test_starts_in_expand(self, simple_game):
        """Test that bot starts in EXPAND phase."""
        bot = StrategySwitchingBot(simple_game, player=2)
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.EXPAND

    def test_transitions_to_buildup(self, simple_game):
        """Test transition from EXPAND to BUILDUP based on turn number."""
        bot = StrategySwitchingBot(simple_game, player=2, expand_until=2, buildup_until=5)
        # Simulate being on turn 3
        simple_game.turn_number = 3
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.BUILDUP

    def test_transitions_to_attack(self, simple_game):
        """Test transition from BUILDUP to ATTACK based on turn number."""
        bot = StrategySwitchingBot(simple_game, player=2, expand_until=2, buildup_until=5)
        simple_game.turn_number = 6
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.ATTACK

    def test_army_advantage_triggers_attack(self, simple_game):
        """Test that having a big army advantage triggers ATTACK early."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 2000
        bot = StrategySwitchingBot(simple_game, player=2, expand_until=5, buildup_until=10)

        # Create 4 units for player 2 and none for player 1
        # to trigger army advantage override
        simple_game.create_unit("W", 9, 9, 2)
        simple_game.create_unit("W", 9, 8, 2)
        simple_game.create_unit("W", 8, 9, 2)
        simple_game.create_unit("W", 8, 8, 2)

        simple_game.turn_number = 2  # Still early game
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.ATTACK

    def test_fewer_structures_stays_expand(self, game_with_towers):
        """Test that having fewer structures keeps bot in EXPAND."""
        bot = StrategySwitchingBot(game_with_towers, player=2, expand_until=2, buildup_until=8)

        # Player 1 has more structures (HQ + building + tower = 3)
        # Player 2 has (HQ + building + tower = 3) - equal, so turn-based applies
        # Give player 1 an extra building
        game_with_towers.grid.tiles[5][5].player = 1
        game_with_towers.grid.tiles[5][5].health = game_with_towers.grid.tiles[5][5].max_health

        game_with_towers.turn_number = 4  # Past expand_until
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.EXPAND


class TestStrategySwitchingBotTurns:
    """Test StrategySwitchingBot turn execution."""

    def test_take_turn_completes(self, simple_game):
        """Test that take_turn completes without errors."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 500

        bot = StrategySwitchingBot(simple_game, player=2)
        bot.take_turn()

        # Turn should have ended (current player advances)
        assert simple_game.current_player != 2 or simple_game.game_over

    def test_expand_turn_purchases_units(self, simple_game):
        """Test that EXPAND phase purchases units."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000

        bot = StrategySwitchingBot(simple_game, player=2)
        initial_units = len([u for u in simple_game.units if u.player == 2])

        bot.take_turn()

        final_units = len([u for u in simple_game.units if u.player == 2])
        assert final_units > initial_units

    def test_buildup_turn_uses_advanced_logic(self, simple_game):
        """Test that BUILDUP phase delegates to AdvancedBot."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        simple_game.turn_number = 5

        bot = StrategySwitchingBot(simple_game, player=2, expand_until=2, buildup_until=8)
        bot.take_turn()

        # AdvancedBot should have analyzed the map
        assert bot._advanced.map_analyzed

    def test_attack_turn_purchases_damage_units(self, simple_game):
        """Test that ATTACK phase prioritizes damage-dealing units."""
        simple_game.current_player = 2
        simple_game.player_gold[2] = 1000
        simple_game.turn_number = 10

        bot = StrategySwitchingBot(simple_game, player=2, expand_until=2, buildup_until=5)
        bot.take_turn()

        # Should have purchased units with attack priorities
        units = [u for u in simple_game.units if u.player == 2]
        assert len(units) > 0

    def test_multiple_turns_switch_phases(self, simple_game):
        """Test that phases switch across multiple turns."""
        simple_game.player_gold[2] = 500
        bot = StrategySwitchingBot(simple_game, player=2, expand_until=1, buildup_until=2)

        # Turn 1: EXPAND
        simple_game.current_player = 2
        simple_game.turn_number = 1
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.EXPAND

        # Turn 2: BUILDUP
        simple_game.turn_number = 2
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.BUILDUP

        # Turn 3: ATTACK
        simple_game.turn_number = 3
        bot._evaluate_phase()
        assert bot.current_phase == StrategySwitchingBot.ATTACK


class TestStrategySwitchingBotIntegration:
    """Integration tests for StrategySwitchingBot."""

    def test_tournament_descriptor(self):
        """Test that StrategySwitchingBot has a tournament descriptor."""
        descriptor = BotDescriptor.strategy_switching_bot()
        assert descriptor.name == "StrategySwitchingBot"
        assert descriptor.bot_type == BotType.STRATEGY_SWITCHING

    def test_tournament_discovery(self):
        """Test that StrategySwitchingBot is discovered by discover_builtin_bots."""
        bots = discover_builtin_bots()
        names = [b.name for b in bots]
        assert "StrategySwitchingBot" in names

    def test_create_from_descriptor(self, simple_game):
        """Test creating StrategySwitchingBot from tournament descriptor."""
        descriptor = BotDescriptor.strategy_switching_bot()
        bot = create_bot_instance(descriptor, simple_game, player=2)
        assert isinstance(bot, StrategySwitchingBot)
        assert bot.bot_player == 2

    def test_full_game_simulation(self, simple_game):
        """Test that StrategySwitchingBot can play a full game against SimpleBot."""
        bot1 = SimpleBot(simple_game, player=1)
        bot2 = StrategySwitchingBot(simple_game, player=2)

        # Run up to 20 turns
        for _ in range(20):
            if simple_game.game_over:
                break

            simple_game.current_player = 1
            simple_game.player_gold[1] = max(simple_game.player_gold[1], 300)
            bot1.take_turn()

            if simple_game.game_over:
                break

            simple_game.current_player = 2
            simple_game.player_gold[2] = max(simple_game.player_gold[2], 300)
            bot2.take_turn()

        # Game should have progressed
        assert simple_game.turn_number > 1
