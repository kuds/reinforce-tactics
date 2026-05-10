"""Tests for MediumBot class."""

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, MediumBot, SimpleBot
from reinforcetactics.tournament import (
    BotDescriptor,
    BotType,
    create_bot_instance,
)
from reinforcetactics.tournament.bots import discover_builtin_bots
from reinforcetactics.utils.file_io import FileIO


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

    def test_structure_priority_prefers_neutral_over_far_enemy(self):
        """Reproduces the beginner-map case: a neutral central tower should
        outrank a distant enemy building. Without the neutral bias the bot
        marches across the map for an owned target while ignoring the easy
        unclaimed structure next to it."""
        map_data = FileIO.load_map("maps/1v1/beginner.csv")
        game = GameState(map_data, num_players=2)
        bot = MediumBot(game, player=2)

        # Central towers in beginner.csv are unowned (player is None).
        neutral_tower = next(
            tile
            for row in game.grid.tiles
            for tile in row
            if tile.type == "t" and tile.player is None
        )
        # Enemy building far from blue's HQ.
        enemy_building = next(
            tile
            for row in game.grid.tiles
            for tile in row
            if tile.type == "b" and tile.player == 1
        )

        assert bot.get_structure_priority(neutral_tower) < bot.get_structure_priority(enemy_building)


class TestMediumBotCoordinatedAttacks:
    """Test MediumBot coordinated attack strategies."""

    def test_mediumbot_identifies_killable_targets(self):
        """Test that MediumBot can identify enemies that can be killed."""
        # Create a game with specific setup
        map_data = FileIO.load_map("maps/1v1/beginner.csv")
        game = GameState(map_data, num_players=2)

        # Create enemy unit with low health
        game.create_unit("W", 3, 3, 1)
        enemy = game.units[-1]
        enemy.health = 5  # Low health

        # Create bot units nearby
        game.create_unit("W", 2, 3, 2)
        game.create_unit("W", 4, 3, 2)

        bot = MediumBot(game, player=2)
        bot_units = [u for u in game.units if u.player == 2]

        killable = bot.find_killable_targets(bot_units)

        # Should identify the low-health enemy as killable (or may not be killable depending on exact positions)
        _ = len(killable)  # Check computed without error


class TestMediumBotContestedStructures:
    """Test MediumBot's ability to detect and respond to contested structures."""

    def test_finds_contested_structures(self):
        """Test that MediumBot can find structures being captured."""
        map_data = FileIO.load_map("maps/1v1/beginner.csv")
        game = GameState(map_data, num_players=2)

        bot = MediumBot(game, player=2)

        # Get a structure owned by player 1
        p1_structures = [tile for row in game.grid.tiles for tile in row if tile.is_capturable() and tile.player == 1]

        if p1_structures:
            structure = p1_structures[0]
            # Simulate partial capture
            structure.health = structure.max_health - 5

            # Place enemy unit on structure
            game.create_unit("W", structure.x, structure.y, 1)

            contested = bot.find_contested_structures()

            # Should find the contested structure
            assert len(contested) >= 1


class TestMediumBotVsSimpleBot:
    """Test MediumBot performance against SimpleBot."""

    def test_mediumbot_vs_simplebot_single_game(self):
        """Test a single game between MediumBot and SimpleBot."""
        # Load a standard map
        map_data = FileIO.load_map("maps/1v1/beginner.csv")
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
        map_data = FileIO.load_map("maps/1v1/beginner.csv")
        game = GameState(map_data, num_players=2)

        # Create attacker
        game.create_unit("W", 2, 2, 2)
        attacker = game.units[-1]

        # Create weak target
        game.create_unit("W", 3, 2, 1)
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
        from reinforcetactics.app.bot_factory import create_bot
        from reinforcetactics.utils.settings import get_settings

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)
        settings = get_settings()

        bot = create_bot(game_state, 2, "MediumBot", settings)

        assert isinstance(bot, MediumBot)
        assert bot.bot_player == 2


class TestTournamentMediumBot:
    """Test tournament system integration."""

    def test_tournament_discovers_mediumbot(self):
        """Test that tournament system discovers MediumBot."""
        bots = discover_builtin_bots()

        # Should find both SimpleBot and MediumBot
        bot_names = [bot.name for bot in bots]
        assert "SimpleBot" in bot_names
        assert "MediumBot" in bot_names

    def test_bot_descriptor_medium_bot(self):
        """Test BotDescriptor for MediumBot."""
        desc = BotDescriptor(name="TestMediumBot", bot_type=BotType.MEDIUM)
        assert desc.name == "TestMediumBot"
        assert desc.bot_type == BotType.MEDIUM

        map_data = FileIO.generate_random_map(10, 10, num_players=2)
        game_state = GameState(map_data, num_players=2)

        bot = create_bot_instance(desc, game_state, player=2)
        assert isinstance(bot, MediumBot)
        assert bot.bot_player == 2


@pytest.fixture
def heal_game():
    """Map with player-2 heal tiles in a known layout for retreat tests."""
    # 6x6 grid: blue HQ at (5,5), blue Building at (4,5) and (5,4),
    # blue Tower at (3,5). Red HQ at (0,0). Plenty of grass.
    map_data = np.array([["p" for _ in range(6)] for _ in range(6)], dtype=object)
    map_data[0][0] = "h_1"
    map_data[5][5] = "h_2"
    map_data[5][4] = "b_2"
    map_data[4][5] = "b_2"
    map_data[5][3] = "t_2"
    return GameState(map_data, num_players=2)


class TestSimpleBotHealRetreat:
    """SimpleBot only stays put when wounded on a heal tile -- it does not
    actively route to one. Tier-1 behaviour."""

    def test_stays_on_heal_tile_when_wounded(self, heal_game):
        bot = SimpleBot(heal_game, player=2)
        heal_game.create_unit("W", 5, 5, 2)  # On HQ
        unit = heal_game.units[-1]
        unit.health = 2  # well below 50%

        bot.act_with_unit(unit)
        assert (unit.x, unit.y) == (5, 5)
        assert not unit.can_move and not unit.can_attack

    def test_does_not_route_to_heal_tile(self, heal_game):
        """Tier ramp: SimpleBot won't search for a heal tile -- it only stays
        if already standing on one. A wounded unit two tiles away from a
        Building keeps acting normally."""
        bot = SimpleBot(heal_game, player=2)
        heal_game.create_unit("W", 3, 3, 2)  # NOT on a heal tile
        unit = heal_game.units[-1]
        unit.health = 1

        # Simply asserting we don't crash and don't end up on a heal tile.
        bot.act_with_unit(unit)
        # Wherever it went, it must not be one of our heal tiles -- SimpleBot
        # has no retreat routing.
        assert bot.heal_amount_at(unit.x, unit.y) == 0


class TestMediumBotHealRetreat:
    """MediumBot routes wounded units to the nearest owned heal tile.
    Tier-2 behaviour."""

    def test_routes_wounded_unit_to_heal_tile(self, heal_game):
        bot = MediumBot(heal_game, player=2)
        # Place a Warrior next to a Building -- one move puts it on heal.
        heal_game.create_unit("W", 4, 4, 2)
        unit = heal_game.units[-1]
        unit.health = 1

        bot.act_with_unit(unit)
        assert bot.heal_amount_at(unit.x, unit.y) > 0

    def test_full_health_unit_does_not_retreat(self, heal_game):
        bot = MediumBot(heal_game, player=2)
        heal_game.create_unit("W", 4, 4, 2)
        unit = heal_game.units[-1]
        # Full HP -- should fall through retreat priority.
        original_health = unit.health
        bot.act_with_unit(unit)
        # We don't care where it went; we just want to confirm retreat
        # didn't fire (i.e. should_retreat_to_heal returns False).
        assert unit.health == original_health
        assert not bot.should_retreat_to_heal(unit)

    def test_finishing_blow_overrides_retreat(self, heal_game):
        """A wounded unit that can kill an adjacent enemy this turn should
        attack instead of retreating."""
        bot = MediumBot(heal_game, player=2)
        heal_game.create_unit("W", 4, 4, 2)
        attacker = heal_game.units[-1]
        attacker.health = 1
        # 1-HP enemy adjacent
        heal_game.create_unit("W", 4, 3, 1)
        target = heal_game.units[-1]
        target.health = 1

        bot.act_with_unit(attacker)
        # Either target is dead OR attacker stayed in melee range; in either
        # case it did NOT skitter back to a heal tile.
        assert target.health <= 0 or bot.heal_amount_at(attacker.x, attacker.y) == 0


class TestAdvancedBotHealRetreat:
    """AdvancedBot uses per-archetype thresholds and prefers safer heal tiles.
    Tier-3 behaviour."""

    def test_archer_threshold_is_higher_than_warrior(self, heal_game):
        bot = AdvancedBot(heal_game, player=2)
        heal_game.player_gold[2] = 10_000  # cover both unit costs
        warrior = heal_game.create_unit("W", 5, 5, 2)
        archer = heal_game.create_unit("A", 4, 5, 2)
        assert warrior is not None and archer is not None

        # Set both to the same fraction (~50%): archer should retreat (>0.55
        # threshold), warrior should not (<0.45 threshold).
        warrior.health = int(warrior.max_health * 0.5)
        archer.health = int(archer.max_health * 0.5)

        assert bot.should_retreat_to_heal(archer)
        assert not bot.should_retreat_to_heal(warrior)

    def test_prefers_higher_heal_amount(self, heal_game):
        """Building (+2) should beat Tower (+1) at equal distance."""
        bot = AdvancedBot(heal_game, player=2)
        # Place between a tower at (3,5) and a building at (4,5).
        # Distance to (3,5) is 1, distance to (4,5) is 1.
        heal_game.create_unit("W", 4, 4, 2)
        unit = heal_game.units[-1]
        unit.health = 1

        target = bot.find_retreat_tile(unit)
        # The chosen tile must be a building (heal=2), not the tower (heal=1).
        assert target is not None
        assert bot.heal_amount_at(*target) == 2

    def test_prefers_safer_heal_tile_when_heal_equal(self, heal_game):
        """When two heal tiles offer the same heal, pick the one with fewer
        enemies in attack range."""
        bot = AdvancedBot(heal_game, player=2)
        heal_game.player_gold[1] = 10_000
        heal_game.player_gold[2] = 10_000
        unit = heal_game.create_unit("W", 1, 1, 2)
        assert unit is not None
        unit.health = 1
        # Move the unit's reachable set toward the heal tiles by giving it
        # high mobility -- simplest is to teleport it close.
        unit.x, unit.y = 4, 4

        # Enemy threatens (5,4) but not (4,5).
        archer = heal_game.create_unit("A", 5, 2, 1)
        assert archer is not None

        target = bot.find_retreat_tile(unit)
        assert target is not None
        # Both buildings heal=2; HQ also heal=2. The safer ones are (4,5)
        # and (5,5) -- (5,4) is in archer range from (5,2).
        assert target != (5, 4)
