"""Tests for the GameMechanics class."""
import pytest
import numpy as np
from reinforcetactics.core.unit import Unit
from reinforcetactics.core.grid import TileGrid
from reinforcetactics.game.mechanics import GameMechanics


@pytest.fixture
def simple_grid():
    """Create a simple test grid."""
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[0][0] = 'w'  # Water at (0, 0)
    map_data[1][1] = 'w'  # Water at (1, 1)
    return TileGrid(map_data)


@pytest.fixture
def grid_with_structures():
    """Create a grid with structures for capture testing."""
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[2][2] = 't_1'  # Tower owned by player 1
    map_data[3][3] = 'h_2'  # HQ owned by player 2
    map_data[4][4] = 'b_1'  # Building owned by player 1
    return TileGrid(map_data)


class TestMovement:
    """Test movement validation."""

    def test_can_move_to_valid_position(self, simple_grid):
        """Test movement to valid walkable position."""
        units = []
        result = GameMechanics.can_move_to_position(5, 5, simple_grid, units)
        assert result is True

    def test_cannot_move_out_of_bounds(self, simple_grid):
        """Test movement to out of bounds position."""
        units = []
        result = GameMechanics.can_move_to_position(-1, 5, simple_grid, units)
        assert result is False

        result = GameMechanics.can_move_to_position(5, 15, simple_grid, units)
        assert result is False

    def test_cannot_move_to_water(self, simple_grid):
        """Test movement to non-walkable tile (water)."""
        units = []
        result = GameMechanics.can_move_to_position(0, 0, simple_grid, units)
        assert result is False

    def test_cannot_move_to_occupied_position(self, simple_grid):
        """Test movement to position occupied by another unit."""
        unit = Unit('W', 5, 5, 1)
        units = [unit]
        result = GameMechanics.can_move_to_position(5, 5, simple_grid, units)
        assert result is False


class TestAdjacentUnits:
    """Test getting adjacent units."""

    def test_get_adjacent_enemies(self):
        """Test finding adjacent enemy units."""
        unit = Unit('W', 5, 5, 1)
        enemy1 = Unit('W', 6, 5, 2)  # Right
        enemy2 = Unit('M', 5, 4, 2)  # Up
        ally = Unit('C', 4, 5, 1)    # Left (same player)

        units = [unit, enemy1, enemy2, ally]

        adjacent_enemies = GameMechanics.get_adjacent_enemies(unit, units)

        assert len(adjacent_enemies) == 2
        assert enemy1 in adjacent_enemies
        assert enemy2 in adjacent_enemies
        assert ally not in adjacent_enemies

    def test_get_adjacent_allies(self):
        """Test finding damaged adjacent allies."""
        unit = Unit('C', 5, 5, 1)
        ally1 = Unit('W', 6, 5, 1)  # Right, damaged
        ally1.health = 10
        ally2 = Unit('M', 5, 4, 1)  # Up, full health
        ally2.health = ally2.max_health
        enemy = Unit('W', 4, 5, 2)  # Left, different player

        units = [unit, ally1, ally2, enemy]

        adjacent_allies = GameMechanics.get_adjacent_allies(unit, units)

        assert len(adjacent_allies) == 1
        assert ally1 in adjacent_allies
        assert ally2 not in adjacent_allies  # Full health
        assert enemy not in adjacent_allies  # Different player

    def test_get_adjacent_paralyzed_allies(self):
        """Test finding paralyzed adjacent allies."""
        unit = Unit('C', 5, 5, 1)
        ally1 = Unit('W', 6, 5, 1)  # Right, paralyzed
        ally1.paralyzed_turns = 2
        ally2 = Unit('M', 5, 4, 1)  # Up, not paralyzed
        ally2.paralyzed_turns = 0

        units = [unit, ally1, ally2]

        adjacent_paralyzed = GameMechanics.get_adjacent_paralyzed_allies(unit, units)

        assert len(adjacent_paralyzed) == 1
        assert ally1 in adjacent_paralyzed
        assert ally2 not in adjacent_paralyzed


class TestCombat:
    """Test combat mechanics."""

    def test_attack_kills_target(self):
        """Test attacker kills target."""
        attacker = Unit('W', 5, 5, 1)
        target = Unit('C', 6, 5, 2)  # Cleric with 8 HP

        result = GameMechanics.attack_unit(attacker, target)

        assert result['attacker_alive'] is True
        assert result['target_alive'] is False
        assert result['damage'] == 10
        assert target.health == 0

    def test_attack_target_survives_and_counters(self):
        """Test target survives and counter-attacks."""
        attacker = Unit('W', 5, 5, 1)
        target = Unit('W', 6, 5, 2)
        target.health = 15  # Full health

        result = GameMechanics.attack_unit(attacker, target)

        assert result['target_alive'] is True
        assert target.health == 5  # 15 - 10 damage
        # Counter attack should happen
        assert result['counter_damage'] > 0
        # Attacker should take counter damage (10 * 0.9 = 9)
        assert attacker.health < 15

    def test_paralyzed_target_no_counter(self):
        """Test paralyzed target doesn't counter-attack."""
        attacker = Unit('W', 5, 5, 1)
        target = Unit('W', 6, 5, 2)
        target.paralyzed_turns = 2

        original_attacker_health = attacker.health
        result = GameMechanics.attack_unit(attacker, target)

        assert result['counter_damage'] == 0
        assert attacker.health == original_attacker_health


class TestMageAbilities:
    """Test mage special abilities."""

    def test_mage_paralyzes_enemy(self):
        """Test mage can paralyze enemy unit."""
        mage = Unit('M', 5, 5, 1)
        enemy = Unit('W', 6, 5, 2)

        result = GameMechanics.paralyze_unit(mage, enemy)

        assert result is True
        assert enemy.paralyzed_turns == 3  # PARALYZE_DURATION

    def test_non_mage_cannot_paralyze(self):
        """Test non-mage cannot paralyze."""
        warrior = Unit('W', 5, 5, 1)
        enemy = Unit('W', 6, 5, 2)

        result = GameMechanics.paralyze_unit(warrior, enemy)

        assert result is False
        assert enemy.paralyzed_turns == 0

    def test_mage_cannot_paralyze_ally(self):
        """Test mage cannot paralyze friendly unit."""
        mage = Unit('M', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)

        result = GameMechanics.paralyze_unit(mage, ally)

        assert result is False
        assert ally.paralyzed_turns == 0


class TestClericAbilities:
    """Test cleric special abilities."""

    def test_cleric_heals_damaged_ally(self):
        """Test cleric heals damaged ally."""
        cleric = Unit('C', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)
        ally.health = 10  # Damaged

        healed = GameMechanics.heal_unit(cleric, ally)

        assert healed > 0
        assert ally.health == 15  # 10 + 5 HEAL_AMOUNT

    def test_cleric_cannot_heal_full_health(self):
        """Test cleric cannot heal unit at full health."""
        cleric = Unit('C', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)
        ally.health = ally.max_health

        healed = GameMechanics.heal_unit(cleric, ally)

        assert healed == -1

    def test_cleric_cannot_heal_enemy(self):
        """Test cleric cannot heal enemy unit."""
        cleric = Unit('C', 5, 5, 1)
        enemy = Unit('W', 6, 5, 2)
        enemy.health = 5

        healed = GameMechanics.heal_unit(cleric, enemy)

        assert healed == -1
        assert enemy.health == 5  # Unchanged

    def test_non_cleric_cannot_heal(self):
        """Test non-cleric cannot heal."""
        warrior = Unit('W', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)
        ally.health = 10

        healed = GameMechanics.heal_unit(warrior, ally)

        assert healed == -1

    def test_cleric_cures_paralyzed_ally(self):
        """Test cleric cures paralyzed ally."""
        cleric = Unit('C', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)
        ally.paralyzed_turns = 2

        result = GameMechanics.cure_unit(cleric, ally)

        assert result is True
        assert ally.paralyzed_turns == 0
        assert ally.can_move is True
        assert ally.can_attack is True


class TestStructureCapture:
    """Test structure capture mechanics."""

    def test_seize_partial_capture(self, grid_with_structures):
        """Test partial capture reduces structure HP."""
        unit = Unit('W', 2, 2, 2)  # Enemy unit
        unit.health = 10
        tile = grid_with_structures.get_tile(2, 2)

        result = GameMechanics.seize_structure(unit, tile)

        assert result['captured'] is False
        assert result['game_over'] is False
        assert tile.health == 20  # 30 - 10 damage

    def test_seize_full_capture(self, grid_with_structures):
        """Test full capture changes ownership."""
        unit = Unit('W', 2, 2, 2)
        unit.health = 15
        tile = grid_with_structures.get_tile(2, 2)
        tile.health = 10  # Low health

        result = GameMechanics.seize_structure(unit, tile)

        assert result['captured'] is True
        assert result['game_over'] is False
        assert tile.player == 2  # Changed to attacker's player
        assert tile.health == tile.max_health  # Restored

    def test_hq_capture_triggers_game_over(self, grid_with_structures):
        """Test HQ capture triggers game over."""
        unit = Unit('W', 3, 3, 1)
        unit.health = 15
        tile = grid_with_structures.get_tile(3, 3)
        tile.health = 10

        result = GameMechanics.seize_structure(unit, tile)

        assert result['captured'] is True
        assert result['game_over'] is True

    def test_cannot_seize_own_structure(self, grid_with_structures):
        """Test cannot seize own structure."""
        unit = Unit('W', 2, 2, 1)
        tile = grid_with_structures.get_tile(2, 2)

        result = GameMechanics.seize_structure(unit, tile)

        assert result['captured'] is False
        assert result['game_over'] is False


class TestIncome:
    """Test income calculation."""

    def test_calculate_income(self, grid_with_structures):
        """Test income calculation counts structures correctly."""
        income = GameMechanics.calculate_income(1, grid_with_structures)

        # Player 1 has 1 tower and 1 building
        assert income['towers'] == 1
        assert income['buildings'] == 1
        assert income['headquarters'] == 0
        # Total: 50 (tower) + 100 (building) = 150
        assert income['total'] == 150

    def test_calculate_income_player_2(self, grid_with_structures):
        """Test income calculation for player 2."""
        income = GameMechanics.calculate_income(2, grid_with_structures)

        # Player 2 has 1 HQ
        assert income['headquarters'] == 1
        assert income['towers'] == 0
        assert income['buildings'] == 0
        # Total: 150 (HQ)
        assert income['total'] == 150


class TestParalysisDecrement:
    """Test paralysis duration decrement."""

    def test_decrement_paralysis(self):
        """Test paralysis counter decrements at turn start."""
        unit1 = Unit('W', 5, 5, 1)
        unit1.paralyzed_turns = 2
        unit2 = Unit('M', 6, 6, 1)
        unit2.paralyzed_turns = 1
        unit3 = Unit('C', 7, 7, 2)
        unit3.paralyzed_turns = 3

        units = [unit1, unit2, unit3]

        cured = GameMechanics.decrement_paralysis(units, 1)

        assert unit1.paralyzed_turns == 1
        assert unit2.paralyzed_turns == 0
        assert unit3.paralyzed_turns == 3  # Different player
        assert len(cured) == 1  # Only unit2 was cured
        assert unit2 in cured


class TestArcherCounterAttack:
    """Test Archer counter-attack restrictions."""

    def test_archer_attacks_warrior_no_counter(self, simple_grid):
        """Test archer attacking warrior gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        warrior = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_unit(archer, warrior, simple_grid)

        # Archer should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert warrior.health == 10  # 15 - 5 damage

        # Warrior should not counter-attack
        assert result['counter_damage'] == 0
        assert archer.health == 15  # No damage taken

    def test_archer_attacks_cleric_no_counter(self, simple_grid):
        """Test archer attacking cleric gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        cleric = Unit('C', 6, 5, 2)

        result = GameMechanics.attack_unit(archer, cleric, simple_grid)

        # Archer should deal damage and kill cleric
        assert result['damage'] == 5
        assert result['target_alive'] is False
        assert cleric.health == 0

        # Cleric should not counter-attack
        assert result['counter_damage'] == 0
        assert archer.health == 15

    def test_archer_attacks_barbarian_no_counter(self, simple_grid):
        """Test archer attacking barbarian gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        barbarian = Unit('B', 6, 5, 2)

        result = GameMechanics.attack_unit(archer, barbarian, simple_grid)

        # Archer should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert barbarian.health == 15  # 20 - 5 damage

        # Barbarian should not counter-attack
        assert result['counter_damage'] == 0
        assert archer.health == 15

    def test_archer_attacks_archer_gets_counter(self, simple_grid):
        """Test archer attacking another archer gets counter-attack."""
        archer1 = Unit('A', 5, 5, 1)
        archer2 = Unit('A', 6, 5, 2)

        result = GameMechanics.attack_unit(archer1, archer2, simple_grid)

        # Archer1 should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert archer2.health == 10  # 15 - 5 damage

        # Archer2 should counter-attack (5 * 0.9 = 4.5, int = 4)
        assert result['counter_damage'] == 4
        assert archer1.health == 11  # 15 - 4 damage

    def test_archer_attacks_mage_gets_counter(self, simple_grid):
        """Test archer attacking mage gets counter-attack if in range."""
        archer = Unit('A', 5, 5, 1)
        mage = Unit('M', 6, 5, 2)

        result = GameMechanics.attack_unit(archer, mage, simple_grid)

        # Archer should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert mage.health == 5  # 10 - 5 damage

        # Mage should counter-attack at distance 1 (8 * 0.9 = 7.2, int = 7)
        assert result['counter_damage'] == 7
        assert archer.health == 8  # 15 - 7 damage

    def test_archer_attacks_mage_at_distance_2_gets_counter(self, simple_grid):
        """Test archer attacking mage at distance 2 gets counter-attack."""
        archer = Unit('A', 5, 5, 1)
        mage = Unit('M', 7, 5, 2)

        result = GameMechanics.attack_unit(archer, mage, simple_grid)

        # Archer should deal damage at distance 2
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert mage.health == 5  # 10 - 5 damage

        # Mage should counter-attack at distance 2 (12 * 0.9 = 10.8, int = 10)
        assert result['counter_damage'] == 10
        assert archer.health == 5  # 15 - 10 damage

    def test_archer_on_mountain_attacks_at_distance_3(self):
        """Test archer on mountain can attack at distance 3."""
        import numpy as np
        # Create a grid with a mountain
        map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[5][5] = 'm'  # Mountain at archer position
        grid = TileGrid(map_data)

        archer = Unit('A', 5, 5, 1)
        warrior = Unit('W', 8, 5, 2)

        result = GameMechanics.attack_unit(archer, warrior, grid)

        # Archer should deal damage at distance 3 from mountain
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert warrior.health == 10

        # Warrior cannot counter from distance 3
        assert result['counter_damage'] == 0
        assert archer.health == 15
