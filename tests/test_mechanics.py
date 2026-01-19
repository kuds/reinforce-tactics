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
        """Test movement to position occupied by another unit (legacy behavior without moving_unit)."""
        unit = Unit('W', 5, 5, 1)
        units = [unit]
        result = GameMechanics.can_move_to_position(5, 5, simple_grid, units)
        assert result is False

    def test_can_pass_through_friendly_unit(self, simple_grid):
        """Test pathfinding allows passing through same-team units."""
        moving_unit = Unit('W', 3, 3, 1)
        friendly_unit = Unit('M', 5, 5, 1)  # Same player
        units = [moving_unit, friendly_unit]

        # During pathfinding (is_destination=False), should allow passing through friendly
        result = GameMechanics.can_move_to_position(
            5, 5, simple_grid, units, moving_unit=moving_unit, is_destination=False
        )
        assert result is True

    def test_cannot_stop_on_friendly_unit(self, simple_grid):
        """Test final destination cannot be on a friendly unit's tile."""
        moving_unit = Unit('W', 3, 3, 1)
        friendly_unit = Unit('M', 5, 5, 1)  # Same player
        units = [moving_unit, friendly_unit]

        # As final destination (is_destination=True), should block friendly units
        result = GameMechanics.can_move_to_position(
            5, 5, simple_grid, units, moving_unit=moving_unit, is_destination=True
        )
        assert result is False

    def test_cannot_pass_through_enemy_unit(self, simple_grid):
        """Test pathfinding blocks enemy units."""
        moving_unit = Unit('W', 3, 3, 1)
        enemy_unit = Unit('M', 5, 5, 2)  # Different player
        units = [moving_unit, enemy_unit]

        # During pathfinding, should block enemy units
        result = GameMechanics.can_move_to_position(
            5, 5, simple_grid, units, moving_unit=moving_unit, is_destination=False
        )
        assert result is False

    def test_cannot_stop_on_enemy_unit(self, simple_grid):
        """Test final destination cannot be on an enemy unit's tile."""
        moving_unit = Unit('W', 3, 3, 1)
        enemy_unit = Unit('M', 5, 5, 2)  # Different player
        units = [moving_unit, enemy_unit]

        # As final destination, should block enemy units
        result = GameMechanics.can_move_to_position(
            5, 5, simple_grid, units, moving_unit=moving_unit, is_destination=True
        )
        assert result is False

    def test_pathfinding_reaches_beyond_friendly_unit(self, simple_grid):
        """Test that pathfinding can reach tiles beyond a friendly unit."""
        # Create a unit at (2, 2) with movement range 3
        moving_unit = Unit('W', 2, 2, 1)
        moving_unit.movement_range = 3

        # Place a friendly unit at (3, 2) blocking the direct path
        friendly_unit = Unit('M', 3, 2, 1)  # Same player
        units = [moving_unit, friendly_unit]

        # Get reachable positions - should be able to reach (4, 2) and beyond
        reachable = moving_unit.get_reachable_positions(
            simple_grid.width,
            simple_grid.height,
            lambda x, y: GameMechanics.can_move_to_position(
                x, y, simple_grid, units, moving_unit=moving_unit, is_destination=False
            )
        )

        # Should be able to reach (4, 2) by passing through (3, 2)
        assert (4, 2) in reachable
        # Should be able to reach (5, 2) with movement 3
        assert (5, 2) in reachable
        # But should NOT include the friendly unit's position (3, 2) as reachable destination
        # because we check is_destination=True in game logic

    def test_pathfinding_blocked_by_enemy_unit(self, simple_grid):
        """Test that pathfinding cannot reach tiles beyond an enemy unit."""
        # Create a unit at (2, 2) with movement range 3
        moving_unit = Unit('W', 2, 2, 1)
        moving_unit.movement_range = 3

        # Place an enemy unit at (3, 2) blocking the path
        enemy_unit = Unit('M', 3, 2, 2)  # Different player
        units = [moving_unit, enemy_unit]

        # Get reachable positions - should NOT be able to reach (4, 2) through enemy
        reachable = moving_unit.get_reachable_positions(
            simple_grid.width,
            simple_grid.height,
            lambda x, y: GameMechanics.can_move_to_position(
                x, y, simple_grid, units, moving_unit=moving_unit, is_destination=False
            )
        )

        # Should NOT be able to reach (4, 2) because enemy blocks at (3, 2)
        assert (4, 2) not in reachable
        # Should be able to reach (2, 3) going around
        assert (2, 3) in reachable


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


class TestAttackableEnemies:
    """Test finding enemies within attack range."""

    def test_warrior_attackable_enemies(self, simple_grid):
        """Test warrior can only attack adjacent enemies."""
        warrior = Unit('W', 5, 5, 1)
        enemy1 = Unit('W', 6, 5, 2)  # Adjacent (distance 1)
        enemy2 = Unit('W', 7, 5, 2)  # Distance 2
        units = [warrior, enemy1, enemy2]

        attackable = GameMechanics.get_attackable_enemies(warrior, units, simple_grid)

        assert len(attackable) == 1
        assert enemy1 in attackable
        assert enemy2 not in attackable

    def test_mage_attackable_enemies(self, simple_grid):
        """Test mage can attack at distance 1-2."""
        mage = Unit('M', 5, 5, 1)
        enemy1 = Unit('W', 6, 5, 2)  # Adjacent (distance 1)
        enemy2 = Unit('W', 7, 5, 2)  # Distance 2
        enemy3 = Unit('W', 8, 5, 2)  # Distance 3
        units = [mage, enemy1, enemy2, enemy3]

        attackable = GameMechanics.get_attackable_enemies(mage, units, simple_grid)

        assert len(attackable) == 2
        assert enemy1 in attackable
        assert enemy2 in attackable
        assert enemy3 not in attackable

    def test_archer_attackable_enemies_no_mountain(self, simple_grid):
        """Test archer can attack at distance 2-3 (not 1)."""
        archer = Unit('A', 5, 5, 1)
        enemy1 = Unit('W', 6, 5, 2)  # Adjacent (distance 1) - should NOT be attackable
        enemy2 = Unit('W', 7, 5, 2)  # Distance 2 - should be attackable
        enemy3 = Unit('W', 8, 5, 2)  # Distance 3 - should be attackable (no mountain needed)
        enemy4 = Unit('W', 9, 5, 2)  # Distance 4 - should NOT be attackable (no mountain)
        units = [archer, enemy1, enemy2, enemy3, enemy4]

        attackable = GameMechanics.get_attackable_enemies(archer, units, simple_grid)

        assert len(attackable) == 2
        assert enemy1 not in attackable  # Can't attack at distance 1
        assert enemy2 in attackable
        assert enemy3 in attackable  # Can attack at distance 3
        assert enemy4 not in attackable

    def test_archer_attackable_enemies_on_mountain(self):
        """Test archer can attack at distance 2-4 on mountain."""
        # Create grid with mountain at archer position
        map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[5][5] = 'm'  # Mountain at (5, 5)
        grid = TileGrid(map_data)

        archer = Unit('A', 5, 5, 1)
        enemy1 = Unit('W', 6, 5, 2)  # Adjacent (distance 1) - should NOT be attackable
        enemy2 = Unit('W', 7, 5, 2)  # Distance 2 - should be attackable
        enemy3 = Unit('W', 8, 5, 2)  # Distance 3 - should be attackable
        enemy4 = Unit('W', 9, 5, 2)  # Distance 4 - should be attackable (on mountain)
        enemy5 = Unit('W', 0, 5, 2)  # Distance 5 - should NOT be attackable
        units = [archer, enemy1, enemy2, enemy3, enemy4, enemy5]

        attackable = GameMechanics.get_attackable_enemies(archer, units, grid)

        assert len(attackable) == 3
        assert enemy1 not in attackable  # Can't attack at distance 1
        assert enemy2 in attackable
        assert enemy3 in attackable
        assert enemy4 in attackable  # Can attack at distance 4 on mountain
        assert enemy5 not in attackable

    def test_no_attackable_enemies_when_none_in_range(self, simple_grid):
        """Test returns empty list when no enemies in range."""
        warrior = Unit('W', 5, 5, 1)
        enemy = Unit('W', 8, 8, 2)  # Far away
        units = [warrior, enemy]

        attackable = GameMechanics.get_attackable_enemies(warrior, units, simple_grid)

        assert len(attackable) == 0

    def test_ignores_allies_and_self(self, simple_grid):
        """Test only returns enemies, not allies or self."""
        unit = Unit('W', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)  # Same player
        enemy = Unit('W', 4, 5, 2)  # Different player
        units = [unit, ally, enemy]

        attackable = GameMechanics.get_attackable_enemies(unit, units, simple_grid)

        assert len(attackable) == 1
        assert enemy in attackable
        assert ally not in attackable


class TestCombat:
    """Test combat mechanics."""

    def test_attack_kills_target(self):
        """Test attacker kills target."""
        attacker = Unit('W', 5, 5, 1)
        target = Unit('C', 6, 5, 2)  # Cleric with 8 HP, 4 defence

        result = GameMechanics.attack_unit(attacker, target)

        assert result['attacker_alive'] is True
        assert result['target_alive'] is False
        # Warrior (10 attack) vs Cleric (4 defence) = 10 * (1 - 0.20) = 8 damage
        assert result['damage'] == 8
        assert target.health == 0

    def test_attack_target_survives_and_counters(self):
        """Test target survives and counter-attacks."""
        attacker = Unit('W', 5, 5, 1)
        target = Unit('W', 6, 5, 2)
        target.health = 15  # Full health

        result = GameMechanics.attack_unit(attacker, target)

        assert result['target_alive'] is True
        # Warrior (10 attack) vs Warrior (6 defence) = 10 * (1 - 0.30) = 7 damage
        assert target.health == 8  # 15 - 7 damage
        # Counter attack should happen
        assert result['counter_damage'] > 0
        # Attacker should take counter damage (10 * 0.8 counter mult * 0.7 defence = 5.6 → 5)
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
        warrior = Unit('W', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, warrior, simple_grid)

        # Archer (5 attack) vs Warrior (6 defence) = 5 * 0.7 = 3.5 → 3 damage
        assert result['damage'] == 3
        assert result['target_alive'] is True
        assert warrior.health == 12  # 15 - 3 damage

        # Warrior should not counter-attack (can't reach distance 2)
        assert result['counter_damage'] == 0
        assert archer.health == 15  # No damage taken

    def test_archer_attacks_cleric_no_counter(self, simple_grid):
        """Test archer attacking cleric gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        cleric = Unit('C', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, cleric, simple_grid)

        # Archer (5 attack) vs Cleric (4 defence) = 5 * 0.8 = 4 damage
        assert result['damage'] == 4
        assert result['target_alive'] is True
        assert cleric.health == 4  # 8 - 4 damage

        # Cleric should not counter-attack (can't reach distance 2)
        assert result['counter_damage'] == 0
        assert archer.health == 15

    def test_archer_attacks_barbarian_no_counter(self, simple_grid):
        """Test archer attacking barbarian gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        barbarian = Unit('B', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, barbarian, simple_grid)

        # Archer (5 attack) vs Barbarian (2 defence) = 5 * 0.9 = 4.5 → 4 damage
        assert result['damage'] == 4
        assert result['target_alive'] is True
        assert barbarian.health == 16  # 20 - 4 damage

        # Barbarian should not counter-attack (can't reach distance 2)
        assert result['counter_damage'] == 0
        assert archer.health == 15

    def test_archer_attacks_archer_gets_counter(self, simple_grid):
        """Test archer attacking another archer gets counter-attack."""
        archer1 = Unit('A', 5, 5, 1)
        archer2 = Unit('A', 7, 5, 2)  # Distance 2 from archer1

        result = GameMechanics.attack_unit(archer1, archer2, simple_grid)

        # Archer1 (5 attack) vs Archer2 (1 defence) = 5 * 0.95 = 4.75 → 4 damage
        assert result['damage'] == 4
        assert result['target_alive'] is True
        assert archer2.health == 11  # 15 - 4 damage

        # Archer2 counter-attack: 5 * 0.8 counter mult = 4, then 4 * 0.95 defence = 3.8 → 3
        assert result['counter_damage'] == 3
        assert archer1.health == 12  # 15 - 3 damage

    def test_archer_attacks_mage_gets_counter(self, simple_grid):
        """Test archer attacking mage gets counter-attack if in range."""
        archer = Unit('A', 5, 5, 1)
        mage = Unit('M', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, mage, simple_grid)

        # Archer (5 attack) vs Mage (4 defence) = 5 * 0.8 = 4 damage
        assert result['damage'] == 4
        assert result['target_alive'] is True
        assert mage.health == 6  # 10 - 4 damage

        # Mage counter: 12 ranged * 0.8 = 9.6 → 9, then 9 * 0.95 (1 defence) = 8.55 → 8
        assert result['counter_damage'] == 8
        assert archer.health == 7  # 15 - 8 damage

    def test_archer_attacks_mage_at_distance_2_gets_counter(self, simple_grid):
        """Test archer attacking mage at distance 2 gets counter-attack."""
        archer = Unit('A', 5, 5, 1)
        mage = Unit('M', 7, 5, 2)

        result = GameMechanics.attack_unit(archer, mage, simple_grid)

        # Archer (5 attack) vs Mage (4 defence) = 5 * 0.8 = 4 damage
        assert result['damage'] == 4
        assert result['target_alive'] is True
        assert mage.health == 6  # 10 - 4 damage

        # Mage counter: 12 * 0.8 = 9.6 → 9, then 9 * 0.95 = 8.55 → 8
        assert result['counter_damage'] == 8
        assert archer.health == 7  # 15 - 8 damage

    def test_archer_on_mountain_attacks_at_distance_3(self):
        """Test archer on mountain can attack at distance 3."""
        # Create a grid with a mountain
        map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[5][5] = 'm'  # Mountain at archer position
        grid = TileGrid(map_data)

        archer = Unit('A', 5, 5, 1)
        warrior = Unit('W', 8, 5, 2)

        result = GameMechanics.attack_unit(archer, warrior, grid)

        # Archer (5 attack) vs Warrior (6 defence) = 5 * 0.7 = 3.5 → 3 damage
        assert result['damage'] == 3
        assert result['target_alive'] is True
        assert warrior.health == 12  # 15 - 3 damage

        # Warrior cannot counter from distance 3
        assert result['counter_damage'] == 0
        assert archer.health == 15


class TestDefenceSystem:
    """Test the new defence damage reduction system."""

    def test_defence_reduces_damage(self, simple_grid):
        """Test that defence reduces incoming damage by 5% per point."""
        # Warrior (10 attack) vs Warrior (6 defence)
        # 6 defence = 30% reduction, so 10 * 0.7 = 7 damage
        attacker = Unit('W', 5, 5, 1)
        defender = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_unit(attacker, defender, simple_grid)

        assert result['damage'] == 7

    def test_defence_minimum_damage_is_one(self, simple_grid):
        """Test that minimum damage is always at least 1."""
        # Create a low-attack unit vs high-defence
        attacker = Unit('C', 5, 5, 1)  # Cleric: 2 attack
        defender = Unit('W', 6, 5, 2)  # Warrior: 6 defence (30% reduction)
        # 2 * 0.7 = 1.4 → 1 (minimum)

        result = GameMechanics.attack_unit(attacker, defender, simple_grid)

        assert result['damage'] >= 1


class TestKnightChargeAbility:
    """Test Knight's Charge ability."""

    def test_knight_charge_bonus_with_3_tiles_moved(self, simple_grid):
        """Test Knight gets +50% damage when moving 3+ tiles."""
        knight = Unit('K', 2, 2, 1)
        knight.original_x = 2
        knight.original_y = 2

        # Simulate moving 3 tiles
        knight.move_to(5, 2)  # Moved 3 tiles right

        enemy = Unit('W', 6, 2, 2)  # Adjacent to knight after move

        # Knight (8 attack) with charge (+50%) = 12 base
        # vs Warrior (6 defence, 30% reduction) = 12 * 0.7 = 8.4 → 8 damage
        result = GameMechanics.attack_unit(knight, enemy, simple_grid)

        assert result['charge_bonus'] is True
        assert result['damage'] == 8

    def test_knight_no_charge_bonus_with_2_tiles_moved(self, simple_grid):
        """Test Knight doesn't get charge bonus when moving less than 3 tiles."""
        knight = Unit('K', 3, 2, 1)
        knight.original_x = 3
        knight.original_y = 2

        # Simulate moving only 2 tiles
        knight.move_to(5, 2)  # Moved 2 tiles right

        enemy = Unit('W', 6, 2, 2)

        # Knight (8 attack) without charge
        # vs Warrior (6 defence) = 8 * 0.7 = 5.6 → 5 damage
        result = GameMechanics.attack_unit(knight, enemy, simple_grid)

        assert result['charge_bonus'] is False
        assert result['damage'] == 5


class TestRogueFlankAbility:
    """Test Rogue's Flank ability."""

    def test_rogue_flank_bonus_when_enemy_adjacent_to_ally(self, simple_grid):
        """Test Rogue gets +50% damage when enemy is adjacent to another ally."""
        rogue = Unit('R', 5, 5, 1)
        ally = Unit('W', 7, 5, 1)  # Ally adjacent to target
        target = Unit('W', 6, 5, 2)  # Target between rogue and ally

        units = [rogue, ally, target]

        # Rogue (9 attack) with flank (+50%) = 13.5 → 13 base
        # vs Warrior (6 defence, 30% reduction) = 13 * 0.7 = 9.1 → 9 damage
        result = GameMechanics.attack_unit(rogue, target, simple_grid, units)

        assert result['flank_bonus'] is True
        assert result['damage'] == 9

    def test_rogue_no_flank_bonus_when_alone(self, simple_grid):
        """Test Rogue doesn't get flank bonus when no allies adjacent to enemy."""
        rogue = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)

        units = [rogue, target]

        # Rogue (9 attack) without flank
        # vs Warrior (6 defence) = 9 * 0.7 = 6.3 → 6 damage
        result = GameMechanics.attack_unit(rogue, target, simple_grid, units)

        assert result['flank_bonus'] is False
        assert result['damage'] == 6


class TestRogueEvadeAbility:
    """Test Rogue's Evade ability (25% dodge counter-attacks)."""

    def test_rogue_evade_triggers_when_random_below_threshold(self, simple_grid, monkeypatch):
        """Test Rogue evades counter-attack when random roll is below 0.25."""
        # Mock random.random to return a value below 0.25
        import reinforcetactics.game.mechanics as mechanics_module
        monkeypatch.setattr(mechanics_module.random, 'random', lambda: 0.1)

        rogue = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)  # Warrior can counter-attack

        result = GameMechanics.attack_unit(rogue, target, simple_grid)

        assert result['evade'] is True
        assert result['counter_damage'] == 0
        assert rogue.health == 12  # Full health, no counter damage taken

    def test_rogue_no_evade_when_random_above_threshold(self, simple_grid, monkeypatch):
        """Test Rogue doesn't evade when random roll is above 0.25."""
        # Mock random.random to return a value above 0.25
        import reinforcetactics.game.mechanics as mechanics_module
        monkeypatch.setattr(mechanics_module.random, 'random', lambda: 0.5)

        rogue = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)  # Warrior can counter-attack

        result = GameMechanics.attack_unit(rogue, target, simple_grid)

        assert result['evade'] is False
        assert result['counter_damage'] > 0
        assert rogue.health < 12  # Took counter damage

    def test_non_rogue_cannot_evade(self, simple_grid, monkeypatch):
        """Test non-Rogue units cannot evade counter-attacks."""
        # Even with favorable random roll, non-Rogues shouldn't evade
        import reinforcetactics.game.mechanics as mechanics_module
        monkeypatch.setattr(mechanics_module.random, 'random', lambda: 0.1)

        warrior = Unit('W', 5, 5, 1)
        target = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_unit(warrior, target, simple_grid)

        assert result['evade'] is False
        # Warrior should take counter damage
        assert warrior.health < 15


class TestSorcererHasteAbility:
    """Test Sorcerer's Haste ability."""

    def test_sorcerer_can_haste_ally(self, simple_grid):
        """Test Sorcerer can grant Haste to nearby ally."""
        sorcerer = Unit('S', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)  # Adjacent ally

        result = GameMechanics.haste_unit(sorcerer, ally)

        assert result is True
        assert ally.is_hasted is True
        assert ally.can_move is True
        assert ally.can_attack is True
        assert sorcerer.haste_cooldown == 3

    def test_sorcerer_cannot_haste_when_on_cooldown(self, simple_grid):
        """Test Sorcerer cannot use Haste when on cooldown."""
        sorcerer = Unit('S', 5, 5, 1)
        sorcerer.haste_cooldown = 2
        ally = Unit('W', 6, 5, 1)

        result = GameMechanics.haste_unit(sorcerer, ally)

        assert result is False
        assert ally.is_hasted is False

    def test_sorcerer_cannot_haste_enemy(self, simple_grid):
        """Test Sorcerer cannot Haste enemy units."""
        sorcerer = Unit('S', 5, 5, 1)
        enemy = Unit('W', 6, 5, 2)

        result = GameMechanics.haste_unit(sorcerer, enemy)

        assert result is False

    def test_sorcerer_cannot_haste_self(self, simple_grid):
        """Test Sorcerer cannot Haste itself."""
        sorcerer = Unit('S', 5, 5, 1)

        result = GameMechanics.haste_unit(sorcerer, sorcerer)

        assert result is False

    def test_sorcerer_haste_range_limit(self, simple_grid):
        """Test Sorcerer Haste has range 1-2."""
        sorcerer = Unit('S', 5, 5, 1)
        ally_far = Unit('W', 8, 5, 1)  # Distance 3, out of range

        result = GameMechanics.haste_unit(sorcerer, ally_far)

        assert result is False

    def test_haste_cooldown_decrements(self, simple_grid):
        """Test Haste cooldown decrements each turn."""
        sorcerer = Unit('S', 5, 5, 1)
        sorcerer.haste_cooldown = 2

        units = [sorcerer]
        ready = GameMechanics.decrement_haste_cooldowns(units, 1)

        assert sorcerer.haste_cooldown == 1
        assert len(ready) == 0

        ready = GameMechanics.decrement_haste_cooldowns(units, 1)

        assert sorcerer.haste_cooldown == 0
        assert len(ready) == 1
        assert sorcerer in ready


class TestSorcererAttacks:
    """Test Sorcerer attack mechanics."""

    def test_sorcerer_adjacent_damage(self, simple_grid):
        """Test Sorcerer deals 6 damage at distance 1."""
        sorcerer = Unit('S', 5, 5, 1)
        enemy = Unit('A', 6, 5, 2)  # Archer: 1 defence

        # Sorcerer adjacent (6 attack) vs Archer (1 defence, 5% reduction)
        # 6 * 0.95 = 5.7 → 5 damage
        result = GameMechanics.attack_unit(sorcerer, enemy, simple_grid)

        assert result['damage'] == 5

    def test_sorcerer_range_damage(self, simple_grid):
        """Test Sorcerer deals 8 damage at distance 2."""
        sorcerer = Unit('S', 5, 5, 1)
        enemy = Unit('A', 7, 5, 2)  # Archer: 1 defence, distance 2

        # Sorcerer ranged (8 attack) vs Archer (1 defence)
        # 8 * 0.95 = 7.6 → 7 damage
        result = GameMechanics.attack_unit(sorcerer, enemy, simple_grid)

        assert result['damage'] == 7

    def test_sorcerer_cannot_attack_at_distance_3(self, simple_grid):
        """Test Sorcerer cannot attack at distance 3."""
        sorcerer = Unit('S', 5, 5, 1)
        enemy = Unit('W', 8, 5, 2)  # Distance 3

        damage = sorcerer.get_attack_damage(enemy.x, enemy.y)

        assert damage == 0


class TestIsEnemyFlanked:
    """Test the flanking detection helper."""

    def test_enemy_flanked_when_ally_adjacent(self, simple_grid):
        """Test enemy is flanked when attacker's ally is adjacent to target."""
        attacker = Unit('R', 5, 5, 1)
        ally = Unit('W', 7, 5, 1)
        target = Unit('W', 6, 5, 2)

        units = [attacker, ally, target]

        assert GameMechanics.is_enemy_flanked(attacker, target, units) is True

    def test_enemy_not_flanked_when_only_attacker(self, simple_grid):
        """Test enemy is not flanked when only attacker is nearby."""
        attacker = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)

        units = [attacker, target]

        assert GameMechanics.is_enemy_flanked(attacker, target, units) is False

    def test_enemy_not_flanked_by_enemy_units(self, simple_grid):
        """Test enemy is not flanked by its own allies."""
        attacker = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)
        enemy_ally = Unit('M', 7, 5, 2)  # Same team as target

        units = [attacker, target, enemy_ally]

        assert GameMechanics.is_enemy_flanked(attacker, target, units) is False


@pytest.fixture
def forest_grid():
    """Create a grid with forest tiles for testing Rogue evade bonus."""
    map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
    map_data[5][5] = 'f'  # Forest at (5, 5)
    map_data[5][6] = 'f'  # Forest at (6, 5)
    return TileGrid(map_data)


class TestRogueForestEvadeBonus:
    """Test Rogue's additional evade chance when in forest."""

    def test_rogue_evade_in_forest_triggers_at_higher_threshold(self, forest_grid, monkeypatch):
        """Test Rogue in forest evades at 0.30 (above 0.25 but below 0.35)."""
        import reinforcetactics.game.mechanics as mechanics_module
        monkeypatch.setattr(mechanics_module.random, 'random', lambda: 0.30)

        # Place rogue on forest tile (5, 5)
        rogue = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_unit(rogue, target, forest_grid)

        # Should evade because 0.30 < 0.35 (base 0.25 + forest bonus 0.10)
        assert result['evade'] is True
        assert result['counter_damage'] == 0
        assert rogue.health == 12

    def test_rogue_evade_in_forest_no_evade_above_threshold(self, forest_grid, monkeypatch):
        """Test Rogue in forest doesn't evade when random is above 0.35."""
        import reinforcetactics.game.mechanics as mechanics_module
        monkeypatch.setattr(mechanics_module.random, 'random', lambda: 0.40)

        # Place rogue on forest tile (5, 5)
        rogue = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_unit(rogue, target, forest_grid)

        # Should NOT evade because 0.40 > 0.35
        assert result['evade'] is False
        assert result['counter_damage'] > 0
        assert rogue.health < 12

    def test_rogue_evade_on_grass_uses_base_chance(self, simple_grid, monkeypatch):
        """Test Rogue on grass uses base 25% evade chance, not forest bonus."""
        import reinforcetactics.game.mechanics as mechanics_module
        monkeypatch.setattr(mechanics_module.random, 'random', lambda: 0.30)

        # Place rogue on grass tile (not forest)
        rogue = Unit('R', 5, 5, 1)
        target = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_unit(rogue, target, simple_grid)

        # Should NOT evade because 0.30 > 0.25 (base chance without forest bonus)
        assert result['evade'] is False
        assert result['counter_damage'] > 0
        assert rogue.health < 12


class TestSorcererDefenceBuff:
    """Test Sorcerer's Defence Buff ability."""

    def test_sorcerer_can_defence_buff_ally(self, simple_grid):
        """Test Sorcerer can grant Defence Buff to nearby ally."""
        sorcerer = Unit('S', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)

        result = GameMechanics.defence_buff_unit(sorcerer, ally)

        assert result is True
        assert ally.has_defence_buff() is True
        assert ally.defence_buff_turns == 3
        assert sorcerer.defence_buff_cooldown == 3

    def test_sorcerer_can_defence_buff_self(self, simple_grid):
        """Test Sorcerer can grant Defence Buff to itself."""
        sorcerer = Unit('S', 5, 5, 1)

        result = GameMechanics.defence_buff_unit(sorcerer, sorcerer)

        assert result is True
        assert sorcerer.has_defence_buff() is True
        assert sorcerer.defence_buff_turns == 3

    def test_sorcerer_cannot_defence_buff_when_on_cooldown(self, simple_grid):
        """Test Sorcerer cannot use Defence Buff when on cooldown."""
        sorcerer = Unit('S', 5, 5, 1)
        sorcerer.defence_buff_cooldown = 2
        ally = Unit('W', 6, 5, 1)

        result = GameMechanics.defence_buff_unit(sorcerer, ally)

        assert result is False
        assert ally.has_defence_buff() is False

    def test_sorcerer_cannot_defence_buff_enemy(self, simple_grid):
        """Test Sorcerer cannot Defence Buff enemy units."""
        sorcerer = Unit('S', 5, 5, 1)
        enemy = Unit('W', 6, 5, 2)

        result = GameMechanics.defence_buff_unit(sorcerer, enemy)

        assert result is False

    def test_sorcerer_cannot_defence_buff_already_buffed(self, simple_grid):
        """Test Sorcerer cannot buff unit that already has defence buff."""
        sorcerer = Unit('S', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)
        ally.defence_buff_turns = 2  # Already buffed

        result = GameMechanics.defence_buff_unit(sorcerer, ally)

        assert result is False

    def test_sorcerer_defence_buff_range_limit(self, simple_grid):
        """Test Sorcerer Defence Buff has range 0-2."""
        sorcerer = Unit('S', 5, 5, 1)
        ally_far = Unit('W', 8, 5, 1)  # Distance 3, out of range

        result = GameMechanics.defence_buff_unit(sorcerer, ally_far)

        assert result is False


class TestSorcererAttackBuff:
    """Test Sorcerer's Attack Buff ability."""

    def test_sorcerer_can_attack_buff_ally(self, simple_grid):
        """Test Sorcerer can grant Attack Buff to nearby ally."""
        sorcerer = Unit('S', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)

        result = GameMechanics.attack_buff_unit(sorcerer, ally)

        assert result is True
        assert ally.has_attack_buff() is True
        assert ally.attack_buff_turns == 3
        assert sorcerer.attack_buff_cooldown == 3

    def test_sorcerer_can_attack_buff_self(self, simple_grid):
        """Test Sorcerer can grant Attack Buff to itself."""
        sorcerer = Unit('S', 5, 5, 1)

        result = GameMechanics.attack_buff_unit(sorcerer, sorcerer)

        assert result is True
        assert sorcerer.has_attack_buff() is True
        assert sorcerer.attack_buff_turns == 3

    def test_sorcerer_cannot_attack_buff_when_on_cooldown(self, simple_grid):
        """Test Sorcerer cannot use Attack Buff when on cooldown."""
        sorcerer = Unit('S', 5, 5, 1)
        sorcerer.attack_buff_cooldown = 2
        ally = Unit('W', 6, 5, 1)

        result = GameMechanics.attack_buff_unit(sorcerer, ally)

        assert result is False
        assert ally.has_attack_buff() is False

    def test_sorcerer_cannot_attack_buff_enemy(self, simple_grid):
        """Test Sorcerer cannot Attack Buff enemy units."""
        sorcerer = Unit('S', 5, 5, 1)
        enemy = Unit('W', 6, 5, 2)

        result = GameMechanics.attack_buff_unit(sorcerer, enemy)

        assert result is False

    def test_sorcerer_cannot_attack_buff_already_buffed(self, simple_grid):
        """Test Sorcerer cannot buff unit that already has attack buff."""
        sorcerer = Unit('S', 5, 5, 1)
        ally = Unit('W', 6, 5, 1)
        ally.attack_buff_turns = 2  # Already buffed

        result = GameMechanics.attack_buff_unit(sorcerer, ally)

        assert result is False

    def test_sorcerer_attack_buff_range_limit(self, simple_grid):
        """Test Sorcerer Attack Buff has range 0-2."""
        sorcerer = Unit('S', 5, 5, 1)
        ally_far = Unit('W', 8, 5, 1)  # Distance 3, out of range

        result = GameMechanics.attack_buff_unit(sorcerer, ally_far)

        assert result is False


class TestBuffDamageModifiers:
    """Test buff effects on damage calculations."""

    def test_attack_buff_increases_damage(self, simple_grid):
        """Test Attack Buff increases damage by 50%."""
        attacker = Unit('W', 5, 5, 1)
        attacker.attack_buff_turns = 3  # Has attack buff
        target = Unit('A', 6, 5, 2)  # Archer: 1 defence

        # Warrior (10 attack) with attack buff (+50%) = 15 attack
        # vs Archer (1 defence, 5% reduction) = 15 * 0.95 = 14.25 → 14 damage
        result = GameMechanics.attack_unit(attacker, target, simple_grid)

        assert result['attack_buff'] is True
        assert result['damage'] == 14

    def test_defence_buff_reduces_damage(self, simple_grid):
        """Test Defence Buff reduces incoming damage by 50%."""
        attacker = Unit('W', 5, 5, 1)
        target = Unit('A', 6, 5, 2)  # Archer: 1 defence
        target.defence_buff_turns = 3  # Has defence buff

        # Warrior (10 attack) vs Archer (1 defence, 5% reduction) = 10 * 0.95 = 9.5 → 9 damage
        # Then reduced by defence buff (-50%) = 9 * 0.5 = 4.5 → 4 damage (minimum 1)
        result = GameMechanics.attack_unit(attacker, target, simple_grid)

        assert result['defence_buff'] is True
        assert result['damage'] == 4

    def test_attack_buff_applies_to_counter_attack(self, simple_grid):
        """Test Attack Buff increases counter-attack damage."""
        attacker = Unit('W', 5, 5, 1)  # Warrior: 6 defence
        target = Unit('W', 6, 5, 2)  # Warrior: will counter-attack
        target.attack_buff_turns = 3  # Counter-attacker has attack buff

        # Normal counter: 10 * 0.8 = 8 base, with attack buff: 8 * 1.5 = 12
        # vs Warrior (6 defence, 30% reduction) = 12 * 0.7 = 8.4 → 8 counter damage
        result = GameMechanics.attack_unit(attacker, target, simple_grid)

        # Counter damage should be higher than normal (without buff it would be ~5)
        assert result['counter_damage'] == 8

    def test_defence_buff_applies_to_counter_attack_received(self, simple_grid):
        """Test Defence Buff reduces counter-attack damage received."""
        attacker = Unit('W', 5, 5, 1)  # Warrior: 6 defence
        attacker.defence_buff_turns = 3  # Attacker has defence buff
        target = Unit('W', 6, 5, 2)  # Warrior: will counter-attack

        # Counter: 10 * 0.8 = 8 base vs Warrior (6 defence, 30% reduction) = 8 * 0.7 = 5.6 → 5
        # Then reduced by defence buff (-50%) = 5 * 0.5 = 2.5 → 2 counter damage (minimum 1)
        result = GameMechanics.attack_unit(attacker, target, simple_grid)

        assert result['counter_damage'] == 2


class TestBuffCooldownDecrement:
    """Test buff cooldown decrement mechanics."""

    def test_defence_buff_cooldown_decrements(self, simple_grid):
        """Test defence buff cooldown decrements each turn."""
        sorcerer = Unit('S', 5, 5, 1)
        sorcerer.defence_buff_cooldown = 2

        units = [sorcerer]
        result = GameMechanics.decrement_buff_cooldowns(units, 1)

        assert sorcerer.defence_buff_cooldown == 1
        assert len(result['defence_ready']) == 0

        result = GameMechanics.decrement_buff_cooldowns(units, 1)

        assert sorcerer.defence_buff_cooldown == 0
        assert len(result['defence_ready']) == 1
        assert sorcerer in result['defence_ready']

    def test_attack_buff_cooldown_decrements(self, simple_grid):
        """Test attack buff cooldown decrements each turn."""
        sorcerer = Unit('S', 5, 5, 1)
        sorcerer.attack_buff_cooldown = 2

        units = [sorcerer]
        result = GameMechanics.decrement_buff_cooldowns(units, 1)

        assert sorcerer.attack_buff_cooldown == 1
        assert len(result['attack_ready']) == 0

        result = GameMechanics.decrement_buff_cooldowns(units, 1)

        assert sorcerer.attack_buff_cooldown == 0
        assert len(result['attack_ready']) == 1
        assert sorcerer in result['attack_ready']


class TestBuffDurationDecrement:
    """Test buff duration decrement mechanics."""

    def test_defence_buff_duration_decrements(self, simple_grid):
        """Test defence buff duration decrements each turn."""
        warrior = Unit('W', 5, 5, 1)
        warrior.defence_buff_turns = 2

        units = [warrior]
        result = GameMechanics.decrement_buff_durations(units, 1)

        assert warrior.defence_buff_turns == 1
        assert len(result['defence_expired']) == 0
        assert warrior.has_defence_buff() is True

        result = GameMechanics.decrement_buff_durations(units, 1)

        assert warrior.defence_buff_turns == 0
        assert len(result['defence_expired']) == 1
        assert warrior in result['defence_expired']
        assert warrior.has_defence_buff() is False

    def test_attack_buff_duration_decrements(self, simple_grid):
        """Test attack buff duration decrements each turn."""
        warrior = Unit('W', 5, 5, 1)
        warrior.attack_buff_turns = 2

        units = [warrior]
        result = GameMechanics.decrement_buff_durations(units, 1)

        assert warrior.attack_buff_turns == 1
        assert len(result['attack_expired']) == 0
        assert warrior.has_attack_buff() is True

        result = GameMechanics.decrement_buff_durations(units, 1)

        assert warrior.attack_buff_turns == 0
        assert len(result['attack_expired']) == 1
        assert warrior in result['attack_expired']
        assert warrior.has_attack_buff() is False
