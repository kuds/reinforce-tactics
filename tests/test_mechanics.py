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
        warrior = Unit('W', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, warrior, simple_grid)

        # Archer should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert warrior.health == 10  # 15 - 5 damage

        # Warrior should not counter-attack (can't reach distance 2)
        assert result['counter_damage'] == 0
        assert archer.health == 15  # No damage taken

    def test_archer_attacks_cleric_no_counter(self, simple_grid):
        """Test archer attacking cleric gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        cleric = Unit('C', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, cleric, simple_grid)

        # Archer should deal damage but cleric survives (8 - 5 = 3 HP)
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert cleric.health == 3

        # Cleric should not counter-attack (can't reach distance 2)
        assert result['counter_damage'] == 0
        assert archer.health == 15

    def test_archer_attacks_barbarian_no_counter(self, simple_grid):
        """Test archer attacking barbarian gets no counter-attack."""
        archer = Unit('A', 5, 5, 1)
        barbarian = Unit('B', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, barbarian, simple_grid)

        # Archer should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert barbarian.health == 15  # 20 - 5 damage

        # Barbarian should not counter-attack (can't reach distance 2)
        assert result['counter_damage'] == 0
        assert archer.health == 15

    def test_archer_attacks_archer_gets_counter(self, simple_grid):
        """Test archer attacking another archer gets counter-attack."""
        archer1 = Unit('A', 5, 5, 1)
        archer2 = Unit('A', 7, 5, 2)  # Distance 2 from archer1

        result = GameMechanics.attack_unit(archer1, archer2, simple_grid)

        # Archer1 should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert archer2.health == 10  # 15 - 5 damage

        # Archer2 should counter-attack at distance 2 (5 * 0.9 = 4.5, int = 4)
        assert result['counter_damage'] == 4
        assert archer1.health == 11  # 15 - 4 damage

    def test_archer_attacks_mage_gets_counter(self, simple_grid):
        """Test archer attacking mage gets counter-attack if in range."""
        archer = Unit('A', 5, 5, 1)
        mage = Unit('M', 7, 5, 2)  # Distance 2 from archer

        result = GameMechanics.attack_unit(archer, mage, simple_grid)

        # Archer should deal damage
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert mage.health == 5  # 10 - 5 damage

        # Mage should counter-attack at distance 2 (12 * 0.8 = 9.6, int = 9)
        assert result['counter_damage'] == 9
        assert archer.health == 6  # 15 - 9 damage

    def test_archer_attacks_mage_at_distance_2_gets_counter(self, simple_grid):
        """Test archer attacking mage at distance 2 gets counter-attack."""
        archer = Unit('A', 5, 5, 1)
        mage = Unit('M', 7, 5, 2)

        result = GameMechanics.attack_unit(archer, mage, simple_grid)

        # Archer should deal damage at distance 2
        assert result['damage'] == 5
        assert result['target_alive'] is True
        assert mage.health == 5  # 10 - 5 damage

        # Mage should counter-attack at distance 2 (12 * 0.8 = 9.6, int = 9)
        assert result['counter_damage'] == 9
        assert archer.health == 6  # 15 - 9 damage

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
