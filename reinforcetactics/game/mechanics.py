"""
Core game mechanics including combat, movement, income, and structure capture.
"""
from typing import Optional

from reinforcetactics.constants import (
    COUNTER_ATTACK_MULTIPLIER, PARALYZE_DURATION, HEAL_AMOUNT,
    STRUCTURE_REGEN_RATE, HEADQUARTERS_INCOME, BUILDING_INCOME, TOWER_INCOME
)


class GameMechanics:
    """Handles core game mechanics and rules."""

    def __init__(self, counter_attack_multiplier: Optional[float] = None) -> None:
        """Initialize game mechanics with configurable parameters.

        Args:
            counter_attack_multiplier: Multiplier for counter-attack damage (0.0 to 1.0).
                                       Defaults to COUNTER_ATTACK_MULTIPLIER from constants.
        """
        self.counter_attack_multiplier = (
            counter_attack_multiplier if counter_attack_multiplier is not None
            else COUNTER_ATTACK_MULTIPLIER
        )

    @staticmethod
    def can_move_to_position(x, y, grid, units, moving_unit=None, is_destination=False):
        """
        Check if a position is valid for unit movement.

        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            grid: TileGrid instance
            units: List of Unit instances
            moving_unit: The unit that is moving (optional, for team checking)
            is_destination: If True, blocks all units. If False (pathfinding),
                           only blocks enemy units (default: False)

        Returns:
            True if position is valid for movement
        """
        if not (0 <= x < grid.width and 0 <= y < grid.height):
            return False

        tile = grid.get_tile(x, y)
        if not tile.is_walkable():
            return False

        # Check if another unit is already there
        for unit in units:
            if unit.x == x and unit.y == y:
                # If this is the final destination, block all units
                if is_destination:
                    return False

                # For pathfinding, allow passing through friendly units
                if moving_unit is not None and unit.player == moving_unit.player:
                    continue  # Allow passing through friendly units

                # Block enemy units or if no moving_unit specified (legacy behavior)
                return False

        return True

    @staticmethod
    def get_adjacent_enemies(unit, units):
        """Get list of enemy units adjacent to the given unit."""
        adjacent_enemies = []
        adjacent_positions = [
            (unit.x, unit.y - 1),
            (unit.x, unit.y + 1),
            (unit.x - 1, unit.y),
            (unit.x + 1, unit.y)
        ]

        for enemy in units:
            if enemy.player != unit.player and enemy.health > 0:
                if (enemy.x, enemy.y) in adjacent_positions:
                    adjacent_enemies.append(enemy)

        return adjacent_enemies

    @staticmethod
    def get_attackable_enemies(unit, units, grid):
        """
        Get list of enemy units within the given unit's attack range.

        Args:
            unit: The unit to check attack range for
            units: List of all units
            grid: TileGrid instance (for checking mountain tiles)

        Returns:
            List of enemy units within attack range
        """
        attackable_enemies = []

        # Check if unit is on a mountain (for Archer range bonus)
        on_mountain = False
        if grid:
            tile = grid.get_tile(unit.x, unit.y)
            on_mountain = tile.type == 'm'

        # Get the unit's attack range
        min_range, max_range = unit.get_attack_range(on_mountain)

        # Check all enemies
        for enemy in units:
            if enemy.player != unit.player and enemy.health > 0:
                distance = abs(unit.x - enemy.x) + abs(unit.y - enemy.y)
                if min_range <= distance <= max_range:
                    attackable_enemies.append(enemy)

        return attackable_enemies

    @staticmethod
    def get_adjacent_allies(unit, units):
        """Get list of damaged friendly units adjacent to the given unit."""
        adjacent_allies = []
        adjacent_positions = [
            (unit.x, unit.y - 1),
            (unit.x, unit.y + 1),
            (unit.x - 1, unit.y),
            (unit.x + 1, unit.y)
        ]

        for ally in units:
            if ally.player == unit.player and ally.health > 0 and ally != unit:
                if (ally.x, ally.y) in adjacent_positions:
                    if ally.health < ally.max_health:
                        adjacent_allies.append(ally)

        return adjacent_allies

    @staticmethod
    def get_adjacent_paralyzed_allies(unit, units):
        """Get list of paralyzed friendly units adjacent to the given unit."""
        adjacent_paralyzed = []
        adjacent_positions = [
            (unit.x, unit.y - 1),
            (unit.x, unit.y + 1),
            (unit.x - 1, unit.y),
            (unit.x + 1, unit.y)
        ]

        for ally in units:
            if ally.player == unit.player and ally.health > 0 and ally != unit:
                if (ally.x, ally.y) in adjacent_positions:
                    if ally.is_paralyzed():
                        adjacent_paralyzed.append(ally)

        return adjacent_paralyzed

    def _calculate_counter_damage(self, unit, target_x, target_y, grid):
        """
        Calculate counter-attack damage for a unit.

        Args:
            unit: The unit that would counter-attack
            target_x: X coordinate of the target
            target_y: Y coordinate of the target
            grid: TileGrid instance (optional, for checking mountain tiles)

        Returns:
            Counter-attack damage as integer
        """
        on_mountain = False
        if grid:
            tile = grid.get_tile(unit.x, unit.y)
            on_mountain = tile.type == 'm'

        return int(
            unit.get_attack_damage(target_x, target_y, on_mountain) * self.counter_attack_multiplier
        )

    def attack_unit(self, attacker, target, grid=None):
        """
        Execute an attack from attacker to target.

        Args:
            attacker: The attacking unit
            target: The target unit
            grid: TileGrid instance (optional, for checking mountain tiles)

        Returns:
            dict with 'attacker_alive' and 'target_alive' booleans
        """
        # Check if attacker is on mountain for range calculation
        attacker_on_mountain = False
        if grid:
            attacker_tile = grid.get_tile(attacker.x, attacker.y)
            attacker_on_mountain = attacker_tile.type == 'm'

        attack_damage = attacker.get_attack_damage(target.x, target.y, attacker_on_mountain)
        target_alive = target.take_damage(attack_damage)

        attacker_alive = True
        counter_damage = 0

        # Counter-attack logic with Archer restrictions
        if target_alive and not target.is_paralyzed():
            # Determine if counter-attack is allowed
            can_counter = True

            # If attacker is an Archer, only Archers and Mages can counter
            if attacker.type == 'A':
                if target.type not in ['A', 'M']:
                    can_counter = False

            if can_counter:
                counter_damage = self._calculate_counter_damage(
                    target, attacker.x, attacker.y, grid
                )
                if counter_damage > 0:
                    attacker_alive = attacker.take_damage(counter_damage)

        # Calculate counter damage for response (even if 0)
        counter_damage_for_response = 0
        if target_alive and not target.is_paralyzed():
            # Adjust counter_damage if Archer attacked melee unit
            if attacker.type == 'A' and target.type not in ['A', 'M']:
                counter_damage_for_response = 0
            else:
                counter_damage_for_response = self._calculate_counter_damage(
                    target, attacker.x, attacker.y, grid
                )

        return {
            'attacker_alive': attacker_alive,
            'target_alive': target_alive,
            'damage': attack_damage,
            'counter_damage': counter_damage_for_response
        }

    @staticmethod
    def paralyze_unit(paralyzer, target):
        """Mage paralyzes the target unit."""
        if paralyzer.type != 'M':
            return False

        if target.player == paralyzer.player:
            return False

        target.paralyzed_turns = PARALYZE_DURATION
        return True

    @staticmethod
    def heal_unit(healer, target):
        """
        Healer heals the target unit.

        Args:
            healer: The unit doing the healing (must be Cleric)
            target: The target unit to heal

        Returns:
            int: Actual amount healed, or -1 if heal failed
        """
        if healer.type != 'C':
            return -1

        if target.player != healer.player:
            return -1

        if target.health >= target.max_health:
            return -1

        old_health = target.health
        target.health = min(target.health + HEAL_AMOUNT, target.max_health)
        return target.health - old_health

    @staticmethod
    def cure_unit(curer, target):
        """Cleric cures the target unit's paralysis."""
        if curer.type != 'C':
            return False

        if target.player != curer.player:
            return False

        if not target.is_paralyzed():
            return False

        target.paralyzed_turns = 0
        target.can_move = True
        target.can_attack = True
        return True

    @staticmethod
    def seize_structure(unit, tile):
        """
        Unit seizes a structure (tower, building, or HQ).

        Args:
            unit: The unit seizing
            tile: The structure tile

        Returns:
            dict with 'captured' boolean and 'game_over' boolean
        """
        if not tile.is_capturable():
            return {'captured': False, 'game_over': False}

        if tile.player == unit.player:
            return {'captured': False, 'game_over': False}

        if tile.regenerating:
            tile.regenerating = False

        damage = unit.health
        tile.health -= damage

        captured = False
        game_over = False

        if tile.health <= 0:
            tile.health = tile.max_health
            tile.player = unit.player
            tile.regenerating = False
            captured = True

            if tile.type == 'h':
                game_over = True

        return {
            'captured': captured,
            'game_over': game_over,
            'damage': damage,
            'remaining_hp': tile.health
        }

    @staticmethod
    def reset_structure_if_vacated(tile, units):
        """Reset structure HP if no unit is on it."""
        if not tile.is_capturable():
            return False

        # Check if any unit is on this tile
        for unit in units:
            if unit.x == tile.x and unit.y == tile.y:
                return False

        if tile.health < tile.max_health:
            tile.health = tile.max_health
            tile.regenerating = False
            return True

        return False

    @staticmethod
    def regenerate_structures(grid, units):
        """Regenerate HP for structures that are marked for regeneration."""
        regenerated = []
        for row in grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.regenerating:
                    # Check if there's a unit on this tile
                    unit_on_tile = False
                    for unit in units:
                        if unit.x == tile.x and unit.y == tile.y:
                            unit_on_tile = True
                            tile.regenerating = False
                            break

                    if not unit_on_tile:
                        regen_amount = int(tile.max_health * STRUCTURE_REGEN_RATE)
                        old_health = tile.health
                        tile.health = min(tile.health + regen_amount, tile.max_health)

                        if tile.health >= tile.max_health:
                            tile.regenerating = False

                        regenerated.append({
                            'tile': tile,
                            'amount': tile.health - old_health
                        })

        return regenerated

    @staticmethod
    def calculate_income(player, grid):
        """Calculate income for a player based on controlled structures."""
        headquarters_count = 0
        building_count = 0
        tower_count = 0

        for row in grid.tiles:
            for tile in row:
                if tile.player == player:
                    if tile.type == 'h':
                        headquarters_count += 1
                    elif tile.type == 'b':
                        building_count += 1
                    elif tile.type == 't':
                        tower_count += 1

        total_income = (
            headquarters_count * HEADQUARTERS_INCOME +
            building_count * BUILDING_INCOME +
            tower_count * TOWER_INCOME
        )

        return {
            'total': total_income,
            'headquarters': headquarters_count,
            'buildings': building_count,
            'towers': tower_count
        }

    @staticmethod
    def decrement_paralysis(units, player):
        """Decrement paralysis counters for a player's units at turn start."""
        cured = []
        for unit in units:
            if unit.player == player and unit.is_paralyzed():
                unit.paralyzed_turns -= 1
                if unit.paralyzed_turns <= 0:
                    cured.append(unit)
        return cured
