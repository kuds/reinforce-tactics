"""
Unit class representing a game unit.
"""
from collections import deque
from reinforcetactics.constants import UNIT_DATA


class Unit:
    """Represents a unit on the map."""

    def __init__(self, unit_type, x, y, player):
        """
        Initialize a unit.

        Args:
            unit_type: 'W', 'M', or 'C'
            x: X coordinate on grid
            y: Y coordinate on grid
            player: Player number who owns this unit
        """
        self.type = unit_type
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.player = player
        self.can_move = False
        self.can_attack = False
        self.selected = False
        self.has_moved = False
        self.movement_range = UNIT_DATA[unit_type]['movement']
        self.max_health = UNIT_DATA[unit_type]['health']
        self.health = self.max_health
        self.attack_data = UNIT_DATA[unit_type]['attack']
        self.paralyzed_turns = 0

    def get_attack_damage(self, target_x, target_y):
        """
        Calculate attack damage based on distance to target.

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate

        Returns:
            Attack damage value
        """
        distance = abs(self.x - target_x) + abs(self.y - target_y)

        if self.type == 'M':
            if distance == 1:
                return self.attack_data['adjacent']
            elif distance == 2:
                return self.attack_data['range']
            else:
                return 0
        else:
            if distance == 1:
                return self.attack_data
            else:
                return 0

    def take_damage(self, damage):
        """
        Apply damage to the unit.

        Args:
            damage: Amount of damage to take

        Returns:
            True if unit is still alive, False if dead
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            return False
        return True

    def is_paralyzed(self):
        """Check if this unit is currently paralyzed."""
        return self.paralyzed_turns > 0

    def get_reachable_positions(self, grid_width, grid_height, can_move_to_func):
        """
        Get all positions reachable within movement range using BFS.

        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            can_move_to_func: Function to check if a position is valid for movement

        Returns:
            List of (x, y) tuples for all reachable positions
        """
        reachable = []
        visited = set()
        queue = deque([(self.x, self.y, 0)])
        visited.add((self.x, self.y))

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        while queue:
            curr_x, curr_y, distance = queue.popleft()

            if distance > 0:
                reachable.append((curr_x, curr_y))

            if distance < self.movement_range:
                for dx, dy in directions:
                    new_x = curr_x + dx
                    new_y = curr_y + dy

                    if (new_x, new_y) not in visited:
                        if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
                            if can_move_to_func(new_x, new_y):
                                visited.add((new_x, new_y))
                                queue.append((new_x, new_y, distance + 1))

        return reachable

    def move_to(self, x, y):
        """Move the unit to a new position."""
        self.x = x
        self.y = y
        self.has_moved = True
        self.selected = False

    def cancel_move(self):
        """Cancel the unit's movement and return to original position."""
        if self.has_moved:
            self.x = self.original_x
            self.y = self.original_y
            self.has_moved = False
            return True
        return False

    def end_unit_turn(self):
        """End this unit's turn."""
        self.can_move = False
        self.can_attack = False
        self.selected = False
        self.has_moved = False
        self.original_x = self.x
        self.original_y = self.y

    def to_dict(self):
        """Convert unit to dictionary for serialization."""
        return {
            'type': self.type,
            'x': self.x,
            'y': self.y,
            'player': self.player,
            'health': self.health,
            'paralyzed_turns': self.paralyzed_turns,
            'can_move': self.can_move,
            'can_attack': self.can_attack
        }

    @classmethod
    def from_dict(cls, data):
        """Create unit from dictionary."""
        unit = cls(data['type'], data['x'], data['y'], data['player'])
        unit.health = data['health']
        unit.paralyzed_turns = data.get('paralyzed_turns', 0)
        unit.can_move = data.get('can_move', True)
        unit.can_attack = data.get('can_attack', True)
        unit.original_x = unit.x
        unit.original_y = unit.y
        return unit
