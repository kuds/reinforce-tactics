"""
Unit class representing a game unit.
"""

import heapq
from collections import deque

from reinforcetactics.rules import UNIT_DATA


class Unit:
    """Represents a unit on the map."""

    def __init__(self, unit_type, x, y, player, stats=None):
        """
        Initialize a unit.

        Args:
            unit_type: 'W' (Warrior), 'M' (Mage), 'C' (Cleric), 'A' (Archer),
                       'K' (Knight), 'R' (Rogue), 'S' (Sorcerer), 'B' (Barbarian)
            x: X coordinate on grid
            y: Y coordinate on grid
            player: Player number who owns this unit
            stats: Optional resolved stat block (cost/health/attack/defence/
                movement) for this unit type. When ``None`` the global
                :data:`reinforcetactics.rules.UNIT_DATA` entry is used,
                preserving behaviour for direct/legacy callers. ``GameState``
                passes its per-game resolved table so engine-override sweeps
                (balance experiments) flow through here without mutating the
                shared module constant.
        """
        if stats is None:
            stats = UNIT_DATA[unit_type]
        # Stable per-game identifier. ``None`` here is a placeholder for
        # the legacy direct-constructor path (e.g. ``Unit.from_dict``).
        # ``GameState.create_unit`` assigns a real id from its
        # ``_next_unit_id`` counter before the unit is added to
        # ``self.units``. Replay schema v3 (PR #360 follow-up) keys
        # every recorded action by this id instead of (x, y), so
        # position-drift bugs can't silently mis-route actions in the
        # replay player.
        self.unit_id = None
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
        self.movement_range = stats["movement"]
        self.max_health = stats["health"]
        self.health = self.max_health
        self.attack_data = stats["attack"]
        self.defence = stats["defence"]
        self.paralyzed_turns = 0

        # Knight charge tracking
        self.distance_moved = 0

        # Mage paralyze ability tracking
        self.paralyze_cooldown = 0  # Turns remaining before can use Paralyze again

        # Sorcerer haste ability tracking
        self.haste_cooldown = 0  # Turns remaining before can use Haste again

        # Haste buff tracking (for any unit that receives Haste)
        self.is_hasted = False  # True if unit has extra action this turn
        # True from the moment the engine spends this unit's haste on a new
        # action (GameState._consume_action) until the unit moves or acts
        # again; lets GameState.end_unit_turn tell "finish the action that
        # was just refreshed" (keep the extra action) from a Wait.
        self.haste_refreshed = False

        # Sorcerer buff ability tracking (cooldowns for the Sorcerer)
        self.defence_buff_cooldown = 0  # Turns remaining before can use Defence Buff again
        self.attack_buff_cooldown = 0  # Turns remaining before can use Attack Buff again

        # Buff status tracking (for any unit that receives buffs)
        self.defence_buff_turns = 0  # Turns remaining with defence buff active
        self.attack_buff_turns = 0  # Turns remaining with attack buff active

        # Fog of war: Track which enemy positions were visible when this unit started its action
        # This prevents "move to discover, then attack" exploitation
        self.visible_enemies_at_action_start = None  # Set of (x, y) tuples, or None if not captured
        # Fog of war: this action's move was ambushed, so it can't be
        # cancelled (see GameState.move_unit and GameState.can_cancel_move)
        self.ambushed = False
        # Fog of war: the owner's VisibilityMap as it was before this unit's
        # latest move, kept until the unit acts or the turn ends so
        # GameState.cancel_move can take back what the move revealed.
        self.pre_move_visibility = None

    def get_attack_damage(self, target_x, target_y, on_mountain=False):
        """
        Calculate attack damage based on distance to target.

        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            on_mountain: Whether the unit is on a mountain tile (for Archer range bonus)

        Returns:
            Attack damage value
        """
        distance = abs(self.x - target_x) + abs(self.y - target_y)

        if self.type in ["M", "S"]:
            # Mage and Sorcerer have ranged attacks
            if distance == 1:
                return self.attack_data["adjacent"]
            elif distance == 2:
                return self.attack_data["range"]
            else:
                return 0
        elif self.type == "A":
            # Archer has range 2-3 normally, 2-4 on mountain (cannot attack at distance 1)
            max_range = 4 if on_mountain else 3
            if 2 <= distance <= max_range:
                return self.attack_data
            else:
                return 0
        else:
            if distance == 1:
                return self.attack_data
            else:
                return 0

    def get_attack_range(self, on_mountain=False):
        """
        Get the min and max attack range for this unit.

        Args:
            on_mountain: Whether the unit is on a mountain tile (for Archer range bonus)

        Returns:
            Tuple of (min_range, max_range)
        """
        if self.type in ["M", "S"]:
            # Mage and Sorcerer: distance 1-2
            return (1, 2)
        elif self.type == "A":
            # Archer: distance 2-3, or 2-4 on mountain
            max_range = 4 if on_mountain else 3
            return (2, max_range)
        else:
            # Warrior, Cleric, Barbarian, Knight, Rogue: distance 1 only
            return (1, 1)

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

    def get_reachable_positions(self, grid_width, grid_height, can_move_to_func, move_cost=None, came_from=None):
        """
        Get all positions reachable within movement range.

        Args:
            grid_width: Width of the grid
            grid_height: Height of the grid
            can_move_to_func: Function to check if a position is valid for movement
            move_cost: Optional ``(x, y) -> cost`` of entering a tile (see
                ``find_paths``); None means every tile costs 1
            came_from: Optional dict filled with each reachable position's
                predecessor on the path the search found to it, so a caller
                can rebuild the route a move takes (the fog-of-war ambush
                rule walks it)

        Returns:
            List of (x, y) tuples for all reachable positions, in search order
        """
        return list(self.find_paths(grid_width, grid_height, can_move_to_func, move_cost, came_from))

    # Neighbour order is part of the search order, which fixes the order of
    # move actions (the flat_discrete encoding and every bot tiebreak).
    _DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def find_paths(self, grid_width, grid_height, can_enter, move_cost=None, came_from=None):
        """Every tile this unit can reach this turn, mapped to the tiles stepped to get there.

        A tile is reachable when some path of enterable tiles (``can_enter``)
        costs at most ``movement_range`` in total, each step costing
        ``move_cost(x, y)`` of the tile entered (1 when ``move_cost`` is
        None). The start tile is not included. The step count is that of
        the cheapest path found, which is what a "path" Knight Charge
        counts.

        Uniform cost is a breadth-first search; any other cost is a Dijkstra
        search whose ties are broken by discovery order, so with every cost
        1 it finds exactly the tiles the BFS finds, in the same order
        (tests/test_pathfinding_core.py checks this on every shipped map).

        ``came_from``, if given, is filled with each reached tile's
        predecessor on the path found to it, so a caller can walk the route a
        move takes (the fog-of-war ambush rule does).

        Returns:
            dict ``{(x, y): steps}`` in the order tiles were settled.
        """
        if move_cost is None:
            return self._find_paths_uniform(grid_width, grid_height, can_enter, came_from)

        budget = self.movement_range + 1e-9  # float costs such as 0.5 may accumulate rounding error
        start = (self.x, self.y)
        best_cost = {start: 0}
        steps = {start: 0}
        settled = {}
        seq = 0  # discovery order: the tiebreak that makes cost-1 Dijkstra match BFS
        heap = [(0, seq, self.x, self.y)]
        while heap:
            cost, _, x, y = heapq.heappop(heap)
            if (x, y) in settled or cost > best_cost[(x, y)]:
                continue  # a stale entry for a tile since reached more cheaply
            settled[(x, y)] = steps[(x, y)]
            for dx, dy in self._DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid_width and 0 <= ny < grid_height):
                    continue
                new_cost = cost + move_cost(nx, ny)
                if new_cost > budget or new_cost >= best_cost.get((nx, ny), float("inf")):
                    continue
                if not can_enter(nx, ny):
                    continue
                best_cost[(nx, ny)] = new_cost
                steps[(nx, ny)] = steps[(x, y)] + 1
                if came_from is not None:
                    came_from[(nx, ny)] = (x, y)
                seq += 1
                heapq.heappush(heap, (new_cost, seq, nx, ny))
        del settled[start]
        return settled

    def _find_paths_uniform(self, grid_width, grid_height, can_enter, came_from=None):
        """``find_paths`` with every step costing 1: a plain BFS (the fast default)."""
        reachable = {}
        visited = {(self.x, self.y)}
        queue = deque([(self.x, self.y, 0)])

        while queue:
            curr_x, curr_y, distance = queue.popleft()

            if distance > 0:
                reachable[(curr_x, curr_y)] = distance

            if distance < self.movement_range:
                for dx, dy in self._DIRECTIONS:
                    new_x = curr_x + dx
                    new_y = curr_y + dy

                    if (new_x, new_y) not in visited:
                        if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
                            if can_enter(new_x, new_y):
                                visited.add((new_x, new_y))
                                queue.append((new_x, new_y, distance + 1))
                                if came_from is not None:
                                    came_from[(new_x, new_y)] = (curr_x, curr_y)

        return reachable

    def move_to(self, x, y):
        """Move the unit to a new position."""
        # Calculate Manhattan distance moved (for Knight's Charge ability)
        self.distance_moved = abs(x - self.original_x) + abs(y - self.original_y)
        self.x = x
        self.y = y
        self.has_moved = True
        self.selected = False

    def cancel_move(self):
        """Cancel the unit's movement and return to original position.

        Also resets can_move to True so the unit can move again.
        """
        if self.has_moved:
            self.x = self.original_x
            self.y = self.original_y
            self.has_moved = False
            self.can_move = True  # Allow unit to move again after cancel
            self.distance_moved = 0  # Reset distance for Knight's Charge
            return True
        return False

    def end_unit_turn(self, force_end=False):
        """End this unit's turn.

        If the unit has haste (is_hasted=True) and force_end is False,
        the haste is consumed and the unit gets another full action instead
        of ending its turn.

        Args:
            force_end: If True, always end the turn even if hasted

        Returns:
            bool: True if the unit can still act (haste was consumed),
                  False if the turn actually ended
        """
        # If hasted and not forcing end, consume haste and refresh for another action
        self.haste_refreshed = False
        if self.is_hasted and not force_end:
            self.is_hasted = False
            self.can_move = True
            self.can_attack = True
            self.has_moved = False
            self.original_x = self.x
            self.original_y = self.y
            self.distance_moved = 0
            self.selected = False
            self.visible_enemies_at_action_start = None  # Clear FOW snapshot for new action
            return True  # Unit can still act

        # Normal turn end
        self.can_move = False
        self.can_attack = False
        self.selected = False
        self.has_moved = False
        self.original_x = self.x
        self.original_y = self.y
        self.distance_moved = 0
        self.is_hasted = False
        self.visible_enemies_at_action_start = None  # Clear FOW snapshot
        return False  # Turn ended

    def can_use_paralyze(self):
        """Check if this Mage can use Paralyze ability."""
        return self.type == "M" and self.paralyze_cooldown == 0

    def can_use_haste(self):
        """Check if this Sorcerer can use Haste ability."""
        return self.type == "S" and self.haste_cooldown == 0

    def can_use_defence_buff(self):
        """Check if this Sorcerer can use Defence Buff ability."""
        return self.type == "S" and self.defence_buff_cooldown == 0

    def can_use_attack_buff(self):
        """Check if this Sorcerer can use Attack Buff ability."""
        return self.type == "S" and self.attack_buff_cooldown == 0

    def has_defence_buff(self):
        """Check if this unit has an active defence buff."""
        return self.defence_buff_turns > 0

    def has_attack_buff(self):
        """Check if this unit has an active attack buff."""
        return self.attack_buff_turns > 0

    def to_dict(self):
        """Convert unit to dictionary for serialization."""
        return {
            "unit_id": self.unit_id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "player": self.player,
            "health": self.health,
            "paralyzed_turns": self.paralyzed_turns,
            "paralyze_cooldown": self.paralyze_cooldown,
            "can_move": self.can_move,
            "can_attack": self.can_attack,
            "haste_cooldown": self.haste_cooldown,
            "is_hasted": self.is_hasted,
            "distance_moved": self.distance_moved,
            "defence_buff_cooldown": self.defence_buff_cooldown,
            "attack_buff_cooldown": self.attack_buff_cooldown,
            "defence_buff_turns": self.defence_buff_turns,
            "attack_buff_turns": self.attack_buff_turns,
            "original_x": self.original_x,
            "original_y": self.original_y,
            # end_turn resets a structure the unit stepped off only if it
            # has_moved, so a mid-turn save must keep it.
            "has_moved": self.has_moved,
            # Fog of war: the enemies it may attack this action (those in
            # sight when the action began); None = not captured yet.
            "visible_enemies_at_action_start": (
                sorted([x, y] for x, y in self.visible_enemies_at_action_start)
                if self.visible_enemies_at_action_start is not None
                else None
            ),
            # An ambushed move can't be cancelled, after a reload too.
            "ambushed": self.ambushed,
        }

    @classmethod
    def from_dict(cls, data, stats=None):
        """Create unit from dictionary.

        Args:
            data: A dict written by :meth:`to_dict`
            stats: The stat block to build the unit with, as in ``__init__``.
                ``GameState.from_dict`` passes its game's (engine-override)
                table; ``None`` uses the module defaults. Saved health is
                capped at the resulting ``max_health``.
        """
        unit = cls(data["type"], data["x"], data["y"], data["player"], stats=stats)
        # ``None`` for old saves that pre-date the unit_id field; the
        # owning GameState restores ``_next_unit_id`` so newly-created
        # units after load still get fresh non-colliding ids.
        unit.unit_id = data.get("unit_id")
        unit.health = min(data["health"], unit.max_health)
        unit.paralyzed_turns = data.get("paralyzed_turns", 0)
        unit.paralyze_cooldown = data.get("paralyze_cooldown", 0)
        unit.can_move = data.get("can_move", True)
        unit.can_attack = data.get("can_attack", True)
        unit.haste_cooldown = data.get("haste_cooldown", 0)
        unit.is_hasted = data.get("is_hasted", False)
        unit.distance_moved = data.get("distance_moved", 0)
        unit.defence_buff_cooldown = data.get("defence_buff_cooldown", 0)
        unit.attack_buff_cooldown = data.get("attack_buff_cooldown", 0)
        unit.defence_buff_turns = data.get("defence_buff_turns", 0)
        unit.attack_buff_turns = data.get("attack_buff_turns", 0)
        unit.original_x = data.get("original_x", unit.x)
        unit.original_y = data.get("original_y", unit.y)
        # Saves before has_moved was recorded: a unit away from where its
        # action started has moved (the only case end_turn acts on).
        unit.has_moved = data.get("has_moved", (unit.x, unit.y) != (unit.original_x, unit.original_y))
        snapshot = data.get("visible_enemies_at_action_start")
        unit.visible_enemies_at_action_start = {(x, y) for x, y in snapshot} if snapshot is not None else None
        unit.ambushed = data.get("ambushed", False)
        return unit
