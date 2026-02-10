"""
AI bots for computer opponents with support for all unit types.
"""
from typing import List, Dict, Optional, Tuple
from reinforcetactics.constants import (
    UNIT_DATA, COUNTER_ATTACK_MULTIPLIER,
    CHARGE_BONUS, CHARGE_MIN_DISTANCE, FLANK_BONUS,
    ROGUE_EVADE_CHANCE, ROGUE_FOREST_EVADE_BONUS
)

# Maximum recursion depth for haste-triggered re-actions
MAX_RECURSION_DEPTH = 10


class BotUnitMixin:
    """Shared methods for handling enabled/disabled units across all bots."""

    # Unit categories for strategic decisions
    MELEE_UNITS = ['W', 'K', 'R', 'B']
    RANGED_UNITS = ['A', 'M', 'S']
    SUPPORT_UNITS = ['C', 'S']

    def get_enabled_units(self) -> List[str]:
        """Get list of currently enabled unit types."""
        return self.game_state.enabled_units

    def is_unit_enabled(self, unit_type: str) -> bool:
        """Check if a specific unit type is enabled."""
        return self.game_state.is_unit_type_enabled(unit_type)

    def get_enabled_melee_units(self) -> List[str]:
        """Get enabled melee unit types (W, K, R, B)."""
        return [u for u in self.MELEE_UNITS if self.is_unit_enabled(u)]

    def get_enabled_ranged_units(self) -> List[str]:
        """Get enabled ranged unit types (A, M, S)."""
        return [u for u in self.RANGED_UNITS if self.is_unit_enabled(u)]

    def get_enabled_support_units(self) -> List[str]:
        """Get enabled support unit types (C, S)."""
        return [u for u in self.SUPPORT_UNITS if self.is_unit_enabled(u)]

    def has_charge_units(self) -> bool:
        """Check if Knight (charge ability) is enabled."""
        return self.is_unit_enabled('K')

    def has_flank_units(self) -> bool:
        """Check if Rogue (flank ability) is enabled."""
        return self.is_unit_enabled('R')

    def has_buff_units(self) -> bool:
        """Check if Sorcerer (buff abilities) is enabled."""
        return self.is_unit_enabled('S')

    def has_heal_units(self) -> bool:
        """Check if Cleric (heal ability) is enabled."""
        return self.is_unit_enabled('C')

    def has_paralyze_units(self) -> bool:
        """Check if Mage (paralyze ability) is enabled."""
        return self.is_unit_enabled('M')

    def manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two points."""
        return abs(x1 - x2) + abs(y1 - y2)

    def find_best_move_position(self, unit, target_x, target_y):
        """Find the best position to move towards a target."""
        reachable = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
            )
        )

        if not reachable:
            return None

        best_pos = None
        best_distance = float('inf')

        for pos in reachable:
            distance = self.manhattan_distance(pos[0], pos[1], target_x, target_y)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos

        return best_pos


class SimpleBot(BotUnitMixin):
    """Simple AI bot for player 2 with basic unit type awareness."""

    # Unit purchase priorities (lower = higher priority)
    UNIT_PRIORITIES = {
        'W': 1,  # Warrior - cheap, good for capturing
        'B': 2,  # Barbarian - fast, good mobility
        'A': 3,  # Archer - safe ranged damage
        'K': 4,  # Knight - heavy hitter
        'R': 5,  # Rogue - flanking potential
        'M': 6,  # Mage - ranged + paralyze
        'C': 7,  # Cleric - healing support
        'S': 8,  # Sorcerer - buff support
    }

    def __init__(self, game_state, player=2):
        """
        Initialize the bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
        """
        self.game_state = game_state
        self.bot_player = player

    def take_turn(self):
        """Execute the bot's turn."""
        # Phase 1: Purchase units
        self.purchase_units()

        # Phase 2: Move and act with units
        self.move_and_act_units()

        # Phase 3: End turn
        self.game_state.end_turn()

    def purchase_units(self):
        """Purchase units based on priority from enabled types."""
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions['create_unit']
            # Note: legal_actions already filters by enabled_units

            if not create_actions:
                break

            available_gold = self.game_state.player_gold[self.bot_player]

            # Filter to affordable units
            affordable = [
                a for a in create_actions
                if UNIT_DATA[a['unit_type']]['cost'] <= available_gold
            ]
            if not affordable:
                break

            # Sort by priority (lower = buy first), then by cost (cheaper first)
            affordable.sort(
                key=lambda a: (
                    self.UNIT_PRIORITIES.get(a['unit_type'], 99),
                    UNIT_DATA[a['unit_type']]['cost']
                )
            )

            action = affordable[0]
            self.game_state.create_unit(
                action['unit_type'],
                action['x'],
                action['y'],
                self.bot_player
            )

    def move_and_act_units(self):
        """Move and act with all bot units."""
        bot_units = [
            u for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack)
            and not u.is_paralyzed()
        ]

        for unit in bot_units:
            self.act_with_unit(unit)

    def act_with_unit(self, unit, _depth=0):
        """Determine and execute best action for a single unit."""
        if _depth >= MAX_RECURSION_DEPTH:
            return

        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if (tile.is_capturable() and tile.player != self.bot_player and
                tile.health < tile.max_health):
            self.game_state.seize(unit)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return

        # Cleric: try to heal damaged allies or cure paralyzed allies
        if unit.type == 'C' and unit.can_attack:
            if self._try_cleric_abilities(unit, _depth):
                return

        # Mage: try to paralyze high-value enemies before normal attack
        if unit.type == 'M' and unit.can_attack:
            if self._try_mage_paralyze(unit, _depth):
                return

        # Find best target
        target = self.find_best_target(unit)

        if target:
            target_type, target_obj, _ = target

            if target_type == 'enemy_unit':
                self.attack_enemy(unit, target_obj, _depth)
            elif target_type in ['enemy_tower', 'enemy_building', 'enemy_hq']:
                self.move_to_and_seize(unit, target_obj, _depth)
        else:
            can_still_act = unit.end_unit_turn()
            if can_still_act:
                self.act_with_unit(unit, _depth + 1)

    def _try_cleric_abilities(self, unit, _depth=0):
        """Try Cleric heal or cure on nearby allies. Returns True if ability used."""
        # Priority 1: Cure paralyzed allies
        curable = self.game_state.mechanics.get_curable_allies(unit, self.game_state.units)
        if curable:
            target = curable[0]
            self.game_state.cure(unit, target)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return True

        # Priority 2: Heal damaged allies
        healable = self.game_state.mechanics.get_healable_allies(unit, self.game_state.units)
        if healable:
            # Heal the most damaged ally
            target = min(healable, key=lambda a: a.health)
            self.game_state.heal(unit, target)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
            return True

        return False

    def _try_mage_paralyze(self, unit, _depth=0):
        """Try Mage paralyze on a high-value target. Returns True if used."""
        if not unit.can_use_paralyze():
            return False

        enemies = [
            u for u in self.game_state.units
            if u.player != self.bot_player and u.health > 0
            and not u.is_paralyzed()
        ]
        if not enemies:
            return False

        # Find enemies in range (Mage range 1-2)
        in_range = []
        for enemy in enemies:
            dist = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
            if 1 <= dist <= 2:
                in_range.append(enemy)

        if not in_range:
            return False

        # Paralyze the most expensive enemy in range
        target = max(in_range, key=lambda e: UNIT_DATA[e.type]['cost'])
        self.game_state.paralyze(unit, target)
        if unit.can_move or unit.can_attack:
            self.act_with_unit(unit, _depth + 1)
        return True

    def find_best_target(self, unit):
        """Find the best target for a unit (enemy unit or structure)."""
        opponent_has_bases = any(
            tile.player and tile.player != self.bot_player and tile.type in ['b', 't']
            for row in self.game_state.grid.tiles
            for tile in row
        )

        opponent_has_units = any(
            u.player != self.bot_player
            for u in self.game_state.units
        )

        # Find enemy units
        enemy_units = [
            (u, self.manhattan_distance(unit.x, unit.y, u.x, u.y))
            for u in self.game_state.units if u.player != self.bot_player
        ]

        # Find capturable structures (enemy-owned and neutral)
        enemy_structures = []
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.player != self.bot_player:
                    dist = self.manhattan_distance(unit.x, unit.y, tile.x, tile.y)
                    if tile.type == 't':
                        enemy_structures.append(('enemy_tower', tile, dist))
                    elif tile.type == 'b':
                        enemy_structures.append(('enemy_building', tile, dist))
                    elif tile.type == 'h' and tile.player is not None and (
                        not opponent_has_bases or not opponent_has_units
                    ):
                        enemy_structures.append(('enemy_hq', tile, dist))

        # Combine targets
        all_targets = [('enemy_unit', u, d) for u, d in enemy_units]
        all_targets.extend(enemy_structures)

        if not all_targets:
            return None

        # Sort by distance, prioritize buildings/towers
        def sort_key(target):
            target_type, _target_obj, distance = target
            priority = (
                0 if target_type in ['enemy_building', 'enemy_tower', 'enemy_hq']
                else 1
            )
            return (distance, priority)

        all_targets.sort(key=sort_key)
        return all_targets[0]

    def attack_enemy(self, unit, enemy, _depth=0):
        """Attack an enemy unit with unit-type awareness."""
        distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)

        # Handle Archer (range 2-3, cannot attack adjacent)
        if unit.type == 'A':
            self._attack_as_archer(unit, enemy, distance)
            return

        # Handle Mage/Sorcerer (range 1-2, prefer ranged)
        if unit.type in ['M', 'S']:
            self._attack_as_ranged_caster(unit, enemy, distance)
            return

        # Standard melee attack (W, K, R, B, C)
        if distance == 1:
            self.game_state.attack(unit, enemy)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
        else:
            target_pos = self.find_best_move_position(unit, enemy.x, enemy.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
                if new_distance == 1:
                    self.game_state.attack(unit, enemy)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                else:
                    can_still_act = unit.end_unit_turn()
                    if can_still_act:
                        self.act_with_unit(unit, _depth + 1)
            else:
                can_still_act = unit.end_unit_turn()
                if can_still_act:
                    self.act_with_unit(unit, _depth + 1)

    def _attack_as_archer(self, unit, enemy, distance):
        """Handle Archer attacks (range 2-3, cannot attack at distance 1)."""
        # Check if on mountain for extended range (2-4)
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        max_range = 4 if tile.type == 'm' else 3
        min_range = 2

        # Already in valid range
        if min_range <= distance <= max_range:
            self.game_state.attack(unit, enemy)
            return

        # Need to move to valid range
        target_pos = self._find_ranged_attack_position(unit, enemy, min_range, max_range)
        if target_pos:
            self.game_state.move_unit(unit, target_pos[0], target_pos[1])
            new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
            if min_range <= new_distance <= max_range:
                self.game_state.attack(unit, enemy)
            else:
                unit.end_unit_turn()
        else:
            unit.end_unit_turn()

    def _attack_as_ranged_caster(self, unit, enemy, distance):
        """Handle Mage/Sorcerer attacks (range 1-2, prefer distance 2)."""
        # Can attack at distance 1 or 2
        if 1 <= distance <= 2:
            self.game_state.attack(unit, enemy)
            return

        # Need to move into range
        target_pos = self._find_ranged_attack_position(unit, enemy, 1, 2)
        if target_pos:
            self.game_state.move_unit(unit, target_pos[0], target_pos[1])
            new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
            if 1 <= new_distance <= 2:
                self.game_state.attack(unit, enemy)
            else:
                unit.end_unit_turn()
        else:
            unit.end_unit_turn()

    def _find_ranged_attack_position(
        self, unit, enemy, min_range: int, max_range: int
    ) -> Optional[Tuple[int, int]]:
        """Find a position from which unit can attack enemy at valid range."""
        reachable = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
            )
        )

        if not reachable:
            return None

        # Find positions within attack range of enemy
        valid_positions = []
        for pos in reachable:
            dist = self.manhattan_distance(pos[0], pos[1], enemy.x, enemy.y)
            if min_range <= dist <= max_range:
                valid_positions.append((pos, dist))

        if not valid_positions:
            # No valid attack position, move closer
            return self.find_best_move_position(unit, enemy.x, enemy.y)

        # Prefer positions at max range (safer)
        valid_positions.sort(key=lambda x: -x[1])
        return valid_positions[0][0]

    def move_to_and_seize(self, unit, structure, _depth=0):
        """Move towards and seize a structure."""
        if unit.x == structure.x and unit.y == structure.y:
            self.game_state.seize(unit)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit, _depth + 1)
        else:
            target_pos = self.find_best_move_position(unit, structure.x, structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == structure.x and unit.y == structure.y:
                    self.game_state.seize(unit)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                else:
                    can_still_act = unit.end_unit_turn()
                    if can_still_act:
                        self.act_with_unit(unit, _depth + 1)
            else:
                can_still_act = unit.end_unit_turn()
                if can_still_act:
                    self.act_with_unit(unit, _depth + 1)


class MediumBot(BotUnitMixin):
    """Medium difficulty AI bot with improved strategic decision-making."""

    # Unit purchase priorities for MediumBot (lower = higher priority)
    # Priorities consider tactical value and cost efficiency
    UNIT_PRIORITIES = {
        'W': (0, 'capture'),    # Warrior - cheap, good HP for capturing
        'B': (1, 'mobility'),   # Barbarian - fast capturing
        'A': (2, 'ranged'),     # Archer - safe ranged damage
        'K': (3, 'damage'),     # Knight - heavy damage with charge
        'R': (4, 'flank'),      # Rogue - flanking potential
        'M': (5, 'control'),    # Mage - ranged + paralyze
        'C': (6, 'support'),    # Cleric - healing
        'S': (7, 'buff'),       # Sorcerer - buff support
    }

    def __init__(self, game_state, player=2):
        """
        Initialize the bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
        """
        self.game_state = game_state
        self.bot_player = player

    def take_turn(self):
        """Execute the bot's turn with improved strategy."""
        # Phase 1: Purchase units - maximize unit production
        self.purchase_units()

        # Phase 2: Move and act with units using coordinated strategy
        self.move_and_act_units()

        # Phase 3: End turn
        self.game_state.end_turn()

    def find_our_hq(self):
        """
        Locate the bot's headquarters.

        Returns:
            Tuple of (x, y) for HQ location, or None if not found
        """
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type == 'h' and tile.player == self.bot_player:
                    return (tile.x, tile.y)
        return None

    def get_structure_priority(self, structure):
        """
        Score structures by proximity to HQ and income value.

        Args:
            structure: Tile object representing a structure

        Returns:
            Priority score (lower is better)
        """
        # Find our HQ
        our_hq = self.find_our_hq()
        if not our_hq:
            # Fallback to simple distance
            return self.manhattan_distance(0, 0, structure.x, structure.y)

        # Distance from structure to our HQ
        distance_to_hq = self.manhattan_distance(our_hq[0], our_hq[1], structure.x, structure.y)

        # Income value (higher income = higher priority = lower score)
        income_weights = {'h': 150, 'b': 100, 't': 50}
        income_bonus = income_weights.get(structure.type, 0)

        # Lower score = higher priority
        # Prioritize closer structures and higher income
        priority = distance_to_hq - (income_bonus / 10.0)
        return priority

    def purchase_units(self):
        """Purchase units based on priority from all enabled types."""
        # Keep buying units until we can't afford any more
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions['create_unit']
            # Note: legal_actions already filters by enabled_units

            if not create_actions:
                break

            # Available gold
            available_gold = self.game_state.player_gold[self.bot_player]

            # Find affordable units
            affordable_actions = [
                action for action in create_actions
                if UNIT_DATA[action['unit_type']]['cost'] <= available_gold
            ]

            if not affordable_actions:
                break

            # Sort by priority (uses UNIT_PRIORITIES), then by cost
            def unit_priority(action):
                unit_type = action['unit_type']
                cost = UNIT_DATA[unit_type]['cost']
                priority = self.UNIT_PRIORITIES.get(unit_type, (99, 'unknown'))[0]
                return (priority, cost)

            affordable_actions.sort(key=unit_priority)

            # Buy the top priority affordable unit
            action = affordable_actions[0]
            self.game_state.create_unit(
                action['unit_type'],
                action['x'],
                action['y'],
                self.bot_player
            )

    def move_and_act_units(self):
        """Move and act with all bot units using coordinated strategy."""
        # Get all bot units that can act
        bot_units = [
            u for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack)
            and not u.is_paralyzed()
        ]

        # First, coordinate attacks to kill targets
        self.coordinate_attacks(bot_units)

        # Then, act with remaining units
        for unit in bot_units:
            if unit.can_move or unit.can_attack:
                self.act_with_unit(unit)

    def find_killable_targets(self, available_units):
        """
        Identify enemies that can be killed this turn with coordinated attacks.

        Args:
            available_units: List of units that can still act

        Returns:
            List of (enemy, attackers) tuples where attackers can kill the enemy
        """
        killable = []
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]

        for enemy in enemy_units:
            # Find all units that can attack this enemy
            potential_attackers = []
            for unit in available_units:
                if not (unit.can_move or unit.can_attack):
                    continue

                # Check if unit can reach and attack enemy
                attackable = self.game_state.mechanics.get_attackable_enemies(
                    unit, [enemy], self.game_state.grid
                )
                if enemy in attackable:
                    # Direct attack possible
                    potential_attackers.append(unit)
                else:
                    # Check if can move and attack
                    reachable = unit.get_reachable_positions(
                        self.game_state.grid.width,
                        self.game_state.grid.height,
                        lambda x, y: self.game_state.mechanics.can_move_to_position(
                            x, y, self.game_state.grid, self.game_state.units,
                            moving_unit=unit, is_destination=False
                        )
                    )

                    for pos in reachable:
                        # Temporarily check if attacking from this position is possible
                        old_x, old_y = unit.x, unit.y
                        unit.x, unit.y = pos[0], pos[1]

                        attackable_from_pos = self.game_state.mechanics.get_attackable_enemies(
                            unit, [enemy], self.game_state.grid
                        )

                        unit.x, unit.y = old_x, old_y

                        if enemy in attackable_from_pos:
                            potential_attackers.append(unit)
                            break

            if potential_attackers:
                # Calculate total damage
                total_damage = 0
                for attacker in potential_attackers:
                    tile = self.game_state.grid.get_tile(attacker.x, attacker.y)
                    on_mountain = tile.type == 'm'
                    damage = attacker.get_attack_damage(enemy.x, enemy.y, on_mountain)
                    total_damage += damage

                # Check if we can kill the enemy
                if total_damage >= enemy.health:
                    # Find minimal set of attackers needed
                    attackers_needed = []
                    damage_so_far = 0
                    for attacker in potential_attackers:
                        tile = self.game_state.grid.get_tile(attacker.x, attacker.y)
                        on_mountain = tile.type == 'm'
                        damage = attacker.get_attack_damage(enemy.x, enemy.y, on_mountain)
                        attackers_needed.append(attacker)
                        damage_so_far += damage
                        if damage_so_far >= enemy.health:
                            break

                    killable.append((enemy, attackers_needed))

        return killable

    def coordinate_attacks(self, bot_units):
        """
        Plan and execute multi-unit attacks on single targets.

        Args:
            bot_units: List of bot units that can act
        """
        # Find killable targets
        killable_targets = self.find_killable_targets(bot_units)

        if not killable_targets:
            return

        # Prioritize targets: enemies on structures, high-value units, etc.
        def target_priority(item):
            enemy, attackers = item
            # Check if enemy is capturing a structure
            tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
            if tile.is_capturable() and tile.player != self.bot_player and tile.health < tile.max_health:
                return (0,)  # Highest priority - tuple for consistent sorting
            # Otherwise, prioritize by unit cost (kill expensive units first)
            cost = UNIT_DATA[enemy.type]['cost']
            # Also consider minimizing overkill (fewer attackers needed = better)
            return (1, -cost, len(attackers))

        killable_targets.sort(key=target_priority)

        # Execute coordinated attacks
        for enemy, attackers in killable_targets:
            # Check if enemy is still alive and attackers are still available
            if enemy.health <= 0:
                continue

            for attacker in attackers:
                if not (attacker.can_move or attacker.can_attack):
                    continue

                # Check if can attack directly
                attackable = self.game_state.mechanics.get_attackable_enemies(
                    attacker, [enemy], self.game_state.grid
                )

                if enemy in attackable and enemy.health > 0:
                    self.game_state.attack(attacker, enemy)
                else:
                    # Move towards enemy and attack if possible
                    target_pos = self.find_best_move_position(attacker, enemy.x, enemy.y)
                    if target_pos:
                        self.game_state.move_unit(attacker, target_pos[0], target_pos[1])

                        # Try to attack after moving
                        attackable_after_move = self.game_state.mechanics.get_attackable_enemies(
                            attacker, [enemy], self.game_state.grid
                        )
                        if enemy in attackable_after_move and enemy.health > 0:
                            self.game_state.attack(attacker, enemy)

    def find_contested_structures(self):
        """
        Find structures currently being captured by enemies.

        Returns:
            List of (structure, enemy_unit, capture_progress) tuples
        """
        contested = []

        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.player != self.bot_player:
                    # Check if health is below max (being captured)
                    if tile.health < tile.max_health:
                        # Find enemy unit on this structure
                        enemy_on_structure = None
                        for unit in self.game_state.units:
                            if unit.player != self.bot_player and unit.x == tile.x and unit.y == tile.y:
                                enemy_on_structure = unit
                                break

                        if enemy_on_structure:
                            # Calculate capture progress (0 to 1)
                            progress = 1.0 - (tile.health / tile.max_health)
                            contested.append((tile, enemy_on_structure, progress))

        return contested

    def calculate_attack_value(self, attacker, target, move_distance: int = 0):
        """
        Evaluate attack efficiency considering damage dealt, received, and abilities.

        Args:
            attacker: Unit that would attack
            target: Enemy unit to attack
            move_distance: Distance moved before attacking (for Knight charge)

        Returns:
            Value score (higher is better)
        """
        # Calculate damage attacker would deal
        attacker_tile = self.game_state.grid.get_tile(attacker.x, attacker.y)
        on_mountain = attacker_tile.type == 'm'
        damage_dealt = attacker.get_attack_damage(target.x, target.y, on_mountain)

        # Apply Knight charge bonus if applicable
        if attacker.type == 'K' and self.has_charge_units():
            if move_distance >= CHARGE_MIN_DISTANCE:
                damage_dealt = int(damage_dealt * (1 + CHARGE_BONUS))

        # Apply Rogue flank bonus if applicable
        if attacker.type == 'R' and self.has_flank_units():
            if self._can_flank(attacker, target):
                damage_dealt = int(damage_dealt * (1 + FLANK_BONUS))

        # Check if this kills the target (no counter-attack)
        if damage_dealt >= target.health:
            # Killing is very valuable - no counter-attack
            return 1000 + damage_dealt

        # Calculate counter-attack damage if target survives
        counter_damage = 0
        # Check if target can counter-attack
        target_tile = self.game_state.grid.get_tile(target.x, target.y)
        target_on_mountain = target_tile.type == 'm'
        target_damage = target.get_attack_damage(attacker.x, attacker.y, target_on_mountain)

        # Archers can't be counter-attacked by melee units
        distance = self.manhattan_distance(attacker.x, attacker.y, target.x, target.y)
        if distance > 1 and target.type not in ['M', 'A', 'S']:
            # Target can't counter-attack ranged attacker
            counter_damage = 0
        elif target_damage > 0:
            # Counter-attacks deal reduced damage
            counter_damage = int(target_damage * COUNTER_ATTACK_MULTIPLIER)

            # Rogue evade reduces expected counter-damage
            if attacker.type == 'R' and self.has_flank_units():
                evade_chance = ROGUE_EVADE_CHANCE
                if attacker_tile.type == 'f':  # Forest
                    evade_chance += ROGUE_FOREST_EVADE_BONUS
                counter_damage = int(counter_damage * (1 - evade_chance))

        # Value = damage dealt - damage received
        # Also consider unit costs
        attacker_cost = UNIT_DATA[attacker.type]['cost']
        target_cost = UNIT_DATA[target.type]['cost']

        # Prefer favorable trades
        value = damage_dealt - counter_damage
        # Bonus for attacking high-value targets
        value += target_cost / 100.0
        # Penalty for risking high-value units
        value -= (counter_damage * attacker_cost) / 1000.0

        return value

    def _can_flank(self, attacker, target) -> bool:
        """Check if attacker can flank target (target adjacent to a friendly unit)."""
        if attacker.type != 'R':
            return False

        # Check if any friendly unit is adjacent to target
        for unit in self.game_state.units:
            if unit.player == self.bot_player and unit != attacker:
                dist = self.manhattan_distance(unit.x, unit.y, target.x, target.y)
                if dist == 1:
                    return True
        return False

    def _find_flank_targets(self, rogue) -> List:
        """Find enemies that can be flanked by the Rogue."""
        if rogue.type != 'R' or not self.has_flank_units():
            return []

        flankable = []
        enemies = [u for u in self.game_state.units
                   if u.player != self.bot_player and u.health > 0]

        for enemy in enemies:
            if self._can_flank(rogue, enemy):
                flankable.append(enemy)

        return flankable

    def act_with_unit(self, unit, _depth=0):
        """Execute actions for a single unit based on strategic priorities."""
        if _depth >= MAX_RECURSION_DEPTH:
            return

        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if (tile.is_capturable() and tile.player != self.bot_player and
                tile.health < tile.max_health):
            self.game_state.seize(unit)
            return

        # Priority 1: Interrupt enemy captures
        contested = self.find_contested_structures()
        if contested:
            # Sort by capture progress (higher progress = higher priority)
            contested.sort(key=lambda x: x[2], reverse=True)

            for _, enemy_unit, __ in contested:
                # Check if we can attack this enemy
                attackable = self.game_state.mechanics.get_attackable_enemies(
                    unit, [enemy_unit], self.game_state.grid
                )

                if enemy_unit in attackable:
                    # Attack to interrupt capture
                    self.game_state.attack(unit, enemy_unit)
                    return

                # Try to move towards enemy and attack
                target_pos = self.find_best_move_position(unit, enemy_unit.x, enemy_unit.y)
                if target_pos:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    # Check if can attack after moving
                    attackable_after = self.game_state.mechanics.get_attackable_enemies(
                        unit, [enemy_unit], self.game_state.grid
                    )
                    if enemy_unit in attackable_after:
                        self.game_state.attack(unit, enemy_unit)
                        return

        # Priority 2: Attack enemies with good value trades
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player and u.health > 0]
        if enemy_units:
            # Evaluate all possible attacks
            best_value = -1000
            best_target = None

            for enemy in enemy_units:
                attackable = self.game_state.mechanics.get_attackable_enemies(
                    unit, [enemy], self.game_state.grid
                )

                if enemy in attackable:
                    value = self.calculate_attack_value(unit, enemy)
                    if value > best_value:
                        best_value = value
                        best_target = enemy

            # Attack if value is positive
            if best_target and best_value > 0:
                self.game_state.attack(unit, best_target)
                return

        # Priority 3: Capture structures (prioritize by proximity to HQ)
        capturable_structures = []
        for row in self.game_state.grid.tiles:
            for structure in row:
                if structure.is_capturable() and structure.player != self.bot_player:
                    priority = self.get_structure_priority(structure)
                    capturable_structures.append((structure, priority))

        capturable_structures.sort(key=lambda x: x[1])

        if capturable_structures:
            target_structure = capturable_structures[0][0]

            # Check if already on structure
            if unit.x == target_structure.x and unit.y == target_structure.y:
                self.game_state.seize(unit)
                # Check if unit can act again (haste)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit(unit, _depth + 1)
                return

            # Move towards structure
            target_pos = self.find_best_move_position(unit, target_structure.x, target_structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                # Check if reached structure
                if unit.x == target_structure.x and unit.y == target_structure.y:
                    self.game_state.seize(unit)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit(unit, _depth + 1)
                    return

        # Fallback: End turn
        can_still_act = unit.end_unit_turn()
        if can_still_act:
            self.act_with_unit(unit, _depth + 1)


class AdvancedBot(MediumBot):
    """Advanced AI bot extending MediumBot with map analysis and enhanced tactics."""

    # Full composition targets for all 8 unit types
    # These will be dynamically adjusted based on enabled units
    FULL_COMPOSITION_TARGETS = {
        'W': 0.25,  # Warriors - capturing, frontline
        'A': 0.20,  # Archers - ranged damage
        'M': 0.15,  # Mages - ranged + paralyze
        'K': 0.10,  # Knights - heavy charge damage
        'R': 0.10,  # Rogues - flanking assassin
        'B': 0.08,  # Barbarians - fast mobility
        'C': 0.07,  # Clerics - healing support
        'S': 0.05,  # Sorcerers - buff support
    }

    def __init__(self, game_state, player=2):
        """
        Initialize the AdvancedBot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
        """
        super().__init__(game_state, player)

        # Map analysis cache
        self.map_analyzed = False
        self.hq_positions = {}
        self.defensive_positions = []  # Mountains
        self.forest_positions = []      # Forests (for Rogue evade bonus)
        self.turn_count = 0

    def take_turn(self):
        """Execute the bot's turn with enhanced strategy."""
        self.turn_count += 1

        # Phase 1: Analyze map on first turn
        if not self.map_analyzed:
            self.analyze_map()
            self.map_analyzed = True

        # Phase 2: Use enhanced purchase strategy
        self.purchase_units_enhanced()

        # Phase 3: Enhanced unit actions with special abilities and better tactics
        self.move_and_act_units_enhanced()

        # Phase 4: End turn
        self.game_state.end_turn()

    def analyze_map(self):
        """Pre-compute strategic map features on first turn."""
        grid = self.game_state.grid

        # Identify HQ positions
        for row in grid.tiles:
            for tile in row:
                if tile.type == 'h' and tile.player:
                    self.hq_positions[tile.player] = (tile.x, tile.y)

        # Identify defensive positions (mountains for Archer range bonus)
        self.defensive_positions = []
        # Identify forest positions (for Rogue evade bonus)
        self.forest_positions = []

        for row in grid.tiles:
            for tile in row:
                if tile.type == 'm':  # Mountains
                    self.defensive_positions.append((tile.x, tile.y))
                elif tile.type == 'f':  # Forests
                    self.forest_positions.append((tile.x, tile.y))

    def get_dynamic_composition_targets(self) -> Dict[str, float]:
        """Calculate target composition based on enabled units."""
        # Filter to only enabled units
        enabled = self.get_enabled_units()
        enabled_targets = {
            k: v for k, v in self.FULL_COMPOSITION_TARGETS.items()
            if k in enabled
        }

        # Redistribute disabled unit ratios proportionally
        if enabled_targets:
            total_enabled = sum(enabled_targets.values())
            if total_enabled > 0:
                # Normalize to 1.0
                enabled_targets = {
                    k: v / total_enabled for k, v in enabled_targets.items()
                }

        return enabled_targets

    def purchase_units_enhanced(self):
        """Enhanced unit purchasing with dynamic composition for all enabled units."""
        # Get dynamic composition targets based on enabled units
        target_ratios = self.get_dynamic_composition_targets()

        # Count existing unit types (only enabled ones)
        my_units = [u for u in self.game_state.units if u.player == self.bot_player]
        enabled = self.get_enabled_units()
        unit_counts = {ut: 0 for ut in enabled}
        for unit in my_units:
            if unit.type in unit_counts:
                unit_counts[unit.type] += 1

        total_units = len(my_units)

        # Enhanced composition: buy units to match target ratios
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions['create_unit']
            # Note: legal_actions already filters by enabled_units

            if not create_actions:
                break

            available_gold = self.game_state.player_gold[self.bot_player]
            affordable_actions = [
                a for a in create_actions
                if UNIT_DATA[a['unit_type']]['cost'] <= available_gold
            ]

            if not affordable_actions:
                break

            # Find unit type most below its target ratio
            best_action = None
            best_priority = -float('inf')

            for action in affordable_actions:
                unit_type = action['unit_type']
                if unit_type not in target_ratios:
                    continue

                # Calculate how far below target we are
                current_ratio = unit_counts.get(unit_type, 0) / max(1, total_units + 1)
                target_ratio = target_ratios[unit_type]
                priority = target_ratio - current_ratio

                # Support units (C, S) need at least 3 units before buying
                if unit_type in ['C', 'S'] and total_units < 3:
                    priority = -1

                if priority > best_priority:
                    best_priority = priority
                    best_action = action

            if best_action and best_priority > -1:
                self.game_state.create_unit(
                    best_action['unit_type'],
                    best_action['x'],
                    best_action['y'],
                    self.bot_player
                )
                unit_counts[best_action['unit_type']] = unit_counts.get(
                    best_action['unit_type'], 0
                ) + 1
                total_units += 1
            else:
                # Fallback: buy any affordable unit
                if affordable_actions:
                    # Prefer cheaper units for economy
                    affordable_actions.sort(key=lambda a: UNIT_DATA[a['unit_type']]['cost'])
                    action = affordable_actions[0]
                    self.game_state.create_unit(
                        action['unit_type'],
                        action['x'],
                        action['y'],
                        self.bot_player
                    )
                    unit_counts[action['unit_type']] = unit_counts.get(
                        action['unit_type'], 0
                    ) + 1
                    total_units += 1
                else:
                    break

    def move_and_act_units_enhanced(self):
        """Enhanced version of MediumBot's unit movement with special abilities."""
        # Get all bot units that can act
        bot_units = [
            u for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack)
            and not u.is_paralyzed()
        ]

        # First, coordinate attacks to kill targets (from MediumBot)
        self.coordinate_attacks(bot_units)

        # Then, act with remaining units (enhanced)
        for unit in bot_units:
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit)

    def act_with_unit_enhanced(self, unit, _depth=0):
        """Enhanced version of MediumBot's act_with_unit with superior tactics."""
        if _depth >= MAX_RECURSION_DEPTH:
            return

        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if (tile.is_capturable() and tile.player != self.bot_player and
                tile.health < tile.max_health):
            self.game_state.seize(unit)
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit, _depth + 1)
            return

        # Try special abilities first (Cleric heal, Mage paralyze, Sorcerer buffs)
        if self.try_use_special_ability(unit):
            # Check if unit can act again (haste)
            if unit.can_move or unit.can_attack:
                self.act_with_unit_enhanced(unit, _depth + 1)
            return

        # PRIORITY 1: Knight charge attack (move 3+ tiles for +50% damage)
        if unit.type == 'K' and self.has_charge_units():
            if self._try_knight_charge(unit):
                return

        # PRIORITY 2: Rogue flank attack (+50% damage when target adjacent to ally)
        if unit.type == 'R' and self.has_flank_units():
            if self._try_rogue_flank(unit):
                return

        # PRIORITY 3: Position Rogues in forests for evade bonus
        if unit.type == 'R' and self.has_flank_units() and tile.type != 'f':
            if self._try_rogue_forest_position(unit):
                return

        # PRIORITY 4: Position on mountains for attack bonus (Archers get range bonus)
        if unit.type in ['W', 'B', 'A', 'M', 'K', 'S'] and tile.type != 'm':
            nearby_mountains = [
                pos for pos in self.defensive_positions
                if self.manhattan_distance(unit.x, unit.y, pos[0], pos[1]) <= unit.movement_range
            ]
            if nearby_mountains:
                # Check if an enemy is nearby
                enemy_units = [u for u in self.game_state.units
                              if u.player != self.bot_player and u.health > 0]
                for enemy in enemy_units:
                    if self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y) <= 4:
                        # Move to mountain to get attack bonus
                        for mountain_pos in nearby_mountains:
                            target_pos = self.find_best_move_position(
                                unit, mountain_pos[0], mountain_pos[1]
                            )
                            if target_pos and target_pos == mountain_pos:
                                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                                # Try to attack after positioning
                                if self.try_ranged_attack(unit):
                                    # Check if unit can act again (haste)
                                    if unit.can_move or unit.can_attack:
                                        self.act_with_unit_enhanced(unit, _depth + 1)
                                    return
                                break

        # PRIORITY 5: Ranged attacks (Archers/Mages/Sorcerers should attack from range)
        if unit.type in self.get_enabled_ranged_units() and self.try_ranged_attack(unit):
            return

        # PRIORITY 6: Move to attack range and attack
        enemy_units = [u for u in self.game_state.units
                      if u.player != self.bot_player and u.health > 0]

        if enemy_units:
            # Find closest attackable enemy
            best_target = None
            best_score = -float('inf')

            for enemy in enemy_units:
                # Calculate if we can attack or move to attack
                attackable = self.game_state.mechanics.get_attackable_enemies(
                    unit, [enemy], self.game_state.grid
                )

                if enemy in attackable:
                    # Can attack now
                    value = self.calculate_attack_value(unit, enemy)
                    # Bonus for killing
                    on_mountain = tile.type == 'm'
                    damage = unit.get_attack_damage(enemy.x, enemy.y, on_mountain)
                    if damage >= enemy.health:
                        value += 500

                    if value > best_score:
                        best_score = value
                        best_target = enemy

            if best_target and best_score > -500:  # More aggressive threshold
                self.game_state.attack(unit, best_target)
                # Check if unit can act again (haste)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit_enhanced(unit, _depth + 1)
                return

            # Try to move towards nearest enemy
            if enemy_units:
                nearest_enemy = min(
                    enemy_units,
                    key=lambda e: self.manhattan_distance(unit.x, unit.y, e.x, e.y)
                )
                target_pos = self.find_best_move_position(
                    unit, nearest_enemy.x, nearest_enemy.y
                )
                if target_pos:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    # Try to attack after moving
                    attackable_after = self.game_state.mechanics.get_attackable_enemies(
                        unit, enemy_units, self.game_state.grid
                    )
                    if attackable_after:
                        best_after_move = max(
                            attackable_after,
                            key=lambda e: self.calculate_attack_value(unit, e)
                        )
                        self.game_state.attack(unit, best_after_move)
                        # Check if unit can act again (haste)
                        if unit.can_move or unit.can_attack:
                            self.act_with_unit_enhanced(unit, _depth + 1)
                        return

        # PRIORITY 7: Interrupt enemy captures (from MediumBot)
        contested = self.find_contested_structures()
        if contested:
            contested.sort(key=lambda x: x[2], reverse=True)
            for _, enemy_unit, __ in contested:
                target_pos = self.find_best_move_position(unit, enemy_unit.x, enemy_unit.y)
                if target_pos:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    attackable = self.game_state.mechanics.get_attackable_enemies(
                        unit, [enemy_unit], self.game_state.grid
                    )
                    if enemy_unit in attackable:
                        self.game_state.attack(unit, enemy_unit)
                        # Check if unit can act again (haste)
                        if unit.can_move or unit.can_attack:
                            self.act_with_unit_enhanced(unit, _depth + 1)
                        return

        # PRIORITY 5: Capture structures (fallback)
        capturable_structures = []
        for row in self.game_state.grid.tiles:
            for structure in row:
                if structure.is_capturable() and structure.player != self.bot_player:
                    priority = self.get_structure_priority(structure)
                    capturable_structures.append((structure, priority))

        capturable_structures.sort(key=lambda x: x[1])

        if capturable_structures:
            target_structure = capturable_structures[0][0]
            if unit.x == target_structure.x and unit.y == target_structure.y:
                self.game_state.seize(unit)
                # Check if unit can act again (haste)
                if unit.can_move or unit.can_attack:
                    self.act_with_unit_enhanced(unit, _depth + 1)
                return

            target_pos = self.find_best_move_position(unit, target_structure.x, target_structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == target_structure.x and unit.y == target_structure.y:
                    self.game_state.seize(unit)
                    # Check if unit can act again (haste)
                    if unit.can_move or unit.can_attack:
                        self.act_with_unit_enhanced(unit, _depth + 1)
                    return

        # Fallback: End turn
        can_still_act = unit.end_unit_turn()
        if can_still_act:
            self.act_with_unit_enhanced(unit, _depth + 1)

    def _try_knight_charge(self, unit) -> bool:
        """Attempt Knight charge attack for +50% damage (requires 3+ tile move)."""
        if unit.type != 'K' or not self.has_charge_units():
            return False

        enemies = [u for u in self.game_state.units
                   if u.player != self.bot_player and u.health > 0]

        if not enemies:
            return False

        # Get all reachable positions
        reachable = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
            )
        )

        best_charge = None
        best_value = -float('inf')

        for pos in reachable:
            # Calculate move distance
            move_distance = self.manhattan_distance(unit.x, unit.y, pos[0], pos[1])
            if move_distance < CHARGE_MIN_DISTANCE:
                continue  # Not a valid charge

            # Check if we can attack any enemy from this position
            for enemy in enemies:
                attack_distance = self.manhattan_distance(pos[0], pos[1], enemy.x, enemy.y)
                if attack_distance == 1:  # Knight is melee
                    # Calculate value with charge bonus
                    value = self.calculate_attack_value(unit, enemy, move_distance)
                    if value > best_value:
                        best_value = value
                        best_charge = (pos, enemy, move_distance)

        if best_charge and best_value > 0:
            pos, enemy, _ = best_charge
            self.game_state.move_unit(unit, pos[0], pos[1])
            self.game_state.attack(unit, enemy)
            return True

        return False

    def _try_rogue_flank(self, unit) -> bool:
        """Attempt Rogue flank attack for +50% damage (target adjacent to ally)."""
        if unit.type != 'R' or not self.has_flank_units():
            return False

        # Find enemies that can be flanked
        flankable = self._find_flank_targets(unit)
        if not flankable:
            return False

        # Check if we can attack any flankable target directly
        attackable = self.game_state.mechanics.get_attackable_enemies(
            unit, flankable, self.game_state.grid
        )

        if attackable:
            # Attack the highest value flankable target
            best_target = max(attackable, key=lambda e: self.calculate_attack_value(unit, e))
            self.game_state.attack(unit, best_target)
            return True

        # Try to move to flank position
        reachable = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
            )
        )

        best_flank_pos = None
        best_target = None
        best_value = -float('inf')

        for pos in reachable:
            for enemy in flankable:
                # Check if we can attack enemy from this position (distance 1 for melee)
                if self.manhattan_distance(pos[0], pos[1], enemy.x, enemy.y) == 1:
                    value = self.calculate_attack_value(unit, enemy)
                    # Bonus for forest positions (evade bonus)
                    tile = self.game_state.grid.get_tile(pos[0], pos[1])
                    if tile.type == 'f':
                        value += 50  # Forest bonus

                    if value > best_value:
                        best_value = value
                        best_flank_pos = pos
                        best_target = enemy

        if best_flank_pos and best_target:
            self.game_state.move_unit(unit, best_flank_pos[0], best_flank_pos[1])
            self.game_state.attack(unit, best_target)
            return True

        return False

    def _try_rogue_forest_position(self, unit) -> bool:
        """Try to position Rogue in a forest for evade bonus."""
        if unit.type != 'R' or not self.has_flank_units():
            return False

        # Already in forest
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if tile.type == 'f':
            return False

        # Check if there are nearby enemies (only position if combat expected)
        enemies = [u for u in self.game_state.units
                   if u.player != self.bot_player and u.health > 0]
        if not enemies:
            return False

        nearest_enemy_dist = min(
            self.manhattan_distance(unit.x, unit.y, e.x, e.y) for e in enemies
        )
        if nearest_enemy_dist > 5:  # Only position if enemy is close
            return False

        # Find reachable forest positions
        reachable = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
            )
        )

        # Find best forest position (closest to enemies)
        best_forest = None
        best_dist = float('inf')

        for pos in reachable:
            forest_tile = self.game_state.grid.get_tile(pos[0], pos[1])
            if forest_tile.type == 'f':
                # Find distance to nearest enemy from this forest
                min_enemy_dist = min(
                    self.manhattan_distance(pos[0], pos[1], e.x, e.y) for e in enemies
                )
                if min_enemy_dist < best_dist:
                    best_dist = min_enemy_dist
                    best_forest = pos

        if best_forest:
            self.game_state.move_unit(unit, best_forest[0], best_forest[1])
            # Try to attack after moving to forest
            attackable = self.game_state.mechanics.get_attackable_enemies(
                unit, enemies, self.game_state.grid
            )
            if attackable:
                best_target = max(attackable, key=lambda e: self.calculate_attack_value(unit, e))
                self.game_state.attack(unit, best_target)
            return True

        return False

    def try_use_special_ability(self, unit):
        """Try to use unit special abilities effectively."""
        # Mage Paralyze on high-value targets (only if Mages enabled)
        if unit.type == 'M' and self.has_paralyze_units() and unit.can_attack:
            enemies = [u for u in self.game_state.units
                      if u.player != self.bot_player and u.health > 0]

            for enemy in enemies:
                # Check if enemy is capturing a structure
                enemy_tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
                if (enemy_tile.is_capturable() and
                    enemy_tile.player != self.bot_player and
                    enemy_tile.health < enemy_tile.max_health):

                    # Check if in range
                    attackable = self.game_state.mechanics.get_attackable_enemies(
                        unit, [enemy], self.game_state.grid
                    )
                    if enemy in attackable:
                        self.game_state.attack(unit, enemy)
                        return True

        # Cleric Heal (only if Clerics enabled)
        if unit.type == 'C' and self.has_heal_units():
            adjacent_allies = self.game_state.mechanics.get_adjacent_allies(
                unit, self.game_state.units
            )

            if adjacent_allies:
                # Only heal if ally is damaged
                damaged_allies = [a for a in adjacent_allies if a.health < a.max_health]
                if damaged_allies:
                    # Prioritize frontline units (Warriors, Barbarians, Knights)
                    frontline_allies = [
                        a for a in damaged_allies if a.type in ['W', 'B', 'K']
                    ]
                    if frontline_allies:
                        target = min(frontline_allies, key=lambda a: a.health)
                    else:
                        target = min(damaged_allies, key=lambda a: a.health)

                    self.game_state.heal(unit, target)
                    return True

        # Sorcerer abilities (only if Sorcerers enabled)
        if unit.type == 'S' and self.has_buff_units():
            if self._try_sorcerer_abilities(unit):
                return True

        return False

    def _try_sorcerer_abilities(self, unit) -> bool:
        """Use Sorcerer abilities strategically (Haste, Attack Buff, Defence Buff)."""
        if unit.type != 'S' or not self.has_buff_units():
            return False

        allies = [u for u in self.game_state.units
                  if u.player == self.bot_player and u != unit]
        enemies = [u for u in self.game_state.units
                   if u.player != self.bot_player and u.health > 0]

        if not allies:
            return False

        # Check range (Sorcerer buffs have range 0-2)
        def in_buff_range(target):
            dist = self.manhattan_distance(unit.x, unit.y, target.x, target.y)
            return dist <= 2

        allies_in_range = [a for a in allies if in_buff_range(a)]
        if not allies_in_range:
            return False

        # Priority 1: Haste a Knight that can charge (if Knight is enabled)
        if self.has_charge_units():
            knights_in_range = [
                a for a in allies_in_range
                if a.type == 'K' and a.can_attack and not getattr(a, 'hasted', False)
            ]
            for knight in knights_in_range:
                # Check if knight has a potential charge target
                for enemy in enemies:
                    dist = self.manhattan_distance(knight.x, knight.y, enemy.x, enemy.y)
                    if dist >= CHARGE_MIN_DISTANCE and dist <= knight.movement_range + 1:
                        # Good haste target
                        if hasattr(self.game_state, 'haste'):
                            self.game_state.haste(unit, knight)
                            return True

        # Priority 2: Haste a Rogue for flank opportunity (if Rogue is enabled)
        if self.has_flank_units():
            rogues_in_range = [
                a for a in allies_in_range
                if a.type == 'R' and a.can_attack and not getattr(a, 'hasted', False)
            ]
            for rogue in rogues_in_range:
                flankable = self._find_flank_targets(rogue)
                if flankable:
                    if hasattr(self.game_state, 'haste'):
                        self.game_state.haste(unit, rogue)
                        return True

        # Priority 3: Attack buff on frontline unit about to engage
        frontline_in_range = [
            a for a in allies_in_range
            if a.type in ['W', 'K', 'B', 'R'] and a.can_attack
            and not getattr(a, 'has_attack_buff', lambda: False)()
        ]
        if frontline_in_range:
            # Find unit closest to enemy
            best_frontline = min(
                frontline_in_range,
                key=lambda a: min(
                    self.manhattan_distance(a.x, a.y, e.x, e.y) for e in enemies
                ) if enemies else float('inf')
            )
            if hasattr(self.game_state, 'attack_buff'):
                self.game_state.attack_buff(unit, best_frontline)
                return True

        # Priority 4: Defence buff on unit capturing contested structure
        for ally in allies_in_range:
            tile = self.game_state.grid.get_tile(ally.x, ally.y)
            if (tile.is_capturable() and tile.health < tile.max_health
                and not getattr(ally, 'has_defence_buff', lambda: False)()):
                if hasattr(self.game_state, 'defence_buff'):
                    self.game_state.defence_buff(unit, ally)
                    return True

        # Priority 5: Defence buff on low-health frontline unit
        low_health_frontline = [
            a for a in allies_in_range
            if a.type in ['W', 'K', 'B'] and a.health < a.max_health * 0.5
            and not getattr(a, 'has_defence_buff', lambda: False)()
        ]
        if low_health_frontline:
            target = min(low_health_frontline, key=lambda a: a.health)
            if hasattr(self.game_state, 'defence_buff'):
                self.game_state.defence_buff(unit, target)
                return True

        return False

    def try_ranged_attack(self, unit):
        """Try to use ranged attacks to minimize counter-attack damage."""
        # Ranged units: Archer (A), Mage (M), Sorcerer (S)
        if unit.type not in ['A', 'M', 'S']:
            return False

        enemy_units = [u for u in self.game_state.units
                      if u.player != self.bot_player and u.health > 0]

        attackable = self.game_state.mechanics.get_attackable_enemies(
            unit, enemy_units, self.game_state.grid
        )

        if not attackable:
            return False

        # Prioritize melee units (they can't counter-attack ranged attacks)
        # Ranged units that can counter: A, M, S
        melee_targets = [e for e in attackable if e.type not in ['A', 'M', 'S']]

        if melee_targets:
            # Attack the one with lowest health (finish off)
            target = min(melee_targets, key=lambda e: e.health)
            self.game_state.attack(unit, target)
            return True

        # Otherwise attack any target
        target = min(attackable, key=lambda e: e.health)
        self.game_state.attack(unit, target)
        return True
