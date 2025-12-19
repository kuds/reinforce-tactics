"""
Simple AI bot for computer opponents.
"""
import random
import copy
from reinforcetactics.constants import (
    UNIT_DATA, COUNTER_ATTACK_MULTIPLIER, PARALYZE_DURATION, HEAL_AMOUNT
)


class SimpleBot:
    """Simple AI bot for player 2."""

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
        """Purchase units with available gold, most expensive first."""
        legal_actions = self.game_state.get_legal_actions(self.bot_player)
        create_actions = legal_actions['create_unit']

        # Sort by cost (most expensive first) using UNIT_DATA
        create_actions.sort(key=lambda a: UNIT_DATA[a['unit_type']]['cost'], reverse=True)

        for action in create_actions:
            unit_cost = UNIT_DATA[action['unit_type']]['cost']
            if self.game_state.player_gold[self.bot_player] >= unit_cost:
                self.game_state.create_unit(
                    action['unit_type'],
                    action['x'],
                    action['y'],
                    self.bot_player
                )
                # Refresh legal actions after purchase
                break

    def move_and_act_units(self):
        """Move and act with all bot units."""
        bot_units = [
            u for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack)
            and not u.is_paralyzed()
        ]

        for unit in bot_units:
            self.act_with_unit(unit)

    def act_with_unit(self, unit):
        """Determine and execute best action for a single unit."""
        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if (tile.is_capturable() and tile.player != self.bot_player and
                tile.health < tile.max_health):
            self.game_state.seize(unit)
            return

        # Find best target
        target = self.find_best_target(unit)

        if target:
            target_type, target_obj, _ = target

            if target_type == 'enemy_unit':
                self.attack_enemy(unit, target_obj)
            elif target_type in ['enemy_tower', 'enemy_building', 'enemy_hq']:
                self.move_to_and_seize(unit, target_obj)
        else:
            unit.end_unit_turn()

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

        # Find enemy structures
        enemy_structures = []
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.player and tile.player != self.bot_player:
                    dist = self.manhattan_distance(unit.x, unit.y, tile.x, tile.y)
                    if tile.type == 't':
                        enemy_structures.append(('enemy_tower', tile, dist))
                    elif tile.type == 'b':
                        enemy_structures.append(('enemy_building', tile, dist))
                    elif tile.type == 'h' and not opponent_has_bases and not opponent_has_units:
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

    def manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance between two points."""
        return abs(x1 - x2) + abs(y1 - y2)

    def attack_enemy(self, unit, enemy):
        """Attack an enemy unit."""
        distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)

        if distance == 1:
            self.game_state.attack(unit, enemy)
        else:
            target_pos = self.find_best_move_position(unit, enemy.x, enemy.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                new_distance = self.manhattan_distance(unit.x, unit.y, enemy.x, enemy.y)
                if new_distance == 1:
                    self.game_state.attack(unit, enemy)
                else:
                    unit.end_unit_turn()
            else:
                unit.end_unit_turn()

    def move_to_and_seize(self, unit, structure):
        """Move towards and seize a structure."""
        if unit.x == structure.x and unit.y == structure.y:
            self.game_state.seize(unit)
        else:
            target_pos = self.find_best_move_position(unit, structure.x, structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == structure.x and unit.y == structure.y:
                    self.game_state.seize(unit)
                else:
                    unit.end_unit_turn()
            else:
                unit.end_unit_turn()

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


class MediumBot:
    """Medium difficulty AI bot with improved strategic decision-making."""

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
        """Purchase as many units as possible with available gold."""
        # Keep buying units until we can't afford any more
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions['create_unit']

            if not create_actions:
                break

            # Available gold
            available_gold = self.game_state.player_gold[self.bot_player]

            # Find affordable units
            affordable_actions = []
            for action in create_actions:
                unit_cost = UNIT_DATA[action['unit_type']]['cost']
                if available_gold >= unit_cost:
                    affordable_actions.append(action)

            if not affordable_actions:
                break

            # Prefer a mix of Warriors (cheap, good HP for capturing) and other units
            # Sort by: Warriors first, then by cost (cheaper first for quantity)
            def unit_priority(action):
                unit_type = action['unit_type']
                cost = UNIT_DATA[unit_type]['cost']
                # Warriors get priority (return 0), others sorted by cost
                if unit_type == 'W':
                    return (0, cost)
                # Archers are also good (ranged)
                elif unit_type == 'A':
                    return (1, cost)
                # Mages are decent (ranged, but expensive)
                elif unit_type == 'M':
                    return (2, cost)
                else:
                    return (3, cost)

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

    def calculate_attack_value(self, attacker, target):
        """
        Evaluate attack efficiency considering damage dealt and received.

        Args:
            attacker: Unit that would attack
            target: Enemy unit to attack

        Returns:
            Value score (higher is better)
        """
        # Calculate damage attacker would deal
        attacker_tile = self.game_state.grid.get_tile(attacker.x, attacker.y)
        on_mountain = attacker_tile.type == 'm'
        damage_dealt = attacker.get_attack_damage(target.x, target.y, on_mountain)

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
        if distance > 1 and target.type not in ['M', 'A']:
            # Target can't counter-attack ranged attacker
            counter_damage = 0
        elif target_damage > 0:
            # Counter-attacks deal reduced damage (COUNTER_ATTACK_MULTIPLIER = 0.9)
            counter_damage = int(target_damage * COUNTER_ATTACK_MULTIPLIER)

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

    def act_with_unit(self, unit):
        """Execute actions for a single unit based on strategic priorities."""
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

            for structure, enemy_unit, progress in contested:
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
                return

            # Move towards structure
            target_pos = self.find_best_move_position(unit, target_structure.x, target_structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                # Check if reached structure
                if unit.x == target_structure.x and unit.y == target_structure.y:
                    self.game_state.seize(unit)
                    return

        # Fallback: End turn
        unit.end_unit_turn()

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


class AdvancedBot(MediumBot):
    """Advanced AI bot extending MediumBot with map analysis and enhanced tactics."""

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
        self.defensive_positions = []
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

        # Identify defensive positions (mountains for attack bonuses)
        self.defensive_positions = []
        for row in grid.tiles:
            for tile in row:
                if tile.type == 'm':  # Mountains give attack bonuses
                    self.defensive_positions.append((tile.x, tile.y))

    def purchase_units_enhanced(self):
        """Enhanced unit purchasing with better composition."""
        # Count existing unit types
        my_units = [u for u in self.game_state.units if u.player == self.bot_player]
        unit_counts = {'W': 0, 'A': 0, 'M': 0, 'C': 0, 'B': 0}
        for unit in my_units:
            unit_counts[unit.type] = unit_counts.get(unit.type, 0) + 1

        total_units = len(my_units)

        # Enhanced composition: more ranged units (Archers/Mages) for better trades
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions['create_unit']

            if not create_actions:
                break

            available_gold = self.game_state.player_gold[self.bot_player]
            affordable_actions = [
                a for a in create_actions
                if UNIT_DATA[a['unit_type']]['cost'] <= available_gold
            ]

            if not affordable_actions:
                break

            # Prioritize ranged units (Archers/Mages) early, Warriors for capturing
            archer_ratio = unit_counts.get('A', 0) / max(1, total_units + 1)
            mage_ratio = unit_counts.get('M', 0) / max(1, total_units + 1)
            warrior_ratio = unit_counts.get('W', 0) / max(1, total_units + 1)
            cleric_ratio = unit_counts.get('C', 0) / max(1, total_units + 1)

            # Desired ratios: 35% Warriors, 30% Archers, 25% Mages, 10% Clerics
            best_action = None
            best_priority = -float('inf')

            for action in affordable_actions:
                unit_type = action['unit_type']
                priority = 0

                if unit_type == 'W' and warrior_ratio < 0.35:
                    priority = 0.35 - warrior_ratio
                elif unit_type == 'A' and archer_ratio < 0.30:
                    priority = 0.30 - archer_ratio + 0.1  # Slight boost for Archers
                elif unit_type == 'M' and mage_ratio < 0.25:
                    priority = 0.25 - mage_ratio + 0.05  # Slight boost for Mages
                elif unit_type == 'C' and cleric_ratio < 0.10 and total_units >= 3:
                    priority = 0.10 - cleric_ratio
                else:
                    priority = -1  # Fallback

                if priority > best_priority:
                    best_priority = priority
                    best_action = action

            if best_action:
                self.game_state.create_unit(
                    best_action['unit_type'],
                    best_action['x'],
                    best_action['y'],
                    self.bot_player
                )
                unit_counts[best_action['unit_type']] += 1
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

    def act_with_unit_enhanced(self, unit):
        """Enhanced version of MediumBot's act_with_unit with superior tactics."""
        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if (tile.is_capturable() and tile.player != self.bot_player and
                tile.health < tile.max_health):
            self.game_state.seize(unit)
            return

        # Try special abilities first (Cleric heal, Mage paralyze)
        if self.try_use_special_ability(unit):
            return

        # PRIORITY 1: Position on mountains for attack bonus before attacking
        if unit.type in ['W', 'B', 'A', 'M'] and not tile.type == 'm':
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
                            target_pos = self.find_best_move_position(unit, mountain_pos[0], mountain_pos[1])
                            if target_pos and target_pos == mountain_pos:
                                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                                # Try to attack after positioning
                                if self.try_ranged_attack(unit):
                                    return
                                break

        # PRIORITY 2: Ranged attacks (Archers/Mages should attack from range)
        if unit.type in ['A', 'M'] and self.try_ranged_attack(unit):
            return

        # PRIORITY 3: Move to attack range and attack
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
                return

            # Try to move towards nearest enemy
            if enemy_units:
                nearest_enemy = min(enemy_units,
                                   key=lambda e: self.manhattan_distance(unit.x, unit.y, e.x, e.y))
                target_pos = self.find_best_move_position(unit, nearest_enemy.x, nearest_enemy.y)
                if target_pos:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    # Try to attack after moving
                    attackable_after = self.game_state.mechanics.get_attackable_enemies(
                        unit, enemy_units, self.game_state.grid
                    )
                    if attackable_after:
                        best_after_move = max(attackable_after,
                                             key=lambda e: self.calculate_attack_value(unit, e))
                        self.game_state.attack(unit, best_after_move)
                        return

        # PRIORITY 4: Interrupt enemy captures (from MediumBot)
        contested = self.find_contested_structures()
        if contested:
            contested.sort(key=lambda x: x[2], reverse=True)
            for structure, enemy_unit, progress in contested:
                target_pos = self.find_best_move_position(unit, enemy_unit.x, enemy_unit.y)
                if target_pos:
                    self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                    attackable = self.game_state.mechanics.get_attackable_enemies(
                        unit, [enemy_unit], self.game_state.grid
                    )
                    if enemy_unit in attackable:
                        self.game_state.attack(unit, enemy_unit)
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
                return

            target_pos = self.find_best_move_position(unit, target_structure.x, target_structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == target_structure.x and unit.y == target_structure.y:
                    self.game_state.seize(unit)
                    return

        # Fallback: End turn
        unit.end_unit_turn()

    def try_use_special_ability(self, unit):
        """Try to use unit special abilities effectively."""
        # Mage Paralyze on high-value targets
        if unit.type == 'M' and unit.can_attack:
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

        # Cleric Heal
        if unit.type == 'C':
            adjacent_allies = self.game_state.mechanics.get_adjacent_allies(
                unit, self.game_state.units
            )

            if adjacent_allies:
                # Prioritize frontline units (Warriors, Barbarians)
                frontline_allies = [a for a in adjacent_allies if a.type in ['W', 'B']]
                if frontline_allies:
                    target = min(frontline_allies, key=lambda a: a.health)
                else:
                    target = min(adjacent_allies, key=lambda a: a.health)

                self.game_state.heal(unit, target)
                return True

        return False

    def try_ranged_attack(self, unit):
        """Try to use ranged attacks to minimize counter-attack damage."""
        if unit.type not in ['A', 'M']:
            return False

        enemy_units = [u for u in self.game_state.units
                      if u.player != self.bot_player and u.health > 0]

        attackable = self.game_state.mechanics.get_attackable_enemies(
            unit, enemy_units, self.game_state.grid
        )

        if not attackable:
            return False

        # Prioritize melee units (they can't counter-attack ranged)
        melee_targets = [e for e in attackable if e.type not in ['A', 'M']]

        if melee_targets:
            # Attack the one with lowest health (finish off)
            target = min(melee_targets, key=lambda e: e.health)
            self.game_state.attack(unit, target)
            return True

        # Otherwise attack any target
        target = min(attackable, key=lambda e: e.health)
        self.game_state.attack(unit, target)
        return True
