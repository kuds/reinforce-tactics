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
                return 0  # Highest priority
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


class AdvancedBot:
    """Advanced AI bot using MCTS and strategic analysis."""

    def __init__(self, game_state, player=2, mcts_iterations=20, mcts_depth=2):
        """
        Initialize the AdvancedBot.

        Args:
            game_state: GameState instance
            player: Player number for this bot
            mcts_iterations: Number of MCTS simulations per action (default: 20)
            mcts_depth: Maximum depth for MCTS rollouts (default: 2)
        """
        self.game_state = game_state
        self.bot_player = player
        self.mcts_iterations = mcts_iterations
        self.mcts_depth = mcts_depth
        
        # Map analysis cache
        self.map_analyzed = False
        self.chokepoints = []
        self.distance_maps = {}
        self.factory_clusters = []
        self.defensive_positions = []
        self.hq_positions = {}

    def take_turn(self):
        """Execute the bot's turn with advanced strategy."""
        # Phase 1: Analyze map on first turn
        if not self.map_analyzed:
            self.analyze_map()
            self.map_analyzed = True

        # Phase 2: Strategic assessment
        income_diff = self.calculate_income_differential()
        threat_level = self.assess_threat_level()
        
        # Phase 3: Purchase units with smart composition
        self.purchase_units_advanced(income_diff)

        # Phase 4: Execute actions with MCTS evaluation
        self.move_and_act_units_advanced(income_diff, threat_level)
        
        # Phase 5: End turn
        self.game_state.end_turn()

    def analyze_map(self):
        """Pre-compute strategic map features on first turn."""
        grid = self.game_state.grid
        
        # Identify HQ positions
        for row in grid.tiles:
            for tile in row:
                if tile.type == 'h' and tile.player:
                    self.hq_positions[tile.player] = (tile.x, tile.y)
        
        # Identify chokepoints (tiles with few walkable neighbors)
        self.chokepoints = []
        for row in grid.tiles:
            for tile in row:
                if tile.is_walkable():
                    walkable_neighbors = 0
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = tile.x + dx, tile.y + dy
                        if 0 <= nx < grid.width and 0 <= ny < grid.height:
                            neighbor = grid.get_tile(nx, ny)
                            if neighbor.is_walkable():
                                walkable_neighbors += 1
                    
                    # Chokepoint if <= 2 walkable neighbors
                    if walkable_neighbors <= 2:
                        self.chokepoints.append((tile.x, tile.y))
        
        # Pre-calculate distance maps from both HQs
        for player, hq_pos in self.hq_positions.items():
            self.distance_maps[player] = self.calculate_distance_map(hq_pos)
        
        # Identify factory clusters (groups of nearby buildings)
        self.factory_clusters = self.identify_factory_clusters()
        
        # Identify defensive positions (mountains, forests)
        self.defensive_positions = []
        for row in grid.tiles:
            for tile in row:
                if tile.type in ['m', 'f']:  # Mountain or forest
                    self.defensive_positions.append((tile.x, tile.y))

    def calculate_distance_map(self, start_pos):
        """Calculate Manhattan distance from start_pos to all tiles."""
        distance_map = {}
        grid = self.game_state.grid
        
        for row in grid.tiles:
            for tile in row:
                dist = abs(tile.x - start_pos[0]) + abs(tile.y - start_pos[1])
                distance_map[(tile.x, tile.y)] = dist
        
        return distance_map

    def identify_factory_clusters(self):
        """Group nearby buildings for income optimization targets."""
        buildings = []
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.type in ['b', 't'] and tile.player != self.bot_player:
                    buildings.append((tile.x, tile.y))
        
        # Simple clustering: group buildings within distance 3
        clusters = []
        visited = set()
        
        for building in buildings:
            if building in visited:
                continue
            
            cluster = [building]
            visited.add(building)
            
            for other in buildings:
                if other in visited:
                    continue
                if self.manhattan_distance(building[0], building[1], other[0], other[1]) <= 3:
                    cluster.append(other)
                    visited.add(other)
            
            clusters.append(cluster)
        
        return clusters

    def calculate_income_differential(self):
        """Calculate income advantage/disadvantage."""
        my_income = 0
        enemy_income = 0
        
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.player:
                    income = 0
                    if tile.type == 'h':
                        income = 150
                    elif tile.type == 'b':
                        income = 100
                    elif tile.type == 't':
                        income = 50
                    
                    if tile.player == self.bot_player:
                        my_income += income
                    else:
                        enemy_income += income
        
        return my_income - enemy_income

    def assess_threat_level(self):
        """Identify enemy attack vectors and potential threats."""
        threats = 0
        my_units = [u for u in self.game_state.units if u.player == self.bot_player]
        enemy_units = [u for u in self.game_state.units if u.player != self.bot_player]
        
        # Check if any of our structures are under immediate threat
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.player == self.bot_player:
                    for enemy in enemy_units:
                        dist = self.manhattan_distance(tile.x, tile.y, enemy.x, enemy.y)
                        if dist <= 3:  # Enemy within 3 tiles
                            threats += 1
        
        return threats

    def purchase_units_advanced(self, income_diff):
        """Purchase units with smart composition based on game state."""
        available_gold = self.game_state.player_gold[self.bot_player]
        
        # Count existing unit types
        my_units = [u for u in self.game_state.units if u.player == self.bot_player]
        unit_counts = {'W': 0, 'A': 0, 'M': 0, 'C': 0, 'B': 0}
        for unit in my_units:
            unit_counts[unit.type] = unit_counts.get(unit.type, 0) + 1
        
        # Determine desired composition based on income
        desired_composition = self.get_desired_composition(income_diff, len(my_units))
        
        while True:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)
            create_actions = legal_actions['create_unit']
            
            if not create_actions:
                break
            
            affordable_actions = [
                a for a in create_actions 
                if UNIT_DATA[a['unit_type']]['cost'] <= self.game_state.player_gold[self.bot_player]
            ]
            
            if not affordable_actions:
                break
            
            # Choose unit type based on composition
            best_action = self.choose_unit_by_composition(affordable_actions, unit_counts, desired_composition)
            
            if best_action:
                self.game_state.create_unit(
                    best_action['unit_type'],
                    best_action['x'],
                    best_action['y'],
                    self.bot_player
                )
                unit_counts[best_action['unit_type']] += 1
            else:
                break

    def get_desired_composition(self, income_diff, unit_count):
        """Determine desired unit composition based on game state."""
        if income_diff > 200:  # Ahead - aggressive
            return {'W': 0.3, 'A': 0.3, 'M': 0.3, 'B': 0.1, 'C': 0.0}
        elif income_diff < -200:  # Behind - defensive
            return {'W': 0.4, 'A': 0.2, 'M': 0.2, 'C': 0.2, 'B': 0.0}
        else:  # Balanced
            return {'W': 0.35, 'A': 0.25, 'M': 0.25, 'C': 0.1, 'B': 0.05}

    def choose_unit_by_composition(self, affordable_actions, current_counts, desired_comp):
        """Choose unit that best fits desired composition."""
        total_units = sum(current_counts.values()) + 1
        
        best_action = None
        best_score = -float('inf')
        
        for action in affordable_actions:
            unit_type = action['unit_type']
            current_ratio = current_counts.get(unit_type, 0) / max(1, total_units)
            desired_ratio = desired_comp.get(unit_type, 0)
            
            # Prefer units we're lacking
            score = desired_ratio - current_ratio
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def move_and_act_units_advanced(self, income_diff, threat_level):
        """Execute unit actions with MCTS evaluation."""
        bot_units = [
            u for u in self.game_state.units
            if u.player == self.bot_player and (u.can_move or u.can_attack)
            and not u.is_paralyzed()
        ]
        
        # Sort units by priority: ranged in back, melee in front
        bot_units.sort(key=lambda u: self.get_unit_action_priority(u), reverse=True)
        
        for unit in bot_units:
            if unit.can_move or unit.can_attack:
                self.act_with_unit_advanced(unit, income_diff, threat_level)

    def get_unit_action_priority(self, unit):
        """Determine action priority for unit (higher = act first)."""
        # Clerics act first (to heal before combat)
        if unit.type == 'C':
            return 100
        # Mages act second (to paralyze before melee engages)
        elif unit.type == 'M':
            return 90
        # Then Warriors and Barbarians
        elif unit.type in ['W', 'B']:
            return 50
        # Archers last
        else:
            return 30

    def act_with_unit_advanced(self, unit, income_diff, threat_level):
        """Execute actions for a single unit with advanced strategy."""
        # Check if already seizing a structure
        tile = self.game_state.grid.get_tile(unit.x, unit.y)
        if (tile.is_capturable() and tile.player != self.bot_player and
                tile.health < tile.max_health):
            self.game_state.seize(unit)
            return

        # Special ability usage
        if self.try_use_special_ability(unit):
            return

        # Smart ranged combat
        if unit.type in ['A', 'M'] and self.try_ranged_attack(unit):
            return

        # Evaluate possible actions with MCTS
        possible_actions = self.generate_possible_actions(unit)
        
        if possible_actions:
            best_action = self.mcts_evaluate(unit, possible_actions)
            if best_action:
                self.execute_action(unit, best_action)
                return

        # Fallback to structure capture
        if self.try_capture_structure(unit):
            return

        unit.end_unit_turn()

    def try_use_special_ability(self, unit):
        """Try to use unit special abilities effectively."""
        # Mage Paralyze
        if unit.type == 'M' and unit.can_attack:
            # Find high-value targets to paralyze
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
                        # Use paralyze on capturing enemy
                        self.game_state.attack(unit, enemy)
                        return True
        
        # Cleric Heal
        if unit.type == 'C':
            # Find damaged allies adjacent to unit
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

    def generate_possible_actions(self, unit):
        """Generate possible actions for MCTS evaluation."""
        actions = []
        
        # Get reachable positions
        reachable = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
            )
        )
        
        # Limit to top positions to keep MCTS tractable
        scored_positions = []
        for pos in reachable:
            score = self.evaluate_position(unit, pos)
            scored_positions.append((pos, score))
        
        scored_positions.sort(key=lambda x: x[1], reverse=True)
        top_positions = scored_positions[:min(5, len(scored_positions))]
        
        for pos, _ in top_positions:
            actions.append({
                'type': 'move',
                'position': pos,
                'then_attack': None
            })
        
        return actions

    def evaluate_position(self, unit, position):
        """Evaluate strategic value of a position for unit."""
        score = 0
        x, y = position
        
        # Check tile type
        tile = self.game_state.grid.get_tile(x, y)
        if tile.type == 'm':  # Mountain
            score += 10
        elif tile.type == 'f':  # Forest
            score += 5
        
        # Distance to enemy HQ
        enemy_player = 1 if self.bot_player == 2 else 2
        if enemy_player in self.hq_positions:
            enemy_hq = self.hq_positions[enemy_player]
            dist_to_enemy_hq = self.manhattan_distance(x, y, enemy_hq[0], enemy_hq[1])
            score -= dist_to_enemy_hq * 2
        
        # Distance to nearest enemy unit
        enemies = [u for u in self.game_state.units if u.player != self.bot_player]
        if enemies:
            min_enemy_dist = min(self.manhattan_distance(x, y, e.x, e.y) for e in enemies)
            if unit.type in ['W', 'B']:  # Melee units should get closer
                score -= min_enemy_dist * 3
            elif unit.type in ['A', 'M']:  # Ranged should maintain distance
                if min_enemy_dist < 2:
                    score -= 20  # Too close
                elif min_enemy_dist <= 3:
                    score += 10  # Good distance
        
        # Check for capturable structures
        if tile.is_capturable() and tile.player != self.bot_player:
            score += 50
        
        return score

    def mcts_evaluate(self, unit, possible_actions):
        """Monte Carlo Tree Search to evaluate possible actions."""
        if not possible_actions:
            return None
        
        action_scores = {}
        
        for action in possible_actions:
            total_score = 0
            
            # Run multiple simulations
            for _ in range(self.mcts_iterations):
                # Simulate this action
                score = self.simulate_action(unit, action)
                total_score += score
            
            avg_score = total_score / self.mcts_iterations
            action_scores[self.action_to_key(action)] = (action, avg_score)
        
        # Return action with best average score
        if action_scores:
            best_key = max(action_scores.keys(), key=lambda k: action_scores[k][1])
            return action_scores[best_key][0]
        
        return None

    def action_to_key(self, action):
        """Convert action to hashable key."""
        if action['type'] == 'move':
            return ('move', action['position'])
        return ('unknown',)

    def simulate_action(self, unit, action):
        """Simulate an action and return its expected value."""
        score = 0
        
        if action['type'] == 'move':
            target_pos = action['position']
            
            # Basic score from position evaluation
            score += self.evaluate_position(unit, target_pos)
            
            # Check if this position allows attacking
            old_x, old_y = unit.x, unit.y
            unit.x, unit.y = target_pos[0], target_pos[1]
            
            enemies = [u for u in self.game_state.units 
                      if u.player != self.bot_player and u.health > 0]
            attackable = self.game_state.mechanics.get_attackable_enemies(
                unit, enemies, self.game_state.grid
            )
            
            if attackable:
                # Bonus for positions that enable attacks
                score += 20
                
                # Additional bonus if we can kill an enemy
                for enemy in attackable:
                    tile = self.game_state.grid.get_tile(target_pos[0], target_pos[1])
                    on_mountain = tile.type == 'm'
                    damage = unit.get_attack_damage(enemy.x, enemy.y, on_mountain)
                    if damage >= enemy.health:
                        score += 50  # Killing is very valuable
            
            # Restore unit position
            unit.x, unit.y = old_x, old_y
            
            # Check for structure capture opportunity
            tile = self.game_state.grid.get_tile(target_pos[0], target_pos[1])
            if tile.is_capturable() and tile.player != self.bot_player:
                # Calculate capture priority
                score += self.calculate_capture_priority(tile) * 10
        
        # Add randomness for exploration
        score += random.uniform(-5, 5)
        
        return score

    def try_capture_structure(self, unit):
        """Try to capture a structure with advanced priority scoring."""
        capturable_structures = []
        
        for row in self.game_state.grid.tiles:
            for tile in row:
                if tile.is_capturable() and tile.player != self.bot_player:
                    priority = self.calculate_capture_priority(tile)
                    capturable_structures.append((tile, priority))
        
        if not capturable_structures:
            return False
        
        capturable_structures.sort(key=lambda x: x[1], reverse=True)
        
        for structure, _ in capturable_structures:
            if unit.x == structure.x and unit.y == structure.y:
                self.game_state.seize(unit)
                return True
            
            # Try to move towards structure
            target_pos = self.find_best_move_position(unit, structure.x, structure.y)
            if target_pos:
                self.game_state.move_unit(unit, target_pos[0], target_pos[1])
                if unit.x == structure.x and unit.y == structure.y:
                    self.game_state.seize(unit)
                return True
        
        return False

    def calculate_capture_priority(self, structure):
        """Calculate advanced priority score for capturing a structure."""
        score = 0
        
        # Income value
        if structure.type == 'h':
            score += 150
        elif structure.type == 'b':
            score += 100
        elif structure.type == 't':
            score += 50
        
        # Distance from our HQ (closer = better)
        if self.bot_player in self.hq_positions:
            my_hq = self.hq_positions[self.bot_player]
            dist_from_hq = self.manhattan_distance(
                structure.x, structure.y, my_hq[0], my_hq[1]
            )
            score -= dist_from_hq * 2
        
        # Check if it's in a factory cluster
        for cluster in self.factory_clusters:
            if (structure.x, structure.y) in cluster:
                score += len(cluster) * 10  # Bigger clusters are better
                break
        
        # Check enemy proximity (risky if enemies nearby)
        enemies = [u for u in self.game_state.units if u.player != self.bot_player]
        if enemies:
            min_enemy_dist = min(
                self.manhattan_distance(structure.x, structure.y, e.x, e.y) 
                for e in enemies
            )
            if min_enemy_dist <= 2:
                score -= 50  # Too risky
            elif min_enemy_dist <= 4:
                score -= 20  # Somewhat risky
        
        # Check if it's a defensive position
        if (structure.x, structure.y) in self.defensive_positions:
            score += 15
        
        # Check if it's a chokepoint
        if (structure.x, structure.y) in self.chokepoints:
            score += 10
        
        return score

    def execute_action(self, unit, action):
        """Execute the chosen action."""
        if action['type'] == 'move':
            target_pos = action['position']
            self.game_state.move_unit(unit, target_pos[0], target_pos[1])
            
            # Try to attack after moving
            enemies = [u for u in self.game_state.units 
                      if u.player != self.bot_player and u.health > 0]
            attackable = self.game_state.mechanics.get_attackable_enemies(
                unit, enemies, self.game_state.grid
            )
            
            if attackable and unit.can_attack:
                # Attack best target
                best_target = self.choose_best_attack_target(unit, attackable)
                if best_target:
                    self.game_state.attack(unit, best_target)

    def choose_best_attack_target(self, unit, attackable_enemies):
        """Choose the best enemy to attack from available targets."""
        if not attackable_enemies:
            return None
        
        best_target = None
        best_score = -float('inf')
        
        for enemy in attackable_enemies:
            score = 0
            
            # Prioritize enemies we can kill
            tile = self.game_state.grid.get_tile(unit.x, unit.y)
            on_mountain = tile.type == 'm'
            damage = unit.get_attack_damage(enemy.x, enemy.y, on_mountain)
            
            if damage >= enemy.health:
                score += 1000  # Killing is top priority
            
            # Prioritize enemies on structures
            enemy_tile = self.game_state.grid.get_tile(enemy.x, enemy.y)
            if enemy_tile.is_capturable() and enemy_tile.player != self.bot_player:
                score += 500
            
            # Prioritize high-value units
            score += UNIT_DATA[enemy.type]['cost'] / 10
            
            # Prioritize low-health enemies
            score += (enemy.max_health - enemy.health) * 5
            
            if score > best_score:
                best_score = score
                best_target = enemy
        
        return best_target

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
