"""
Simple AI bot for computer opponents.
"""
from reinforcetactics.constants import UNIT_DATA


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

    def purchase_units(self):
        """Purchase units with available gold, most expensive first."""
        legal_actions = self.game_state.get_legal_actions(self.bot_player)
        create_actions = legal_actions['create_unit']

        # Sort by cost (most expensive first)
        unit_costs = {'M': 250, 'W': 200, 'C': 200}
        create_actions.sort(key=lambda a: unit_costs[a['unit_type']], reverse=True)

        for action in create_actions:
            if self.game_state.player_gold[self.bot_player] >= UNIT_DATA[action['unit_type']]['cost']:
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
                x, y, self.game_state.grid, self.game_state.units
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
