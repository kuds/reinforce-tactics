"""
Core game state management without rendering dependencies.
Updated with structure healing feature.
"""
import numpy as np
from datetime import datetime
from reinforcetactics.core.unit import Unit
from reinforcetactics.core.grid import TileGrid
from reinforcetactics.game.mechanics import GameMechanics
from reinforcetactics.constants import STARTING_GOLD, UNIT_DATA


class GameState:
    """Manages the core game state without rendering."""

    def __init__(self, map_data, num_players=2):
        """
        Initialize the game state.

        Args:
            map_data: 2D array containing map information
            num_players: Number of players (default 2)
        """
        self.grid = TileGrid(map_data)
        self.units = []
        self.current_player = 1
        self.num_players = num_players
        self.player_gold = {i: STARTING_GOLD for i in range(1, num_players + 1)}
        self.game_over = False
        self.winner = None
        self.turn_number = 0
        self.mechanics = GameMechanics()

        # Action history for replay
        self.action_history = []
        self.game_start_time = datetime.now()

    def reset(self, map_data):
        """Reset the game state."""
        self.__init__(map_data, self.num_players)

    def get_unit_at_position(self, x, y):
        """Get the unit at a grid position."""
        for unit in self.units:
            if unit.x == x and unit.y == y:
                return unit
        return None

    def create_unit(self, unit_type, x, y, player=None):
        """
        Create a unit at the specified position.

        Args:
            unit_type: 'W', 'M', or 'C'
            x: Grid x coordinate
            y: Grid y coordinate
            player: Player number (defaults to current player)

        Returns:
            Unit if created, None if failed
        """
        if player is None:
            player = self.current_player

        if self.get_unit_at_position(x, y):
            return None

        cost = UNIT_DATA[unit_type]['cost']
        if self.player_gold[player] < cost:
            return None

        self.player_gold[player] -= cost
        unit = Unit(unit_type, x, y, player)
        self.units.append(unit)

        return unit

    def move_unit(self, unit, to_x, to_y):
        """
        Move a unit to a new position.

        Args:
            unit: Unit to move
            to_x: Target x coordinate
            to_y: Target y coordinate

        Returns:
            bool: True if move successful
        """
        reachable = unit.get_reachable_positions(
            self.grid.width,
            self.grid.height,
            lambda x, y: self.mechanics.can_move_to_position(x, y, self.grid, self.units)
        )

        if (to_x, to_y) not in reachable:
            return False

        if not self.mechanics.can_move_to_position(to_x, to_y, self.grid, self.units):
            return False

        old_tile = self.grid.get_tile(unit.x, unit.y)
        unit.move_to(to_x, to_y)

        return True

    def attack(self, attacker, target):
        """
        Execute an attack.

        Args:
            attacker: Attacking unit
            target: Target unit

        Returns:
            dict: Attack results
        """
        result = self.mechanics.attack_unit(attacker, target)

        if not result['target_alive']:
            target_tile = self.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            self.units.remove(target)

        if not result['attacker_alive']:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            self.units.remove(attacker)

        attacker.can_move = False
        attacker.can_attack = False

        return result

    def paralyze(self, paralyzer, target):
        """Paralyze a target unit."""
        result = self.mechanics.paralyze_unit(paralyzer, target)
        if result:
            paralyzer.can_move = False
            paralyzer.can_attack = False
        return result

    def heal(self, healer, target):
        """Heal a target unit."""
        amount = self.mechanics.heal_unit(healer, target)
        if amount > 0:
            healer.can_move = False
            healer.can_attack = False
        return amount

    def cure(self, curer, target):
        """Cure a target unit's paralysis."""
        result = self.mechanics.cure_unit(curer, target)
        if result:
            curer.can_move = False
            curer.can_attack = False
        return result

    def seize(self, unit):
        """Seize the structure the unit is on."""
        tile = self.grid.get_tile(unit.x, unit.y)
        result = self.mechanics.seize_structure(unit, tile)

        if result['game_over']:
            self.game_over = True
            self.winner = unit.player

        unit.can_move = False
        unit.can_attack = False

        return result

    def heal_units_on_structures(self, player):
        """
        Heal units on owned structures at the start of their turn.
        
        Healing amounts:
        - Tower: 1 HP
        - HQ/Building: 2 HP
        
        Cost formula: (heal_amount / unit_max_hp) * unit_cost (rounded)
        
        Priority: Units closest to enemy HQ if insufficient gold.
        
        Args:
            player: Player number whose units to heal
        """
        # Find enemy HQ for distance calculations
        enemy_hq_pos = None
        for row in self.grid.tiles:
            for tile in row:
                if tile.type == 'h' and tile.player and tile.player != player:
                    enemy_hq_pos = (tile.x, tile.y)
                    break
            if enemy_hq_pos:
                break
        
        # Collect units that need healing on owned structures
        units_to_heal = []
        
        for unit in self.units:
            # Skip if unit doesn't belong to current player
            if unit.player != player:
                continue
            
            # Skip if unit is already at full health
            if unit.health >= unit.max_health:
                continue
            
            # Check if unit is on an owned structure
            tile = self.grid.get_tile(unit.x, unit.y)
            if not tile or tile.player != player:
                continue
            
            # Determine heal amount based on structure type
            heal_amount = 0
            structure_name = ""
            
            if tile.type == 't':  # Tower
                heal_amount = 1
                structure_name = "Tower"
            elif tile.type == 'h':  # Headquarters
                heal_amount = 2
                structure_name = "Headquarters"
            elif tile.type == 'b':  # Building
                heal_amount = 2
                structure_name = "Building"
            
            if heal_amount > 0:
                # Calculate distance to enemy HQ for priority
                distance = float('inf')
                if enemy_hq_pos:
                    distance = abs(unit.x - enemy_hq_pos[0]) + abs(unit.y - enemy_hq_pos[1])
                
                units_to_heal.append({
                    'unit': unit,
                    'heal_amount': heal_amount,
                    'structure_name': structure_name,
                    'distance': distance
                })
        
        # Sort by distance to enemy HQ (closest first)
        units_to_heal.sort(key=lambda x: x['distance'])
        
        # Process healing for each unit
        total_healing_cost = 0
        healed_any = False
        
        for heal_data in units_to_heal:
            unit = heal_data['unit']
            requested_heal = heal_data['heal_amount']
            structure_name = heal_data['structure_name']
            
            # Calculate maximum HP that can be healed (don't exceed max HP)
            max_possible_heal = unit.max_health - unit.health
            desired_heal = min(requested_heal, max_possible_heal)
            
            # Calculate cost per HP
            unit_cost = UNIT_DATA[unit.type]['cost']
            cost_per_hp = unit_cost / unit.max_health
            
            # Try to heal the full amount, but allow partial healing for HQ/Building
            actual_heal = 0
            actual_cost = 0
            
            if structure_name == "Tower":
                # Towers: All or nothing (1 HP)
                total_cost = round(cost_per_hp * desired_heal)
                
                if self.player_gold[player] >= total_cost:
                    actual_heal = desired_heal
                    actual_cost = total_cost
                else:
                    # Can't afford tower healing, skip
                    continue
            
            else:  # HQ or Building
                # HQ/Building: Allow partial healing
                for hp in range(desired_heal, 0, -1):
                    cost = round(cost_per_hp * hp)
                    if self.player_gold[player] >= cost:
                        actual_heal = hp
                        actual_cost = cost
                        break
            
            # Apply healing if any
            if actual_heal > 0:
                old_health = unit.health
                unit.health = min(unit.health + actual_heal, unit.max_health)
                self.player_gold[player] -= actual_cost
                total_healing_cost += actual_cost
                healed_any = True
                
                # Log healing
                print(f"  {unit.type} on {structure_name} at ({unit.x}, {unit.y}): "
                      f"Healed {actual_heal} HP ({old_health} → {unit.health}) for ${actual_cost}")
        
        if healed_any:
            print(f"  Total healing cost: ${total_healing_cost}")

    def end_turn(self):
        """End the current player's turn and pass to the next player."""
        # Reset structures that were vacated this turn
        for unit in self.units:
            if unit.player == self.current_player and unit.has_moved:
                old_tile = self.grid.get_tile(unit.original_x, unit.original_y)
                if (unit.x, unit.y) != (unit.original_x, unit.original_y):
                    self.mechanics.reset_structure_if_vacated(old_tile, self.units)

        # Regenerate structures
        self.mechanics.regenerate_structures(self.grid, self.units)

        # Move to next player
        self.current_player += 1
        if self.current_player > self.num_players:
            self.current_player = 1
            self.turn_number += 1

        # Handle paralysis and enable units
        self.mechanics.decrement_paralysis(self.units, self.current_player)

        for unit in self.units:
            if unit.player == self.current_player:
                if not unit.is_paralyzed():
                    unit.can_move = True
                    unit.can_attack = True
                else:
                    unit.can_move = False
                    unit.can_attack = False

                unit.original_x = unit.x
                unit.original_y = unit.y
                unit.has_moved = False
            unit.selected = False

        # Calculate and apply income
        income_data = self.mechanics.calculate_income(self.current_player, self.grid)
        self.player_gold[self.current_player] += income_data['total']
        
        # Heal units on structures after income collection
        print(f"  Structure healing:")
        self.heal_units_on_structures(self.current_player)

        return income_data

    def resign(self, player=None):
        """Player resigns."""
        if player is None:
            player = self.current_player

        if self.num_players == 2:
            self.winner = 2 if player == 1 else 1
        else:
            self.winner = player + 1 if player < self.num_players else 1

        self.game_over = True

    def get_legal_actions(self, player=None):
        """
        Get all legal actions for the current player.

        Returns:
            dict: Legal actions organized by type
        """
        if player is None:
            player = self.current_player

        legal_actions = {
            'create_unit': [],
            'move': [],
            'attack': [],
            'paralyze': [],
            'heal': [],
            'cure': [],
            'seize': [],
            'end_turn': True
        }

        # Building units
        for tile in self.grid.get_capturable_tiles(player):
            if tile.type == 'b' and not self.get_unit_at_position(tile.x, tile.y):
                for unit_type in ['W', 'M', 'C']:
                    if self.player_gold[player] >= UNIT_DATA[unit_type]['cost']:
                        legal_actions['create_unit'].append({
                            'unit_type': unit_type,
                            'x': tile.x,
                            'y': tile.y
                        })

        # Unit actions
        for unit in self.units:
            if unit.player == player and not unit.is_paralyzed():
                # Movement
                if unit.can_move:
                    reachable = unit.get_reachable_positions(
                        self.grid.width,
                        self.grid.height,
                        lambda x, y: self.mechanics.can_move_to_position(x, y, self.grid, self.units)
                    )
                    for pos in reachable:
                        legal_actions['move'].append({
                            'unit': unit,
                            'from_x': unit.x,
                            'from_y': unit.y,
                            'to_x': pos[0],
                            'to_y': pos[1]
                        })

                # Combat actions
                if unit.can_attack:
                    adjacent_enemies = self.mechanics.get_adjacent_enemies(unit, self.units)
                    for enemy in adjacent_enemies:
                        legal_actions['attack'].append({
                            'attacker': unit,
                            'target': enemy
                        })

                        if unit.type == 'M':
                            legal_actions['paralyze'].append({
                                'paralyzer': unit,
                                'target': enemy
                            })

                    # Healing (Cleric only)
                    if unit.type == 'C':
                        adjacent_allies = self.mechanics.get_adjacent_allies(unit, self.units)
                        for ally in adjacent_allies:
                            legal_actions['heal'].append({
                                'healer': unit,
                                'target': ally
                            })

                        adjacent_paralyzed = self.mechanics.get_adjacent_paralyzed_allies(unit, self.units)
                        for ally in adjacent_paralyzed:
                            legal_actions['cure'].append({
                                'curer': unit,
                                'target': ally
                            })

                    # Seizing
                    tile = self.grid.get_tile(unit.x, unit.y)
                    if tile.is_capturable() and tile.player != player:
                        legal_actions['seize'].append({
                            'unit': unit,
                            'tile': tile
                        })

        return legal_actions

    def to_dict(self):
        """Convert game state to dictionary for serialization."""
        return {
            'timestamp': self.game_start_time.strftime("%Y-%m-%d %H-%M-%S"),
            'current_player': self.current_player,
            'num_players': self.num_players,
            'player_gold': self.player_gold,
            'turn_number': self.turn_number,
            'game_over': self.game_over,
            'winner': self.winner,
            'units': [unit.to_dict() for unit in self.units],
            'tiles': self.grid.to_dict()['tiles']
        }

    def to_numpy(self):
        """
        Convert game state to numpy arrays for RL.

        Returns:
            dict with numpy arrays
        """
        # Grid representation
        grid_state = self.grid.to_numpy()

        # Unit representation (height x width x 3)
        # Channels: unit_type (0=none, 1=W, 2=M, 3=C), owner, hp_percentage
        unit_state = np.zeros((self.grid.height, self.grid.width, 3), dtype=np.float32)

        unit_type_encoding = {'W': 1, 'M': 2, 'C': 3}

        for unit in self.units:
            unit_state[unit.y, unit.x, 0] = unit_type_encoding[unit.type]
            unit_state[unit.y, unit.x, 1] = unit.player
            unit_state[unit.y, unit.x, 2] = (unit.health / unit.max_health) * 100

        return {
            'grid': grid_state,
            'units': unit_state,
            'gold': np.array([self.player_gold[i] for i in range(1, self.num_players + 1)], dtype=np.float32),
            'current_player': self.current_player,
            'turn_number': self.turn_number
        }

    """
    Add these methods to the GameState class in reinforcetactics/core/game_state.py
    These should be added to the existing GameState class.
    """

    def record_action(self, action_type, **kwargs):
        """
        Record an action for replay purposes.
        
        Args:
            action_type: Type of action (move, attack, create_unit, etc.)
            **kwargs: Action-specific parameters
        """
        action_record = {
            'turn': self.turn_number,
            'player': self.current_player,
            'type': action_type,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.action_history.append(action_record)

    def save_to_file(self, filepath=None):
        """
        Save game state to file.
        
        Args:
            filepath: Path to save file (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        from reinforcetactics.utils.file_io import FileIO
        return FileIO.save_game(self, filepath)

    def save_replay_to_file(self, filepath=None):
        """
        Save replay to file.
        
        Args:
            filepath: Path to replay file (auto-generated if None)
        
        Returns:
            Path to saved replay
        """
        from reinforcetactics.utils.file_io import FileIO
        
        game_info = {
            'num_players': self.num_players,
            'total_turns': self.turn_number,
            'winner': self.winner,
            'game_over': self.game_over,
            'start_time': self.game_start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        return FileIO.save_replay(self.action_history, game_info, filepath)

    @classmethod
    def from_dict(cls, save_data, map_data):
        """
        Restore game state from dictionary.
        
        Args:
            save_data: Dictionary with saved game data
            map_data: Map data (2D array)
        
        Returns:
            Restored GameState instance
        """
        # Create new game state
        game = cls(map_data, save_data.get('num_players', 2))
        
        # Restore basic state
        game.current_player = save_data.get('current_player', 1)
        game.turn_number = save_data.get('turn_number', 0)
        game.game_over = save_data.get('game_over', False)
        game.winner = save_data.get('winner')
        game.player_gold = save_data.get('player_gold', game.player_gold)
        
        # Restore units
        game.units = []
        for unit_data in save_data.get('units', []):
            from reinforcetactics.core.unit import Unit
            unit = Unit.from_dict(unit_data)
            game.units.append(unit)
        
        # Restore tiles
        from reinforcetactics.core.grid import TileGrid
        game.grid = TileGrid.from_dict(save_data, map_data)
        
        return game


    # UPDATED METHODS - Replace existing methods in GameState with these:

    def create_unit(self, unit_type, x, y, player=None):
        """
        Create a unit at the specified position.
        (Updated to record action)
        """
        if player is None:
            player = self.current_player

        if self.get_unit_at_position(x, y):
            return None

        cost = UNIT_DATA[unit_type]['cost']
        if self.player_gold[player] < cost:
            return None

        self.player_gold[player] -= cost
        unit = Unit(unit_type, x, y, player)
        self.units.append(unit)
        
        # Record action
        self.record_action('create_unit', unit_type=unit_type, x=x, y=y, player=player)

        return unit

    def move_unit(self, unit, to_x, to_y):
        """
        Move a unit to a new position.
        (Updated to record action)
        """
        from_x, from_y = unit.x, unit.y
        
        reachable = unit.get_reachable_positions(
            self.grid.width,
            self.grid.height,
            lambda x, y: self.mechanics.can_move_to_position(x, y, self.grid, self.units)
        )

        if (to_x, to_y) not in reachable:
            return False

        if not self.mechanics.can_move_to_position(to_x, to_y, self.grid, self.units):
            return False

        old_tile = self.grid.get_tile(unit.x, unit.y)
        unit.move_to(to_x, to_y)
        
        # Record action
        self.record_action('move', unit_type=unit.type, from_x=from_x, from_y=from_y, 
                        to_x=to_x, to_y=to_y, player=unit.player)

        return True

    def attack(self, attacker, target):
        """
        Execute an attack.
        (Updated to record action)
        """
        result = self.mechanics.attack_unit(attacker, target)

        # Record action
        self.record_action('attack', 
                        attacker_type=attacker.type,
                        attacker_pos=(attacker.x, attacker.y),
                        target_type=target.type,
                        target_pos=(target.x, target.y),
                        damage=result['damage'],
                        target_killed=not result['target_alive'],
                        player=attacker.player)

        if not result['target_alive']:
            target_tile = self.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            self.units.remove(target)

        if not result['attacker_alive']:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            self.units.remove(attacker)

        attacker.can_move = False
        attacker.can_attack = False

        return result

    def seize(self, unit):
        """
        Seize the structure the unit is on.
        (Updated to record action)
        """
        tile = self.grid.get_tile(unit.x, unit.y)
        result = self.mechanics.seize_structure(unit, tile)
        
        # Record action
        self.record_action('seize',
                        unit_type=unit.type,
                        position=(unit.x, unit.y),
                        structure_type=tile.type,
                        captured=result['captured'],
                        player=unit.player)

        if result['game_over']:
            self.game_over = True
            self.winner = unit.player

        unit.can_move = False
        unit.can_attack = False

        return result

    def end_turn(self):
        """
        End the current player's turn and pass to the next player.
        (Updated to record action)
        """
        # Record action
        self.record_action('end_turn', player=self.current_player)
        
        # Reset structures that were vacated this turn
        for unit in self.units:
            if unit.player == self.current_player and unit.has_moved:
                old_tile = self.grid.get_tile(unit.original_x, unit.original_y)
                if (unit.x, unit.y) != (unit.original_x, unit.original_y):
                    self.mechanics.reset_structure_if_vacated(old_tile, self.units)

        # Regenerate structures
        self.mechanics.regenerate_structures(self.grid, self.units)

        # Move to next player
        self.current_player += 1
        if self.current_player > self.num_players:
            self.current_player = 1
            self.turn_number += 1

        # Handle paralysis and enable units
        self.mechanics.decrement_paralysis(self.units, self.current_player)

        for unit in self.units:
            if unit.player == self.current_player:
                if not unit.is_paralyzed():
                    unit.can_move = True
                    unit.can_attack = True
                else:
                    unit.can_move = False
                    unit.can_attack = False

                unit.original_x = unit.x
                unit.original_y = unit.y
                unit.has_moved = False
            unit.selected = False

        # Calculate and apply income
        income_data = self.mechanics.calculate_income(self.current_player, self.grid)
        self.player_gold[self.current_player] += income_data['total']
        
        # Heal units on structures after income collection
        self.heal_units_on_structures(self.current_player)

        return income_data