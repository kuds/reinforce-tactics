"""
Core game state management without rendering dependencies.
Fixed version: removed duplicate methods, added type hints, controlled logging.
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable

import numpy as np
import pandas as pd

from reinforcetactics.core.unit import Unit
from reinforcetactics.core.grid import TileGrid
from reinforcetactics.game.mechanics import GameMechanics
from reinforcetactics.constants import STARTING_GOLD, UNIT_DATA, TileType

# Configure logging
logger = logging.getLogger(__name__)


class GameState:
    """Manages the core game state without rendering."""

    def __init__(self, map_data, num_players: int = 2) -> None:
        """
        Initialize the game state.

        Args:
            map_data: 2D array containing map information
            num_players: Number of players (default 2)
        """
        self.grid = TileGrid(map_data)
        self.units: List[Unit] = []
        self.current_player: int = 1
        self.num_players: int = num_players
        self.player_gold: Dict[int, int] = {i: STARTING_GOLD for i in range(1, num_players + 1)}
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.turn_number: int = 0
        self.mechanics = GameMechanics()

        # Optional map file reference for saving
        self.map_file_used: Optional[str] = None

        # Original map dimensions (before padding)
        # These default to the current grid dimensions if not set
        self.original_map_width: int = self.grid.width
        self.original_map_height: int = self.grid.height
        self.map_padding_offset_x: int = 0
        self.map_padding_offset_y: int = 0

        # Store initial map data for replays (as 2D list of tile codes)
        # This stores the PADDED map by default
        if isinstance(map_data, pd.DataFrame):
            self.initial_map_data: List[List[str]] = map_data.values.tolist()
        elif isinstance(map_data, np.ndarray):
            self.initial_map_data: List[List[str]] = map_data.tolist()
        else:
            self.initial_map_data: List[List[str]] = [list(row) for row in map_data]

        # Store original unpadded map data (will be set via set_map_metadata if map was padded)
        # If not set, defaults to the same as initial_map_data (no padding)
        self.original_map_data: Optional[List[List[str]]] = None

        # Player configurations (human vs bot)
        self.player_configs: List[Dict[str, Any]] = []

        # Maximum turns for the game (None = unlimited)
        self.max_turns: Optional[int] = None

        # Draw reason (only set when game ends in a draw)
        # Possible values: "max_turns", None
        self.draw_reason: Optional[str] = None

        # Action history for replay
        self.action_history: List[Dict[str, Any]] = []
        self.game_start_time: datetime = datetime.now()

        # Cached values for performance
        self._unit_count_cache: Dict[int, int] = {}
        self._cache_valid: bool = False

    def reset(self, map_data) -> None:
        """Reset the game state."""
        self.__init__(map_data, self.num_players)

    def set_map_metadata(self, original_width: int, original_height: int,
                         padding_offset_x: int, padding_offset_y: int,
                         map_file: Optional[str] = None,
                         original_map_data: Optional[List[List[str]]] = None) -> None:
        """
        Set metadata about the original map before padding.

        Args:
            original_width: Width of the map before padding
            original_height: Height of the map before padding
            padding_offset_x: X offset added by padding (left side)
            padding_offset_y: Y offset added by padding (top side)
            map_file: Path to the map file
            original_map_data: The unpadded map data (2D list of tile codes)
        """
        self.original_map_width = original_width
        self.original_map_height = original_height
        self.map_padding_offset_x = padding_offset_x
        self.map_padding_offset_y = padding_offset_y
        if map_file:
            self.map_file_used = map_file
        if original_map_data:
            self.original_map_data = original_map_data

    def padded_to_original_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert padded map coordinates to original map coordinates.

        Args:
            x: X coordinate in padded map
            y: Y coordinate in padded map

        Returns:
            Tuple of (original_x, original_y)
        """
        return (x - self.map_padding_offset_x, y - self.map_padding_offset_y)

    def original_to_padded_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert original map coordinates to padded map coordinates.

        Args:
            x: X coordinate in original map
            y: Y coordinate in original map

        Returns:
            Tuple of (padded_x, padded_y)
        """
        return (x + self.map_padding_offset_x, y + self.map_padding_offset_y)

    def _invalidate_cache(self) -> None:
        """Invalidate cached values."""
        self._cache_valid = False
        self._unit_count_cache.clear()

    def get_unit_count(self, player: int) -> int:
        """Get cached unit count for a player."""
        if not self._cache_valid:
            self._unit_count_cache = {}
            for unit in self.units:
                self._unit_count_cache[unit.player] = self._unit_count_cache.get(unit.player, 0) + 1
            self._cache_valid = True
        return self._unit_count_cache.get(player, 0)

    def get_unit_at_position(self, x: int, y: int) -> Optional[Unit]:
        """Get the unit at a grid position."""
        for unit in self.units:
            if unit.x == x and unit.y == y:
                return unit
        return None

    def record_action(self, action_type: str, **kwargs) -> None:
        """
        Record an action for replay purposes.
        
        Automatically converts any coordinate parameters from padded to original coordinates.

        Args:
            action_type: Type of action (move, attack, create_unit, etc.)
            **kwargs: Action-specific parameters (coordinates will be converted)
        """
        # Convert coordinate parameters from padded to original
        converted_kwargs = {}
        for key, value in kwargs.items():
            if key in ['x', 'y', 'from_x', 'from_y', 'to_x', 'to_y']:
                # Single coordinate value
                if key.endswith('_x'):
                    # Store x coordinate to pair with y
                    converted_kwargs[key] = value
                elif key.endswith('_y'):
                    # Convert the x,y pair
                    x_key = key.replace('_y', '_x')
                    if x_key in kwargs:
                        orig_x, orig_y = self.padded_to_original_coords(kwargs[x_key], value)
                        converted_kwargs[x_key] = orig_x
                        converted_kwargs[key] = orig_y
                    else:
                        converted_kwargs[key] = value
                elif key == 'x':
                    # Will be converted when we see 'y'
                    converted_kwargs[key] = value
                elif key == 'y':
                    # Convert x,y pair
                    if 'x' in kwargs:
                        orig_x, orig_y = self.padded_to_original_coords(kwargs['x'], value)
                        converted_kwargs['x'] = orig_x
                        converted_kwargs[key] = orig_y
                    else:
                        converted_kwargs[key] = value
            elif key in ['position', 'attacker_pos', 'target_pos', 'healer_pos', 'paralyzer_pos', 'curer_pos']:
                # Tuple/list of (x, y) coordinates
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    orig_x, orig_y = self.padded_to_original_coords(value[0], value[1])
                    converted_kwargs[key] = (orig_x, orig_y)
                else:
                    converted_kwargs[key] = value
            else:
                # Non-coordinate parameter, keep as-is
                converted_kwargs[key] = value
        
        action_record = {
            'turn': self.turn_number,
            'player': self.current_player,
            'type': action_type,
            'timestamp': datetime.now().isoformat(),
            **converted_kwargs
        }
        self.action_history.append(action_record)

    def create_unit(self, unit_type: str, x: int, y: int,
                    player: Optional[int] = None) -> Optional[Unit]:
        """
        Create a unit at the specified position.

        Args:
            unit_type: 'W', 'M', 'C', 'B', or 'A'
            x: Grid x coordinate
            y: Grid y coordinate
            player: Player number (defaults to current player)

        Returns:
            Unit if created, None if failed
        """
        if player is None:
            player = self.current_player

        # Check if position is occupied
        if self.get_unit_at_position(x, y):
            logger.debug(f"Cannot create unit at ({x}, {y}): position occupied")
            return None

        # Check if player can afford
        if unit_type not in UNIT_DATA:
            logger.warning(f"Unknown unit type: {unit_type}")
            return None

        cost = UNIT_DATA[unit_type]['cost']
        if self.player_gold[player] < cost:
            logger.debug(f"Cannot create unit: insufficient gold ({self.player_gold[player]} < {cost})")
            return None

        # Create the unit
        self.player_gold[player] -= cost
        unit = Unit(unit_type, x, y, player)
        self.units.append(unit)
        self._invalidate_cache()

        # Record action
        self.record_action('create_unit', unit_type=unit_type, x=x, y=y, player=player)

        logger.debug(f"Player {player} created {unit_type} at ({x}, {y})")
        return unit

    def move_unit(self, unit: Unit, to_x: int, to_y: int) -> bool:
        """
        Move a unit to a new position.

        Args:
            unit: Unit to move
            to_x: Target x coordinate
            to_y: Target y coordinate

        Returns:
            bool: True if move successful
        """
        from_x, from_y = unit.x, unit.y

        # Check if move is valid
        reachable = unit.get_reachable_positions(
            self.grid.width,
            self.grid.height,
            lambda x, y: self.mechanics.can_move_to_position(
                x, y, self.grid, self.units, moving_unit=unit, is_destination=False
            )
        )

        if (to_x, to_y) not in reachable:
            logger.debug(f"Cannot move to ({to_x}, {to_y}): not reachable")
            return False

        if not self.mechanics.can_move_to_position(
            to_x, to_y, self.grid, self.units, moving_unit=unit, is_destination=True
        ):
            logger.debug(f"Cannot move to ({to_x}, {to_y}): position blocked")
            return False

        # Execute move
        unit.move_to(to_x, to_y)

        # Record action
        self.record_action('move', unit_type=unit.type, from_x=from_x, from_y=from_y,
                          to_x=to_x, to_y=to_y, player=unit.player)

        logger.debug(f"Moved {unit.type} from ({from_x}, {from_y}) to ({to_x}, {to_y})")
        return True

    def attack(self, attacker: Unit, target: Unit) -> Dict[str, Any]:
        """
        Execute an attack.

        Args:
            attacker: Attacking unit
            target: Target unit

        Returns:
            dict: Attack results
        """
        result = self.mechanics.attack_unit(attacker, target, self.grid)

        # Record action
        self.record_action('attack',
                          attacker_type=attacker.type,
                          attacker_pos=(attacker.x, attacker.y),
                          target_type=target.type,
                          target_pos=(target.x, target.y),
                          damage=result['damage'],
                          target_killed=not result['target_alive'],
                          player=attacker.player)

        # Handle unit deaths
        if not result['target_alive']:
            target_tile = self.grid.get_tile(target.x, target.y)
            if target_tile.is_capturable() and target_tile.health < target_tile.max_health:
                target_tile.regenerating = True
            defeated_player = target.player
            self.units.remove(target)
            self._invalidate_cache()

            # Check if defeated player has any remaining units
            remaining_units = [u for u in self.units if u.player == defeated_player]
            if len(remaining_units) == 0:
                self.game_over = True
                # Determine winner using same logic as resign method
                if self.num_players == 2:
                    self.winner = 2 if defeated_player == 1 else 1
                else:
                    self.winner = defeated_player + 1 if defeated_player < self.num_players else 1

        if not result['attacker_alive']:
            attacker_tile = self.grid.get_tile(attacker.x, attacker.y)
            if attacker_tile.is_capturable() and attacker_tile.health < attacker_tile.max_health:
                attacker_tile.regenerating = True
            defeated_player = attacker.player
            self.units.remove(attacker)
            self._invalidate_cache()

            # Check if defeated player has any remaining units
            remaining_units = [u for u in self.units if u.player == defeated_player]
            if len(remaining_units) == 0:
                self.game_over = True
                # Determine winner using same logic as resign method
                if self.num_players == 2:
                    self.winner = 2 if defeated_player == 1 else 1
                else:
                    self.winner = defeated_player + 1 if defeated_player < self.num_players else 1

        attacker.can_move = False
        attacker.can_attack = False

        return result

    def paralyze(self, paralyzer: Unit, target: Unit) -> bool:
        """Paralyze a target unit."""
        result = self.mechanics.paralyze_unit(paralyzer, target)
        if result:
            paralyzer.can_move = False
            paralyzer.can_attack = False
            self.record_action('paralyze',
                              paralyzer_pos=(paralyzer.x, paralyzer.y),
                              target_pos=(target.x, target.y),
                              player=paralyzer.player)
        return result

    def heal(self, healer: Unit, target: Unit) -> int:
        """Heal a target unit."""
        amount = self.mechanics.heal_unit(healer, target)
        if amount > 0:
            healer.can_move = False
            healer.can_attack = False
            self.record_action('heal',
                              healer_pos=(healer.x, healer.y),
                              target_pos=(target.x, target.y),
                              amount=amount,
                              player=healer.player)
        return amount

    def cure(self, curer: Unit, target: Unit) -> bool:
        """Cure a target unit's paralysis."""
        result = self.mechanics.cure_unit(curer, target)
        if result:
            curer.can_move = False
            curer.can_attack = False
            self.record_action('cure',
                              curer_pos=(curer.x, curer.y),
                              target_pos=(target.x, target.y),
                              player=curer.player)
        return result

    def seize(self, unit: Unit) -> Dict[str, Any]:
        """Seize the structure the unit is on."""
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

    def heal_units_on_structures(self, player: int) -> Dict[str, Any]:
        """
        Heal units on owned structures at the start of their turn.

        Healing amounts:
        - Tower: 1 HP
        - HQ/Building: 2 HP

        Cost formula: (heal_amount / unit_max_hp) * unit_cost (rounded)

        Args:
            player: Player number whose units to heal

        Returns:
            Dict with healing statistics
        """
        stats = {'total_healed': 0, 'total_cost': 0, 'units_healed': []}

        # Find enemy HQ for distance calculations
        enemy_hq_pos = None
        for row in self.grid.tiles:
            for tile in row:
                if tile.type == TileType.HEADQUARTERS.value and tile.player and tile.player != player:
                    enemy_hq_pos = (tile.x, tile.y)
                    break
            if enemy_hq_pos:
                break

        # Collect units that need healing on owned structures
        units_to_heal = []

        for unit in self.units:
            if unit.player != player:
                continue
            if unit.health >= unit.max_health:
                continue

            tile = self.grid.get_tile(unit.x, unit.y)
            if not tile or tile.player != player:
                continue

            # Determine heal amount based on structure type
            heal_amount = 0
            structure_name = ""

            if tile.type == TileType.TOWER.value:
                heal_amount = 1
                structure_name = "Tower"
            elif tile.type == TileType.HEADQUARTERS.value:
                heal_amount = 2
                structure_name = "Headquarters"
            elif tile.type == TileType.BUILDING.value:
                heal_amount = 2
                structure_name = "Building"

            if heal_amount > 0:
                distance = float('inf')
                if enemy_hq_pos:
                    distance = abs(unit.x - enemy_hq_pos[0]) + abs(unit.y - enemy_hq_pos[1])

                units_to_heal.append({
                    'unit': unit,
                    'heal_amount': heal_amount,
                    'structure_name': structure_name,
                    'distance': distance
                })

        # Sort by distance to enemy HQ (closest first - priority)
        units_to_heal.sort(key=lambda x: x['distance'])

        # Process healing for each unit
        for heal_data in units_to_heal:
            unit = heal_data['unit']
            requested_heal = heal_data['heal_amount']
            structure_name = heal_data['structure_name']

            max_possible_heal = unit.max_health - unit.health
            desired_heal = min(requested_heal, max_possible_heal)

            unit_cost = UNIT_DATA[unit.type]['cost']
            cost_per_hp = unit_cost / unit.max_health

            actual_heal = 0
            actual_cost = 0

            if structure_name == "Tower":
                # Towers: All or nothing (1 HP)
                total_cost = round(cost_per_hp * desired_heal)
                if self.player_gold[player] >= total_cost:
                    actual_heal = desired_heal
                    actual_cost = total_cost
            else:  # HQ or Building - allow partial healing
                for hp in range(desired_heal, 0, -1):
                    cost = round(cost_per_hp * hp)
                    if self.player_gold[player] >= cost:
                        actual_heal = hp
                        actual_cost = cost
                        break

            if actual_heal > 0:
                old_health = unit.health
                unit.health = min(unit.health + actual_heal, unit.max_health)
                self.player_gold[player] -= actual_cost

                stats['total_healed'] += actual_heal
                stats['total_cost'] += actual_cost
                stats['units_healed'].append({
                    'unit_type': unit.type,
                    'position': (unit.x, unit.y),
                    'structure': structure_name,
                    'healed': actual_heal,
                    'cost': actual_cost,
                    'old_health': old_health,
                    'new_health': unit.health
                })

                logger.debug(f"Healed {unit.type} on {structure_name} at ({unit.x}, {unit.y}): "
                           f"{actual_heal} HP ({old_health} → {unit.health}) for ${actual_cost}")

        return stats

    def end_turn(self) -> Dict[str, Any]:
        """End the current player's turn and pass to the next player."""
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

            # Check for draw due to max turns after completing a full round
            if self.max_turns is not None and self.turn_number >= self.max_turns:
                self.game_over = True
                self.winner = 0  # 0 indicates a draw
                self.draw_reason = "max_turns"
                self.record_action('draw', reason='max_turns', turn=self.turn_number)
                logger.info(f"Game ended in draw: max turns ({self.max_turns}) reached")

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
        healing_stats = self.heal_units_on_structures(self.current_player)
        income_data['healing'] = healing_stats

        return income_data

    def resign(self, player: Optional[int] = None) -> None:
        """Player resigns."""
        if player is None:
            player = self.current_player

        self.record_action('resign', player=player)

        if self.num_players == 2:
            self.winner = 2 if player == 1 else 1
        else:
            self.winner = player + 1 if player < self.num_players else 1

        self.game_over = True

    @property
    def is_draw(self) -> bool:
        """Check if the game ended in a draw."""
        return self.game_over and self.winner == 0

    def check_draw_condition(self) -> bool:
        """
        Check if draw conditions are met and trigger draw if so.

        Returns:
            True if game ended in a draw, False otherwise
        """
        if self.game_over:
            return self.is_draw

        # Check max turns condition
        if self.max_turns is not None and self.turn_number >= self.max_turns:
            self.game_over = True
            self.winner = 0
            self.draw_reason = "max_turns"
            self.record_action('draw', reason='max_turns', turn=self.turn_number)
            logger.info(f"Game ended in draw: max turns ({self.max_turns}) reached")
            return True

        return False

    def get_legal_actions(self, player: Optional[int] = None) -> Dict[str, List[Any]]:
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

        # Building units (only at Buildings, not HQ)
        for tile in self.grid.get_capturable_tiles(player):
            if tile.type == TileType.BUILDING.value and not self.get_unit_at_position(tile.x, tile.y):
                for unit_type in ['W', 'M', 'C', 'A']:
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
                    # For Archers and Mages, find enemies within range (not just adjacent)
                    if unit.type in ['M', 'A']:
                        # Check if unit is on mountain (for Archer range bonus)
                        unit_tile = self.grid.get_tile(unit.x, unit.y)
                        on_mountain = (unit_tile.type == 'm')

                        for enemy in self.units:
                            if enemy.player != player:
                                damage = unit.get_attack_damage(enemy.x, enemy.y, on_mountain)
                                if damage > 0:
                                    legal_actions['attack'].append({
                                        'attacker': unit,
                                        'target': enemy
                                    })

                                    if unit.type == 'M':
                                        # Mages can also paralyze at range
                                        distance = abs(unit.x - enemy.x) + abs(unit.y - enemy.y)
                                        if distance <= 2:
                                            legal_actions['paralyze'].append({
                                                'paralyzer': unit,
                                                'target': enemy
                                            })
                    else:
                        # For other units, only adjacent enemies
                        adjacent_enemies = self.mechanics.get_adjacent_enemies(unit, self.units)
                        for enemy in adjacent_enemies:
                            legal_actions['attack'].append({
                                'attacker': unit,
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert game state to dictionary for serialization."""
        return {
            'timestamp': self.game_start_time.strftime("%Y-%m-%d %H-%M-%S"),
            'current_player': self.current_player,
            'num_players': self.num_players,
            'player_gold': self.player_gold,
            'turn_number': self.turn_number,
            'game_over': self.game_over,
            'winner': self.winner,
            'map_file': self.map_file_used,
            'player_configs': self.player_configs,
            'units': [unit.to_dict() for unit in self.units],
            'tiles': self.grid.to_dict()['tiles']
        }

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        Convert game state to numpy arrays for RL.

        Returns:
            dict with numpy arrays
        """
        # Grid representation
        grid_state = self.grid.to_numpy()

        # Unit representation (height x width x 3)
        unit_state = np.zeros((self.grid.height, self.grid.width, 3), dtype=np.float32)

        unit_type_encoding = {'W': 1, 'M': 2, 'C': 3, 'B': 4, 'A': 5}

        for unit in self.units:
            unit_state[unit.y, unit.x, 0] = unit_type_encoding.get(unit.type, 0)
            unit_state[unit.y, unit.x, 1] = unit.player
            unit_state[unit.y, unit.x, 2] = (unit.health / unit.max_health) * 100

        return {
            'grid': grid_state,
            'units': unit_state,
            'gold': np.array([self.player_gold[i] for i in range(1, self.num_players + 1)], dtype=np.float32),
            'current_player': self.current_player,
            'turn_number': self.turn_number
        }

    def save_to_file(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Save game state to file.

        Args:
            filepath: Path to save file (auto-generated if None)

        Returns:
            Path to saved file
        """
        from reinforcetactics.utils.file_io import FileIO
        return FileIO.save_game(self, filepath)

    def _get_player_type(self, config: Dict[str, Any]) -> str:
        """
        Get the standardized player type for replay logs.

        Args:
            config: Player configuration dictionary

        Returns:
            Player type string: 'human', 'bot', 'llm', or 'rl'
        """
        if config.get('type') == 'human':
            return 'human'

        bot_type = config.get('bot_type', '')

        # LLM bots
        if bot_type in ('OpenAIBot', 'ClaudeBot', 'GeminiBot'):
            return 'llm'

        # RL model bots
        if bot_type == 'ModelBot':
            return 'rl'

        # Standard bots (SimpleBot, MediumBot, AdvancedBot)
        return 'bot'

    def save_replay_to_file(self, filepath: Optional[str] = None) -> Optional[str]:
        """
        Save replay to file.

        Args:
            filepath: Path to replay file (auto-generated if None)

        Returns:
            Path to saved replay
        """
        from reinforcetactics.utils.file_io import FileIO

        # Use original unpadded map if available, otherwise use initial_map_data
        map_to_save = self.original_map_data if self.original_map_data else self.initial_map_data

        # Build enhanced player_configs from player_configs
        enhanced_player_configs = []
        for i, config in enumerate(self.player_configs):
            player_num = i + 1
            player_name = config.get('player_name', 'Unknown')

            # Build enhanced config with standardized structure
            enhanced_config = {
                'player_no': player_num,
                'type': config.get('player_type', self._get_player_type(config)),
                'name': player_name
            }

            # Add LLM-specific fields if applicable
            if enhanced_config['type'] == 'llm':
                enhanced_config['temperature'] = config.get('temperature', None)
                enhanced_config['max_tokens'] = config.get('max_tokens', None)

            enhanced_player_configs.append(enhanced_config)

        game_info = {
            'num_players': self.num_players,
            'max_turns': self.max_turns,
            'total_turns': self.turn_number,
            'winner': self.winner,
            'game_over': self.game_over,
            'draw_reason': self.draw_reason,  # Only set when winner == 0 (draw)
            'start_time': self.game_start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'map_file': self.map_file_used,
            'initial_map': map_to_save,
            'player_configs': enhanced_player_configs
        }

        return FileIO.save_replay(self.action_history, game_info, filepath)

    @classmethod
    def from_dict(cls, save_data: Dict[str, Any], map_data) -> 'GameState':
        """
        Restore game state from dictionary.

        Args:
            save_data: Dictionary with saved game data
            map_data: Map data (2D array)

        Returns:
            Restored GameState instance
        """
        game = cls(map_data, save_data.get('num_players', 2))

        game.current_player = save_data.get('current_player', 1)
        game.turn_number = save_data.get('turn_number', 0)
        game.game_over = save_data.get('game_over', False)
        game.winner = save_data.get('winner')

        # Fix player_gold dictionary key type (JSON serializes as strings)
        saved_gold = save_data.get('player_gold', {})
        game.player_gold = {int(k): v for k, v in saved_gold.items()}

        game.map_file_used = save_data.get('map_file')

        # Restore player_configs (backward compatible with old saves)
        game.player_configs = save_data.get('player_configs', [])

        # Restore units
        game.units = []
        for unit_data in save_data.get('units', []):
            unit = Unit.from_dict(unit_data)
            game.units.append(unit)

        # Restore tile states
        for tile_data in save_data.get('tiles', []):
            x, y = tile_data['x'], tile_data['y']
            if 0 <= x < game.grid.width and 0 <= y < game.grid.height:
                tile = game.grid.tiles[y][x]
                if tile_data.get('player'):
                    tile.player = tile_data['player']
                if tile_data.get('health') is not None:
                    tile.health = tile_data['health']
                if tile_data.get('regenerating') is not None:
                    tile.regenerating = tile_data['regenerating']

        game._invalidate_cache()
        return game
