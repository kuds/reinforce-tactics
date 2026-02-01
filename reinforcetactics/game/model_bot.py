"""
Model-based bot that uses trained Stable-Baselines3 models.
"""
import logging
from pathlib import Path
from typing import Optional, Any

import numpy as np

from reinforcetactics.constants import UNIT_DATA

# Configure logging
logger = logging.getLogger(__name__)


class ModelBot:  # pylint: disable=too-few-public-methods
    """Bot that uses a trained Stable-Baselines3 model for decision-making."""

    # Canonical unit type ordering — must match StrategyGameEnv action space indices
    ALL_UNIT_TYPES = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    # Number of action types in the environment (0-9)
    NUM_ACTION_TYPES = 10

    def __init__(self, game_state, player: int = 2, model_path: Optional[str] = None):
        """
        Initialize the model bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot (default 2)
            model_path: Path to the trained model .zip file
        """
        self.game_state = game_state
        self.bot_player = player
        self.model_path = model_path
        self.model = None
        self.env = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """
        Load a Stable-Baselines3 model from file.

        Args:
            model_path: Path to the model .zip file
        """
        try:
            # Import here to avoid dependency issues if SB3 not installed
            from stable_baselines3 import PPO, A2C, DQN
            from reinforcetactics.rl.gym_env import StrategyGameEnv

            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Try to load with different algorithms
            for algorithm_class in [PPO, A2C, DQN]:
                try:
                    self.model = algorithm_class.load(str(model_path))
                    logger.info("Successfully loaded model as %s: %s",
                               algorithm_class.__name__, model_path)
                    break
                except Exception as e:
                    logger.debug("Failed to load as %s: %s", algorithm_class.__name__, e)
                    continue

            if self.model is None:
                raise ValueError(f"Could not load model with any supported algorithm: {model_path}")

            # Create a dummy environment for observation space info
            # We'll use the actual game state for observations
            self.env = StrategyGameEnv(
                map_file=None,
                opponent=None,
                render_mode=None
            )

        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for ModelBot. "
                "Install it with: pip install stable-baselines3"
            ) from e
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def take_turn(self) -> None:
        """Execute the bot's turn using the trained model."""
        if self.model is None:
            logger.warning("No model loaded, ending turn")
            self.game_state.end_turn()
            return

        try:
            # Keep taking actions until we decide to end turn or hit invalid actions
            max_actions_per_turn = 50  # Safety limit
            actions_taken = 0

            while actions_taken < max_actions_per_turn:
                # Get observation from current game state
                obs = self._get_observation()

                # Predict action using the model
                action, _states = self.model.predict(obs, deterministic=True)

                # Execute the action
                action_valid = self._execute_action(action)

                # If action was end_turn or invalid, stop
                if not action_valid or self._is_end_turn_action(action):
                    break

                actions_taken += 1

            # Make sure turn is ended
            if self.game_state.current_player == self.bot_player:
                self.game_state.end_turn()

        except Exception as e:
            logger.error("Error during model bot turn: %s", e)
            # Fallback: just end turn
            if self.game_state.current_player == self.bot_player:
                self.game_state.end_turn()

    def _get_observation(self) -> Any:
        """
        Get observation from current game state in the format expected by the model.

        Returns:
            Observation dict compatible with StrategyGameEnv
        """
        # Convert game state to observation format
        state_arrays = self.game_state.to_numpy()

        obs = {
            'grid': state_arrays['grid'].astype(np.float32),
            'units': state_arrays['units'].astype(np.float32),
            'global_features': np.array([
                self.game_state.player_gold.get(1, 0),
                self.game_state.player_gold.get(2, 0),
                self.game_state.turn_number,
                len([u for u in self.game_state.units if u.player == 1]),
                len([u for u in self.game_state.units if u.player == 2]),
                self.game_state.current_player
            ], dtype=np.float32),
            'action_mask': np.ones(
                self.NUM_ACTION_TYPES * self.game_state.grid.width * self.game_state.grid.height,
                dtype=np.float32
            )
        }

        return obs

    def _execute_action(self, action) -> bool:  # pylint: disable=too-many-return-statements
        """
        Execute a model action in the game.

        Args:
            action: Action from the model (MultiDiscrete format)

        Returns:
            True if action was valid and executed, False otherwise
        """
        try:
            # Action format: [action_type, unit_type, from_x, from_y, to_x, to_y]
            if isinstance(action, np.ndarray):
                action = action.tolist()

            if not isinstance(action, (list, tuple)) or len(action) < 6:
                logger.warning("Invalid action format: %s", action)
                return False

            action_type, unit_type, from_x, from_y, to_x, to_y = action[:6]

            # Map action types: 0=create, 1=move, 2=attack, 3=seize, 4=heal/cure,
            # 5=end_turn, 6=paralyze, 7=haste, 8=defence_buff, 9=attack_buff
            if action_type == 0:  # Create unit
                return self._create_unit(unit_type, to_x, to_y)
            if action_type == 1:  # Move
                return self._move_unit(from_x, from_y, to_x, to_y)
            if action_type == 2:  # Attack
                return self._attack(from_x, from_y, to_x, to_y)
            if action_type == 3:  # Seize
                return self._seize(from_x, from_y)
            if action_type == 4:  # Heal/Cure
                return self._heal(from_x, from_y, to_x, to_y)
            if action_type == 5:  # End turn
                return True  # Will be handled by caller
            if action_type == 6:  # Paralyze (Mage/Sorcerer)
                return self._paralyze(from_x, from_y, to_x, to_y)
            if action_type == 7:  # Haste (Sorcerer)
                return self._buff(from_x, from_y, to_x, to_y, 'haste')
            if action_type == 8:  # Defence Buff (Sorcerer)
                return self._buff(from_x, from_y, to_x, to_y, 'defence_buff')
            if action_type == 9:  # Attack Buff (Sorcerer)
                return self._buff(from_x, from_y, to_x, to_y, 'attack_buff')

            logger.warning("Unknown action type: %s", action_type)
            return False

        except Exception as e:
            logger.warning("Error executing action: %s", e)
            return False

    def _create_unit(self, unit_type: int, x: int, y: int) -> bool:  # pylint: disable=too-many-return-statements
        """Create a unit at the specified location."""
        try:
            # Map unit_type index to unit code using canonical ordering
            if unit_type < 0 or unit_type >= len(self.ALL_UNIT_TYPES):
                return False

            unit_code = self.ALL_UNIT_TYPES[unit_type]

            # Check if this unit type is enabled in the current game
            if hasattr(self.game_state, 'is_unit_type_enabled'):
                if not self.game_state.is_unit_type_enabled(unit_code):
                    return False

            # Check if we have enough gold
            cost = UNIT_DATA[unit_code]['cost']
            if self.game_state.player_gold.get(self.bot_player, 0) < cost:
                return False

            # Check if location is valid for creation
            if not (0 <= x < self.game_state.grid.width and 0 <= y < self.game_state.grid.height):
                return False

            tile = self.game_state.grid.get_tile(x, y)
            if tile.player != self.bot_player or tile.type != 'b':
                return False

            # Check if location is occupied
            if self.game_state.get_unit_at_position(x, y):
                return False

            # Create the unit
            self.game_state.create_unit(unit_code, x, y, self.bot_player)
            return True

        except Exception as e:
            logger.debug("Failed to create unit: %s", e)
            return False

    def _move_unit(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Move a unit from one location to another."""
        try:
            unit = self.game_state.get_unit_at_position(from_x, from_y)
            if not unit or unit.player != self.bot_player or not unit.can_move:
                return False

            self.game_state.move_unit(unit, to_x, to_y)
            return True

        except Exception as e:
            logger.debug("Failed to move unit: %s", e)
            return False

    def _attack(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Attack with a unit."""
        try:
            attacker = self.game_state.get_unit_at_position(from_x, from_y)
            target = self.game_state.get_unit_at_position(to_x, to_y)

            if not attacker or not target:
                return False

            if attacker.player != self.bot_player or target.player == self.bot_player:
                return False

            if not attacker.can_attack:
                return False

            self.game_state.attack(attacker, target)
            return True

        except Exception as e:
            logger.debug("Failed to attack: %s", e)
            return False

    def _seize(self, x: int, y: int) -> bool:
        """Seize a structure."""
        try:
            unit = self.game_state.get_unit_at_position(x, y)
            if not unit or unit.player != self.bot_player:
                return False

            tile = self.game_state.grid.get_tile(x, y)
            if not tile.is_capturable() or tile.player == self.bot_player:
                return False

            self.game_state.seize(unit)
            return True

        except Exception as e:
            logger.debug("Failed to seize: %s", e)
            return False

    def _heal(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Heal or cure a unit (Cleric only). Prioritizes cure if target is paralyzed."""
        try:
            healer = self.game_state.get_unit_at_position(from_x, from_y)
            target = self.game_state.get_unit_at_position(to_x, to_y)

            if not healer or not target:
                return False

            if healer.player != self.bot_player or target.player != self.bot_player:
                return False

            if healer.type != 'C':  # Only clerics can heal/cure
                return False

            # Priority: cure if paralyzed, otherwise heal (matches gym_env logic)
            if target.is_paralyzed():
                result = self.game_state.cure(healer, target)
                if result:
                    return True

            heal_amount = self.game_state.heal(healer, target)
            return heal_amount > 0

        except Exception as e:
            logger.debug("Failed to heal: %s", e)
            return False

    def _paralyze(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Paralyze an enemy unit (Mage/Sorcerer)."""
        try:
            unit = self.game_state.get_unit_at_position(from_x, from_y)
            target = self.game_state.get_unit_at_position(to_x, to_y)

            if not unit or not target:
                return False

            if unit.player != self.bot_player or target.player == self.bot_player:
                return False

            if unit.type not in ('M', 'S'):
                return False

            return self.game_state.paralyze(unit, target)

        except Exception as e:
            logger.debug("Failed to paralyze: %s", e)
            return False

    def _buff(self, from_x: int, from_y: int, to_x: int, to_y: int,
              buff_type: str) -> bool:
        """Apply a Sorcerer buff (haste, defence_buff, or attack_buff) to a friendly unit."""
        try:
            unit = self.game_state.get_unit_at_position(from_x, from_y)
            target = self.game_state.get_unit_at_position(to_x, to_y)

            if not unit or not target:
                return False

            if unit.player != self.bot_player or target.player != self.bot_player:
                return False

            if unit.type != 'S':  # Only Sorcerers can buff
                return False

            buff_fn = getattr(self.game_state, buff_type, None)
            if buff_fn is None:
                logger.warning("Unknown buff type: %s", buff_type)
                return False

            return buff_fn(unit, target)

        except Exception as e:
            logger.debug("Failed to apply %s: %s", buff_type, e)
            return False

    def _is_end_turn_action(self, action) -> bool:
        """Check if the action is an end turn action."""
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) > 0:
            return action[0] == 5  # End turn action type
        return False
