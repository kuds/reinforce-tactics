"""
Model-based bot that uses trained Stable-Baselines3 models.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from reinforcetactics.constants import ALL_UNIT_TYPES, UNIT_DATA
from reinforcetactics.game.bot_base import BaseBot

# Configure logging
logger = logging.getLogger(__name__)


class ModelBot(BaseBot):  # pylint: disable=too-few-public-methods
    """Bot that uses a trained Stable-Baselines3 model for decision-making."""

    ALL_UNIT_TYPES = ALL_UNIT_TYPES

    # Number of action types in the environment (0-9)
    NUM_ACTION_TYPES = 10

    def __init__(self, game_state, player: int = 2, model_path: Optional[str] = None):
        """
        Initialize the model bot.

        Args:
            game_state: GameState instance
            player: Player number for this bot (default 2)
            model_path: Path to the trained model. ``.zip`` -> SB3 (PPO,
                MaskablePPO, A2C, DQN). ``.pt`` -> ``FeudalRLAgent`` checkpoint.
        """
        self.game_state = game_state
        self.bot_player = player
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.env: Optional[Any] = None
        # Set when loading a .pt feudal checkpoint. Take-turn flow branches on this.
        # Typed as Any rather than ``Optional["FeudalRLAgent"]`` to avoid pulling
        # the heavy reinforcetactics.rl.feudal_rl import at module load time.
        self.feudal_agent: Optional[Any] = None

        if model_path:
            self._load_model(model_path)

    @property
    def is_feudal(self) -> bool:
        """True if this bot was loaded from a feudal ``.pt`` checkpoint."""
        return self.feudal_agent is not None

    def _load_model(self, model_path: str) -> None:
        """Load either an SB3 ``.zip`` or a feudal ``.pt`` checkpoint.

        File extension picks the loader: ``.pt`` goes through
        :meth:`_load_feudal`, anything else falls back to the SB3 path.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if path.suffix == ".pt":
            self._load_feudal(path)
            return

        try:
            # Import here to avoid dependency issues if SB3 not installed
            from stable_baselines3 import A2C, DQN, PPO

            from reinforcetactics.rl.gym_env import StrategyGameEnv

            # Also try MaskablePPO from sb3-contrib
            algorithm_classes: list[Any] = [PPO, A2C, DQN]
            try:
                from sb3_contrib import MaskablePPO

                algorithm_classes.insert(0, MaskablePPO)
            except ImportError:
                pass

            # Try to load with different algorithms
            for algorithm_class in algorithm_classes:
                try:
                    self.model = algorithm_class.load(str(path))
                    logger.info("Successfully loaded model as %s: %s", algorithm_class.__name__, model_path)
                    break
                except Exception as e:
                    logger.debug("Failed to load as %s: %s", algorithm_class.__name__, e)
                    continue

            if self.model is None:
                raise ValueError(f"Could not load model with any supported algorithm: {model_path}")

            # Create a dummy environment for observation space info
            # We'll use the actual game state for observations
            self.env = StrategyGameEnv(map_file=None, opponent=None, render_mode=None)

        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for ModelBot. Install it with: pip install stable-baselines3"
            ) from e
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def _load_feudal(self, path: Path) -> None:
        """Load a ``FeudalRLAgent`` ``.pt`` checkpoint and stash it for inference.

        We construct the agent from the checkpoint's ``hyperparams`` blob so the
        worker head, grid dims, and manager horizon match training. The
        observation space is built directly from constants + the saved grid
        dims (no dummy env round-trip) — the live ``self.game_state`` is what
        observations and masks are built from at turn time.
        """
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from gymnasium import spaces

            from reinforcetactics.rl.feudal_rl import FeudalRLAgent
            from reinforcetactics.rl.observation import (
                GLOBAL_FEATURES_DIM,
                GRID_CHANNELS,
                UNIT_CHANNELS,
            )
        except ImportError as e:
            raise ImportError("torch and reinforcetactics.rl.feudal_rl are required to load .pt checkpoints") from e

        # Probe the checkpoint to recover the runtime config without running the
        # network. The hyperparams dict was added in the recent feudal review;
        # older checkpoints fall back to defaults that match the legacy agent.
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        hp = ckpt.get("hyperparams") or {}

        # The feature extractor's linear projection bakes in grid_width *
        # grid_height, so the agent must be constructed with the saved dims.
        gw = int(hp.get("grid_width") or self.game_state.grid.width)
        gh = int(hp.get("grid_height") or self.game_state.grid.height)
        # If the saved dims don't match the live game's, fail loudly rather
        # than risk a silent shape mismatch deeper in the encoder.
        live_w = self.game_state.grid.width
        live_h = self.game_state.grid.height
        if (gw, gh) != (live_w, live_h):
            raise ValueError(
                f"Feudal checkpoint trained on {gw}x{gh} grid; this game is {live_w}x{live_h}. "
                f"Pick a checkpoint whose grid matches the map."
            )

        # Build the observation space directly. The CNN trunk is shape-driven
        # by the spatial dims; the linear projection by their product times
        # the channel counts; both must match the saved state_dict exactly.
        obs_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(gh, gw, GRID_CHANNELS), dtype=np.float32),
                "units": spaces.Box(low=0.0, high=1.0, shape=(gh, gw, UNIT_CHANNELS), dtype=np.float32),
                # ``global_features`` are tanh-squashed in ``build_observation``
                # (see ``reinforcetactics.rl.observation``), so all five
                # entries land in [0, 1). Matches the env-side bounds in
                # ``StrategyGameEnv``.
                "global_features": spaces.Box(low=0.0, high=1.0, shape=(GLOBAL_FEATURES_DIM,), dtype=np.float32),
            }
        )
        autoregressive = bool(hp.get("autoregressive_worker", ckpt.get("autoregressive_worker", False)))
        self.feudal_agent = FeudalRLAgent(
            observation_space=obs_space,
            grid_width=gw,
            grid_height=gh,
            agent_player=self.bot_player,
            device="cpu",
            autoregressive_worker=autoregressive,
        )
        self.feudal_agent.load_checkpoint(str(path))
        # Inference mode — no dropout / batchnorm twiddling.
        self.feudal_agent.feature_extractor.eval()
        self.feudal_agent.manager.eval()
        self.feudal_agent.worker.eval()
        # Reset goal so we don't carry over any stale goal_step_counter from
        # training (load_checkpoint doesn't touch the in-memory goal).
        self.feudal_agent.reset_goal()
        logger.info("Loaded feudal agent from %s (AR=%s, grid=%dx%d)", path, autoregressive, gw, gh)

    def take_turn(self) -> None:
        """Execute the bot's turn using the trained model.

        Feudal agents share the SB3 take-turn loop (predict → execute →
        check for end_turn) but route action selection through
        ``FeudalRLAgent.select_action`` with stage-conditional or per-dim
        masks built from the live ``self.game_state`` rather than an env's.
        """
        if self.model is None and self.feudal_agent is None:
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

                if self.feudal_agent is not None:
                    action = self._predict_feudal(obs)
                else:
                    # Guarded above: take_turn returns early when both
                    # ``self.model`` and ``self.feudal_agent`` are None.
                    assert self.model is not None
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

    def _predict_feudal(self, obs):
        """Sample an action from the loaded ``FeudalRLAgent``.

        Builds either stage-conditional (``StructuredActionMasks``) or
        per-dimension masks against the live ``self.game_state`` — same
        canonical layout the env uses, just bypassing the env wrapper so
        the bot can play in the GUI / tournament without one.
        """
        from reinforcetactics.rl.gym_env import (  # pylint: disable=import-outside-toplevel
            build_per_dim_masks,
            build_structured_masks,
        )

        gw = self.game_state.grid.width
        gh = self.game_state.grid.height
        if self.feudal_agent.autoregressive_worker:
            structured = build_structured_masks(self.game_state, gw, gh)
            action, _goal = self.feudal_agent.select_action(obs, deterministic=True, structured_masks=structured)
        else:
            _, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask = build_per_dim_masks(self.game_state, gw, gh)
            action, _goal = self.feudal_agent.select_action(
                obs,
                deterministic=True,
                action_masks=(at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask),
            )
        return action

    def _get_observation(self) -> Any:
        """
        Get observation from current game state in the format expected by the model.

        The observation is built from ``self.bot_player``'s perspective so
        global_features start with this bot's own gold / unit count, matching
        the agent-relative contract used by the training environment.

        Returns:
            Observation dict compatible with StrategyGameEnv
        """
        # Lazy import: reinforcetactics.game is imported before reinforcetactics.rl
        # during package init, so a top-level import would be circular.
        from reinforcetactics.rl.observation import build_observation

        # ``action_mask`` is no longer part of the PPO observation space;
        # the policy receives the mask via ``predict(action_masks=...)``.
        return build_observation(
            self.game_state,
            perspective_player=self.bot_player,
            action_mask=None,
        )

    def _compute_action_mask(self) -> np.ndarray:
        """Compute action mask from legal actions.

        The mask uses the same flat layout as gym_env: action_type * W * H + y * W + x.
        Each action type maps to a target position on the grid.
        """
        w = self.game_state.grid.width
        h = self.game_state.grid.height
        area = w * h
        mask_size = self.NUM_ACTION_TYPES * area
        mask = np.zeros(mask_size, dtype=np.float32)

        try:
            legal_actions = self.game_state.get_legal_actions(self.bot_player)

            # End turn always valid at canonical position (0,0)
            mask[5 * area] = 1.0

            # Create unit actions (action_type=0): target is building position
            for action in legal_actions.get("create_unit", []):
                idx = 0 * area + action["y"] * w + action["x"]
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Move actions (action_type=1): target is destination
            for action in legal_actions.get("move", []):
                idx = 1 * area + action["to_y"] * w + action["to_x"]
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Attack actions (action_type=2): target is enemy unit position
            for action in legal_actions.get("attack", []):
                target = action["target"]
                idx = 2 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Seize actions (action_type=3): target is tile position
            for action in legal_actions.get("seize", []):
                tile = action["tile"]
                idx = 3 * area + tile.y * w + tile.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Heal actions (action_type=4): target is ally position
            for action in legal_actions.get("heal", []):
                target = action["target"]
                idx = 4 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Cure actions (action_type=4): same slot as heal
            for action in legal_actions.get("cure", []):
                target = action["target"]
                idx = 4 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Paralyze (action_type=6): target is enemy position
            for action in legal_actions.get("paralyze", []):
                target = action["target"]
                idx = 6 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Haste (action_type=7): target is ally position
            for action in legal_actions.get("haste", []):
                target = action["target"]
                idx = 7 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Defence buff (action_type=8): target is ally position
            for action in legal_actions.get("defence_buff", []):
                target = action["target"]
                idx = 8 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

            # Attack buff (action_type=9): target is ally position
            for action in legal_actions.get("attack_buff", []):
                target = action["target"]
                idx = 9 * area + target.y * w + target.x
                if 0 <= idx < mask_size:
                    mask[idx] = 1.0

        except Exception as e:
            logger.warning("Failed to compute action mask, using all-ones: %s", e)
            mask[:] = 1.0

        # Ensure at least end_turn is valid
        if mask.sum() == 0:
            mask[5 * area] = 1.0

        return mask

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
                return self._buff(from_x, from_y, to_x, to_y, "haste")
            if action_type == 8:  # Defence Buff (Sorcerer)
                return self._buff(from_x, from_y, to_x, to_y, "defence_buff")
            if action_type == 9:  # Attack Buff (Sorcerer)
                return self._buff(from_x, from_y, to_x, to_y, "attack_buff")

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
            if hasattr(self.game_state, "is_unit_type_enabled"):
                if not self.game_state.is_unit_type_enabled(unit_code):
                    return False

            # Check if we have enough gold
            cost = UNIT_DATA[unit_code]["cost"]
            if self.game_state.player_gold.get(self.bot_player, 0) < cost:
                return False

            # Check if location is valid for creation
            if not (0 <= x < self.game_state.grid.width and 0 <= y < self.game_state.grid.height):
                return False

            tile = self.game_state.grid.get_tile(x, y)
            if tile.player != self.bot_player or tile.type != "b":
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

            if healer.type != "C":  # Only clerics can heal/cure
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

            if unit.type != "M":  # Only Mages can paralyze
                return False

            return self.game_state.paralyze(unit, target)

        except Exception as e:
            logger.debug("Failed to paralyze: %s", e)
            return False

    def _buff(self, from_x: int, from_y: int, to_x: int, to_y: int, buff_type: str) -> bool:
        """Apply a Sorcerer buff (haste, defence_buff, or attack_buff) to a friendly unit."""
        try:
            unit = self.game_state.get_unit_at_position(from_x, from_y)
            target = self.game_state.get_unit_at_position(to_x, to_y)

            if not unit or not target:
                return False

            if unit.player != self.bot_player or target.player != self.bot_player:
                return False

            if unit.type != "S":  # Only Sorcerers can buff
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
