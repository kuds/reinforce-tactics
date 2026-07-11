"""
Model-based bot that uses trained Stable-Baselines3 models.
"""

import logging
from pathlib import Path
from typing import Any

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

    def __init__(self, game_state, player: int = 2, model_path: str | None = None):
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
        self.model: Any | None = None
        # Set when loading a .pt feudal checkpoint. Take-turn flow branches on this.
        # Typed as Any rather than ``Optional["FeudalRLAgent"]`` to avoid pulling
        # the heavy reinforcetactics.rl.feudal_rl import at module load time.
        self.feudal_agent: Any | None = None

        # SB3-checkpoint runtime config, derived from the loaded model's
        # saved observation/action spaces by ``_configure_from_sb3_model``:
        #   _action_mode:          "flat" (Discrete policy; actions decoded
        #                          via the shared build_flat_actions table)
        #                          or "multi_discrete" (6-vector actions).
        #   _max_flat_actions:     Discrete(n) size in flat mode.
        #   _pad_to:               (pad_h, pad_w) when the checkpoint was
        #                          trained on padded observations larger
        #                          than the live map; None = no padding.
        #   _accepts_action_masks: model.predict takes ``action_masks``
        #                          (MaskablePPO); plain PPO/A2C/DQN don't.
        #   _flat_actions:         the decode table built for the most
        #                          recent flat-mode predict call.
        self._action_mode: str = "multi_discrete"
        self._max_flat_actions: int = 0
        self._pad_to: tuple | None = None
        self._accepts_action_masks: bool = False
        self._flat_actions: list = []

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

            # Derive action mode / padding / masking support from the
            # checkpoint's saved spaces, and validate them against the live
            # game so a mismatch fails loud at load time instead of as a
            # silently-passive bot at play time.
            self._configure_from_sb3_model()

        except ImportError as e:
            raise ImportError(
                "stable-baselines3 is required for ModelBot. Install it with: pip install stable-baselines3"
            ) from e
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def _configure_from_sb3_model(self) -> None:
        """Configure inference from the checkpoint's saved spaces.

        SB3 checkpoints carry the training-time ``observation_space`` and
        ``action_space``; together with the live ``self.game_state`` they
        determine everything ModelBot needs to replicate the training env's
        encode/decode contract:

          * ``Discrete(n)`` action space -> ``flat_discrete`` mode: actions
            are indices into the legal-action table from
            :func:`reinforcetactics.rl.gym_env.build_flat_actions`
            (``n`` = the training env's ``max_flat_actions``).
          * ``MultiDiscrete([10, 8, W, H, W, H])`` -> 6-vector mode. ``W``/``H``
            are the *unpadded* training grid dims and must match the live map.
          * Obs ``grid`` shape ``(pad_h, pad_w, C)`` larger than the live map
            -> observations are zero-padded to ``(pad_h, pad_w)`` exactly like
            the curriculum runner's ``pad_to_size`` (smaller than the live
            map fails loud: the policy literally cannot see the whole board).
          * A ``visibility`` key in the obs space means the checkpoint was
            trained under fog of war; the live game must match either way.

        Raises ``ValueError`` with an actionable message on any mismatch.
        """
        from gymnasium import spaces as gym_spaces

        from reinforcetactics.rl.evaluation import _model_accepts_action_masks
        from reinforcetactics.rl.observation import GRID_CHANNELS, UNIT_CHANNELS

        obs_space = getattr(self.model, "observation_space", None)
        act_space = getattr(self.model, "action_space", None)
        if not isinstance(obs_space, gym_spaces.Dict) or "grid" not in obs_space.spaces or "units" not in obs_space.spaces:
            raise ValueError(
                f"Checkpoint observation space {obs_space} is not the reinforcetactics Dict(grid, units, "
                "global_features[, visibility]) contract; this does not look like a StrategyGameEnv checkpoint."
            )

        live_h = self.game_state.grid.height
        live_w = self.game_state.grid.width

        grid_shape = obs_space.spaces["grid"].shape  # (H, W, C)
        units_shape = obs_space.spaces["units"].shape
        # Box.shape is Optional in gymnasium's typing; a Dict-of-Box obs
        # space from a real checkpoint always carries concrete 3-D shapes,
        # so anything else is a malformed/foreign checkpoint.
        if grid_shape is None or len(grid_shape) != 3 or units_shape is None or len(units_shape) != 3:
            raise ValueError(
                f"Checkpoint grid/units spaces have unexpected shapes ({grid_shape}, {units_shape}); "
                "expected (H, W, C) Boxes from StrategyGameEnv."
            )
        if grid_shape[2] != GRID_CHANNELS or units_shape[2] != UNIT_CHANNELS:
            raise ValueError(
                f"Checkpoint obs channels (grid={grid_shape[2]}, units={units_shape[2]}) do not match this "
                f"build's encoding (grid={GRID_CHANNELS}, units={UNIT_CHANNELS}). The checkpoint was trained "
                "on an older observation schema."
            )
        obs_h, obs_w = int(grid_shape[0]), int(grid_shape[1])
        if obs_h < live_h or obs_w < live_w:
            raise ValueError(
                f"Checkpoint was trained on a ({obs_h}, {obs_w}) observation but this map is "
                f"({live_h}, {live_w}); the policy cannot see the whole board. Pick a checkpoint whose "
                "(padded) observation covers the map."
            )
        # Equal dims need no padding; strictly-larger dims mean the
        # checkpoint trained with pad_to_size and the live obs must be
        # padded up to match (build_observation pads with the all-zero
        # signature the policy saw in training).
        self._pad_to = (obs_h, obs_w) if (obs_h, obs_w) != (live_h, live_w) else None

        expects_visibility = "visibility" in obs_space.spaces
        live_fow = bool(getattr(self.game_state, "fog_of_war", False))
        if expects_visibility != live_fow:
            raise ValueError(
                f"Checkpoint fog-of-war mismatch: checkpoint {'expects' if expects_visibility else 'does not expect'} "
                f"a visibility layer but the live game has fog_of_war={live_fow}. Start the game with matching "
                "fog-of-war settings for this checkpoint."
            )

        if isinstance(act_space, gym_spaces.Discrete):
            self._action_mode = "flat"
            self._max_flat_actions = int(act_space.n)
        elif isinstance(act_space, gym_spaces.MultiDiscrete):
            self._action_mode = "multi_discrete"
            nvec = [int(x) for x in act_space.nvec]
            # Layout: [10 action types, 8 unit types, W, H, W, H]. The grid
            # dims are baked into the action space, so a map-size mismatch
            # makes coordinate sub-actions meaningless -- fail loud, exactly
            # like the feudal loader does.
            if len(nvec) == 6 and (nvec[2] != live_w or nvec[3] != live_h):
                raise ValueError(
                    f"MultiDiscrete checkpoint was trained on a {nvec[2]}x{nvec[3]} grid; this game is "
                    f"{live_w}x{live_h}. Pick a checkpoint whose grid matches the map."
                )
        else:
            raise ValueError(
                f"Unsupported action space {act_space} in checkpoint; expected Discrete (flat_discrete) or MultiDiscrete."
            )

        # MaskablePPO.predict accepts ``action_masks``; plain PPO/A2C/DQN
        # raise TypeError on the kwarg, so probe the signature once here.
        self._accepts_action_masks = _model_accepts_action_masks(self.model)
        logger.info(
            "ModelBot configured: mode=%s%s pad_to=%s masked_predict=%s",
            self._action_mode,
            f"(n={self._max_flat_actions})" if self._action_mode == "flat" else "",
            self._pad_to,
            self._accepts_action_masks,
        )

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
                    action = self._predict_sb3(obs)

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

    def _predict_sb3(self, obs) -> np.ndarray:
        """Sample a 6-vector action from the loaded SB3 model.

        Mirrors the training env's encode/decode contract per
        ``_configure_from_sb3_model``:

          * flat mode: build the legal-action table via the shared
            :func:`reinforcetactics.rl.gym_env.build_flat_actions`, predict a
            ``Discrete`` index with the exact "first ``len(table)`` entries
            legal" mask, and decode the index through the table. An
            out-of-range index (only possible without masking, e.g. plain
            DQN) falls back to end_turn -- same as ``StrategyGameEnv.step``.
          * multi_discrete mode: predict the 6-vector directly, forwarding
            the per-dimension masks (concatenated, as MaskablePPO expects)
            when the model supports them.
        """
        # Lazy import: mirrors _get_observation (reinforcetactics.game loads
        # before reinforcetactics.rl during package init).
        from reinforcetactics.rl.gym_env import build_flat_actions, build_per_dim_masks

        # take_turn guards this already; the assert narrows the Optional for
        # mypy and for any direct callers.
        assert self.model is not None

        predict_kwargs: dict = {"deterministic": True}

        if self._action_mode == "flat":
            self._flat_actions = build_flat_actions(self.game_state, self.bot_player, self._max_flat_actions)
            if self._accepts_action_masks:
                mask = np.zeros(self._max_flat_actions, dtype=bool)
                mask[: len(self._flat_actions)] = True
                predict_kwargs["action_masks"] = mask
            raw_action, _states = self.model.predict(obs, **predict_kwargs)
            idx = int(np.asarray(raw_action).reshape(-1)[0])
            if 0 <= idx < len(self._flat_actions):
                return self._flat_actions[idx]
            logger.debug(
                "Flat action index %d out of range (%d legal); falling back to end_turn", idx, len(self._flat_actions)
            )
            return np.array([5, 0, 0, 0, 0, 0], dtype=np.int32)

        if self._accepts_action_masks:
            _, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask = build_per_dim_masks(
                self.game_state,
                self.game_state.grid.width,
                self.game_state.grid.height,
                enabled_units=getattr(self.game_state, "enabled_units", None),
            )
            predict_kwargs["action_masks"] = np.concatenate(
                [m.astype(np.bool_) for m in (at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask)]
            )
        raw_action, _states = self.model.predict(obs, **predict_kwargs)
        return np.asarray(raw_action)

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
        the agent-relative contract used by the training environment. When
        the loaded checkpoint was trained on padded observations
        (``_pad_to`` set from its saved observation space), the live obs is
        zero-padded to the same shape -- without this, padded-curriculum
        checkpoints fail SB3's obs-shape check on every predict.

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
            pad_to=self._pad_to,
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
