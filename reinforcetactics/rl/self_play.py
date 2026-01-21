"""
Self-play utilities for training RL agents against themselves.

Self-play is a powerful technique where an agent learns by playing against
copies of itself, enabling the agent to improve through adversarial training
without requiring hand-crafted opponents.

Features:
- SelfPlayEnv: Environment wrapper for self-play training
- OpponentPool: Manages historical model checkpoints for diverse opponents
- SelfPlayCallback: Stable-Baselines3 callback for opponent updates

Usage:
    from reinforcetactics.rl.self_play import (
        SelfPlayEnv,
        make_self_play_env,
        make_self_play_vec_env,
        SelfPlayCallback
    )

    # Create self-play environment
    env = make_self_play_env()

    # Train with self-play
    model = MaskablePPO("MultiInputPolicy", env)
    callback = SelfPlayCallback(env, update_freq=10000)
    model.learn(total_timesteps=1000000, callback=callback)
"""

import copy
import logging
import os
import random
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reinforcetactics.core.game_state import GameState
from reinforcetactics.rl.gym_env import StrategyGameEnv
from reinforcetactics.utils.file_io import FileIO

logger = logging.getLogger(__name__)


class OpponentPool:
    """
    Manages a pool of opponent models for diverse self-play training.

    This implements "Fictitious Self-Play" where the agent trains against
    a mixture of historical versions of itself, preventing overfitting
    to a single opponent strategy.

    Attributes:
        max_size: Maximum number of models to keep in the pool
        models: Deque of (model, metadata) tuples
        selection_strategy: How to select opponents ('uniform', 'recent', 'prioritized')
    """

    def __init__(
        self,
        max_size: int = 10,
        selection_strategy: str = 'uniform',
        save_dir: Optional[str] = None
    ):
        """
        Initialize the opponent pool.

        Args:
            max_size: Maximum number of models to keep
            selection_strategy: 'uniform' (equal probability), 'recent' (favor recent),
                              'prioritized' (favor strong opponents)
            save_dir: Directory to save/load pool checkpoints
        """
        self.max_size = max_size
        self.selection_strategy = selection_strategy
        self.save_dir = Path(save_dir) if save_dir else None
        self.models: deque = deque(maxlen=max_size)
        self.metadata: deque = deque(maxlen=max_size)
        self._selection_weights: List[float] = []

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_model(
        self,
        model: Any,
        timestep: int = 0,
        win_rate: float = 0.5,
        save_to_disk: bool = True
    ) -> None:
        """
        Add a model to the pool.

        Args:
            model: The trained model to add
            timestep: Training timestep when model was saved
            win_rate: Model's win rate (for prioritized selection)
            save_to_disk: Whether to save to disk
        """
        # Deep copy the model's policy parameters
        model_copy = self._copy_model_params(model)

        metadata = {
            'timestep': timestep,
            'win_rate': win_rate,
            'index': len(self.models)
        }

        self.models.append(model_copy)
        self.metadata.append(metadata)
        self._update_selection_weights()

        if save_to_disk and self.save_dir:
            save_path = self.save_dir / f"opponent_{timestep}.zip"
            try:
                model.save(str(save_path))
                logger.info(f"Saved opponent to pool: {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save opponent to disk: {e}")

    def _copy_model_params(self, model: Any) -> Dict[str, np.ndarray]:
        """Create a lightweight copy of model parameters."""
        try:
            # For SB3 models, get policy parameters
            params = {}
            for name, param in model.policy.state_dict().items():
                params[name] = param.cpu().numpy().copy()
            return params
        except Exception as e:
            logger.warning(f"Could not copy model params: {e}")
            return {}

    def _load_model_params(self, model: Any, params: Dict[str, np.ndarray]) -> None:
        """Load parameters into a model's policy."""
        try:
            import torch
            state_dict = {
                name: torch.tensor(param)
                for name, param in params.items()
            }
            model.policy.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Could not load model params: {e}")

    def _update_selection_weights(self) -> None:
        """Update selection weights based on strategy."""
        n = len(self.models)
        if n == 0:
            self._selection_weights = []
            return

        if self.selection_strategy == 'uniform':
            self._selection_weights = [1.0 / n] * n

        elif self.selection_strategy == 'recent':
            # Exponentially favor more recent models
            weights = [2.0 ** i for i in range(n)]
            total = sum(weights)
            self._selection_weights = [w / total for w in weights]

        elif self.selection_strategy == 'prioritized':
            # Favor models with higher win rates
            win_rates = [m.get('win_rate', 0.5) for m in self.metadata]
            # Add small epsilon to avoid zero weights
            weights = [max(0.1, wr) for wr in win_rates]
            total = sum(weights)
            self._selection_weights = [w / total for w in weights]

    def sample_opponent(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample an opponent from the pool.

        Returns:
            Model parameters dict, or None if pool is empty
        """
        if not self.models:
            return None

        idx = random.choices(
            range(len(self.models)),
            weights=self._selection_weights,
            k=1
        )[0]

        return self.models[idx]

    def sample_opponent_with_metadata(self) -> Optional[Tuple[Dict, Dict]]:
        """Sample an opponent and return with metadata."""
        if not self.models:
            return None

        idx = random.choices(
            range(len(self.models)),
            weights=self._selection_weights,
            k=1
        )[0]

        return self.models[idx], self.metadata[idx]

    def update_win_rate(self, model_idx: int, new_win_rate: float) -> None:
        """Update a model's win rate in the pool."""
        if 0 <= model_idx < len(self.metadata):
            self.metadata[model_idx]['win_rate'] = new_win_rate
            self._update_selection_weights()

    def load_from_disk(self, model_class: Any) -> int:
        """
        Load all saved opponents from disk.

        Args:
            model_class: SB3 model class (e.g., PPO, MaskablePPO)

        Returns:
            Number of models loaded
        """
        if not self.save_dir or not self.save_dir.exists():
            return 0

        loaded = 0
        for path in sorted(self.save_dir.glob("opponent_*.zip")):
            try:
                model = model_class.load(str(path))
                params = self._copy_model_params(model)
                timestep = int(path.stem.split('_')[1])
                self.models.append(params)
                self.metadata.append({
                    'timestep': timestep,
                    'win_rate': 0.5,
                    'index': len(self.models) - 1
                })
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load opponent {path}: {e}")

        self._update_selection_weights()
        logger.info(f"Loaded {loaded} opponents from {self.save_dir}")
        return loaded

    @property
    def size(self) -> int:
        """Return number of models in pool."""
        return len(self.models)

    def __len__(self) -> int:
        return self.size


class SelfPlayEnv(gym.Wrapper):
    """
    Gymnasium wrapper that enables self-play training.

    The agent controls player 1, and the opponent (controlled by a copy
    of the agent's policy) controls player 2. The environment handles
    turn alternation and opponent action execution automatically.

    Features:
    - Symmetric gameplay (same observations/actions for both players)
    - Configurable opponent update frequency
    - Support for opponent pool (multiple historical models)
    - Optional random starting player to encourage robust play

    Attributes:
        opponent_model: The model used for opponent decisions
        opponent_pool: Pool of historical opponents (optional)
        swap_players: Whether to randomly swap player order
    """

    def __init__(
        self,
        env: StrategyGameEnv,
        opponent_model: Optional[Any] = None,
        opponent_pool: Optional[OpponentPool] = None,
        swap_players: bool = True,
        opponent_deterministic: bool = False
    ):
        """
        Initialize the self-play environment.

        Args:
            env: The base StrategyGameEnv
            opponent_model: Initial opponent model (can be updated later)
            opponent_pool: Pool of historical opponents for diverse training
            swap_players: Randomly swap which player agent controls each episode
            opponent_deterministic: Use deterministic opponent actions
        """
        super().__init__(env)
        self.opponent_model = opponent_model
        self.opponent_pool = opponent_pool
        self.swap_players = swap_players
        self.opponent_deterministic = opponent_deterministic

        # Track which player the learning agent controls
        self.agent_player = 1
        self._opponent_params: Optional[Dict] = None

        # Statistics
        self.stats = {
            'agent_wins': 0,
            'opponent_wins': 0,
            'draws': 0,
            'total_games': 0
        }

    def set_opponent_model(self, model: Any) -> None:
        """Set or update the opponent model."""
        self.opponent_model = model

    def update_opponent_from_pool(self) -> bool:
        """
        Sample a new opponent from the pool.

        Returns:
            True if opponent was updated, False if pool is empty
        """
        if self.opponent_pool is None or self.opponent_pool.size == 0:
            return False

        self._opponent_params = self.opponent_pool.sample_opponent()
        return self._opponent_params is not None

    def update_opponent_from_current(self) -> None:
        """Update opponent to use current model's parameters."""
        if self.opponent_model is not None:
            try:
                self._opponent_params = {}
                for name, param in self.opponent_model.policy.state_dict().items():
                    self._opponent_params[name] = param.cpu().numpy().copy()
            except Exception as e:
                logger.warning(f"Could not copy current model params: {e}")

    def _get_opponent_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get opponent's action using the opponent model.

        Args:
            obs: Observation from opponent's perspective

        Returns:
            Action array
        """
        if self.opponent_model is None:
            # Fallback to random valid action
            return self._get_random_valid_action()

        try:
            # If we have stored params, temporarily load them
            original_params = None
            if self._opponent_params:
                import torch
                original_params = {
                    name: param.cpu().numpy().copy()
                    for name, param in self.opponent_model.policy.state_dict().items()
                }
                # Load opponent params
                state_dict = {
                    name: torch.tensor(param)
                    for name, param in self._opponent_params.items()
                }
                self.opponent_model.policy.load_state_dict(state_dict)

            # Get action from model
            action, _ = self.opponent_model.predict(
                obs,
                deterministic=self.opponent_deterministic
            )

            # Restore original params if we swapped
            if original_params:
                import torch
                state_dict = {
                    name: torch.tensor(param)
                    for name, param in original_params.items()
                }
                self.opponent_model.policy.load_state_dict(state_dict)

            return action

        except Exception as e:
            logger.warning(f"Error getting opponent action: {e}")
            return self._get_random_valid_action()

    def _get_random_valid_action(self) -> np.ndarray:
        """Get a random valid action (fallback)."""
        # Get action masks
        masks = self.env.action_masks()

        # Sample valid action for each dimension
        action = []
        for mask in masks:
            valid_indices = np.where(mask)[0]
            if len(valid_indices) > 0:
                action.append(np.random.choice(valid_indices))
            else:
                action.append(0)

        return np.array(action)

    def _flip_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Flip observation to opponent's perspective.

        For symmetric self-play, the opponent sees the game from their
        perspective (as player 1).
        """
        flipped = {}

        # Grid: swap player ownership (channel 1)
        if 'grid' in obs:
            grid = obs['grid'].copy()
            # Ownership channel: 1 -> 2, 2 -> 1
            ownership = grid[:, :, 1]
            new_ownership = np.where(ownership == 1, 2,
                                     np.where(ownership == 2, 1, ownership))
            grid[:, :, 1] = new_ownership
            flipped['grid'] = grid

        # Units: swap ownership (channel 1)
        if 'units' in obs:
            units = obs['units'].copy()
            ownership = units[:, :, 1]
            new_ownership = np.where(ownership == 1, 2,
                                     np.where(ownership == 2, 1, ownership))
            units[:, :, 1] = new_ownership
            flipped['units'] = units

        # Global features: swap player-specific features
        if 'global_features' in obs:
            gf = obs['global_features'].copy()
            # [gold_p1, gold_p2, turn, units_p1, units_p2, current_player]
            # Swap gold
            gf[0], gf[1] = gf[1], gf[0]
            # Swap unit counts
            gf[3], gf[4] = gf[4], gf[3]
            # Flip current player
            gf[5] = 3 - gf[5]  # 1 -> 2, 2 -> 1
            flipped['global_features'] = gf

        # Action mask: recalculate for opponent's perspective
        if 'action_mask' in obs:
            flipped['action_mask'] = obs['action_mask'].copy()

        return flipped

    def _execute_opponent_turn(self) -> None:
        """Execute the opponent's turn."""
        game_state = self.env.game_state

        # Safety limit to prevent infinite loops
        max_actions = 50
        actions_taken = 0

        while (game_state.current_player == 2 and
               not game_state.game_over and
               actions_taken < max_actions):

            # Get observation from opponent's perspective
            obs = self._get_obs_for_player(2)

            # Get opponent's action
            action = self._get_opponent_action(obs)

            # Execute action
            action_executed = self._execute_opponent_action(action)

            # If action was end_turn or failed, break
            if action[0] == 5 or not action_executed:
                game_state.end_turn()
                break

            actions_taken += 1

        # Ensure turn ends if still opponent's turn
        if game_state.current_player == 2 and not game_state.game_over:
            game_state.end_turn()

    def _get_obs_for_player(self, player: int) -> Dict[str, np.ndarray]:
        """Get observation from a specific player's perspective."""
        obs = self.env._get_obs()
        if player == 2:
            obs = self._flip_observation(obs)
        return obs

    def _execute_opponent_action(self, action: np.ndarray) -> bool:
        """
        Execute opponent's action in the game.

        Returns:
            True if action was valid, False otherwise
        """
        game_state = self.env.game_state

        action_type = int(action[0])
        unit_type_idx = int(action[1])
        from_x, from_y = int(action[2]), int(action[3])
        to_x, to_y = int(action[4]), int(action[5])

        unit_types = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']
        unit_type = unit_types[unit_type_idx % 8]

        try:
            if action_type == 0:  # Create unit
                unit = game_state.create_unit(unit_type, to_x, to_y, player=2)
                return unit is not None

            elif action_type == 1:  # Move
                unit = game_state.get_unit_at_position(from_x, from_y)
                if unit and unit.player == 2 and unit.can_move:
                    return game_state.move_unit(unit, to_x, to_y)
                return False

            elif action_type == 2:  # Attack
                unit = game_state.get_unit_at_position(from_x, from_y)
                target = game_state.get_unit_at_position(to_x, to_y)
                if unit and target and unit.player == 2 and target.player != 2:
                    game_state.attack(unit, target)
                    return True
                return False

            elif action_type == 3:  # Seize
                unit = game_state.get_unit_at_position(from_x, from_y)
                if unit and unit.player == 2:
                    result = game_state.seize(unit)
                    return True
                return False

            elif action_type == 4:  # Heal/Cure
                unit = game_state.get_unit_at_position(from_x, from_y)
                target = game_state.get_unit_at_position(to_x, to_y)
                if unit and target and unit.type == 'C' and unit.player == 2:
                    if target.is_paralyzed():
                        return game_state.cure(unit, target)
                    return game_state.heal(unit, target) > 0
                return False

            elif action_type == 5:  # End turn
                return True  # Signal to end turn

            elif action_type == 6:  # Paralyze
                unit = game_state.get_unit_at_position(from_x, from_y)
                target = game_state.get_unit_at_position(to_x, to_y)
                if unit and target and unit.type in ['M', 'S'] and unit.player == 2:
                    return game_state.paralyze(unit, target)
                return False

            elif action_type == 7:  # Haste
                unit = game_state.get_unit_at_position(from_x, from_y)
                target = game_state.get_unit_at_position(to_x, to_y)
                if unit and target and unit.type == 'S' and unit.player == 2:
                    return game_state.haste(unit, target)
                return False

            elif action_type == 8:  # Defence buff
                unit = game_state.get_unit_at_position(from_x, from_y)
                target = game_state.get_unit_at_position(to_x, to_y)
                if unit and target and unit.type == 'S' and unit.player == 2:
                    return game_state.defence_buff(unit, target)
                return False

            elif action_type == 9:  # Attack buff
                unit = game_state.get_unit_at_position(from_x, from_y)
                target = game_state.get_unit_at_position(to_x, to_y)
                if unit and target and unit.type == 'S' and unit.player == 2:
                    return game_state.attack_buff(unit, target)
                return False

        except Exception as e:
            logger.debug(f"Opponent action failed: {e}")
            return False

        return False

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute agent's action and then opponent's turn.

        Args:
            action: Agent's action

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Execute agent's action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If game ended or action was end_turn, let opponent play
        if not terminated and action[0] == 5:
            # Agent ended turn, opponent plays
            self._execute_opponent_turn()

            # Get new observation after opponent's turn
            obs = self.env._get_obs()
            terminated = self.env.game_state.game_over

            # Adjust reward for game end
            if terminated:
                winner = self.env.game_state.winner
                if winner == 1:
                    reward += self.env.reward_config['win']
                    self.stats['agent_wins'] += 1
                elif winner == 2:
                    reward += self.env.reward_config['loss']
                    self.stats['opponent_wins'] += 1
                else:
                    self.stats['draws'] += 1
                self.stats['total_games'] += 1

        info['self_play_stats'] = self.stats.copy()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset the environment.

        Optionally swaps player order and updates opponent from pool.
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Potentially update opponent from pool
        if self.opponent_pool and self.opponent_pool.size > 0:
            self.update_opponent_from_pool()

        # Optionally swap player order
        if self.swap_players and random.random() < 0.5:
            # Agent plays as player 2 this game
            # For simplicity, we just let opponent go first
            self._execute_opponent_turn()
            obs = self.env._get_obs()

        return obs, info

    def action_masks(self) -> Tuple[np.ndarray, ...]:
        """Get action masks for agent."""
        return self.env.action_masks()

    def get_win_rate(self) -> float:
        """Get agent's win rate against opponents."""
        total = self.stats['total_games']
        if total == 0:
            return 0.5
        return self.stats['agent_wins'] / total


class SelfPlayCallback:
    """
    Callback for Stable-Baselines3 that manages self-play opponent updates.

    This callback:
    1. Periodically updates the opponent model to the current policy
    2. Optionally adds models to the opponent pool
    3. Tracks win rates and training progress

    Usage:
        callback = SelfPlayCallback(
            env,
            update_freq=10000,
            add_to_pool_freq=50000
        )
        model.learn(total_timesteps=1000000, callback=callback)
    """

    def __init__(
        self,
        env: Union[SelfPlayEnv, Any],
        update_freq: int = 10000,
        add_to_pool_freq: int = 50000,
        min_win_rate_for_pool: float = 0.55,
        verbose: int = 1
    ):
        """
        Initialize the callback.

        Args:
            env: The SelfPlayEnv or vectorized environment
            update_freq: How often to update opponent to current model
            add_to_pool_freq: How often to add model to opponent pool
            min_win_rate_for_pool: Minimum win rate to add to pool
            verbose: Verbosity level
        """
        self.env = env
        self.update_freq = update_freq
        self.add_to_pool_freq = add_to_pool_freq
        self.min_win_rate_for_pool = min_win_rate_for_pool
        self.verbose = verbose

        self.n_calls = 0
        self.model = None

    def _get_self_play_envs(self) -> List[SelfPlayEnv]:
        """Get all SelfPlayEnv instances from the environment."""
        envs = []

        # Handle vectorized environments
        if hasattr(self.env, 'envs'):
            for env in self.env.envs:
                if isinstance(env, SelfPlayEnv):
                    envs.append(env)
                elif hasattr(env, 'env') and isinstance(env.env, SelfPlayEnv):
                    envs.append(env.env)
        elif isinstance(self.env, SelfPlayEnv):
            envs.append(self.env)
        elif hasattr(self.env, 'env') and isinstance(self.env.env, SelfPlayEnv):
            envs.append(self.env.env)

        return envs

    def _init_callback(self, model: Any) -> bool:
        """Initialize callback with model reference."""
        self.model = model
        return True

    def _on_step(self) -> bool:
        """Called after each step."""
        self.n_calls += 1

        # Update opponent model
        if self.n_calls % self.update_freq == 0:
            self._update_opponents()

        # Add to pool
        if self.n_calls % self.add_to_pool_freq == 0:
            self._add_to_pool()

        return True

    def _update_opponents(self) -> None:
        """Update all opponents to current model."""
        for env in self._get_self_play_envs():
            env.set_opponent_model(self.model)
            env.update_opponent_from_current()

        if self.verbose >= 1:
            win_rates = [env.get_win_rate() for env in self._get_self_play_envs()]
            avg_win_rate = np.mean(win_rates) if win_rates else 0.5
            logger.info(
                f"Step {self.n_calls}: Updated opponents. "
                f"Avg win rate: {avg_win_rate:.2%}"
            )

    def _add_to_pool(self) -> None:
        """Add current model to opponent pool if win rate is good enough."""
        envs = self._get_self_play_envs()
        if not envs:
            return

        # Check win rate
        win_rates = [env.get_win_rate() for env in envs]
        avg_win_rate = np.mean(win_rates) if win_rates else 0.5

        if avg_win_rate >= self.min_win_rate_for_pool:
            for env in envs:
                if env.opponent_pool is not None:
                    env.opponent_pool.add_model(
                        self.model,
                        timestep=self.n_calls,
                        win_rate=avg_win_rate
                    )
                    if self.verbose >= 1:
                        logger.info(
                            f"Step {self.n_calls}: Added model to pool "
                            f"(win rate: {avg_win_rate:.2%}, "
                            f"pool size: {env.opponent_pool.size})"
                        )
                    break  # Only add once


def make_self_play_env(
    map_file: Optional[str] = None,
    max_steps: int = 500,
    reward_config: Optional[Dict[str, float]] = None,
    opponent_pool: Optional[OpponentPool] = None,
    swap_players: bool = True,
    enabled_units: Optional[List[str]] = None
) -> SelfPlayEnv:
    """
    Create a single self-play environment.

    Args:
        map_file: Path to map CSV file. None for random map.
        max_steps: Maximum steps per episode
        reward_config: Custom reward configuration
        opponent_pool: Pool of historical opponents
        swap_players: Whether to randomly swap player order
        enabled_units: List of enabled unit types

    Returns:
        SelfPlayEnv ready for training

    Example:
        env = make_self_play_env()
        model = MaskablePPO("MultiInputPolicy", env)

        # Set opponent after model creation
        env.set_opponent_model(model)

        callback = SelfPlayCallback(env, update_freq=10000)
        model.learn(total_timesteps=1000000, callback=callback)
    """
    from reinforcetactics.rl.masking import ActionMaskedEnv

    base_env = StrategyGameEnv(
        map_file=map_file,
        opponent=None,  # No built-in opponent for self-play
        render_mode=None,
        max_steps=max_steps,
        reward_config=reward_config,
        enabled_units=enabled_units
    )

    # Wrap with action masking first
    masked_env = ActionMaskedEnv(base_env)

    # Then wrap with self-play
    self_play_env = SelfPlayEnv(
        masked_env,
        opponent_pool=opponent_pool,
        swap_players=swap_players
    )

    return self_play_env


def _make_self_play_env_fn(
    rank: int,
    seed: int,
    map_file: Optional[str],
    max_steps: int,
    reward_config: Optional[Dict[str, float]],
    opponent_pool: Optional[OpponentPool],
    swap_players: bool,
    enabled_units: Optional[List[str]]
) -> Callable[[], SelfPlayEnv]:
    """Create a function that creates a self-play environment."""
    from reinforcetactics.rl.masking import ActionMaskedEnv

    def _init() -> SelfPlayEnv:
        base_env = StrategyGameEnv(
            map_file=map_file,
            opponent=None,
            render_mode=None,
            max_steps=max_steps,
            reward_config=reward_config,
            enabled_units=enabled_units
        )
        base_env.reset(seed=seed + rank)

        masked_env = ActionMaskedEnv(base_env)
        self_play_env = SelfPlayEnv(
            masked_env,
            opponent_pool=opponent_pool,
            swap_players=swap_players
        )
        return self_play_env

    return _init


def make_self_play_vec_env(
    n_envs: int = 4,
    map_file: Optional[str] = None,
    max_steps: int = 500,
    reward_config: Optional[Dict[str, float]] = None,
    seed: int = 0,
    use_subprocess: bool = True,
    opponent_pool: Optional[OpponentPool] = None,
    swap_players: bool = True,
    enabled_units: Optional[List[str]] = None
):
    """
    Create vectorized self-play environments for parallel training.

    Args:
        n_envs: Number of parallel environments
        map_file: Path to map CSV file. None for random maps.
        max_steps: Maximum steps per episode
        reward_config: Custom reward configuration
        seed: Random seed (each env gets seed + rank)
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)
        opponent_pool: Shared opponent pool
        swap_players: Whether to randomly swap player order
        enabled_units: List of enabled unit types

    Returns:
        Vectorized environment ready for MaskablePPO

    Example:
        pool = OpponentPool(max_size=10)
        vec_env = make_self_play_vec_env(
            n_envs=8,
            opponent_pool=pool
        )
        model = MaskablePPO("MultiInputPolicy", vec_env)

        callback = SelfPlayCallback(vec_env, update_freq=10000)
        model.learn(total_timesteps=1000000, callback=callback)
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        _make_self_play_env_fn(
            i, seed, map_file, max_steps, reward_config,
            opponent_pool, swap_players, enabled_units
        )
        for i in range(n_envs)
    ]

    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env
