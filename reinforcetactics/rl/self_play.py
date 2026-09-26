"""
Self-play utilities for training RL agents against themselves.

Self-play is a powerful technique where an agent learns by playing against
copies of itself, enabling the agent to improve through adversarial training
without requiring hand-crafted opponents.

Features:
- SelfPlayEnv: Environment wrapper for self-play training
- OpponentPool: Manages historical model checkpoints for diverse opponents
- SelfPlayCallback: Stable-Baselines3 callback for opponent updates

How the pieces fit:
- The opponent is a *frozen snapshot* of the learner's policy
  (:func:`policy_snapshot`: policy class, constructor kwargs and CPU
  weights). Snapshots are plain picklable data, so the callback pushes them
  to every worker through ``VecEnv.env_method`` -- this is what makes
  ``SubprocVecEnv`` workers actually play against the learner.
- The opponent plays inside the base env's ``opponent='self'`` slot, so its
  turns get exactly the reward accounting (opponent-turn penalties,
  terminal bonuses, potential-shaping terminal charge) that a scripted bot
  gets. It observes, masks and decodes actions for its *own* seat.
- ``swap_players`` puts the agent in seat 2 for about half the episodes
  (drawn from the env's ``np_random``); the base env then plays player 1's
  opening turn before the agent's first observation.

Usage:
    from reinforcetactics.rl.self_play import (
        SelfPlayCallback,
        make_self_play_vec_env,
    )

    vec_env = make_self_play_vec_env(n_envs=8, map_file="maps/1v1/beginner.csv")
    model = MaskablePPO("MultiInputPolicy", vec_env)
    # The callback pushes the current policy to the opponents at the start
    # of training and every ``update_freq`` calls (vec steps).
    callback = SelfPlayCallback(vec_env, update_freq=1250)
    model.learn(total_timesteps=1000000, callback=callback)
"""

import copy
import logging
import random
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from reinforcetactics.rl.gym_env import StrategyGameEnv, build_flat_actions, build_per_dim_masks
from reinforcetactics.rl.observation import build_observation

logger = logging.getLogger(__name__)

# Conditionally import BaseCallback so the module is importable without SB3 installed.
try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
except ImportError:  # pragma: no cover
    _BaseCallback = None

# The canonical end_turn action in the 6-vector layout shared by both action
# spaces (see ``StrategyGameEnv._encode_action``).
_END_TURN = (5, 0, 0, 0, 0, 0)

# Safety cap on opponent actions per game-turn (same limit ModelBot uses).
_MAX_OPPONENT_ACTIONS = 50
# Consecutive rejected actions after which the opponent's turn is ended.
# Only multi_discrete can produce these: its per-dimension masks
# over-approximate the legal set, while flat_discrete masks are exact.
_MAX_CONSECUTIVE_INVALID = 5


def _state_dict_to_numpy(policy: Any) -> dict[str, np.ndarray]:
    """Detached CPU numpy copy of a torch module's ``state_dict``."""
    return {name: param.detach().cpu().numpy().copy() for name, param in policy.state_dict().items()}


def _load_numpy_state_dict(policy: Any, params: Mapping[str, np.ndarray]) -> None:
    """Load a numpy state dict (as produced by :func:`_state_dict_to_numpy`)."""
    import torch

    policy.load_state_dict({name: torch.as_tensor(value) for name, value in params.items()})


def params_checksum(params: Mapping[str, np.ndarray]) -> float:
    """Cheap fingerprint of a numpy state dict (float64 sum of all entries).

    Lets callers verify *which* weights an opponent holds -- e.g. that an
    update pushed from the trainer reached a ``SubprocVecEnv`` worker --
    without shipping the weights back across the process boundary.
    """
    return float(sum(np.sum(value, dtype=np.float64) for value in params.values()))


def policy_snapshot(model: Any) -> dict[str, Any]:
    """Picklable, frozen description of ``model``'s policy.

    Returns ``{"policy_class", "policy_kwargs", "state_dict"}``: everything
    needed to rebuild an independent copy of the policy (the same recipe
    SB3's ``BasePolicy.save``/``load`` uses). This, not the model, is what
    crosses into ``SubprocVecEnv`` workers: the live model cannot be pickled
    (it owns the vec env and its pipes), and holding a separate copy also
    avoids swapping weights in and out of the learner on every opponent
    action.

    Args:
        model: An SB3 model (anything with a ``.policy``) or a policy itself.
    """
    policy = getattr(model, "policy", model)
    return {
        "policy_class": type(policy),
        # Deep-copied: the constructor kwargs reference the learner policy's
        # own config dicts (optimizer_kwargs, net_arch, ...), which an
        # in-process rebuild would otherwise share with the learner.
        "policy_kwargs": copy.deepcopy(policy._get_constructor_parameters()),
        "state_dict": _state_dict_to_numpy(policy),
    }


def _accepts_action_masks(predictor: Any) -> bool:
    # Lazy import: rl.evaluation is light, but keep module import order
    # independent of it.
    from reinforcetactics.rl.evaluation import _model_accepts_action_masks

    return _model_accepts_action_masks(predictor)


class OpponentPool:
    """
    Manages a pool of opponent models for diverse self-play training.

    This implements "Fictitious Self-Play" where the agent trains against
    a mixture of historical versions of itself, preventing overfitting
    to a single opponent strategy.

    Pool entries are numpy ``state_dict`` copies of the policy; the
    architecture comes from the latest :func:`policy_snapshot` the env
    received.

    Attributes:
        max_size: Maximum number of models to keep in the pool
        models: Deque of parameter dicts
        selection_strategy: How to select opponents ('uniform', 'recent', 'prioritized')
    """

    def __init__(self, max_size: int = 10, selection_strategy: str = "uniform", save_dir: str | None = None):
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
        self._selection_weights: list[float] = []

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def add_model(
        self, model: Any, timestep: int = 0, win_rate: float = 0.5, save_to_disk: bool = True
    ) -> dict[str, np.ndarray] | None:
        """
        Add a model to the pool.

        Args:
            model: The trained model to add
            timestep: Training timestep when model was saved
            win_rate: Model's win rate (for prioritized selection)
            save_to_disk: Whether to save to disk

        Returns:
            The parameter dict that was added (so callers can mirror it into
            pools living in other processes), or None if the parameters
            could not be copied and nothing was added.
        """
        # Deep copy the model's policy parameters
        model_copy = self._copy_model_params(model)
        if not model_copy:
            # An empty entry would be sampled like any other and then fail
            # to load on every episode it is drawn for.
            logger.warning("Not adding opponent at timestep %d to the pool: its parameters could not be copied", timestep)
            return None

        self.add_params(model_copy, timestep=timestep, win_rate=win_rate)

        if save_to_disk and self.save_dir:
            save_path = self.save_dir / f"opponent_{timestep}.zip"
            try:
                # Atomic: a sync or a kill mid-save must not leave a truncated
                # snapshot that later loads (or uploads) as an opponent.
                from reinforcetactics.rl.callbacks import save_model_atomically

                save_model_atomically(model, save_path)
                logger.info("Saved opponent to pool: %s", save_path)
            except Exception as exc:
                logger.warning("Failed to save opponent to disk: %s", exc)

        return model_copy

    def add_params(self, params: dict[str, np.ndarray], timestep: int = 0, win_rate: float = 0.5) -> None:
        """Add an already-copied parameter dict (no disk write)."""
        metadata = {"timestep": timestep, "win_rate": win_rate, "index": len(self.models)}
        self.models.append(params)
        self.metadata.append(metadata)
        self._update_selection_weights()

    def _copy_model_params(self, model: Any) -> dict[str, np.ndarray]:
        """Create a lightweight copy of model parameters."""
        try:
            # For SB3 models, get policy parameters
            return _state_dict_to_numpy(model.policy)
        except Exception as exc:
            logger.warning("Could not copy model params: %s", exc)
            return {}

    def _load_model_params(self, model: Any, params: dict[str, np.ndarray]) -> None:
        """Load parameters into a model's policy."""
        try:
            _load_numpy_state_dict(model.policy, params)
        except Exception as exc:
            logger.warning("Could not load model params: %s", exc)

    def _update_selection_weights(self) -> None:
        """Update selection weights based on strategy."""
        num_models = len(self.models)
        if num_models == 0:
            self._selection_weights = []
            return

        if self.selection_strategy == "uniform":
            self._selection_weights = [1.0 / num_models] * num_models

        elif self.selection_strategy == "recent":
            # Exponentially favor more recent models
            weights = [2.0**i for i in range(num_models)]
            total = sum(weights)
            self._selection_weights = [w / total for w in weights]

        elif self.selection_strategy == "prioritized":
            # Favor models with higher win rates
            win_rates = [m.get("win_rate", 0.5) for m in self.metadata]
            # Add small epsilon to avoid zero weights
            weights = [max(0.1, wr) for wr in win_rates]
            total = sum(weights)
            self._selection_weights = [w / total for w in weights]

    def _sample_index(self, rng: np.random.Generator | None) -> int:
        if len(self._selection_weights) != len(self.models):
            # Entries appended to ``models`` directly (bypassing add_params)
            # leave stale weights behind; recompute rather than fail.
            self._update_selection_weights()
        if rng is None:
            return random.choices(range(len(self.models)), weights=self._selection_weights, k=1)[0]
        probs = np.asarray(self._selection_weights, dtype=np.float64)
        return int(rng.choice(len(self.models), p=probs / probs.sum()))

    def sample_opponent(self, rng: np.random.Generator | None = None) -> dict[str, np.ndarray] | None:
        """
        Sample an opponent from the pool.

        Args:
            rng: Generator to draw from. SelfPlayEnv passes its env's
                ``np_random`` so the draw is reproducible under
                ``reset(seed=...)``; ``None`` falls back to the global
                ``random`` module.

        Returns:
            Model parameters dict, or None if pool is empty
        """
        if not self.models:
            return None
        return self.models[self._sample_index(rng)]

    def sample_opponent_with_metadata(self, rng: np.random.Generator | None = None) -> tuple[dict, dict] | None:
        """Sample an opponent and return with metadata."""
        if not self.models:
            return None
        idx = self._sample_index(rng)
        return self.models[idx], self.metadata[idx]

    def update_win_rate(self, model_idx: int, new_win_rate: float) -> None:
        """Update a model's win rate in the pool."""
        if 0 <= model_idx < len(self.metadata):
            self.metadata[model_idx]["win_rate"] = new_win_rate
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
                timestep = int(path.stem.split("_")[1])
                self.models.append(params)
                self.metadata.append({"timestep": timestep, "win_rate": 0.5, "index": len(self.models) - 1})
                loaded += 1
            except Exception as exc:
                logger.warning("Failed to load opponent %s: %s", path, exc)

        self._update_selection_weights()
        logger.info("Loaded %d opponents from %s", loaded, self.save_dir)
        return loaded

    @property
    def size(self) -> int:
        """Return number of models in pool."""
        return len(self.models)

    def __len__(self) -> int:
        return self.size


class _SelfPlayOpponent:
    """Bot-shaped adapter the base env's ``opponent='self'`` slot plays.

    ``StrategyGameEnv.reset`` builds one per episode through the factory
    SelfPlayEnv registers, and calls ``take_turn()`` on it for player 1's
    opening turn (agent in seat 2) and after every agent end_turn.
    """

    def __init__(self, self_play_env: "SelfPlayEnv", game_state: Any, player: int):
        self._self_play_env = self_play_env
        self.game_state = game_state
        self.bot_player = player

    def take_turn(self) -> None:
        self._self_play_env._execute_opponent_turn(self.game_state, self.bot_player)


class SelfPlayEnv(gym.Wrapper):
    """
    Gymnasium wrapper that enables self-play training.

    The learning agent controls one seat and a frozen snapshot of its own
    policy controls the other. The opponent plays through the base env's
    ``opponent='self'`` hook, so an opponent turn is scored exactly like a
    scripted bot's turn (see ``StrategyGameEnv._execute_action``).

    Features:
    - The opponent observes, is masked and decodes actions for its own seat
    - Configurable opponent update frequency (via SelfPlayCallback)
    - Support for opponent pool (multiple historical models)
    - Optional random seat per episode (``swap_players``)

    Which weights the opponent plays:

    - ``set_opponent_snapshot`` (SelfPlayCallback, at training start and
      every ``update_freq`` calls) installs the *latest* snapshot at once,
      mid-episode included.
    - While the pool is empty, every episode plays the latest snapshot.
    - Once the pool has entries, each reset draws the episode's opponent:
      the latest snapshot with probability ``latest_opponent_prob``, else a
      pool sample (the pool's own ``selection_strategy``). The default 0.0 is
      the long-standing behaviour, which replaces the latest snapshot with a
      pool sample at every reset: a fresh snapshot then only plays out the
      episodes already running when it arrives. The draw uses the env's
      ``np_random`` and is only made for 0 < p < 1, so the default leaves
      the random stream exactly as it was.

    Attributes:
        opponent_model: The model used for opponent decisions until the
            first snapshot is installed (in-process only)
        opponent_pool: Pool of historical opponents (optional)
        swap_players: Whether the agent's seat is drawn per episode
        latest_opponent_prob: See above.
    """

    def __init__(
        self,
        env: gym.Env,
        opponent_model: Any | None = None,
        opponent_pool: OpponentPool | None = None,
        swap_players: bool = True,
        opponent_deterministic: bool = False,
        latest_opponent_prob: float = 0.0,
    ):
        """
        Initialize the self-play environment.

        Args:
            env: A StrategyGameEnv (optionally wrapped, e.g. by
                ActionMaskedEnv) built with ``opponent=None`` or ``'self'``.
            opponent_model: Initial opponent model (can be updated later)
            opponent_pool: Pool of historical opponents for diverse training
            swap_players: Randomly swap which player agent controls each episode
            opponent_deterministic: Use deterministic opponent actions
            latest_opponent_prob: Probability, per episode, of playing the
                latest snapshot instead of a pool sample once the pool has
                entries (in [0, 1]; default 0.0, see the class docstring).
        """
        if not 0.0 <= float(latest_opponent_prob) <= 1.0:
            raise ValueError(f"latest_opponent_prob must be in [0, 1]; got {latest_opponent_prob}")
        super().__init__(env)
        base = env.unwrapped
        if not isinstance(base, StrategyGameEnv):
            raise TypeError(f"SelfPlayEnv needs a StrategyGameEnv underneath; got {type(base).__name__}")
        if base.opponent_type not in (None, "self"):
            raise ValueError(
                f"SelfPlayEnv drives the opponent itself; build the base env with opponent=None or 'self' "
                f"(got {base.opponent_type!r})"
            )
        # Play the opponent through the base env's own opponent slot rather
        # than after step() returns: that path already charges opponent-turn
        # damage/capture penalties and scores a game that ends on the
        # opponent's turn (win_by_*, speed bonus, draw, -Phi terminal),
        # which a wrapper-side re-implementation kept getting wrong.
        base.opponent_type = "self"
        base.set_self_play_opponent_factory(self._build_opponent)

        self.opponent_model = opponent_model
        self.opponent_pool = opponent_pool
        self.opponent_deterministic = opponent_deterministic
        self.latest_opponent_prob = float(latest_opponent_prob)
        self.swap_players = swap_players

        # Frozen opponent policy (built from a policy_snapshot) and the
        # weights it currently holds: the latest snapshot, or a pool sample.
        self._opponent_policy: Any | None = None
        self._opponent_accepts_masks = False
        self._latest_params: dict[str, np.ndarray] | None = None
        self._opponent_params: dict[str, np.ndarray] | None = None
        self._opponent_source = "random"

        # Statistics. ``total_games`` counts every finished episode,
        # including step-limit truncations (recorded as ``truncations``), so
        # the win rate is never inflated by leaving unfinished games out.
        self.stats = {"agent_wins": 0, "opponent_wins": 0, "draws": 0, "truncations": 0, "total_games": 0}

    @property
    def _base_env(self) -> StrategyGameEnv:
        """The StrategyGameEnv under all wrappers (checked in ``__init__``)."""
        return cast(StrategyGameEnv, self.env.unwrapped)

    # ------------------------------------------------------------------
    # Seat
    # ------------------------------------------------------------------

    @property
    def agent_player(self) -> int:
        """The seat the learning agent plays this episode.

        Read from the base env, which is the only copy: it is what builds
        the masks, scores rewards and the shaping potential, and executes
        the agent's actions.
        """
        return int(self._base_env.agent_player)

    @property
    def swap_players(self) -> bool:
        return self._swap_players

    @swap_players.setter
    def swap_players(self, value: bool) -> None:
        # The seat is configured on the base env *before* its reset() runs:
        # reset draws it from np_random (after seeding), binds the opponent
        # to the other seat, plays player 1's opening turn if the agent is
        # player 2, and only then computes Phi(s_0) and the first obs.
        self._swap_players = bool(value)
        self._base_env.set_agent_seat("random" if self._swap_players else 1)

    # ------------------------------------------------------------------
    # Opponent management
    # ------------------------------------------------------------------

    def set_opponent_model(self, model: Any) -> None:
        """Set the model the opponent plays with (in-process only).

        The opponent uses the live model's policy until a frozen snapshot
        is installed with :meth:`update_opponent_from_current` (or
        :meth:`set_opponent_snapshot`, which is what SelfPlayCallback uses
        and which also works across processes).
        """
        self.opponent_model = model

    def set_opponent_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Install a frozen opponent policy built from a :func:`policy_snapshot`.

        Safe to call through ``VecEnv.env_method``: the snapshot is plain
        picklable data, so this is how opponent updates reach
        ``SubprocVecEnv`` workers.
        """
        policy = snapshot["policy_class"](**snapshot["policy_kwargs"])
        params = snapshot["state_dict"]
        _load_numpy_state_dict(policy, params)
        policy.set_training_mode(False)
        self._opponent_policy = policy
        self._opponent_accepts_masks = _accepts_action_masks(policy)
        self._latest_params = params
        self._opponent_params = params
        self._opponent_source = "latest"

    def update_opponent_from_current(self) -> None:
        """Snapshot ``opponent_model``'s current weights as the opponent."""
        if self.opponent_model is not None:
            self.set_opponent_snapshot(policy_snapshot(self.opponent_model))

    def update_opponent_from_pool(self) -> bool:
        """
        Sample a new opponent from the pool (drawn from the env's np_random).

        Returns:
            True if the opponent now plays a pool member, False if the pool
            is empty, no opponent architecture is known yet (no snapshot has
            been installed), or the sampled weights could not be loaded.
        """
        if self.opponent_pool is None or self.opponent_pool.size == 0:
            return False
        params = self.opponent_pool.sample_opponent(rng=self._base_env.np_random)
        if params is None:
            return False
        return self._load_opponent_params(params, source="pool")

    def add_opponent_to_pool(self, params: dict[str, np.ndarray], metadata: Mapping[str, Any] | None = None) -> bool:
        """Mirror a pool addition made in the trainer process into this env's pool.

        Under ``SubprocVecEnv`` each worker holds its own copy of the pool
        (pickled with the env factory), so ``OpponentPool.add_model`` in the
        trainer never reaches it. SelfPlayCallback forwards every addition
        here through ``env_method``. In-process envs usually share the
        trainer's pool object, where the entry is already present.

        Returns:
            True if the entry was added to this env's pool.
        """
        if self.opponent_pool is None:
            return False
        if any(existing is params for existing in self.opponent_pool.models):
            return False
        meta = dict(metadata or {})
        self.opponent_pool.add_params(params, timestep=int(meta.get("timestep", 0)), win_rate=float(meta.get("win_rate", 0.5)))
        return True

    def has_opponent_pool(self) -> bool:
        return self.opponent_pool is not None

    def describe_opponent(self) -> dict[str, Any]:
        """Which opponent this env currently plays (diagnostics / tests)."""
        params = self._opponent_params
        if self._opponent_policy is not None:
            source = self._opponent_source
        else:
            source = "live" if self.opponent_model is not None else "random"
        return {
            "source": source,
            "params_checksum": params_checksum(params) if params is not None else None,
            "pool_size": self.opponent_pool.size if self.opponent_pool is not None else 0,
        }

    def _load_opponent_params(self, params: dict[str, np.ndarray], source: str) -> bool:
        if self._opponent_policy is None:
            logger.debug("No opponent architecture known yet; ignoring %s parameters", source)
            return False
        if params is self._opponent_params:
            self._opponent_source = source
            return True
        try:
            _load_numpy_state_dict(self._opponent_policy, params)
        except Exception as exc:
            logger.warning("Could not load %s opponent parameters (%s); keeping the latest snapshot", source, exc)
            if self._latest_params is not None and self._opponent_params is not self._latest_params:
                _load_numpy_state_dict(self._opponent_policy, self._latest_params)
                self._opponent_params = self._latest_params
                self._opponent_source = "latest"
            return False
        self._opponent_params = params
        self._opponent_source = source
        return True

    def _select_episode_opponent(self) -> None:
        """Choose this episode's opponent weights: latest snapshot or a pool sample.

        See the class docstring. With the pool empty (or absent) the latest
        snapshot simply stays in place.
        """
        if self.opponent_pool is None or self.opponent_pool.size == 0:
            return
        p = self.latest_opponent_prob
        if p >= 1.0 or (p > 0.0 and float(self._base_env.np_random.random()) < p):
            if self._latest_params is not None:
                self._load_opponent_params(self._latest_params, source="latest")
            return
        self.update_opponent_from_pool()

    def _build_opponent(self, game_state: Any, opponent_player: int) -> _SelfPlayOpponent:
        """Opponent factory registered with the base env.

        Called from ``StrategyGameEnv.reset`` after ``np_random`` has been
        seeded and before player 1's opening turn, so the per-episode
        opponent draw is reproducible and in place for the opponent's first
        move.
        """
        self._select_episode_opponent()
        return _SelfPlayOpponent(self, game_state, opponent_player)

    # ------------------------------------------------------------------
    # Opponent turn
    # ------------------------------------------------------------------

    def _execute_opponent_turn(self, game_state: Any | None = None, player: int | None = None) -> None:
        """Play one full turn for the opponent seat, then hand the turn back."""
        base_env = self._base_env
        game_state = base_env.game_state if game_state is None else game_state
        opponent_player = 3 - self.agent_player if player is None else player

        # Mirror the agent's per-turn budget: the same policy was trained to
        # see an end_turn-only mask once ``max_actions_per_turn`` is used up.
        max_actions = _MAX_OPPONENT_ACTIONS
        if base_env.max_actions_per_turn is not None:
            max_actions = min(max_actions, base_env.max_actions_per_turn)

        actions_taken = 0
        consecutive_invalid = 0
        while game_state.current_player == opponent_player and not game_state.game_over and actions_taken < max_actions:
            action_arr = self._get_opponent_action(opponent_player)
            if int(action_arr[0]) == 5:
                break

            action_dict = base_env._encode_action(action_arr)
            _, is_valid = base_env.execute_game_action(action_dict, opponent_player)

            if is_valid:
                consecutive_invalid = 0
                actions_taken += 1
            else:
                consecutive_invalid += 1
                if consecutive_invalid >= _MAX_CONSECUTIVE_INVALID:
                    break

        if game_state.current_player == opponent_player and not game_state.game_over:
            game_state.end_turn()

    def _get_opponent_action(self, player: int) -> np.ndarray:
        """Choose the opponent's next action as a 6-vector, for its own seat.

        The observation, the masks passed to ``predict`` and (for
        flat_discrete) the index -> action table are all built for
        ``player``. Using the base env's ``action_masks()`` /
        ``_current_actions`` instead would describe the *agent's* legal
        moves, which is how the opponent used to end up executing nothing.
        """
        policy = self._opponent_policy
        accepts_masks = self._opponent_accepts_masks
        if policy is None and self.opponent_model is not None:
            policy = self.opponent_model.policy
            accepts_masks = _accepts_action_masks(policy)
        if policy is None:
            return self._get_random_valid_action(player)

        base_env = self._base_env
        try:
            obs = self._build_obs_for_player(player)
            if base_env.action_space_type == "flat_discrete":
                actions = build_flat_actions(
                    base_env.game_state, player, base_env.max_flat_actions, version=base_env.flat_action_version
                )
                mask = np.zeros(base_env.max_flat_actions, dtype=bool)
                mask[: len(actions)] = True
                raw = self._predict_opponent(policy, obs, mask if accepts_masks else None)
                idx = int(np.asarray(raw).reshape(-1)[0])
                if 0 <= idx < len(actions):
                    return actions[idx]
                return np.array(_END_TURN, dtype=np.int32)

            per_dim = self._opponent_per_dim_masks(player)
            mask = np.concatenate([m.astype(np.bool_) for m in per_dim])
            raw = self._predict_opponent(policy, obs, mask if accepts_masks else None)
            return np.asarray(raw).reshape(-1)
        except Exception as exc:
            logger.warning("Error getting opponent action: %s", exc)
            return self._get_random_valid_action(player)

    def _predict_opponent(self, policy: Any, obs: dict[str, np.ndarray], action_masks: np.ndarray | None) -> Any:
        kwargs: dict[str, Any] = {"deterministic": self.opponent_deterministic}
        if action_masks is not None:
            kwargs["action_masks"] = action_masks
        if self.opponent_deterministic:
            action, _ = policy.predict(obs, **kwargs)
            return action

        import torch

        # Stochastic predict() samples from torch's global CPU generator.
        # Seed it from the env's np_random inside fork_rng so the opponent
        # is reproducible under reset(seed=...) and the trainer's own torch
        # stream is left exactly as it was. Only the CPU generator is
        # touched (``default_generator``, not ``torch.manual_seed``, which
        # would also reseed CUDA): snapshot policies always live on the CPU.
        seed = int(self._base_env.np_random.integers(0, 2**31 - 1))
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            action, _ = policy.predict(obs, **kwargs)
        return action

    def _opponent_per_dim_masks(self, player: int) -> tuple[np.ndarray, ...]:
        """Per-dimension multi_discrete masks for ``player``'s legal actions."""
        base_env = self._base_env
        _, at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask = build_per_dim_masks(
            base_env.game_state,
            base_env.grid_width,
            base_env.grid_height,
            enabled_units=base_env.enabled_units,
            player=player,
        )
        return (at_mask, ut_mask, fx_mask, fy_mask, tx_mask, ty_mask)

    def _get_random_valid_action(self, player: int | None = None) -> np.ndarray:
        """A uniformly random legal action (6-vector) for ``player``.

        The fallback opponent before any policy is installed. Drawn from
        the exact legal list in both action spaces (sampling each
        multi_discrete dimension independently mostly yields illegal
        combinations) and from the env's np_random, not the global RNGs.
        Defaults to the opponent seat.
        """
        base_env = self._base_env
        if player is None:
            player = 3 - self.agent_player
        actions = build_flat_actions(
            base_env.game_state, player, base_env.max_flat_actions, version=base_env.flat_action_version
        )
        idx = int(base_env.np_random.integers(len(actions)))
        return np.array(actions[idx])

    def _flip_observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Build an observation from the opponent's perspective.

        Previously this method swapped ownership channels and gold/unit-count
        slots in-place on the agent's observation. That produced the right
        labels under perfect information but was wrong under fog of war:
        the visibility filter applied by ``GameState.to_numpy(for_player=...)``
        belonged to the agent, not the opponent, so the "flipped" obs leaked
        the agent's vision and hid the opponent's own information.

        The fix is to rebuild the observation from scratch via
        ``build_observation(perspective_player=opponent)``, which applies the
        opponent's visibility (or none, when FOW is off) and computes the
        agent-relative global_features for the opponent. The ``obs`` argument
        is ignored — kept only for API compatibility — because nothing the
        agent's obs carries is reusable from the opponent's POV.

        Args:
            obs: Unused. Retained to preserve the existing call signature.

        Returns:
            Observation dict from the opponent's perspective.
        """
        del obs  # See docstring; rebuilt from game state instead.
        return self._build_obs_for_player(3 - self.agent_player)

    def _build_obs_for_player(self, player: int) -> dict[str, np.ndarray]:
        """Build a fresh observation from ``player``'s perspective.

        Delegates to the shared ``build_observation`` helper so FOW handling
        and padding match gym_env and ModelBot. The action mask is not
        embedded in the obs dict (policies consume it through
        ``predict(action_masks=...)``).
        """
        base_env = self._base_env
        pad_to: tuple[int, int] | None = None
        if base_env.pad_height != base_env.grid_height or base_env.pad_width != base_env.grid_width:
            pad_to = (base_env.pad_height, base_env.pad_width)
        return build_observation(
            base_env.game_state,
            perspective_player=player,
            action_mask=None,
            fog_of_war=base_env.fog_of_war,
            pad_to=pad_to,
            gold_scale=base_env.gold_scale,
            turn_scale=base_env.turn_scale,
            unit_count_scale=base_env.unit_count_scale,
        )

    def _get_obs_for_player(self, player: int) -> dict[str, np.ndarray]:
        """Get observation from a specific player's perspective.

        For the agent, this returns the env's standard observation. For the
        opponent, it rebuilds the observation from the opponent's POV via
        ``build_observation`` so that fog-of-war visibility is applied
        correctly (rather than reusing the agent's filtered grid/units).
        """
        if player == self.agent_player:
            return self._base_env._get_obs()
        return self._build_obs_for_player(player)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute the agent's action. On end_turn the base env plays the
        opponent's whole turn before returning.

        Args:
            action: Agent's action (array for multi_discrete, scalar for flat_discrete)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            self._record_outcome(info.get("winner") if terminated else None, terminated)

        info["self_play_stats"] = self.stats.copy()
        info["agent_player"] = self.agent_player
        return obs, reward, terminated, truncated, info

    def _record_outcome(self, winner: int | None, terminated: bool) -> None:
        """Count a finished episode, whichever move (agent's or opponent's) ended it."""
        if not terminated:
            self.stats["truncations"] += 1
        elif winner == self.agent_player:
            self.stats["agent_wins"] += 1
        elif winner is not None:
            self.stats["opponent_wins"] += 1
        else:
            self.stats["draws"] += 1
        self.stats["total_games"] += 1

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """
        Reset the environment.

        The base env draws the seat (when ``swap_players``), samples a pool
        opponent through the registered factory, and plays player 1's
        opening turn if the agent is player 2.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["agent_player"] = self.agent_player
        return obs, info

    def action_masks(self) -> np.ndarray:
        """Agent masks in the layout MaskablePPO expects.

        flat_discrete: the ``(max_flat_actions,)`` mask. multi_discrete: the
        six per-dimension masks concatenated into one 1-D array (what
        ``ActionMaskedEnv.action_masks`` returns and what sb3-contrib's
        ``get_action_masks`` stacks across envs). Returning the tuple here
        made MaskablePPO crash on its first rollout.
        """
        return np.concatenate([m.astype(np.bool_) for m in self.get_action_masks_tuple()])

    def get_action_masks_tuple(self) -> tuple[np.ndarray, ...]:
        """The agent's masks as a tuple (one array per action dimension)."""
        return tuple(self._base_env.action_masks())

    def get_self_play_stats(self) -> dict[str, int]:
        return self.stats.copy()

    def get_win_rate(self) -> float:
        """Get agent's win rate against opponents."""
        total = self.stats["total_games"]
        if total == 0:
            return 0.5
        return self.stats["agent_wins"] / total


def _find_self_play_envs(env: Any) -> list[SelfPlayEnv]:
    """In-process SelfPlayEnvs reachable from ``env`` (a gym env or DummyVecEnv)."""
    candidates = env.envs if hasattr(env, "envs") else [env]
    found = []
    for candidate in candidates:
        current = candidate
        while current is not None:
            if isinstance(current, SelfPlayEnv):
                found.append(current)
                break
            current = getattr(current, "env", None)
    return found


def _is_vec_env(env: Any) -> bool:
    return hasattr(env, "env_method") and hasattr(env, "env_is_wrapped")


def _make_callback_class():
    """Build SelfPlayCallback with BaseCallback as parent when SB3 is available."""
    base = _BaseCallback if _BaseCallback is not None else object

    class _SelfPlayCallback(base):
        """
        Callback for Stable-Baselines3 that manages self-play opponent updates.

        When ``stable-baselines3`` is installed this class inherits from
        ``BaseCallback``, so it can be passed directly to
        ``model.learn(callback=...)``.  The parent handles ``self.model``,
        ``self.n_calls``, ``self.num_timesteps``, and the full lifecycle
        (``init_callback`` → ``on_training_start`` → ``on_step`` → …).

        This callback:
        1. Pushes a snapshot of the current policy to every opponent at
           training start and every ``update_freq`` calls. The snapshot
           becomes each env's *latest* opponent at once; whether later
           episodes keep playing it depends on the pool (see 2).
        2. Optionally adds the model to the opponent pool (and mirrors the
           addition into worker-process pools). Once the pool has entries,
           every episode's opponent is drawn at reset: the latest snapshot
           with probability ``SelfPlayEnv.latest_opponent_prob``, else a
           pool sample. With the default 0.0 a pushed snapshot is replaced
           by a pool sample at the next reset, i.e. with a non-empty pool
           ``update_freq`` only refreshes the episodes in flight.
        3. Tracks win rates and logs stats (incl. tensorboard when available)

        Vectorized envs (DummyVecEnv, SubprocVecEnv, VecMonitor over either,
        or a mixed bot/self-play VecEnv) are driven through
        ``env_method`` on exactly the workers wrapped in SelfPlayEnv, so the
        same code reaches in-process envs and worker processes. Finding no
        SelfPlayEnv is an error, not a silent no-op.

        Frequencies count ``n_calls`` (one per vec-env step, i.e. ``n_envs``
        timesteps), as in SB3's own callbacks.

        Usage:
            callback = SelfPlayCallback(vec_env, update_freq=1250, opponent_pool=pool)
            # Or with an explicit list of in-process SelfPlayEnvs:
            callback = SelfPlayCallback(envs=self_play_envs, opponent_pool=pool)
            model.learn(total_timesteps=1000000, callback=callback)
        """

        def __init__(
            self,
            env: SelfPlayEnv | Any = None,
            update_freq: int = 10000,
            add_to_pool_freq: int = 50000,
            min_win_rate_for_pool: float = 0.55,
            verbose: int = 1,
            *,
            envs: list[SelfPlayEnv] | None = None,
            opponent_pool: Any = None,
        ):
            """
            Initialize the callback.

            Args:
                env: The SelfPlayEnv or vectorized environment whose
                    self-play opponents to manage. Mutually exclusive with
                    ``envs``.
                update_freq: How often (in calls) to push a snapshot of the
                    current model to the opponents as their latest snapshot.
                    It is played until the next reset, and after it only
                    while the pool is empty or when the per-episode draw
                    picks it (``SelfPlayEnv.latest_opponent_prob``; with the
                    default 0.0 a non-empty pool always wins the draw).
                add_to_pool_freq: How often (in calls) to consider adding
                    the model to the opponent pool
                min_win_rate_for_pool: Minimum win rate, over the games
                    finished since the previous pool check, to add to pool
                verbose: Verbosity level
                envs: Explicit list of in-process ``SelfPlayEnv`` instances
                    (skips discovery).
                opponent_pool: Shared :class:`OpponentPool` to add snapshots
                    to. Defaults to the first pool found on in-process envs;
                    required when the envs live in worker processes.
            """
            if _BaseCallback is not None:
                super().__init__(verbose=verbose)
            else:
                # Fallback attributes when SB3 is not installed
                self.n_calls = 0
                self.num_timesteps = 0
                self.model = None
                self.verbose = verbose

            if env is None and envs is None:
                raise ValueError("SelfPlayCallback needs either env= or envs=")
            if envs is not None and len(envs) == 0:
                raise ValueError("SelfPlayCallback got envs=[]: there are no self-play opponents to update")

            self.env = env
            self._explicit_envs = list(envs) if envs is not None else None
            self.opponent_pool = opponent_pool
            self.update_freq = update_freq
            self.add_to_pool_freq = add_to_pool_freq
            self.min_win_rate_for_pool = min_win_rate_for_pool

            self.win_rate_history: list[float] = []
            self.pool_additions = 0
            self._vec_indices: list[int] | None = None
            # (agent_wins, total_games) at the previous pool check: the pool
            # gate judges the games played since then, not the whole run.
            self._pool_window_start = (0, 0)

        # -- target plumbing ------------------------------------------------

        def _uses_env_method(self) -> bool:
            return self._explicit_envs is None and _is_vec_env(self.env)

        def _self_play_indices(self) -> list[int]:
            """Indices of the vec-env workers wrapped in SelfPlayEnv."""
            if self._vec_indices is None:
                flags = self.env.env_is_wrapped(SelfPlayEnv)
                indices = [i for i, wrapped in enumerate(flags) if wrapped]
                if not indices:
                    raise ValueError(
                        f"SelfPlayCallback: none of the {len(flags)} envs in {type(self.env).__name__} is a "
                        "SelfPlayEnv, so there is no opponent to update. Build the envs with "
                        "make_self_play_vec_env."
                    )
                self._vec_indices = indices
            return self._vec_indices

        def _call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
            """Call ``method`` on every self-play env, in-process or in a worker."""
            if self._uses_env_method():
                return self.env.env_method(method, *args, indices=self._self_play_indices(), **kwargs)
            return [getattr(env, method)(*args, **kwargs) for env in self._get_self_play_envs()]

        def _get_self_play_envs(self) -> list[SelfPlayEnv]:
            """In-process SelfPlayEnv instances (explicit list or discovered).

            Raises for SubprocVecEnv (whose envs live in other processes and
            are reached through ``env_method``) and when none are found.
            """
            if self._explicit_envs is not None:
                return self._explicit_envs
            if _is_vec_env(self.env) and not hasattr(self.env, "envs"):
                raise TypeError(
                    f"{type(self.env).__name__} runs its envs in worker processes; SelfPlayCallback reaches "
                    "them through env_method and cannot return the env objects."
                )
            envs = _find_self_play_envs(self.env)
            if not envs:
                raise ValueError(f"SelfPlayCallback: no SelfPlayEnv found in {type(self.env).__name__}")
            return envs

        def _resolve_pool(self):
            """The shared pool if given, else the first in-process env pool found."""
            if self.opponent_pool is not None:
                return self.opponent_pool
            if self._uses_env_method() and not hasattr(self.env, "envs"):
                return None
            for env in self._get_self_play_envs():
                if env.opponent_pool is not None:
                    return env.opponent_pool
            return None

        def _collect_stats(self) -> tuple[int, int]:
            """(agent_wins, total_games) summed over all self-play envs."""
            stats = self._call("get_self_play_stats")
            return sum(s["agent_wins"] for s in stats), sum(s["total_games"] for s in stats)

        # -- lifecycle ----------------------------------------------------

        def _init_callback(self) -> None:
            """Called by BaseCallback.init_callback() after self.model is set."""

        def _on_training_start(self) -> None:
            """Validate the wiring and initialize opponents with the current model."""
            if (
                self.opponent_pool is None
                and self._uses_env_method()
                and not hasattr(self.env, "envs")
                and any(self._call("has_opponent_pool"))
            ):
                # Each worker holds a private copy of the pool; without the
                # trainer-side pool nothing could ever be added to it.
                raise ValueError(
                    "The self-play workers have an opponent pool but SelfPlayCallback got no opponent_pool=. "
                    "Pass the same OpponentPool given to make_self_play_vec_env."
                )
            if self.verbose >= 1:
                logger.info("Initializing self-play opponents with current model...")
            self._update_opponents(log=False)

        def _on_step(self) -> bool:
            """Called after each env.step() by BaseCallback.on_step()."""
            # Update opponent model
            if self.n_calls % self.update_freq == 0:
                self._update_opponents()
                self._log_stats()

            # Add to pool
            if self.n_calls % self.add_to_pool_freq == 0:
                self._add_to_pool()

            return True

        def _get_average_win_rate(self) -> float:
            """Agent win rate over every game finished so far, pooled across envs."""
            wins, games = self._collect_stats()
            return wins / games if games else 0.5

        def _update_opponents(self, log: bool = True) -> None:
            """Install a snapshot of the current model as every env's latest opponent."""
            self._call("set_opponent_snapshot", policy_snapshot(self.model))

            if log and self.verbose >= 1:
                logger.info(
                    "Step %d: Updated opponents. Avg win rate: %.2f%%",
                    self.num_timesteps,
                    self._get_average_win_rate() * 100,
                )

        def _log_stats(self) -> None:
            """Log training statistics (and tensorboard series when attached)."""
            if self.verbose < 1 and getattr(self, "logger", None) is None:
                return
            total_wins, total_games = self._collect_stats()
            avg_win_rate = total_wins / total_games if total_games else 0.5

            if self.verbose >= 1:
                logger.info(
                    "Step %d: Win rate: %.2f%%, Total games: %d, Wins: %d",
                    self.num_timesteps,
                    avg_win_rate * 100,
                    total_games,
                    total_wins,
                )

            # Tensorboard, via SB3's logger (present once attached to a model).
            sb3_logger = getattr(self, "logger", None)
            if sb3_logger is not None:
                sb3_logger.record("self_play/win_rate", avg_win_rate)
                sb3_logger.record("self_play/total_games", total_games)
                pool = self._resolve_pool()
                if pool is not None:
                    sb3_logger.record("self_play/pool_size", pool.size)

        def _add_to_pool(self) -> None:
            """Add current model to opponent pool if its recent win rate is good enough."""
            pool = self._resolve_pool()
            if pool is None:
                return

            wins, games = self._collect_stats()
            prev_wins, prev_games = self._pool_window_start
            self._pool_window_start = (wins, games)
            window_games = games - prev_games
            if window_games <= 0:
                if self.verbose >= 1:
                    logger.info("Step %d: No self-play games finished since the last pool check", self.num_timesteps)
                return
            win_rate = (wins - prev_wins) / window_games
            self.win_rate_history.append(win_rate)

            if win_rate >= self.min_win_rate_for_pool:
                params = pool.add_model(self.model, timestep=self.num_timesteps, win_rate=win_rate)
                if params is None:
                    return
                self.pool_additions += 1
                # Mirror into the envs' own pools (worker processes hold
                # copies; in-process envs sharing ``pool`` skip it).
                self._call("add_opponent_to_pool", params, {"timestep": self.num_timesteps, "win_rate": win_rate})
                if self.verbose >= 1:
                    logger.info(
                        "Step %d: Added model to pool (win rate: %.2f%% over %d games, pool size: %d)",
                        self.num_timesteps,
                        win_rate * 100,
                        window_games,
                        pool.size,
                    )
            elif self.verbose >= 1:
                logger.info(
                    "Step %d: Win rate %.2f%% over %d games below threshold %.2f%%, not adding to pool",
                    self.num_timesteps,
                    win_rate * 100,
                    window_games,
                    self.min_win_rate_for_pool * 100,
                )

    return _SelfPlayCallback


SelfPlayCallback = _make_callback_class()


def _env_kwargs(
    *,
    map_file: str | None,
    max_steps: int,
    max_turns: int | None,
    reward_config: dict[str, float] | None,
    enabled_units: list[str] | None,
    action_space_type: str,
    max_flat_actions: int,
    max_actions_per_turn: int | None,
    gamma: float,
    pad_to_size: tuple[int, int] | None,
    fog_of_war: bool,
    engine_overrides: dict[str, Any] | None,
    gold_scale: float | None,
    turn_scale: float | None,
    unit_count_scale: float | None,
    flat_action_version: int | None,
) -> dict[str, Any]:
    """The StrategyGameEnv construction kwargs shared by self-play and bot workers.

    Every EnvConfig field a self-play run can honour. ``engine_overrides``,
    the observation scales and ``fog_of_war`` used to be hard-coded to the
    defaults here, so ``train_self_play.py --config`` with a balance overlay
    or fog of war silently trained the default game. ``opponent_kwargs`` is
    set per worker kind (self-play workers take none).
    """
    return {
        "map_file": map_file,
        "render_mode": None,
        "max_steps": max_steps,
        "max_turns": max_turns,
        "reward_config": reward_config,
        "enabled_units": enabled_units,
        "action_space_type": action_space_type,
        "max_flat_actions": max_flat_actions,
        "max_actions_per_turn": max_actions_per_turn,
        "opponent_kwargs": None,
        "gamma": gamma,
        "pad_to_size": tuple(pad_to_size) if pad_to_size is not None else None,
        "engine_overrides": engine_overrides,
        "gold_scale": gold_scale,
        "turn_scale": turn_scale,
        "unit_count_scale": unit_count_scale,
        "fog_of_war": bool(fog_of_war),
        "flat_action_version": flat_action_version,
    }


def _build_self_play_env(
    env_kwargs: dict[str, Any],
    opponent_pool: OpponentPool | None,
    swap_players: bool,
    latest_opponent_prob: float = 0.0,
) -> SelfPlayEnv:
    from reinforcetactics.rl.masking import ActionMaskedEnv, _build_strategy_env

    base_env = _build_strategy_env(opponent="self", **env_kwargs)
    return SelfPlayEnv(
        ActionMaskedEnv(base_env),
        opponent_pool=opponent_pool,
        swap_players=swap_players,
        latest_opponent_prob=latest_opponent_prob,
    )


def make_self_play_env(
    map_file: str | None = None,
    max_steps: int = 500,
    reward_config: dict[str, float] | None = None,
    opponent_pool: OpponentPool | None = None,
    swap_players: bool = True,
    enabled_units: list[str] | None = None,
    action_space_type: str = "multi_discrete",
    max_flat_actions: int = 512,
    max_turns: int | None = None,
    max_actions_per_turn: int | None = None,
    gamma: float = 0.99,
    pad_to_size: tuple[int, int] | None = None,
    seed: int | None = None,
    fog_of_war: bool = False,
    engine_overrides: dict[str, Any] | None = None,
    gold_scale: float | None = None,
    turn_scale: float | None = None,
    unit_count_scale: float | None = None,
    flat_action_version: int | None = None,
    latest_opponent_prob: float = 0.0,
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
        action_space_type: 'multi_discrete' (default) or 'flat_discrete'
        max_flat_actions: Max actions for flat_discrete mode (default 512)
        max_turns: Game-turn limit before a draw (None = unlimited)
        max_actions_per_turn: Optional per-turn action budget (both seats)
        gamma: Discount for potential-based shaping; match the trainer's
        pad_to_size: Optional ``(pad_h, pad_w)`` observation padding
            (flat_discrete only)
        seed: Optional seed for an initial ``reset(seed=...)``
        fog_of_war: Partial observability (each seat sees its own view)
        engine_overrides: Sparse overlay over the engine constants
        gold_scale / turn_scale / unit_count_scale: Observation tanh
            divisors (``None`` keeps the env defaults)
        flat_action_version: flat_discrete decode-table layout (``None``
            keeps the env default, the latest version)
        latest_opponent_prob: See :class:`SelfPlayEnv`.

    Returns:
        SelfPlayEnv ready for training

    Example:
        env = make_self_play_env()
        model = MaskablePPO("MultiInputPolicy", env)
        callback = SelfPlayCallback(env, update_freq=10000)
        model.learn(total_timesteps=1000000, callback=callback)
    """
    env = _build_self_play_env(
        _env_kwargs(
            map_file=map_file,
            max_steps=max_steps,
            max_turns=max_turns,
            reward_config=reward_config,
            enabled_units=enabled_units,
            action_space_type=action_space_type,
            max_flat_actions=max_flat_actions,
            max_actions_per_turn=max_actions_per_turn,
            gamma=gamma,
            pad_to_size=pad_to_size,
            fog_of_war=fog_of_war,
            engine_overrides=engine_overrides,
            gold_scale=gold_scale,
            turn_scale=turn_scale,
            unit_count_scale=unit_count_scale,
            flat_action_version=flat_action_version,
        ),
        opponent_pool,
        swap_players,
        latest_opponent_prob,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def _make_self_play_env_fn(
    rank: int,
    seed: int,
    *,
    env_kwargs: dict[str, Any],
    opponent_pool: OpponentPool | None,
    swap_players: bool,
    latest_opponent_prob: float = 0.0,
) -> Callable[[], SelfPlayEnv]:
    """Create a function that creates a self-play environment."""

    def _init() -> SelfPlayEnv:
        env = _build_self_play_env(env_kwargs, opponent_pool, swap_players, latest_opponent_prob)
        # Seed through the wrapper so the seat draw is part of the seeded
        # stream too.
        env.reset(seed=seed + rank)
        return env

    return _init


def _make_bot_env_fn(
    rank: int,
    seed: int,
    *,
    env_kwargs: dict[str, Any],
    opponent: str,
    opponent_kwargs: dict[str, Any] | None = None,
) -> Callable[[], gym.Env]:
    """A scripted-bot worker for a mixed VecEnv (same spaces as the self-play workers)."""

    def _init() -> gym.Env:
        from reinforcetactics.rl.masking import ActionMaskedEnv, _build_strategy_env

        env = _build_strategy_env(opponent=opponent, **{**env_kwargs, "opponent_kwargs": opponent_kwargs})
        env.reset(seed=seed + rank)
        # No Monitor here, matching the self-play workers: the caller wraps
        # the whole VecEnv in VecMonitor.
        return ActionMaskedEnv(env)

    return _init


def make_self_play_vec_env(
    n_envs: int = 4,
    map_file: str | None = None,
    max_steps: int = 500,
    reward_config: dict[str, float] | None = None,
    seed: int = 0,
    use_subprocess: bool = True,
    opponent_pool: OpponentPool | None = None,
    swap_players: bool = True,
    enabled_units: list[str] | None = None,
    action_space_type: str = "multi_discrete",
    max_flat_actions: int = 512,
    max_turns: int | None = None,
    max_actions_per_turn: int | None = None,
    gamma: float = 0.99,
    pad_to_size: tuple[int, int] | None = None,
    bot_ratio: float = 0.0,
    bot_opponent: str = "bot",
    fog_of_war: bool = False,
    engine_overrides: dict[str, Any] | None = None,
    gold_scale: float | None = None,
    turn_scale: float | None = None,
    unit_count_scale: float | None = None,
    flat_action_version: int | None = None,
    bot_opponent_kwargs: dict[str, Any] | None = None,
    latest_opponent_prob: float = 0.0,
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
        action_space_type: 'multi_discrete' (default) or 'flat_discrete'
        max_flat_actions: Max actions for flat_discrete mode (default 512)
        max_turns: Game-turn limit before a draw (None = unlimited)
        max_actions_per_turn: Optional per-turn action budget (both seats)
        gamma: Discount for potential-based shaping; match the trainer's
        pad_to_size: Optional ``(pad_h, pad_w)`` observation padding
        bot_ratio: Fraction of workers that play a scripted bot instead of
            self-play (mixed training). ``round(n_envs * bot_ratio)``
            workers, which must leave at least one of each kind when > 0.
        bot_opponent: Opponent type for the bot workers: a scripted bot
            from the registry (``bot_registry.accepted_names()``).
        fog_of_war, engine_overrides, gold_scale, turn_scale,
            unit_count_scale, flat_action_version: Forwarded to every
            worker's env (see :func:`make_self_play_env`).
        bot_opponent_kwargs: Constructor kwargs for the bot workers'
            opponent (validated against it).
        latest_opponent_prob: Per-episode probability that a self-play
            worker plays the latest snapshot rather than a pool sample once
            the pool has entries (see :class:`SelfPlayEnv`). The default 0.0
            keeps the long-standing pool-only behaviour.

    Returns:
        Vectorized environment ready for MaskablePPO. Wrap it in
        ``VecMonitor`` for episode statistics, and pass it (not a list of
        envs) to :class:`SelfPlayCallback`.

    Example:
        pool = OpponentPool(max_size=10)
        vec_env = make_self_play_vec_env(n_envs=8, opponent_pool=pool)
        model = MaskablePPO("MultiInputPolicy", vec_env)
        callback = SelfPlayCallback(vec_env, update_freq=1250, opponent_pool=pool)
        model.learn(total_timesteps=1000000, callback=callback)
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if not 0.0 <= bot_ratio < 1.0:
        raise ValueError(f"bot_ratio must be in [0, 1); got {bot_ratio}")
    n_bot = int(round(n_envs * bot_ratio))
    if bot_ratio > 0 and not 0 < n_bot < n_envs:
        raise ValueError(
            f"bot_ratio={bot_ratio} with n_envs={n_envs} gives {n_bot} bot workers; mixed training needs at "
            "least one bot worker and one self-play worker"
        )
    if not 0.0 <= latest_opponent_prob <= 1.0:
        raise ValueError(f"latest_opponent_prob must be in [0, 1]; got {latest_opponent_prob}")
    if n_bot:
        # Checked here, in the trainer process: inside a SubprocVecEnv worker
        # the env's own ValueError surfaces only as a broken pipe. 'self' (or
        # None) would leave the bot workers with no opponent at all.
        from reinforcetactics.game.bot_registry import accepted_names, is_scripted_name
        from reinforcetactics.rl.env_schema import validate_opponent_kwargs

        if not (isinstance(bot_opponent, str) and is_scripted_name(bot_opponent)):
            raise ValueError(f"bot_opponent must be a scripted bot ({', '.join(accepted_names())}); got {bot_opponent!r}")
        validate_opponent_kwargs(bot_opponent, bot_opponent_kwargs)

    env_kwargs = _env_kwargs(
        map_file=map_file,
        max_steps=max_steps,
        max_turns=max_turns,
        reward_config=reward_config,
        enabled_units=enabled_units,
        action_space_type=action_space_type,
        max_flat_actions=max_flat_actions,
        max_actions_per_turn=max_actions_per_turn,
        gamma=gamma,
        pad_to_size=pad_to_size,
        fog_of_war=fog_of_war,
        engine_overrides=engine_overrides,
        gold_scale=gold_scale,
        turn_scale=turn_scale,
        unit_count_scale=unit_count_scale,
        flat_action_version=flat_action_version,
    )
    env_fns: list[Callable[[], gym.Env]] = [
        _make_self_play_env_fn(
            i,
            seed,
            env_kwargs=env_kwargs,
            opponent_pool=opponent_pool,
            swap_players=swap_players,
            latest_opponent_prob=latest_opponent_prob,
        )
        for i in range(n_envs - n_bot)
    ]
    env_fns += [
        _make_bot_env_fn(i, seed, env_kwargs=env_kwargs, opponent=bot_opponent, opponent_kwargs=bot_opponent_kwargs)
        for i in range(n_envs - n_bot, n_envs)
    ]

    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env
