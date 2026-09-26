"""
Action masking utilities for Reinforce Tactics.

Provides wrappers and helper functions for training with MaskablePPO from sb3-contrib.
This significantly improves training efficiency by preventing the agent from attempting
invalid actions.

Usage:
    from reinforcetactics.rl.masking import make_maskable_env, make_maskable_vec_env

    # Single environment
    env = make_maskable_env(map_file="maps/1v1/small.csv", opponent="bot")

    # Vectorized environments for faster training
    vec_env = make_maskable_vec_env(n_envs=4, opponent="bot")

    # Train with MaskablePPO
    from sb3_contrib import MaskablePPO
    model = MaskablePPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100000)
"""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from reinforcetactics.rl.gym_env import StrategyGameEnv


class ActionMaskedEnv(gym.Wrapper):
    """
    Wrapper that ensures the environment properly supports action masking
    for sb3-contrib's MaskablePPO.

    This wrapper:
    1. Ensures action_masks() is properly exposed
    2. Validates that masks are consistent
    3. Optionally tracks masking statistics for debugging
    """

    def __init__(self, env: StrategyGameEnv, track_stats: bool = False):
        """
        Initialize the wrapper.

        Args:
            env: The StrategyGameEnv to wrap
            track_stats: Whether to track action masking statistics
        """
        super().__init__(env)
        self.track_stats = track_stats

        if track_stats:
            self.stats = {
                "total_actions": 0,
                "masked_actions_attempted": 0,
                "action_type_distribution": np.zeros(10),
            }

        # From here on, writes to names the wrapper does not own are
        # forwarded to the wrapped env (see ``__setattr__``). Set through
        # ``object.__setattr__`` so the flag itself is not subject to the
        # forwarding rule it switches on.
        object.__setattr__(self, "_attrs_forwarded", True)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment.

        ``__getattr__`` only runs after normal lookup failed. During
        unpickling / ``copy.deepcopy`` the instance is created without
        running ``__init__``, so ``self.env`` does not exist yet and an
        unguarded ``getattr(self.env, name)`` re-enters ``__getattr__('env')``
        until RecursionError. Dunder lookups (``__setstate__``,
        ``__deepcopy__``, ...) are protocol probes on *this* object and must
        not be answered by the wrapped env either.
        """
        env = self.__dict__.get("env")
        if env is None or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward writes of the wrapped env's attributes; refuse unknown ones.

        ``gym.Wrapper`` forwards neither reads nor writes, and this class
        forwards reads (``__getattr__``). Without a matching write rule,
        ``masked_env.agent_player = 2`` would read back as 2 through the
        wrapper while the ``StrategyGameEnv`` that actually scores rewards,
        builds masks and executes actions kept player 1 -- exactly how
        SelfPlayEnv's seat swap used to be lost. So once construction is
        done:

        * names the wrapper itself owns (instance attributes set in
          ``__init__``, or class-level descriptors such as gym.Wrapper's
          ``action_space`` property) are set on the wrapper;
        * names that exist on the wrapped env are set there;
        * anything else raises instead of silently creating a wrapper-only
          attribute that the env never sees.
        """
        if not self.__dict__.get("_attrs_forwarded", False) or name in self.__dict__ or hasattr(type(self), name):
            super().__setattr__(name, value)
            return
        env = self.__dict__["env"]
        if hasattr(env, name):
            setattr(env, name, value)
            return
        raise AttributeError(
            f"Cannot set {name!r}: neither {type(self).__name__} nor the wrapped "
            f"{type(env).__name__} has that attribute. Set it on env.unwrapped explicitly "
            "if a new attribute on the base env is really intended."
        )

    def action_masks(self) -> np.ndarray:
        """
        Get action masks in the format expected by MaskablePPO.

        For MultiDiscrete spaces, returns a concatenated 1D boolean array
        of all dimension masks.

        Returns:
            Concatenated boolean mask array
        """
        masks = self.env.action_masks()
        # Concatenate all dimension masks into a single array
        return np.concatenate([m.astype(np.bool_) for m in masks])

    def get_action_masks_tuple(self):
        """
        Get action masks as a tuple of arrays (one per dimension).

        This is the native format from the environment.
        """
        return self.env.action_masks()

    def step(self, action):
        """Execute action and optionally track statistics."""
        if self.track_stats:
            self.stats["total_actions"] += 1
            # For MultiDiscrete, action is an array; for flat Discrete, it's an int
            if isinstance(action, np.ndarray) and action.ndim > 0:
                action_type = int(action[0])
                self.stats["action_type_distribution"][action_type] += 1

        return super().step(action)

    def reset(self, **kwargs):
        """Reset environment and optionally reset stats."""
        return super().reset(**kwargs)

    def get_masking_stats(self) -> dict[str, Any]:
        """Get action masking statistics."""
        if not self.track_stats:
            return {}

        stats = self.stats.copy()
        # Compute action type percentages
        total = stats["action_type_distribution"].sum()
        if total > 0:
            stats["action_type_percentages"] = (stats["action_type_distribution"] / total * 100).tolist()
        return stats


def _build_strategy_env(
    *,
    map_file: str | None,
    opponent: str,
    render_mode: str | None,
    max_steps: int,
    max_turns: int | None,
    reward_config: dict[str, float] | None,
    enabled_units: list[str] | None,
    action_space_type: str,
    max_flat_actions: int,
    max_actions_per_turn: int | None,
    opponent_kwargs: dict[str, Any] | None,
    gamma: float,
    pad_to_size: tuple[int, int] | None,
    engine_overrides: dict[str, Any] | None,
    gold_scale: float | None,
    turn_scale: float | None,
    unit_count_scale: float | None,
    fog_of_war: bool,
    flat_action_version: int | None,
) -> StrategyGameEnv:
    """Construct a ``StrategyGameEnv`` from the shared parameter set.

    The single place the env-construction kwargs are spelled out for the
    masking wrappers — ``make_maskable_env``, the per-rank vec-env builder
    and the self-play builders all delegate here, so a new env parameter is
    added exactly once (and every caller must pass it: the parameters are
    keyword-only with no defaults). Scale factors and the flat_discrete
    table version are forwarded only when explicitly set so the env's own
    defaults stay in charge otherwise.
    """
    optional_kwargs: dict[str, Any] = {}
    if gold_scale is not None:
        optional_kwargs["gold_scale"] = gold_scale
    if turn_scale is not None:
        optional_kwargs["turn_scale"] = turn_scale
    if unit_count_scale is not None:
        optional_kwargs["unit_count_scale"] = unit_count_scale
    if flat_action_version is not None:
        optional_kwargs["flat_action_version"] = flat_action_version

    return StrategyGameEnv(
        map_file=map_file,
        opponent=opponent,
        render_mode=render_mode,
        max_steps=max_steps,
        max_turns=max_turns,
        reward_config=reward_config,
        enabled_units=enabled_units,
        action_space_type=action_space_type,
        max_flat_actions=max_flat_actions,
        max_actions_per_turn=max_actions_per_turn,
        opponent_kwargs=opponent_kwargs,
        gamma=gamma,
        pad_to_size=pad_to_size,
        engine_overrides=engine_overrides,
        fog_of_war=fog_of_war,
        **optional_kwargs,
    )


def make_maskable_env(
    map_file: str | None = None,
    opponent: str = "bot",
    render_mode: str | None = None,
    max_steps: int = 200,
    max_turns: int | None = None,
    reward_config: dict[str, float] | None = None,
    track_stats: bool = False,
    enabled_units: list[str] | None = None,
    action_space_type: str = "multi_discrete",
    max_flat_actions: int = 512,
    max_actions_per_turn: int | None = None,
    seed: int | None = None,
    opponent_kwargs: dict[str, Any] | None = None,
    gamma: float = 0.99,
    pad_to_size: tuple[int, int] | None = None,
    gold_scale: float | None = None,
    turn_scale: float | None = None,
    unit_count_scale: float | None = None,
    engine_overrides: dict[str, Any] | None = None,
    fog_of_war: bool = False,
    flat_action_version: int | None = None,
) -> ActionMaskedEnv:
    """
    Create a single environment ready for use with MaskablePPO.

    Args:
        map_file: Path to map CSV file. None for random map.
        opponent: Opponent type ('bot', 'random', 'self', None)
        render_mode: 'human', 'rgb_array', or None
        max_steps: Maximum steps per episode
        reward_config: Custom reward configuration
        track_stats: Whether to track action masking statistics
        enabled_units: List of enabled unit types (default all)
        action_space_type: 'multi_discrete' (default) or 'flat_discrete'
        max_flat_actions: Max actions for flat_discrete mode (default 512)
        seed: Optional seed for reproducibility. When provided, the env's
            ``np_random`` (and the random opponent's RNG) are seeded so that
            episodes are deterministic across runs.
        fog_of_war: Partial observability for the agent (and the opponent's
            observation in self-play).
        flat_action_version: flat_discrete decode-table layout; ``None``
            keeps the env's default (``FLAT_ACTION_VERSION_LATEST``).

    Returns:
        ActionMaskedEnv ready for training

    Example:
        # MultiDiscrete (default, per-dimension masks):
        env = make_maskable_env(opponent="bot")

        # Flat Discrete (exact per-action masks, recommended):
        env = make_maskable_env(opponent="bot", action_space_type="flat_discrete")

        # Reproducible eval against a random opponent:
        env = make_maskable_env(opponent="random", seed=42)
    """
    env = _build_strategy_env(
        map_file=map_file,
        opponent=opponent,
        render_mode=render_mode,
        max_steps=max_steps,
        max_turns=max_turns,
        reward_config=reward_config,
        enabled_units=enabled_units,
        action_space_type=action_space_type,
        max_flat_actions=max_flat_actions,
        max_actions_per_turn=max_actions_per_turn,
        opponent_kwargs=opponent_kwargs,
        gamma=gamma,
        pad_to_size=pad_to_size,
        engine_overrides=engine_overrides,
        gold_scale=gold_scale,
        turn_scale=turn_scale,
        unit_count_scale=unit_count_scale,
        fog_of_war=fog_of_war,
        flat_action_version=flat_action_version,
    )
    if seed is not None:
        env.reset(seed=seed)
    return ActionMaskedEnv(env, track_stats=track_stats)


def _make_env_fn(
    rank: int,
    seed: int,
    *,
    map_file: str | None,
    opponent: str,
    max_steps: int,
    reward_config: dict[str, float] | None,
    enabled_units: list[str] | None = None,
    action_space_type: str = "multi_discrete",
    max_flat_actions: int = 512,
    max_turns: int | None = None,
    opponent_kwargs: dict[str, Any] | None = None,
    gamma: float = 0.99,
    pad_to_size: tuple[int, int] | None = None,
    gold_scale: float | None = None,
    turn_scale: float | None = None,
    unit_count_scale: float | None = None,
    max_actions_per_turn: int | None = None,
    engine_overrides: dict[str, Any] | None = None,
    fog_of_war: bool = False,
    flat_action_version: int | None = None,
) -> Callable[[], gym.Env]:
    """
    Create a function that creates an environment.

    Used for vectorized environment creation. Each sub-env is
    ``Monitor(ActionMaskedEnv(StrategyGameEnv))`` — the Monitor layer
    feeds SB3's ``ep_info_buffer`` (rollout/ep_rew_mean, ep_len_mean).

    Env parameters are keyword-only: this used to be called with 18
    positional arguments, where any parameter reordering silently
    mis-wired every sub-env.
    """

    def _init() -> gym.Env:
        from stable_baselines3.common.monitor import Monitor

        env = _build_strategy_env(
            map_file=map_file,
            opponent=opponent,
            render_mode=None,  # No rendering in vectorized envs
            max_steps=max_steps,
            max_turns=max_turns,
            reward_config=reward_config,
            enabled_units=enabled_units,
            action_space_type=action_space_type,
            max_flat_actions=max_flat_actions,
            max_actions_per_turn=max_actions_per_turn,
            opponent_kwargs=opponent_kwargs,
            gamma=gamma,
            pad_to_size=pad_to_size,
            engine_overrides=engine_overrides,
            gold_scale=gold_scale,
            turn_scale=turn_scale,
            unit_count_scale=unit_count_scale,
            fog_of_war=fog_of_war,
            flat_action_version=flat_action_version,
        )
        env.reset(seed=seed + rank)
        # Monitor must be the outer wrapper: it injects the ``episode``
        # key into ``info`` at episode end, which is the only source SB3
        # reads to fill ``model.ep_info_buffer``. Without it,
        # rollout/ep_rew_mean and ep_len_mean are silently absent from
        # every training log and chart (SB3 auto-wraps single envs in
        # Monitor, but hand-built VecEnvs like this one skip that path).
        # Mask lookups still work: ``env_method("action_masks")`` reaches
        # ActionMaskedEnv through Monitor's attribute forwarding — the
        # same chain sb3-contrib relies on for its own auto-wrapped envs.
        return Monitor(ActionMaskedEnv(env))

    return _init


def make_maskable_vec_env(
    n_envs: int = 4,
    map_file: str | None = None,
    opponent: str = "bot",
    max_steps: int = 200,
    max_turns: int | None = None,
    reward_config: dict[str, float] | None = None,
    seed: int = 0,
    use_subprocess: bool = True,
    enabled_units: list[str] | None = None,
    action_space_type: str = "multi_discrete",
    max_flat_actions: int = 512,
    max_actions_per_turn: int | None = None,
    opponent_kwargs: dict[str, Any] | None = None,
    gamma: float = 0.99,
    pad_to_size: tuple[int, int] | None = None,
    gold_scale: float | None = None,
    turn_scale: float | None = None,
    unit_count_scale: float | None = None,
    engine_overrides: dict[str, Any] | None = None,
    fog_of_war: bool = False,
    flat_action_version: int | None = None,
):
    """
    Create vectorized environments for parallel training with MaskablePPO.

    Uses sb3-contrib's vectorized environment wrappers that properly handle
    action masking across multiple environments.

    Args:
        n_envs: Number of parallel environments
        map_file: Path to map CSV file. None for random maps.
        opponent: Opponent type ('bot', 'random', 'self')
        max_steps: Maximum steps per episode
        reward_config: Custom reward configuration
        seed: Random seed (each env gets seed + rank)
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)
        enabled_units: List of enabled unit types (default all)
        action_space_type: 'multi_discrete' (default) or 'flat_discrete'
        max_flat_actions: Max actions for flat_discrete mode (default 512)

    Returns:
        Vectorized environment ready for MaskablePPO

    Example:
        vec_env = make_maskable_vec_env(n_envs=8, opponent="bot",
                                         action_space_type="flat_discrete")
        model = MaskablePPO("MultiInputPolicy", vec_env)
        model.learn(total_timesteps=1000000)
    """
    try:
        from sb3_contrib.common.maskable.utils import get_action_masks  # noqa: F401
        from sb3_contrib.common.wrappers import ActionMasker  # noqa: F401
    except ImportError:
        raise ImportError("sb3-contrib is required for action masking. Install it with: pip install sb3-contrib")

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        _make_env_fn(
            i,
            seed,
            map_file=map_file,
            opponent=opponent,
            max_steps=max_steps,
            reward_config=reward_config,
            enabled_units=enabled_units,
            action_space_type=action_space_type,
            max_flat_actions=max_flat_actions,
            max_turns=max_turns,
            opponent_kwargs=opponent_kwargs,
            gamma=gamma,
            pad_to_size=pad_to_size,
            gold_scale=gold_scale,
            turn_scale=turn_scale,
            unit_count_scale=unit_count_scale,
            max_actions_per_turn=max_actions_per_turn,
            engine_overrides=engine_overrides,
            fog_of_war=fog_of_war,
            flat_action_version=flat_action_version,
        )
        for i in range(n_envs)
    ]

    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    return vec_env


def validate_action_mask(env: StrategyGameEnv) -> dict[str, Any]:
    """
    Validate that action masks are correctly computed.

    Useful for debugging and testing.

    Args:
        env: Environment to validate

    Returns:
        Dict with validation results
    """
    masks = env.action_masks()
    legal_actions = env.game_state.get_legal_actions(player=env.game_state.current_player)

    results = {"valid": True, "errors": [], "warnings": [], "mask_summary": {}}

    # Check each mask dimension
    action_type_names = [
        "create",
        "move",
        "attack",
        "seize",
        "heal",
        "end_turn",
        "paralyze",
        "haste",
        "defence_buff",
        "attack_buff",
    ]

    for i, name in enumerate(action_type_names):
        mask_enabled = masks[0][i]
        has_actions = bool(legal_actions.get(name, []))

        # Special case: end_turn is always valid
        if name == "end_turn":
            has_actions = True

        # Special case: heal and cure share action type 4
        if name == "heal":
            has_actions = bool(legal_actions.get("heal", [])) or bool(legal_actions.get("cure", []))

        results["mask_summary"][name] = {"mask_enabled": mask_enabled, "has_legal_actions": has_actions}

        if has_actions and not mask_enabled:
            results["valid"] = False
            results["errors"].append(f"Action type '{name}' has legal actions but mask is False")

    # Check that at least one action is always available (end_turn should always work)
    if not masks[0].any():
        results["valid"] = False
        results["errors"].append("No action types are masked as valid")

    # Check position masks have at least one valid option
    for i, dim_name in enumerate(["action_type", "unit_type", "from_x", "from_y", "to_x", "to_y"]):
        if not masks[i].any():
            results["warnings"].append(f"Dimension '{dim_name}' has no valid values in mask")

    return results
