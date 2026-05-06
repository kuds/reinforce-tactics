"""
Evaluation utilities for trained RL agents.

Provides a reusable evaluation function that works with both MaskablePPO
and standard PPO models across all environment configurations.

Usage:
    from reinforcetactics.rl.evaluation import evaluate_model

    results = evaluate_model(model, env, n_episodes=50)
    print(f"Win rate: {results['win_rate']:.1%}")
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


# Action types are emitted as integers in info["action_type"]; this list maps
# them to human-readable names. Matches the encoding in
# StrategyGameEnv._encode_action (gym_env.py).
ACTION_TYPE_NAMES = (
    "create_unit",
    "move",
    "attack",
    "seize",
    "heal",
    "end_turn",
    "paralyze",
    "haste",
    "defence_buff",
    "attack_buff",
)

# Reward components emitted in info["reward_breakdown"] by StrategyGameEnv.
REWARD_COMPONENTS = ("action", "shaping_delta", "invalid_penalty", "terminal")


def evaluate_model(
    model: Any,
    env: Any,
    n_episodes: int = 50,
    deterministic: bool = True,
    seed: Any = None,
    track_breakdown: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a trained model and return summary statistics.

    Works with both MaskablePPO (using action_masks) and standard PPO.
    The environment can be a raw gym env or an ActionMaskedEnv wrapper.

    Args:
        model: Trained SB3 model (PPO, MaskablePPO, etc.)
        env: Gymnasium environment (single, not vectorized).
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic actions (recommended for eval).
        seed: Optional integer seed. When provided, episode ``i`` is reset
            with ``seed + i`` so results are reproducible across runs.
        track_breakdown: When True, also accumulate per-step
            ``info["action_type"]`` counts and ``info["reward_breakdown"]``
            sums across the evaluation, returned under ``action_counts``
            and ``reward_components``. Disabled by default to keep the hot
            path lean; the per-step ``info`` dict is read either way.

    Returns:
        Dict with keys: win_rate, avg_reward, std_reward, avg_length,
        std_length, wins, losses, draws, episodes, rewards, lengths.

        ``rewards`` and ``lengths`` are the raw per-episode arrays (lists
        of plain floats / ints), exposed so callers can plot full
        distributions rather than only the mean ± std summary.

        When ``track_breakdown=True`` the dict also includes
        ``action_counts`` (dict keyed by ACTION_TYPE_NAMES, summed over
        every step of every episode) and ``reward_components`` (dict
        keyed by REWARD_COMPONENTS, summed analogously).
    """
    wins, losses, draws = 0, 0, 0
    rewards = []
    lengths = []
    has_action_masks = hasattr(env, "action_masks")

    if track_breakdown:
        action_counts = {name: 0 for name in ACTION_TYPE_NAMES}
        reward_components = {name: 0.0 for name in REWARD_COMPONENTS}

    for ep_idx in range(n_episodes):
        if seed is not None:
            obs, _ = env.reset(seed=int(seed) + ep_idx)
        else:
            obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            predict_kwargs: Dict[str, Any] = {"deterministic": deterministic}
            if has_action_masks:
                masks = env.action_masks()
                # MaskablePPO expects a flat concatenated mask
                if isinstance(masks, tuple):
                    masks = np.concatenate([m.astype(np.bool_) for m in masks])
                predict_kwargs["action_masks"] = masks

            action, _ = model.predict(obs, **predict_kwargs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated

            if track_breakdown:
                at = info.get("action_type")
                if isinstance(at, (int, np.integer)) and 0 <= int(at) < len(ACTION_TYPE_NAMES):
                    action_counts[ACTION_TYPE_NAMES[int(at)]] += 1
                for k, v in info.get("reward_breakdown", {}).items():
                    if k in reward_components:
                        reward_components[k] += float(v)

        rewards.append(ep_reward)
        lengths.append(ep_len)

        # Determine outcome from info
        episode_stats = info.get("episode_stats", {})
        winner = episode_stats.get("winner", info.get("winner"))
        agent_player = getattr(env, "agent_player", None)
        if agent_player is None:
            agent_player = getattr(getattr(env, "unwrapped", None), "agent_player", 1)

        if winner == agent_player:
            wins += 1
        elif winner is not None:
            losses += 1
        else:
            draws += 1

    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)

    result: Dict[str, Any] = {
        "win_rate": wins / n_episodes if n_episodes > 0 else 0.0,
        "avg_reward": float(rewards_arr.mean()) if n_episodes > 0 else 0.0,
        "std_reward": float(rewards_arr.std()) if n_episodes > 0 else 0.0,
        "avg_length": float(lengths_arr.mean()) if n_episodes > 0 else 0.0,
        "std_length": float(lengths_arr.std()) if n_episodes > 0 else 0.0,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "episodes": n_episodes,
        "rewards": [float(r) for r in rewards],
        "lengths": [int(length) for length in lengths],
    }
    if track_breakdown:
        result["action_counts"] = action_counts
        result["reward_components"] = reward_components
    return result
