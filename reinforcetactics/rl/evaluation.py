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


def evaluate_model(
    model: Any,
    env: Any,
    n_episodes: int = 50,
    deterministic: bool = True,
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

    Returns:
        Dict with keys: win_rate, avg_reward, std_reward, avg_length,
        std_length, wins, losses, draws, episodes.
    """
    wins, losses, draws = 0, 0, 0
    rewards = []
    lengths = []
    has_action_masks = hasattr(env, "action_masks")

    for _ in range(n_episodes):
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

    return {
        "win_rate": wins / n_episodes if n_episodes > 0 else 0.0,
        "avg_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "avg_length": float(lengths_arr.mean()),
        "std_length": float(lengths_arr.std()),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "episodes": n_episodes,
    }
