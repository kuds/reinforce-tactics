"""
Evaluation script for trained agents.
"""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

from stable_baselines3 import PPO
from reinforcetactics.rl.gym_env import StrategyGameEnv


def evaluate_agent(
    model_path: str,
    n_episodes: int = 100,
    opponent: str = 'bot',
    render: bool = False,
    save_replays: bool = False
):
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to trained model
        n_episodes: Number of episodes to evaluate
        opponent: Opponent type
        render: Whether to render
        save_replays: Whether to save game replays

    Returns:
        Dict with evaluation statistics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Agent: {Path(model_path).name}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model = PPO.load(model_path)

    # Create environment
    env = StrategyGameEnv(
        opponent=opponent,
        render_mode='human' if render else None
    )

    # Evaluation statistics
    wins = 0
    losses = 0
    total_rewards = []
    episode_lengths = []
    invalid_actions = []

    # Run episodes
    print(f"Running {n_episodes} evaluation episodes...")
    for ep in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_invalid = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if not info.get('valid_action', True):
                episode_invalid += 1

            done = terminated or truncated

            if render:
                env.render()

        # Record statistics
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        invalid_actions.append(episode_invalid)

        if info.get('winner') == 1:
            wins += 1
        else:
            losses += 1

        # Save replay if requested
        if save_replays:
            replay_dir = Path("replays") / Path(model_path).stem
            replay_dir.mkdir(parents=True, exist_ok=True)
            # TODO: Implement replay saving

    env.close()

    # Compute statistics
    win_rate = wins / n_episodes
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_length = np.mean(episode_lengths)
    avg_invalid = np.mean(invalid_actions)

    # Print results
    print(f"\n{'='*60}")
    print("📊 Evaluation Results")
    print(f"{'='*60}")
    print(f"Episodes:        {n_episodes}")
    print(f"Win Rate:        {win_rate*100:.1f}% ({wins}/{n_episodes})")
    print(f"Avg Reward:      {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Avg Length:      {avg_length:.1f} steps")
    print(f"Avg Invalid:     {avg_invalid:.1f} per episode")
    print(f"{'='*60}\n")

    results = {
        'model_path': model_path,
        'n_episodes': n_episodes,
        'opponent': opponent,
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_length': avg_length,
        'avg_invalid': avg_invalid,
        'rewards': total_rewards,
        'lengths': episode_lengths,
        'invalid_actions': invalid_actions
    }

    return results


def compare_agents(model_paths: list, n_episodes: int = 100, opponent: str = 'bot'):
    """Compare multiple agents."""
    print(f"\n{'='*60}")
    print(f"Comparing {len(model_paths)} Agents")
    print(f"{'='*60}\n")

    results = []
    for model_path in model_paths:
        result = evaluate_agent(model_path, n_episodes, opponent, render=False)
        results.append(result)

    # Create comparison table
    df = pd.DataFrame([
        {
            'Model': Path(r['model_path']).stem,
            'Win Rate': f"{r['win_rate']*100:.1f}%",
            'Avg Reward': f"{r['avg_reward']:.2f}",
            'Avg Length': f"{r['avg_length']:.1f}",
            'Avg Invalid': f"{r['avg_invalid']:.1f}"
        }
        for r in results
    ])

    print("\n" + "="*60)
    print("📊 Comparison Results")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (or directory for comparison)')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--opponent', type=str, default='bot',
                       choices=['bot', 'random'],
                       help='Opponent type')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--save-replays', action='store_true',
                       help='Save game replays')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models in directory')

    args = parser.parse_args()

    if args.compare:
        # Compare all models in directory
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"Error: {model_dir} is not a directory")
            return

        model_paths = list(model_dir.glob("*.zip"))
        if not model_paths:
            print(f"No .zip models found in {model_dir}")
            return

        compare_agents(model_paths, args.n_episodes, args.opponent)
    else:
        # Evaluate single model
        evaluate_agent(
            args.model,
            args.n_episodes,
            args.opponent,
            args.render,
            args.save_replays
        )


if __name__ == '__main__':
    main()
