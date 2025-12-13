"""
CLI Commands for Reinforce Tactics.

This module contains the command implementations for different modes:
- train: RL agent training
- evaluate: Model evaluation
- play: Interactive gameplay
- stats: Display training statistics
"""

import sys
from pathlib import Path


def train_mode(args):
    """Training mode for RL agents."""
    try:
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError:
        print("❌ Stable-Baselines3 not installed.")
        print("Install with: pip install stable-baselines3[extra]")
        return

    # Import from correct paths
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))

        try:
            from reinforcetactics.rl.rl_gym_env import StrategyGameEnv
        except ImportError:
            try:
                from reinforcetactics.rl.gym_env import StrategyGameEnv
            except ImportError:
                print("❌ Could not import StrategyGameEnv")
                print("Please ensure reinforcetactics/rl/gym_env.py exists")
                print("\nSearching for RL environment files...")
                rl_dir = Path("reinforcetactics/rl")
                if rl_dir.exists():
                    print(f"Files in {rl_dir}:")
                    for f in rl_dir.glob("*.py"):
                        print(f"  - {f.name}")
                return
    except Exception as e:
        print(f"❌ Error importing modules: {e}")
        return

    print(f"\n{'='*60}")
    print(f"🚀 Training {args.algorithm.upper()} Agent")
    print(f"{'='*60}\n")

    # Create environment
    print("Creating environment...")
    try:
        env = StrategyGameEnv(
            map_file=args.map_file,
            opponent=args.opponent,
            render_mode=None,  # Headless for training
            reward_config={
                'win': 1000.0,
                'loss': -1000.0,
                'income_diff': args.reward_income,
                'unit_diff': args.reward_units,
                'structure_control': args.reward_structures,
                'invalid_action': -10.0
            }
        )
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    env = Monitor(env)

    # Create model
    print(f"Creating {args.algorithm.upper()} model...")

    if args.algorithm.lower() == 'ppo':
        model = PPO(
            'MultiInputPolicy',
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10
        )
    elif args.algorithm.lower() == 'a2c':
        model = A2C(
            'MultiInputPolicy',
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            learning_rate=3e-4,
            n_steps=5
        )
    elif args.algorithm.lower() == 'dqn':
        model = DQN(
            'MultiInputPolicy',
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            learning_rate=1e-4,
            buffer_size=50000
        )
    else:
        print(f"❌ Unknown algorithm: {args.algorithm}")
        return

    # Setup callbacks
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix=f"{args.algorithm}_strategy"
    )

    # Train
    print(f"🎮 Training for {args.timesteps} timesteps...")
    print(f"Opponent: {args.opponent}")
    print("Tensorboard: tensorboard --logdir ./tensorboard/\n")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Save final model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_name = args.model_name or f"{args.algorithm}_final"
    model_path = models_dir / model_name

    model.save(str(model_path))
    print(f"\n✅ Model saved to {model_path}.zip")

    env.close()


def evaluate_mode(args):
    """Evaluation mode for trained agents."""
    try:
        from stable_baselines3 import PPO, A2C, DQN
        import numpy as np
    except ImportError:
        print("❌ Stable-Baselines3 not installed.")
        return

    # Import environment
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        try:
            from reinforcetactics.rl.rl_gym_env import StrategyGameEnv
        except ImportError:
            from reinforcetactics.rl.gym_env import StrategyGameEnv
    except ImportError:
        print("❌ Could not import StrategyGameEnv")
        return

    if not args.model:
        print("❌ --model path required for evaluation")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"\n{'='*60}")
    print(f"📊 Evaluating Model: {model_path.name}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    try:
        if 'ppo' in str(model_path).lower():
            model = PPO.load(str(model_path))
        elif 'a2c' in str(model_path).lower():
            model = A2C.load(str(model_path))
        elif 'dqn' in str(model_path).lower():
            model = DQN.load(str(model_path))
        else:
            print("⚠️  Could not detect algorithm, trying PPO...")
            model = PPO.load(str(model_path))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create environment
    try:
        env = StrategyGameEnv(
            opponent=args.opponent,
            render_mode='human' if args.render else None
        )
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return

    # Evaluate
    wins = 0
    total_rewards = []

    print(f"Running {args.episodes} evaluation episodes...\n")

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

            if args.render:
                env.render()

        total_rewards.append(episode_reward)

        if info.get('game_over') and info.get('winner') == 1:
            wins += 1
            print(f"Episode {ep+1}: WIN  | Reward: {episode_reward:.1f} | Steps: {steps}")
        else:
            print(f"Episode {ep+1}: LOSS | Reward: {episode_reward:.1f} | Steps: {steps}")

    # Print results
    win_rate = wins / args.episodes
    avg_reward = np.mean(total_rewards)

    print(f"\n{'='*60}")
    print("📊 Evaluation Results:")
    print(f"Win Rate:     {win_rate*100:.1f}%")
    print(f"Wins:         {wins}/{args.episodes}")
    print(f"Avg Reward:   {avg_reward:.2f}")
    print(f"Best Reward:  {max(total_rewards):.2f}")
    print(f"Worst Reward: {min(total_rewards):.2f}")
    print(f"{'='*60}\n")

    env.close()


def play_mode(_args):
    """Interactive play mode with GUI."""
    print("\n🎮 Starting Interactive Play Mode...\n")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Import game components to validate they exist
        import pygame
        from reinforcetactics.ui.menus import MainMenu
        from game.game_loop import start_new_game, load_saved_game, watch_replay
    except ImportError as e:
        print(f"❌ Error importing game components: {e}")
        print("\nMake sure all required modules are in reinforcetactics/")
        return

    # Main game loop - keep showing menu until user quits
    while True:
        # Initialize/reinitialize Pygame (needed after game sessions quit pygame)
        pygame.init()

        # Show main menu
        main_menu = MainMenu()
        menu_result = main_menu.run()

        if not menu_result or menu_result['type'] == 'exit':
            print("Exiting...")
            pygame.quit()
            return

        # Handle menu selection
        game_result = None
        if menu_result['type'] == 'new_game':
            game_result = start_new_game(
                mode=menu_result.get('mode', 'human_vs_computer'),
                selected_map=menu_result.get('map'),
                player_configs=menu_result.get('players')
            )
        elif menu_result['type'] == 'load_game':
            game_result = load_saved_game()
        elif menu_result['type'] == 'watch_replay':
            game_result = watch_replay(menu_result.get('replay_path'))

        # Handle game result
        # - 'main_menu': Continue loop to show menu again
        # - 'new_game': Continue loop to show menu again (user can start new game from menu)
        # - 'quit' or None: Exit completely
        if game_result == 'quit':
            print("Exiting...")
            return
        # For 'main_menu' or 'new_game', continue to next iteration


def stats_mode(_args):
    """Display training statistics."""
    print("\n📊 Training Statistics\n")

    # Check for saved models
    models_dir = Path("models")
    if models_dir.exists():
        models = list(models_dir.glob("*.zip"))
        print(f"Saved models: {len(models)}")
        for model in models[:10]:  # Show first 10
            print(f"  - {model.name}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
    else:
        print("No saved models found")

    print()

    # Check for checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        print(f"Checkpoints: {len(checkpoints)}")
    else:
        print("No checkpoints found")

    print()

    # Check for tensorboard logs
    tb_dir = Path("tensorboard")
    if tb_dir.exists():
        print("Tensorboard logs found")
        print("View with: tensorboard --logdir ./tensorboard/")
    else:
        print("No tensorboard logs found")

    print()
