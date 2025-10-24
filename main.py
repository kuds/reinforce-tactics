"""
Reinforce Tactics - Main Entry Point
Updated to work with current reinforcetactics folder structure

Requirements:
pip install pygame numpy gymnasium stable-baselines3 tensorboard pandas

Usage:
    # Train with Stable-Baselines3 (PPO)
    python main.py --mode train --algorithm ppo --timesteps 100000

    # Evaluate trained agent
    python main.py --mode evaluate --model models/ppo_model.zip --episodes 10

    # Play manually (GUI mode)
    python main.py --mode play

    # View training stats
    python main.py --mode stats
"""

import argparse
import os
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import pygame
    except ImportError:
        missing.append("pygame")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True

def train_mode(args):
    """Training mode for RL agents."""
    try:
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback
        import os
    except ImportError:
        print("❌ Stable-Baselines3 not installed.")
        print("Install with: pip install stable-baselines3[extra]")
        return
    
    # Import from correct paths based on actual file structure
    try:
        # Add the project root to path if needed
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import RL environment - check if it exists
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
    print(f"Tensorboard: tensorboard --logdir ./tensorboard/\n")
    
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
        sys.path.insert(0, str(Path(__file__).parent))
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
    print(f"📊 Evaluation Results:")
    print(f"Win Rate:     {win_rate*100:.1f}%")
    print(f"Wins:         {wins}/{args.episodes}")
    print(f"Avg Reward:   {avg_reward:.2f}")
    print(f"Best Reward:  {max(total_rewards):.2f}")
    print(f"Worst Reward: {min(total_rewards):.2f}")
    print(f"{'='*60}\n")
    
    env.close()

def play_mode(args):
    """Interactive play mode with GUI."""
    print("\n🎮 Starting Interactive Play Mode...\n")
    
    try:
        import pygame
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import game components
        from reinforcetactics.core.game_state import GameState
        from reinforcetactics.ui.renderer import Renderer
        from reinforcetactics.ui.menus import MainMenu, MapSelectionMenu
        from reinforcetactics.utils.file_io import FileIO
        from reinforcetactics.game.bot import SimpleBot
    except ImportError as e:
        print(f"❌ Error importing game components: {e}")
        print("\nMake sure all required modules are in reinforcetactics/")
        return
    
    # Initialize Pygame
    pygame.init()
    
    # Show main menu
    main_menu = MainMenu()
    menu_result = main_menu.run()
    
    if not menu_result or menu_result['type'] == 'exit':
        print("Exiting...")
        pygame.quit()
        return
    
    # Handle menu selection
    if menu_result['type'] == 'new_game':
        start_new_game(menu_result.get('mode', 'human_vs_computer'))
    elif menu_result['type'] == 'load_game':
        load_saved_game()
    elif menu_result['type'] == 'watch_replay':
        watch_replay()
    elif menu_result['type'] == 'settings':
        print("Settings menu not yet implemented")
        pygame.quit()

def start_new_game(mode='human_vs_computer'):
    """Start a new game with the specified mode."""
    import pygame
    from pathlib import Path
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.ui.renderer import Renderer
    from reinforcetactics.ui.menus import (
        MapSelectionMenu, SaveGameMenu,
        BuildingMenu
    )
    from reinforcetactics.utils.file_io import FileIO
    from reinforcetactics.game.bot import SimpleBot
    from reinforcetactics.constants import TILE_SIZE

    print(f"\n🎮 Starting new game: {mode}\n")

    # Show map selection
    map_menu = MapSelectionMenu()
    selected_map = map_menu.run()

    if not selected_map:
        print("Map selection cancelled")
        return

    try:
        # Load or generate map
        if selected_map == "random":
            print("Generating random map...")
            map_data = FileIO.generate_random_map(20, 20, num_players=2)
            map_file_used = None
        else:
            print(f"Loading map: {selected_map}")
            map_data = FileIO.load_map(selected_map)
            map_file_used = selected_map

        if map_data is None:
            print("Failed to load map")
            return

        # Create game state
        game = GameState(map_data, num_players=2)

        # Store map file for saving
        if map_file_used:
            game.map_file_used = map_file_used

        # Create renderer
        renderer = Renderer(game)

        # Create bot if needed
        bot = None
        if mode == 'human_vs_computer':
            bot = SimpleBot(game, player=2)
            print("Bot created for Player 2")

        # Game loop variables
        clock = pygame.time.Clock()
        running = True
        selected_unit = None
        active_menu = None
        menu_opened_time = 0  # Track when menu was opened (in milliseconds)

        print("\n🎮 Game started!")
        print("Controls:")
        print("  - Click units to select")
        print("  - Click buildings to create units")
        print("  - Click tiles to move")
        print("  - Press SPACE to end turn")
        print("  - Press S to save game")
        print("  - Press ESC to quit")
        print()

        while running and not game.game_over:
            # Get mouse position ONCE per frame
            mouse_pos = pygame.mouse.get_pos()
            current_time = pygame.time.get_ticks()  # Get current time in milliseconds

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if active_menu:
                            # Close menu with ESC
                            active_menu = None
                        else:
                            running = False

                    elif event.key == pygame.K_s and not active_menu:
                        # Save game
                        save_menu = SaveGameMenu(game)
                        result = save_menu.run()
                        if result:
                            print(f"✅ Game saved to {result}")

                    elif event.key == pygame.K_SPACE and not active_menu:
                        # End turn
                        print(f"\nPlayer {game.current_player} ended turn")
                        selected_unit = None
                        game.end_turn()

                        # Bot plays if it's player 2's turn and bot exists
                        if bot and game.current_player == 2:
                            print("Bot is thinking...")
                            bot.take_turn()
                            game.end_turn()
                            print(f"Bot finished. Player {game.current_player}'s turn\n")

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # CRITICAL: Check if clicking on a menu FIRST
                        if active_menu:
                            # Ignore clicks for 200ms after menu opens
                            # This prevents the same click that opened the menu from closing it
                            if current_time - menu_opened_time < 200:
                                continue

                            menu_result = active_menu.handle_click(mouse_pos)
                            if menu_result:
                                if menu_result['type'] == 'close':
                                    active_menu = None
                                elif menu_result['type'] == 'create_unit':
                                    # Create unit at building
                                    unit_type = menu_result['unit_type']
                                    building_pos = menu_result['building_pos']
                                    unit = game.create_unit(unit_type, building_pos[0], building_pos[1])
                                    if unit:
                                        print(f"Created {unit_type} at {building_pos}")
                                    else:
                                        print(f"Failed to create {unit_type}")
                                    active_menu = None
                                elif menu_result['type'] == 'unit_action':
                                    # Handle unit action
                                    action = menu_result['action']
                                    if action == 'wait':
                                        selected_unit.end_unit_turn()
                                        selected_unit = None
                                    active_menu = None
                            # Continue to prevent further processing
                            continue

                        # Check if clicking on UI buttons
                        if renderer.end_turn_button.collidepoint(mouse_pos):
                            print(f"\nPlayer {game.current_player} ended turn")
                            selected_unit = None
                            game.end_turn()

                            if bot and game.current_player == 2:
                                print("Bot is thinking...")
                                bot.take_turn()
                                game.end_turn()
                                print(f"Bot finished. Player {game.current_player}'s turn\n")
                            continue

                        if renderer.resign_button.collidepoint(mouse_pos):
                            print(f"\nPlayer {game.current_player} resigned")
                            game.resign()
                            continue

                        # Convert mouse position to grid coordinates
                        grid_x = mouse_pos[0] // TILE_SIZE
                        grid_y = mouse_pos[1] // TILE_SIZE

                        # Check bounds
                        if not (0 <= grid_x < game.grid.width and 0 <= grid_y < game.grid.height):
                            continue

                        # Check what was clicked
                        clicked_unit = game.get_unit_at_position(grid_x, grid_y)
                        clicked_tile = game.grid.get_tile(grid_x, grid_y)

                        # Priority 1: Own unit clicked
                        if clicked_unit and clicked_unit.player == game.current_player:
                            selected_unit = clicked_unit
                            print(f"Selected {clicked_unit.type} at ({grid_x}, {grid_y})")
                            continue  # Stop processing more events this frame

                        # Priority 2: Own building clicked (for creating units)
                        elif (clicked_tile and clicked_tile.type == 'b' and
                              clicked_tile.player == game.current_player and
                              not clicked_unit):  # Only if no unit on it
                            active_menu = BuildingMenu(game, (grid_x, grid_y))
                            menu_opened_time = current_time  # Record when menu opened
                            print(f"Opened building menu at ({grid_x}, {grid_y})")
                            continue  # CRITICAL: Stop processing remaining events!

                        # Priority 3: Movement with selected unit
                        elif selected_unit and selected_unit.can_move:
                            if game.move_unit(selected_unit, grid_x, grid_y):
                                print(f"Moved {selected_unit.type} to ({grid_x}, {grid_y})")
                                selected_unit = None
                            continue  # Stop processing more events this frame

                        # Priority 4: Deselect
                        else:
                            selected_unit = None
                            continue  # Stop processing more events this frame

            # === RENDERING SECTION ===
            # Render game state (this clears and redraws everything)
            renderer.render()

            # Draw movement overlay if unit selected
            if selected_unit:
                renderer.draw_movement_overlay(selected_unit)

            # Draw active menu LAST (on top of everything)
            if active_menu:
                active_menu.draw(renderer.screen)

            # Flip display ONCE per frame
            pygame.display.flip()
            clock.tick(60)

        # Game over
        if game.game_over:
            print(f"\n🎉 Game Over! Player {game.winner} wins!")

            # Automatically save replay
            replay_path = game.save_replay_to_file()
            if replay_path:
                print(f"📼 Replay saved to {replay_path}")

        pygame.quit()

    except Exception as e:
        print(f"❌ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()

def load_saved_game():
    """Load a saved game."""
    import pygame
    from pathlib import Path
    from reinforcetactics.core.game_state import GameState
    from reinforcetactics.ui.renderer import Renderer
    from reinforcetactics.ui.menus import LoadGameMenu, SaveGameMenu
    from reinforcetactics.utils.file_io import FileIO
    from reinforcetactics.game.bot import SimpleBot
    from reinforcetactics.constants import TILE_SIZE
    
    print("\n💾 Loading saved game...\n")
    
    # Show load menu
    load_menu = LoadGameMenu()
    save_data = load_menu.run()
    
    if not save_data:
        print("Load cancelled")
        return
    
    try:
        # Get the map file that was used
        # For now, we'll need to regenerate or store it
        # In the save data, we should have stored the initial map
        
        # Load map (you may need to save map info in save_data)
        if 'map_file' in save_data:
            map_data = FileIO.load_map(save_data['map_file'])
        else:
            # Try to reconstruct from tiles
            print("⚠️  Map file not in save, reconstructing from tiles...")
            # This is a simplified reconstruction
            # You should save map_file path in save_data for better results
            map_data = FileIO.generate_random_map(20, 20, num_players=save_data.get('num_players', 2))
        
        # Restore game state
        game = GameState.from_dict(save_data, map_data)
        
        # Create renderer
        renderer = Renderer(game)
        
        # Create bot for player 2 if needed
        bot = SimpleBot(game, player=2) if game.num_players == 2 else None
        
        # Game loop
        clock = pygame.time.Clock()
        running = True
        selected_unit = None
        
        print(f"\n✅ Game loaded! Turn {game.turn_number}, Player {game.current_player}'s turn")
        print("\nControls:")
        print("  - Click units to select")
        print("  - Click tiles to move")
        print("  - Press SPACE to end turn")
        print("  - Press S to save game")
        print("  - Press ESC to quit")
        print()
        
        while running and not game.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Auto-save before exit
                    print("\nAuto-saving before exit...")
                    game.save_to_file()
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # Ask if want to save
                        print("\nAuto-saving before exit...")
                        game.save_to_file()
                        running = False
                    
                    elif event.key == pygame.K_s:
                        # Save game
                        save_menu = SaveGameMenu(game)
                        result = save_menu.run()
                        if result:
                            print(f"✅ Game saved to {result}")
                    
                    elif event.key == pygame.K_SPACE:
                        # End turn
                        print(f"\nPlayer {game.current_player} ended turn")
                        game.end_turn()
                        
                        # Bot plays if it's player 2's turn and bot exists
                        if bot and game.current_player == 2:
                            print("Bot is thinking...")
                            bot.take_turn()
                            game.end_turn()
                            print(f"Bot finished. Player {game.current_player}'s turn\n")
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        grid_x = mouse_pos[0] // TILE_SIZE
                        grid_y = mouse_pos[1] // TILE_SIZE
                        
                        # Check if clicked on unit
                        clicked_unit = game.get_unit_at_position(grid_x, grid_y)
                        
                        if clicked_unit and clicked_unit.player == game.current_player:
                            selected_unit = clicked_unit
                            print(f"Selected {clicked_unit.type} at ({grid_x}, {grid_y})")
                        elif selected_unit and selected_unit.can_move:
                            # Try to move selected unit
                            if game.move_unit(selected_unit, grid_x, grid_y):
                                print(f"Moved {selected_unit.type} to ({grid_x}, {grid_y})")
                                selected_unit = None
            
            # Render
            renderer.render()
            
            # Show movement overlay for selected unit
            if selected_unit:
                renderer.draw_movement_overlay(selected_unit)
            
            pygame.display.flip()
            clock.tick(60)
        
        # Game over
        if game.game_over:
            print(f"\n🎉 Game Over! Player {game.winner} wins!")
            # Save replay
            replay_path = game.save_replay_to_file()
            if replay_path:
                print(f"📼 Replay saved to {replay_path}")
        
        pygame.quit()
        
    except Exception as e:
        print(f"❌ Error loading game: {e}")
        import traceback
        traceback.print_exc()

def watch_replay():
    """Watch a replay."""
    import pygame
    from pathlib import Path
    from reinforcetactics.ui.menus import ReplaySelectionMenu
    from reinforcetactics.utils.file_io import FileIO
    from reinforcetactics.utils.replay_player import ReplayPlayer
    
    print("\n📼 Loading replay...\n")
    
    # Show replay selection menu
    replay_menu = ReplaySelectionMenu()
    replay_path = replay_menu.run()
    
    if not replay_path:
        print("Replay selection cancelled")
        return
    
    try:
        # Load replay data
        replay_data = FileIO.load_replay(replay_path)
        
        if not replay_data:
            print("Failed to load replay")
            return
        
        # Load or generate initial map
        # Ideally, the replay should store the initial map configuration
        game_info = replay_data.get('game_info', {})
        
        # For now, generate a map based on saved info
        # TODO: Store and load actual map used in replay
        map_data = FileIO.generate_random_map(20, 20, 
                                              num_players=game_info.get('num_players', 2))
        
        # Create replay player
        player = ReplayPlayer(replay_data, map_data)
        
        # Run replay
        player.run()
        
        pygame.quit()
        
    except Exception as e:
        print(f"❌ Error playing replay: {e}")
        import traceback
        traceback.print_exc()

def stats_mode(args):
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

def main():
    """Main entry point."""
    # Initialize settings first
    from reinforcetactics.utils.settings import get_settings
    from reinforcetactics.utils.language import get_language
    
    settings = get_settings()
    lang = get_language()
    
    # Ensure all directories exist
    settings.ensure_directories()
    
    parser = argparse.ArgumentParser(
        description="Reinforce Tactics - Turn-Based Strategy with RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO agent
  python main.py --mode train --algorithm ppo --timesteps 100000
  
  # Evaluate trained model
  python main.py --mode evaluate --model models/ppo_final.zip --episodes 20
  
  # Play manually
  python main.py --mode play
  
  # View stats
  python main.py --mode stats
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="play",
        choices=["train", "evaluate", "play", "stats"],
        help="Mode: train, evaluate, play, or stats"
    )
    
    # Training arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "a2c", "dqn"],
        help="RL algorithm for training"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="bot",
        choices=["bot", "random", "self"],
        help="Opponent type"
    )
    parser.add_argument(
        "--map-file",
        type=str,
        default=None,
        help="Path to map file (None for random)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name for saved model"
    )
    
    # Reward shaping
    parser.add_argument(
        "--reward-income",
        type=float,
        default=0.0,
        help="Reward coefficient for income difference"
    )
    parser.add_argument(
        "--reward-units",
        type=float,
        default=0.0,
        help="Reward coefficient for unit advantage"
    )
    parser.add_argument(
        "--reward-structures",
        type=float,
        default=0.0,
        help="Reward coefficient for structure control"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model for evaluation"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Route to appropriate mode
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "evaluate":
        evaluate_mode(args)
    elif args.mode == "play":
        play_mode(args)
    elif args.mode == "stats":
        stats_mode(args)
    
    print("\n✅ Done!\n")

if __name__ == "__main__":
    main()