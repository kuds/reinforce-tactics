"""
Game Loop and Session Management for Reinforce Tactics.

This module manages the main game loop, game session, and game modes.
"""
# pylint: disable=cyclic-import

import pygame
import pandas as pd
from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui.renderer import Renderer
from reinforcetactics.ui.menus import (
    MapSelectionMenu, SaveGameMenu, GameOverMenu, LoadGameMenu, ReplaySelectionMenu, PauseMenu
)
from reinforcetactics.ui.menus.settings.settings_menu import SettingsMenu
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.settings import get_settings
from reinforcetactics.utils.replay_player import ReplayPlayer
from game.bot_factory import create_bots_from_config
from game.input_handler import InputHandler


class GameSession:  # pylint: disable=too-few-public-methods
    """
    Manages a game session including initialization and game loop.

    Attributes:
        game: The GameState instance
        renderer: The Renderer instance
        bots: Dictionary mapping player numbers to bot instances
        input_handler: InputHandler instance
        clock: pygame.Clock for frame timing
        running: Whether the game loop is running
    """

    def __init__(self, game, renderer, bots, num_players):
        """
        Initialize a GameSession.

        Args:
            game: The GameState instance
            renderer: The Renderer instance
            bots: Dictionary mapping player numbers to bot instances
            num_players: Total number of players
        """
        self.game = game
        self.renderer = renderer
        self.bots = bots
        self.input_handler = InputHandler(game, renderer, bots, num_players)
        self.clock = pygame.time.Clock()
        self.running = True

    def run(self):
        """
        Run the main game loop.

        Returns:
            'new_game', 'main_menu', or 'quit' based on game over menu selection
        """
        print("\n🎮 Game started!")
        print("Controls:")
        print("  - Click units to select")
        print("  - Click buildings to create units")
        print("  - Click tiles to move")
        print("  - Right-click and hold on a unit to preview attack range")
        print("  - Press SPACE to end turn")
        print("  - Press S to save game")
        print("  - Press ESC to pause")
        print()

        while self.running and not self.game.game_over:
            # Get mouse position once per frame
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    result = self.input_handler.handle_keyboard_event(event)
                    if result == 'quit':
                        self.running = False
                    elif result == 'save':
                        self._handle_save_game()
                    elif result == 'pause':
                        pause_result = self._handle_pause_menu()
                        if pause_result == 'quit':
                            return 'quit'
                        elif pause_result == 'main_menu':
                            return 'main_menu'
                        # 'resume' or None: continue playing

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        result = self.input_handler.handle_mouse_click(mouse_pos)
                        if result == 'continue':
                            continue
                    elif event.button == 3:  # Right click
                        self.input_handler.handle_right_click_press(mouse_pos)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # Right click release
                        self.input_handler.handle_right_click_release()

                elif event.type == pygame.MOUSEMOTION:
                    self.input_handler.handle_mouse_motion(mouse_pos)

            # Rendering
            self._render_frame()

            # Frame timing
            self.clock.tick(60)

        # Handle game over
        if self.game.game_over:
            return self._handle_game_over()

        return 'quit'

    def _handle_save_game(self):
        """Handle save game request."""
        save_menu = SaveGameMenu(self.game)
        result = save_menu.run()
        if result:
            print(f"✅ Game saved to {result}")

    def _handle_pause_menu(self):
        """
        Handle pause menu display and interaction.

        Returns:
            'resume', 'main_menu', 'quit', or None
        """
        pause_menu = PauseMenu(self.renderer.screen)
        result = pause_menu.run()
        pygame.event.clear()

        if result == 'resume':
            return 'resume'
        elif result == 'save':
            self._handle_save_game()
            return 'resume'
        elif result == 'load':
            # Loading from pause menu - return to main menu to handle properly
            return 'main_menu'
        elif result == 'settings':
            settings_menu = SettingsMenu(self.renderer.screen)
            settings_menu.run()
            pygame.event.clear()
            return 'resume'
        elif result == 'main_menu':
            return 'main_menu'
        elif result == 'quit':
            return 'quit'

        return 'resume'

    def _render_frame(self):
        """Render a single frame."""
        self.renderer.render()

        # Draw movement overlay if unit selected
        if self.input_handler.selected_unit:
            self.renderer.draw_movement_overlay(self.input_handler.selected_unit)

        # Draw attack range preview if right-clicking on a unit
        if (self.input_handler.right_click_preview_active and
            self.input_handler.preview_positions):
            self.renderer.draw_attack_range_overlay(
                self.input_handler.preview_positions
            )

        # Draw target overlay if in target selection mode
        if (self.input_handler.target_selection_mode and
            self.input_handler.target_selection_action):
            self.renderer.draw_target_overlay(
                self.input_handler.target_selection_action['targets']
            )

        # Draw active menu last (on top)
        if self.input_handler.active_menu:
            self.input_handler.active_menu.draw(self.renderer.screen)

        pygame.display.flip()

    def _handle_game_over(self):
        """
        Handle game over state.

        Returns:
            'new_game', 'main_menu', or 'quit'
        """
        print(f"\n🎉 Game Over! Player {self.game.winner} wins!")

        # Automatically save replay
        replay_path = self.game.save_replay_to_file()
        if replay_path:
            print(f"📼 Replay saved to {replay_path}")

        # Show game over screen
        game_over_menu = GameOverMenu(self.game.winner, self.game, self.renderer.screen)
        result = game_over_menu.run()

        return result if result else 'quit'


def start_new_game(mode='human_vs_computer', selected_map=None, player_configs=None):
    """
    Start a new game with the specified mode, map, and player configurations.

    Args:
        mode: Game mode string
        selected_map: Map file path or 'random'
        player_configs: List of player configuration dictionaries
    """
    print(f"\n🎮 Starting new game: {mode}\n")

    # Use provided map or show map selection
    if selected_map is None:
        map_menu = MapSelectionMenu()
        selected_map = map_menu.run()

    if not selected_map:
        print("Map selection cancelled")
        return

    # Determine number of players
    num_players = 2
    if mode == '2v2':
        num_players = 4
    elif player_configs:
        num_players = len(player_configs)

    try:
        # Load or generate map
        if selected_map == "random":
            print("Generating random map...")
            map_data = FileIO.generate_random_map(20, 20, num_players=num_players)
            map_file_used = None
        else:
            print(f"Loading map: {selected_map}")
            map_data = FileIO.load_map(selected_map)
            map_file_used = selected_map

        if map_data is None:
            print("Failed to load map")
            return

        # Create game state
        game = GameState(map_data, num_players=num_players)

        # Store map file for saving
        if map_file_used:
            game.map_file_used = map_file_used

        # Set player configurations
        if player_configs:
            game.player_configs = player_configs
        else:
            # Generate default player configs
            game.player_configs = []
            for i in range(num_players):
                if i == 0:
                    game.player_configs.append({'type': 'human', 'bot_type': None})
                else:
                    game.player_configs.append({'type': 'computer', 'bot_type': 'SimpleBot'})

        # Create renderer
        renderer = Renderer(game)

        # Create bots
        settings = get_settings()
        bots = create_bots_from_config(game, game.player_configs, settings)

        # Legacy mode: Ensure bot for player 2 in human_vs_computer
        if mode == 'human_vs_computer' and 2 not in bots:
            from reinforcetactics.game.bot import SimpleBot
            bots[2] = SimpleBot(game, player=2)
            print("Bot created for Player 2")

        # Create and run game session
        session = GameSession(game, renderer, bots, num_players)
        result = session.run()

        pygame.quit()

        # Return result to let caller handle navigation
        return result

    except Exception as e:
        print(f"❌ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()


def load_saved_game():
    """Load and play a saved game."""
    print("\n💾 Loading saved game...\n")

    # Show load menu
    load_menu = LoadGameMenu()
    save_data = load_menu.run()

    if not save_data:
        print("Load cancelled")
        return

    try:
        # Load map
        if 'map_file' in save_data:
            map_data = FileIO.load_map(save_data['map_file'])
        else:
            print("⚠️  Map file not in save, reconstructing from tiles...")
            map_data = FileIO.generate_random_map(
                20, 20, num_players=save_data.get('num_players', 2)
            )

        # Restore game state
        game = GameState.from_dict(save_data, map_data)

        # Create renderer
        renderer = Renderer(game)

        # Create bots
        settings = get_settings()
        bots = {}
        if game.player_configs:
            bots = create_bots_from_config(game, game.player_configs, settings)
        else:
            # Fallback for old saves
            from reinforcetactics.game.bot import SimpleBot
            for player_num in range(2, game.num_players + 1):
                bots[player_num] = SimpleBot(game, player=player_num)
                print(f"Bot created for Player {player_num} (loaded game - legacy)")

        print(f"\n✅ Game loaded! Turn {game.turn_number}, Player {game.current_player}'s turn")
        print("\nControls:")
        print("  - Click units to select")
        print("  - Click tiles to move")
        print("  - Right-click and hold on a unit to preview attack range")
        print("  - Press SPACE to end turn")
        print("  - Press S to save game")
        print("  - Press ESC to pause")
        print()

        # Create and run game session
        session = GameSession(game, renderer, bots, game.num_players)
        result = session.run()

        pygame.quit()

        # Return result to let caller handle navigation
        return result

    except Exception as e:
        print(f"❌ Error loading game: {e}")
        import traceback
        traceback.print_exc()


def watch_replay(replay_path=None):
    """
    Watch a replay.

    Args:
        replay_path: Path to replay file. If None, shows replay selection menu.
    """
    print("\n📼 Loading replay...\n")

    # Show replay selection menu if path not provided
    if not replay_path:
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

        # Load initial map
        game_info = replay_data.get('game_info', {})

        if 'initial_map' in game_info:
            map_data = pd.DataFrame(game_info['initial_map'])
            print("✅ Using stored map from replay")
        else:
            print("⚠️  Replay doesn't have stored map data. Generating random map...")
            map_data = FileIO.generate_random_map(
                20, 20, num_players=game_info.get('num_players', 2)
            )

        # Create and run replay player
        player = ReplayPlayer(replay_data, map_data)
        player.run()

        pygame.quit()
        return 'main_menu'  # Return to main menu after watching replay

    except Exception as e:
        print(f"❌ Error playing replay: {e}")
        import traceback
        traceback.print_exc()
