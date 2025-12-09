"""
Menu system for the strategy game.
Self-contained menus that manage their own pygame screen and navigation.
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from typing import Optional, List, Tuple, Callable, Any, Dict
from pathlib import Path

import pygame

from reinforcetactics.constants import TILE_SIZE, UNIT_DATA
from reinforcetactics.utils.language import get_language, reset_language


class Menu:
    """Base class for game menus. Manages its own screen if not provided."""

    def __init__(self, screen: Optional[pygame.Surface] = None, title: str = "") -> None:
        """
        Initialize the menu.

        Args:
            screen: Optional pygame display surface. If None, creates its own.
            title: Menu title
        """
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Create screen if not provided
        self.owns_screen = screen is None
        if self.owns_screen:
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Reinforce Tactics")
        else:
            self.screen = screen

        self.title = title
        self.running = True
        self.selected_index = 0
        self.options: List[Tuple[str, Callable[[], Any]]] = []

        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.selected_color = (255, 200, 50)
        self.hover_color = (200, 180, 100)
        self.title_color = (100, 200, 255)
        self.option_bg_color = (50, 50, 65)
        self.option_bg_hover_color = (70, 70, 90)
        self.option_bg_selected_color = (80, 80, 100)

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.option_font = pygame.font.Font(None, 36)
        self.indicator_font = pygame.font.Font(None, 24)

        # Mouse tracking
        self.hover_index = -1
        self.option_rects: List[pygame.Rect] = []

        # Scrolling support
        self.scroll_offset = 0
        self.max_visible_options = 8  # Maximum options visible at once
        self.option_spacing = 60

        # Get language instance
        self.lang = get_language()

    def add_option(self, text: str, callback: Callable[[], Any]) -> None:
        """Add a menu option."""
        self.options.append((text, callback))

    def handle_input(self, event: pygame.event.Event) -> Optional[Any]:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            Result of selected option callback, if any
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.options)
                self._ensure_selected_visible()
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)
                self._ensure_selected_visible()
            elif event.key == pygame.K_RETURN:
                if self.options:
                    _, callback = self.options[self.selected_index]
                    return callback()
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos
                # Check if any option was clicked
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(mouse_pos):
                        # Get the actual option index accounting for scroll
                        actual_index = i + self.scroll_offset
                        if actual_index < len(self.options):
                            self.selected_index = actual_index
                            _, callback = self.options[actual_index]
                            return callback()
            elif event.button == 4:  # Mouse wheel up
                self.scroll_offset = max(0, self.scroll_offset - 1)
            elif event.button == 5:  # Mouse wheel down
                max_scroll = max(0, len(self.options) - self.max_visible_options)
                self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_index = -1
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(mouse_pos):
                    self.hover_index = i + self.scroll_offset
                    break

        return None
    
    def _ensure_selected_visible(self) -> None:
        """Ensure the selected option is visible by adjusting scroll offset."""
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + self.max_visible_options:
            self.scroll_offset = self.selected_index - self.max_visible_options + 1

    def draw(self) -> None:
        """Draw the menu."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
            self.screen.blit(title_surface, title_rect)

        # Draw options with scrolling support
        start_y = screen_height // 3
        spacing = self.option_spacing
        self.option_rects = []

        # Calculate maximum option width for uniform sizing
        padding_x = 40
        padding_y = 10
        max_text_width = 0
        for text, _ in self.options:
            display_text = f"> {text}"  # Use the selected format for consistent width
            text_surface = self.option_font.render(display_text, True, self.text_color)
            max_text_width = max(max_text_width, text_surface.get_width())

        uniform_width = max_text_width + 2 * padding_x

        # Determine which options to display (with scrolling)
        total_options = len(self.options)
        start_index = self.scroll_offset
        end_index = min(total_options, start_index + self.max_visible_options)

        # Draw visible options
        for display_i, option_i in enumerate(range(start_index, end_index)):
            text, _ = self.options[option_i]
            
            # Determine styling based on state
            is_selected = option_i == self.selected_index
            is_hovered = option_i == self.hover_index

            # Choose colors
            if is_selected:
                text_color = self.selected_color
                bg_color = self.option_bg_selected_color
            elif is_hovered:
                text_color = self.hover_color
                bg_color = self.option_bg_hover_color
            else:
                text_color = self.text_color
                bg_color = self.option_bg_color

            # Add selection indicator
            display_text = f"> {text}" if is_selected else f"  {text}"

            # Render text
            text_surface = self.option_font.render(display_text, True, text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2, 
                                              y=start_y + display_i * spacing)

            # Create background rectangle with uniform width
            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,  # Center the uniform-width box
                text_rect.y - padding_y,
                uniform_width,
                text_rect.height + 2 * padding_y
            )

            # Draw rounded background rectangle
            pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=8)

            # Draw border for selected/hovered
            if is_selected or is_hovered:
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(self.screen, border_color, bg_rect, width=2, border_radius=8)

            # Draw text
            self.screen.blit(text_surface, text_rect)

            # Store rect for click detection
            self.option_rects.append(bg_rect)

        # Draw scroll indicators if needed
        if total_options > self.max_visible_options:
            # Show up arrow if not at top
            if self.scroll_offset > 0:
                up_text = self.indicator_font.render("▲ Scroll Up", True, self.hover_color)
                up_rect = up_text.get_rect(centerx=screen_width // 2, y=start_y - 30)
                self.screen.blit(up_text, up_rect)
            
            # Show down arrow if not at bottom
            if end_index < total_options:
                down_text = self.indicator_font.render("▼ Scroll Down", True, self.hover_color)
                down_y = start_y + self.max_visible_options * spacing + 10
                down_rect = down_text.get_rect(centerx=screen_width // 2, y=down_y)
                self.screen.blit(down_text, down_rect)
            
            # Show position indicator (e.g., "3 / 15")
            pos_text = self.indicator_font.render(
                f"{self.scroll_offset + 1}-{end_index} / {total_options}",
                True, self.text_color
            )
            pos_rect = pos_text.get_rect(right=screen_width - 20, bottom=screen_height - 20)
            self.screen.blit(pos_text, pos_rect)

        pygame.display.flip()

    def run(self) -> Optional[Any]:
        """
        Run the menu loop.

        Returns:
            Result from selected option, or None
        """
        result = None
        clock = pygame.time.Clock()

        # Clear any residual events from before this menu started
        pygame.event.clear()

        # Don't draw before the event loop - causes double-click issue
        # option_rects will be populated on first draw() call in the loop

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                result = self.handle_input(event)
                if result is not None:
                    return result

            self.draw()
            clock.tick(30)

        return result


class MainMenu(Menu):
    """Main menu for the game. Handles navigation to sub-menus internally."""

    def __init__(self) -> None:
        """Initialize main menu with self-managed screen."""
        super().__init__(None, self._get_title())
        self._setup_options()

    def _get_title(self) -> str:
        return get_language().get('main_menu.title', 'Reinforce Tactics')

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('main_menu.new_game', 'New Game'), self._new_game)
        self.add_option(lang.get('main_menu.load_game', 'Load Game'), self._load_game)
        self.add_option(lang.get('main_menu.watch_replay', 'Watch Replay'), self._watch_replay)
        self.add_option(lang.get('main_menu.settings', 'Settings'), self._settings)
        self.add_option(lang.get('main_menu.quit', 'Quit'), self._quit)

    def _new_game(self) -> Optional[Dict[str, Any]]:
        """Handle new game - show game mode selection, map selection, and player configuration."""
        # Step 1: Select game mode
        mode_menu = GameModeMenu(self.screen)
        selected_mode = mode_menu.run()
        pygame.event.clear()

        if not selected_mode:
            return None  # User cancelled

        # Step 2: Select map from chosen mode
        map_menu = MapSelectionMenu(self.screen, game_mode=selected_mode)
        selected_map = map_menu.run()
        pygame.event.clear()

        if not selected_map:
            return None  # User cancelled

        # Step 3: Configure players
        player_config_menu = PlayerConfigMenu(self.screen, game_mode=selected_mode)
        player_config_result = player_config_menu.run()
        pygame.event.clear()

        if player_config_result:
            return {
                'type': 'new_game',
                'map': selected_map,
                'mode': selected_mode,  # Contains "1v1" or "2v2"
                'players': player_config_result['players']
            }
        return None

    def _load_game(self) -> Optional[Dict[str, Any]]:
        """Handle load game - show load menu and return result."""
        load_menu = LoadGameMenu(self.screen)
        save_path = load_menu.run()
        pygame.event.clear()

        if save_path:
            return {
                'type': 'load_game',
                'save_path': save_path
            }
        return None  # Cancelled

    def _watch_replay(self) -> Optional[Dict[str, Any]]:
        """Handle watch replay - show replay menu and return result."""
        replay_menu = ReplaySelectionMenu(self.screen)
        replay_path = replay_menu.run()
        pygame.event.clear()

        if replay_path:
            return {
                'type': 'watch_replay',
                'replay_path': replay_path
            }
        return None  # Cancelled

    def _settings(self) -> None:
        """Handle settings - show settings menu."""
        settings_menu = SettingsMenu(self.screen)
        settings_menu.run()
        pygame.event.clear()
        # Return to main menu after settings

    def _quit(self) -> Dict[str, Any]:
        """Handle quit."""
        return {'type': 'exit'}

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the main menu loop with internal navigation.

        Returns:
            Dict with 'type' key indicating action, or None if cancelled
        """
        result = None
        clock = pygame.time.Clock()

        # Clear any residual events from before this menu started
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return {'type': 'exit'}

                result = self.handle_input(event)
                if result is not None:
                    # If we got a dict back, that's our final result
                    if isinstance(result, dict):
                        return result
                    # Otherwise stay in menu loop

            self.draw()
            clock.tick(30)

        return result if isinstance(result, dict) else {'type': 'exit'}


class GameModeMenu(Menu):
    """Menu for selecting game mode (1v1 or 2v2)."""

    def __init__(self, screen: Optional[pygame.Surface] = None, maps_dir: str = "maps") -> None:
        """
        Initialize game mode menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            maps_dir: Directory containing map subdirectories
        """
        super().__init__(screen, get_language().get('new_game.select_mode', 'Select Game Mode'))
        self.maps_dir = maps_dir
        self.available_modes: List[str] = []
        self._load_modes()
        self._setup_options()

    def _load_modes(self) -> None:
        """Discover available game mode folders."""
        if os.path.exists(self.maps_dir):
            for item in os.listdir(self.maps_dir):
                item_path = os.path.join(self.maps_dir, item)
                if os.path.isdir(item_path):
                    # Check if folder contains .csv maps
                    try:
                        if any(f.endswith('.csv') for f in os.listdir(item_path)):
                            self.available_modes.append(item)
                    except (OSError, PermissionError):
                        # Skip directories that can't be read
                        continue
        self.available_modes.sort()

    def _setup_options(self) -> None:
        """Setup menu options for available game modes."""
        for mode in self.available_modes:
            self.add_option(mode, lambda m=mode: m)
        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def run(self) -> Optional[str]:
        """
        Run game mode selection menu.

        Returns:
            Selected game mode string (e.g., "1v1" or "2v2"), or None if cancelled
        """
        return super().run()


class MapSelectionMenu(Menu):
    """Menu for selecting a map when starting a new game."""

    def __init__(self, screen: Optional[pygame.Surface] = None, maps_dir: str = "maps",
                 game_mode: Optional[str] = None) -> None:
        """
        Initialize map selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            maps_dir: Directory containing map files
            game_mode: Optional game mode to filter maps (e.g., "1v1", "2v2")
        """
        super().__init__(screen, get_language().get('new_game.title', 'Select Map'))
        self.maps_dir = maps_dir
        self.game_mode = game_mode
        self.available_maps: List[str] = []
        self._load_maps()
        self._setup_options()

    def _load_maps(self) -> None:
        """Load available map files."""
        if os.path.exists(self.maps_dir):
            if self.game_mode:
                # Load maps only from the specified game mode subfolder
                subdir_path = os.path.join(self.maps_dir, self.game_mode)
                if os.path.exists(subdir_path):
                    for f in os.listdir(subdir_path):
                        if f.endswith('.csv'):
                            # Store full path including maps/ prefix
                            map_path = os.path.join(self.maps_dir, self.game_mode, f)
                            self.available_maps.append(map_path)
            else:
                # Load maps from all subdirectories (backward compatibility)
                for subdir in ['1v1', '2v2']:
                    subdir_path = os.path.join(self.maps_dir, subdir)
                    if os.path.exists(subdir_path):
                        for f in os.listdir(subdir_path):
                            if f.endswith('.csv'):
                                # Store full path including maps/ prefix
                                self.available_maps.append(os.path.join(self.maps_dir, subdir, f))

        # Add random map option
        self.available_maps.insert(0, "random")

    def _setup_options(self) -> None:
        """Setup menu options for available maps."""
        for map_file in self.available_maps:
            if map_file == "random":
                display_name = get_language().get('new_game.random_map', 'Random Map')
            else:
                if self.game_mode:
                    # Show just the filename when game mode is already selected
                    display_name = os.path.splitext(os.path.basename(map_file))[0]
                else:
                    # Include the subdirectory in the display name to distinguish duplicates
                    # e.g., "1v1/6x6_beginner" instead of just "6x6_beginner"
                    relative_path = map_file.replace(self.maps_dir + os.sep, '')
                    # Remove .csv extension
                    display_name = os.path.splitext(relative_path)[0]
            self.add_option(display_name, lambda m=map_file: m)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def draw(self) -> None:
        """Draw the map selection menu with reduced spacing."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
            self.screen.blit(title_surface, title_rect)

        # Draw options with reduced spacing (25% less gap from title)
        # Original: start_y = screen_height // 3 (200 for 600px height)
        # Gap from title at y=50: 150px
        # 25% reduction: 150 * 0.75 = 112.5
        # New start_y: 50 + 112.5 = 162.5
        start_y = int(screen_height * 0.27)  # Approximately 162 for 600px height
        spacing = 60
        self.option_rects = []

        # Calculate maximum option width for uniform sizing
        padding_x = 40
        padding_y = 10
        max_text_width = 0
        for text, _ in self.options:
            display_text = f"> {text}"  # Use the selected format for consistent width
            text_surface = self.option_font.render(display_text, True, self.text_color)
            max_text_width = max(max_text_width, text_surface.get_width())

        uniform_width = max_text_width + 2 * padding_x

        for i, (text, _) in enumerate(self.options):
            # Determine styling based on state
            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index

            # Choose colors
            if is_selected:
                text_color = self.selected_color
                bg_color = self.option_bg_selected_color
            elif is_hovered:
                text_color = self.hover_color
                bg_color = self.option_bg_hover_color
            else:
                text_color = self.text_color
                bg_color = self.option_bg_color

            # Add selection indicator
            display_text = f"> {text}" if is_selected else f"  {text}"

            # Render text
            text_surface = self.option_font.render(display_text, True, text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2, y=start_y + i * spacing)

            # Create background rectangle with uniform width
            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,  # Center the uniform-width box
                text_rect.y - padding_y,
                uniform_width,
                text_rect.height + 2 * padding_y
            )

            # Draw rounded background rectangle
            pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=8)

            # Draw border for selected/hovered
            if is_selected or is_hovered:
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(self.screen, border_color, bg_rect, width=2, border_radius=8)

            # Draw text
            self.screen.blit(text_surface, text_rect)

            # Store rect for click detection
            self.option_rects.append(bg_rect)

        pygame.display.flip()

    def run(self) -> Optional[str]:
        """
        Run map selection menu.

        Returns:
            Selected map path string, or None if cancelled
        """
        return super().run()


class PlayerConfigMenu:
    """Menu for configuring players (Human vs Computer) with difficulty settings."""

    def __init__(self, screen: Optional[pygame.Surface] = None, game_mode: str = "1v1") -> None:
        """
        Initialize player configuration menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            game_mode: Game mode ("1v1" or "2v2")
        
        Raises:
            ValueError: If game_mode is not "1v1" or "2v2"
        """
        # Validate game_mode
        if game_mode not in ["1v1", "2v2"]:
            raise ValueError(f"Invalid game_mode: {game_mode}. Must be '1v1' or '2v2'")
        
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Create screen if not provided
        self.owns_screen = screen is None
        if self.owns_screen:
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Reinforce Tactics")
        else:
            self.screen = screen

        self.game_mode = game_mode
        self.num_players = 2 if game_mode == "1v1" else 4
        self.running = True

        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.selected_color = (255, 200, 50)
        self.hover_color = (200, 180, 100)
        self.title_color = (100, 200, 255)
        self.option_bg_color = (50, 50, 65)
        self.option_bg_hover_color = (70, 70, 90)
        self.option_bg_selected_color = (80, 80, 100)
        self.disabled_color = (100, 100, 120)

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.label_font = pygame.font.Font(None, 32)
        self.option_font = pygame.font.Font(None, 28)

        # Player configurations
        # Default: Player 1 is Human, others are Computer (SimpleBot)
        self.player_configs = []
        for i in range(self.num_players):
            self.player_configs.append({
                'type': 'human' if i == 0 else 'computer',
                'bot_type': None if i == 0 else 'SimpleBot'
            })

        # UI interaction tracking
        self.hover_element = None
        self.selected_element = None
        self.interactive_elements: List[Dict[str, Any]] = []

        # Get language instance
        self.lang = get_language()

    def handle_input(self, event: pygame.event.Event) -> Optional[Dict[str, Any]]:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            Result dict with player configurations, or None
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return None
            elif event.key == pygame.K_RETURN:
                # Start game with current configuration
                return self._get_result()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos
                for element in self.interactive_elements:
                    if element['rect'].collidepoint(mouse_pos):
                        if element['type'] == 'type_toggle':
                            # Toggle between human and computer
                            player_idx = element['player_idx']
                            config = self.player_configs[player_idx]
                            if config['type'] == 'human':
                                config['type'] = 'computer'
                                config['bot_type'] = 'SimpleBot'
                            else:
                                config['type'] = 'human'
                                config['bot_type'] = None

                        elif element['type'] == 'difficulty_select':
                            # Cycle through difficulty levels (only SimpleBot is enabled)
                            player_idx = element['player_idx']
                            config = self.player_configs[player_idx]
                            if config['type'] == 'computer':
                                # Only SimpleBot is available; others are disabled
                                config['bot_type'] = 'SimpleBot'

                        elif element['type'] == 'start_button':
                            return self._get_result()

                        elif element['type'] == 'back_button':
                            self.running = False
                            return None

        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_element = None
            for element in self.interactive_elements:
                if element['rect'].collidepoint(mouse_pos):
                    self.hover_element = element
                    break

        return None

    def _get_result(self) -> Dict[str, Any]:
        """Get the configured player settings as a result dict."""
        return {
            'players': self.player_configs
        }

    def draw(self) -> None:
        """Draw the player configuration menu."""
        self.screen.fill(self.bg_color)
        self.interactive_elements = []

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        title = self.lang.get('player_config.title', 'Configure Players')
        title_surface = self.title_font.render(title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=30)
        self.screen.blit(title_surface, title_rect)

        # Starting Y position for player configurations
        # Use more compact spacing for 2v2 to fit all elements on screen
        start_y = 80
        spacing_y = 85 if self.num_players > 2 else 100

        # Draw each player's configuration
        for i in range(self.num_players):
            config = self.player_configs[i]
            y_pos = start_y + i * spacing_y

            # Player label
            player_label = self.lang.get('player_config.player', 'Player {number}').format(number=i + 1)
            label_surface = self.label_font.render(player_label, True, self.text_color)
            label_rect = label_surface.get_rect(x=50, y=y_pos)
            self.screen.blit(label_surface, label_rect)

            # Type toggle button (Human/Computer)
            type_x = 200
            type_text = self.lang.get('player_config.type_human', 'Human') if config['type'] == 'human' else self.lang.get('player_config.type_computer', 'Computer')
            type_rect = self._draw_button(type_x, y_pos, type_text, 'type_toggle', i)

            # Difficulty selection (only shown if computer)
            if config['type'] == 'computer':
                diff_x = 400
                diff_text = self.lang.get('player_config.difficulty_simple', 'SimpleBot')
                self._draw_button(diff_x, y_pos, diff_text, 'difficulty_select', i, disabled=False)

        # Draw Start Game button
        start_y_pos = start_y + self.num_players * spacing_y + 20
        start_text = self.lang.get('player_config.start_game', 'Start Game')
        self._draw_button(screen_width // 2 - 100, start_y_pos, start_text, 'start_button', centered=True)

        # Draw Back button
        back_text = self.lang.get('common.back', 'Back')
        self._draw_button(screen_width // 2 - 100, start_y_pos + 60, back_text, 'back_button', centered=True)

        pygame.display.flip()

    def _draw_button(self, x: int, y: int, text: str, element_type: str, 
                     player_idx: int = -1, centered: bool = False, disabled: bool = False) -> pygame.Rect:
        """
        Draw a button and register it as an interactive element.

        Args:
            x: X position
            y: Y position
            text: Button text
            element_type: Type of element ('type_toggle', 'difficulty_select', 'start_button', 'back_button')
            player_idx: Player index for player-specific buttons
            centered: Whether to center the button at x position
            disabled: Whether the button is disabled

        Returns:
            Button rect
        """
        padding_x = 20
        padding_y = 10
        # Container width for centered buttons
        button_container_width = 200

        # Render text
        text_color = self.disabled_color if disabled else self.text_color
        text_surface = self.option_font.render(text, True, text_color)
        text_rect = text_surface.get_rect()

        # Calculate button dimensions
        button_width = text_rect.width + 2 * padding_x
        button_height = text_rect.height + 2 * padding_y

        # Adjust position if centered
        if centered:
            button_x = x + (button_container_width - button_width) // 2
        else:
            button_x = x

        button_rect = pygame.Rect(button_x, y, button_width, button_height)

        # Determine styling
        is_hovered = self.hover_element and self.hover_element.get('rect') == button_rect and not disabled

        if is_hovered:
            bg_color = self.option_bg_hover_color
            border_color = self.hover_color
        else:
            bg_color = self.option_bg_color
            border_color = self.option_bg_color if disabled else (60, 60, 80)

        # Draw button background
        pygame.draw.rect(self.screen, bg_color, button_rect, border_radius=8)

        # Draw border
        if is_hovered:
            pygame.draw.rect(self.screen, border_color, button_rect, width=2, border_radius=8)

        # Draw text
        text_rect.center = button_rect.center
        self.screen.blit(text_surface, text_rect)

        # Register as interactive element if not disabled
        if not disabled:
            self.interactive_elements.append({
                'type': element_type,
                'rect': button_rect,
                'player_idx': player_idx
            })

        return button_rect

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the player configuration menu loop.

        Returns:
            Dict with player configurations, or None if cancelled
        """
        result = None
        clock = pygame.time.Clock()

        # Clear any residual events from before this menu started
        pygame.event.clear()

        # Draw once before event loop to populate interactive_elements
        self.draw()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                result = self.handle_input(event)
                if result is not None or not self.running:
                    return result

            self.draw()
            clock.tick(30)

        return result


class SaveGameMenu(Menu):
    """Menu for saving the game."""

    def __init__(self, game: Any, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize save game menu.

        Args:
            game: Game state object to save
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('save_game.title', 'Save Game'))
        self.game = game
        self.input_text = f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.input_active = True

    def handle_input(self, event: pygame.event.Event) -> Optional[str]:
        """Handle keyboard input for filename entry."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.input_text:
                    return self._save_game()
            elif event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                # Add character if printable
                if event.unicode.isprintable() and len(self.input_text) < 50:
                    self.input_text += event.unicode

        return None

    def _save_game(self) -> Optional[str]:
        """Save the game with current filename."""
        # Ensure saves directory exists
        saves_dir = Path("saves")
        saves_dir.mkdir(exist_ok=True)

        filepath = self.game.save_to_file(f"saves/{self.input_text}.json")
        return filepath

    def run(self) -> Optional[str]:
        """
        Run save game menu.

        Returns:
            Path to saved file, or None if cancelled
        """
        return super().run()

    def draw(self) -> None:
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        title_surface = self.title_font.render(self.title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
        self.screen.blit(title_surface, title_rect)

        # Draw input prompt
        prompt = get_language().get('save_game.enter_name', 'Enter save name:')
        prompt_surface = self.option_font.render(prompt, True, self.text_color)
        prompt_rect = prompt_surface.get_rect(centerx=screen_width // 2, y=screen_height // 3)
        self.screen.blit(prompt_surface, prompt_rect)

        # Draw input box
        input_width = 400
        input_height = 40
        input_rect = pygame.Rect(
            (screen_width - input_width) // 2,
            screen_height // 3 + 50,
            input_width,
            input_height
        )
        pygame.draw.rect(self.screen, (50, 50, 60), input_rect)
        pygame.draw.rect(self.screen, self.selected_color, input_rect, 2)

        # Draw input text
        text_surface = self.option_font.render(self.input_text, True, self.text_color)
        text_rect = text_surface.get_rect(midleft=(input_rect.x + 10, input_rect.centery))
        self.screen.blit(text_surface, text_rect)

        # Draw cursor
        cursor_x = text_rect.right + 2
        pygame.draw.line(
            self.screen,
            self.text_color,
            (cursor_x, input_rect.y + 5),
            (cursor_x, input_rect.y + input_height - 5),
            2
        )

        # Draw instructions
        lang = get_language()
        instructions = lang.get('save_game.instructions',
                               'Press ENTER to save, ESC to cancel')
        inst_surface = self.option_font.render(instructions, True, (150, 150, 150))
        inst_rect = inst_surface.get_rect(centerx=screen_width // 2, y=screen_height // 2 + 50)
        self.screen.blit(inst_surface, inst_rect)

        pygame.display.flip()


class LoadGameMenu(Menu):
    """Menu for loading saved games."""

    def __init__(self, screen: Optional[pygame.Surface] = None, saves_dir: str = "saves") -> None:
        """
        Initialize load game menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            saves_dir: Directory containing save files
        """
        super().__init__(screen, get_language().get('load_game.title', 'Load Game'))
        self.saves_dir = saves_dir
        self.save_files: List[str] = []
        self._load_saves()
        self._setup_options()

    def _load_saves(self) -> None:
        """Load available save files."""
        if os.path.exists(self.saves_dir):
            self.save_files = [
                f for f in os.listdir(self.saves_dir)
                if f.endswith('.json')
            ]
            # Sort by modification time (newest first)
            self.save_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.saves_dir, f)),
                reverse=True
            )

    def _setup_options(self) -> None:
        """Setup menu options for available save files."""
        if not self.save_files:
            # No saves available
            lang = get_language()
            self.add_option(lang.get('load_game.no_saves', 'No saved games found'), lambda: None)
        else:
            for save_file in self.save_files:
                display_name = os.path.splitext(save_file)[0]
                filepath = os.path.join(self.saves_dir, save_file)
                self.add_option(display_name, lambda p=filepath: p)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run load game menu.

        Returns:
            Dict with loaded save data, or None if cancelled
        """
        selected_path = super().run()

        if not selected_path:
            return None

        # Load the actual save data from the file
        try:
            with open(selected_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            return save_data
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            print(f"Error loading save file: {e}")
            return None


class ReplaySelectionMenu(Menu):
    """Menu for selecting a replay to watch."""

    def __init__(self, screen: Optional[pygame.Surface] = None,
                 replays_dir: str = "replays") -> None:
        """
        Initialize replay selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            replays_dir: Directory containing replay files
        """
        super().__init__(screen, get_language().get('replay.title', 'Select Replay'))
        self.replays_dir = replays_dir
        self.replay_files: List[str] = []
        self._load_replays()
        self._setup_options()

    def _load_replays(self) -> None:
        """Load available replay files."""
        if os.path.exists(self.replays_dir):
            self.replay_files = [
                f for f in os.listdir(self.replays_dir)
                if f.endswith('.json')
            ]
            # Sort by modification time (newest first)
            self.replay_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.replays_dir, f)),
                reverse=True
            )

    def _setup_options(self) -> None:
        """Setup menu options for available replay files."""
        if not self.replay_files:
            # No replays available
            lang = get_language()
            self.add_option(lang.get('replay.no_replays', 'No replays found'), lambda: None)
        else:
            for replay_file in self.replay_files:
                display_name = os.path.splitext(replay_file)[0]
                filepath = os.path.join(self.replays_dir, replay_file)
                self.add_option(display_name, lambda p=filepath: p)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def run(self) -> Optional[str]:
        """
        Run replay selection menu.

        Returns:
            Path to selected replay file, or None if cancelled
        """
        return super().run()


class SettingsMenu(Menu):
    """Settings menu."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize settings menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('settings.title', 'Settings'))
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('settings.language', 'Language'), self._change_language)
        self.add_option(lang.get('settings.sound', 'Sound'), self._toggle_sound)
        self.add_option(lang.get('settings.fullscreen', 'Fullscreen'), self._toggle_fullscreen)
        self.add_option(lang.get('common.back', 'Back'), lambda: None)

    def _change_language(self) -> str:
        """Open language selection menu."""
        return 'language_menu'

    def _toggle_sound(self) -> None:
        """Toggle sound on/off. Currently not implemented."""
        # Sound system not yet implemented in the game

    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        pygame.display.toggle_fullscreen()


class LanguageMenu(Menu):
    """Language selection menu."""

    LANGUAGES = {
        'en': 'English',
        'fr': 'Français',
        'ko': '한국어',
        'es': 'Español'
    }

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize language menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('language.title', 'Select Language'))
        self._setup_options()

    def _setup_options(self) -> None:
        for code, name in self.LANGUAGES.items():
            self.add_option(name, lambda c=code: self._set_language(c))

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def _set_language(self, lang_code: str) -> str:
        """Set the game language."""
        reset_language(lang_code)
        self.lang = get_language()  # Refresh our reference
        return lang_code


class PauseMenu(Menu):
    """In-game pause menu."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize pause menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('pause.title', 'Paused'))
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('pause.resume', 'Resume'), lambda: 'resume')
        self.add_option(lang.get('pause.save', 'Save Game'), lambda: 'save')
        self.add_option(lang.get('pause.load', 'Load Game'), lambda: 'load')
        self.add_option(lang.get('pause.settings', 'Settings'), lambda: 'settings')
        self.add_option(lang.get('pause.main_menu', 'Main Menu'), lambda: 'main_menu')
        self.add_option(lang.get('pause.quit', 'Quit'), lambda: 'quit')


class GameOverMenu(Menu):
    """Game over screen."""

    def __init__(self, winner: int, game_state: Any,
                 screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize game over menu.

        Args:
            winner: Player number who won
            game_state: Game state object
            screen: Optional pygame surface. If None, creates its own.
        """
        lang = get_language()
        title = lang.get('game_over.title', 'Game Over')
        super().__init__(screen, title)

        self.winner = winner
        self.game_state = game_state
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('game_over.save_replay', 'Save Replay'), self._save_replay)
        self.add_option(lang.get('game_over.new_game', 'New Game'), lambda: 'new_game')
        self.add_option(lang.get('game_over.main_menu', 'Main Menu'), lambda: 'main_menu')
        self.add_option(lang.get('game_over.quit', 'Quit'), lambda: 'quit')

    def _save_replay(self) -> Optional[str]:
        """Save game replay."""
        return self.game_state.save_replay_to_file()

    def draw(self) -> None:
        super().draw()

        # Draw winner announcement
        lang = get_language()
        winner_template = lang.get('game_over.winner', 'Player {player} Wins!')
        winner_text = winner_template.format(player=self.winner)

        winner_surface = self.title_font.render(winner_text, True, self.selected_color)
        winner_rect = winner_surface.get_rect(
            centerx=self.screen.get_width() // 2,
            y=100
        )
        self.screen.blit(winner_surface, winner_rect)

        pygame.display.flip()


class UnitPurchaseMenu:
    """In-game overlay menu for purchasing units on buildings."""

    def __init__(self, screen: pygame.Surface, game_state: Any, building_pos: Tuple[int, int]) -> None:
        """
        Initialize unit purchase menu.

        Args:
            screen: Pygame surface to draw on
            game_state: Game state object
            building_pos: (x, y) tuple of building position in grid coordinates
        """
        self.screen = screen
        self.game_state = game_state
        self.building_pos = building_pos
        self.running = True

        # Colors
        self.bg_color = (40, 40, 50)
        self.text_color = (255, 255, 255)
        self.hover_color = (200, 180, 100)
        self.disabled_color = (100, 100, 120)
        self.disabled_bg_color = (60, 60, 70)
        self.border_color = (100, 150, 200)
        self.close_button_color = (200, 50, 50)
        self.close_button_hover_color = (255, 80, 80)

        # Fonts
        self.title_font = pygame.font.Font(None, 28)
        self.option_font = pygame.font.Font(None, 24)

        # Unit types to display (basic units only: Warrior, Mage, Cleric)
        # Barbarian is excluded as it costs 400g vs 200-250g for basic units
        self.unit_types = ['W', 'M', 'C']

        # Interactive elements
        self.interactive_elements: List[Dict[str, Any]] = []
        self.hover_element = None

        # Calculate menu position and size
        self._calculate_menu_rect()

    def _calculate_menu_rect(self) -> None:
        """Calculate the menu rectangle position and size."""
        # Menu dimensions
        menu_width = 220
        menu_height = 180
        
        # Convert building position to screen coordinates
        building_screen_x = self.building_pos[0] * TILE_SIZE
        building_screen_y = self.building_pos[1] * TILE_SIZE
        
        # Position menu to the right of the building
        menu_x = building_screen_x + TILE_SIZE + 10
        menu_y = building_screen_y
        
        # Ensure menu stays within screen bounds
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        if menu_x + menu_width > screen_width:
            # Position to the left instead
            menu_x = building_screen_x - menu_width - 10
        
        if menu_y + menu_height > screen_height:
            # Position higher
            menu_y = screen_height - menu_height - 10
        
        # Ensure minimum position
        menu_x = max(10, menu_x)
        menu_y = max(10, menu_y)
        
        self.menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)

    def handle_click(self, mouse_pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Handle mouse clicks.

        Args:
            mouse_pos: (x, y) tuple of mouse position

        Returns:
            Dict with action result, or None
        """
        # Check if click is outside menu (close menu)
        if not self.menu_rect.collidepoint(mouse_pos):
            return {'type': 'close'}

        # Check interactive elements
        for element in self.interactive_elements:
            if element['rect'].collidepoint(mouse_pos):
                if element['type'] == 'close_button':
                    return {'type': 'close'}
                elif element['type'] == 'unit_button' and not element['disabled']:
                    unit_type = element['unit_type']
                    # Try to create the unit
                    unit = self.game_state.create_unit(
                        unit_type,
                        self.building_pos[0],
                        self.building_pos[1]
                    )
                    if unit:
                        return {'type': 'unit_created', 'unit': unit}
                    else:
                        # Failed to create - position occupied or insufficient gold
                        # This shouldn't happen if UI logic is correct
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to create unit {unit_type} at {self.building_pos}")
                        return None

        return None

    def handle_mouse_motion(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Handle mouse motion for hover effects.

        Args:
            mouse_pos: (x, y) tuple of mouse position
        """
        self.hover_element = None
        for element in self.interactive_elements:
            if element['rect'].collidepoint(mouse_pos):
                self.hover_element = element
                break

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the unit purchase menu.

        Args:
            screen: Pygame surface to draw on
        """
        self.interactive_elements = []

        # Draw semi-transparent background overlay for entire screen (to show menu is modal)
        overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        screen.blit(overlay, (0, 0))

        # Draw menu background
        pygame.draw.rect(screen, self.bg_color, self.menu_rect, border_radius=10)
        pygame.draw.rect(screen, self.border_color, self.menu_rect, width=2, border_radius=10)

        # Draw title
        title = "Purchase Unit"
        title_surface = self.title_font.render(title, True, self.text_color)
        title_rect = title_surface.get_rect(
            centerx=self.menu_rect.centerx,
            y=self.menu_rect.y + 10
        )
        screen.blit(title_surface, title_rect)

        # Draw close button (X) in upper right
        close_button_size = 20
        close_button_x = self.menu_rect.right - close_button_size - 10
        close_button_y = self.menu_rect.y + 10
        close_button_rect = pygame.Rect(
            close_button_x,
            close_button_y,
            close_button_size,
            close_button_size
        )

        # Check if hovering over close button
        is_close_hover = (self.hover_element and 
                         self.hover_element.get('type') == 'close_button')
        close_color = self.close_button_hover_color if is_close_hover else self.close_button_color

        pygame.draw.rect(screen, close_color, close_button_rect, border_radius=3)
        
        # Draw X
        x_margin = 4
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (close_button_rect.left + x_margin, close_button_rect.top + x_margin),
            (close_button_rect.right - x_margin, close_button_rect.bottom - x_margin),
            2
        )
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (close_button_rect.right - x_margin, close_button_rect.top + x_margin),
            (close_button_rect.left + x_margin, close_button_rect.bottom - x_margin),
            2
        )

        self.interactive_elements.append({
            'type': 'close_button',
            'rect': close_button_rect
        })

        # Draw unit options
        start_y = self.menu_rect.y + 50
        spacing = 35
        current_player = self.game_state.current_player
        player_gold = self.game_state.player_gold[current_player]

        for i, unit_type in enumerate(self.unit_types):
            unit_data = UNIT_DATA[unit_type]
            unit_name = unit_data['name']
            unit_cost = unit_data['cost']
            
            # Check if player can afford
            can_afford = player_gold >= unit_cost
            
            y_pos = start_y + i * spacing
            
            # Draw unit option
            button_width = 190
            button_height = 28
            button_x = self.menu_rect.x + 15
            button_rect = pygame.Rect(button_x, y_pos, button_width, button_height)
            
            # Check if hovering
            is_hovered = (self.hover_element and 
                         self.hover_element.get('rect') == button_rect and
                         can_afford)
            
            # Choose colors
            if not can_afford:
                bg_color = self.disabled_bg_color
                text_color = self.disabled_color
            elif is_hovered:
                bg_color = (70, 70, 90)
                text_color = self.hover_color
            else:
                bg_color = (50, 50, 65)
                text_color = self.text_color
            
            # Draw button background
            pygame.draw.rect(screen, bg_color, button_rect, border_radius=5)
            
            if is_hovered:
                pygame.draw.rect(screen, self.hover_color, button_rect, width=2, border_radius=5)
            
            # Draw text: "Unit Name - Cost"
            text = f"{unit_name} - {unit_cost}g"
            text_surface = self.option_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(
                left=button_rect.left + 10,
                centery=button_rect.centery
            )
            screen.blit(text_surface, text_rect)
            
            # Register as interactive element
            self.interactive_elements.append({
                'type': 'unit_button',
                'rect': button_rect,
                'unit_type': unit_type,
                'disabled': not can_afford
            })
