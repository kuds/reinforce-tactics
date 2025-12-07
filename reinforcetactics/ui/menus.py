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

        # Mouse tracking
        self.hover_index = -1
        self.option_rects: List[pygame.Rect] = []

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
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)
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
                        self.selected_index = i
                        if self.options:
                            _, callback = self.options[i]
                            return callback()
        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_index = -1
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(mouse_pos):
                    self.hover_index = i
                    break

        return None

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

        # Draw options
        start_y = screen_height // 3
        spacing = 60
        self.option_rects = []

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

            # Create background rectangle with padding
            padding_x = 40
            padding_y = 10
            bg_rect = pygame.Rect(
                text_rect.x - padding_x,
                text_rect.y - padding_y,
                text_rect.width + 2 * padding_x,
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

    def run(self) -> Optional[Any]:
        """
        Run the menu loop.

        Returns:
            Result from selected option, or None
        """
        result = None
        clock = pygame.time.Clock()

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
        """Handle new game - show map selection and return result."""
        map_menu = MapSelectionMenu(self.screen)
        selected_map = map_menu.run()
        
        # Clear event queue to prevent double-processing
        pygame.event.clear()

        if selected_map:
            # Return dictionary with new game info
            return {
                'type': 'new_game',
                'map': selected_map,
                'mode': 'human_vs_computer'  # Default mode
            }
        # Return None to stay in menu when cancelled
        return None

    def _load_game(self) -> Optional[Dict[str, Any]]:
        """Handle load game - show load menu and return result."""
        load_menu = LoadGameMenu(self.screen)
        save_path = load_menu.run()

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


class MapSelectionMenu(Menu):
    """Menu for selecting a map when starting a new game."""

    def __init__(self, screen: Optional[pygame.Surface] = None, maps_dir: str = "maps") -> None:
        """
        Initialize map selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            maps_dir: Directory containing map files
        """
        super().__init__(screen, get_language().get('new_game.title', 'Select Map'))
        self.maps_dir = maps_dir
        self.available_maps: List[str] = []
        self._load_maps()
        self._setup_options()

    def _load_maps(self) -> None:
        """Load available map files."""
        if os.path.exists(self.maps_dir):
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
                # Include the subdirectory in the display name to distinguish duplicates
                # e.g., "1v1/6x6_beginner" instead of just "6x6_beginner"
                relative_path = map_file.replace(self.maps_dir + os.sep, '')
                # Remove .csv extension
                display_name = os.path.splitext(relative_path)[0]
            self.add_option(display_name, lambda m=map_file: m)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def run(self) -> Optional[str]:
        """
        Run map selection menu.

        Returns:
            Selected map path string, or None if cancelled
        """
        return super().run()


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


class BuildingMenu:
    """In-game menu for creating units from buildings."""

    def __init__(self, game: Any, building_pos: Tuple[int, int]) -> None:
        """
        Initialize building menu.

        Args:
            game: Game state object
            building_pos: (x, y) position of the building
        """
        self.game = game
        self.building_pos = building_pos
        self.screen = None  # Will use renderer's screen

        # Menu appearance
        self.width = 200
        self.height = 250
        self.bg_color = (40, 40, 50, 230)  # Semi-transparent
        self.text_color = (255, 255, 255)
        self.selected_color = (255, 200, 50)
        self.border_color = (100, 100, 120)

        # Unit types that can be created (use UNIT_DATA keys: W=Warrior, M=Mage, C=Cleric, B=Barbarian)
        self.unit_types = ['W', 'M', 'C']  # Warrior, Mage, Cleric
        self.selected_index = 0

        # Font
        self.font = pygame.font.Font(None, 24)

    def handle_click(self, mouse_pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Handle mouse click on the menu.

        Args:
            mouse_pos: (x, y) mouse position

        Returns:
            Dict with action info, or None
        """
        # Calculate menu position
        menu_x = self.building_pos[0] * TILE_SIZE + TILE_SIZE
        menu_y = self.building_pos[1] * TILE_SIZE

        # Check if click is inside menu
        if not (menu_x <= mouse_pos[0] <= menu_x + self.width and
                menu_y <= mouse_pos[1] <= menu_y + self.height):
            return {'type': 'close'}

        # Check which option was clicked
        option_height = 40
        start_y = menu_y + 40

        for i, unit_type in enumerate(self.unit_types):
            option_y = start_y + i * option_height
            if option_y <= mouse_pos[1] <= option_y + option_height:
                return {
                    'type': 'create_unit',
                    'unit_type': unit_type,
                    'building_pos': self.building_pos
                }

        # Check close button
        close_y = start_y + len(self.unit_types) * option_height
        if close_y <= mouse_pos[1] <= close_y + option_height:
            return {'type': 'close'}

        return None

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the building menu.

        Args:
            screen: Pygame surface to draw on
        """
        self.screen = screen

        # Calculate menu position (to the right of the building)
        menu_x = self.building_pos[0] * TILE_SIZE + TILE_SIZE
        menu_y = self.building_pos[1] * TILE_SIZE

        # Create semi-transparent surface
        menu_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        menu_surface.fill(self.bg_color)

        # Draw border
        pygame.draw.rect(menu_surface, self.border_color,
                        (0, 0, self.width, self.height), 2)

        # Draw title
        title = get_language().get('building_menu.title', 'Create Unit')
        title_surface = self.font.render(title, True, self.text_color)
        menu_surface.blit(title_surface, (10, 10))

        # Draw unit options
        option_height = 40
        start_y = 40

        for i, unit_type in enumerate(self.unit_types):
            option_y = start_y + i * option_height

            # Get unit name and cost from UNIT_DATA
            unit_info = UNIT_DATA.get(unit_type, {})
            unit_name = unit_info.get('name', unit_type)
            cost = unit_info.get('cost', 100)

            # Display name
            display_text = f"{unit_name} (${cost})"
            text_surface = self.font.render(display_text, True, self.text_color)
            menu_surface.blit(text_surface, (10, option_y + 10))

        # Draw close button
        close_y = start_y + len(self.unit_types) * option_height
        close_text = get_language().get('common.close', 'Close')
        close_surface = self.font.render(close_text, True, self.text_color)
        menu_surface.blit(close_surface, (10, close_y + 10))

        # Blit to screen
        screen.blit(menu_surface, (menu_x, menu_y))


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
