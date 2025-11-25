"""
Menu system for the strategy game.
Fixed version: Added datetime import, fixed language reset mechanism.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime  # FIX: Added missing import
from typing import Optional, List, Tuple, Callable, Any, Dict

import pygame

from reinforcetactics.constants import TILE_SIZE
from reinforcetactics.utils.language import Language, get_language, reset_language


class Menu:
    """Base class for game menus."""

    def __init__(self, screen: pygame.Surface, title: str = "") -> None:
        """
        Initialize the menu.
        
        Args:
            screen: Pygame display surface
            title: Menu title
        """
        self.screen = screen
        self.title = title
        self.running = True
        self.selected_index = 0
        self.options: List[Tuple[str, Callable[[], Any]]] = []
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.selected_color = (255, 200, 50)
        self.title_color = (100, 200, 255)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.option_font = pygame.font.Font(None, 36)
        
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
        spacing = 50
        
        for i, (text, _) in enumerate(self.options):
            color = self.selected_color if i == self.selected_index else self.text_color
            
            # Add selection indicator
            display_text = f"> {text}" if i == self.selected_index else f"  {text}"
            
            text_surface = self.option_font.render(display_text, True, color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2, y=start_y + i * spacing)
            self.screen.blit(text_surface, text_rect)
        
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
    """Main menu for the game."""

    def __init__(self, screen: pygame.Surface) -> None:
        super().__init__(screen, self._get_title())
        self._setup_options()

    def _get_title(self) -> str:
        return get_language().get('main_menu.title', 'Reinforce Tactics')

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('main_menu.new_game', 'New Game'), lambda: 'new_game')
        self.add_option(lang.get('main_menu.load_game', 'Load Game'), lambda: 'load_game')
        self.add_option(lang.get('main_menu.settings', 'Settings'), lambda: 'settings')
        self.add_option(lang.get('main_menu.quit', 'Quit'), lambda: 'quit')


class NewGameMenu(Menu):
    """Menu for starting a new game."""

    def __init__(self, screen: pygame.Surface, maps_dir: str = "maps") -> None:
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
                            self.available_maps.append(os.path.join(subdir, f))

    def _setup_options(self) -> None:
        for map_file in self.available_maps:
            display_name = os.path.splitext(os.path.basename(map_file))[0]
            self.add_option(display_name, lambda m=map_file: m)
        
        self.add_option(get_language().get('common.back', 'Back'), lambda: None)


class SaveGameMenu(Menu):
    """Menu for saving the game."""

    def __init__(self, screen: pygame.Surface, game_state: Any) -> None:
        super().__init__(screen, get_language().get('save_game.title', 'Save Game'))
        self.game_state = game_state
        # FIX: datetime is now properly imported at module level
        self.input_text = f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.input_active = True

    def handle_input(self, event: pygame.event.Event) -> Optional[str]:
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
        filepath = self.game_state.save_to_file(f"saves/{self.input_text}.json")
        return filepath

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
        instructions = get_language().get('save_game.instructions', 'Press ENTER to save, ESC to cancel')
        inst_surface = self.option_font.render(instructions, True, (150, 150, 150))
        inst_rect = inst_surface.get_rect(centerx=screen_width // 2, y=screen_height // 2 + 50)
        self.screen.blit(inst_surface, inst_rect)
        
        pygame.display.flip()


class LoadGameMenu(Menu):
    """Menu for loading saved games."""

    def __init__(self, screen: pygame.Surface, saves_dir: str = "saves") -> None:
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
        for save_file in self.save_files:
            display_name = os.path.splitext(save_file)[0]
            filepath = os.path.join(self.saves_dir, save_file)
            self.add_option(display_name, lambda p=filepath: p)
        
        self.add_option(get_language().get('common.back', 'Back'), lambda: None)


class SettingsMenu(Menu):
    """Settings menu."""

    def __init__(self, screen: pygame.Surface) -> None:
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
        """Toggle sound on/off."""
        # TODO: Implement sound toggle
        pass

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

    def __init__(self, screen: pygame.Surface) -> None:
        super().__init__(screen, get_language().get('language.title', 'Select Language'))
        self._setup_options()

    def _setup_options(self) -> None:
        for code, name in self.LANGUAGES.items():
            self.add_option(name, lambda c=code: self._set_language(c))
        
        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def _set_language(self, lang_code: str) -> str:
        """
        Set the game language.
        
        FIX: Use proper reset_language() function instead of trying to
        mutate module-level variable.
        """
        reset_language(lang_code)
        self.lang = get_language()  # Refresh our reference
        return lang_code


class PauseMenu(Menu):
    """In-game pause menu."""

    def __init__(self, screen: pygame.Surface) -> None:
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

    def __init__(self, screen: pygame.Surface, winner: int, game_state: Any) -> None:
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
        winner_text = lang.get('game_over.winner', 'Player {player} Wins!').format(player=self.winner)
        
        winner_surface = self.title_font.render(winner_text, True, self.selected_color)
        winner_rect = winner_surface.get_rect(
            centerx=self.screen.get_width() // 2,
            y=100
        )
        self.screen.blit(winner_surface, winner_rect)
        
        pygame.display.flip()
