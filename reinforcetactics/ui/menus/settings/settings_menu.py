"""Settings menu."""
from typing import Optional, Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.settings.language_menu import LanguageMenu
from reinforcetactics.ui.menus.settings.api_keys_menu import APIKeysMenu
from reinforcetactics.ui.menus.settings.graphics_menu import GraphicsMenu
from reinforcetactics.ui.menus.settings.units_menu import UnitsMenu
from reinforcetactics.utils.language import get_language


class SettingsMenu(Menu):
    """Settings menu."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize settings menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('settings.title', 'Settings'))
        self._fullscreen = bool(pygame.display.get_surface() and
                                pygame.display.get_surface().get_flags() & pygame.FULLSCREEN)
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('settings.language', 'Language'), self._change_language)
        self.add_option(lang.get('settings.graphics', 'Graphics'), self._configure_graphics)
        self.add_option(lang.get('settings.units', 'Unit Settings'), self._configure_units)
        self.add_option(
            f"{lang.get('settings.sound', 'Sound')} (not implemented)",
            self._toggle_sound
        )
        fullscreen_status = "ON" if self._fullscreen else "OFF"
        self.add_option(
            f"{lang.get('settings.fullscreen', 'Fullscreen')}: {fullscreen_status}",
            self._toggle_fullscreen
        )
        self.add_option(lang.get('settings.api_keys', 'LLM API Keys'), self._configure_api_keys)
        self.add_option(lang.get('common.back', 'Back'), lambda: None)

    def _refresh_options(self) -> None:
        """Rebuild options with current language strings."""
        self.options.clear()
        self.title = get_language().get('settings.title', 'Settings')
        self._setup_options()

    def _change_language(self) -> str:
        """Open language selection menu."""
        return 'language_menu'

    def _configure_graphics(self) -> str:
        """Open graphics configuration menu."""
        return 'graphics_menu'

    def _configure_units(self) -> str:
        """Open unit settings menu."""
        return 'units_menu'

    def _toggle_sound(self) -> str:
        """Toggle sound on/off. Currently not implemented."""
        return 'toggled'

    def _toggle_fullscreen(self) -> str:
        """Toggle fullscreen mode."""
        pygame.display.toggle_fullscreen()
        self._fullscreen = not self._fullscreen
        self._refresh_options()
        return 'toggled'

    def _configure_api_keys(self) -> str:
        """Open API keys configuration menu."""
        return 'api_keys_menu'

    def run(self) -> Optional[Any]:
        """
        Run the settings menu loop with submenu handling.

        Returns:
            Result from selected option, or None
        """
        result = None
        clock = pygame.time.Clock()

        # Populate option_rects before event loop for click detection
        self._populate_option_rects()

        # Clear any residual events AFTER option_rects are populated
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                result = self.handle_input(event)
                if result is not None:
                    # Handle toggled settings (stay in menu)
                    if result == 'toggled':
                        self._populate_option_rects()
                    # Handle submenu navigation
                    elif result == 'language_menu':
                        language_menu = LanguageMenu(self.screen)
                        language_menu.run()
                        pygame.event.clear()
                        # Refresh options with new language strings
                        self._refresh_options()
                        self._populate_option_rects()
                    elif result == 'graphics_menu':
                        graphics_menu = GraphicsMenu(self.screen)
                        graphics_menu.run()
                        pygame.event.clear()
                        # Continue in settings menu
                    elif result == 'units_menu':
                        units_menu = UnitsMenu(self.screen)
                        units_menu.run()
                        pygame.event.clear()
                        # Continue in settings menu
                    elif result == 'api_keys_menu':
                        api_keys_menu = APIKeysMenu(self.screen)
                        api_keys_menu.run()
                        pygame.event.clear()
                        # Continue in settings menu
                    else:
                        # For other results (like None from Back button), exit
                        return result

            self.draw()
            clock.tick(30)

        return result
