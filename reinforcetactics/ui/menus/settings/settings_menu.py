"""Settings menu."""
import sys
from typing import Optional, Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.settings.language_menu import LanguageMenu
from reinforcetactics.ui.menus.settings.api_keys_menu import APIKeysMenu
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
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('settings.language', 'Language'), self._change_language)
        self.add_option(lang.get('settings.sound', 'Sound'), self._toggle_sound)
        self.add_option(lang.get('settings.fullscreen', 'Fullscreen'), self._toggle_fullscreen)
        self.add_option(lang.get('settings.api_keys', 'LLM API Keys'), self._configure_api_keys)
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
                    pygame.quit()
                    sys.exit()

                result = self.handle_input(event)
                if result is not None:
                    # Handle submenu navigation
                    if result == 'language_menu':
                        language_menu = LanguageMenu(self.screen)
                        language_menu.run()
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
