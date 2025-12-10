"""In-game pause menu."""
from typing import Optional

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language


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
