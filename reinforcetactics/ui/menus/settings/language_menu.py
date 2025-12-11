"""Language selection menu."""
from typing import Optional

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language, reset_language


class LanguageMenu(Menu):
    """Language selection menu."""

    LANGUAGES = {
        'en': 'English',
        'fr': 'Français',
        'ko': '한국어',
        'es': 'Español',
        'zh': '中文'
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
