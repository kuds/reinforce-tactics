"""Language selection menu."""

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import LANGUAGE_CODES, get_language, reset_language


class LanguageMenu(Menu):
    """Language selection menu."""

    LANGUAGES = {"en": "English", "fr": "Français", "ko": "한국어", "es": "Español", "zh": "中文"}

    def __init__(self, screen: pygame.Surface | None = None) -> None:
        """
        Initialize language menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get("language.title", "Select Language"))
        self._setup_options()

    def _setup_options(self) -> None:
        current = get_language().current_language
        for code, name in self.LANGUAGES.items():

            def make_callback(c: str = code) -> str:
                return self._set_language(c)

            # Mark the active language so users can see the current setting
            label = f"● {name}" if LANGUAGE_CODES.get(code, code) == current else f"   {name}"
            self.add_option(label, make_callback)

        self.add_option(get_language().get("common.back", "Back"), lambda: None)

    def _set_language(self, lang_code: str) -> str:
        """Set the game language."""
        reset_language(lang_code)
        self.lang = get_language()  # Refresh our reference

        # Clear existing options
        self.clear_options()

        # Update menu title to use the new language
        self.title = self.lang.get("language.title", "Select Language")

        # Rebuild menu options with new language
        self._setup_options()

        return lang_code
