"""Settings menu."""

from typing import Any

import pygame

from reinforcetactics.ui.menus.base import Menu, drain_events
from reinforcetactics.ui.menus.settings.api_keys_menu import APIKeysMenu
from reinforcetactics.ui.menus.settings.graphics_menu import GraphicsMenu
from reinforcetactics.ui.menus.settings.language_menu import LanguageMenu
from reinforcetactics.ui.menus.settings.units_menu import UnitsMenu
from reinforcetactics.utils.language import get_language


class SettingsMenu(Menu):
    """Settings menu."""

    def __init__(self, screen: pygame.Surface | None = None) -> None:
        """
        Initialize settings menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get("settings.title", "Settings"))
        surface = pygame.display.get_surface()
        self._fullscreen = bool(surface and surface.get_flags() & pygame.FULLSCREEN)
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get("settings.language", "Language"), self._change_language)
        self.add_option(lang.get("settings.graphics", "Graphics"), self._configure_graphics)
        self.add_option(lang.get("settings.units", "Unit Settings"), self._configure_units)
        self.add_option(f"{lang.get('settings.sound', 'Sound')} (not implemented)", self._toggle_sound)
        fullscreen_status = "ON" if self._fullscreen else "OFF"
        self.add_option(f"{lang.get('settings.fullscreen', 'Fullscreen')}: {fullscreen_status}", self._toggle_fullscreen)
        self.add_option(lang.get("settings.api_keys", "LLM API Keys"), self._configure_api_keys)
        self.add_option(lang.get("common.back", "Back"), lambda: None)

    def _refresh_options(self) -> None:
        """Rebuild options with current language strings."""
        self.clear_options()
        self.title = get_language().get("settings.title", "Settings")
        self._setup_options()

    def _change_language(self) -> str:
        """Open language selection menu."""
        return "language_menu"

    def _configure_graphics(self) -> str:
        """Open graphics configuration menu."""
        return "graphics_menu"

    def _configure_units(self) -> str:
        """Open unit settings menu."""
        return "units_menu"

    def _toggle_sound(self) -> str:
        """Toggle sound on/off. Currently not implemented."""
        return "toggled"

    def _toggle_fullscreen(self) -> str:
        """Toggle fullscreen mode."""
        pygame.display.toggle_fullscreen()
        self._fullscreen = not self._fullscreen
        self._refresh_options()
        return "toggled"

    def _configure_api_keys(self) -> str:
        """Open API keys configuration menu."""
        return "api_keys_menu"

    # Sentinel returned by an option -> sub-menu class to open for it.
    _SUBMENUS = {
        "language_menu": LanguageMenu,
        "graphics_menu": GraphicsMenu,
        "units_menu": UnitsMenu,
        "api_keys_menu": APIKeysMenu,
    }

    def _on_result(self, result: Any) -> tuple[bool, Any]:
        """Open the requested sub-menu (or absorb a toggle) and stay put."""
        if result == "toggled":
            return False, None

        submenu_cls = self._SUBMENUS.get(result)
        if submenu_cls is None:
            # Anything else (e.g. Back) is the caller's answer.
            return True, result

        submenu_cls(self.screen).run()
        drain_events()
        if result == "language_menu":
            # Rebuild with the newly selected language's strings.
            self._refresh_options()
        return False, None
