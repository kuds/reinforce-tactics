"""Menu for selecting game mode."""

import os

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.game_setup.modes import GAME_MODE_PLAYER_COUNTS
from reinforcetactics.utils.language import get_language


class GameModeMenu(Menu):
    """Menu for selecting game mode (1v1, 1v1v1 or 2v2)."""

    def __init__(self, screen: pygame.Surface | None = None, maps_dir: str = "maps") -> None:
        """
        Initialize game mode menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            maps_dir: Directory containing map subdirectories
        """
        super().__init__(screen, get_language().get("new_game.select_mode", "Select Game Mode"))
        self.maps_dir = maps_dir
        self.available_modes: list[str] = []
        self._load_modes()
        self._setup_options()

    def _load_modes(self) -> None:
        """Discover the supported game modes that have at least one map.

        Only modes in ``GAME_MODE_PLAYER_COUNTS`` are offered. Listing every
        ``maps/`` subfolder used to offer folders the rest of the New Game
        flow could not start (``1v1v1`` crashed PlayerConfigMenu), and would
        do the same for any future non-mode folder such as scenarios.
        """
        if os.path.exists(self.maps_dir):
            for item in GAME_MODE_PLAYER_COUNTS:
                item_path = os.path.join(self.maps_dir, item)
                if os.path.isdir(item_path):
                    # Check if folder contains .csv maps
                    try:
                        if any(f.endswith(".csv") for f in os.listdir(item_path)):
                            self.available_modes.append(item)
                    except (OSError, PermissionError):
                        # Skip directories that can't be read
                        continue
        self.available_modes.sort()

    def _setup_options(self) -> None:
        """Setup menu options for available game modes."""
        for mode in self.available_modes:

            def make_callback(m: str = mode) -> str:
                return m

            self.add_option(mode, make_callback)
        self.add_option(get_language().get("common.back", "Back"), lambda: None)

    def run(self) -> str | None:
        """
        Run game mode selection menu.

        Returns:
            Selected game mode string (e.g., "1v1", "1v1v1" or "2v2"), or None if cancelled
        """
        return super().run()
