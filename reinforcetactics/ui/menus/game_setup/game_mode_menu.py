"""Menu for selecting game mode."""
import os
from typing import Optional, List

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language


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
