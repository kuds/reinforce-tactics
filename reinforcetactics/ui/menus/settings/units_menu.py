"""Units settings menu for enabling/disabling unit types."""
from typing import Optional

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.settings import get_settings
from reinforcetactics.constants import UNIT_DATA


class UnitsMenu(Menu):
    """Menu for configuring which unit types are available in the game."""

    # All available unit types in order
    ALL_UNIT_TYPES = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize units menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('units.title', 'Unit Settings'))
        self.settings = get_settings()
        self._setup_options()

    def _setup_options(self) -> None:
        """Setup menu options with current unit enable/disable states."""
        lang = get_language()

        # Add toggle for each unit type
        for unit_type in self.ALL_UNIT_TYPES:
            unit_name = UNIT_DATA[unit_type]['name']
            unit_cost = UNIT_DATA[unit_type]['cost']
            is_enabled = self.settings.is_unit_enabled(unit_type)
            status = "ON" if is_enabled else "OFF"

            # Create option text with unit info
            option_text = f"{unit_name} ({unit_type}) - {unit_cost}g: [{status}]"
            self.add_option(option_text, lambda ut=unit_type: self._toggle_unit(ut))

        # Add separator-like options
        self.add_option("---", lambda: 'separator')

        # Quick actions
        self.add_option(
            lang.get('units.enable_all', 'Enable All Units'),
            self._enable_all
        )
        self.add_option(
            lang.get('units.disable_all', 'Disable All Units'),
            self._disable_all
        )
        self.add_option(
            lang.get('units.basic_only', 'Basic Units Only (W,M,C,A)'),
            self._basic_only
        )
        self.add_option(
            lang.get('units.advanced_only', 'Advanced Units Only (K,R,S,B)'),
            self._advanced_only
        )

        # Back option
        self.add_option(lang.get('common.back', 'Back'), lambda: None)

    def _refresh_options(self) -> None:
        """Refresh menu options after settings change."""
        self.options.clear()
        self._setup_options()

    def _toggle_unit(self, unit_type: str) -> str:
        """Toggle a specific unit type on/off."""
        # Ensure at least one unit remains enabled
        enabled = self.settings.get_enabled_units()
        if unit_type in enabled and len(enabled) <= 1:
            # Cannot disable the last unit
            return 'cannot_disable_last'

        self.settings.toggle_unit(unit_type)
        self._refresh_options()
        return 'toggled'

    def _enable_all(self) -> str:
        """Enable all unit types."""
        self.settings.set_enabled_units(self.ALL_UNIT_TYPES.copy())
        self._refresh_options()
        return 'toggled'

    def _disable_all(self) -> str:
        """Disable all units except Warrior (ensure at least one remains)."""
        # Keep at least Warrior enabled
        self.settings.set_enabled_units(['W'])
        self._refresh_options()
        return 'toggled'

    def _basic_only(self) -> str:
        """Enable only basic units (Warrior, Mage, Cleric, Archer)."""
        self.settings.set_enabled_units(['W', 'M', 'C', 'A'])
        self._refresh_options()
        return 'toggled'

    def _advanced_only(self) -> str:
        """Enable only advanced units (Knight, Rogue, Sorcerer, Barbarian)."""
        self.settings.set_enabled_units(['K', 'R', 'S', 'B'])
        self._refresh_options()
        return 'toggled'

    def run(self) -> Optional[str]:
        """Run the units menu loop."""
        result = None
        clock = pygame.time.Clock()

        self._populate_option_rects()
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                result = self.handle_input(event)
                if result is not None:
                    # Handle special results
                    if result in ('toggled', 'separator', 'cannot_disable_last'):
                        # Stay in menu, refresh display
                        self._populate_option_rects()
                        result = None
                    else:
                        return result

            self.draw()
            clock.tick(30)

        return result
