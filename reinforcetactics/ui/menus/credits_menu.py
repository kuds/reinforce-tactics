"""Credits menu."""

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language


class CreditsMenu(Menu):
    """Credits menu."""

    def __init__(self, screen: pygame.Surface | None = None) -> None:
        """
        Initialize credits menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get("credits.title", "Credits"))
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get("common.back", "Back"), lambda: None)

    def _options_start_y(self) -> int:
        """Start the option list (Back button) below the credits text."""
        return int(self.screen.get_height() * 0.7)

    def _options_min_top(self) -> int:
        """Never let the layout pull the Back button up into the credits."""
        return self._options_start_y()

    def _draw_content(self) -> None:
        """Draw the credits menu with additional information."""
        # Draw base menu (title, background, options)
        super()._draw_content()

        # Get language for translations
        lang = get_language()
        screen_cx = self.screen.get_width() // 2

        # Draw game title
        game_title = lang.get("credits.game_title", "REINFORCE TACTICS")
        title_surface = self.title_font.render(game_title, True, self.selected_color)
        y = 140
        self.screen.blit(title_surface, title_surface.get_rect(centerx=screen_cx, y=y))
        y += title_surface.get_height() + 40

        # Draw developer info
        developer_label = lang.get("credits.developer", "Developer:")
        developer_name = lang.get("credits.developer_name", "Michael Kudlaty")

        developer_label_surface = self.option_font.render(developer_label, True, self.text_color)
        self.screen.blit(developer_label_surface, developer_label_surface.get_rect(centerx=screen_cx, y=y))
        y += developer_label_surface.get_height() + 6

        developer_name_surface = self.option_font.render(developer_name, True, self.selected_color)
        self.screen.blit(developer_name_surface, developer_name_surface.get_rect(centerx=screen_cx, y=y))
        y += developer_name_surface.get_height() + 30

        # Draw description
        description = lang.get("credits.description", "Turn-Based Strategy Game with Reinforcement Learning")
        description_surface = self.option_font.render(description, True, self.text_color)
        self.screen.blit(description_surface, description_surface.get_rect(centerx=screen_cx, y=y))
