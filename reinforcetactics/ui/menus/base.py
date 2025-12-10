"""
Base menu class and helper functions for the menu system.
Self-contained menus that manage their own pygame screen and navigation.
"""
from __future__ import annotations
import sys
from typing import Optional, List, Tuple, Callable, Any

import pygame

from reinforcetactics.utils.language import get_language, TRANSLATIONS


# Cache all "Back" button translations from the language system
_BACK_TRANSLATIONS_CACHE = None


def _get_back_translations() -> set:
    """
    Get all translations of the "Back" button text.

    Returns:
        Set of lowercase, stripped back button translations
    """
    global _BACK_TRANSLATIONS_CACHE
    if _BACK_TRANSLATIONS_CACHE is None:
        back_translations = set()
        for lang_dict in TRANSLATIONS.values():
            back_text = lang_dict.get('common.back')
            if back_text:
                # Strip whitespace to match the checking logic
                back_translations.add(back_text.lower().strip())
        _BACK_TRANSLATIONS_CACHE = back_translations
    return _BACK_TRANSLATIONS_CACHE


class Menu:
    """Base class for game menus. Manages its own screen if not provided."""

    def __init__(self, screen: Optional[pygame.Surface] = None, title: str = "") -> None:
        """
        Initialize the menu.

        Args:
            screen: Optional pygame display surface. If None, creates its own.
            title: Menu title
        """
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Create screen if not provided
        self.owns_screen = screen is None
        if self.owns_screen:
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Reinforce Tactics")
            # Initialize clipboard support when we own the screen
            try:
                pygame.scrap.init()
            except pygame.error:
                # Clipboard not available on this platform
                pass
        else:
            self.screen = screen

        self.title = title
        self.running = True
        self.selected_index = 0
        self.options: List[Tuple[str, Callable[[], Any]]] = []

        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.selected_color = (255, 200, 50)
        self.hover_color = (200, 180, 100)
        self.title_color = (100, 200, 255)
        self.option_bg_color = (50, 50, 65)
        self.option_bg_hover_color = (70, 70, 90)
        self.option_bg_selected_color = (80, 80, 100)

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.option_font = pygame.font.Font(None, 36)
        self.indicator_font = pygame.font.Font(None, 24)

        # Mouse tracking
        self.hover_index = -1
        self.option_rects: List[pygame.Rect] = []

        # Scrolling support
        self.scroll_offset = 0
        self.max_visible_options = 8  # Maximum options visible at once
        self.option_spacing = 60

        # Get language instance
        self.lang = get_language()

    def add_option(self, text: str, callback: Callable[[], Any]) -> None:
        """Add a menu option."""
        self.options.append((text, callback))

    def handle_input(self, event: pygame.event.Event) -> Optional[Any]:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            Result of selected option callback, if any
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.options)
                self._ensure_selected_visible()
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)
                self._ensure_selected_visible()
            elif event.key == pygame.K_RETURN:
                if self.options:
                    text, callback = self.options[self.selected_index]
                    result = callback()
                    # If callback returns None and it's a Back button, exit the menu
                    if result is None and self._is_back_option(text):
                        self.running = False
                    return result
            elif event.key == pygame.K_ESCAPE:
                self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos
                # Check if any option was clicked
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(mouse_pos):
                        # Get the actual option index accounting for scroll
                        actual_index = i + self.scroll_offset
                        if actual_index < len(self.options):
                            self.selected_index = actual_index
                            text, callback = self.options[actual_index]
                            result = callback()
                            # If callback returns None and it's a Back button, exit the menu
                            if result is None and self._is_back_option(text):
                                self.running = False
                            return result
            elif event.button == 4:  # Mouse wheel up
                self.scroll_offset = max(0, self.scroll_offset - 1)
            elif event.button == 5:  # Mouse wheel down
                max_scroll = max(0, len(self.options) - self.max_visible_options)
                self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_index = -1
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(mouse_pos):
                    self.hover_index = i + self.scroll_offset
                    break

        return None

    def _is_back_option(self, text: str) -> bool:
        """
        Check if an option text represents a Back button.

        Args:
            text: The option text to check

        Returns:
            True if the option is a Back button, False otherwise
        """
        # Check if the text matches any Back translation (using cached list)
        return text.lower().strip() in _get_back_translations()

    def _ensure_selected_visible(self) -> None:
        """Ensure the selected option is visible by adjusting scroll offset."""
        if self.selected_index < self.scroll_offset:
            self.scroll_offset = self.selected_index
        elif self.selected_index >= self.scroll_offset + self.max_visible_options:
            self.scroll_offset = self.selected_index - self.max_visible_options + 1

    def _populate_option_rects(self) -> None:
        """Populate option_rects for click detection without drawing to screen."""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        start_y = screen_height // 3
        spacing = self.option_spacing
        self.option_rects = []

        # Calculate maximum option width for uniform sizing
        padding_x = 40
        padding_y = 10
        max_text_width = 0
        for text, _ in self.options:
            display_text = f"> {text}"
            text_surface = self.option_font.render(display_text, True, self.text_color)
            max_text_width = max(max_text_width, text_surface.get_width())

        uniform_width = max_text_width + 2 * padding_x

        # Determine which options to display (with scrolling)
        total_options = len(self.options)
        start_index = self.scroll_offset
        end_index = min(total_options, start_index + self.max_visible_options)

        # Calculate rects for visible options
        for display_i, option_i in enumerate(range(start_index, end_index)):
            text, _ = self.options[option_i]
            is_selected = option_i == self.selected_index
            display_text = f"> {text}" if is_selected else f"  {text}"

            text_surface = self.option_font.render(display_text, True, self.text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2,
                                              y=start_y + display_i * spacing)

            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,
                text_rect.y - padding_y,
                uniform_width,
                text_rect.height + 2 * padding_y
            )

            self.option_rects.append(bg_rect)

    def draw(self) -> None:
        """Draw the menu."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
            self.screen.blit(title_surface, title_rect)

        # Draw options with scrolling support
        start_y = screen_height // 3
        spacing = self.option_spacing
        self.option_rects = []

        # Calculate maximum option width for uniform sizing
        padding_x = 40
        padding_y = 10
        max_text_width = 0
        for text, _ in self.options:
            display_text = f"> {text}"  # Use the selected format for consistent width
            text_surface = self.option_font.render(display_text, True, self.text_color)
            max_text_width = max(max_text_width, text_surface.get_width())

        uniform_width = max_text_width + 2 * padding_x

        # Determine which options to display (with scrolling)
        total_options = len(self.options)
        start_index = self.scroll_offset
        end_index = min(total_options, start_index + self.max_visible_options)

        # Draw visible options
        for display_i, option_i in enumerate(range(start_index, end_index)):
            text, _ = self.options[option_i]

            # Determine styling based on state
            is_selected = option_i == self.selected_index
            is_hovered = option_i == self.hover_index

            # Choose colors
            if is_selected:
                text_color = self.selected_color
                bg_color = self.option_bg_selected_color
            elif is_hovered:
                text_color = self.hover_color
                bg_color = self.option_bg_hover_color
            else:
                text_color = self.text_color
                bg_color = self.option_bg_color

            # Add selection indicator
            display_text = f"> {text}" if is_selected else f"  {text}"

            # Render text
            text_surface = self.option_font.render(display_text, True, text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2,
                                              y=start_y + display_i * spacing)

            # Create background rectangle with uniform width
            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,  # Center the uniform-width box
                text_rect.y - padding_y,
                uniform_width,
                text_rect.height + 2 * padding_y
            )

            # Draw rounded background rectangle
            pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=8)

            # Draw border for selected/hovered
            if is_selected or is_hovered:
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(self.screen, border_color, bg_rect, width=2, border_radius=8)

            # Draw text
            self.screen.blit(text_surface, text_rect)

            # Store rect for click detection
            self.option_rects.append(bg_rect)

        # Draw scroll indicators if needed
        if total_options > self.max_visible_options:
            # Show up arrow if not at top
            if self.scroll_offset > 0:
                up_text = self.indicator_font.render("▲ Scroll Up", True, self.hover_color)
                up_rect = up_text.get_rect(centerx=screen_width // 2, y=start_y - 30)
                self.screen.blit(up_text, up_rect)

            # Show down arrow if not at bottom
            if end_index < total_options:
                down_text = self.indicator_font.render("▼ Scroll Down", True, self.hover_color)
                down_y = start_y + self.max_visible_options * spacing + 10
                down_rect = down_text.get_rect(centerx=screen_width // 2, y=down_y)
                self.screen.blit(down_text, down_rect)

            # Show position indicator (e.g., "3 / 15")
            pos_text = self.indicator_font.render(
                f"{self.scroll_offset + 1}-{end_index} / {total_options}",
                True, self.text_color
            )
            pos_rect = pos_text.get_rect(right=screen_width - 20, bottom=screen_height - 20)
            self.screen.blit(pos_text, pos_rect)

        pygame.display.flip()

    def run(self) -> Optional[Any]:
        """
        Run the menu loop.

        Returns:
            Result from selected option, or None
        """
        result = None
        clock = pygame.time.Clock()

        # Populate option_rects before event loop for click detection
        # Don't call draw() here to avoid double-display issue
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
                    return result

            self.draw()
            clock.tick(30)

        return result
