"""
Base menu class and helper functions for the menu system.
Self-contained menus that manage their own pygame screen and navigation.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.icons import get_arrow_down_icon, get_arrow_up_icon
from reinforcetactics.utils.clipboard import init_clipboard
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import TRANSLATIONS, get_language

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
            back_text = lang_dict.get("common.back")
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
            self.screen = pygame.display.set_mode((900, 700))
            pygame.display.set_caption("Reinforce Tactics")
            init_clipboard()
        else:
            assert screen is not None
            self.screen = screen

        self.title = title
        self.running = True
        self.selected_index = 0
        self.options: List[Tuple[str, Callable[[], Any]]] = []
        # Parallel list: self.option_enabled[i] gates whether options[i] can
        # receive focus or clicks. Kept separate from `options` to preserve
        # the public 2-tuple shape subclasses may rely on.
        self.option_enabled: List[bool] = []

        # Colors (from shared theme)
        self.bg_color = theme.BG
        self.text_color = theme.TEXT
        self.text_disabled_color = theme.TEXT_DISABLED
        self.selected_color = theme.SELECTED
        self.hover_color = theme.HOVER
        self.title_color = theme.TITLE
        self.option_bg_color = theme.OPTION_BG
        self.option_bg_hover_color = theme.OPTION_BG_HOVER
        self.option_bg_selected_color = theme.OPTION_BG_SELECTED
        self.option_bg_disabled_color = theme.OPTION_BG_DISABLED

        # Fonts
        self.title_font = get_font(theme.FONT_SIZE_TITLE)
        self.option_font = get_font(theme.FONT_SIZE_OPTION)
        self.indicator_font = get_font(theme.FONT_SIZE_INDICATOR)

        # Mouse tracking
        self.hover_index = -1
        self.option_rects: List[pygame.Rect] = []

        # Scrolling support
        self.scroll_offset = 0
        self.max_visible_options = 8  # Maximum options visible at once
        self.option_spacing = theme.MENU_OPTION_SPACING

        # Get language instance
        self.lang = get_language()

    def add_option(self, text: str, callback: Callable[[], Any], enabled: bool = True) -> None:
        """Add a menu option.

        Args:
            text: Label for the option.
            callback: Called when the option is selected.
            enabled: When False, the option is grayed out and cannot be
                focused or clicked. Useful for unavailable actions without
                hiding them from the user.
        """
        # Guard against empty text to prevent pygame "Text has zero width" error
        if not text:
            text = "(Empty)"
        self.options.append((text, callback))
        self.option_enabled.append(enabled)

    def _is_enabled(self, index: int) -> bool:
        """Whether the option at ``index`` is enabled. Defaults to True if
        a subclass appends to ``self.options`` without going through
        ``add_option`` (keeps backward compatibility with legacy call sites).
        """
        return index < len(self.option_enabled) and self.option_enabled[index]

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
                self._move_selection(-1)
                self._ensure_selected_visible()
            elif event.key == pygame.K_DOWN:
                self._move_selection(1)
                self._ensure_selected_visible()
            elif event.key == pygame.K_RETURN:
                if self.options and self._is_enabled(self.selected_index):
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
                        if actual_index < len(self.options) and self._is_enabled(actual_index):
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

    def _move_selection(self, delta: int) -> None:
        """Advance ``self.selected_index`` by ``delta`` while skipping
        disabled options. Wraps around. No-op if every option is disabled.
        """
        n = len(self.options)
        if n == 0:
            return
        # If every option is disabled, just wrap normally without infinite loop.
        if not any(self.option_enabled[: len(self.option_enabled)]) and len(self.option_enabled) >= n:
            self.selected_index = (self.selected_index + delta) % n
            return
        step = 1 if delta >= 0 else -1
        idx = self.selected_index
        for _ in range(n):
            idx = (idx + step) % n
            if self._is_enabled(idx):
                self.selected_index = idx
                return
        # Fallback: leave selection as-is.

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

    def _layout_visible_options(self) -> List[Tuple[int, str, pygame.Rect]]:
        """Compute the on-screen layout for the currently visible options.

        Single source of truth for geometry — used by both click-hit testing
        (``_populate_option_rects``) and drawing (``_draw_content``) so the
        two cannot drift apart.

        Returns:
            List of ``(option_index, safe_text, bg_rect)`` for each visible
            option. ``safe_text`` is the option label guarded against the
            empty string (pygame raises on zero-width text).
        """
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        start_y = screen_height // 3
        spacing = self.option_spacing
        padding_x = theme.OPTION_PADDING_X
        padding_y = theme.OPTION_PADDING_Y

        def safe(text: str) -> str:
            return text if text else "(Empty)"

        # Uniform option width: use the widest rendered label (in "> " form
        # since that's the wider of the two prefixes).
        max_text_width = 0
        for text, _ in self.options:
            surface = self.option_font.render(f"> {safe(text)}", True, self.text_color)
            max_text_width = max(max_text_width, surface.get_width())
        uniform_width = max_text_width + 2 * padding_x

        start_index = self.scroll_offset
        end_index = min(len(self.options), start_index + self.max_visible_options)

        # Approximate row height from a single rendered line so bg_rect y-pos
        # doesn't depend on which label happens to be tallest.
        sample = self.option_font.render("Ag", True, self.text_color)
        row_height = sample.get_height()

        layout: List[Tuple[int, str, pygame.Rect]] = []
        for display_i, option_i in enumerate(range(start_index, end_index)):
            text, _ = self.options[option_i]
            text = safe(text)
            row_y = start_y + display_i * spacing
            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,
                row_y - padding_y,
                uniform_width,
                row_height + 2 * padding_y,
            )
            layout.append((option_i, text, bg_rect))
        return layout

    def _populate_option_rects(self) -> None:
        """Populate option_rects for click detection without drawing to screen."""
        self.option_rects = [bg_rect for _, _, bg_rect in self._layout_visible_options()]

    def _draw_content(self) -> None:
        """Draw the menu content without flipping the display.

        Subclasses can override this to add custom content (e.g., winner text,
        credits info) and call super()._draw_content() for the base rendering.
        The draw() method calls this then flips the display once.
        """
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
            self.screen.blit(title_surface, title_rect)

        layout = self._layout_visible_options()
        self.option_rects = [bg_rect for _, _, bg_rect in layout]

        for option_i, text, bg_rect in layout:
            is_enabled = self._is_enabled(option_i)
            is_selected = option_i == self.selected_index
            is_hovered = option_i == self.hover_index

            if not is_enabled:
                text_color = self.text_disabled_color
                bg_color = self.option_bg_disabled_color
            elif is_selected:
                text_color = self.selected_color
                bg_color = self.option_bg_selected_color
            elif is_hovered:
                text_color = self.hover_color
                bg_color = self.option_bg_hover_color
            else:
                text_color = self.text_color
                bg_color = self.option_bg_color

            display_text = f"> {text}" if is_selected and is_enabled else f"  {text}"
            text_surface = self.option_font.render(display_text, True, text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2, centery=bg_rect.centery)

            pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=theme.BORDER_RADIUS)

            if is_enabled and (is_selected or is_hovered):
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(
                    self.screen,
                    border_color,
                    bg_rect,
                    width=theme.BORDER_WIDTH_HOVER,
                    border_radius=theme.BORDER_RADIUS,
                )

            self.screen.blit(text_surface, text_rect)

        # Data needed for the scroll indicator below.
        start_y = screen_height // 3
        spacing = self.option_spacing
        total_options = len(self.options)
        end_index = min(total_options, self.scroll_offset + self.max_visible_options)

        # Draw scroll indicators if needed
        if total_options > self.max_visible_options:
            # Show up arrow if not at top
            if self.scroll_offset > 0:
                up_icon = get_arrow_up_icon(size=20, color=self.hover_color)
                up_text = self.indicator_font.render(" Scroll Up", True, self.hover_color)
                # Calculate total width for centering
                total_width = up_icon.get_width() + up_text.get_width()
                icon_x = (screen_width - total_width) // 2
                text_x = icon_x + up_icon.get_width()
                icon_y = start_y - 30
                self.screen.blit(up_icon, (icon_x, icon_y))
                self.screen.blit(up_text, (text_x, icon_y))

            # Show down arrow if not at bottom
            if end_index < total_options:
                down_icon = get_arrow_down_icon(size=20, color=self.hover_color)
                down_text = self.indicator_font.render(" Scroll Down", True, self.hover_color)
                # Calculate total width for centering
                total_width = down_icon.get_width() + down_text.get_width()
                icon_x = (screen_width - total_width) // 2
                text_x = icon_x + down_icon.get_width()
                down_y = start_y + self.max_visible_options * spacing + 10
                self.screen.blit(down_icon, (icon_x, down_y))
                self.screen.blit(down_text, (text_x, down_y))

            # Show position indicator (e.g., "3 / 15")
            pos_text = self.indicator_font.render(
                f"{self.scroll_offset + 1}-{end_index} / {total_options}", True, self.text_color
            )
            pos_rect = pos_text.get_rect(right=screen_width - 20, bottom=screen_height - 20)
            self.screen.blit(pos_text, pos_rect)

    def draw(self) -> None:
        """Draw the menu and flip the display."""
        self._draw_content()
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
                    self.running = False
                    # Re-post QUIT event for parent menu if we don't own the screen
                    # This ensures the parent can handle the quit properly
                    if not self.owns_screen:
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
                    return None

                result = self.handle_input(event)
                if result is not None:
                    return result

            self.draw()
            clock.tick(theme.MENU_FRAMERATE)

        return result
