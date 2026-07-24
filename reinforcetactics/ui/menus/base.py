"""
Base menu class and helper functions for the menu system.
Self-contained menus that manage their own pygame screen and navigation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.icons import get_arrow_down_icon, get_arrow_up_icon
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.clipboard import init_clipboard
from reinforcetactics.utils.fonts import get_display_font, get_font
from reinforcetactics.utils.language import TRANSLATIONS, get_language

# Below this window height the screen switches to a compact layout (no
# footer hint, icon-only scroll arrows): in-game screens are sized to the
# map (``grid * TILE_SIZE``) and can be as small as 320px, where every pixel
# of vertical space belongs to the options themselves.
COMPACT_LAYOUT_MIN_HEIGHT = 480

# Cache all "Back" button translations from the language system
_BACK_TRANSLATIONS_CACHE = None


def drain_events() -> None:
    """Drop input left over from a sub-screen, keeping any window-close.

    Menus flush the event queue after a sub-menu returns so a stray click
    made on the sub-menu doesn't land on whatever is now under the cursor.
    A blanket ``pygame.event.clear()`` also swallowed the QUIT that the
    sub-menu re-posts on its way out, so clicking the window's close button
    from a nested screen left the window open.
    """
    had_quit = bool(pygame.event.get(pygame.QUIT))
    pygame.event.clear()
    if had_quit:
        pygame.event.post(pygame.event.Event(pygame.QUIT))


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

    def __init__(self, screen: pygame.Surface | None = None, title: str = "") -> None:
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
        self.options: list[tuple[str, Callable[[], Any]]] = []
        # Parallel list: self.option_enabled[i] gates whether options[i] can
        # receive focus or clicks. Kept separate from `options` to preserve
        # the public 2-tuple shape subclasses may rely on.
        self.option_enabled: list[bool] = []

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

        # Fonts (titles use the pixel-styled display font)
        self.title_font = get_display_font(theme.FONT_SIZE_TITLE)
        self.option_font = get_font(theme.FONT_SIZE_OPTION)
        self.indicator_font = get_font(theme.FONT_SIZE_INDICATOR)

        # Mouse tracking
        self.hover_index = -1
        self.option_rects: list[pygame.Rect] = []

        # Scrolling support. ``max_visible_options`` is recomputed from the
        # window on every layout pass (see ``_options_geometry``) so options
        # can never be drawn past the bottom edge; the theme value is only
        # the upper bound.
        self.scroll_offset = 0
        self.max_visible_options = theme.MAX_VISIBLE_OPTIONS
        # Guarantee rows never overlap even if the option font is taller
        # than the theme's nominal spacing allows for.
        row_height = self.option_font.get_height()
        self.option_spacing = max(theme.MENU_OPTION_SPACING, row_height + 2 * theme.OPTION_PADDING_Y + 4)

        # Memoized uniform option width, invalidated when the labels change.
        self._width_cache: tuple[tuple[str, ...], int] | None = None

        # Get language instance
        self.lang = get_language()

        # One-line control hint drawn at the bottom of the screen. Set to
        # None in a subclass to suppress it.
        self.footer_hint: str | None = self.lang.get("common.menu_hint", "Arrows: Move   Enter: Select   Esc: Back")

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

    def clear_options(self) -> None:
        """Remove all options (for menus that rebuild their option list).

        Clears ``option_enabled`` together with ``options`` — clearing only
        one desyncs the parallel lists and leaves stale enabled flags
        gating the rebuilt options.
        """
        self.options.clear()
        self.option_enabled.clear()

    def _is_enabled(self, index: int) -> bool:
        """Whether the option at ``index`` is enabled. Defaults to True if
        a subclass appends to ``self.options`` without going through
        ``add_option`` (keeps backward compatibility with legacy call sites).
        """
        return index < len(self.option_enabled) and self.option_enabled[index]

    def handle_input(self, event: pygame.event.Event) -> Any | None:
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

    def _options_start_y(self) -> int:
        """Preferred Y coordinate where the option list starts.

        Subclasses that draw custom content above the options (e.g. the
        credits screen) can override this to push the list down. The value
        is a preference: ``_options_geometry`` moves the list up when that
        is what it takes to fit more rows on screen, but never above
        ``_options_min_top``.
        """
        return self.screen.get_height() // 3

    def _options_min_top(self) -> int:
        """Hard floor for the option list — nothing is drawn above this.

        Defaults to just below the screen title, leaving room for the
        "Scroll Up" affordance. Subclasses that draw a banner between the
        title and the options (e.g. the game-over screen) raise this so the
        two can never overlap on a short window.
        """
        title_bottom = theme.TITLE_MARGIN_Y + self.title_font.get_height() if self.title else 0
        return title_bottom + theme.OPTIONS_MIN_TOP_GAP

    def _options_geometry(self) -> tuple[int, int]:
        """Resolve where the option list starts and how many rows fit.

        The in-game screen is sized to the map (``grid * TILE_SIZE``), so it
        can be as small as 320x320 — far too short for the eight rows the
        theme nominally allows. Deriving the row count from the window (and
        pulling the list up when that lets the whole list fit) is what keeps
        every option reachable instead of silently drawn past the bottom
        edge.

        Returns:
            ``(start_y, capacity)`` — the Y of the first row and the number
            of rows that fit between it and the reserved bottom strip.
        """
        spacing = self.option_spacing
        min_top = self._options_min_top()
        usable_bottom = self.screen.get_height() - theme.OPTIONS_BOTTOM_RESERVE

        needed = max(1, len(self.options)) * spacing
        latest_start = max(min_top, usable_bottom - needed)
        start_y = min(max(self._options_start_y(), min_top), latest_start)

        capacity = max(1, (usable_bottom - start_y) // spacing)
        return start_y, min(capacity, theme.MAX_VISIBLE_OPTIONS)

    def _natural_option_width(self) -> int:
        """Width the option rows want, before any clamping to the window.

        Measured from the widest label in its "> " (selected) form, since
        that is the wider of the two prefixes. Cached because it only
        changes when the option list does, and re-rendering every label to
        measure it once per frame is pure waste.
        """
        labels = tuple(text for text, _ in self.options)
        if self._width_cache is not None and self._width_cache[0] == labels:
            return self._width_cache[1]

        max_text_width = 0
        for text in labels:
            surface = self.option_font.render(f"> {text if text else '(Empty)'}", True, self.text_color)
            max_text_width = max(max_text_width, surface.get_width())
        width = max_text_width + 2 * theme.OPTION_PADDING_X
        self._width_cache = (labels, width)
        return width

    def _layout_visible_options(self) -> list[tuple[int, str, pygame.Rect]]:
        """Compute the on-screen layout for the currently visible options.

        Single source of truth for geometry — used by both click-hit testing
        (``_populate_option_rects``) and drawing (``_draw_content``) so the
        two cannot drift apart. Also refreshes ``max_visible_options`` and
        clamps ``scroll_offset`` to whatever the current window allows.

        Returns:
            List of ``(option_index, display_text, bg_rect)`` for each
            visible option. ``display_text`` is the option label guarded
            against the empty string (pygame raises on zero-width text) and
            ellipsized to fit the row.
        """
        screen_width = self.screen.get_width()

        start_y, capacity = self._options_geometry()
        self.max_visible_options = capacity
        self.scroll_offset = max(0, min(self.scroll_offset, len(self.options) - capacity))

        spacing = self.option_spacing
        padding_x = theme.OPTION_PADDING_X
        padding_y = theme.OPTION_PADDING_Y

        # Never wider than the window: without the clamp a long translated
        # label grows the row until it runs off both edges.
        natural_width = self._natural_option_width()
        uniform_width = min(natural_width, max(screen_width - 2 * theme.SCREEN_MARGIN_X, 1))
        # Only shorten labels when the clamp actually bit, so the common
        # case renders exactly the text the caller supplied.
        label_budget = 0
        if uniform_width < natural_width:
            label_budget = max(uniform_width - 2 * padding_x - self.option_font.size("> ")[0], 1)

        start_index = self.scroll_offset
        end_index = min(len(self.options), start_index + capacity)

        # Approximate row height from a single rendered line so bg_rect y-pos
        # doesn't depend on which label happens to be tallest.
        sample = self.option_font.render("Ag", True, self.text_color)
        row_height = sample.get_height()

        layout: list[tuple[int, str, pygame.Rect]] = []
        for display_i, option_i in enumerate(range(start_index, end_index)):
            text, _ = self.options[option_i]
            text = text if text else "(Empty)"
            if label_budget:
                text = ellipsize(text, self.option_font, label_budget)
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
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=theme.TITLE_MARGIN_Y)
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

        # Scroll affordances, anchored to the same geometry the rows used so
        # they cannot drift off-screen the way a recomputed start_y did.
        total_options = len(self.options)
        visible = self.max_visible_options
        end_index = min(total_options, self.scroll_offset + visible)
        rows = self.option_rects
        list_top = rows[0].top if rows else 0
        list_bottom = rows[-1].bottom if rows else 0

        if total_options > visible:
            if self.scroll_offset > 0:
                self._draw_scroll_hint(get_arrow_up_icon, " Scroll Up", above=list_top)
            if end_index < total_options:
                self._draw_scroll_hint(get_arrow_down_icon, " Scroll Down", below=list_bottom)

            # Position readout (e.g. "3-10 / 22")
            pos_text = self.indicator_font.render(
                f"{self.scroll_offset + 1}-{end_index} / {total_options}", True, self.text_color
            )
            pos_rect = pos_text.get_rect(right=screen_width - theme.SCREEN_MARGIN_X, bottom=screen_height - 8)
            self.screen.blit(pos_text, pos_rect)

        self._draw_footer_hint()

    def _is_compact(self) -> bool:
        """Whether the window is too short for the full bottom furniture.

        The in-game screen is sized to the map, so a small map leaves no
        room for a footer hint or labelled scroll arrows. Compact mode drops
        the words and keeps the arrows, matching the selection screens.
        """
        return self.screen.get_height() < COMPACT_LAYOUT_MIN_HEIGHT

    def _draw_scroll_hint(
        self,
        icon_factory: Callable[..., pygame.Surface],
        label: str,
        *,
        above: int | None = None,
        below: int | None = None,
    ) -> None:
        """Draw one scroll affordance, clamped inside the window.

        Args:
            icon_factory: ``get_arrow_up_icon`` / ``get_arrow_down_icon``.
            label: Text shown beside the arrow (dropped in compact mode).
            above: Y of the list's top edge; the hint sits just above it.
            below: Y of the list's bottom edge; the hint sits just below it.
        """
        icon = icon_factory(size=20, color=self.hover_color)
        text = None if self._is_compact() else self.indicator_font.render(label, True, self.hover_color)

        total_width = icon.get_width() + (text.get_width() if text else 0)
        # Height of the whole affordance, not just the arrow: clamping on the
        # arrow alone let a taller text label hang off the bottom edge.
        height = max(icon.get_height(), text.get_height() if text else 0)

        x = (self.screen.get_width() - total_width) // 2
        if above is not None:
            y = max(0, above - height - 8)
        else:
            y = min((below or 0) + 6, self.screen.get_height() - height)

        self.screen.blit(icon, (x, y))
        if text:
            self.screen.blit(text, (x + icon.get_width(), y))

    def _draw_footer_hint(self) -> None:
        """Draw the one-line control hint along the bottom of the screen.

        Skipped on short windows, where the space belongs to the options.
        """
        if not self.footer_hint or self._is_compact():
            return
        hint_font = get_font(theme.FONT_SIZE_HINT)
        hint_surface = hint_font.render(self.footer_hint, True, theme.TEXT_MUTED)
        hint_rect = hint_surface.get_rect(x=theme.SCREEN_MARGIN_X, bottom=self.screen.get_height() - 8)
        self.screen.blit(hint_surface, hint_rect)

    def draw(self) -> None:
        """Draw the menu and flip the display."""
        self._draw_content()
        pygame.display.flip()

    def _on_quit_event(self) -> tuple[bool, Any]:
        """React to a window-close (QUIT) event.

        Returns:
            ``(stop, value)`` — whether to leave the loop, and what
            :meth:`run` should return if so.
        """
        self.running = False
        # Re-post QUIT for the parent menu if we don't own the screen, so
        # the close request propagates all the way out instead of only
        # dismissing this screen.
        if not self.owns_screen:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return True, None

    def _on_result(self, result: Any) -> tuple[bool, Any]:
        """Decide what a callback's return value means for the menu loop.

        The default is "this is the answer, stop". Screens that toggle a
        setting or open a sub-menu override this to act on their sentinel
        values and return ``(False, None)`` to stay put — that is the whole
        reason they used to need a copy of :meth:`run`.

        Returns:
            ``(stop, value)`` — whether to leave the loop, and what
            :meth:`run` should return if so.
        """
        return True, result

    def run(self) -> Any | None:
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
                    stop, value = self._on_quit_event()
                    if stop:
                        return value
                    continue

                result = self.handle_input(event)
                if result is not None:
                    stop, value = self._on_result(result)
                    if stop:
                        return value
                    # Sub-menus and toggles rebuild the option list, so the
                    # click targets have to be recomputed before the next
                    # event is dispatched against them.
                    self._populate_option_rects()
                    result = None

            self.draw()
            clock.tick(theme.MENU_FRAMERATE)

        return result
