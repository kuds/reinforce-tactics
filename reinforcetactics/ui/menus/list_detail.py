"""Shared split-panel list + detail screen base.

The three selection screens (map / save / replay) all draw the same
shell: a title, a scrollable list panel on the left whose rows highlight
on hover/selection, and a detail panel on the right showing a preview of
the active item. Each used to copy the scaffolding around ``ScrollList``
verbatim; :class:`ListDetailMenu` owns it once, and subclasses supply
only their row content, detail content, and layout constants.
"""

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.components.list_panel import ScrollList, draw_panel, split_panels
from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.fonts import get_font


def draw_preview_or_placeholder(
    screen: pygame.Surface,
    preview: pygame.Surface | None,
    x: int,
    y: int,
    size: int,
) -> None:
    """Blit a square preview with border, or the standard "No Preview" card."""
    rect = pygame.Rect(x, y, size, size)
    if preview:
        screen.blit(preview, (x, y))
        pygame.draw.rect(screen, theme.FRAME_BORDER, rect, width=theme.BORDER_WIDTH_HOVER)
    else:
        pygame.draw.rect(screen, theme.PLACEHOLDER_BG, rect)
        pygame.draw.rect(screen, theme.FRAME_BORDER, rect, width=theme.BORDER_WIDTH_HOVER)

        placeholder_font = get_font(theme.FONT_SIZE_BODY)
        placeholder_text = placeholder_font.render("No Preview", True, theme.TEXT_PLACEHOLDER)
        screen.blit(placeholder_text, placeholder_text.get_rect(center=rect.center))


class ListDetailMenu(Menu):
    """Base for split-panel screens: scrollable list left, item detail right.

    Subclasses set the class attributes below, implement
    :meth:`_detail_items` (the list backing the detail panel — options
    beyond its length, e.g. the Back row, get no detail),
    :meth:`_draw_row_content` and :meth:`_draw_detail_content`.
    """

    # Fraction of the window width given to the list panel.
    LIST_FRACTION = 0.55
    # Row height in pixels (includes ROW_GAP of trailing space).
    ITEM_HEIGHT = 50
    # Space between rows; 5 is ScrollList's own default.
    ROW_GAP = 5
    # Placeholder shown in the detail panel when nothing is active.
    EMPTY_HINT = "Select an item to preview"
    EMPTY_HINT_FONT_SIZE = theme.FONT_SIZE_SUBHEADING

    # -- geometry (shared by hit-testing and drawing) ----------------------

    def _panels(self) -> tuple[pygame.Rect, pygame.Rect]:
        """The list and detail panel rectangles for the current window."""
        return split_panels(self.screen, self.LIST_FRACTION)

    def _scroll_list(self) -> ScrollList:
        """The list geometry, shared by hit-testing and drawing."""
        left_panel, _ = self._panels()
        return ScrollList(left_panel, self.ITEM_HEIGHT, row_gap=self.ROW_GAP)

    def _populate_option_rects(self) -> None:
        """Populate option_rects for click detection matching the panel layout.

        Without this override the base class's centred full-width rows would
        be used for the first frame's hit-testing, so an early click landed
        on the wrong option (or on nothing at all).
        """
        scroll_list = self._scroll_list()
        self.max_visible_options = scroll_list.capacity
        self.option_rects = scroll_list.item_rects(self.scroll_offset, len(self.options))

    # -- hooks -------------------------------------------------------------

    def _detail_items(self) -> list:
        """The items backing the detail panel, aligned with option indices."""
        raise NotImplementedError

    def _draw_row_content(self, index: int, item_rect: pygame.Rect, text: str, text_color) -> None:
        """Draw one row's content inside ``item_rect`` (row bg already drawn)."""
        raise NotImplementedError

    def _draw_detail_content(self, panel_rect: pygame.Rect, active_index: int) -> None:
        """Draw the detail panel for ``_detail_items()[active_index]``."""
        raise NotImplementedError

    # -- drawing -----------------------------------------------------------

    def draw(self) -> None:
        """Draw the split-panel screen and flip the display."""
        self.screen.fill(self.bg_color)

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=self.screen.get_width() // 2, y=20)
            self.screen.blit(title_surface, title_rect)

        left_panel, right_panel = self._panels()
        draw_panel(self.screen, left_panel)
        draw_panel(self.screen, right_panel)

        self._draw_list()
        self._draw_detail_panel(right_panel)

        pygame.display.flip()

    def _row_text_color(self, is_selected: bool, is_hovered: bool):
        """The row label color for the given highlight state."""
        if is_selected:
            return self.selected_color
        if is_hovered:
            return self.hover_color
        return self.text_color

    def _draw_list(self) -> None:
        """Draw the scrollable list: row backgrounds, content, indicators."""
        scroll_list = self._scroll_list()
        # Sync with base class so keyboard scrolling and mouse-wheel bounds
        # match the visible count.
        self.max_visible_options = scroll_list.capacity

        total = len(self.options)
        start_idx, _ = scroll_list.visible_range(self.scroll_offset, total)
        self.option_rects = scroll_list.item_rects(self.scroll_offset, total)

        for display_idx, item_rect in enumerate(self.option_rects):
            i = start_idx + display_idx
            text, _ = self.options[i]

            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index
            scroll_list.draw_row(self.screen, item_rect, selected=is_selected, hovered=is_hovered)

            self._draw_row_content(i, item_rect, text, self._row_text_color(is_selected, is_hovered))

        scroll_list.draw_scroll_indicators(self.screen, self.scroll_offset, total)

    def _draw_detail_panel(self, panel_rect: pygame.Rect) -> None:
        """Resolve the active item and draw its detail (or the empty hint)."""
        # Hover wins over selection so mousing down the list live-previews.
        active_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        if active_index < 0 or active_index >= len(self._detail_items()):
            hint_font = get_font(self.EMPTY_HINT_FONT_SIZE)
            ScrollList.draw_empty_hint(self.screen, panel_rect, self.EMPTY_HINT, hint_font)
            return

        self._draw_detail_content(panel_rect, active_index)
