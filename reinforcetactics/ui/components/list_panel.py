"""Split-panel scrolling list shared by the selection screens.

The map, save and replay pickers all use the same shape: a scrolling list
of rows on the left, a preview/details panel on the right. Each screen used
to hand-roll that geometry twice — once in ``draw`` and once in
``_populate_option_rects`` — which is how the two drifted apart (map
selection never grew a ``_populate_option_rects`` override at all, so its
first frame hit-tested against the base class's centred rows).

:class:`ScrollList` is the single source of truth for that geometry, and
draws the chrome (row backgrounds, selection borders, scroll arrows) so the
three screens stay visually identical. Row *content* stays with each screen,
since a map thumbnail, a save summary and a replay card have nothing in
common.
"""

from __future__ import annotations

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.icons import get_arrow_down_icon, get_arrow_up_icon


def split_panels(screen: pygame.Surface, left_fraction: float, top: int = theme.PANEL_TOP) -> tuple[pygame.Rect, pygame.Rect]:
    """Lay out the left (list) and right (details) panels.

    Args:
        screen: Surface being drawn to.
        left_fraction: Share of the window width given to the list panel.
        top: Y coordinate where both panels start (below the title).

    Returns:
        ``(left_rect, right_rect)``.
    """
    width = screen.get_width()
    height = screen.get_height()
    margin = theme.PANEL_MARGIN
    panel_height = height - top - margin

    left_width = int(width * left_fraction)
    left = pygame.Rect(margin, top, left_width - 2 * margin, panel_height)
    right = pygame.Rect(left_width, top, width - left_width - margin, panel_height)
    return left, right


def draw_panel(screen: pygame.Surface, rect: pygame.Rect) -> None:
    """Draw a panel's background and border in the shared style."""
    pygame.draw.rect(screen, theme.PANEL_BG, rect, border_radius=theme.BORDER_RADIUS)
    pygame.draw.rect(
        screen,
        theme.PANEL_BORDER,
        rect,
        width=theme.BORDER_WIDTH_HOVER,
        border_radius=theme.BORDER_RADIUS,
    )


class ScrollList:
    """Geometry and chrome for a scrolling list of fixed-height rows.

    Attributes:
        panel_rect: Panel the list lives in.
        item_height: Row pitch, including the gap to the next row.
        capacity: How many rows fit in the panel (at least one).
    """

    def __init__(self, panel_rect: pygame.Rect, item_height: int, *, row_gap: int = 5, padding: int | None = None) -> None:
        """
        Args:
            panel_rect: Panel the list is drawn inside.
            item_height: Vertical pitch between consecutive rows.
            row_gap: Space between a row's bottom and the next row's top.
            padding: Inset from the panel edges. Defaults to the theme's
                panel margin.
        """
        self.panel_rect = panel_rect
        self.item_height = item_height
        self.row_gap = row_gap
        self.padding = theme.PANEL_MARGIN if padding is None else padding
        self.capacity = max(1, (panel_rect.height - 2 * self.padding) // item_height)

    def visible_range(self, scroll_offset: int, total: int) -> tuple[int, int]:
        """Return the ``[start, end)`` item indices visible at this offset."""
        start = max(0, min(scroll_offset, max(0, total - self.capacity)))
        return start, min(total, start + self.capacity)

    def item_rects(self, scroll_offset: int, total: int) -> list[pygame.Rect]:
        """Screen rectangles for the currently visible rows, in display order."""
        start, end = self.visible_range(scroll_offset, total)
        list_y = self.panel_rect.y + self.padding
        width = self.panel_rect.width - 2 * self.padding
        return [
            pygame.Rect(
                self.panel_rect.x + self.padding,
                list_y + display_i * self.item_height,
                width,
                self.item_height - self.row_gap,
            )
            for display_i in range(end - start)
        ]

    def draw_row(self, screen: pygame.Surface, rect: pygame.Rect, *, selected: bool = False, hovered: bool = False) -> None:
        """Draw a row's background and its selection/hover border."""
        if selected:
            bg_color = theme.OPTION_BG_SELECTED
        elif hovered:
            bg_color = theme.OPTION_BG_HOVER
        else:
            bg_color = theme.OPTION_BG

        pygame.draw.rect(screen, bg_color, rect, border_radius=theme.BORDER_RADIUS_SMALL)
        if selected or hovered:
            border_color = theme.SELECTED if selected else theme.HOVER
            pygame.draw.rect(
                screen,
                border_color,
                rect,
                width=theme.BORDER_WIDTH_HOVER,
                border_radius=theme.BORDER_RADIUS_SMALL,
            )

    def draw_scroll_indicators(self, screen: pygame.Surface, scroll_offset: int, total: int) -> None:
        """Draw the up/down arrows when there is more list than panel."""
        if total <= self.capacity:
            return
        start, end = self.visible_range(scroll_offset, total)

        if start > 0:
            icon = get_arrow_up_icon(size=16, color=theme.HOVER)
            screen.blit(icon, icon.get_rect(centerx=self.panel_rect.centerx, y=self.panel_rect.y + 2))
        if end < total:
            icon = get_arrow_down_icon(size=16, color=theme.HOVER)
            screen.blit(icon, icon.get_rect(centerx=self.panel_rect.centerx, bottom=self.panel_rect.bottom - 2))

    @staticmethod
    def draw_empty_hint(screen: pygame.Surface, panel_rect: pygame.Rect, text: str, font: pygame.font.Font) -> None:
        """Centre a "nothing selected" message inside a details panel."""
        surface = font.render(text, True, theme.TEXT_PLACEHOLDER)
        screen.blit(surface, surface.get_rect(center=panel_rect.center))
