"""Reusable button widgets.

A :class:`Button` owns its rect, label, and a :class:`ButtonStyle`, and
renders all visual states (normal / hovered / selected / disabled) the same
way on every screen. Menus remain in charge of input: they decide when a
button is hovered or clicked (usually via ``rect.collidepoint``) and pass
the state into :meth:`Button.draw`.
"""

from dataclasses import dataclass
from typing import Any, Self

import pygame

from reinforcetactics.ui import theme

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ButtonStyle:
    """Visual style for a button. Instances are shared and immutable."""

    bg: Color = theme.OPTION_BG
    bg_hover: Color = theme.OPTION_BG_HOVER
    bg_disabled: Color = theme.OPTION_BG_DISABLED
    text_color: Color = theme.TEXT
    text_hover_color: Color = theme.TEXT
    text_disabled_color: Color = theme.TEXT_DISABLED
    border_color: Color | None = None
    border_hover_color: Color | None = theme.HOVER
    border_radius: int = theme.BORDER_RADIUS
    border_width: int = 1
    border_hover_width: int = theme.BORDER_WIDTH_HOVER


# Shared styles. Dialog action buttons use the theme's semantic colors;
# MENU_OPTION is the neutral style used by overlay menus and config screens.
MENU_OPTION = ButtonStyle(text_hover_color=theme.HOVER)
MENU_OPTION_SMALL = ButtonStyle(text_hover_color=theme.HOVER, border_radius=theme.BORDER_RADIUS_SMALL)
PANEL_BUTTON = ButtonStyle(border_color=(60, 60, 80))
CONFIRM = ButtonStyle(
    bg=theme.BTN_CONFIRM,
    bg_hover=theme.BTN_CONFIRM_HOVER,
    border_hover_color=theme.BTN_CONFIRM_BORDER_HOVER,
)
CANCEL = ButtonStyle(
    bg=theme.BTN_CANCEL,
    bg_hover=theme.BTN_CANCEL_HOVER,
    border_hover_color=theme.BTN_CANCEL_BORDER_HOVER,
)
QUIT = ButtonStyle(
    bg=theme.BTN_QUIT,
    bg_hover=theme.BTN_QUIT_HOVER,
    border_hover_color=theme.BTN_QUIT_BORDER_HOVER,
)


class Button:
    """A rectangular button with a text label.

    Attributes:
        rect: Screen-space rectangle of the button.
        text: Button label.
        enabled: Disabled buttons render grayed out; menus should also skip
            them in hit-testing.
        payload: Optional caller data (e.g. an action dict) carried by the
            button for convenience.
    """

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        *,
        style: ButtonStyle = MENU_OPTION,
        enabled: bool = True,
        text_align: str = "center",
        text_padding: int = 10,
        payload: Any = None,
    ) -> None:
        """
        Args:
            rect: Button rectangle.
            text: Label text.
            font: Font used to render the label.
            style: Visual style.
            enabled: Whether the button is interactive.
            text_align: "center" or "left" label alignment.
            text_padding: Horizontal padding used for left alignment.
            payload: Optional caller data attached to this button.
        """
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.style = style
        self.enabled = enabled
        self.text_align = text_align
        self.text_padding = text_padding
        self.payload = payload

    @classmethod
    def with_label(
        cls,
        x: int,
        y: int,
        text: str,
        font: pygame.font.Font,
        *,
        padding_x: int = 20,
        padding_y: int = 10,
        **kwargs: Any,
    ) -> Self:
        """Create a button sized to fit its label plus padding at (x, y)."""
        width, height = font.size(text)
        rect = pygame.Rect(x, y, width + 2 * padding_x, height + 2 * padding_y)
        return cls(rect, text, font, **kwargs)

    def collidepoint(self, pos: tuple[int, int]) -> bool:
        """Whether ``pos`` is inside the button (regardless of enabled state)."""
        return self.rect.collidepoint(pos)

    def draw(self, screen: pygame.Surface, *, hovered: bool = False, selected: bool = False) -> None:
        """Draw the button.

        Args:
            screen: Target surface.
            hovered: Mouse-hover state (highlights like selection).
            selected: Keyboard-selection state (same highlight as hover, so
                keyboard users get identical visual feedback).
        """
        style = self.style
        highlighted = (hovered or selected) and self.enabled

        if not self.enabled:
            bg = style.bg_disabled
            text_color = style.text_disabled_color
        elif highlighted:
            bg = style.bg_hover
            text_color = style.text_hover_color
        else:
            bg = style.bg
            text_color = style.text_color

        pygame.draw.rect(screen, bg, self.rect, border_radius=style.border_radius)

        if highlighted and style.border_hover_color is not None:
            pygame.draw.rect(
                screen,
                style.border_hover_color,
                self.rect,
                width=style.border_hover_width,
                border_radius=style.border_radius,
            )
        elif self.enabled and style.border_color is not None:
            pygame.draw.rect(
                screen,
                style.border_color,
                self.rect,
                width=style.border_width,
                border_radius=style.border_radius,
            )

        text_surface = self.font.render(self.text, True, text_color)
        if self.text_align == "left":
            text_rect = text_surface.get_rect(left=self.rect.left + self.text_padding, centery=self.rect.centery)
        else:
            text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)


class CloseButton:
    """The small red "X" button used by in-game overlay menus."""

    SIZE = 20

    def __init__(self, x: int, y: int, size: int = SIZE) -> None:
        """
        Args:
            x: Left edge.
            y: Top edge.
            size: Square size in pixels.
        """
        self.rect = pygame.Rect(x, y, size, size)

    def collidepoint(self, pos: tuple[int, int]) -> bool:
        """Whether ``pos`` is inside the button."""
        return self.rect.collidepoint(pos)

    def draw(self, screen: pygame.Surface, *, hovered: bool = False) -> None:
        """Draw the close button."""
        color = theme.BTN_CLOSE_HOVER if hovered else theme.BTN_CLOSE
        pygame.draw.rect(screen, color, self.rect, border_radius=3)

        margin = 4
        pygame.draw.line(
            screen,
            theme.TEXT,
            (self.rect.left + margin, self.rect.top + margin),
            (self.rect.right - margin, self.rect.bottom - margin),
            2,
        )
        pygame.draw.line(
            screen,
            theme.TEXT,
            (self.rect.right - margin, self.rect.top + margin),
            (self.rect.left + margin, self.rect.bottom - margin),
            2,
        )
