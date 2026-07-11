"""Modal dialog widget.

A :class:`Dialog` renders a dimmed overlay, a centered panel with a title,
a word-wrapped message, an optional keyboard hint, and a row of action
buttons. The panel is sized to fit its content (long or translated strings
grow the dialog instead of overflowing it).

Each button resolves the dialog to a caller-supplied value; ``None`` is
reserved for "still running" and cannot be used as a button value.
"""

from typing import Any

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.widgets.button import Button, ButtonStyle
from reinforcetactics.ui.widgets.text import wrap_text
from reinforcetactics.utils.fonts import get_display_font, get_font

# (label, resolve value, style) for each action button, left to right.
DialogButtonSpec = tuple[str, Any, ButtonStyle]

_PADDING = 20
_BUTTON_HEIGHT = 40
_BUTTON_MIN_WIDTH = 120
_BUTTON_SPACING = 20


class Dialog:
    """A modal dialog with a title, message, and action buttons."""

    def __init__(
        self,
        screen: pygame.Surface,
        title: str,
        message: str,
        buttons: list[DialogButtonSpec],
        *,
        hint: str | None = None,
        keymap: dict[int, Any] | None = None,
        cancel_value: Any = None,
        quit_value: Any = None,
        min_width: int = 400,
    ) -> None:
        """
        Args:
            screen: Surface to draw on (the dialog dims it).
            title: Dialog title.
            message: Body text; word-wrapped to the dialog width.
            buttons: Action buttons, left to right.
            hint: Optional keyboard hint shown under the message.
            keymap: Extra key -> value shortcuts (e.g. ``{pygame.K_y: True}``).
            cancel_value: Value resolved by ESC or clicking outside the dialog.
            quit_value: Value resolved by a window-close (QUIT) event.
            min_width: Minimum dialog width in pixels.
        """
        self.screen = screen
        self.title = title
        self.message = message
        self.hint = hint
        self.keymap = keymap or {}
        self.cancel_value = cancel_value
        self.quit_value = quit_value
        self.running = True
        self.result: Any = None

        self.title_font = get_display_font(theme.FONT_SIZE_HEADING)
        self.message_font = get_font(theme.FONT_SIZE_BODY)
        self.button_font = get_font(theme.FONT_SIZE_BODY)
        self.hint_font = get_font(theme.FONT_SIZE_HINT)

        # Cached overlay surface to avoid per-frame allocation
        self._overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        self._overlay.fill(theme.DIALOG_OVERLAY_COLOR)

        self._build_layout(buttons, min_width)

    def _build_layout(self, button_specs: list[DialogButtonSpec], min_width: int) -> None:
        """Size the dialog to its content and lay out the buttons."""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        max_dialog_width = max(min_width, screen_width - 80)

        # Uniform button width fitting the widest label.
        label_widths = [self.button_font.size(label)[0] for label, _, _ in button_specs]
        button_width = max(_BUTTON_MIN_WIDTH, max(label_widths) + 30 if label_widths else _BUTTON_MIN_WIDTH)
        buttons_total = len(button_specs) * button_width + (len(button_specs) - 1) * _BUTTON_SPACING

        content_width = max(
            min_width - 2 * _PADDING,
            self.title_font.size(self.title)[0],
            self.hint_font.size(self.hint)[0] if self.hint else 0,
            buttons_total,
            min(self.message_font.size(self.message)[0], max_dialog_width - 2 * _PADDING),
        )
        dialog_width = min(content_width + 2 * _PADDING, max_dialog_width)

        self.message_lines = wrap_text(self.message, self.message_font, dialog_width - 2 * _PADDING)
        line_height = self.message_font.get_height()

        dialog_height = (
            _PADDING
            + self.title_font.get_height()
            + 12
            + len(self.message_lines) * line_height
            + 10
            + (self.hint_font.get_height() + 16 if self.hint else 0)
            + _BUTTON_HEIGHT
            + _PADDING
        )

        self.dialog_rect = pygame.Rect(
            (screen_width - dialog_width) // 2,
            (screen_height - dialog_height) // 2,
            dialog_width,
            dialog_height,
        )

        buttons_start_x = self.dialog_rect.centerx - buttons_total // 2
        button_y = self.dialog_rect.bottom - _BUTTON_HEIGHT - _PADDING
        self.buttons: list[Button] = []
        for i, (label, value, style) in enumerate(button_specs):
            rect = pygame.Rect(buttons_start_x + i * (button_width + _BUTTON_SPACING), button_y, button_width, _BUTTON_HEIGHT)
            self.buttons.append(Button(rect, label, self.button_font, style=style, payload=value))

    def _finish(self, value: Any) -> Any:
        """Resolve the dialog with ``value``."""
        self.result = value
        self.running = False
        return value

    def handle_event(self, event: pygame.event.Event) -> Any | None:
        """Handle a pygame event.

        Returns:
            The resolved value if this event finished the dialog, else None.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._finish(self.cancel_value)
            if event.key in self.keymap:
                return self._finish(self.keymap[event.key])

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.collidepoint(event.pos):
                    return self._finish(button.payload)
            if not self.dialog_rect.collidepoint(event.pos):
                # Click outside dialog - treat as cancel
                return self._finish(self.cancel_value)

        return None

    def draw(self) -> None:
        """Draw the dialog (overlay, panel, texts, buttons)."""
        self.screen.blit(self._overlay, (0, 0))

        pygame.draw.rect(self.screen, theme.PANEL_BG, self.dialog_rect, border_radius=theme.BORDER_RADIUS_DIALOG)
        pygame.draw.rect(
            self.screen,
            theme.BORDER,
            self.dialog_rect,
            width=theme.BORDER_WIDTH_DIALOG,
            border_radius=theme.BORDER_RADIUS_DIALOG,
        )

        y = self.dialog_rect.y + _PADDING
        title_surface = self.title_font.render(self.title, True, theme.SELECTED)
        self.screen.blit(title_surface, title_surface.get_rect(centerx=self.dialog_rect.centerx, y=y))
        y += self.title_font.get_height() + 12

        for line in self.message_lines:
            if line:
                line_surface = self.message_font.render(line, True, theme.TEXT)
                self.screen.blit(line_surface, line_surface.get_rect(centerx=self.dialog_rect.centerx, y=y))
            y += self.message_font.get_height()
        y += 10

        if self.hint:
            hint_surface = self.hint_font.render(self.hint, True, theme.TEXT_MUTED)
            self.screen.blit(hint_surface, hint_surface.get_rect(centerx=self.dialog_rect.centerx, y=y))

        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, hovered=button.collidepoint(mouse_pos))

    def run(self) -> Any:
        """Run the dialog loop until resolved.

        Returns:
            The value of the chosen button (or ``cancel_value``/``quit_value``).
        """
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return self._finish(self.quit_value)

                result = self.handle_event(event)
                if result is not None:
                    return result

            self.draw()
            pygame.display.flip()
            clock.tick(theme.MENU_FRAMERATE)

        return self.result
