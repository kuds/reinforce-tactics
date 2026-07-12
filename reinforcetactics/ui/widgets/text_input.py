"""Single-line text input widget.

Consolidates the keyboard editing behavior (backspace, clipboard paste,
printable-character filtering with modifier guards) and the box rendering
(background, focus border, blinking cursor, overflow ellipsis) that several
menus previously each implemented by hand.
"""

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.clipboard import get_clipboard_text

Color = tuple[int, int, int]

CURSOR_BLINK_MS = 500


class TextInput:
    """Editable single-line text state plus drawing.

    The widget only consumes *editing* keys (backspace, paste, printable
    characters). Navigation/submission keys (Enter, Escape, Tab) are left to
    the owning menu so each screen keeps its own flow.
    """

    def __init__(self, text: str = "", max_length: int | None = None) -> None:
        """
        Args:
            text: Initial contents.
            max_length: Maximum number of characters, or None for unlimited.
        """
        self.text = text
        self.max_length = max_length
        # Enable OS-style key repeat so holding Backspace (or a character
        # key) keeps editing instead of requiring one press per character.
        # pygame's default is repeat-off; this is global but matches
        # expected behavior in menus too (held arrows keep scrolling).
        if pygame.get_init():
            pygame.key.set_repeat(400, 50)

    def _append(self, addition: str) -> None:
        if self.max_length is not None:
            addition = addition[: max(0, self.max_length - len(self.text))]
        self.text += addition

    def handle_key(self, event: pygame.event.Event) -> bool:
        """Apply an editing key to the input.

        Args:
            event: A pygame.KEYDOWN event.

        Returns:
            True if the event was consumed (text may have changed); False if
            the key is not an editing key and the caller should handle it.
        """
        if event.type != pygame.KEYDOWN:
            return False

        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True

        # Ctrl+V (Windows/Linux) or Cmd+V (macOS) pastes from the clipboard.
        if event.key == pygame.K_v and event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META):
            clipboard_text = get_clipboard_text()
            if clipboard_text:
                self._append("".join(c for c in clipboard_text if c.isprintable()))
            return True

        # Plain printable characters. The modifier guard prevents shortcuts
        # like Cmd+V from also inserting their letter on macOS.
        if (
            event.unicode
            and event.unicode.isprintable()
            and not event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META | pygame.KMOD_ALT)
        ):
            self._append(event.unicode)
            return True

        return False

    def draw(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        font: pygame.font.Font,
        *,
        active: bool = True,
        display_text: str | None = None,
        text_color: Color = theme.TEXT,
        bg_color: Color = theme.OPTION_BG,
        bg_active_color: Color = theme.OPTION_BG_HOVER,
        border_color: Color = theme.PANEL_BORDER,
        border_active_color: Color = theme.SELECTED,
    ) -> None:
        """Draw the input box.

        Args:
            screen: Target surface.
            rect: Box rectangle.
            font: Font for the contents.
            active: Whether the input has focus (focus border + cursor).
            display_text: Optional override for the rendered text (e.g. a
                masked API key). Defaults to the actual contents.
            text_color: Contents color.
            bg_color: Box background when inactive.
            bg_active_color: Box background when active.
            border_color: Border when inactive.
            border_active_color: Border when active.
        """
        pygame.draw.rect(screen, bg_active_color if active else bg_color, rect, border_radius=theme.BORDER_RADIUS)
        pygame.draw.rect(
            screen,
            border_active_color if active else border_color,
            rect,
            width=theme.BORDER_WIDTH_HOVER,
            border_radius=theme.BORDER_RADIUS,
        )

        padding = 10
        max_text_width = rect.width - 2 * padding - 4  # room for the cursor
        shown = display_text if display_text is not None else self.text
        # Keep the end of the text visible while typing (paths, long keys).
        shown = ellipsize(shown, font, max_text_width, keep_end=True)

        text_surface = font.render(shown, True, text_color)
        text_rect = text_surface.get_rect(midleft=(rect.left + padding, rect.centery))
        screen.blit(text_surface, text_rect)

        if active and (pygame.time.get_ticks() // CURSOR_BLINK_MS) % 2 == 0:
            cursor_x = text_rect.right + 2
            pygame.draw.line(
                screen,
                text_color,
                (cursor_x, rect.top + 8),
                (cursor_x, rect.bottom - 8),
                2,
            )
