"""Three-button quit confirmation dialog: Save & Quit, Quit, Cancel."""
from typing import Optional

import pygame

from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import get_language


class QuitConfirmDialog:
    """A modal dialog shown when the player attempts to quit mid-game."""

    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize quit confirmation dialog.

        Args:
            screen: Pygame surface to draw on
        """
        self.screen = screen
        self.running = True
        self.result: Optional[str] = None

        lang = get_language()
        self.title = lang.get('quit_confirm.title', 'Quit Game')
        self.message = lang.get('quit_confirm.message', 'Save before quitting?')
        self.save_quit_text = lang.get('quit_confirm.save_quit', 'Save & Quit')
        self.quit_text = lang.get('quit_confirm.quit', 'Quit')
        self.cancel_text = lang.get('quit_confirm.cancel', 'Cancel')

        # Colors
        self.bg_color = (40, 40, 50)
        self.text_color = (255, 255, 255)
        self.title_color = (255, 200, 50)
        self.border_color = (100, 150, 200)
        self.save_quit_color = (80, 150, 80)
        self.save_quit_hover_color = (100, 180, 100)
        self.quit_color = (150, 120, 50)
        self.quit_hover_color = (180, 150, 70)
        self.cancel_color = (150, 80, 80)
        self.cancel_hover_color = (180, 100, 100)

        # Fonts
        self.title_font = get_font(32)
        self.message_font = get_font(24)
        self.button_font = get_font(24)

        # Button state
        self.hover_button: Optional[str] = None

        # Calculate layout
        self._calculate_layout()

    def _calculate_layout(self) -> None:
        """Calculate dialog and button positions."""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        dialog_width = 500
        dialog_height = 210
        dialog_x = (screen_width - dialog_width) // 2
        dialog_y = (screen_height - dialog_height) // 2

        self.dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)

        # Three buttons laid out horizontally
        button_width = 140
        button_height = 40
        button_y = self.dialog_rect.bottom - button_height - 20
        button_spacing = 20

        total_width = 3 * button_width + 2 * button_spacing
        start_x = self.dialog_rect.centerx - total_width // 2

        self.cancel_rect = pygame.Rect(start_x, button_y, button_width, button_height)
        self.quit_rect = pygame.Rect(
            start_x + button_width + button_spacing, button_y,
            button_width, button_height
        )
        self.save_quit_rect = pygame.Rect(
            start_x + 2 * (button_width + button_spacing), button_y,
            button_width, button_height
        )

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle pygame events.

        Returns:
            'save_quit', 'quit', 'cancel', or None if still running
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._finish('cancel')
            elif event.key == pygame.K_s:
                return self._finish('save_quit')
            elif event.key == pygame.K_q:
                return self._finish('quit')

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = event.pos
                if self.save_quit_rect.collidepoint(mouse_pos):
                    return self._finish('save_quit')
                elif self.quit_rect.collidepoint(mouse_pos):
                    return self._finish('quit')
                elif self.cancel_rect.collidepoint(mouse_pos):
                    return self._finish('cancel')
                elif not self.dialog_rect.collidepoint(mouse_pos):
                    return self._finish('cancel')

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            if self.save_quit_rect.collidepoint(mouse_pos):
                self.hover_button = 'save_quit'
            elif self.quit_rect.collidepoint(mouse_pos):
                self.hover_button = 'quit'
            elif self.cancel_rect.collidepoint(mouse_pos):
                self.hover_button = 'cancel'
            else:
                self.hover_button = None

        return None

    def _finish(self, result: str) -> str:
        """Set result and stop the dialog loop."""
        self.result = result
        self.running = False
        return result

    def draw(self) -> None:
        """Draw the quit confirmation dialog."""
        # Semi-transparent overlay
        overlay = pygame.Surface(
            (self.screen.get_width(), self.screen.get_height()),
            pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # Dialog background
        pygame.draw.rect(self.screen, self.bg_color, self.dialog_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.border_color, self.dialog_rect, width=3, border_radius=12)

        # Title
        title_surface = self.title_font.render(self.title, True, self.title_color)
        title_rect = title_surface.get_rect(
            centerx=self.dialog_rect.centerx,
            y=self.dialog_rect.y + 20
        )
        self.screen.blit(title_surface, title_rect)

        # Message
        message_surface = self.message_font.render(self.message, True, self.text_color)
        message_rect = message_surface.get_rect(
            centerx=self.dialog_rect.centerx,
            y=self.dialog_rect.y + 65
        )
        self.screen.blit(message_surface, message_rect)

        # Hint
        hint_font = get_font(18)
        hint_surface = hint_font.render(
            "S = Save & Quit, Q = Quit, ESC = Cancel", True, (150, 150, 150)
        )
        hint_rect = hint_surface.get_rect(
            centerx=self.dialog_rect.centerx,
            y=self.dialog_rect.y + 100
        )
        self.screen.blit(hint_surface, hint_rect)

        # Cancel button
        self._draw_button(
            self.cancel_rect, self.cancel_text,
            self.cancel_color, self.cancel_hover_color,
            'cancel', (255, 150, 150)
        )

        # Quit button
        self._draw_button(
            self.quit_rect, self.quit_text,
            self.quit_color, self.quit_hover_color,
            'quit', (255, 220, 120)
        )

        # Save & Quit button
        self._draw_button(
            self.save_quit_rect, self.save_quit_text,
            self.save_quit_color, self.save_quit_hover_color,
            'save_quit', (150, 255, 150)
        )

    def _draw_button(self, rect: pygame.Rect, text: str,
                     color: tuple, hover_color: tuple,
                     button_id: str, border_highlight: tuple) -> None:
        """Draw a single button."""
        is_hovered = self.hover_button == button_id
        bg = hover_color if is_hovered else color
        pygame.draw.rect(self.screen, bg, rect, border_radius=8)
        if is_hovered:
            pygame.draw.rect(self.screen, border_highlight, rect, width=2, border_radius=8)

        text_surface = self.button_font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def run(self) -> str:
        """
        Run the quit confirmation dialog.

        Returns:
            'save_quit', 'quit', or 'cancel'
        """
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._finish('quit')
                    break

                result = self.handle_event(event)
                if result is not None:
                    return result

            self.draw()
            pygame.display.flip()
            clock.tick(60)

        return self.result if self.result is not None else 'cancel'
