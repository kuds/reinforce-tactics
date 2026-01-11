"""Confirmation dialog for critical actions."""
from typing import Optional, Tuple

import pygame

from reinforcetactics.utils.fonts import get_font


class ConfirmationDialog:
    """A modal confirmation dialog for critical actions like resign."""

    def __init__(self, screen: pygame.Surface, title: str, message: str,
                 confirm_text: str = "Confirm", cancel_text: str = "Cancel") -> None:
        """
        Initialize confirmation dialog.

        Args:
            screen: Pygame surface to draw on
            title: Dialog title
            message: Message to display
            confirm_text: Text for confirm button
            cancel_text: Text for cancel button
        """
        self.screen = screen
        self.title = title
        self.message = message
        self.confirm_text = confirm_text
        self.cancel_text = cancel_text
        self.running = True
        self.result: Optional[bool] = None

        # Colors
        self.bg_color = (40, 40, 50)
        self.text_color = (255, 255, 255)
        self.title_color = (255, 200, 50)
        self.border_color = (100, 150, 200)
        self.confirm_color = (80, 150, 80)
        self.confirm_hover_color = (100, 180, 100)
        self.cancel_color = (150, 80, 80)
        self.cancel_hover_color = (180, 100, 100)

        # Fonts
        self.title_font = get_font(32)
        self.message_font = get_font(24)
        self.button_font = get_font(28)

        # Calculate dialog dimensions
        self._calculate_dialog_rect()

        # Button rects (will be set in _calculate_dialog_rect)
        self.confirm_rect: Optional[pygame.Rect] = None
        self.cancel_rect: Optional[pygame.Rect] = None
        self.hover_button: Optional[str] = None

    def _calculate_dialog_rect(self) -> None:
        """Calculate the dialog rectangle position and size."""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Dialog dimensions
        dialog_width = 400
        dialog_height = 200

        # Center dialog on screen
        dialog_x = (screen_width - dialog_width) // 2
        dialog_y = (screen_height - dialog_height) // 2

        self.dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)

        # Calculate button positions
        button_width = 120
        button_height = 40
        button_y = self.dialog_rect.bottom - button_height - 20
        button_spacing = 30

        total_buttons_width = 2 * button_width + button_spacing
        buttons_start_x = self.dialog_rect.centerx - total_buttons_width // 2

        self.cancel_rect = pygame.Rect(
            buttons_start_x,
            button_y,
            button_width,
            button_height
        )
        self.confirm_rect = pygame.Rect(
            buttons_start_x + button_width + button_spacing,
            button_y,
            button_width,
            button_height
        )

    def handle_event(self, event: pygame.event.Event) -> Optional[bool]:
        """
        Handle pygame events.

        Args:
            event: Pygame event

        Returns:
            True if confirmed, False if cancelled, None if still running
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.result = False
                self.running = False
                return False
            elif event.key == pygame.K_RETURN:
                self.result = True
                self.running = False
                return True
            elif event.key == pygame.K_y:
                self.result = True
                self.running = False
                return True
            elif event.key == pygame.K_n:
                self.result = False
                self.running = False
                return False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = event.pos
                if self.confirm_rect and self.confirm_rect.collidepoint(mouse_pos):
                    self.result = True
                    self.running = False
                    return True
                elif self.cancel_rect and self.cancel_rect.collidepoint(mouse_pos):
                    self.result = False
                    self.running = False
                    return False
                # Click outside dialog - treat as cancel
                elif not self.dialog_rect.collidepoint(mouse_pos):
                    self.result = False
                    self.running = False
                    return False

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            if self.confirm_rect and self.confirm_rect.collidepoint(mouse_pos):
                self.hover_button = 'confirm'
            elif self.cancel_rect and self.cancel_rect.collidepoint(mouse_pos):
                self.hover_button = 'cancel'
            else:
                self.hover_button = None

        return None

    def draw(self) -> None:
        """Draw the confirmation dialog."""
        # Draw semi-transparent overlay
        overlay = pygame.Surface(
            (self.screen.get_width(), self.screen.get_height()),
            pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # Draw dialog background
        pygame.draw.rect(self.screen, self.bg_color, self.dialog_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.border_color, self.dialog_rect, width=3, border_radius=12)

        # Draw title
        title_surface = self.title_font.render(self.title, True, self.title_color)
        title_rect = title_surface.get_rect(
            centerx=self.dialog_rect.centerx,
            y=self.dialog_rect.y + 20
        )
        self.screen.blit(title_surface, title_rect)

        # Draw message
        message_surface = self.message_font.render(self.message, True, self.text_color)
        message_rect = message_surface.get_rect(
            centerx=self.dialog_rect.centerx,
            y=self.dialog_rect.y + 70
        )
        self.screen.blit(message_surface, message_rect)

        # Draw hint
        hint_font = get_font(18)
        hint_surface = hint_font.render("Press Y to confirm, N or ESC to cancel", True, (150, 150, 150))
        hint_rect = hint_surface.get_rect(
            centerx=self.dialog_rect.centerx,
            y=self.dialog_rect.y + 105
        )
        self.screen.blit(hint_surface, hint_rect)

        # Draw cancel button
        if self.cancel_rect:
            cancel_color = self.cancel_hover_color if self.hover_button == 'cancel' else self.cancel_color
            pygame.draw.rect(self.screen, cancel_color, self.cancel_rect, border_radius=8)
            if self.hover_button == 'cancel':
                pygame.draw.rect(self.screen, (255, 150, 150), self.cancel_rect, width=2, border_radius=8)

            cancel_surface = self.button_font.render(self.cancel_text, True, (255, 255, 255))
            cancel_text_rect = cancel_surface.get_rect(center=self.cancel_rect.center)
            self.screen.blit(cancel_surface, cancel_text_rect)

        # Draw confirm button
        if self.confirm_rect:
            confirm_color = self.confirm_hover_color if self.hover_button == 'confirm' else self.confirm_color
            pygame.draw.rect(self.screen, confirm_color, self.confirm_rect, border_radius=8)
            if self.hover_button == 'confirm':
                pygame.draw.rect(self.screen, (150, 255, 150), self.confirm_rect, width=2, border_radius=8)

            confirm_surface = self.button_font.render(self.confirm_text, True, (255, 255, 255))
            confirm_text_rect = confirm_surface.get_rect(center=self.confirm_rect.center)
            self.screen.blit(confirm_surface, confirm_text_rect)

    def run(self) -> bool:
        """
        Run the confirmation dialog loop.

        Returns:
            True if confirmed, False if cancelled
        """
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.result = False
                    self.running = False
                    break

                result = self.handle_event(event)
                if result is not None:
                    return result

            self.draw()
            pygame.display.flip()
            clock.tick(60)

        return self.result if self.result is not None else False
