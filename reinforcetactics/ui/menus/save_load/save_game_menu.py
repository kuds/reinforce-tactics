"""Menu for saving the game."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.widgets import TextInput
from reinforcetactics.utils.language import get_language


class SaveGameMenu(Menu):
    """Menu for saving the game."""

    def __init__(self, game: Any, screen: pygame.Surface | None = None) -> None:
        """
        Initialize save game menu.

        Args:
            game: Game state object to save
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get("save_game.title", "Save Game"))
        self.game = game
        self.name_input = TextInput(text=f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}", max_length=50)

    @property
    def input_text(self) -> str:
        """Current save-name text (kept for backward compatibility)."""
        return self.name_input.text

    def handle_input(self, event: pygame.event.Event) -> str | None:
        """Handle keyboard input for filename entry."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if self.input_text:
                    return self._save_game()
            elif event.key == pygame.K_ESCAPE:
                self.running = False
            else:
                self.name_input.handle_key(event)

        return None

    def _save_game(self) -> str | None:
        """Save the game with current filename."""
        # Ensure saves directory exists
        saves_dir = Path("saves")
        saves_dir.mkdir(exist_ok=True)

        filepath = self.game.save_to_file(f"saves/{self.input_text}.json")
        return filepath

    def run(self) -> str | None:
        """
        Run save game menu.

        Returns:
            Path to saved file, or None if cancelled
        """
        return super().run()

    def draw(self) -> None:
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        title_surface = self.title_font.render(self.title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
        self.screen.blit(title_surface, title_rect)

        # Draw input prompt
        prompt = get_language().get("save_game.enter_name", "Enter save name:")
        prompt_surface = self.option_font.render(prompt, True, self.text_color)
        prompt_rect = prompt_surface.get_rect(centerx=screen_width // 2, y=screen_height // 3)
        self.screen.blit(prompt_surface, prompt_rect)

        # Draw input box
        input_width = 500
        input_height = 44
        input_rect = pygame.Rect((screen_width - input_width) // 2, screen_height // 3 + 50, input_width, input_height)
        self.name_input.draw(self.screen, input_rect, self.indicator_font)

        # Draw instructions
        lang = get_language()
        instructions = lang.get("save_game.instructions", "Press ENTER to save, ESC to cancel")
        inst_surface = self.option_font.render(instructions, True, (150, 150, 150))
        inst_rect = inst_surface.get_rect(centerx=screen_width // 2, y=screen_height // 2 + 50)
        self.screen.blit(inst_surface, inst_rect)

        pygame.display.flip()
