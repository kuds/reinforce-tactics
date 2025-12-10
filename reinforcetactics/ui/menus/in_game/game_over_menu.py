"""Game over screen."""
from typing import Optional, Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language


class GameOverMenu(Menu):
    """Game over screen."""

    def __init__(self, winner: int, game_state: Any,
                 screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize game over menu.

        Args:
            winner: Player number who won
            game_state: Game state object
            screen: Optional pygame surface. If None, creates its own.
        """
        lang = get_language()
        title = lang.get('game_over.title', 'Game Over')
        super().__init__(screen, title)

        self.winner = winner
        self.game_state = game_state
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('game_over.save_replay', 'Save Replay'), self._save_replay)
        self.add_option(lang.get('game_over.new_game', 'New Game'), lambda: 'new_game')
        self.add_option(lang.get('game_over.main_menu', 'Main Menu'), lambda: 'main_menu')
        self.add_option(lang.get('game_over.quit', 'Quit'), lambda: 'quit')

    def _save_replay(self) -> Optional[str]:
        """Save game replay."""
        return self.game_state.save_replay_to_file()

    def draw(self) -> None:
        super().draw()

        # Draw winner announcement
        lang = get_language()
        winner_template = lang.get('game_over.winner', 'Player {player} Wins!')
        winner_text = winner_template.format(player=self.winner)

        winner_surface = self.title_font.render(winner_text, True, self.selected_color)
        winner_rect = winner_surface.get_rect(
            centerx=self.screen.get_width() // 2,
            y=100
        )
        self.screen.blit(winner_surface, winner_rect)

        pygame.display.flip()
