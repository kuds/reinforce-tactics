"""Game over screen."""

from typing import Any, Optional

import pygame

from reinforcetactics.constants import PLAYER_COLORS
from reinforcetactics.ui import theme
from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import get_language


class GameOverMenu(Menu):
    """Game over screen."""

    def __init__(self, winner: int, game_state: Any, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize game over menu.

        Args:
            winner: Player number who won
            game_state: Game state object
            screen: Optional pygame surface. If None, creates its own.
        """
        lang = get_language()
        title = lang.get("game_over.title", "Game Over")
        super().__init__(screen, title)

        self.winner = winner
        self.game_state = game_state
        self.winner_color = PLAYER_COLORS.get(winner, self.selected_color)
        self.label_font = get_font(20)
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get("game_over.save_replay", "Save Replay"), self._save_replay)
        self.add_option(lang.get("game_over.new_game", "New Game"), lambda: "new_game")
        self.add_option(lang.get("game_over.main_menu", "Main Menu"), lambda: "main_menu")
        self.add_option(lang.get("game_over.quit", "Quit"), lambda: "quit")

    def _save_replay(self) -> Optional[str]:
        """Save game replay."""
        return self.game_state.save_replay_to_file()

    def _draw_content(self) -> None:
        super()._draw_content()

        screen_cx = self.screen.get_width() // 2

        # Draw winner announcement with player color
        lang = get_language()
        winner_template = lang.get("game_over.winner", "Player {player} Wins!")
        winner_text = winner_template.format(player=self.winner)

        winner_surface = self.title_font.render(winner_text, True, self.winner_color)
        winner_rect = winner_surface.get_rect(centerx=screen_cx, y=100)
        self.screen.blit(winner_surface, winner_rect)

        # Draw decorative underline in player color
        line_y = winner_rect.bottom + 6
        line_half = winner_rect.width // 2 + 20
        pygame.draw.line(
            self.screen,
            (*self.winner_color[:3], 120) if len(self.winner_color) > 3 else self.winner_color,
            (screen_cx - line_half, line_y),
            (screen_cx + line_half, line_y),
            2,
        )

        # Draw turn count
        turn_count = getattr(self.game_state, "turn", None)
        if turn_count is not None:
            turn_label = (
                lang.get("game_over.turns", "Turns: {turns}").format(turns=turn_count)
                if hasattr(lang, "get")
                else f"Turns: {turn_count}"
            )
            turn_surface = self.label_font.render(turn_label, True, theme.TEXT_MUTED)
            turn_rect = turn_surface.get_rect(centerx=screen_cx, y=line_y + 12)
            self.screen.blit(turn_surface, turn_rect)
