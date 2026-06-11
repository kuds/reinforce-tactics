"""Three-button quit confirmation dialog: Save & Quit, Quit, Cancel."""

import pygame

from reinforcetactics.ui import widgets
from reinforcetactics.ui.widgets.dialog import Dialog
from reinforcetactics.utils.language import get_language


class QuitConfirmDialog(Dialog):
    """A modal dialog shown when the player attempts to quit mid-game."""

    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize quit confirmation dialog.

        Args:
            screen: Pygame surface to draw on
        """
        lang = get_language()
        super().__init__(
            screen,
            lang.get("quit_confirm.title", "Quit Game"),
            lang.get("quit_confirm.message", "Save before quitting?"),
            buttons=[
                (lang.get("quit_confirm.cancel", "Cancel"), "cancel", widgets.CANCEL),
                (lang.get("quit_confirm.quit", "Quit"), "quit", widgets.QUIT),
                (lang.get("quit_confirm.save_quit", "Save & Quit"), "save_quit", widgets.CONFIRM),
            ],
            hint="S = Save & Quit, Q = Quit, ESC = Cancel",
            keymap={pygame.K_s: "save_quit", pygame.K_q: "quit"},
            cancel_value="cancel",
            quit_value="quit",
            min_width=500,
        )

    def run(self) -> str:
        """
        Run the quit confirmation dialog.

        Returns:
            'save_quit', 'quit', or 'cancel'
        """
        return str(super().run())
