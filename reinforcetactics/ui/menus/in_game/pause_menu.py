"""In-game pause menu with sub-menu navigation."""
from typing import Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language


class PauseMenu(Menu):
    """
    In-game pause menu that provides access to Resume, Save, Settings,
    Main Menu, and Quit options.

    Sub-menus (Save, Settings) are opened inline and return to the pause
    menu when closed. Main Menu and Quit show confirmation dialogs.
    """

    def __init__(self, screen: pygame.Surface, game: Any) -> None:
        """
        Initialize pause menu.

        Args:
            screen: Pygame surface to draw on (the game screen).
            game: The GameState instance (needed for saving).
        """
        super().__init__(screen, get_language().get('pause.title', 'Paused'))
        self.game = game
        self._setup_options()

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('pause.resume', 'Resume'), lambda: 'resume')
        self.add_option(lang.get('pause.save', 'Save Game'), lambda: 'save')
        self.add_option(lang.get('pause.settings', 'Settings'), lambda: 'settings')
        self.add_option(lang.get('pause.main_menu', 'Main Menu'), lambda: 'main_menu')
        self.add_option(lang.get('pause.quit', 'Quit'), lambda: 'quit')

    def run(self) -> str:
        """
        Run the pause menu loop with sub-menu handling.

        Returns:
            'resume'    - Player wants to continue playing
            'main_menu' - Player wants to return to the main menu
            'save_quit' - Player wants to save and then quit the application
            'quit'      - Player wants to quit without saving
        """
        clock = pygame.time.Clock()

        self._populate_option_rects()
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Window close button -> treat as quit request
                    return self._handle_quit_option()

                result = self.handle_input(event)
                if result is not None:
                    if result == 'resume':
                        return 'resume'

                    elif result == 'save':
                        self._handle_save_option()
                        # Stay in pause menu after saving
                        self._populate_option_rects()

                    elif result == 'settings':
                        self._handle_settings_option()
                        # Stay in pause menu after settings
                        self._populate_option_rects()

                    elif result == 'main_menu':
                        confirmed = self._handle_main_menu_option()
                        if confirmed:
                            return 'main_menu'
                        # User cancelled -> stay in pause menu
                        self.running = True
                        self._populate_option_rects()

                    elif result == 'quit':
                        quit_result = self._handle_quit_option()
                        if quit_result in ('save_quit', 'quit'):
                            return quit_result
                        # 'cancel' -> stay in pause menu
                        self.running = True
                        self._populate_option_rects()

            self.draw()
            clock.tick(30)

        # Loop exited via ESC (base class sets self.running = False)
        return 'resume'

    def _handle_save_option(self) -> None:
        """Open the SaveGameMenu as a sub-menu."""
        from reinforcetactics.ui.menus.save_load.save_game_menu import SaveGameMenu

        save_menu = SaveGameMenu(self.game, self.screen)
        save_result = save_menu.run()
        pygame.event.clear()

        if save_result:
            print(f"Game saved to {save_result}")

    def _handle_settings_option(self) -> None:
        """Open the SettingsMenu as a sub-menu."""
        from reinforcetactics.ui.menus.settings.settings_menu import SettingsMenu

        settings_menu = SettingsMenu(self.screen)
        settings_menu.run()
        pygame.event.clear()

    def _handle_main_menu_option(self) -> bool:
        """
        Show a confirmation dialog before returning to the main menu.

        Returns:
            True if the user confirmed, False if cancelled.
        """
        from reinforcetactics.ui.menus.in_game.confirmation_dialog import ConfirmationDialog

        lang = get_language()
        dialog = ConfirmationDialog(
            self.screen,
            lang.get('pause.main_menu_confirm_title', 'Return to Main Menu'),
            lang.get('pause.main_menu_confirm_msg',
                      'Unsaved progress will be lost. Continue?'),
            confirm_text=lang.get('common.confirm', 'Confirm'),
            cancel_text=lang.get('common.cancel', 'Cancel')
        )
        confirmed = dialog.run()
        pygame.event.clear()
        return confirmed

    def _handle_quit_option(self) -> str:
        """
        Show the quit confirmation dialog (Save & Quit / Quit / Cancel).

        Returns:
            'save_quit', 'quit', or 'cancel'
        """
        from reinforcetactics.ui.menus.in_game.quit_confirm_dialog import QuitConfirmDialog

        dialog = QuitConfirmDialog(self.screen)
        quit_result = dialog.run()
        pygame.event.clear()
        return quit_result
