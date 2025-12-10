"""Main menu for the game."""
from typing import Optional, Dict, Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.game_setup.game_mode_menu import GameModeMenu
from reinforcetactics.ui.menus.game_setup.map_selection_menu import MapSelectionMenu
from reinforcetactics.ui.menus.game_setup.player_config_menu import PlayerConfigMenu
from reinforcetactics.ui.menus.save_load.load_game_menu import LoadGameMenu
from reinforcetactics.ui.menus.save_load.replay_selection_menu import ReplaySelectionMenu
from reinforcetactics.ui.menus.settings.settings_menu import SettingsMenu
from reinforcetactics.utils.language import get_language


class MainMenu(Menu):
    """Main menu for the game. Handles navigation to sub-menus internally."""

    def __init__(self) -> None:
        """Initialize main menu with self-managed screen."""
        super().__init__(None, self._get_title())
        self._setup_options()

    def _get_title(self) -> str:
        return get_language().get('main_menu.title', 'Reinforce Tactics')

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get('main_menu.new_game', 'New Game'), self._new_game)
        self.add_option(lang.get('main_menu.load_game', 'Load Game'), self._load_game)
        self.add_option(lang.get('main_menu.watch_replay', 'Watch Replay'), self._watch_replay)
        self.add_option(lang.get('main_menu.settings', 'Settings'), self._settings)
        self.add_option(lang.get('main_menu.quit', 'Quit'), self._quit)

    def _new_game(self) -> Optional[Dict[str, Any]]:
        """Handle new game - show game mode selection, map selection, and player configuration."""
        # Step 1: Select game mode
        mode_menu = GameModeMenu(self.screen)
        selected_mode = mode_menu.run()
        pygame.event.clear()

        if not selected_mode:
            return None  # User cancelled

        # Step 2: Select map from chosen mode
        map_menu = MapSelectionMenu(self.screen, game_mode=selected_mode)
        selected_map = map_menu.run()
        pygame.event.clear()

        if not selected_map:
            return None  # User cancelled

        # Step 3: Configure players
        player_config_menu = PlayerConfigMenu(self.screen, game_mode=selected_mode)
        player_config_result = player_config_menu.run()
        pygame.event.clear()

        if player_config_result:
            return {
                'type': 'new_game',
                'map': selected_map,
                'mode': selected_mode,  # Contains "1v1" or "2v2"
                'players': player_config_result['players']
            }
        return None

    def _load_game(self) -> Optional[Dict[str, Any]]:
        """Handle load game - show load menu and return result."""
        load_menu = LoadGameMenu(self.screen)
        save_path = load_menu.run()
        pygame.event.clear()

        if save_path:
            return {
                'type': 'load_game',
                'save_path': save_path
            }
        return None  # Cancelled

    def _watch_replay(self) -> Optional[Dict[str, Any]]:
        """Handle watch replay - show replay menu and return result."""
        replay_menu = ReplaySelectionMenu(self.screen)
        replay_path = replay_menu.run()
        pygame.event.clear()

        if replay_path:
            return {
                'type': 'watch_replay',
                'replay_path': replay_path
            }
        return None  # Cancelled

    def _settings(self) -> None:
        """Handle settings - show settings menu."""
        settings_menu = SettingsMenu(self.screen)
        settings_menu.run()
        pygame.event.clear()
        # Return to main menu after settings

    def _quit(self) -> Dict[str, Any]:
        """Handle quit."""
        return {'type': 'exit'}

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the main menu loop with internal navigation.

        Returns:
            Dict with 'type' key indicating action, or None if cancelled
        """
        result = None
        clock = pygame.time.Clock()

        # Populate option_rects before event loop for click detection
        self._populate_option_rects()

        # Clear any residual events AFTER option_rects are populated
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return {'type': 'exit'}

                result = self.handle_input(event)
                if result is not None:
                    # If we got a dict back, that's our final result
                    if isinstance(result, dict):
                        return result
                    # Otherwise stay in menu loop

            self.draw()
            clock.tick(30)

        return result if isinstance(result, dict) else {'type': 'exit'}
