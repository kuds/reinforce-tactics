"""Main menu for the game."""

from typing import Any

import pygame

from reinforcetactics.ui.menus.base import Menu, drain_events
from reinforcetactics.ui.menus.credits_menu import CreditsMenu
from reinforcetactics.ui.menus.game_setup.game_mode_menu import GameModeMenu
from reinforcetactics.ui.menus.game_setup.map_selection_menu import MapSelectionMenu
from reinforcetactics.ui.menus.game_setup.player_config_menu import PlayerConfigMenu
from reinforcetactics.ui.menus.in_game.confirmation_dialog import ConfirmationDialog
from reinforcetactics.ui.menus.map_editor.map_editor_menu import MapEditorMenu
from reinforcetactics.ui.menus.save_load.load_game_menu import LoadGameMenu
from reinforcetactics.ui.menus.save_load.replay_selection_menu import ReplaySelectionMenu
from reinforcetactics.ui.menus.settings.settings_menu import SettingsMenu
from reinforcetactics.utils.language import get_language


class MainMenu(Menu):
    """Main menu for the game. Handles navigation to sub-menus internally."""

    def __init__(self) -> None:
        """Initialize main menu with self-managed screen."""
        super().__init__(None, self._get_title())
        # ESC quits from here rather than going "back", so say so.
        self.footer_hint = get_language().get("main_menu.menu_hint", "Arrows: Move   Enter: Select   Esc: Quit")
        self._setup_options()

    def _get_title(self) -> str:
        return get_language().get("main_menu.title", "Reinforce Tactics")

    def _setup_options(self) -> None:
        lang = get_language()
        self.add_option(lang.get("main_menu.new_game", "New Game"), self._new_game)
        self.add_option(lang.get("main_menu.load_game", "Load Game"), self._load_game)
        self.add_option(lang.get("main_menu.watch_replay", "Watch Replay"), self._watch_replay)
        self.add_option(lang.get("map_editor.title", "Map Editor"), self._map_editor)
        self.add_option(lang.get("main_menu.credits", "Credits"), self._credits)
        self.add_option(lang.get("main_menu.settings", "Settings"), self._settings)
        self.add_option(lang.get("main_menu.quit", "Quit"), self._quit)

    def _refresh_options(self) -> None:
        """Rebuild options with current language strings."""
        self.clear_options()
        self.title = self._get_title()
        self._setup_options()

    def _new_game(self) -> dict[str, Any] | None:
        """Handle new game - show game mode selection, map selection, and player configuration.

        Supports back-navigation: cancelling a step returns to the previous step
        instead of exiting all the way to the main menu.
        """
        step = 1
        selected_mode = None
        selected_map = None

        while step > 0:
            if step == 1:
                # Select game mode
                mode_menu = GameModeMenu(self.screen)
                selected_mode = mode_menu.run()
                drain_events()
                if not selected_mode:
                    return None  # Back to main menu
                step = 2

            elif step == 2:
                # Select map from chosen mode
                map_menu = MapSelectionMenu(self.screen, game_mode=selected_mode)
                selected_map = map_menu.run()
                drain_events()
                if not selected_map:
                    step = 1  # Back to mode selection
                else:
                    step = 3

            elif step == 3:
                # Configure players
                assert selected_mode is not None
                player_config_menu = PlayerConfigMenu(self.screen, game_mode=selected_mode)
                player_config_result = player_config_menu.run()
                drain_events()
                if player_config_result:
                    return {
                        "type": "new_game",
                        "map": selected_map,
                        "mode": selected_mode,
                        "players": player_config_result["players"],
                        "fog_of_war": player_config_result.get("fog_of_war", False),
                    }
                step = 2  # Back to map selection

        return None

    def _load_game(self) -> dict[str, Any] | None:
        """Handle load game - show load menu and return result."""
        load_menu = LoadGameMenu(self.screen)
        save_path = load_menu.run()
        drain_events()

        if save_path:
            return {"type": "load_game", "save_path": save_path}
        return None  # Cancelled

    def _watch_replay(self) -> dict[str, Any] | None:
        """Handle watch replay - show replay menu and return result."""
        replay_menu = ReplaySelectionMenu(self.screen)
        replay_path = replay_menu.run()
        drain_events()

        if replay_path:
            return {"type": "watch_replay", "replay_path": replay_path}
        return None  # Cancelled

    def _map_editor(self) -> None:
        """Handle map editor - show map editor menu."""
        map_editor_menu = MapEditorMenu(self.screen)
        map_editor_menu.run()
        drain_events()
        # Return to main menu after map editor

    def _settings(self) -> None:
        """Handle settings - show settings menu."""
        settings_menu = SettingsMenu(self.screen)
        settings_menu.run()
        drain_events()
        # Refresh options in case language was changed in settings
        self._refresh_options()
        self._populate_option_rects()

    def _credits(self) -> None:
        """Handle credits - show credits menu."""
        credits_menu = CreditsMenu(self.screen)
        credits_menu.run()
        drain_events()
        # Return to main menu after credits

    def _quit(self) -> dict[str, Any] | None:
        """Handle quit, confirming first so a stray click can't end the app."""
        if self._confirm_quit():
            return {"type": "exit"}
        return "cancelled"

    def _confirm_quit(self) -> bool:
        """Ask the player to confirm closing the game."""
        lang = get_language()
        dialog = ConfirmationDialog(
            self.screen,
            lang.get("main_menu.quit_confirm_title", "Quit Reinforce Tactics"),
            lang.get("main_menu.quit_confirm_msg", "Close the game?"),
            confirm_text=lang.get("main_menu.quit", "Quit"),
            cancel_text=lang.get("common.cancel", "Cancel"),
        )
        confirmed = dialog.run()
        drain_events()
        return confirmed

    def handle_input(self, event: pygame.event.Event) -> Any | None:
        """Route ESC through the same confirmation as the Quit option.

        The base class treats ESC as "leave this screen", but the main menu
        is the last screen — leaving it closes the game, which is too
        destructive to happen on a single keypress.
        """
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return self._quit()
        return super().handle_input(event)

    def _on_quit_event(self) -> tuple[bool, dict[str, Any]]:
        """A window-close on the main menu exits the application."""
        return True, {"type": "exit"}

    def _on_result(self, result: Any) -> tuple[bool, dict[str, Any] | None]:
        """Only a dict (a navigation command) ends the main menu."""
        if isinstance(result, dict):
            return True, result
        # A cancelled sub-flow or a declined quit: stay on the menu.
        return False, None

    def run(self) -> dict[str, Any] | None:
        """
        Run the main menu loop with internal navigation.

        Returns:
            Dict with 'type' key indicating action, or None if cancelled
        """
        result = super().run()
        return result if isinstance(result, dict) else {"type": "exit"}
