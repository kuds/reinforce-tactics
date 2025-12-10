"""
Menu system for the strategy game.
Self-contained menus that manage their own pygame screen and navigation.

This package provides backward compatibility by re-exporting all menu classes.
"""

# Base menu class and helper function
from reinforcetactics.ui.menus.base import Menu, _get_back_translations

# Main menu
from reinforcetactics.ui.menus.main_menu import MainMenu

# Game setup menus
from reinforcetactics.ui.menus.game_setup.game_mode_menu import GameModeMenu
from reinforcetactics.ui.menus.game_setup.map_selection_menu import MapSelectionMenu
from reinforcetactics.ui.menus.game_setup.player_config_menu import PlayerConfigMenu

# Save/load menus
from reinforcetactics.ui.menus.save_load.save_game_menu import SaveGameMenu
from reinforcetactics.ui.menus.save_load.load_game_menu import LoadGameMenu
from reinforcetactics.ui.menus.save_load.replay_selection_menu import ReplaySelectionMenu

# Settings menus
from reinforcetactics.ui.menus.settings.settings_menu import SettingsMenu
from reinforcetactics.ui.menus.settings.language_menu import LanguageMenu
from reinforcetactics.ui.menus.settings.api_keys_menu import APIKeysMenu

# In-game menus
from reinforcetactics.ui.menus.in_game.pause_menu import PauseMenu
from reinforcetactics.ui.menus.in_game.game_over_menu import GameOverMenu
from reinforcetactics.ui.menus.in_game.unit_purchase_menu import UnitPurchaseMenu

__all__ = [
    # Base
    'Menu',
    '_get_back_translations',
    # Main
    'MainMenu',
    # Game setup
    'GameModeMenu',
    'MapSelectionMenu',
    'PlayerConfigMenu',
    # Save/load
    'SaveGameMenu',
    'LoadGameMenu',
    'ReplaySelectionMenu',
    # Settings
    'SettingsMenu',
    'LanguageMenu',
    'APIKeysMenu',
    # In-game
    'PauseMenu',
    'GameOverMenu',
    'UnitPurchaseMenu',
]
