"""Map editor menu - entry point for creating or editing maps."""
from typing import Optional, Dict, Any
from pathlib import Path
import pygame
import pandas as pd
import numpy as np

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.map_editor.new_map_dialog import NewMapDialog
from reinforcetactics.ui.menus.map_editor.map_editor import MapEditor
from reinforcetactics.ui.menus.game_setup.map_selection_menu import MapSelectionMenu
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.constants import MIN_MAP_SIZE


class MapEditorMenu(Menu):
    """Menu for map editor options."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize the map editor menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        lang = get_language()
        super().__init__(screen, lang.get('map_editor.title', 'Map Editor'))
        self._setup_options()

    def _setup_options(self) -> None:
        """Setup menu options."""
        lang = get_language()
        self.add_option(lang.get('map_editor.new_map', 'New Map'), self._new_map)
        self.add_option(lang.get('map_editor.edit_map', 'Edit Existing Map'), self._edit_map)
        self.add_option(lang.get('common.back', 'Back'), lambda: None)

    def _new_map(self) -> Optional[Dict[str, Any]]:
        """
        Create a new map.

        Returns:
            Result dict or None
        """
        # Show new map dialog
        dialog = NewMapDialog(self.screen)
        result = dialog.run()
        pygame.event.clear()
        
        if not result:
            return None  # User cancelled
        
        # Create empty map with specified dimensions
        width = result['width']
        height = result['height']
        num_players = result['num_players']
        
        # Create map filled with ocean tiles
        map_data = pd.DataFrame(
            np.full((height, width), 'o', dtype=object)
        )
        
        # Place headquarters for each player in corners
        if num_players >= 1:
            map_data.iloc[1, 1] = 'h_1'  # Player 1 HQ (top-left)
        
        if num_players >= 2:
            map_data.iloc[height-2, width-2] = 'h_2'  # Player 2 HQ (bottom-right)
        
        if num_players >= 3:
            map_data.iloc[1, width-2] = 'h_3'  # Player 3 HQ (top-right)
        
        if num_players >= 4:
            map_data.iloc[height-2, 1] = 'h_4'  # Player 4 HQ (bottom-left)
        
        # Launch editor
        editor = MapEditor(
            self.screen,
            map_data,
            None,  # No filename yet
            num_players
        )
        editor_result = editor.run()
        pygame.event.clear()
        
        return editor_result

    def _edit_map(self) -> Optional[Dict[str, Any]]:
        """
        Edit an existing map.

        Returns:
            Result dict or None
        """
        # Show map selection for 1v1 maps
        map_menu_1v1 = MapSelectionMenu(self.screen, game_mode="1v1")
        selected_map = map_menu_1v1.run()
        pygame.event.clear()
        
        if not selected_map:
            # Try 2v2 maps if user cancelled 1v1
            map_menu_2v2 = MapSelectionMenu(self.screen, game_mode="2v2")
            selected_map = map_menu_2v2.run()
            pygame.event.clear()
        
        if not selected_map or selected_map == "random":
            return None  # User cancelled or selected random
        
        # Load the map
        map_data = FileIO.load_map(selected_map, for_ui=True, border_size=2)
        if map_data is None:
            print(f"Failed to load map: {selected_map}")
            return None
        
        # Detect number of players
        num_players = self._detect_num_players(map_data)
        
        # Launch editor
        editor = MapEditor(
            self.screen,
            map_data,
            selected_map,
            num_players
        )
        editor_result = editor.run()
        pygame.event.clear()
        
        return editor_result

    def _detect_num_players(self, map_data: pd.DataFrame) -> int:
        """
        Detect the number of players in a map.

        Args:
            map_data: Map data

        Returns:
            Number of players (2-4)
        """
        # Count unique player numbers in headquarters
        max_player = 0
        
        for row in range(map_data.shape[0]):
            for col in range(map_data.shape[1]):
                cell = str(map_data.iloc[row, col])
                if cell.startswith('h_'):
                    parts = cell.split('_')
                    if len(parts) == 2 and parts[1].isdigit():
                        player_num = int(parts[1])
                        max_player = max(max_player, player_num)
        
        # Default to 2 if no headquarters found
        return max(2, max_player)

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the map editor menu loop.

        Returns:
            Result dict or None
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
                    pygame.quit()
                    import sys
                    sys.exit()
                
                result = self.handle_input(event)
                if result is not None:
                    return result
            
            self.draw()
            clock.tick(30)
        
        return result
