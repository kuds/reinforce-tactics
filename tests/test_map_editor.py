"""
Tests for the map editor functionality.
"""
import os
import pytest
import pygame
import pandas as pd
import numpy as np
from pathlib import Path

from reinforcetactics.ui.menus.map_editor.new_map_dialog import NewMapDialog
from reinforcetactics.ui.menus.map_editor.tile_palette import TilePalette
from reinforcetactics.ui.menus.map_editor.editor_canvas import EditorCanvas
from reinforcetactics.ui.menus.map_editor.map_editor import MapEditor
from reinforcetactics.ui.menus.map_editor.map_editor_menu import MapEditorMenu
from reinforcetactics.constants import MIN_MAP_SIZE
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def dummy_display():
    """Set up dummy display for headless testing."""
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    yield screen
    pygame.quit()


class TestNewMapDialog:
    """Test the new map dialog."""

    def test_initialization(self, dummy_display):
        """Test dialog initialization."""
        dialog = NewMapDialog(dummy_display)
        assert dialog.width_value >= MIN_MAP_SIZE
        assert dialog.height_value >= MIN_MAP_SIZE
        assert 2 <= dialog.num_players <= 4

    def test_create_map(self, dummy_display):
        """Test map creation."""
        dialog = NewMapDialog(dummy_display)
        result = dialog._create()
        assert result is not None
        assert 'width' in result
        assert 'height' in result
        assert 'num_players' in result
        assert result['width'] >= MIN_MAP_SIZE
        assert result['height'] >= MIN_MAP_SIZE


class TestTilePalette:
    """Test the tile palette."""

    def test_initialization(self, dummy_display):
        """Test palette initialization."""
        palette = TilePalette(20, 20, 250, 500, num_players=2)
        assert palette.selected_tile == 'p'
        assert palette.selected_player == 1

    def test_terrain_tile_selection(self, dummy_display):
        """Test terrain tile selection."""
        palette = TilePalette(20, 20, 250, 500, num_players=2)
        palette.selected_tile = 'p'
        assert palette.get_selected_tile() == 'p'
        
        palette.selected_tile = 'w'
        assert palette.get_selected_tile() == 'w'

    def test_structure_tile_selection(self, dummy_display):
        """Test structure tile with player ownership."""
        palette = TilePalette(20, 20, 250, 500, num_players=2)
        
        # Headquarters with player 1
        palette.selected_tile = 'h'
        palette.selected_player = 1
        assert palette.get_selected_tile() == 'h_1'
        
        # Building with player 2
        palette.selected_tile = 'b'
        palette.selected_player = 2
        assert palette.get_selected_tile() == 'b_2'


class TestEditorCanvas:
    """Test the editor canvas."""

    def test_initialization(self, dummy_display):
        """Test canvas initialization."""
        map_data = pd.DataFrame(np.full((30, 30), 'p', dtype=object))
        canvas = EditorCanvas(20, 20, 600, 500, map_data)
        assert canvas.tile_size == 32
        assert canvas.grid_enabled is True

    def test_grid_toggle(self, dummy_display):
        """Test grid toggle."""
        map_data = pd.DataFrame(np.full((30, 30), 'p', dtype=object))
        canvas = EditorCanvas(20, 20, 600, 500, map_data)
        assert canvas.grid_enabled is True
        canvas.toggle_grid()
        assert canvas.grid_enabled is False
        canvas.toggle_grid()
        assert canvas.grid_enabled is True

    def test_zoom(self, dummy_display):
        """Test zoom functionality."""
        map_data = pd.DataFrame(np.full((30, 30), 'p', dtype=object))
        canvas = EditorCanvas(20, 20, 600, 500, map_data)
        initial_size = canvas.tile_size
        
        canvas.zoom_in()
        assert canvas.tile_size > initial_size
        
        canvas.zoom_out()
        assert canvas.tile_size == initial_size


class TestMapEditor:
    """Test the main map editor."""

    def test_initialization(self, dummy_display):
        """Test editor initialization."""
        map_data = pd.DataFrame(np.full((25, 25), 'o', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[23, 23] = 'h_2'
        
        editor = MapEditor(dummy_display, map_data, None, num_players=2)
        assert editor.map_data is not None
        assert editor.num_players == 2

    def test_valid_map_validation(self, dummy_display):
        """Test validation of a valid map."""
        map_data = pd.DataFrame(np.full((25, 25), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[23, 23] = 'h_2'
        
        editor = MapEditor(dummy_display, map_data, None, num_players=2)
        errors = editor._validate_map()
        assert len(errors) == 0

    def test_missing_hq_validation(self, dummy_display):
        """Test validation fails when HQ is missing."""
        map_data = pd.DataFrame(np.full((25, 25), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        # Player 2 HQ missing
        
        editor = MapEditor(dummy_display, map_data, None, num_players=2)
        errors = editor._validate_map()
        assert len(errors) > 0

    def test_small_map_validation(self, dummy_display):
        """Test validation fails for maps smaller than minimum size."""
        map_data = pd.DataFrame(np.full((10, 10), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[8, 8] = 'h_2'
        
        editor = MapEditor(dummy_display, map_data, None, num_players=2)
        errors = editor._validate_map()
        assert len(errors) > 0

    def test_save_map(self, dummy_display, tmp_path):
        """Test map saving functionality."""
        map_data = pd.DataFrame(np.full((25, 25), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[23, 23] = 'h_2'
        
        editor = MapEditor(dummy_display, map_data, None, num_players=2)
        
        # Set temporary save path
        save_path = tmp_path / "test_map.csv"
        editor.map_filename = str(save_path)
        
        # Save the map
        success = editor._save_map()
        assert success is True
        assert save_path.exists()
        
        # Load and verify
        loaded_map = FileIO.load_map(str(save_path))
        assert loaded_map is not None
        assert loaded_map.iloc[1, 1] == 'h_1'
        assert loaded_map.iloc[23, 23] == 'h_2'


class TestMapEditorMenu:
    """Test the map editor menu."""

    def test_initialization(self, dummy_display):
        """Test menu initialization."""
        menu = MapEditorMenu(dummy_display)
        assert len(menu.options) > 0
        option_names = [opt[0] for opt in menu.options]
        assert 'New Map' in option_names
        assert 'Edit Existing Map' in option_names

    def test_detect_num_players(self, dummy_display):
        """Test player detection from map."""
        menu = MapEditorMenu(dummy_display)
        
        # 2-player map
        map_data = pd.DataFrame(np.full((25, 25), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[23, 23] = 'h_2'
        assert menu._detect_num_players(map_data) == 2
        
        # 4-player map
        map_data.iloc[1, 23] = 'h_3'
        map_data.iloc[23, 1] = 'h_4'
        assert menu._detect_num_players(map_data) == 4
