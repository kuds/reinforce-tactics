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
from reinforcetactics.constants import MIN_MAP_SIZE, MIN_STRIP_SIZE
from reinforcetactics.utils.file_io import FileIO


@pytest.fixture
def dummy_display():
    """Set up dummy display for headless testing."""
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    pygame.init()
    screen = pygame.display.set_mode((1500, 1000))
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
        assert palette.selected_player == 0  # Defaults to neutral

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

    def test_neutral_structure_selection(self, dummy_display):
        """Test neutral structure selection (player 0)."""
        palette = TilePalette(20, 20, 250, 500, num_players=2)
        
        # Neutral tower
        palette.selected_tile = 't'
        palette.selected_player = 0
        assert palette.get_selected_tile() == 't'
        
        # Neutral building
        palette.selected_tile = 'b'
        palette.selected_player = 0
        assert palette.get_selected_tile() == 'b'
        
        # Neutral headquarters
        palette.selected_tile = 'h'
        palette.selected_player = 0
        assert palette.get_selected_tile() == 'h'
        
    def test_player_owned_vs_neutral_tower(self, dummy_display):
        """Test that towers can be neutral or player-owned."""
        palette = TilePalette(20, 20, 250, 500, num_players=2)
        
        # Tower with player 1
        palette.selected_tile = 't'
        palette.selected_player = 1
        assert palette.get_selected_tile() == 't_1'
        
        # Neutral tower (player 0)
        palette.selected_player = 0
        assert palette.get_selected_tile() == 't'



class TestEditorCanvas:
    """Test the editor canvas."""

    def test_initialization(self, dummy_display):
        """Test canvas initialization."""
        map_data = pd.DataFrame(np.full((30, 30), 'p', dtype=object))
        canvas = EditorCanvas(20, 20, 600, 500, map_data)
        assert canvas.tile_size == 24
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


class TestWaterBorderStripping:
    """Test water border stripping and restoration."""

    def test_strip_water_border_fully_surrounded(self, dummy_display):
        """Test stripping when map is fully surrounded by ocean."""
        # Create a 10x10 map with 2 layers of ocean border
        map_data = pd.DataFrame(np.full((10, 10), 'o', dtype=object))
        # Inner 6x6 area has grass
        for i in range(2, 8):
            for j in range(2, 8):
                map_data.iloc[i, j] = 'p'
        
        stripped = FileIO.strip_water_border(map_data, min_size=MIN_STRIP_SIZE)
        
        # Should strip to 6x6
        assert stripped.shape == (6, 6)
        # All tiles should be grass
        assert (stripped == 'p').all().all()

    def test_strip_water_border_partial_ocean(self, dummy_display):
        """Test that partial ocean borders are not stripped."""
        # Create map with partial ocean border
        map_data = pd.DataFrame(np.full((10, 10), 'o', dtype=object))
        # Make one tile on the border not ocean
        map_data.iloc[0, 5] = 'p'  # Top row has one grass tile
        
        stripped = FileIO.strip_water_border(map_data, min_size=MIN_STRIP_SIZE)
        
        # Should not strip at all
        assert stripped.shape == (10, 10)

    def test_strip_water_border_respects_min_size(self, dummy_display):
        """Test that stripping stops at minimum size."""
        # Create an 8x8 map fully of ocean
        map_data = pd.DataFrame(np.full((8, 8), 'o', dtype=object))
        
        stripped = FileIO.strip_water_border(map_data, min_size=MIN_STRIP_SIZE)
        
        # Should strip to exactly 6x6, not smaller
        assert stripped.shape == (6, 6)

    def test_strip_water_border_no_ocean_border(self, dummy_display):
        """Test map with no ocean borders."""
        # Create map with grass borders
        map_data = pd.DataFrame(np.full((10, 10), 'p', dtype=object))
        
        stripped = FileIO.strip_water_border(map_data, min_size=MIN_STRIP_SIZE)
        
        # Should not strip anything
        assert stripped.shape == (10, 10)

    def test_strip_water_border_iterative(self, dummy_display):
        """Test iterative stripping of multiple layers."""
        # Create a 14x14 map with 4 layers of ocean border
        map_data = pd.DataFrame(np.full((14, 14), 'o', dtype=object))
        # Inner 6x6 area has grass
        for i in range(4, 10):
            for j in range(4, 10):
                map_data.iloc[i, j] = 'p'
        
        stripped = FileIO.strip_water_border(map_data, min_size=MIN_STRIP_SIZE)
        
        # Should strip all 4 layers to get to 6x6
        assert stripped.shape == (6, 6)
        assert (stripped == 'p').all().all()

    def test_strip_water_border_mixed_tiles(self, dummy_display):
        """Test stripping with various tile types inside."""
        # Create map with ocean border
        map_data = pd.DataFrame(np.full((12, 12), 'o', dtype=object))
        # Inner area has various tiles
        map_data.iloc[2:10, 2:10] = 'p'
        map_data.iloc[3, 3] = 'h_1'
        map_data.iloc[8, 8] = 'h_2'
        map_data.iloc[5, 5] = 'w'  # Water (not ocean)
        map_data.iloc[6, 6] = 'm'  # Mountain
        
        stripped = FileIO.strip_water_border(map_data, min_size=MIN_STRIP_SIZE)
        
        # Should strip 2 layers
        assert stripped.shape == (8, 8)
        # Check that interior tiles are preserved
        assert stripped.iloc[1, 1] == 'h_1'
        assert stripped.iloc[6, 6] == 'h_2'
        assert stripped.iloc[3, 3] == 'w'
        assert stripped.iloc[4, 4] == 'm'

    def test_add_water_border(self, dummy_display):
        """Test adding water borders."""
        # Create a small 6x6 map
        map_data = pd.DataFrame(np.full((6, 6), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[4, 4] = 'h_2'
        
        bordered = FileIO.add_water_border(map_data, border_size=2)
        
        # Should be 10x10 now
        assert bordered.shape == (10, 10)
        
        # Check borders are ocean
        assert all(bordered.iloc[0, :] == 'o')  # Top row
        assert all(bordered.iloc[-1, :] == 'o')  # Bottom row
        assert all(bordered.iloc[:, 0] == 'o')  # Left column
        assert all(bordered.iloc[:, -1] == 'o')  # Right column
        
        # Check interior is preserved with offset
        assert bordered.iloc[3, 3] == 'h_1'  # 1,1 + 2 offset
        assert bordered.iloc[6, 6] == 'h_2'  # 4,4 + 2 offset

    def test_add_water_border_zero_size(self, dummy_display):
        """Test adding zero-size border returns unchanged map."""
        map_data = pd.DataFrame(np.full((6, 6), 'p', dtype=object))
        
        bordered = FileIO.add_water_border(map_data, border_size=0)
        
        assert bordered.shape == (6, 6)
        assert (bordered == 'p').all().all()

    def test_save_map_strips_border(self, dummy_display, tmp_path):
        """Test that saving a map strips ocean borders."""
        # Create map with ocean border (larger than MIN_MAP_SIZE)
        map_data = pd.DataFrame(np.full((24, 24), 'o', dtype=object))
        # Inner 20x20 area
        for i in range(2, 22):
            for j in range(2, 22):
                map_data.iloc[i, j] = 'p'
        map_data.iloc[3, 3] = 'h_1'
        map_data.iloc[20, 20] = 'h_2'
        
        # Save the map
        save_path = tmp_path / "test_strip.csv"
        FileIO.save_map(map_data, str(save_path))
        
        # Load it back (without UI borders)
        loaded = FileIO.load_map(str(save_path), for_ui=False)
        
        # Should be stripped to 20x20
        assert loaded.shape == (20, 20)
        # Check HQs are at correct positions after stripping
        assert loaded.iloc[1, 1] == 'h_1'
        assert loaded.iloc[18, 18] == 'h_2'

    def test_load_map_for_ui(self, dummy_display, tmp_path):
        """Test loading map with UI borders."""
        # Create and save a map that's already at MIN_MAP_SIZE
        map_data = pd.DataFrame(np.full((20, 20), 'p', dtype=object))
        map_data.iloc[1, 1] = 'h_1'
        map_data.iloc[18, 18] = 'h_2'
        
        save_path = tmp_path / "test_ui_load.csv"
        # Save directly without borders
        map_data.to_csv(save_path, header=False, index=False)
        
        # Load for UI (should add borders)
        loaded_ui = FileIO.load_map(str(save_path), for_ui=True, border_size=2)
        
        # Should be 24x24 now (20 + 2*2)
        assert loaded_ui.shape == (24, 24)
        
        # Check borders are ocean
        assert all(loaded_ui.iloc[0, :] == 'o')
        assert all(loaded_ui.iloc[-1, :] == 'o')
        
        # Check interior is offset correctly
        assert loaded_ui.iloc[3, 3] == 'h_1'  # 1,1 + 2 offset
        assert loaded_ui.iloc[20, 20] == 'h_2'  # 18,18 + 2 offset

    def test_round_trip_save_load(self, dummy_display, tmp_path):
        """Test that save and load preserve map content."""
        # Create original map with borders (larger than MIN_MAP_SIZE)
        original = pd.DataFrame(np.full((24, 24), 'o', dtype=object))
        for i in range(2, 22):
            for j in range(2, 22):
                original.iloc[i, j] = 'p'
        original.iloc[5, 5] = 'h_1'
        original.iloc[20, 20] = 'h_2'
        original.iloc[10, 10] = 'm'
        original.iloc[12, 12] = 'w'
        
        # Save (will strip borders)
        save_path = tmp_path / "test_round_trip.csv"
        FileIO.save_map(original, str(save_path))
        
        # Load for UI (will add borders back)
        loaded = FileIO.load_map(str(save_path), for_ui=True, border_size=2)
        
        # Should have same dimensions
        assert loaded.shape == original.shape
        
        # Check critical tiles match
        assert loaded.iloc[5, 5] == 'h_1'
        assert loaded.iloc[20, 20] == 'h_2'
        assert loaded.iloc[10, 10] == 'm'
        assert loaded.iloc[12, 12] == 'w'
