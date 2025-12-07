"""Tests for the Tile class."""
from reinforcetactics.core.tile import Tile


class TestTileInitialization:
    """Test tile initialization."""

    def test_simple_grass_tile(self):
        """Test grass tile initialization."""
        tile = Tile('p', 5, 5)
        assert tile.type == 'p'
        assert tile.x == 5
        assert tile.y == 5
        assert tile.player is None
        assert tile.team is None
        assert tile.max_health is None
        assert tile.health is None

    def test_tile_with_player_ownership(self):
        """Test tile initialization with player ownership."""
        tile = Tile('h_1', 3, 4)
        assert tile.type == 'h'
        assert tile.player == 1
        assert tile.team is None
        assert tile.x == 3
        assert tile.y == 4
        assert tile.max_health == 50  # HEADQUARTERS_MAX_HEALTH
        assert tile.health == 50

    def test_tile_with_player_and_team(self):
        """Test tile initialization with player and team."""
        tile = Tile('t_2_1', 1, 2)
        assert tile.type == 't'
        assert tile.player == 2
        assert tile.team == 1
        assert tile.max_health == 30  # TOWER_MAX_HEALTH
        assert tile.health == 30

    def test_invalid_tile_defaults_to_grass(self):
        """Test invalid tile type defaults to ocean/grass."""
        tile = Tile('invalid', 0, 0)
        assert tile.type == 'o'

    def test_building_tile_initialization(self):
        """Test building tile initialization."""
        tile = Tile('b_1', 2, 3)
        assert tile.type == 'b'
        assert tile.player == 1
        assert tile.max_health == 40  # BUILDING_MAX_HEALTH
        assert tile.health == 40


class TestTileProperties:
    """Test tile property methods."""

    def test_grass_is_walkable(self):
        """Test grass tile is walkable."""
        tile = Tile('p', 0, 0)
        assert tile.is_walkable() is True

    def test_water_not_walkable(self):
        """Test water tile is not walkable."""
        tile = Tile('w', 0, 0)
        assert tile.is_walkable() is False

    def test_ocean_not_walkable(self):
        """Test ocean tile is not walkable."""
        tile = Tile('o', 0, 0)
        assert tile.is_walkable() is False

    def test_road_is_walkable(self):
        """Test road tile is walkable."""
        tile = Tile('r', 0, 0)
        assert tile.is_walkable() is True

    def test_mountain_is_walkable(self):
        """Test mountain tile is walkable."""
        tile = Tile('m', 0, 0)
        assert tile.is_walkable() is True

    def test_tower_is_capturable(self):
        """Test tower is capturable."""
        tile = Tile('t_1', 0, 0)
        assert tile.is_capturable() is True

    def test_hq_is_capturable(self):
        """Test HQ is capturable."""
        tile = Tile('h_2', 0, 0)
        assert tile.is_capturable() is True

    def test_building_is_capturable(self):
        """Test building is capturable."""
        tile = Tile('b_1', 0, 0)
        assert tile.is_capturable() is True

    def test_grass_not_capturable(self):
        """Test grass is not capturable."""
        tile = Tile('p', 0, 0)
        assert tile.is_capturable() is False


class TestTileColors:
    """Test tile color methods."""

    def test_grass_color(self):
        """Test grass tile returns correct color."""
        tile = Tile('p', 0, 0)
        color = tile.get_color()
        assert isinstance(color, tuple)
        assert len(color) == 3
        # Grass should be greenish
        assert color == (100, 200, 100)

    def test_water_color(self):
        """Test water tile returns correct color."""
        tile = Tile('w', 0, 0)
        color = tile.get_color()
        assert isinstance(color, tuple)
        assert len(color) == 3
        # Water should be bluish
        assert color == (50, 120, 200)

    def test_structure_with_player_color(self):
        """Test structure tile with player ownership has blended color."""
        tile = Tile('h_1', 0, 0)
        color = tile.get_color()
        assert isinstance(color, tuple)
        assert len(color) == 3
        # Should be a blend of base color and player 1 color (red)
        # Player 1 color is (255, 50, 50) and should dominate (70%)
        assert color != (200, 200, 50)  # Not just base color


class TestTileSerialization:
    """Test tile serialization and deserialization."""

    def test_to_dict_simple_tile(self):
        """Test simple tile serialization."""
        tile = Tile('p', 3, 4)
        data = tile.to_dict()

        assert data['x'] == 3
        assert data['y'] == 4
        assert data['type'] == 'p'
        assert data['player'] is None
        assert data['health'] is None
        assert data['regenerating'] is False

    def test_to_dict_structure_tile(self):
        """Test structure tile serialization."""
        tile = Tile('t_1', 5, 6)
        tile.health = 25
        tile.regenerating = True

        data = tile.to_dict()

        assert data['x'] == 5
        assert data['y'] == 6
        assert data['type'] == 't'
        assert data['player'] == 1
        assert data['health'] == 25
        assert data['regenerating'] is True

    def test_from_dict_simple_tile(self):
        """Test simple tile deserialization."""
        data = {
            'x': 2,
            'y': 3,
            'type': 'p',
            'player': None,
            'health': None,
            'regenerating': False
        }

        tile = Tile.from_dict(data)

        assert tile.x == 2
        assert tile.y == 3
        assert tile.type == 'p'
        assert tile.player is None

    def test_from_dict_structure_tile(self):
        """Test structure tile deserialization."""
        data = {
            'x': 4,
            'y': 5,
            'type': 'h',
            'player': 2,
            'health': 40,
            'regenerating': True
        }

        tile = Tile.from_dict(data)

        assert tile.x == 4
        assert tile.y == 5
        assert tile.type == 'h'
        assert tile.player == 2
        assert tile.health == 40
        assert tile.regenerating is True
