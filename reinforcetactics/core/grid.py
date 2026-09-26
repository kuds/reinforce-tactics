"""
Grid class for managing the tile map.
"""

import numpy as np

from reinforcetactics.core.tile import Tile


class TileGrid:
    """Manages the grid of tiles."""

    def __init__(self, map_data):
        """
        Initialize the grid from map data.

        Args:
            map_data: Tile codes by row: a pandas DataFrame, numpy array or
                list of lists
        """
        codes = np.asarray(map_data, dtype=object)
        if codes.ndim != 2:
            raise ValueError(f"map_data must be a 2D grid of tile codes, got shape {codes.shape}")
        self.height, self.width = codes.shape
        self.tiles = [[Tile(codes[y, x], x, y) for x in range(self.width)] for y in range(self.height)]

        # Structure (HQ/building/tower) positions in row-major order. Tile
        # types never change after the grid is built, so fog-of-war code can
        # visit just these instead of scanning every tile (review core-21).
        self.structure_positions: list[tuple[int, int]] = [
            (tile.x, tile.y) for row in self.tiles for tile in row if tile.is_capturable()
        ]

    def get_tile(self, x, y):
        """Get tile at coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.tiles[y][x]
        return None

    def get_capturable_tiles(self, player=None):
        """Get all capturable tiles, optionally filtered by player."""
        tiles = [tile for row in self.tiles for tile in row if tile.is_capturable()]
        if player is not None:
            tiles = [tile for tile in tiles if tile.player == player]
        return tiles

    def to_numpy(self):
        """
        Convert grid to numpy representation for RL.

        Returns:
            numpy array of shape (height, width, channels) where channels are:
            0: tile_type (encoded as int; must stay in sync with
               ``reinforcetactics.rl.observation.TILE_TYPE_ORDER``)
            1: tile_owner (0 for neutral, 1-4 for players)
            2: structure_hp_percentage (0-100)
        """
        result = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Ocean ("o") shares the water ("w") code: both are impassable and
        # mechanically identical, so the RL observation treats them as one
        # terrain class. Without the explicit "o" entry it would fall through
        # the ``.get(..., 0)`` default below and be encoded as grass ("p"),
        # making impassable ocean indistinguishable from open ground.
        tile_type_encoding = {"p": 0, "w": 1, "m": 2, "f": 3, "r": 4, "b": 5, "h": 6, "t": 7, "o": 1}

        for y in range(self.height):
            for x in range(self.width):
                tile = self.tiles[y][x]
                result[y, x, 0] = tile_type_encoding.get(tile.type, 0)
                result[y, x, 1] = tile.player if tile.player else 0

                if tile.health is not None and tile.max_health is not None and tile.max_health > 0:
                    result[y, x, 2] = (tile.health / tile.max_health) * 100
                else:
                    result[y, x, 2] = 0

        return result

    def to_dict(self):
        """Convert grid to dictionary for serialization."""
        return {
            "width": self.width,
            "height": self.height,
            "tiles": [tile.to_dict() for row in self.tiles for tile in row if tile.is_capturable()],
        }
