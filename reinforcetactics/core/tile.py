"""
Tile class representing a single tile in the game grid.
"""

import logging

from reinforcetactics.rules import (
    BUILDING_MAX_HEALTH,
    HEADQUARTERS_MAX_HEALTH,
    TOWER_MAX_HEALTH,
    TileType,
)

logger = logging.getLogger(__name__)

# Tile codes a map may use; anything else loads as ocean.
_VALID_TILE_CODES = frozenset(tile_type.value for tile_type in TileType)
_IMPASSABLE_TILE_CODES = frozenset(tile_type.value for tile_type in TileType if not tile_type.is_walkable())
_CAPTURABLE_TILE_CODES = frozenset(tile_type.value for tile_type in TileType if tile_type.is_capturable())


class Tile:
    """Represents a single tile in the grid with type, player ownership, and team info."""

    def __init__(self, tile_data, x, y):
        """
        Initialize a tile from CSV data.

        Args:
            tile_data: String in format "type" or "type_player" or "type_player_team"
            x: X coordinate in grid
            y: Y coordinate in grid
        """
        # Convert to string and strip whitespace
        tile_str = str(tile_data).strip()

        # Handle NaN, empty, or invalid values
        if tile_str in ["", "nan", "None", "NaN"]:
            tile_str = "o"  # Default to ocean (impassable), not open ground

        # Split by underscore
        parts = tile_str.split("_")

        self.type = parts[0].strip()  # Strip whitespace from type too
        self.player = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
        self.team = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else None
        self.x = x
        self.y = y

        # Validate tile type - if invalid, default to ocean (impassable)
        if self.type not in _VALID_TILE_CODES:
            logger.warning("Invalid tile type %r at (%d, %d), defaulting to ocean", self.type, x, y)
            self.type = "o"

        # Tower/Headquarters/Building-specific properties
        if self.type == "t":
            self.max_health = TOWER_MAX_HEALTH
            self.health = TOWER_MAX_HEALTH
            self.regenerating = False
        elif self.type == "h":
            self.max_health = HEADQUARTERS_MAX_HEALTH
            self.health = HEADQUARTERS_MAX_HEALTH
            self.regenerating = False
        elif self.type == "b":
            self.max_health = BUILDING_MAX_HEALTH
            self.health = BUILDING_MAX_HEALTH
            self.regenerating = False
        else:
            self.max_health = None
            self.health = None
            self.regenerating = False

    def is_walkable(self):
        """Check if this tile can be walked on."""
        return self.type not in _IMPASSABLE_TILE_CODES  # Water and ocean

    def is_capturable(self):
        """Check if this tile can be captured."""
        return self.type in _CAPTURABLE_TILE_CODES

    def to_dict(self):
        """Convert tile to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "player": self.player,
            "health": self.health,
            "regenerating": self.regenerating,
        }

    @classmethod
    def from_dict(cls, data):
        """Create tile from dictionary."""
        tile_str = data["type"]
        if data.get("player"):
            tile_str += f"_{data['player']}"

        tile = cls(tile_str, data["x"], data["y"])
        if data.get("health") is not None:
            tile.health = data["health"]
        if data.get("regenerating") is not None:
            tile.regenerating = data["regenerating"]

        return tile
