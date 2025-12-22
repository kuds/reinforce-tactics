"""
Game constants for the 2D Strategy Game.
"""
from enum import Enum
from typing import Dict, List, TypedDict


class TileType(Enum):
    """Enumeration of tile types in the game."""
    GRASS = 'p'
    WATER = 'w'
    MOUNTAIN = 'm'
    FOREST = 'f'
    ROAD = 'r'
    BUILDING = 'b'
    HEADQUARTERS = 'h'
    TOWER = 't'
    OCEAN = 'o'

    @classmethod
    def from_code(cls, code: str) -> 'TileType':
        """Get TileType from single-letter code."""
        for tile_type in cls:
            if tile_type.value == code:
                return tile_type
        raise ValueError(f"Unknown tile code: {code}")

    def is_walkable(self) -> bool:
        """Check if this tile type can be walked on."""
        return self not in (TileType.WATER, TileType.OCEAN, TileType.MOUNTAIN)

    def is_capturable(self) -> bool:
        """Check if this tile type can be captured."""
        return self in (TileType.TOWER, TileType.HEADQUARTERS, TileType.BUILDING)


# Display settings
TILE_SIZE = 32
FPS = 60
MIN_MAP_SIZE = 20
MIN_STRIP_SIZE = 6  # Minimum size to preserve when stripping water borders

# Tile type colors (fallback when images aren't available)
# Made more distinct and vibrant
TILE_COLORS = {
    TileType.GRASS.value: (100, 200, 100),      # Grass - Bright green
    TileType.WATER.value: (50, 120, 200),       # Water - Blue
    TileType.MOUNTAIN.value: (150, 150, 150),   # Mountain - Light gray
    TileType.FOREST.value: (34, 139, 34),       # Forest - Forest green
    TileType.ROAD.value: (160, 130, 80),        # Road - Brown/tan
    TileType.BUILDING.value: (180, 180, 180),   # Building - Light gray (player-colored)
    TileType.HEADQUARTERS.value: (200, 200, 50),  # Headquarters - Yellow (player-colored)
    TileType.TOWER.value: (220, 220, 220),      # Tower - Light gray
    TileType.OCEAN.value: (0, 39, 232),         # Ocean - Dark Blue
    # Keep string keys for backwards compatibility
    'p': (100, 200, 100),
    'w': (50, 120, 200),
    'm': (150, 150, 150),
    'f': (34, 139, 34),
    'r': (160, 130, 80),
    'b': (180, 180, 180),
    'h': (200, 200, 50),
    't': (220, 220, 220),
    'o': (0, 39, 232),
}

# Player colors - Made more vibrant
PLAYER_COLORS = {
    1: (255, 50, 50),     # Red - Brighter
    2: (77, 121, 255),    # Blue - Brighter
    3: (50, 255, 50),     # Green - Brighter
    4: (255, 255, 50)     # Yellow - Brighter
}

# Unit colors
UNIT_COLORS = {
    'W': (139, 69, 19),      # Brown (Warrior)
    'M': (138, 43, 226),     # Purple (Mage)
    'C': (255, 215, 0),      # Gold (Cleric)
    'B': (0, 215, 0),        # Barbarian (Green)
    'A': (34, 139, 34)       # Archer (Forest Green)
}

# Unit costs and properties
UNIT_DATA = {
    'W': {
        'static_path': 'warrior.png',
        'animation_path': 'warrior',
        'name': 'Warrior',
        'cost': 200,
        'color': (139, 69, 19),
        'movement': 3,
        'health': 15,
        'attack': 10,
        'defence': 6
    },
    'M': {
        'static_path': 'mage.png',
        'animation_path': 'mage',
        'name': 'Mage',
        'cost': 250,
        'color': (138, 43, 226),
        'movement': 2,
        'health': 10,
        'attack': {'adjacent': 8, 'range': 12},
        'defence': 4
    },
    'C': {
        'static_path': 'cleric.png',
        'animation_path': 'cleric',
        'name': 'Cleric',
        'cost': 200,
        'color': (255, 215, 0),
        'movement': 2,
        'health': 8,
        'attack': 2,
        'defence': 4
    },
    'B': {
        'static_path': 'barbarian.png',
        'animation_path': 'barbarian',
        'name': 'Barbarian',
        'cost': 400,
        'color': (0, 215, 0),
        'movement': 5,
        'health': 20,
        'attack': 10,
        'defence': 2
    },
    'A': {
        'static_path': 'archer.png',
        'animation_path': 'archer',
        'name': 'Archer',
        'cost': 250,
        'color': (34, 139, 34),
        'movement': 3,
        'health': 15,
        'attack': 5,
        'defence': 1
    }
}


class UnitConfig(TypedDict, total=False):
    """Configuration for which units are enabled for purchase.

    All units default to True (enabled) if not specified.
    Example: {'M': False, 'A': False} disables Mage and Archer.
    """
    W: bool  # Warrior
    M: bool  # Mage
    C: bool  # Cleric
    B: bool  # Barbarian
    A: bool  # Archer


# All unit type codes
ALL_UNIT_TYPES: List[str] = ['W', 'M', 'C', 'B', 'A']

# Default purchasable units (excludes Barbarian due to high cost)
DEFAULT_PURCHASABLE_UNITS: List[str] = ['W', 'M', 'C', 'A']


def get_enabled_units(config: Dict[str, bool] | None = None) -> List[str]:
    """Get list of enabled unit types based on configuration.

    Args:
        config: Optional dict mapping unit type codes to enabled status.
                Units not in the dict default to True (enabled).
                Only units in DEFAULT_PURCHASABLE_UNITS are considered.

    Returns:
        List of enabled unit type codes.
    """
    if config is None:
        return DEFAULT_PURCHASABLE_UNITS.copy()

    return [u for u in DEFAULT_PURCHASABLE_UNITS if config.get(u, True)]


# Starting gold for each player
STARTING_GOLD = 250

# Income rates
HEADQUARTERS_INCOME = 150
BUILDING_INCOME = 100
TOWER_INCOME = 50

# Structure health
TOWER_MAX_HEALTH = 30
BUILDING_MAX_HEALTH = 40
HEADQUARTERS_MAX_HEALTH = 50

# Structure regeneration rate (percentage of max HP per turn)
STRUCTURE_REGEN_RATE = 0.5

# Combat
COUNTER_ATTACK_MULTIPLIER = 0.8

# Status effects
PARALYZE_DURATION = 3
HEAL_AMOUNT = 5

# Tile type mapping (string code -> display name)
# Kept for backwards compatibility
TILE_TYPES = {
    TileType.GRASS.value: 'GRASS',
    TileType.WATER.value: 'WATER',
    TileType.MOUNTAIN.value: 'MOUNTAIN',
    TileType.FOREST.value: 'FOREST',
    TileType.ROAD.value: 'ROAD',
    TileType.BUILDING.value: 'BUILDING',
    TileType.HEADQUARTERS.value: 'HEADQUARTERS',
    TileType.TOWER.value: 'TOWER',
    TileType.OCEAN.value: 'OCEAN',
    # Also keep simple string keys for backwards compatibility
    'p': 'GRASS',
    'w': 'WATER',
    'm': 'MOUNTAIN',
    'f': 'FOREST',
    'r': 'ROAD',
    'b': 'BUILDING',
    'h': 'HEADQUARTERS',
    't': 'TOWER',
    'o': 'OCEAN'
}

# Tile images
TILE_IMAGES = {
    'GRASS': 'grass.png',
    'WATER': 'water.png',
    'OCEAN': 'ocean.png',
    'MOUNTAIN': 'mountain.png',
    'FOREST': 'forest.png',
    'ROAD': 'road.png',
    'TOWER': 'city.png',
    'BUILDING': 'building.png',
    'HEADQUARTERS': 'headquarters.png'
}
