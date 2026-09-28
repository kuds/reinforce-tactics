"""
Board art for the Pygame interface: colours, sprite files and animation layout.

Keyed by the engine's tile and unit codes (:mod:`reinforcetactics.rules`).
Plain data with no pygame import, so it loads in headless installs too
(:mod:`reinforcetactics.constants` re-exports it). Menu chrome colours live
in :mod:`reinforcetactics.ui.theme`.
"""

from typing import TypedDict

from reinforcetactics.rules import TileType


class UnitAssets(TypedDict):
    """Art for one unit type (an entry of :data:`UNIT_ASSETS`)."""

    static_path: str  # Still sprite, in the unit sprites folder
    animation_path: str  # Sprite-sheet base name, in the animation folder
    color: tuple[int, int, int]  # Letter colour when no sprite loads


# Display settings
TILE_SIZE = 32
FPS = 60

# Tile type colors (fallback when images aren't available)
# Made more distinct and vibrant
TILE_COLORS = {
    TileType.GRASS.value: (100, 200, 100),  # Grass - Bright green
    TileType.WATER.value: (50, 120, 200),  # Water - Blue
    TileType.MOUNTAIN.value: (150, 150, 150),  # Mountain - Light gray
    TileType.FOREST.value: (34, 139, 34),  # Forest - Forest green
    TileType.ROAD.value: (160, 130, 80),  # Road - Brown/tan
    TileType.BUILDING.value: (180, 180, 180),  # Building - Light gray (player-colored)
    TileType.HEADQUARTERS.value: (200, 200, 50),  # Headquarters - Yellow (player-colored)
    TileType.TOWER.value: (220, 220, 220),  # Tower - Light gray
    TileType.OCEAN.value: (0, 39, 232),  # Ocean - Dark Blue
}

# Player colors - Made more vibrant
PLAYER_COLORS = {
    1: (255, 50, 50),  # Red - Brighter
    2: (77, 121, 255),  # Blue - Brighter
    3: (50, 255, 50),  # Green - Brighter
    4: (255, 255, 50),  # Yellow - Brighter
}

# Base sprite sheet palette (blue tones) to be replaced per team.
# These are the exact RGB values in the sprite sheet PNGs that represent
# the unit's "team colour" regions, from darkest to lightest.
BASE_SPRITE_COLORS = [
    (30, 87, 156),  # #1e579c - darkest
    (60, 94, 139),  # #3c5e8b
    (47, 114, 144),  # #2f7290
    (40, 134, 176),  # #2886b0
    (61, 165, 211),  # #3da5d3 - lightest
    # Additional blue tones for units and bases
    (7, 109, 191),  # #076dbf
    (0, 152, 219),  # #0098db
    (79, 143, 186),  # #4f8fba
    (115, 190, 211),  # #73bed3
]

# Per-team replacement palettes (same length / order as BASE_SPRITE_COLORS).
# ``None`` means "keep the base colours as-is" (blue team uses the originals).
TEAM_PALETTES = {
    1: [  # Red
        (156, 47, 38),
        (139, 72, 65),
        (168, 62, 54),
        (194, 58, 48),
        (220, 90, 75),
        # Replacements for additional blue tones
        (202, 31, 15),
        (232, 28, 9),
        (197, 97, 88),
        (224, 134, 126),
    ],
    2: None,  # Blue – sprites are already blue, no swap needed
    3: [  # Green
        (38, 130, 56),
        (65, 125, 78),
        (54, 148, 80),
        (48, 168, 98),
        (75, 206, 122),
        # Replacements for additional blue tones
        (24, 180, 68),
        (21, 206, 73),
        (84, 175, 110),
        (117, 198, 140),
    ],
    4: [  # Yellow
        (156, 142, 30),
        (139, 130, 60),
        (158, 148, 47),
        (186, 176, 40),
        (218, 208, 61),
        # Replacements for additional blue tones
        (199, 181, 5),
        (228, 207, 0),
        (193, 183, 81),
        (219, 210, 119),
    ],
}

# Neutral (unowned) structure palette – white/gray tones used when a
# capturable structure has no owning player (tile.player is None).
NEUTRAL_STRUCTURE_PALETTE = [
    (100, 100, 112),  # dark gray
    (120, 120, 130),  # medium-dark gray
    (142, 142, 152),  # medium gray
    (172, 172, 182),  # medium-light gray
    (204, 204, 214),  # light gray
    # Neutrals for additional blue tones
    (174, 174, 185),
    (200, 200, 212),
    (170, 170, 180),
    (192, 192, 205),
]

# Tile type names (from TILE_TYPES values) that should receive team
# colour recoloring when rendered as sprites.
STRUCTURE_TILE_TYPES = {"TOWER", "BUILDING", "HEADQUARTERS"}

# Per-unit art, keyed by the unit codes of rules.UNIT_DATA (same order)
UNIT_ASSETS: dict[str, UnitAssets] = {
    "W": {"static_path": "warrior.png", "animation_path": "warrior", "color": (139, 69, 19)},  # Brown
    "M": {"static_path": "mage.png", "animation_path": "mage", "color": (138, 43, 226)},  # Purple
    "C": {"static_path": "cleric.png", "animation_path": "cleric", "color": (255, 215, 0)},  # Gold
    "B": {"static_path": "barbarian.png", "animation_path": "barbarian", "color": (0, 215, 0)},  # Green
    "A": {"static_path": "archer.png", "animation_path": "archer", "color": (34, 139, 34)},  # Forest Green
    "K": {"static_path": "knight.png", "animation_path": "knight", "color": (192, 192, 192)},  # Silver
    "R": {"static_path": "rogue.png", "animation_path": "rogue", "color": (64, 64, 64)},  # Dark Gray
    "S": {"static_path": "sorcerer.png", "animation_path": "sorcerer", "color": (0, 191, 255)},  # Deep Sky Blue
}

# Tile images
TILE_IMAGES = {
    "GRASS": "grass.png",
    "WATER": "water.png",
    "OCEAN": "ocean.png",
    "MOUNTAIN": "mountain.png",
    "FOREST": "forest.png",
    "ROAD": "road.png",
    "TOWER": "city.png",
    "BUILDING": "building.png",
    "HEADQUARTERS": "headquarters.png",
}

# Animation configuration for sprite sheets
#
# Sprite sheets use a 6-column grid of 64x64 frames.  Each frame is cropped
# to ``crop`` (48x48, ending just below the feet and drop shadow at y=51) and
# drawn with its bottom edge on the tile's bottom edge, so the sprite
# overflows its 32 px tile upward and sideways.
#
#   Idle (4 frames):         [0,0] [0,1] [0,2] [0,3]
#   Move Left (8 frames):   [0,4] [0,5] [1,0] [1,1] [1,2] [1,3] [1,4] [1,5]
#   Move Down (8 frames):   [2,0] [2,1] [2,2] [2,3] [2,4] [2,5] [3,0] [3,1]
#   Move Up (8 frames):     [3,2] [3,3] [3,4] [3,5] [4,0] [4,1] [4,2] [4,3]
#   Move Right:              auto-generated as horizontal flip of Move Left
#
ANIMATION_CONFIG = {
    # Source frame size on the sprite sheet
    "frame_width": 64,
    "frame_height": 64,
    # Crop of each frame, as (x, y, width, height) within the frame
    "crop": (8, 4, 48, 48),
    # Tile size the sheet art is drawn for; frames are scaled by
    # TILE_SIZE / art_tile_size (nearest-neighbour)
    "art_tile_size": 32,
    # Frame map: animation state -> list of (row, col) coordinates
    # This replaces the old row-based state_rows mapping.
    "frame_map": {
        "idle": [(0, 0), (0, 1), (0, 2), (0, 3)],
        "move_left": [
            (0, 4),
            (0, 5),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
        ],
        "move_down": [
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 0),
            (3, 1),
        ],
        "move_up": [
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ],
    },
    # States that are generated by horizontally flipping another state
    "mirror_states": {
        "move_right": "move_left",
    },
    # Animation state speed (seconds per frame)
    "states": {
        "idle": {"speed": 0.2},
        "move_down": {"speed": 0.1},
        "move_up": {"speed": 0.1},
        "move_left": {"speed": 0.1},
        "move_right": {"speed": 0.1},
    },
    # Per-unit type overrides (optional)
    # Example: 'units': {'W': {'frame_width': 48, 'frame_height': 48}}
    "units": {},
}


def tile_color(tile_type: str, owner: int | None) -> tuple[int, ...]:
    """The flat colour of a ``tile_type`` tile owned by player ``owner`` (None = neutral).

    Drawn when no tile sprite is available. Takes the owner rather than a
    tile so the renderer can draw a structure under fog of war with the owner
    its viewer knows, not the live one.
    """
    base_color = TILE_COLORS.get(tile_type, (0, 0, 0))

    # For structures (buildings, HQ, towers), emphasize player color more
    if owner and owner in PLAYER_COLORS:
        player_color = PLAYER_COLORS[owner]

        if tile_type in ["h", "b", "t"]:
            # Structures: 70% player color, 30% base color
            return tuple(min(int(base * 0.3 + player * 0.7), 255) for base, player in zip(base_color, player_color))
        else:
            # Regular terrain with owner: 60% base, 40% player
            return tuple(min(int(base * 0.6 + player * 0.4), 255) for base, player in zip(base_color, player_color))

    return base_color
