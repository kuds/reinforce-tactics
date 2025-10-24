"""
Game constants for the 2D Strategy Game.
"""

# Display settings
TILE_SIZE = 32
FPS = 60
MIN_MAP_SIZE = 20

# Tile type colors (fallback when images aren't available)
# Made more distinct and vibrant
TILE_COLORS = {
    'p': (100, 200, 100),    # Grass - Bright green
    'w': (50, 120, 200),     # Water - Blue
    'm': (150, 150, 150),    # Mountain - Light gray
    'f': (34, 139, 34),      # Forest - Forest green
    'r': (160, 130, 80),     # Road - Brown/tan
    'b': (180, 180, 180),    # Building - Light gray (will be colored by player)
    'h': (200, 200, 50),     # Headquarters - Yellow base (will be colored by player)
    't': (220, 220, 220),    # Tower - Light gray
    'o': (0, 39, 232)        # Ocean - Dark Blue
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
    'B': (0, 215, 0)         # Barbarian (Green)
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
    }
}

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
COUNTER_ATTACK_MULTIPLIER = 0.9

# Status effects
PARALYZE_DURATION = 3
HEAL_AMOUNT = 5

# Tile type mapping
TILE_TYPES = {
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