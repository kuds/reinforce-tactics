#!/usr/bin/env python3
"""
Generate PNG map previews for the documentation site.
Uses PIL instead of Pygame to avoid display requirements.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw
import pandas as pd


# Color definitions from constants.py
TILE_COLORS = {
    'p': (100, 200, 100),      # Plains/Grass - Bright green
    'w': (50, 120, 200),       # Water - Blue
    'm': (150, 150, 150),      # Mountain - Light gray
    'f': (34, 139, 34),        # Forest - Forest green
    'r': (160, 130, 80),       # Road - Brown/tan
    'b': (180, 180, 180),      # Building - Light gray
    'h': (200, 200, 50),       # HQ - Yellow
    't': (220, 220, 220),      # Tower - Light gray
    'o': (0, 39, 232),         # Ocean - Dark Blue
}

PLAYER_COLORS = {
    1: (255, 50, 50),     # Red
    2: (77, 121, 255),    # Blue
    3: (50, 255, 50),     # Green
    4: (255, 255, 50)     # Yellow
}

TERRAIN_DISPLAY_NAMES = {
    'p': 'Plains',
    'o': 'Ocean',
    'w': 'Water',
    'm': 'Mountain',
    'f': 'Forest',
    'r': 'Road',
    't': 'Tower',
    'h': 'HQ',
    'b': 'Building',
}


def load_map(filepath: str) -> pd.DataFrame:
    """Load a map from CSV file."""
    try:
        return pd.read_csv(filepath, header=None)
    except Exception as e:
        print(f"Error loading map {filepath}: {e}")
        return None


def get_tile_color(tile: str) -> tuple:
    """Get the RGB color for a tile type."""
    if '_' in tile:
        parts = tile.split('_')
        base_type = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            player_num = int(parts[1])
            return PLAYER_COLORS.get(player_num, TILE_COLORS.get(base_type, (128, 128, 128)))
    return TILE_COLORS.get(tile, (128, 128, 128))


def generate_preview(map_path: str, width: int = 300, height: int = 300) -> Image.Image:
    """Generate a preview image for a map."""
    map_data = load_map(map_path)
    if map_data is None:
        return None

    map_height, map_width = map_data.shape

    # Calculate tile size
    tile_width = width / map_width
    tile_height = height / map_height

    # Use smaller tile size to maintain aspect ratio
    tile_size = min(tile_width, tile_height)

    # Recalculate actual image dimensions
    actual_width = int(tile_size * map_width)
    actual_height = int(tile_size * map_height)

    # Create image
    img = Image.new('RGB', (actual_width, actual_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw each tile
    for y in range(map_height):
        for x in range(map_width):
            tile = str(map_data.iloc[y, x])
            color = get_tile_color(tile)

            x1 = int(x * tile_size)
            y1 = int(y * tile_size)
            x2 = int((x + 1) * tile_size)
            y2 = int((y + 1) * tile_size)

            draw.rectangle([x1, y1, x2, y2], fill=color)

            # Add subtle grid lines
            draw.rectangle([x1, y1, x2, y2], outline=(50, 50, 50, 100), width=1)

    return img


def extract_metadata(map_path: str) -> dict:
    """Extract metadata from a map file."""
    map_data = load_map(map_path)
    if map_data is None:
        return {}

    map_height, map_width = map_data.shape

    terrain_counts = {}
    player_count = 0

    for y in range(map_height):
        for x in range(map_width):
            tile = str(map_data.iloc[y, x])

            if '_' in tile:
                parts = tile.split('_')
                base_type = parts[0]
                if len(parts) > 1 and parts[1].isdigit():
                    player_num = int(parts[1])
                    player_count = max(player_count, player_num)
                    terrain_name = TERRAIN_DISPLAY_NAMES.get(base_type, 'Unknown')
                    terrain_counts[terrain_name] = terrain_counts.get(terrain_name, 0) + 1
            else:
                terrain_name = TERRAIN_DISPLAY_NAMES.get(tile, 'Unknown')
                terrain_counts[terrain_name] = terrain_counts.get(terrain_name, 0) + 1

    return {
        'width': map_width,
        'height': map_height,
        'player_count': player_count,
        'terrain_counts': terrain_counts
    }


def main():
    """Generate map preview images for tournament maps."""
    # Maps used in tournaments
    tournament_maps = [
        'maps/1v1/beginner.csv',
        'maps/1v1/funnel_point.csv',
        'maps/1v1/center_mountains.csv',
        'maps/1v1/corner_points.csv',
    ]

    # Output directory
    output_dir = project_root / 'docs-site' / 'static' / 'img' / 'maps'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating map previews in {output_dir}")

    for map_path in tournament_maps:
        full_path = project_root / map_path
        if not full_path.exists():
            print(f"Map not found: {full_path}")
            continue

        # Generate preview
        preview = generate_preview(str(full_path), width=400, height=400)
        if preview is None:
            print(f"Failed to generate preview for {map_path}")
            continue

        # Extract metadata
        metadata = extract_metadata(str(full_path))

        # Save image
        map_name = Path(map_path).stem
        output_path = output_dir / f"{map_name}.png"
        preview.save(str(output_path), 'PNG')

        print(f"Generated: {output_path}")
        print(f"  Size: {metadata['width']}x{metadata['height']}")
        print(f"  Players: {metadata['player_count']}")
        print(f"  Terrain: {metadata['terrain_counts']}")

    print("\nDone!")


if __name__ == '__main__':
    main()
