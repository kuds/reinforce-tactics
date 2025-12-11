"""Map preview generator for creating thumbnails of maps."""
import os
import re
from typing import Dict, Tuple, Optional
import pygame
import pandas as pd

from reinforcetactics.constants import TILE_COLORS, PLAYER_COLORS
from reinforcetactics.utils.file_io import FileIO


# Mapping of terrain codes to display names
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


class MapPreviewGenerator:
    """Generates visual previews and metadata for map files."""

    def __init__(self):
        """Initialize the map preview generator."""
        self._cache: Dict[str, Tuple[pygame.Surface, dict]] = {}

    def generate_preview(self, map_path: str, width: int = 150, height: int = 150,
                        force_regenerate: bool = False) -> Tuple[Optional[pygame.Surface], dict]:
        """
        Generate a preview thumbnail and metadata for a map.

        Args:
            map_path: Path to the map CSV file
            width: Target width for the preview
            height: Target height for the preview
            force_regenerate: If True, regenerate even if cached

        Returns:
            Tuple of (pygame.Surface with preview, dict with metadata)
            Surface is None if generation fails
        """
        # Check cache
        cache_key = f"{map_path}_{width}_{height}"
        if not force_regenerate and cache_key in self._cache:
            return self._cache[cache_key]

        # Handle random map case
        if map_path == "random":
            metadata = {
                'name': "Random Map",
                'width': 0,
                'height': 0,
                'player_count': 0,
                'terrain_breakdown': {},
                'difficulty': 'N/A'
            }
            # Create a simple preview for random map
            preview = pygame.Surface((width, height))
            preview.fill((50, 50, 50))

            # Draw question mark pattern
            font = pygame.font.Font(None, int(height * 0.6))
            text = font.render("?", True, (200, 200, 200))
            text_rect = text.get_rect(center=(width // 2, height // 2))
            preview.blit(text, text_rect)

            self._cache[cache_key] = (preview, metadata)
            return preview, metadata

        # Load map data
        map_data = FileIO.load_map(map_path)
        if map_data is None:
            return None, {}

        # Extract metadata
        metadata = self._extract_metadata(map_data, map_path)

        # Generate preview surface
        preview = self._render_preview(map_data, width, height)

        # Cache result
        self._cache[cache_key] = (preview, metadata)

        return preview, metadata

    def _extract_metadata(self, map_data: pd.DataFrame, map_path: str) -> dict:
        """
        Extract metadata from map data.

        Args:
            map_data: DataFrame containing map data
            map_path: Path to the map file (for name extraction)

        Returns:
            Dictionary with metadata
        """
        map_height, map_width = map_data.shape

        # Count terrain types and player structures
        terrain_counts = {}
        player_count = 0

        for y in range(map_height):
            for x in range(map_width):
                tile = str(map_data.iloc[y, x])

                # Handle player structures (e.g., "h_1", "b_2")
                if '_' in tile:
                    parts = tile.split('_')
                    base_type = parts[0]
                    if len(parts) > 1 and parts[1].isdigit():
                        player_num = int(parts[1])
                        player_count = max(player_count, player_num)

                        # Count as the base type with player prefix
                        terrain_name = TERRAIN_DISPLAY_NAMES.get(base_type, 'Unknown')
                        terrain_counts[terrain_name] = terrain_counts.get(terrain_name, 0) + 1
                else:
                    # Map terrain codes to readable names
                    terrain_name = TERRAIN_DISPLAY_NAMES.get(tile, 'Unknown')
                    terrain_counts[terrain_name] = terrain_counts.get(terrain_name, 0) + 1

        # Calculate percentages
        total_tiles = map_width * map_height
        terrain_breakdown = {}
        for terrain, count in terrain_counts.items():
            percentage = (count / total_tiles) * 100
            terrain_breakdown[terrain] = {
                'count': count,
                'percentage': percentage
            }

        # Calculate difficulty based on map complexity
        difficulty = self._calculate_difficulty(map_width, map_height, terrain_breakdown)

        # Format friendly name from filename
        filename = os.path.basename(map_path)
        friendly_name = self._format_display_name(filename)

        return {
            'name': friendly_name,
            'width': map_width,
            'height': map_height,
            'player_count': player_count,
            'terrain_breakdown': terrain_breakdown,
            'difficulty': difficulty
        }

    def _calculate_difficulty(self, width: int, height: int,
                             terrain_breakdown: dict) -> str:
        """
        Calculate difficulty indicator based on map complexity.

        Args:
            width: Map width
            height: Map height
            terrain_breakdown: Terrain composition data

        Returns:
            Difficulty string: "Beginner", "Easy", "Medium", "Hard", "Expert"
        """
        # Base difficulty on map size
        total_tiles = width * height

        if total_tiles < 100:  # < 10x10
            return "Beginner"
        elif total_tiles < 150:  # < ~12x12
            return "Easy"
        elif total_tiles < 800:  # < ~28x28
            return "Medium"
        elif total_tiles < 1024:  # < 32x32
            return "Hard"
        else:
            return "Expert"

    def _format_display_name(self, filename: str) -> str:
        """
        Format a friendly display name from a filename.

        Args:
            filename: Original filename (e.g., "6x6_beginner.csv")

        Returns:
            Formatted name (e.g., "6×6 Beginner")
        """
        # Remove .csv extension
        name = os.path.splitext(filename)[0]

        # Replace underscores with spaces
        name = name.replace('_', ' ')

        # Replace lowercase x with multiplication symbol in dimensions
        # Match patterns like "6x6" or "10x10"
        name = re.sub(r'(\d+)x(\d+)', r'\1×\2', name)

        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())

        return name

    def _render_preview(self, map_data: pd.DataFrame, width: int, height: int) -> pygame.Surface:
        """
        Render a visual preview of the map.

        Args:
            map_data: DataFrame containing map data
            width: Target width for preview
            height: Target height for preview

        Returns:
            pygame.Surface with rendered preview
        """
        map_height, map_width = map_data.shape

        # Create surface for preview
        preview = pygame.Surface((width, height))
        preview.fill((0, 0, 0))

        # Calculate tile size to fit preview
        tile_width = width / map_width
        tile_height = height / map_height

        # Render each tile
        for y in range(map_height):
            for x in range(map_width):
                tile = str(map_data.iloc[y, x])

                # Get color for tile
                color = self._get_tile_color(tile)

                # Draw rectangle for this tile
                rect = pygame.Rect(
                    int(x * tile_width),
                    int(y * tile_height),
                    int(tile_width) + 1,  # +1 to avoid gaps
                    int(tile_height) + 1
                )
                pygame.draw.rect(preview, color, rect)

        return preview

    def _get_tile_color(self, tile: str) -> Tuple[int, int, int]:
        """
        Get the color for a tile type.

        Args:
            tile: Tile code (e.g., "p", "h_1", "b_2")

        Returns:
            RGB color tuple
        """
        # Handle player structures (e.g., "h_1", "b_2")
        if '_' in tile:
            parts = tile.split('_')
            base_type = parts[0]

            if len(parts) > 1 and parts[1].isdigit():
                player_num = int(parts[1])
                # Use player color for player structures
                return PLAYER_COLORS.get(player_num, TILE_COLORS.get(base_type, (128, 128, 128)))

        # Return color from TILE_COLORS
        return TILE_COLORS.get(tile, (128, 128, 128))

    def clear_cache(self):
        """Clear the preview cache."""
        self._cache.clear()
