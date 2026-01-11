"""Map editing canvas component."""
from typing import Tuple
import pygame
import pandas as pd
from reinforcetactics.constants import TILE_COLORS, PLAYER_COLORS
from reinforcetactics.utils.fonts import get_font


class EditorCanvas:
    """Canvas for editing maps with paint/erase tools."""

    def __init__(self, x: int, y: int, width: int, height: int, map_data: pd.DataFrame) -> None:
        """
        Initialize the editor canvas.

        Args:
            x: X position
            y: Y position
            width: Width of the canvas
            height: Height of the canvas
            map_data: Pandas DataFrame containing the map data
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.map_data = map_data

        # View settings
        self.tile_size = 24
        self.grid_enabled = True
        self.offset_x = 0
        self.offset_y = 0

        # Mouse state
        self.hover_tile = None  # (x, y) of hovered tile
        self.is_painting = False

        # Colors
        self.bg_color = (30, 30, 40)
        self.grid_color = (60, 60, 70)
        self.highlight_color = (255, 255, 0)
        self.hover_color = (255, 255, 255, 100)

        # Fonts
        self.coord_font = get_font(18)

        # Calculate visible tiles
        self._update_visible_area()

    def _update_visible_area(self) -> None:
        """Update the visible area based on canvas size and tile size."""
        self.visible_cols = self.width // self.tile_size
        self.visible_rows = self.height // self.tile_size

    def handle_mouse_move(self, mouse_pos: Tuple[int, int]) -> bool:
        """
        Handle mouse movement over the canvas.

        Args:
            mouse_pos: Mouse position (x, y)

        Returns:
            True if mouse is over canvas, False otherwise
        """
        if not self._is_over_canvas(mouse_pos):
            self.hover_tile = None
            return False

        # Calculate tile coordinates
        tile_x = (mouse_pos[0] - self.x + self.offset_x) // self.tile_size
        tile_y = (mouse_pos[1] - self.y + self.offset_y) // self.tile_size

        # Check bounds
        map_height, map_width = self.map_data.shape
        if 0 <= tile_x < map_width and 0 <= tile_y < map_height:
            self.hover_tile = (tile_x, tile_y)
        else:
            self.hover_tile = None

        return True

    def handle_mouse_click(self, mouse_pos: Tuple[int, int], tile_code: str) -> bool:
        """
        Handle mouse click to paint a tile.

        Args:
            mouse_pos: Mouse position (x, y)
            tile_code: Tile code to place

        Returns:
            True if a tile was painted, False otherwise
        """
        if not self._is_over_canvas(mouse_pos):
            return False

        if self.hover_tile:
            tile_x, tile_y = self.hover_tile
            self.map_data.iloc[tile_y, tile_x] = tile_code
            self.is_painting = True
            return True

        return False

    def handle_mouse_release(self) -> None:
        """Handle mouse button release."""
        self.is_painting = False

    def handle_scroll(self, dx: int, dy: int) -> None:
        """
        Handle scrolling/panning.

        Args:
            dx: Change in X offset
            dy: Change in Y offset
        """
        map_height, map_width = self.map_data.shape
        max_offset_x = max(0, map_width * self.tile_size - self.width)
        max_offset_y = max(0, map_height * self.tile_size - self.height)

        self.offset_x = max(0, min(max_offset_x, self.offset_x + dx))
        self.offset_y = max(0, min(max_offset_y, self.offset_y + dy))

    def toggle_grid(self) -> None:
        """Toggle grid visibility."""
        self.grid_enabled = not self.grid_enabled

    def zoom_in(self) -> None:
        """Zoom in (increase tile size)."""
        if self.tile_size < 64:
            self.tile_size += 4
            self._update_visible_area()

    def zoom_out(self) -> None:
        """Zoom out (decrease tile size)."""
        if self.tile_size > 16:
            self.tile_size -= 4
            self._update_visible_area()

    def _is_over_canvas(self, mouse_pos: Tuple[int, int]) -> bool:
        """
        Check if mouse is over the canvas.

        Args:
            mouse_pos: Mouse position (x, y)

        Returns:
            True if mouse is over canvas, False otherwise
        """
        return (self.x <= mouse_pos[0] < self.x + self.width and
                self.y <= mouse_pos[1] < self.y + self.height)

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the canvas.

        Args:
            screen: Pygame surface to draw on
        """
        # Draw background
        bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, bg_rect)

        # Draw map tiles
        self._draw_tiles(screen)

        # Draw grid if enabled
        if self.grid_enabled:
            self._draw_grid(screen)

        # Draw hover highlight
        if self.hover_tile:
            self._draw_hover_highlight(screen)

        # Draw coordinates
        if self.hover_tile:
            self._draw_coordinates(screen)

        # Draw border
        pygame.draw.rect(screen, (100, 100, 120), bg_rect, width=2)

    def _draw_tiles(self, screen: pygame.Surface) -> None:
        """
        Draw map tiles.

        Args:
            screen: Pygame surface to draw on
        """
        map_height, map_width = self.map_data.shape

        # Calculate visible range
        start_col = self.offset_x // self.tile_size
        start_row = self.offset_y // self.tile_size
        end_col = min(map_width, start_col + self.visible_cols + 1)
        end_row = min(map_height, start_row + self.visible_rows + 1)

        # Draw each visible tile
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                tile_code = str(self.map_data.iloc[row, col])

                # Calculate screen position
                screen_x = self.x + col * self.tile_size - self.offset_x
                screen_y = self.y + row * self.tile_size - self.offset_y

                # Get base tile code (without player number)
                base_code = tile_code.split('_')[0] if '_' in tile_code else tile_code

                # Get tile color
                tile_color = TILE_COLORS.get(base_code, (50, 50, 50))

                # For structures with ownership, blend with player color
                if '_' in tile_code:
                    parts = tile_code.split('_')
                    if len(parts) == 2 and parts[1].isdigit():
                        player_num = int(parts[1])
                        player_color = PLAYER_COLORS.get(player_num, (255, 255, 255))
                        # Blend colors
                        tile_color = tuple((tile_color[i] + player_color[i]) // 2 for i in range(3))

                # Draw tile
                tile_rect = pygame.Rect(screen_x, screen_y, self.tile_size, self.tile_size)
                pygame.draw.rect(screen, tile_color, tile_rect)

    def _draw_grid(self, screen: pygame.Surface) -> None:
        """
        Draw grid lines.

        Args:
            screen: Pygame surface to draw on
        """
        map_height, map_width = self.map_data.shape

        # Calculate visible range
        start_col = self.offset_x // self.tile_size
        start_row = self.offset_y // self.tile_size
        end_col = min(map_width, start_col + self.visible_cols + 1)
        end_row = min(map_height, start_row + self.visible_rows + 1)

        # Draw vertical lines
        for col in range(start_col, end_col + 1):
            screen_x = self.x + col * self.tile_size - self.offset_x
            pygame.draw.line(
                screen,
                self.grid_color,
                (screen_x, self.y),
                (screen_x, self.y + self.height),
                1
            )

        # Draw horizontal lines
        for row in range(start_row, end_row + 1):
            screen_y = self.y + row * self.tile_size - self.offset_y
            pygame.draw.line(
                screen,
                self.grid_color,
                (self.x, screen_y),
                (self.x + self.width, screen_y),
                1
            )

    def _draw_hover_highlight(self, screen: pygame.Surface) -> None:
        """
        Draw highlight over hovered tile.

        Args:
            screen: Pygame surface to draw on
        """
        if not self.hover_tile:
            return

        tile_x, tile_y = self.hover_tile
        screen_x = self.x + tile_x * self.tile_size - self.offset_x
        screen_y = self.y + tile_y * self.tile_size - self.offset_y

        # Create semi-transparent surface
        highlight_surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        highlight_surface.fill(self.hover_color)
        screen.blit(highlight_surface, (screen_x, screen_y))

        # Draw border
        tile_rect = pygame.Rect(screen_x, screen_y, self.tile_size, self.tile_size)
        pygame.draw.rect(screen, self.highlight_color, tile_rect, width=2)

    def _draw_coordinates(self, screen: pygame.Surface) -> None:
        """
        Draw coordinates of hovered tile.

        Args:
            screen: Pygame surface to draw on
        """
        if not self.hover_tile:
            return

        tile_x, tile_y = self.hover_tile
        coord_text = f"X: {tile_x}, Y: {tile_y}"
        coord_surface = self.coord_font.render(coord_text, True, (255, 255, 255))

        # Draw with background
        coord_rect = coord_surface.get_rect(x=self.x + 10, y=self.y + self.height - 30)
        bg_rect = coord_rect.inflate(10, 5)
        pygame.draw.rect(screen, (40, 40, 50), bg_rect)
        pygame.draw.rect(screen, (100, 100, 120), bg_rect, width=1)
        screen.blit(coord_surface, coord_rect)
