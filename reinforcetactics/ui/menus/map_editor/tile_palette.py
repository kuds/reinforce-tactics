"""Tile palette component for the map editor."""
from typing import Optional, Tuple
import pygame
from reinforcetactics.constants import TILE_COLORS, PLAYER_COLORS, TileType
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font


class TilePalette:
    """Tile palette for selecting tiles to place on the map."""

    # Define available tiles
    TERRAIN_TILES = [
        ('p', 'Grass'),
        ('o', 'Ocean'),
        ('w', 'Water'),
        ('m', 'Mountain'),
        ('f', 'Forest'),
        ('r', 'Road')
    ]

    STRUCTURE_TILES = [
        ('t', 'Tower'),
        ('b', 'Building'),
        ('h', 'Headquarters')
    ]

    def __init__(self, x: int, y: int, width: int, height: int, num_players: int = 2) -> None:
        """
        Initialize the tile palette.

        Args:
            x: X position
            y: Y position
            width: Width of the palette
            height: Height of the palette
            num_players: Number of players (for structure ownership)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.num_players = num_players

        # Selected tile
        self.selected_tile = 'p'  # Default to grass
        self.selected_player = 0  # For structures (0 = neutral)

        # Fonts
        self.title_font = get_font(20)
        self.label_font = get_font(16)
        self.tile_font = get_font(14)

        # Colors
        self.bg_color = (40, 40, 50)
        self.border_color = (100, 100, 120)
        self.selected_color = (255, 200, 50)
        self.text_color = (255, 255, 255)

        # Layout
        self.tile_size = 30
        self.padding = 8
        self.section_spacing = 12

        # Clickable rectangles
        self.tile_rects = {}  # tile_code -> rect
        self.player_rects = {}  # player_num -> rect

    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """
        Handle mouse click on the palette.

        Args:
            mouse_pos: Mouse position (x, y)

        Returns:
            True if a tile was selected, False otherwise
        """
        # Check terrain tiles
        for tile_code, rect in self.tile_rects.items():
            if rect.collidepoint(mouse_pos):
                self.selected_tile = tile_code
                return True

        # Check player selection for structures
        for player_num, rect in self.player_rects.items():
            if rect.collidepoint(mouse_pos):
                self.selected_player = player_num
                return True

        return False

    def get_selected_tile(self) -> str:
        """
        Get the currently selected tile code.

        Returns:
            Tile code (e.g., 'p', 'h_1', 'b_2', 't', 'b', 'h')
        """
        # If it's a structure, append player number (unless neutral)
        if self.selected_tile in ('t', 'b', 'h'):
            # Player 0 means neutral (no suffix)
            if self.selected_player == 0:
                return self.selected_tile
            else:
                return f"{self.selected_tile}_{self.selected_player}"
        return self.selected_tile

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the tile palette.

        Args:
            screen: Pygame surface to draw on
        """
        lang = get_language()

        # Draw background
        bg_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.bg_color, bg_rect)
        pygame.draw.rect(screen, self.border_color, bg_rect, width=2)

        # Draw title
        title_text = lang.get('map_editor.tile_palette.title', 'Tile Palette')
        title_surface = self.title_font.render(title_text, True, self.text_color)
        title_rect = title_surface.get_rect(centerx=self.x + self.width // 2, y=self.y + 10)
        screen.blit(title_surface, title_rect)

        current_y = self.y + 35

        # Draw terrain section
        current_y = self._draw_terrain_section(screen, current_y, lang)
        current_y += self.section_spacing

        # Draw structures section
        current_y = self._draw_structures_section(screen, current_y, lang)
        current_y += self.section_spacing

        # Draw player selection for structures
        if self.selected_tile in ('t', 'b', 'h'):
            self._draw_player_selection(screen, current_y, lang)

    def _draw_terrain_section(self, screen: pygame.Surface, start_y: int, lang) -> int:
        """
        Draw terrain tiles section.

        Args:
            screen: Pygame surface
            start_y: Starting Y position
            lang: Language instance

        Returns:
            Next Y position
        """
        # Section label
        label_text = lang.get('map_editor.tile_palette.terrain', 'Terrain')
        label_surface = self.label_font.render(label_text, True, self.text_color)
        label_rect = label_surface.get_rect(x=self.x + self.padding, y=start_y)
        screen.blit(label_surface, label_rect)

        current_y = start_y + 22
        current_x = self.x + self.padding

        # Draw each terrain tile
        for tile_code, tile_name in self.TERRAIN_TILES:
            is_selected = self.selected_tile == tile_code

            # Tile rectangle
            tile_rect = pygame.Rect(current_x, current_y, self.tile_size, self.tile_size)

            # Draw tile with color
            tile_color = TILE_COLORS.get(tile_code, (100, 100, 100))
            pygame.draw.rect(screen, tile_color, tile_rect)

            # Draw border
            border_color = self.selected_color if is_selected else (80, 80, 80)
            border_width = 3 if is_selected else 1
            pygame.draw.rect(screen, border_color, tile_rect, width=border_width)

            # Draw tile name
            name_surface = self.tile_font.render(tile_name, True, self.text_color)
            name_rect = name_surface.get_rect(x=current_x + self.tile_size + 5, centery=current_y + self.tile_size // 2)
            screen.blit(name_surface, name_rect)

            # Store rectangle for click detection
            full_rect = pygame.Rect(current_x, current_y, self.width - 2 * self.padding, self.tile_size)
            self.tile_rects[tile_code] = full_rect

            current_y += self.tile_size + 5

        return current_y

    def _draw_structures_section(self, screen: pygame.Surface, start_y: int, lang) -> int:
        """
        Draw structure tiles section.

        Args:
            screen: Pygame surface
            start_y: Starting Y position
            lang: Language instance

        Returns:
            Next Y position
        """
        # Section label
        label_text = lang.get('map_editor.tile_palette.structures', 'Structures')
        label_surface = self.label_font.render(label_text, True, self.text_color)
        label_rect = label_surface.get_rect(x=self.x + self.padding, y=start_y)
        screen.blit(label_surface, label_rect)

        current_y = start_y + 22
        current_x = self.x + self.padding

        # Draw each structure tile
        for tile_code, tile_name in self.STRUCTURE_TILES:
            is_selected = self.selected_tile == tile_code

            # Tile rectangle
            tile_rect = pygame.Rect(current_x, current_y, self.tile_size, self.tile_size)

            # Draw tile with color (use player color if selected and player-owned)
            if is_selected and self.selected_player > 0:
                # Show with player color
                base_color = TILE_COLORS.get(tile_code, (100, 100, 100))
                player_color = PLAYER_COLORS.get(self.selected_player, (255, 255, 255))
                # Blend colors
                tile_color = tuple((base_color[i] + player_color[i]) // 2 for i in range(3))
            else:
                # Show neutral or unselected color
                tile_color = TILE_COLORS.get(tile_code, (100, 100, 100))

            pygame.draw.rect(screen, tile_color, tile_rect)

            # Draw border
            border_color = self.selected_color if is_selected else (80, 80, 80)
            border_width = 3 if is_selected else 1
            pygame.draw.rect(screen, border_color, tile_rect, width=border_width)

            # Draw tile name
            name_surface = self.tile_font.render(tile_name, True, self.text_color)
            name_rect = name_surface.get_rect(x=current_x + self.tile_size + 5, centery=current_y + self.tile_size // 2)
            screen.blit(name_surface, name_rect)

            # Store rectangle for click detection
            full_rect = pygame.Rect(current_x, current_y, self.width - 2 * self.padding, self.tile_size)
            self.tile_rects[tile_code] = full_rect

            current_y += self.tile_size + 5

        return current_y

    def _draw_player_selection(self, screen: pygame.Surface, start_y: int, lang) -> None:
        """
        Draw player selection buttons for structures.

        Args:
            screen: Pygame surface
            start_y: Starting Y position
            lang: Language instance
        """
        # Label
        label_text = "Owner:"
        label_surface = self.label_font.render(label_text, True, self.text_color)
        label_rect = label_surface.get_rect(x=self.x + self.padding, y=start_y)
        screen.blit(label_surface, label_rect)

        current_y = start_y + 22
        current_x = self.x + self.padding
        button_size = 28

        # Draw neutral button (player 0)
        is_selected = self.selected_player == 0
        button_rect = pygame.Rect(current_x, current_y, button_size, button_size)

        # Neutral color (gray)
        neutral_color = (150, 150, 150)
        pygame.draw.rect(screen, neutral_color, button_rect)

        # Draw border
        border_color = self.selected_color if is_selected else (80, 80, 80)
        border_width = 3 if is_selected else 1
        pygame.draw.rect(screen, border_color, button_rect, width=border_width)

        # Draw "N" for Neutral
        neutral_label = lang.get('map_editor.tile_palette.neutral', 'Neutral')
        neutral_text = neutral_label[0] if neutral_label else 'N'  # First letter, fallback to 'N'
        num_surface = self.tile_font.render(neutral_text, True, (0, 0, 0))
        num_rect = num_surface.get_rect(center=button_rect.center)
        screen.blit(num_surface, num_rect)

        # Store rectangle for click detection
        self.player_rects[0] = button_rect

        current_x += button_size + 5

        # Draw player buttons
        for player_num in range(1, self.num_players + 1):
            is_selected = self.selected_player == player_num

            # Button rectangle
            button_rect = pygame.Rect(current_x, current_y, button_size, button_size)

            # Draw button with player color
            player_color = PLAYER_COLORS.get(player_num, (200, 200, 200))
            pygame.draw.rect(screen, player_color, button_rect)

            # Draw border
            border_color = self.selected_color if is_selected else (80, 80, 80)
            border_width = 3 if is_selected else 1
            pygame.draw.rect(screen, border_color, button_rect, width=border_width)

            # Draw player number
            num_surface = self.tile_font.render(str(player_num), True, (0, 0, 0))
            num_rect = num_surface.get_rect(center=button_rect.center)
            screen.blit(num_surface, num_rect)

            # Store rectangle for click detection
            self.player_rects[player_num] = button_rect

            current_x += button_size + 5
