"""
Pygame rendering for the strategy game.
"""
import pygame
import numpy as np
from reinforcetactics.constants import (
    TILE_SIZE, TILE_COLORS, TILE_TYPES, TILE_IMAGES,
    PLAYER_COLORS, UNIT_COLORS, UNIT_DATA
)


class Renderer:
    """Handles all Pygame rendering."""

    def __init__(self, game_state):
        """
        Initialize the renderer.

        Args:
            game_state: GameState instance to render
        """
        self.game_state = game_state

        # Initialize Pygame if not already initialized
        if not pygame.get_init():
            pygame.init()

        # Setup display
        screen_width = game_state.grid.width * TILE_SIZE
        screen_height = game_state.grid.height * TILE_SIZE
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("2D Strategy Game")

        # Initialize clipboard support
        try:
            pygame.scrap.init()
        except pygame.error:
            # Clipboard not available on this platform
            pass

        # Load tile images
        self.tile_images = self._load_tile_images()

        # UI elements
        self._setup_ui_elements()

    def _load_tile_images(self):
        """Load tile images from files."""
        tile_images = {}
        for tile_type, filename in TILE_IMAGES.items():
            try:
                image = pygame.image.load(filename)
                tile_images[tile_type] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
            except (pygame.error, FileNotFoundError):
                tile_images[tile_type] = None
        return tile_images

    def _setup_ui_elements(self):
        """Setup UI elements like buttons."""
        screen_width = self.game_state.grid.width * TILE_SIZE

        self.end_turn_button = pygame.Rect(screen_width - 150, 10, 140, 40)
        self.resign_button = pygame.Rect(screen_width - 150, 60, 140, 40)

    def render(self):
        """Render the entire game state."""
        self.screen.fill((0, 0, 0))

        # Draw grid
        self._draw_grid()

        # Draw units
        self._draw_units()

        # Draw UI
        self._draw_ui()

        # Note: pygame.display.flip() is called by the main game loop

    def _draw_grid(self):
        """Draw the tile grid."""
        for y in range(self.game_state.grid.height):
            for x in range(self.game_state.grid.width):
                tile = self.game_state.grid.tiles[y][x]
                self._draw_tile(tile)

    def _draw_tile(self, tile):
        """Draw a single tile with improved visuals."""
        rect = pygame.Rect(tile.x * TILE_SIZE, tile.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

        tile_type_name = TILE_TYPES.get(tile.type, 'OCEAN')

        # Draw tile image or color
        if self.tile_images.get(tile_type_name):
            self.screen.blit(self.tile_images[tile_type_name], rect)
        else:
            color = tile.get_color()
            pygame.draw.rect(self.screen, color, rect)

            # Add visual variety to tiles
            if tile.type == 'p':  # Grass - checkerboard pattern
                if (tile.x + tile.y) % 2 == 0:
                    lighter = tuple(min(c + 15, 255) for c in color)
                    pygame.draw.rect(self.screen, lighter, rect)

            elif tile.type == 'o':  # Ocean - checkerboard pattern
                if (tile.x + tile.y) % 2 == 0:
                    lighter = tuple(min(c + 15, 255) for c in color)
                    pygame.draw.rect(self.screen, lighter, rect)

            elif tile.type == 'w':  # Water - darker edges
                darker = tuple(max(c - 30, 0) for c in color)
                pygame.draw.line(self.screen, darker, rect.topleft, rect.topright, 2)
                pygame.draw.line(self.screen, darker, rect.topleft, rect.bottomleft, 2)

            elif tile.type == 'm':  # Mountain - lighter top
                lighter = tuple(min(c + 40, 255) for c in color)
                top_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height // 2)
                pygame.draw.rect(self.screen, lighter, top_rect)

            elif tile.type == 'f':  # Forest - random dots for texture
                import random
                random.seed(tile.x * 1000 + tile.y)  # Deterministic randomness
                for _ in range(3):
                    x = rect.x + random.randint(5, rect.width - 5)
                    y = rect.y + random.randint(5, rect.height - 5)
                    darker = tuple(max(c - 20, 0) for c in color)
                    pygame.draw.circle(self.screen, darker, (x, y), 2)

            elif tile.type == 'r':  # Road - center stripe
                stripe_color = tuple(min(c + 30, 255) for c in color)
                center_y = rect.centery
                pygame.draw.line(self.screen, stripe_color,
                            (rect.left, center_y), (rect.right, center_y), 2)

            elif tile.type in ['h', 'b', 't']:  # Structures - border highlight
                if tile.player:
                    player_color = PLAYER_COLORS.get(tile.player, (255, 255, 255))
                    pygame.draw.rect(self.screen, player_color, rect, 3)

            # Tile border
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Draw structure health bar (only for capturable structures)
        if tile.is_capturable() and tile.health is not None:
            self._draw_structure_health_bar(tile)

    def _draw_structure_health_bar(self, tile):
        """Draw health bar for structures."""
        bar_width = TILE_SIZE - 6
        bar_height = 4
        bar_x = tile.x * TILE_SIZE + 3
        bar_y = tile.y * TILE_SIZE + 3

        # Background
        pygame.draw.rect(self.screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

        # Foreground
        health_percentage = tile.health / tile.max_health
        current_bar_width = int(bar_width * health_percentage)

        if tile.type == 'h':
            health_color = (255, 200, 0)
        elif tile.type == 'b':
            health_color = (0, 200, 200)
        else:
            health_color = (200, 200, 200)

        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 1)

    def _draw_units(self):
        """Draw all units."""
        for unit in self.game_state.units:
            self._draw_unit(unit)

    def _draw_unit(self, unit):
        """Draw a single unit."""
        font = pygame.font.Font(None, 40)
        color = UNIT_COLORS[unit.type]

        # Gray out if can't act
        if not unit.can_move and not unit.can_attack:
            color = tuple(c // 2 for c in color)

        # Purple tint for paralyzed
        if unit.is_paralyzed():
            color = tuple(min(int(c * 0.6 + 128 * 0.4), 255) for c in color)

        # Draw unit letter with outline
        text = font.render(unit.type, True, color)
        text_rect = text.get_rect(center=(
            unit.x * TILE_SIZE + TILE_SIZE // 2,
            unit.y * TILE_SIZE + TILE_SIZE // 2
        ))

        # Black outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            outline_rect = text_rect.copy()
            outline_rect.x += dx
            outline_rect.y += dy
            outline_text = font.render(unit.type, True, (0, 0, 0))
            self.screen.blit(outline_text, outline_rect)

        self.screen.blit(text, text_rect)

        # Draw paralysis indicator
        if unit.is_paralyzed():
            tile_rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.screen, (148, 0, 211), tile_rect, 3)

            indicator_font = pygame.font.Font(None, 24)
            indicator_text = indicator_font.render(f"P:{unit.paralyzed_turns}", True, (148, 0, 211))
            indicator_rect = indicator_text.get_rect(topright=(
                unit.x * TILE_SIZE + TILE_SIZE - 2,
                unit.y * TILE_SIZE + 2
            ))
            bg_rect = indicator_rect.inflate(4, 2)
            pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
            pygame.draw.rect(self.screen, (148, 0, 211), bg_rect, 1)
            self.screen.blit(indicator_text, indicator_rect)

        # Draw health bar
        self._draw_unit_health_bar(unit)

    def _draw_unit_health_bar(self, unit):
        """Draw health bar for a unit."""
        bar_width = TILE_SIZE - 6
        bar_height = 5
        bar_x = unit.x * TILE_SIZE + 3
        bar_y = unit.y * TILE_SIZE + TILE_SIZE - bar_height - 3

        # Background
        pygame.draw.rect(self.screen, (100, 0, 0), (bar_x, bar_y, bar_width, bar_height))

        # Foreground
        health_percentage = unit.health / unit.max_health
        current_bar_width = int(bar_width * health_percentage)

        if health_percentage > 0.5:
            health_color = (0, 200, 0)
        elif health_percentage > 0.25:
            health_color = (255, 165, 0)
        else:
            health_color = (255, 0, 0)

        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 1)

    def _draw_ui(self):
        """Draw UI elements."""
        font = pygame.font.Font(None, 28)
        player_color = PLAYER_COLORS.get(self.game_state.current_player, (255, 255, 255))

        # Draw player info and gold
        gold_text = f"Player {self.game_state.current_player} Gold: ${self.game_state.player_gold[self.game_state.current_player]}"
        text_surface = font.render(gold_text, True, (255, 215, 0))
        text_rect = text_surface.get_rect(topleft=(10, 10))
        bg_rect = text_rect.inflate(10, 5)

        pygame.draw.rect(self.screen, player_color, bg_rect)
        pygame.draw.rect(self.screen, (255, 215, 0), bg_rect, 2)
        self.screen.blit(text_surface, text_rect)

        # Draw End Turn button
        mouse_pos = pygame.mouse.get_pos()
        button_color = (100, 150, 100) if self.end_turn_button.collidepoint(mouse_pos) else (70, 120, 70)

        pygame.draw.rect(self.screen, button_color, self.end_turn_button)
        pygame.draw.rect(self.screen, (255, 255, 255), self.end_turn_button, 2)

        button_font = pygame.font.Font(None, 32)
        button_text = button_font.render("End Turn", True, (255, 255, 255))
        button_text_rect = button_text.get_rect(center=self.end_turn_button.center)
        self.screen.blit(button_text, button_text_rect)

        # Draw Resign button
        resign_color = (150, 70, 70) if self.resign_button.collidepoint(mouse_pos) else (120, 50, 50)

        pygame.draw.rect(self.screen, resign_color, self.resign_button)
        pygame.draw.rect(self.screen, (200, 100, 100), self.resign_button, 2)

        resign_text = button_font.render("Resign", True, (255, 255, 255))
        resign_text_rect = resign_text.get_rect(center=self.resign_button.center)
        self.screen.blit(resign_text, resign_text_rect)

    def draw_movement_overlay(self, unit):
        """Draw movement range overlay for a unit."""
        if not unit.can_move:
            return

        movement_positions = unit.get_reachable_positions(
            self.game_state.grid.width,
            self.game_state.grid.height,
            lambda x, y: self.game_state.mechanics.can_move_to_position(
                x, y, self.game_state.grid, self.game_state.units
            )
        )

        if movement_positions:
            overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
            overlay.set_alpha(100)
            overlay.fill((255, 255, 255))

            for x, y in movement_positions:
                self.screen.blit(overlay, (x * TILE_SIZE, y * TILE_SIZE))
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, (255, 255, 0), rect, 1)

    def draw_target_overlay(self, valid_targets):
        """
        Draw target selection overlay highlighting valid target positions.

        Args:
            valid_targets: List of unit objects that are valid targets
        """
        for target in valid_targets:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
            overlay.set_alpha(120)
            overlay.fill((255, 100, 100))  # Red tint for targets
            self.screen.blit(overlay, (target.x * TILE_SIZE, target.y * TILE_SIZE))

            # Draw border around target
            rect = pygame.Rect(target.x * TILE_SIZE, target.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 0, 0), rect, 3)

    def get_rgb_array(self):
        """Get the current screen as RGB array."""
        # Convert pygame surface to numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)),
            axes=(1, 0, 2)
        )

    def close(self):
        """Close the renderer."""
        pygame.quit()
