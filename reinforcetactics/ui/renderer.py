"""
Pygame rendering for the strategy game.
"""
import os
import time
import pygame
import numpy as np
from reinforcetactics.constants import (
    TILE_SIZE, TILE_TYPES, TILE_IMAGES,
    PLAYER_COLORS, UNIT_COLORS, UNIT_DATA
)
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.settings import get_settings
from reinforcetactics.ui.sprite_animator import SpriteAnimator


class Renderer:
    """Handles all Pygame rendering."""

    def __init__(self, game_state, replay_mode=False):
        """
        Initialize the renderer.

        Args:
            game_state: GameState instance to render
            replay_mode: If True, skip rendering gameplay controls (End Turn, Resign)
        """
        self.game_state = game_state
        self.replay_mode = replay_mode

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

        # Load settings
        self.settings = get_settings()

        # Load tile images
        self.tile_images = self._load_tile_images()

        # Load unit images
        self.unit_images = self._load_unit_images()

        # Initialize sprite animator for animations
        self.animator = self._init_animator()

        # Animation timing
        self.last_frame_time = time.time()
        self.delta_time = 0.0

        # UI elements
        self._setup_ui_elements()

    def _load_tile_images(self):
        """Load tile images from files."""
        tile_images = {}

        # Check if tile sprites are enabled and path is set
        use_tile_sprites = self.settings.get('graphics.use_tile_sprites', False)
        tile_sprites_path = self.settings.get('graphics.tile_sprites_path', '')

        for tile_type, filename in TILE_IMAGES.items():
            try:
                # Build full path if sprites are enabled and path is configured
                if use_tile_sprites and tile_sprites_path:
                    full_path = os.path.join(tile_sprites_path, filename)
                else:
                    full_path = filename

                image = pygame.image.load(full_path)
                tile_images[tile_type] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
            except (pygame.error, FileNotFoundError):
                tile_images[tile_type] = None
        return tile_images

    def _load_unit_images(self):
        """Load unit images from configured sprites path."""
        unit_images = {}

        # Get the configured unit sprites path
        unit_sprites_path = self.settings.get('graphics.unit_sprites_path', '')
        if not unit_sprites_path:
            return unit_images

        # Load sprite for each unit type
        for unit_type, unit_data in UNIT_DATA.items():
            static_path = unit_data.get('static_path', '')
            if static_path:
                try:
                    full_path = os.path.join(unit_sprites_path, static_path)
                    image = pygame.image.load(full_path)
                    # Scale to fit within tile, leaving room for health bar
                    sprite_size = TILE_SIZE - 4  # Slightly smaller than tile
                    unit_images[unit_type] = pygame.transform.scale(image, (sprite_size, sprite_size))
                except (pygame.error, FileNotFoundError):
                    unit_images[unit_type] = None

        return unit_images

    def _init_animator(self):
        """Initialize the sprite animator for unit animations."""
        # Try animation sprites path first, fall back to unit sprites path
        animation_path = self.settings.get('graphics.animation_sprites_path', '')
        if not animation_path:
            animation_path = self.settings.get('graphics.unit_sprites_path', '')

        if not animation_path:
            return None

        return SpriteAnimator(animation_path)

    def reload_sprites(self):
        """Reload sprites after settings change."""
        self.tile_images = self._load_tile_images()
        self.unit_images = self._load_unit_images()
        self.animator = self._init_animator()

    def _setup_ui_elements(self):
        """Setup UI elements like buttons."""
        screen_width = self.game_state.grid.width * TILE_SIZE

        self.end_turn_button = pygame.Rect(screen_width - 150, 10, 140, 40)
        self.resign_button = pygame.Rect(screen_width - 150, 60, 140, 40)

    def render(self):
        """Render the entire game state."""
        # Update animation timing
        current_time = time.time()
        self.delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

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
        """
        Draw a single unit.

        Rendering priority (cascading fallback):
        1. Animated sprite sheet (if available and not disabled)
        2. Static sprite image (if available and not disabled)
        3. Letter representation (always available as fallback)
        """
        # Check settings for what's disabled
        animations_disabled = self.settings.get('graphics.disable_animations', False)
        static_sprites_disabled = self.settings.get('graphics.disable_unit_sprites', False)

        # Try animated sprite first (if not disabled)
        if not animations_disabled:
            if self.animator and self.animator.has_animations(unit.type):
                animated_frame = self.animator.get_frame(unit, self.delta_time)
                if animated_frame:
                    self._draw_unit_sprite(unit, animated_frame)
                    self._draw_paralysis_indicator(unit)
                    self._draw_unit_health_bar(unit)
                    return

        # Fall back to static sprite (if not disabled)
        if not static_sprites_disabled:
            static_sprite = self.unit_images.get(unit.type)
            if static_sprite:
                self._draw_unit_sprite(unit, static_sprite)
                self._draw_paralysis_indicator(unit)
                self._draw_unit_health_bar(unit)
                return

        # Fall back to letter representation
        self._draw_unit_letter(unit)
        self._draw_paralysis_indicator(unit)
        self._draw_unit_health_bar(unit)

    def _draw_paralysis_indicator(self, unit):
        """Draw paralysis indicator if unit is paralyzed."""
        if not unit.is_paralyzed():
            return

        tile_rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.screen, (148, 0, 211), tile_rect, 3)

        indicator_font = get_font(24)
        indicator_text = indicator_font.render(f"P:{unit.paralyzed_turns}", True, (148, 0, 211))
        indicator_rect = indicator_text.get_rect(topright=(
            unit.x * TILE_SIZE + TILE_SIZE - 2,
            unit.y * TILE_SIZE + 2
        ))
        bg_rect = indicator_rect.inflate(4, 2)
        pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
        pygame.draw.rect(self.screen, (148, 0, 211), bg_rect, 1)
        self.screen.blit(indicator_text, indicator_rect)

    def _draw_unit_sprite(self, unit, sprite):
        """Draw a unit using its sprite image."""
        # Create a copy of the sprite for modifications
        display_sprite = sprite.copy()

        # Apply visual effects
        # Gray out if can't act
        if not unit.can_move and not unit.can_attack:
            # Create a darkened version
            dark_surface = pygame.Surface(display_sprite.get_size())
            dark_surface.fill((128, 128, 128))
            display_sprite.blit(dark_surface, (0, 0), special_flags=pygame.BLEND_MULT)

        # Purple tint for paralyzed
        if unit.is_paralyzed():
            purple_surface = pygame.Surface(display_sprite.get_size())
            purple_surface.fill((200, 150, 255))
            display_sprite.blit(purple_surface, (0, 0), special_flags=pygame.BLEND_MULT)

        # Draw player-colored border around sprite
        player_color = PLAYER_COLORS.get(unit.player, (255, 255, 255))
        border_rect = pygame.Rect(
            unit.x * TILE_SIZE + 1,
            unit.y * TILE_SIZE + 1,
            TILE_SIZE - 2,
            TILE_SIZE - 2
        )
        pygame.draw.rect(self.screen, player_color, border_rect, 2)

        # Center the sprite in the tile
        sprite_rect = display_sprite.get_rect(center=(
            unit.x * TILE_SIZE + TILE_SIZE // 2,
            unit.y * TILE_SIZE + TILE_SIZE // 2
        ))
        self.screen.blit(display_sprite, sprite_rect)

    def _draw_unit_letter(self, unit):
        """Draw a unit using its letter representation (fallback)."""
        font = get_font(28)
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
        font = get_font(28)
        player_color = PLAYER_COLORS.get(self.game_state.current_player, (255, 255, 255))

        # Draw player info and gold
        gold_text = f"Player {self.game_state.current_player} Gold: ${self.game_state.player_gold[self.game_state.current_player]}"
        text_surface = font.render(gold_text, True, (255, 215, 0))
        text_rect = text_surface.get_rect(topleft=(10, 10))
        bg_rect = text_rect.inflate(10, 5)

        pygame.draw.rect(self.screen, player_color, bg_rect)
        pygame.draw.rect(self.screen, (255, 215, 0), bg_rect, 2)
        self.screen.blit(text_surface, text_rect)

        # Draw turn counter
        turn_text = f"Turn: {self.game_state.turn_number + 1}"
        if self.game_state.max_turns:
            turn_text += f" / {self.game_state.max_turns}"
        turn_surface = font.render(turn_text, True, (255, 255, 255))
        turn_rect = turn_surface.get_rect(topleft=(10, bg_rect.bottom + 5))
        turn_bg_rect = turn_rect.inflate(10, 5)

        pygame.draw.rect(self.screen, (50, 50, 65), turn_bg_rect)
        pygame.draw.rect(self.screen, (100, 150, 200), turn_bg_rect, 2)
        self.screen.blit(turn_surface, turn_rect)

        # Skip End Turn and Resign buttons in replay mode
        if self.replay_mode:
            return

        # Draw End Turn button
        mouse_pos = pygame.mouse.get_pos()
        button_color = (100, 150, 100) if self.end_turn_button.collidepoint(mouse_pos) else (70, 120, 70)

        pygame.draw.rect(self.screen, button_color, self.end_turn_button)
        pygame.draw.rect(self.screen, (255, 255, 255), self.end_turn_button, 2)

        button_font = get_font(32)
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
                x, y, self.game_state.grid, self.game_state.units,
                moving_unit=unit, is_destination=False
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

    def draw_attack_range_overlay(self, positions):
        """
        Draw attack range preview overlay highlighting attackable positions.

        Args:
            positions: List of (x, y) tuples for positions that can be attacked
        """
        for x, y in positions:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
            overlay.set_alpha(100)
            overlay.fill((255, 150, 50))  # Orange tint for attack range
            self.screen.blit(overlay, (x * TILE_SIZE, y * TILE_SIZE))

            # Draw border around position
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.screen, (255, 100, 0), rect, 2)

    def draw_unit_tooltip(self, mouse_pos):
        """
        Draw a tooltip showing unit stats when hovering over a unit.

        Args:
            mouse_pos: Tuple of (x, y) mouse position
        """
        # Convert mouse position to grid coordinates
        grid_x = mouse_pos[0] // TILE_SIZE
        grid_y = mouse_pos[1] // TILE_SIZE

        # Check bounds
        if not (0 <= grid_x < self.game_state.grid.width and
                0 <= grid_y < self.game_state.grid.height):
            return

        # Find unit at position
        unit = self.game_state.get_unit_at_position(grid_x, grid_y)
        if not unit:
            return

        # Get unit data
        unit_data = UNIT_DATA.get(unit.type, {})
        unit_name = unit_data.get('name', unit.type)

        # Build tooltip lines
        # Handle attack display - can be int or dict (for ranged units like Mage/Sorcerer)
        attack_data = unit.attack_data
        if isinstance(attack_data, dict):
            attack_str = f"{attack_data.get('adjacent', 0)}/{attack_data.get('range', 0)}"
        else:
            attack_str = str(attack_data)

        lines = [
            f"{unit_name} (P{unit.player})",
            f"HP: {unit.health}/{unit.max_health}",
            f"ATK: {attack_str}  DEF: {unit.defence}",
            f"MOV: {unit.movement_range}",
        ]

        # Add status indicators
        status_parts = []
        if unit.can_move:
            status_parts.append("Can Move")
        if unit.can_attack:
            status_parts.append("Can Act")
        if unit.is_paralyzed():
            status_parts.append(f"Paralyzed ({unit.paralyzed_turns})")

        if status_parts:
            lines.append(" | ".join(status_parts))

        # Calculate tooltip dimensions
        font = get_font(20)
        padding = 8
        line_height = 22
        max_width = 0

        for line in lines:
            text_surface = font.render(line, True, (255, 255, 255))
            max_width = max(max_width, text_surface.get_width())

        tooltip_width = max_width + 2 * padding
        tooltip_height = len(lines) * line_height + 2 * padding

        # Position tooltip near mouse but avoid going off-screen
        tooltip_x = mouse_pos[0] + 15
        tooltip_y = mouse_pos[1] + 15

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        if tooltip_x + tooltip_width > screen_width:
            tooltip_x = mouse_pos[0] - tooltip_width - 5
        if tooltip_y + tooltip_height > screen_height:
            tooltip_y = mouse_pos[1] - tooltip_height - 5

        # Ensure tooltip stays on screen
        tooltip_x = max(5, tooltip_x)
        tooltip_y = max(5, tooltip_y)

        # Draw tooltip background
        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, (30, 30, 45), tooltip_rect, border_radius=6)

        # Draw player-colored border
        player_color = PLAYER_COLORS.get(unit.player, (255, 255, 255))
        pygame.draw.rect(self.screen, player_color, tooltip_rect, width=2, border_radius=6)

        # Draw text lines
        for i, line in enumerate(lines):
            # First line (unit name) uses player color
            if i == 0:
                text_color = player_color
            else:
                text_color = (220, 220, 220)

            text_surface = font.render(line, True, text_color)
            text_y = tooltip_y + padding + i * line_height
            self.screen.blit(text_surface, (tooltip_x + padding, text_y))

    def get_rgb_array(self):
        """Get the current screen as RGB array."""
        # Convert pygame surface to numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)),
            axes=(1, 0, 2)
        )

    def set_unit_animation_state(self, unit, state):
        """
        Set the animation state for a unit.

        Args:
            unit: Unit object
            state: Animation state ('idle', 'move_down', 'move_up',
                   'move_left', 'move_right')
        """
        if self.animator:
            self.animator.set_unit_state(unit, state)

    def update_unit_animation_from_movement(self, unit, from_pos, to_pos):
        """
        Update unit animation based on movement direction.

        Args:
            unit: Unit object
            from_pos: Tuple (x, y) of starting position
            to_pos: Tuple (x, y) of ending position
        """
        if self.animator:
            self.animator.update_unit_state_from_movement(unit, from_pos, to_pos)

    def set_unit_idle(self, unit):
        """Set a unit to idle animation state."""
        if self.animator:
            self.animator.set_idle(unit)

    def cleanup_unit_animation(self, unit):
        """Clean up animation data for a removed unit."""
        if self.animator:
            self.animator.cleanup_unit(unit)

    def close(self):
        """Close the renderer."""
        pygame.quit()
