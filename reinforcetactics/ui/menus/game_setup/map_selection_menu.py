"""Menu for selecting a map when starting a new game."""
import os
from typing import Optional, List

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.components.map_preview import MapPreviewGenerator
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font


class MapSelectionMenu(Menu):
    """Menu for selecting a map when starting a new game with visual previews."""

    def __init__(self, screen: Optional[pygame.Surface] = None, maps_dir: str = "maps",
                 game_mode: Optional[str] = None) -> None:
        """
        Initialize map selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            maps_dir: Directory containing map files
            game_mode: Optional game mode to filter maps (e.g., "1v1", "2v2")
        """
        super().__init__(screen, get_language().get('new_game.title', 'Select Map'))
        self.maps_dir = maps_dir
        self.game_mode = game_mode
        self.available_maps: List[str] = []
        self.preview_generator = MapPreviewGenerator()
        self._load_maps()
        self._setup_options()

        # Preload previews for better responsiveness
        self._preload_previews()

    def _load_maps(self) -> None:
        """Load available map files."""
        if os.path.exists(self.maps_dir):
            if self.game_mode:
                # Load maps only from the specified game mode subfolder
                subdir_path = os.path.join(self.maps_dir, self.game_mode)
                if os.path.exists(subdir_path):
                    for f in os.listdir(subdir_path):
                        if f.endswith('.csv'):
                            # Store full path including maps/ prefix
                            map_path = os.path.join(self.maps_dir, self.game_mode, f)
                            self.available_maps.append(map_path)
            else:
                # Load maps from all subdirectories (backward compatibility)
                for subdir in ['1v1', '2v2']:
                    subdir_path = os.path.join(self.maps_dir, subdir)
                    if os.path.exists(subdir_path):
                        for f in os.listdir(subdir_path):
                            if f.endswith('.csv'):
                                # Store full path including maps/ prefix
                                self.available_maps.append(os.path.join(self.maps_dir, subdir, f))

        # Add random map option
        self.available_maps.insert(0, "random")

    def _setup_options(self) -> None:
        """Setup menu options for available maps."""
        for map_file in self.available_maps:
            # Use MapPreviewGenerator to format display names
            if map_file == "random":
                display_name = get_language().get('new_game.random_map', 'Random Map')
            else:
                _, metadata = self.preview_generator.generate_preview(map_file, 50, 50)
                display_name = metadata.get('name', os.path.basename(map_file))

            self.add_option(display_name, lambda m=map_file: m)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def _preload_previews(self) -> None:
        """Preload map previews for better responsiveness."""
        for map_file in self.available_maps:
            if map_file != "random":
                # Generate small thumbnail (50x50) for list
                self.preview_generator.generate_preview(map_file, 50, 50)
                # Generate larger preview (300x300) for detail panel
                self.preview_generator.generate_preview(map_file, 300, 300)

    def draw(self) -> None:
        """Draw the map selection menu with split-panel layout."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=20)
            self.screen.blit(title_surface, title_rect)

        # Define split panel layout
        panel_top = 80
        panel_height = screen_height - panel_top - 20

        # Left panel for map list (40% of width)
        left_panel_width = int(screen_width * 0.4)
        left_panel_rect = pygame.Rect(10, panel_top, left_panel_width - 20, panel_height)

        # Right panel for preview and details (60% of width)
        right_panel_x = left_panel_width
        right_panel_width = screen_width - left_panel_width - 10
        right_panel_rect = pygame.Rect(right_panel_x, panel_top, right_panel_width, panel_height)

        # Draw panel backgrounds
        pygame.draw.rect(self.screen, (40, 40, 50), left_panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (40, 40, 50), right_panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (80, 80, 100), left_panel_rect, width=2, border_radius=8)
        pygame.draw.rect(self.screen, (80, 80, 100), right_panel_rect, width=2, border_radius=8)

        # Draw map list in left panel
        self._draw_map_list(left_panel_rect)

        # Draw preview and details in right panel
        self._draw_preview_panel(right_panel_rect)

        pygame.display.flip()

    def _draw_map_list(self, panel_rect: pygame.Rect) -> None:
        """Draw the scrollable map list with thumbnails."""
        self.option_rects = []

        # Calculate visible area for scrolling
        list_padding = 10
        item_height = 60
        list_y = panel_rect.y + list_padding
        max_visible = (panel_rect.height - 2 * list_padding) // item_height

        # Determine which items to show based on scroll
        start_idx = self.scroll_offset
        end_idx = min(start_idx + max_visible, len(self.options))

        for i in range(start_idx, end_idx):
            display_idx = i - start_idx
            text, _ = self.options[i]
            # Get map file safely - the Back option is at the end and has no corresponding map
            map_file = self.available_maps[i] if i < len(self.available_maps) else None

            # Calculate item position
            item_y = list_y + display_idx * item_height
            item_rect = pygame.Rect(
                panel_rect.x + list_padding,
                item_y,
                panel_rect.width - 2 * list_padding,
                item_height - 5
            )

            # Determine styling
            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index

            # Choose colors
            if is_selected:
                bg_color = self.option_bg_selected_color
                text_color = self.selected_color
            elif is_hovered:
                bg_color = self.option_bg_hover_color
                text_color = self.hover_color
            else:
                bg_color = self.option_bg_color
                text_color = self.text_color

            # Draw background
            pygame.draw.rect(self.screen, bg_color, item_rect, border_radius=5)
            if is_selected or is_hovered:
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(self.screen, border_color, item_rect, width=2, border_radius=5)

            # Draw thumbnail if this is a map (not Back button) and not random
            thumbnail_size = 45
            if map_file and map_file != "random":
                thumbnail, _ = self.preview_generator.generate_preview(map_file, thumbnail_size, thumbnail_size)
                if thumbnail:
                    thumb_x = item_rect.x + 5
                    thumb_y = item_rect.y + (item_rect.height - thumbnail_size) // 2
                    self.screen.blit(thumbnail, (thumb_x, thumb_y))

                    # Draw border around thumbnail
                    thumb_rect = pygame.Rect(thumb_x, thumb_y, thumbnail_size, thumbnail_size)
                    pygame.draw.rect(self.screen, (100, 100, 120), thumb_rect, width=1)

            # Draw text with offset for thumbnail
            text_x = item_rect.x + thumbnail_size + 15
            text_font = get_font(28)
            text_surface = text_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(
                midleft=(text_x, item_rect.centery)
            )
            self.screen.blit(text_surface, text_rect)

            # Store rect for click detection
            # Note: These rects are stored in display order (0 to visible items)
            # The base Menu class handles mapping these to actual option indices
            self.option_rects.append(item_rect)

    def _draw_preview_panel(self, panel_rect: pygame.Rect) -> None:
        """Draw the preview and details panel."""
        # Get currently selected/hovered map
        active_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        if active_index < 0 or active_index >= len(self.available_maps):
            # Draw placeholder
            font = get_font(32)
            text = font.render("Select a map to preview", True, (150, 150, 150))
            text_rect = text.get_rect(center=panel_rect.center)
            self.screen.blit(text, text_rect)
            return

        map_file = self.available_maps[active_index]

        # Generate preview and get metadata
        preview_size = 300
        preview, metadata = self.preview_generator.generate_preview(map_file, preview_size, preview_size)

        if not preview or not metadata:
            return

        # Draw preview image
        preview_x = panel_rect.x + (panel_rect.width - preview_size) // 2
        preview_y = panel_rect.y + 20
        self.screen.blit(preview, (preview_x, preview_y))

        # Draw border around preview
        preview_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
        pygame.draw.rect(self.screen, (100, 100, 120), preview_rect, width=2)

        # Draw metadata below preview
        info_y = preview_y + preview_size + 20
        info_x = panel_rect.x + 20

        info_font = get_font(28)
        label_font = get_font(24)

        # Map name
        name_surface = info_font.render(metadata['name'], True, self.title_color)
        self.screen.blit(name_surface, (info_x, info_y))
        info_y += 35

        # Dimensions
        if metadata['width'] > 0:
            dim_text = f"Size: {metadata['width']}×{metadata['height']}"
            dim_surface = label_font.render(dim_text, True, self.text_color)
            self.screen.blit(dim_surface, (info_x, info_y))
            info_y += 28

        # Player count
        if metadata['player_count'] > 0:
            player_text = f"Players: {metadata['player_count']}"
            player_surface = label_font.render(player_text, True, self.text_color)
            self.screen.blit(player_surface, (info_x, info_y))
            info_y += 28

        # Difficulty
        if metadata.get('difficulty'):
            diff_text = f"Difficulty: {metadata['difficulty']}"
            diff_surface = label_font.render(diff_text, True, self.text_color)
            self.screen.blit(diff_surface, (info_x, info_y))
            info_y += 30

    def run(self) -> Optional[str]:
        """
        Run map selection menu.

        Returns:
            Selected map path string, or None if cancelled
        """
        return super().run()
