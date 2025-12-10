"""Menu for selecting a map when starting a new game."""
import os
from typing import Optional, List

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language


class MapSelectionMenu(Menu):
    """Menu for selecting a map when starting a new game."""

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
        self._load_maps()
        self._setup_options()

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
            if map_file == "random":
                display_name = get_language().get('new_game.random_map', 'Random Map')
            else:
                if self.game_mode:
                    # Show just the filename when game mode is already selected
                    display_name = os.path.splitext(os.path.basename(map_file))[0]
                else:
                    # Include the subdirectory in the display name to distinguish duplicates
                    # e.g., "1v1/6x6_beginner" instead of just "6x6_beginner"
                    relative_path = map_file.replace(self.maps_dir + os.sep, '')
                    # Remove .csv extension
                    display_name = os.path.splitext(relative_path)[0]
            self.add_option(display_name, lambda m=map_file: m)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def draw(self) -> None:
        """Draw the map selection menu with reduced spacing."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
            self.screen.blit(title_surface, title_rect)

        # Draw options with reduced spacing (25% less gap from title)
        # Original: start_y = screen_height // 3 (200 for 600px height)
        # Gap from title at y=50: 150px
        # 25% reduction: 150 * 0.75 = 112.5
        # New start_y: 50 + 112.5 = 162.5
        start_y = int(screen_height * 0.27)  # Approximately 162 for 600px height
        spacing = 60
        self.option_rects = []

        # Calculate maximum option width for uniform sizing
        padding_x = 40
        padding_y = 10
        max_text_width = 0
        for text, _ in self.options:
            display_text = f"> {text}"  # Use the selected format for consistent width
            text_surface = self.option_font.render(display_text, True, self.text_color)
            max_text_width = max(max_text_width, text_surface.get_width())

        uniform_width = max_text_width + 2 * padding_x

        for i, (text, _) in enumerate(self.options):
            # Determine styling based on state
            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index

            # Choose colors
            if is_selected:
                text_color = self.selected_color
                bg_color = self.option_bg_selected_color
            elif is_hovered:
                text_color = self.hover_color
                bg_color = self.option_bg_hover_color
            else:
                text_color = self.text_color
                bg_color = self.option_bg_color

            # Add selection indicator
            display_text = f"> {text}" if is_selected else f"  {text}"

            # Render text
            text_surface = self.option_font.render(display_text, True, text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2, y=start_y + i * spacing)

            # Create background rectangle with uniform width
            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,  # Center the uniform-width box
                text_rect.y - padding_y,
                uniform_width,
                text_rect.height + 2 * padding_y
            )

            # Draw rounded background rectangle
            pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=8)

            # Draw border for selected/hovered
            if is_selected or is_hovered:
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(self.screen, border_color, bg_rect, width=2, border_radius=8)

            # Draw text
            self.screen.blit(text_surface, text_rect)

            # Store rect for click detection
            self.option_rects.append(bg_rect)

        pygame.display.flip()

    def run(self) -> Optional[str]:
        """
        Run map selection menu.

        Returns:
            Selected map path string, or None if cancelled
        """
        return super().run()
