"""Menu for selecting a map when starting a new game."""

import os

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.components.list_panel import ScrollList
from reinforcetactics.ui.components.map_preview import MapPreviewGenerator
from reinforcetactics.ui.menus.list_detail import ListDetailMenu
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import get_language

# Row/preview metrics for this screen.
THUMBNAIL_SIZE = 45
PREVIEW_SIZE = 300


class MapSelectionMenu(ListDetailMenu):
    """Menu for selecting a map when starting a new game with visual previews."""

    LIST_FRACTION = 0.4
    ITEM_HEIGHT = 60
    EMPTY_HINT = "Select a map to preview"
    EMPTY_HINT_FONT_SIZE = theme.FONT_SIZE_HEADING

    def __init__(self, screen: pygame.Surface | None = None, maps_dir: str = "maps", game_mode: str | None = None) -> None:
        """
        Initialize map selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            maps_dir: Directory containing map files
            game_mode: Optional game mode to filter maps (e.g., "1v1", "2v2")
        """
        super().__init__(screen, get_language().get("new_game.title", "Select Map"))
        self.maps_dir = maps_dir
        self.game_mode = game_mode
        self.available_maps: list[str] = []
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
                    for f in sorted(os.listdir(subdir_path)):
                        if f.endswith(".csv"):
                            # Store full path including maps/ prefix
                            map_path = os.path.join(self.maps_dir, self.game_mode, f)
                            self.available_maps.append(map_path)
            else:
                # Load maps from all subdirectories (backward compatibility)
                for subdir in ["1v1", "2v2"]:
                    subdir_path = os.path.join(self.maps_dir, subdir)
                    if os.path.exists(subdir_path):
                        for f in sorted(os.listdir(subdir_path)):
                            if f.endswith(".csv"):
                                # Store full path including maps/ prefix
                                self.available_maps.append(os.path.join(self.maps_dir, subdir, f))

        # Add random map option
        self.available_maps.insert(0, "random")

    def _setup_options(self) -> None:
        """Setup menu options for available maps."""
        for map_file in self.available_maps:
            # Use MapPreviewGenerator to format display names
            if map_file == "random":
                display_name = get_language().get("new_game.random_map", "Random Map")
            else:
                _, metadata = self.preview_generator.generate_preview(map_file, 50, 50)
                display_name = metadata.get("name", os.path.basename(map_file))

            def make_callback(m: str = map_file) -> str:
                return m

            self.add_option(display_name, make_callback)

        self.add_option(get_language().get("common.back", "Back"), lambda: None)

    def _preload_previews(self) -> None:
        """Preload map previews for better responsiveness."""
        for map_file in self.available_maps:
            if map_file != "random":
                # Generate the thumbnail at the exact size the list draws it;
                # a mismatched size defeats the preload.
                self.preview_generator.generate_preview(map_file, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
                # Generate the larger preview for the detail panel
                self.preview_generator.generate_preview(map_file, PREVIEW_SIZE, PREVIEW_SIZE)

    def _detail_items(self) -> list:
        """The maps backing the detail panel (Back has no detail)."""
        return self.available_maps

    def _draw_row_content(self, index: int, item_rect: pygame.Rect, text: str, text_color) -> None:
        """Thumbnail plus label; the Back row has no corresponding map."""
        map_file = self.available_maps[index] if index < len(self.available_maps) else None

        if map_file and map_file != "random":
            thumbnail, _ = self.preview_generator.generate_preview(map_file, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
            if thumbnail:
                thumb_x = item_rect.x + 5
                thumb_y = item_rect.y + (item_rect.height - THUMBNAIL_SIZE) // 2
                self.screen.blit(thumbnail, (thumb_x, thumb_y))
                thumb_rect = pygame.Rect(thumb_x, thumb_y, THUMBNAIL_SIZE, THUMBNAIL_SIZE)
                pygame.draw.rect(self.screen, theme.FRAME_BORDER, thumb_rect, width=1)

        # Offset the label past the thumbnail column so every row's text
        # starts on the same vertical line.
        text_font = get_font(theme.FONT_SIZE_SUBHEADING)
        text_x = item_rect.x + THUMBNAIL_SIZE + 15
        text_surface = text_font.render(text, True, text_color)
        text_rect = text_surface.get_rect(midleft=(text_x, item_rect.centery))
        self.screen.blit(text_surface, text_rect)

    def _draw_detail_content(self, panel_rect: pygame.Rect, active_index: int) -> None:
        """Draw the preview and details panel."""
        map_file = self.available_maps[active_index]

        preview, metadata = self.preview_generator.generate_preview(map_file, PREVIEW_SIZE, PREVIEW_SIZE)

        if not preview or not metadata:
            # "Random Map" has nothing to render; say so rather than leaving
            # the panel blank.
            placeholder_font = get_font(theme.FONT_SIZE_HEADING)
            ScrollList.draw_empty_hint(self.screen, panel_rect, "No preview available", placeholder_font)
            return

        # Draw preview image
        preview_x = panel_rect.x + (panel_rect.width - PREVIEW_SIZE) // 2
        preview_y = panel_rect.y + 20
        self.screen.blit(preview, (preview_x, preview_y))

        # Draw border around preview
        preview_rect = pygame.Rect(preview_x, preview_y, PREVIEW_SIZE, PREVIEW_SIZE)
        pygame.draw.rect(self.screen, theme.FRAME_BORDER, preview_rect, width=theme.BORDER_WIDTH_HOVER)

        # Draw metadata below preview
        info_y = preview_y + PREVIEW_SIZE + 20
        info_x = panel_rect.x + 20

        info_font = get_font(theme.FONT_SIZE_SUBHEADING)
        label_font = get_font(theme.FONT_SIZE_BODY)

        # Map name
        name_surface = info_font.render(metadata["name"], True, self.title_color)
        self.screen.blit(name_surface, (info_x, info_y))
        info_y += 35

        # Dimensions
        if metadata["width"] > 0:
            dim_text = f"Size: {metadata['width']}×{metadata['height']}"
            dim_surface = label_font.render(dim_text, True, self.text_color)
            self.screen.blit(dim_surface, (info_x, info_y))
            info_y += 28

        # Player count
        if metadata["player_count"] > 0:
            player_text = f"Players: {metadata['player_count']}"
            player_surface = label_font.render(player_text, True, self.text_color)
            self.screen.blit(player_surface, (info_x, info_y))
            info_y += 28

        # Difficulty
        if metadata.get("difficulty"):
            diff_text = f"Difficulty: {metadata['difficulty']}"
            diff_surface = label_font.render(diff_text, True, self.text_color)
            self.screen.blit(diff_surface, (info_x, info_y))
            info_y += 30

    def run(self) -> str | None:
        """
        Run map selection menu.

        Returns:
            Selected map path string, or None if cancelled
        """
        return super().run()
