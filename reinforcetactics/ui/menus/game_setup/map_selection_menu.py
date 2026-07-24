"""Menu for selecting a map when starting a new game."""

import os

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.components.list_panel import ScrollList, draw_panel, split_panels
from reinforcetactics.ui.components.map_preview import MapPreviewGenerator
from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import get_language

# Split-panel proportions and row metrics for this screen.
LIST_FRACTION = 0.4
ITEM_HEIGHT = 60
THUMBNAIL_SIZE = 45
PREVIEW_SIZE = 300


class MapSelectionMenu(Menu):
    """Menu for selecting a map when starting a new game with visual previews."""

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

    def _panels(self) -> tuple[pygame.Rect, pygame.Rect]:
        """The list and detail panel rectangles for the current window."""
        return split_panels(self.screen, LIST_FRACTION)

    def _scroll_list(self) -> ScrollList:
        """The list geometry, shared by hit-testing and drawing."""
        left_panel, _ = self._panels()
        return ScrollList(left_panel, ITEM_HEIGHT)

    def _populate_option_rects(self) -> None:
        """Populate option_rects for click detection matching the panel layout.

        Without this override the base class's centred full-width rows were
        used for the first frame's hit-testing, so an early click landed on
        the wrong option (or on nothing at all).
        """
        scroll_list = self._scroll_list()
        self.max_visible_options = scroll_list.capacity
        self.option_rects = scroll_list.item_rects(self.scroll_offset, len(self.options))

    def draw(self) -> None:
        """Draw the map selection menu with split-panel layout."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()

        # Draw title
        if self.title:
            title_surface = self.title_font.render(self.title, True, self.title_color)
            title_rect = title_surface.get_rect(centerx=screen_width // 2, y=20)
            self.screen.blit(title_surface, title_rect)

        left_panel, right_panel = self._panels()
        draw_panel(self.screen, left_panel)
        draw_panel(self.screen, right_panel)

        self._draw_map_list(left_panel)
        self._draw_preview_panel(right_panel)

        pygame.display.flip()

    def _draw_map_list(self, panel_rect: pygame.Rect) -> None:
        """Draw the scrollable map list with thumbnails."""
        scroll_list = ScrollList(panel_rect, ITEM_HEIGHT)
        # Sync with base class so keyboard scrolling matches the visible count
        self.max_visible_options = scroll_list.capacity

        total = len(self.options)
        start_idx, _ = scroll_list.visible_range(self.scroll_offset, total)
        self.option_rects = scroll_list.item_rects(self.scroll_offset, total)

        text_font = get_font(theme.FONT_SIZE_SUBHEADING)

        for display_idx, item_rect in enumerate(self.option_rects):
            i = start_idx + display_idx
            text, _ = self.options[i]
            # The Back option sits at the end and has no corresponding map.
            map_file = self.available_maps[i] if i < len(self.available_maps) else None

            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index
            scroll_list.draw_row(self.screen, item_rect, selected=is_selected, hovered=is_hovered)

            if is_selected:
                text_color = self.selected_color
            elif is_hovered:
                text_color = self.hover_color
            else:
                text_color = self.text_color

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
            text_x = item_rect.x + THUMBNAIL_SIZE + 15
            text_surface = text_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(midleft=(text_x, item_rect.centery))
            self.screen.blit(text_surface, text_rect)

        scroll_list.draw_scroll_indicators(self.screen, self.scroll_offset, total)

    def _draw_preview_panel(self, panel_rect: pygame.Rect) -> None:
        """Draw the preview and details panel."""
        # Get currently selected/hovered map
        active_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        placeholder_font = get_font(theme.FONT_SIZE_HEADING)
        if active_index < 0 or active_index >= len(self.available_maps):
            ScrollList.draw_empty_hint(self.screen, panel_rect, "Select a map to preview", placeholder_font)
            return

        map_file = self.available_maps[active_index]

        preview, metadata = self.preview_generator.generate_preview(map_file, PREVIEW_SIZE, PREVIEW_SIZE)

        if not preview or not metadata:
            # "Random Map" has nothing to render; say so rather than leaving
            # the panel blank.
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
