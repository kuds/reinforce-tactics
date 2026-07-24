"""Menu for loading saved games with enhanced preview and info."""

import json
import os
from typing import Any

import pygame

from reinforcetactics.constants import PLAYER_COLORS
from reinforcetactics.ui import theme
from reinforcetactics.ui.components.list_panel import ScrollList, draw_panel, split_panels
from reinforcetactics.ui.components.map_preview import get_tile_color
from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.in_game.confirmation_dialog import ConfirmationDialog
from reinforcetactics.ui.menus.save_load.utils import extract_date_from_filename, get_player_display_name
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import get_language

# Split-panel proportions and row metrics for this screen.
LIST_FRACTION = 0.55
ITEM_HEIGHT = 50


class LoadGameMenu(Menu):
    """Menu for loading saved games with visual previews and info."""

    def __init__(self, screen: pygame.Surface | None = None, saves_dir: str = "saves") -> None:
        """
        Initialize load game menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            saves_dir: Directory containing save files
        """
        super().__init__(screen, get_language().get("load_game.title", "Load Game"))
        self.saves_dir = saves_dir
        self.save_files: list[str] = []
        self.save_metadata: dict[str, dict[str, Any]] = {}
        self._load_saves()
        self._setup_options()

        # Cache for map previews
        self._preview_cache: dict[str, pygame.Surface] = {}

    def _load_saves(self) -> None:
        """Load available save files and their metadata."""
        if os.path.exists(self.saves_dir):
            all_saves = []
            for f in os.listdir(self.saves_dir):
                if f.endswith(".json"):
                    filepath = os.path.join(self.saves_dir, f)
                    all_saves.append(filepath)

            # Sort by modification time (newest first)
            all_saves.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            self.save_files = all_saves

            # Load metadata for each save
            for filepath in self.save_files:
                self._load_save_metadata(filepath)

    def _load_save_metadata(self, filepath: str) -> None:
        """Load metadata from a save file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # Parse timestamp
            timestamp_str = data.get("timestamp", "")
            try:
                # Try parsing our format "YYYY-MM-DD HH-MM-SS"
                date_str = timestamp_str.split(" ")[0] if timestamp_str else "Unknown"
            except (ValueError, TypeError):
                date_str = self._extract_date_from_filename(os.path.basename(filepath))

            # Get player info
            player_configs = data.get("player_configs", [])
            player1_name = self._get_player_display_name(player_configs, 0)
            player2_name = self._get_player_display_name(player_configs, 1)

            # Get turn info
            turn_number = data.get("turn_number", 0)
            current_player = data.get("current_player", 1)

            # Get gold for each player
            player_gold = data.get("player_gold", {})
            # Convert keys to int if they are strings (JSON serializes dict keys as strings)
            player_gold = {int(k): v for k, v in player_gold.items()}

            # Get map info
            map_file = data.get("map_file", "Unknown Map")
            map_name = os.path.basename(map_file).replace(".csv", "").replace("_", " ").title() if map_file else "Unknown Map"

            # Count units per player
            units = data.get("units", [])
            unit_counts: dict[int, int] = {}
            unit_types_per_player: dict[int, dict[str, int]] = {}
            total_health_per_player: dict[int, int] = {}

            for unit in units:
                player = unit.get("player", 0)
                unit_type = unit.get("type", "W")
                health = unit.get("health", 0)

                unit_counts[player] = unit_counts.get(player, 0) + 1

                if player not in unit_types_per_player:
                    unit_types_per_player[player] = {}
                unit_types_per_player[player][unit_type] = unit_types_per_player[player].get(unit_type, 0) + 1

                if player not in total_health_per_player:
                    total_health_per_player[player] = 0
                total_health_per_player[player] += health

            # Get tile data for preview
            tiles = data.get("tiles", [])

            # Determine map dimensions from tiles
            map_width = 0
            map_height = 0
            for tile in tiles:
                map_width = max(map_width, tile.get("x", 0) + 1)
                map_height = max(map_height, tile.get("y", 0) + 1)

            # Get game state
            game_over = data.get("game_over", False)
            winner = data.get("winner")
            num_players = data.get("num_players", 2)

            self.save_metadata[filepath] = {
                "date": date_str,
                "timestamp": timestamp_str,
                "player1": player1_name,
                "player2": player2_name,
                "turn_number": turn_number,
                "current_player": current_player,
                "player_gold": player_gold,
                "map_name": map_name,
                "map_file": map_file,
                "tiles": tiles,
                "units": units,
                "unit_counts": unit_counts,
                "unit_types_per_player": unit_types_per_player,
                "total_health_per_player": total_health_per_player,
                "map_width": map_width,
                "map_height": map_height,
                "game_over": game_over,
                "winner": winner,
                "num_players": num_players,
            }

        except (OSError, json.JSONDecodeError):
            # Store minimal metadata for failed loads
            self.save_metadata[filepath] = {
                "date": self._extract_date_from_filename(os.path.basename(filepath)),
                "timestamp": "",
                "player1": "Player 1",
                "player2": "Player 2",
                "turn_number": 0,
                "current_player": 1,
                "player_gold": {},
                "map_name": "Unknown",
                "map_file": "",
                "tiles": [],
                "units": [],
                "unit_counts": {},
                "unit_types_per_player": {},
                "total_health_per_player": {},
                "map_width": 0,
                "map_height": 0,
                "game_over": False,
                "winner": None,
                "num_players": 2,
            }

    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from save filename."""
        return extract_date_from_filename(filename)

    def _get_player_display_name(self, player_configs: list[dict], player_idx: int) -> str:
        """Get a display name for a player from config."""
        return get_player_display_name(player_configs, player_idx)

    def _get_display_name(self, filepath: str) -> str:
        """Get user-friendly display name for a save."""
        metadata = self.save_metadata.get(filepath, {})
        date = metadata.get("date", "Unknown")
        p1 = metadata.get("player1", "P1")
        p2 = metadata.get("player2", "P2")

        # Truncate long player names
        max_name_len = 12
        if len(p1) > max_name_len:
            p1 = p1[: max_name_len - 2] + ".."
        if len(p2) > max_name_len:
            p2 = p2[: max_name_len - 2] + ".."

        game_over = metadata.get("game_over", False)
        base = f"{date} - {p1} vs {p2}"
        if game_over:
            base += " [END]"
        return base

    def _setup_options(self) -> None:
        """Setup menu options for available save files."""
        if not self.save_files:
            lang = get_language()
            self.add_option(lang.get("load_game.no_saves", "No saved games found"), lambda: None)
        else:
            for save_file in self.save_files:
                display_name = self._get_display_name(save_file)

                def make_callback(p: str = save_file) -> str:
                    return p

                self.add_option(display_name, make_callback)

        self.add_option(get_language().get("common.back", "Back"), lambda: None)

    def _panels(self) -> tuple[pygame.Rect, pygame.Rect]:
        """The list and detail panel rectangles for the current window."""
        return split_panels(self.screen, LIST_FRACTION)

    def _scroll_list(self) -> ScrollList:
        """The list geometry, shared by hit-testing and drawing."""
        left_panel, _ = self._panels()
        return ScrollList(left_panel, ITEM_HEIGHT)

    def _populate_option_rects(self) -> None:
        """Populate option_rects for click detection matching split-panel layout."""
        scroll_list = self._scroll_list()
        self.max_visible_options = scroll_list.capacity
        self.option_rects = scroll_list.item_rects(self.scroll_offset, len(self.options))

    def _generate_save_map_preview(self, filepath: str, width: int, height: int) -> pygame.Surface | None:
        """Generate a map preview from save's tile data."""
        cache_key = f"{filepath}_{width}_{height}"
        if cache_key in self._preview_cache:
            return self._preview_cache[cache_key]

        metadata = self.save_metadata.get(filepath, {})
        tiles = metadata.get("tiles", [])
        units = metadata.get("units", [])
        map_width = metadata.get("map_width", 0)
        map_height = metadata.get("map_height", 0)

        if not tiles or map_width == 0 or map_height == 0:
            return None

        try:
            # Create preview surface
            preview = pygame.Surface((width, height))
            preview.fill(theme.BG)

            # Calculate tile size
            tile_width = width / map_width
            tile_height = height / map_height

            # Build a 2D grid from tiles list
            tile_grid: dict[tuple[int, int], dict[str, Any]] = {}
            for tile in tiles:
                x, y = tile.get("x", 0), tile.get("y", 0)
                tile_grid[(x, y)] = tile

            # Render each tile
            for y in range(map_height):
                for x in range(map_width):
                    tile_data = tile_grid.get((x, y), {})
                    tile_type = tile_data.get("type", "p")
                    tile_player = tile_data.get("player")
                    # Build combined tile code for get_tile_color (e.g. "h_1")
                    tile_code = f"{tile_type}_{tile_player}" if tile_player else tile_type
                    color = get_tile_color(tile_code)

                    rect = pygame.Rect(int(x * tile_width), int(y * tile_height), int(tile_width) + 1, int(tile_height) + 1)
                    pygame.draw.rect(preview, color, rect)

            # Render units on top
            for unit in units:
                ux, uy = unit.get("x", 0), unit.get("y", 0)
                player = unit.get("player", 1)

                # Draw a small colored circle for units
                center_x = int((ux + 0.5) * tile_width)
                center_y = int((uy + 0.5) * tile_height)
                radius = max(2, int(min(tile_width, tile_height) * 0.35))

                unit_color = PLAYER_COLORS.get(player, (255, 255, 255))
                pygame.draw.circle(preview, unit_color, (center_x, center_y), radius)
                # Add a small dark outline
                pygame.draw.circle(preview, theme.PREVIEW_UNIT_OUTLINE, (center_x, center_y), radius, 1)

            self._preview_cache[cache_key] = preview
            return preview

        except Exception:
            return None

    def draw(self) -> None:
        """Draw the load game menu with split-panel layout."""
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

        self._draw_save_list(left_panel)
        self._draw_preview_panel(right_panel)

        pygame.display.flip()

    def _draw_save_list(self, panel_rect: pygame.Rect) -> None:
        """Draw the scrollable save list."""
        scroll_list = ScrollList(panel_rect, ITEM_HEIGHT)
        # Sync with base class so keyboard scrolling and mouse-wheel bounds
        # match the visible count.
        self.max_visible_options = scroll_list.capacity

        total = len(self.options)
        start_idx, _ = scroll_list.visible_range(self.scroll_offset, total)
        self.option_rects = scroll_list.item_rects(self.scroll_offset, total)

        text_font = get_font(theme.FONT_SIZE_BODY)

        for display_idx, item_rect in enumerate(self.option_rects):
            i = start_idx + display_idx
            text, _ = self.options[i]

            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index
            scroll_list.draw_row(self.screen, item_rect, selected=is_selected, hovered=is_hovered)

            if is_selected:
                text_color = self.selected_color
            elif is_hovered:
                text_color = self.hover_color
            else:
                text_color = self.text_color

            # Ellipsize rather than hard-clipping: a chopped glyph reads as a
            # rendering glitch, "..." reads as "there is more".
            label = ellipsize(text, text_font, item_rect.width - 20)
            text_surface = text_font.render(label, True, text_color)
            text_rect = text_surface.get_rect(midleft=(item_rect.x + 10, item_rect.centery))
            self.screen.blit(text_surface, text_rect)

        scroll_list.draw_scroll_indicators(self.screen, self.scroll_offset, total)

    def _draw_preview_panel(self, panel_rect: pygame.Rect) -> None:
        """Draw the preview and details panel."""
        # Get currently selected/hovered save
        active_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        if active_index < 0 or active_index >= len(self.save_files):
            # Draw placeholder
            font = get_font(theme.FONT_SIZE_SUBHEADING)
            ScrollList.draw_empty_hint(self.screen, panel_rect, "Select a save to preview", font)
            return

        filepath = self.save_files[active_index]
        metadata = self.save_metadata.get(filepath, {})

        # Generate and draw map preview
        preview_size = min(220, panel_rect.width - 40, panel_rect.height // 3)
        preview = self._generate_save_map_preview(filepath, preview_size, preview_size)

        preview_x = panel_rect.x + (panel_rect.width - preview_size) // 2
        preview_y = panel_rect.y + 15

        if preview:
            self.screen.blit(preview, (preview_x, preview_y))
            preview_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
            pygame.draw.rect(self.screen, theme.FRAME_BORDER, preview_rect, width=theme.BORDER_WIDTH_HOVER)
        else:
            # Draw placeholder for missing preview
            placeholder_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
            pygame.draw.rect(self.screen, theme.PLACEHOLDER_BG, placeholder_rect)
            pygame.draw.rect(self.screen, theme.FRAME_BORDER, placeholder_rect, width=theme.BORDER_WIDTH_HOVER)

            placeholder_font = get_font(theme.FONT_SIZE_BODY)
            placeholder_text = placeholder_font.render("No Preview", True, theme.TEXT_PLACEHOLDER)
            placeholder_text_rect = placeholder_text.get_rect(center=placeholder_rect.center)
            self.screen.blit(placeholder_text, placeholder_text_rect)

        # Draw metadata below preview
        info_y = preview_y + preview_size + 20
        info_x = panel_rect.x + 20
        line_spacing = 26

        # Title: Map Name
        title_font = get_font(theme.FONT_SIZE_BODY)
        map_name = metadata.get("map_name", "Unknown Map")
        title_surface = title_font.render(map_name, True, self.title_color)
        self.screen.blit(title_surface, (info_x, info_y))
        info_y += 32

        # Info lines
        info_font = get_font(theme.FONT_SIZE_LABEL)
        label_color = theme.TEXT_INSTRUCTION
        value_color = theme.TEXT

        # Game status
        lang = get_language()
        game_over = metadata.get("game_over", False)
        winner = metadata.get("winner")

        status_label = info_font.render("Status: ", True, label_color)
        self.screen.blit(status_label, (info_x, info_y))

        if game_over:
            status_text = lang.get("load_game.status_completed", "Completed")
            status_color = theme.STATUS_INVALID
            status_value = info_font.render(status_text, True, status_color)
            self.screen.blit(status_value, (info_x + status_label.get_width(), info_y))
            info_y += line_spacing

            if winner:
                winner_label = info_font.render("Winner: ", True, label_color)
                winner_value = info_font.render(f"Player {winner}", True, PLAYER_COLORS.get(winner, value_color))
                self.screen.blit(winner_label, (info_x, info_y))
                self.screen.blit(winner_value, (info_x + winner_label.get_width(), info_y))
            else:
                draw_text = info_font.render("Draw", True, theme.RESULT_DRAW)
                self.screen.blit(draw_text, (info_x, info_y))
            info_y += line_spacing
        else:
            status_text = lang.get("load_game.status_in_progress", "In Progress")
            status_color = theme.STATUS_VALID
            status_value = info_font.render(status_text, True, status_color)
            self.screen.blit(status_value, (info_x + status_label.get_width(), info_y))
            info_y += line_spacing

        # Turn info
        turn_number = metadata.get("turn_number", 0)
        current_player = metadata.get("current_player", 1)
        turn_label = info_font.render("Turn: ", True, label_color)
        turn_value = info_font.render(f"{turn_number}", True, value_color)
        current_label = info_font.render(f"  (P{current_player}'s turn)", True, PLAYER_COLORS.get(current_player, value_color))
        self.screen.blit(turn_label, (info_x, info_y))
        self.screen.blit(turn_value, (info_x + turn_label.get_width(), info_y))
        self.screen.blit(current_label, (info_x + turn_label.get_width() + turn_value.get_width(), info_y))
        info_y += line_spacing

        # Number of players
        num_players = metadata.get("num_players", 2)
        players_label = info_font.render("Players: ", True, label_color)
        players_value = info_font.render(str(num_players), True, value_color)
        self.screen.blit(players_label, (info_x, info_y))
        self.screen.blit(players_value, (info_x + players_label.get_width(), info_y))
        info_y += line_spacing

        # Draw separator
        info_y += 5
        pygame.draw.line(self.screen, theme.PANEL_BORDER, (info_x, info_y), (panel_rect.right - 20, info_y), 1)
        info_y += 10

        # Player info with units and gold
        player_gold = metadata.get("player_gold", {})
        unit_counts = metadata.get("unit_counts", {})
        unit_types_per_player = metadata.get("unit_types_per_player", {})

        for player_num in range(1, num_players + 1):
            player_color = PLAYER_COLORS.get(player_num, theme.TEXT_NEUTRAL)

            # Player header
            player_name = metadata.get(f"player{player_num}", f"Player {player_num}")
            p_label = info_font.render(f"P{player_num}: ", True, player_color)
            p_name = info_font.render(player_name, True, value_color)
            self.screen.blit(p_label, (info_x, info_y))
            self.screen.blit(p_name, (info_x + p_label.get_width(), info_y))
            info_y += line_spacing

            # Gold
            gold = player_gold.get(player_num, 0)
            gold_label = info_font.render("  Gold: ", True, label_color)
            gold_value = info_font.render(f"{gold}", True, theme.HUD_GOLD_TEXT)
            self.screen.blit(gold_label, (info_x, info_y))
            self.screen.blit(gold_value, (info_x + gold_label.get_width(), info_y))
            info_y += line_spacing

            # Unit count with breakdown
            total_units = unit_counts.get(player_num, 0)
            unit_types = unit_types_per_player.get(player_num, {})

            units_label = info_font.render("  Units: ", True, label_color)
            units_value = info_font.render(f"{total_units}", True, value_color)
            self.screen.blit(units_label, (info_x, info_y))
            self.screen.blit(units_value, (info_x + units_label.get_width(), info_y))

            # Unit type breakdown on same line
            if unit_types:
                bx = info_x + units_label.get_width() + units_value.get_width() + 10
                small_font = get_font(theme.FONT_SIZE_CAPTION)
                breakdown_parts = []
                for utype, count in sorted(unit_types.items()):
                    breakdown_parts.append(f"{count}{utype}")
                breakdown_text = " ".join(breakdown_parts)
                breakdown_surf = small_font.render(f"({breakdown_text})", True, theme.TEXT_SUBTLE)
                self.screen.blit(breakdown_surf, (bx, info_y + 2))

            info_y += line_spacing

        # Draw separator before date
        info_y += 5
        pygame.draw.line(self.screen, theme.PANEL_BORDER, (info_x, info_y), (panel_rect.right - 20, info_y), 1)
        info_y += 10

        # Date
        date = metadata.get("date", "Unknown")
        date_label = info_font.render("Saved: ", True, label_color)
        date_value = info_font.render(date, True, value_color)
        self.screen.blit(date_label, (info_x, info_y))
        self.screen.blit(date_value, (info_x + date_label.get_width(), info_y))

    def run(self) -> dict[str, Any] | None:
        """
        Run load game menu.

        Returns:
            Dict with loaded save data, or None if cancelled
        """
        while True:
            selected_path = super().run()

            if not selected_path:
                return None

            # Check if this is a completed game and warn the user
            metadata = self.save_metadata.get(selected_path, {})
            if metadata.get("game_over", False):
                lang = get_language()
                winner = metadata.get("winner")
                if winner:
                    message = f"Player {winner} won. Load anyway?"
                else:
                    message = "Game ended in a draw. Load anyway?"

                dialog = ConfirmationDialog(
                    self.screen,
                    lang.get("load_game.completed_title", "Completed Game"),
                    message,
                    confirm_text=lang.get("common.confirm", "Confirm"),
                    cancel_text=lang.get("common.cancel", "Cancel"),
                )
                if not dialog.run():
                    # User cancelled — reset and let them pick again
                    self.running = True
                    continue

            # Load the actual save data from the file
            try:
                with open(selected_path, encoding="utf-8") as f:
                    save_data = json.load(f)
                return save_data
            except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading save file: {e}")
                return None
