"""Menu for selecting a replay to watch with enhanced preview and info."""

import json
import logging
import os
from datetime import datetime
from typing import Any

import pygame

from reinforcetactics.ui import theme
from reinforcetactics.ui.assets import PLAYER_COLORS
from reinforcetactics.ui.components.map_preview import get_tile_color
from reinforcetactics.ui.menus.list_detail import ListDetailMenu, draw_preview_or_placeholder
from reinforcetactics.ui.menus.save_load.utils import (
    as_dict,
    as_int,
    as_list,
    as_optional_int,
    as_player_count,
    extract_date_from_filename,
    map_display_name,
    mtime_or_zero,
    safe_player_display_name,
)
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.language import get_language

logger = logging.getLogger(__name__)


class ReplaySelectionMenu(ListDetailMenu):
    """Menu for selecting a replay to watch with visual previews and info."""

    LIST_FRACTION = 0.55
    # Replay rows are two-line cards, so they are taller than the save/map
    # rows and use a tighter gap.
    ITEM_HEIGHT = 62
    ROW_GAP = 4
    EMPTY_HINT = "Select a replay to preview"

    def __init__(self, screen: pygame.Surface | None = None, replays_dir: str = "replays") -> None:
        """
        Initialize replay selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            replays_dir: Directory containing replay files
        """
        super().__init__(screen, get_language().get("replay.title", "Select Replay"))
        self.replays_dir = replays_dir
        self.replay_files: list[str] = []
        self.replay_metadata: dict[str, dict[str, Any]] = {}
        self._load_replays()
        self._setup_options()

        # Cache for replay map previews
        self._preview_cache: dict[str, pygame.Surface] = {}

    def _load_replays(self) -> None:
        """Load available replay files and their metadata."""
        # Search in multiple directories
        search_dirs = [self.replays_dir, "tournament_results"]

        all_replays = []

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Walk through directory tree to find all .json replay files
                for root, _, files in os.walk(search_dir):
                    for f in files:
                        if f.endswith(".json") and ("replay" in f.lower() or "game_" in f.lower()):
                            filepath = os.path.join(root, f)
                            all_replays.append(filepath)

        # Sort by modification time (newest first)
        all_replays.sort(key=mtime_or_zero, reverse=True)
        self.replay_files = all_replays

        # Load metadata for each replay
        for filepath in self.replay_files:
            self._load_replay_metadata(filepath)

    def _load_replay_metadata(self, filepath: str) -> None:
        """Load metadata from a replay file.

        Never raises for a bad file. This runs for every file when the menu
        opens, so one replay the menu can't parse used to crash Watch Replay
        (and the app) on every visit; such a file now gets minimal
        "Unknown" metadata instead.
        """
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            self.replay_metadata[filepath] = self._parse_replay_metadata(filepath, data)
        except Exception as e:
            logger.warning("Could not read replay metadata from %s: %s", filepath, e)
            self.replay_metadata[filepath] = self._minimal_metadata(filepath)

    def _parse_replay_metadata(self, filepath: str, data: Any) -> dict[str, Any]:
        """Extract the fields the menu draws, coerced to the types it expects."""
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")

        game_info = as_dict(data.get("game_info"))
        timestamp_str = data.get("timestamp", "")

        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            date_str = timestamp.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # Try to extract date from filename
            filename = os.path.basename(filepath)
            date_str = extract_date_from_filename(filename)

        # Get player info
        player_configs = as_list(game_info.get("player_configs"))
        num_players = as_player_count(game_info.get("num_players"))
        player1_name = safe_player_display_name(player_configs, 0)
        player2_name = safe_player_display_name(player_configs, 1)
        player3_name = safe_player_display_name(player_configs, 2) if num_players > 2 else None
        player4_name = safe_player_display_name(player_configs, 3) if num_players > 3 else None

        # Get winner info
        winner = as_optional_int(game_info.get("winner"))
        game_over = bool(game_info.get("game_over", False))

        if winner == 0 or not game_over:
            result = "Draw" if game_over else "Incomplete"
        else:
            result = f"P{winner} Wins"

        # Get turn count
        total_turns = as_int(game_info.get("total_turns"), 0)

        # Get map info. Random-map games record "map_file": null, which
        # os.path.basename() used to choke on; they are labelled "Random Map".
        map_file = game_info.get("map_file")
        map_name = map_display_name(map_file)

        # Store initial map for preview
        initial_map = as_list(game_info.get("initial_map")) or None

        return {
            "date": date_str,
            "player1": player1_name,
            "player2": player2_name,
            "player3": player3_name,
            "player4": player4_name,
            "winner": winner,
            "result": result,
            "total_turns": total_turns,
            "map_name": map_name,
            "map_file": map_file if isinstance(map_file, str) else "",
            "initial_map": initial_map,
            "num_players": num_players,
            "max_turns": as_optional_int(game_info.get("max_turns")),
        }

    @staticmethod
    def _minimal_metadata(filepath: str) -> dict[str, Any]:
        """Metadata for a replay file that couldn't be read."""
        return {
            "date": extract_date_from_filename(os.path.basename(filepath)),
            "player1": "Player 1",
            "player2": "Player 2",
            "winner": None,
            "result": "Unknown",
            "total_turns": 0,
            "map_name": "Unknown",
            "map_file": "",
            "initial_map": None,
            "num_players": 2,
            "max_turns": None,
        }

    def _get_display_name(self, filepath: str) -> str:
        """Get user-friendly display name for a replay."""
        metadata = self.replay_metadata.get(filepath, {})
        date = metadata.get("date", "Unknown")
        p1 = metadata.get("player1", "P1")
        p2 = metadata.get("player2", "P2")

        # Truncate long player names
        max_name_len = 15
        if len(p1) > max_name_len:
            p1 = p1[: max_name_len - 2] + ".."
        if len(p2) > max_name_len:
            p2 = p2[: max_name_len - 2] + ".."

        return f"{date} - {p1} vs {p2}"

    def _setup_options(self) -> None:
        """Setup menu options for available replay files."""
        if not self.replay_files:
            lang = get_language()
            self.add_option(lang.get("replay.no_replays", "No replays found"), lambda: None)
        else:
            for replay_file in self.replay_files:
                display_name = self._get_display_name(replay_file)

                def make_callback(p: str = replay_file) -> str:
                    return p

                self.add_option(display_name, make_callback)

        self.add_option(get_language().get("common.back", "Back"), lambda: None)

    def _detail_items(self) -> list:
        """The replays backing the detail panel (Back has no detail)."""
        return self.replay_files

    def _generate_replay_map_preview(self, filepath: str, width: int, height: int) -> pygame.Surface | None:
        """Generate a map preview from replay's initial map data."""
        cache_key = f"{filepath}_{width}_{height}"
        if cache_key in self._preview_cache:
            return self._preview_cache[cache_key]

        metadata = self.replay_metadata.get(filepath, {})
        initial_map = metadata.get("initial_map")

        if not initial_map:
            return None

        try:
            # Create preview surface
            preview = pygame.Surface((width, height))
            preview.fill(theme.BG)

            map_height = len(initial_map)
            map_width = len(initial_map[0]) if map_height > 0 else 0

            if map_width == 0 or map_height == 0:
                return None

            # Calculate tile size
            tile_width = width / map_width
            tile_height = height / map_height

            # Render each tile
            for y in range(map_height):
                for x in range(map_width):
                    tile = str(initial_map[y][x])
                    color = get_tile_color(tile)

                    rect = pygame.Rect(int(x * tile_width), int(y * tile_height), int(tile_width) + 1, int(tile_height) + 1)
                    pygame.draw.rect(preview, color, rect)

            self._preview_cache[cache_key] = preview
            return preview

        except Exception:
            return None

    def _draw_row_content(self, index: int, item_rect, text: str, text_color) -> None:
        """Two-line replay card with a winner pip; centered text for Back."""
        main_font = get_font(theme.FONT_SIZE_LABEL)

        if index >= len(self.replay_files):
            # Non-replay items (e.g. "Back" button) - simple centered text
            text_surface = main_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(center=item_rect.center)
            self.screen.blit(text_surface, text_rect)
            return

        sub_font = get_font(theme.FONT_SIZE_CAPTION)
        is_active = index in (self.selected_index, self.hover_index)

        # Get metadata for richer rendering
        filepath = self.replay_files[index]
        metadata = self.replay_metadata.get(filepath, {})
        winner = metadata.get("winner")

        # Draw winner color pip on the left edge
        pip_width = 4
        pip_rect = pygame.Rect(item_rect.x + 2, item_rect.y + 6, pip_width, item_rect.height - 12)
        if winner and winner in PLAYER_COLORS:
            pip_color = PLAYER_COLORS[winner]
        elif metadata.get("result") == "Draw":
            pip_color = theme.RESULT_DRAW
        else:
            pip_color = theme.FRAME_BORDER
        pygame.draw.rect(self.screen, pip_color, pip_rect, border_radius=2)

        # Text area starts after the pip
        text_x = item_rect.x + pip_width + 10
        max_text_width = item_rect.width - pip_width - 20

        # Main line: matchup text
        main_surface = main_font.render(ellipsize(text, main_font, max_text_width), True, text_color)
        self.screen.blit(main_surface, (text_x, item_rect.y + 8))

        # Subtitle: result, turns, map
        result = metadata.get("result", "")
        total_turns = metadata.get("total_turns", 0)
        map_name = metadata.get("map_name", "")
        parts = []
        if result:
            parts.append(result)
        if total_turns:
            parts.append(f"{total_turns} turns")
        if map_name and map_name != "Unknown":
            parts.append(map_name)
        subtitle = "  |  ".join(parts) if parts else ""

        if subtitle:
            sub_color = theme.HOVER if is_active else theme.TEXT_MUTED
            sub_surface = sub_font.render(ellipsize(subtitle, sub_font, max_text_width), True, sub_color)
            self.screen.blit(sub_surface, (text_x, item_rect.y + 32))

    def _draw_detail_content(self, panel_rect, active_index: int) -> None:
        """Draw the preview and details panel."""
        filepath = self.replay_files[active_index]
        metadata = self.replay_metadata.get(filepath, {})

        # Generate and draw map preview
        preview_size = min(250, panel_rect.width - 40, panel_rect.height // 2)
        preview = self._generate_replay_map_preview(filepath, preview_size, preview_size)

        preview_x = panel_rect.x + (panel_rect.width - preview_size) // 2
        preview_y = panel_rect.y + 20

        draw_preview_or_placeholder(self.screen, preview, preview_x, preview_y, preview_size)

        # Draw metadata below preview
        info_y = preview_y + preview_size + 25
        info_x = panel_rect.x + 20
        line_spacing = 28

        # Title: Map Name
        title_font = get_font(theme.FONT_SIZE_SUBHEADING)
        map_name = metadata.get("map_name", "Unknown Map")
        title_surface = title_font.render(map_name, True, self.title_color)
        self.screen.blit(title_surface, (info_x, info_y))
        info_y += 35

        # Info lines
        info_font = get_font(theme.FONT_SIZE_LABEL)
        label_color = theme.TEXT_INSTRUCTION
        value_color = theme.TEXT

        # Result
        result = metadata.get("result", "Unknown")
        winner = metadata.get("winner")
        if winner and winner in PLAYER_COLORS:
            result_color = PLAYER_COLORS[winner]
        elif result == "Draw":
            result_color = theme.RESULT_DRAW
        else:
            result_color = theme.TEXT_PLACEHOLDER

        result_label = info_font.render("Result: ", True, label_color)
        result_value = info_font.render(result, True, result_color)
        self.screen.blit(result_label, (info_x, info_y))
        self.screen.blit(result_value, (info_x + result_label.get_width(), info_y))
        info_y += line_spacing

        # Turns
        total_turns = metadata.get("total_turns", 0)
        max_turns = metadata.get("max_turns")
        if max_turns:
            turns_text = f"{total_turns} / {max_turns}"
        else:
            turns_text = str(total_turns)

        turns_label = info_font.render("Turns: ", True, label_color)
        turns_value = info_font.render(turns_text, True, value_color)
        self.screen.blit(turns_label, (info_x, info_y))
        self.screen.blit(turns_value, (info_x + turns_label.get_width(), info_y))
        info_y += line_spacing

        # Players
        num_players = metadata.get("num_players", 2)
        players_label = info_font.render("Players: ", True, label_color)
        players_value = info_font.render(str(num_players), True, value_color)
        self.screen.blit(players_label, (info_x, info_y))
        self.screen.blit(players_value, (info_x + players_label.get_width(), info_y))
        info_y += line_spacing

        # Player info (supports 2v2)
        player_keys = ["player1", "player2", "player3", "player4"]
        for p_idx in range(num_players):
            p_name = metadata.get(player_keys[p_idx], f"Player {p_idx + 1}")
            if p_name is None:
                p_name = f"Player {p_idx + 1}"
            p_num = p_idx + 1
            p_label = info_font.render(f"P{p_num}: ", True, PLAYER_COLORS.get(p_num, theme.TEXT_NEUTRAL))
            p_value = info_font.render(p_name, True, value_color)
            self.screen.blit(p_label, (info_x, info_y))
            self.screen.blit(p_value, (info_x + p_label.get_width(), info_y))
            info_y += line_spacing

        # Date
        date = metadata.get("date", "Unknown")
        date_label = info_font.render("Date: ", True, label_color)
        date_value = info_font.render(date, True, value_color)
        self.screen.blit(date_label, (info_x, info_y))
        self.screen.blit(date_value, (info_x + date_label.get_width(), info_y))

    def run(self) -> str | None:
        """
        Run replay selection menu.

        Returns:
            Path to selected replay file, or None if cancelled
        """
        return super().run()
