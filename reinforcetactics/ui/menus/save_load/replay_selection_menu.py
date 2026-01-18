"""Menu for selecting a replay to watch with enhanced preview and info."""
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.components.map_preview import MapPreviewGenerator
from reinforcetactics.ui.icons import get_arrow_up_icon, get_arrow_down_icon
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.constants import TILE_COLORS, PLAYER_COLORS


class ReplaySelectionMenu(Menu):
    """Menu for selecting a replay to watch with visual previews and info."""

    def __init__(self, screen: Optional[pygame.Surface] = None,
                 replays_dir: str = "replays") -> None:
        """
        Initialize replay selection menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            replays_dir: Directory containing replay files
        """
        super().__init__(screen, get_language().get('replay.title', 'Select Replay'))
        self.replays_dir = replays_dir
        self.replay_files: List[str] = []
        self.replay_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_replays()
        self._setup_options()

        # Preview generator for map previews
        self.preview_generator = MapPreviewGenerator()

        # Cache for replay map previews
        self._preview_cache: Dict[str, pygame.Surface] = {}

    def _load_replays(self) -> None:
        """Load available replay files and their metadata."""
        # Search in multiple directories
        search_dirs = [
            self.replays_dir,
            "tournament_results"
        ]

        all_replays = []

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Walk through directory tree to find all .json replay files
                for root, _, files in os.walk(search_dir):
                    for f in files:
                        if f.endswith('.json') and ('replay' in f.lower() or 'game_' in f.lower()):
                            filepath = os.path.join(root, f)
                            all_replays.append(filepath)

        # Sort by modification time (newest first)
        all_replays.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        self.replay_files = all_replays

        # Load metadata for each replay
        for filepath in self.replay_files:
            self._load_replay_metadata(filepath)

    def _load_replay_metadata(self, filepath: str) -> None:
        """Load metadata from a replay file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            game_info = data.get('game_info', {})
            timestamp_str = data.get('timestamp', '')

            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                date_str = timestamp.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                # Try to extract date from filename
                filename = os.path.basename(filepath)
                date_str = self._extract_date_from_filename(filename)

            # Get player info
            player_configs = game_info.get('player_configs', [])
            player1_name = self._get_player_display_name(player_configs, 0)
            player2_name = self._get_player_display_name(player_configs, 1)

            # Get winner info
            winner = game_info.get('winner')
            game_over = game_info.get('game_over', False)

            if winner == 0 or not game_over:
                result = "Draw" if game_over else "Incomplete"
            else:
                result = f"P{winner} Wins"

            # Get turn count
            total_turns = game_info.get('total_turns', 0)

            # Get map info
            map_file = game_info.get('map_file', 'Unknown Map')
            map_name = os.path.basename(map_file).replace('.csv', '').replace('_', ' ').title()

            # Store initial map for preview
            initial_map = game_info.get('initial_map')

            self.replay_metadata[filepath] = {
                'date': date_str,
                'player1': player1_name,
                'player2': player2_name,
                'winner': winner,
                'result': result,
                'total_turns': total_turns,
                'map_name': map_name,
                'map_file': map_file,
                'initial_map': initial_map,
                'num_players': game_info.get('num_players', 2),
                'max_turns': game_info.get('max_turns'),
            }

        except (json.JSONDecodeError, IOError):
            # Store minimal metadata for failed loads
            self.replay_metadata[filepath] = {
                'date': self._extract_date_from_filename(os.path.basename(filepath)),
                'player1': 'Player 1',
                'player2': 'Player 2',
                'winner': None,
                'result': 'Unknown',
                'total_turns': 0,
                'map_name': 'Unknown',
                'map_file': '',
                'initial_map': None,
                'num_players': 2,
                'max_turns': None,
            }

    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from replay filename."""
        # Handle formats like "game_20251228_053412_..." or "replay_20251228_053412"
        import re
        match = re.search(r'(\d{8})_(\d{6})', filename)
        if match:
            date_part = match.group(1)
            time_part = match.group(2)
            try:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
        return "Unknown Date"

    def _get_player_display_name(self, player_configs: List[Dict], player_idx: int) -> str:
        """Get a display name for a player from config."""
        if player_idx >= len(player_configs):
            return f"Player {player_idx + 1}"

        config = player_configs[player_idx]
        player_type = config.get('type', 'human')
        bot_type = config.get('bot_type', '')

        if player_type == 'human':
            return "Human"
        elif player_type == 'llm':
            # LLM players have a nice name field with model name
            name = config.get('name', '')
            if name:
                return name
            # Fallback to model if name not available
            model = config.get('model', '')
            if model:
                return model
            return "LLM"
        elif player_type == 'computer' or bot_type:
            # Bot players - use bot_type directly (AdvancedBot or SimpleBot)
            if bot_type:
                return bot_type
            return "Bot"
        else:
            return player_type.title()

    def _get_display_name(self, filepath: str) -> str:
        """Get user-friendly display name for a replay."""
        metadata = self.replay_metadata.get(filepath, {})
        date = metadata.get('date', 'Unknown')
        p1 = metadata.get('player1', 'P1')
        p2 = metadata.get('player2', 'P2')

        # Truncate long player names
        max_name_len = 15
        if len(p1) > max_name_len:
            p1 = p1[:max_name_len-2] + ".."
        if len(p2) > max_name_len:
            p2 = p2[:max_name_len-2] + ".."

        return f"{date} - {p1} vs {p2}"

    def _load_replay_info(self, filepath: str) -> Dict[str, Any]:
        """Load replay info from a file without loading full action history."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            game_info = data.get('game_info', {})
            player_configs = game_info.get('player_configs', [])

            # Get player names
            p1_name = 'P1'
            p2_name = 'P2'
            if player_configs:
                if len(player_configs) > 0:
                    p1_name = player_configs[0].get('name', 'P1')
                if len(player_configs) > 1:
                    p2_name = player_configs[1].get('name', 'P2')

            # Parse timestamp
            timestamp = data.get('timestamp', '')
            date_str = ''
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    date_str = timestamp[:10] if len(timestamp) >= 10 else timestamp

            return {
                'p1_name': p1_name,
                'p2_name': p2_name,
                'winner': game_info.get('winner'),
                'total_turns': game_info.get('total_turns', '?'),
                'num_actions': len(data.get('actions', [])),
                'date': date_str,
                'map_file': game_info.get('map_file', ''),
            }
        except (json.JSONDecodeError, IOError, KeyError):
            return {
                'p1_name': 'Unknown',
                'p2_name': 'Unknown',
                'winner': None,
                'total_turns': '?',
                'num_actions': 0,
                'date': '',
                'map_file': '',
            }

    def _setup_options(self) -> None:
        """Setup menu options for available replay files."""
        if not self.replay_files:
            lang = get_language()
            self.add_option(lang.get('replay.no_replays', 'No replays found'), lambda: None)
        else:
            for replay_file in self.replay_files:
                display_name = self._get_display_name(replay_file)
                self.add_option(display_name, lambda p=replay_file: p)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def _populate_option_rects(self) -> None:
        """Populate option_rects for click detection matching split-panel layout."""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Must match the layout in draw() and _draw_replay_list()
        panel_top = 80
        panel_height = screen_height - panel_top - 20
        left_panel_width = int(screen_width * 0.45)
        left_panel_rect = pygame.Rect(10, panel_top, left_panel_width - 20, panel_height)

        # Calculate visible area for scrolling (same as _draw_replay_list)
        list_padding = 10
        item_height = 50
        list_y = left_panel_rect.y + list_padding
        max_visible = (left_panel_rect.height - 2 * list_padding) // item_height

        # Determine which items to show based on scroll
        start_idx = self.scroll_offset
        end_idx = min(start_idx + max_visible, len(self.options))

        self.option_rects = []
        for i in range(start_idx, end_idx):
            display_idx = i - start_idx
            item_y = list_y + display_idx * item_height
            item_rect = pygame.Rect(
                left_panel_rect.x + list_padding,
                item_y,
                left_panel_rect.width - 2 * list_padding,
                item_height - 5
            )
            self.option_rects.append(item_rect)

    def _generate_replay_map_preview(self, filepath: str, width: int, height: int) -> Optional[pygame.Surface]:
        """Generate a map preview from replay's initial map data."""
        cache_key = f"{filepath}_{width}_{height}"
        if cache_key in self._preview_cache:
            return self._preview_cache[cache_key]

        metadata = self.replay_metadata.get(filepath, {})
        initial_map = metadata.get('initial_map')

        if not initial_map:
            return None

        try:
            # Create preview surface
            preview = pygame.Surface((width, height))
            preview.fill((30, 30, 40))

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
                    color = self._get_tile_color(tile)

                    rect = pygame.Rect(
                        int(x * tile_width),
                        int(y * tile_height),
                        int(tile_width) + 1,
                        int(tile_height) + 1
                    )
                    pygame.draw.rect(preview, color, rect)

            self._preview_cache[cache_key] = preview
            return preview

        except Exception:
            return None

    def _get_tile_color(self, tile: str) -> Tuple[int, int, int]:
        """Get the color for a tile type."""
        if '_' in tile:
            parts = tile.split('_')
            base_type = parts[0]

            if len(parts) > 1 and parts[1].isdigit():
                player_num = int(parts[1])
                return PLAYER_COLORS.get(player_num, TILE_COLORS.get(base_type, (128, 128, 128)))

        return TILE_COLORS.get(tile, (128, 128, 128))

    def draw(self) -> None:
        """Draw the replay selection menu with split-panel layout."""
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

        # Left panel for replay list (45% of width)
        left_panel_width = int(screen_width * 0.45)
        left_panel_rect = pygame.Rect(10, panel_top, left_panel_width - 20, panel_height)

        # Right panel for preview and details (55% of width)
        right_panel_x = left_panel_width
        right_panel_width = screen_width - left_panel_width - 10
        right_panel_rect = pygame.Rect(right_panel_x, panel_top, right_panel_width, panel_height)

        # Draw panel backgrounds
        pygame.draw.rect(self.screen, (40, 40, 50), left_panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (40, 40, 50), right_panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (80, 80, 100), left_panel_rect, width=2, border_radius=8)
        pygame.draw.rect(self.screen, (80, 80, 100), right_panel_rect, width=2, border_radius=8)

        # Draw replay list in left panel
        self._draw_replay_list(left_panel_rect)

        # Draw preview and details in right panel
        self._draw_preview_panel(right_panel_rect)

        pygame.display.flip()

    def _draw_replay_list(self, panel_rect: pygame.Rect) -> None:
        """Draw the scrollable replay list."""
        self.option_rects = []

        # Calculate visible area for scrolling
        list_padding = 10
        item_height = 50
        list_y = panel_rect.y + list_padding
        max_visible = (panel_rect.height - 2 * list_padding) // item_height

        # Determine which items to show based on scroll
        start_idx = self.scroll_offset
        end_idx = min(start_idx + max_visible, len(self.options))

        for i in range(start_idx, end_idx):
            display_idx = i - start_idx
            text, _ = self.options[i]

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

            # Draw text
            text_font = get_font(22)
            text_surface = text_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(
                midleft=(item_rect.x + 10, item_rect.centery)
            )

            # Clip text if too long
            if text_rect.width > item_rect.width - 20:
                # Create clipped surface
                clip_rect = pygame.Rect(0, 0, item_rect.width - 20, text_rect.height)
                text_surface = text_surface.subsurface(clip_rect)
                text_rect = text_surface.get_rect(
                    midleft=(item_rect.x + 10, item_rect.centery)
                )

            self.screen.blit(text_surface, text_rect)

            # Store rect for click detection
            self.option_rects.append(item_rect)

        # Draw scroll indicators if needed
        if len(self.options) > max_visible:
            if self.scroll_offset > 0:
                up_icon = get_arrow_up_icon(size=16, color=self.hover_color)
                up_rect = up_icon.get_rect(
                    centerx=panel_rect.centerx,
                    y=panel_rect.y + 2
                )
                self.screen.blit(up_icon, up_rect)

            if end_idx < len(self.options):
                down_icon = get_arrow_down_icon(size=16, color=self.hover_color)
                down_rect = down_icon.get_rect(
                    centerx=panel_rect.centerx,
                    bottom=panel_rect.bottom - 2
                )
                self.screen.blit(down_icon, down_rect)

    def _draw_preview_panel(self, panel_rect: pygame.Rect) -> None:
        """Draw the preview and details panel."""
        # Get currently selected/hovered replay
        active_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        if active_index < 0 or active_index >= len(self.replay_files):
            # Draw placeholder
            font = get_font(28)
            text = font.render("Select a replay to preview", True, (150, 150, 150))
            text_rect = text.get_rect(center=panel_rect.center)
            self.screen.blit(text, text_rect)
            return

        filepath = self.replay_files[active_index]
        metadata = self.replay_metadata.get(filepath, {})

        # Generate and draw map preview
        preview_size = min(250, panel_rect.width - 40, panel_rect.height // 2)
        preview = self._generate_replay_map_preview(filepath, preview_size, preview_size)

        preview_x = panel_rect.x + (panel_rect.width - preview_size) // 2
        preview_y = panel_rect.y + 20

        if preview:
            self.screen.blit(preview, (preview_x, preview_y))
            preview_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
            pygame.draw.rect(self.screen, (100, 100, 120), preview_rect, width=2)
        else:
            # Draw placeholder for missing preview
            placeholder_rect = pygame.Rect(preview_x, preview_y, preview_size, preview_size)
            pygame.draw.rect(self.screen, (50, 50, 60), placeholder_rect)
            pygame.draw.rect(self.screen, (100, 100, 120), placeholder_rect, width=2)

            placeholder_font = get_font(24)
            placeholder_text = placeholder_font.render("No Preview", True, (120, 120, 120))
            placeholder_text_rect = placeholder_text.get_rect(center=placeholder_rect.center)
            self.screen.blit(placeholder_text, placeholder_text_rect)

        # Draw metadata below preview
        info_y = preview_y + preview_size + 25
        info_x = panel_rect.x + 20
        line_spacing = 28

        # Title: Map Name
        title_font = get_font(26)
        map_name = metadata.get('map_name', 'Unknown Map')
        title_surface = title_font.render(map_name, True, self.title_color)
        self.screen.blit(title_surface, (info_x, info_y))
        info_y += 35

        # Info lines
        info_font = get_font(22)
        label_color = (180, 180, 180)
        value_color = (255, 255, 255)

        # Result
        result = metadata.get('result', 'Unknown')
        winner = metadata.get('winner')
        if winner == 1:
            result_color = PLAYER_COLORS.get(1, (255, 100, 100))
        elif winner == 2:
            result_color = PLAYER_COLORS.get(2, (100, 100, 255))
        elif result == 'Draw':
            result_color = (200, 200, 100)
        else:
            result_color = (150, 150, 150)

        result_label = info_font.render("Result: ", True, label_color)
        result_value = info_font.render(result, True, result_color)
        self.screen.blit(result_label, (info_x, info_y))
        self.screen.blit(result_value, (info_x + result_label.get_width(), info_y))
        info_y += line_spacing

        # Turns
        total_turns = metadata.get('total_turns', 0)
        max_turns = metadata.get('max_turns')
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
        num_players = metadata.get('num_players', 2)
        players_label = info_font.render("Players: ", True, label_color)
        players_value = info_font.render(str(num_players), True, value_color)
        self.screen.blit(players_label, (info_x, info_y))
        self.screen.blit(players_value, (info_x + players_label.get_width(), info_y))
        info_y += line_spacing

        # Player 1 info
        p1_name = metadata.get('player1', 'Player 1')
        p1_label = info_font.render("P1: ", True, PLAYER_COLORS.get(1, (255, 100, 100)))
        p1_value = info_font.render(p1_name, True, value_color)
        self.screen.blit(p1_label, (info_x, info_y))
        self.screen.blit(p1_value, (info_x + p1_label.get_width(), info_y))
        info_y += line_spacing

        # Player 2 info
        p2_name = metadata.get('player2', 'Player 2')
        p2_label = info_font.render("P2: ", True, PLAYER_COLORS.get(2, (100, 100, 255)))
        p2_value = info_font.render(p2_name, True, value_color)
        self.screen.blit(p2_label, (info_x, info_y))
        self.screen.blit(p2_value, (info_x + p2_label.get_width(), info_y))
        info_y += line_spacing

        # Date
        date = metadata.get('date', 'Unknown')
        date_label = info_font.render("Date: ", True, label_color)
        date_value = info_font.render(date, True, value_color)
        self.screen.blit(date_label, (info_x, info_y))
        self.screen.blit(date_value, (info_x + date_label.get_width(), info_y))

    def run(self) -> Optional[str]:
        """
        Run replay selection menu.

        Returns:
            Path to selected replay file, or None if cancelled
        """
        return super().run()
