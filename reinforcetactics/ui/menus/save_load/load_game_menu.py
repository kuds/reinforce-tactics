"""Menu for loading saved games with enhanced preview and info."""
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.icons import get_arrow_up_icon, get_arrow_down_icon
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.constants import TILE_COLORS, PLAYER_COLORS


class LoadGameMenu(Menu):
    """Menu for loading saved games with visual previews and info."""

    def __init__(self, screen: Optional[pygame.Surface] = None,
                 saves_dir: str = "saves") -> None:
        """
        Initialize load game menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            saves_dir: Directory containing save files
        """
        super().__init__(screen, get_language().get('load_game.title', 'Load Game'))
        self.saves_dir = saves_dir
        self.save_files: List[str] = []
        self.save_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_saves()
        self._setup_options()

        # Cache for map previews
        self._preview_cache: Dict[str, pygame.Surface] = {}

    def _load_saves(self) -> None:
        """Load available save files and their metadata."""
        if os.path.exists(self.saves_dir):
            all_saves = []
            for f in os.listdir(self.saves_dir):
                if f.endswith('.json'):
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
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Parse timestamp
            timestamp_str = data.get('timestamp', '')
            try:
                # Try parsing our format "YYYY-MM-DD HH-MM-SS"
                date_str = timestamp_str.split(' ')[0] if timestamp_str else 'Unknown'
            except (ValueError, TypeError):
                date_str = self._extract_date_from_filename(os.path.basename(filepath))

            # Get player info
            player_configs = data.get('player_configs', [])
            player1_name = self._get_player_display_name(player_configs, 0)
            player2_name = self._get_player_display_name(player_configs, 1)

            # Get turn info
            turn_number = data.get('turn_number', 0)
            current_player = data.get('current_player', 1)

            # Get gold for each player
            player_gold = data.get('player_gold', {})
            # Convert keys to int if they are strings (JSON serializes dict keys as strings)
            player_gold = {int(k): v for k, v in player_gold.items()}

            # Get map info
            map_file = data.get('map_file', 'Unknown Map')
            map_name = os.path.basename(map_file).replace('.csv', '').replace('_', ' ').title() if map_file else 'Unknown Map'

            # Count units per player
            units = data.get('units', [])
            unit_counts = {}
            unit_types_per_player: Dict[int, Dict[str, int]] = {}
            total_health_per_player: Dict[int, int] = {}

            for unit in units:
                player = unit.get('player', 0)
                unit_type = unit.get('type', 'W')
                health = unit.get('health', 0)

                unit_counts[player] = unit_counts.get(player, 0) + 1

                if player not in unit_types_per_player:
                    unit_types_per_player[player] = {}
                unit_types_per_player[player][unit_type] = unit_types_per_player[player].get(unit_type, 0) + 1

                if player not in total_health_per_player:
                    total_health_per_player[player] = 0
                total_health_per_player[player] += health

            # Get tile data for preview
            tiles = data.get('tiles', [])

            # Determine map dimensions from tiles
            map_width = 0
            map_height = 0
            for tile in tiles:
                map_width = max(map_width, tile.get('x', 0) + 1)
                map_height = max(map_height, tile.get('y', 0) + 1)

            # Get game state
            game_over = data.get('game_over', False)
            winner = data.get('winner')
            num_players = data.get('num_players', 2)

            self.save_metadata[filepath] = {
                'date': date_str,
                'timestamp': timestamp_str,
                'player1': player1_name,
                'player2': player2_name,
                'turn_number': turn_number,
                'current_player': current_player,
                'player_gold': player_gold,
                'map_name': map_name,
                'map_file': map_file,
                'tiles': tiles,
                'units': units,
                'unit_counts': unit_counts,
                'unit_types_per_player': unit_types_per_player,
                'total_health_per_player': total_health_per_player,
                'map_width': map_width,
                'map_height': map_height,
                'game_over': game_over,
                'winner': winner,
                'num_players': num_players,
            }

        except (json.JSONDecodeError, IOError):
            # Store minimal metadata for failed loads
            self.save_metadata[filepath] = {
                'date': self._extract_date_from_filename(os.path.basename(filepath)),
                'timestamp': '',
                'player1': 'Player 1',
                'player2': 'Player 2',
                'turn_number': 0,
                'current_player': 1,
                'player_gold': {},
                'map_name': 'Unknown',
                'map_file': '',
                'tiles': [],
                'units': [],
                'unit_counts': {},
                'unit_types_per_player': {},
                'total_health_per_player': {},
                'map_width': 0,
                'map_height': 0,
                'game_over': False,
                'winner': None,
                'num_players': 2,
            }

    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from save filename."""
        import re
        # Handle formats like "save_20251228_053412.json"
        match = re.search(r'(\d{8})_(\d{6})', filename)
        if match:
            date_part = match.group(1)
            try:
                dt = datetime.strptime(date_part, "%Y%m%d")
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
            name = config.get('name', '')
            if name:
                return name
            model = config.get('model', '')
            if model:
                return model
            return "LLM"
        elif player_type == 'computer' or bot_type:
            if bot_type:
                return bot_type
            return "Bot"
        else:
            return player_type.title()

    def _get_display_name(self, filepath: str) -> str:
        """Get user-friendly display name for a save."""
        metadata = self.save_metadata.get(filepath, {})
        date = metadata.get('date', 'Unknown')
        turn = metadata.get('turn_number', 0)
        p1 = metadata.get('player1', 'P1')
        p2 = metadata.get('player2', 'P2')

        # Truncate long player names
        max_name_len = 12
        if len(p1) > max_name_len:
            p1 = p1[:max_name_len-2] + ".."
        if len(p2) > max_name_len:
            p2 = p2[:max_name_len-2] + ".."

        return f"{date} - Turn {turn} - {p1} vs {p2}"

    def _setup_options(self) -> None:
        """Setup menu options for available save files."""
        if not self.save_files:
            lang = get_language()
            self.add_option(lang.get('load_game.no_saves', 'No saved games found'), lambda: None)
        else:
            for save_file in self.save_files:
                display_name = self._get_display_name(save_file)
                self.add_option(display_name, lambda p=save_file: p)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def _generate_save_map_preview(self, filepath: str, width: int, height: int) -> Optional[pygame.Surface]:
        """Generate a map preview from save's tile data."""
        cache_key = f"{filepath}_{width}_{height}"
        if cache_key in self._preview_cache:
            return self._preview_cache[cache_key]

        metadata = self.save_metadata.get(filepath, {})
        tiles = metadata.get('tiles', [])
        units = metadata.get('units', [])
        map_width = metadata.get('map_width', 0)
        map_height = metadata.get('map_height', 0)

        if not tiles or map_width == 0 or map_height == 0:
            return None

        try:
            # Create preview surface
            preview = pygame.Surface((width, height))
            preview.fill((30, 30, 40))

            # Calculate tile size
            tile_width = width / map_width
            tile_height = height / map_height

            # Build a 2D grid from tiles list
            tile_grid: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for tile in tiles:
                x, y = tile.get('x', 0), tile.get('y', 0)
                tile_grid[(x, y)] = tile

            # Render each tile
            for y in range(map_height):
                for x in range(map_width):
                    tile_data = tile_grid.get((x, y), {})
                    tile_type = tile_data.get('type', 'p')
                    tile_player = tile_data.get('player')
                    color = self._get_tile_color(tile_type, tile_player)

                    rect = pygame.Rect(
                        int(x * tile_width),
                        int(y * tile_height),
                        int(tile_width) + 1,
                        int(tile_height) + 1
                    )
                    pygame.draw.rect(preview, color, rect)

            # Render units on top
            for unit in units:
                ux, uy = unit.get('x', 0), unit.get('y', 0)
                player = unit.get('player', 1)

                # Draw a small colored circle for units
                center_x = int((ux + 0.5) * tile_width)
                center_y = int((uy + 0.5) * tile_height)
                radius = max(2, int(min(tile_width, tile_height) * 0.35))

                unit_color = PLAYER_COLORS.get(player, (255, 255, 255))
                pygame.draw.circle(preview, unit_color, (center_x, center_y), radius)
                # Add a small dark outline
                pygame.draw.circle(preview, (20, 20, 20), (center_x, center_y), radius, 1)

            self._preview_cache[cache_key] = preview
            return preview

        except Exception:
            return None

    def _get_tile_color(self, tile_type: str, player: Optional[int] = None) -> Tuple[int, int, int]:
        """Get the color for a tile type."""
        # If tile has a player owner and is a capturable structure, use player color tint
        if player and tile_type in ['h', 'b', 't']:
            base_color = TILE_COLORS.get(tile_type, (128, 128, 128))
            player_color = PLAYER_COLORS.get(player, (128, 128, 128))
            # Blend base color with player color
            blended = (
                (base_color[0] + player_color[0]) // 2,
                (base_color[1] + player_color[1]) // 2,
                (base_color[2] + player_color[2]) // 2
            )
            return blended

        return TILE_COLORS.get(tile_type, (128, 128, 128))

    def draw(self) -> None:
        """Draw the load game menu with split-panel layout."""
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

        # Left panel for save list (45% of width)
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

        # Draw save list in left panel
        self._draw_save_list(left_panel_rect)

        # Draw preview and details in right panel
        self._draw_preview_panel(right_panel_rect)

        pygame.display.flip()

    def _draw_save_list(self, panel_rect: pygame.Rect) -> None:
        """Draw the scrollable save list."""
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
            text_font = get_font(20)
            text_surface = text_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(
                midleft=(item_rect.x + 10, item_rect.centery)
            )

            # Clip text if too long
            if text_rect.width > item_rect.width - 20:
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
        # Get currently selected/hovered save
        active_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        if active_index < 0 or active_index >= len(self.save_files):
            # Draw placeholder
            font = get_font(28)
            text = font.render("Select a save to preview", True, (150, 150, 150))
            text_rect = text.get_rect(center=panel_rect.center)
            self.screen.blit(text, text_rect)
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
        info_y = preview_y + preview_size + 20
        info_x = panel_rect.x + 20
        line_spacing = 26

        # Title: Map Name
        title_font = get_font(24)
        map_name = metadata.get('map_name', 'Unknown Map')
        title_surface = title_font.render(map_name, True, self.title_color)
        self.screen.blit(title_surface, (info_x, info_y))
        info_y += 32

        # Info lines
        info_font = get_font(20)
        label_color = (180, 180, 180)
        value_color = (255, 255, 255)

        # Turn info
        turn_number = metadata.get('turn_number', 0)
        current_player = metadata.get('current_player', 1)
        turn_label = info_font.render("Turn: ", True, label_color)
        turn_value = info_font.render(f"{turn_number}", True, value_color)
        current_label = info_font.render(f"  (P{current_player}'s turn)", True, PLAYER_COLORS.get(current_player, value_color))
        self.screen.blit(turn_label, (info_x, info_y))
        self.screen.blit(turn_value, (info_x + turn_label.get_width(), info_y))
        self.screen.blit(current_label, (info_x + turn_label.get_width() + turn_value.get_width(), info_y))
        info_y += line_spacing

        # Number of players
        num_players = metadata.get('num_players', 2)
        players_label = info_font.render("Players: ", True, label_color)
        players_value = info_font.render(str(num_players), True, value_color)
        self.screen.blit(players_label, (info_x, info_y))
        self.screen.blit(players_value, (info_x + players_label.get_width(), info_y))
        info_y += line_spacing

        # Draw separator
        info_y += 5
        pygame.draw.line(self.screen, (80, 80, 100),
                        (info_x, info_y), (panel_rect.right - 20, info_y), 1)
        info_y += 10

        # Player info with units and gold
        player_gold = metadata.get('player_gold', {})
        unit_counts = metadata.get('unit_counts', {})
        unit_types_per_player = metadata.get('unit_types_per_player', {})

        for player_num in range(1, num_players + 1):
            player_color = PLAYER_COLORS.get(player_num, (200, 200, 200))

            # Player header
            player_name = metadata.get(f'player{player_num}', f'Player {player_num}')
            p_label = info_font.render(f"P{player_num}: ", True, player_color)
            p_name = info_font.render(player_name, True, value_color)
            self.screen.blit(p_label, (info_x, info_y))
            self.screen.blit(p_name, (info_x + p_label.get_width(), info_y))
            info_y += line_spacing

            # Gold
            gold = player_gold.get(player_num, 0)
            gold_label = info_font.render("  Gold: ", True, label_color)
            gold_value = info_font.render(f"{gold}", True, (255, 215, 0))  # Gold color
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
                small_font = get_font(16)
                breakdown_parts = []
                for utype, count in sorted(unit_types.items()):
                    breakdown_parts.append(f"{count}{utype}")
                breakdown_text = " ".join(breakdown_parts)
                breakdown_surf = small_font.render(
                    f"({breakdown_text})", True, (140, 140, 140))
                self.screen.blit(breakdown_surf, (bx, info_y + 2))

            info_y += line_spacing

        # Draw separator before date
        info_y += 5
        pygame.draw.line(self.screen, (80, 80, 100),
                        (info_x, info_y), (panel_rect.right - 20, info_y), 1)
        info_y += 10

        # Date
        date = metadata.get('date', 'Unknown')
        date_label = info_font.render("Saved: ", True, label_color)
        date_value = info_font.render(date, True, value_color)
        self.screen.blit(date_label, (info_x, info_y))
        self.screen.blit(date_value, (info_x + date_label.get_width(), info_y))

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run load game menu.

        Returns:
            Dict with loaded save data, or None if cancelled
        """
        selected_path = super().run()

        if not selected_path:
            return None

        # Load the actual save data from the file
        try:
            with open(selected_path, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            return save_data
        except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
            print(f"Error loading save file: {e}")
            return None
