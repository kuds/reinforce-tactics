"""Menu for selecting a replay to watch."""
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font


class ReplaySelectionMenu(Menu):
    """Menu for selecting a replay to watch with game info preview."""

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
        self.replay_info_cache: Dict[str, Dict[str, Any]] = {}
        self._load_replays()
        self._setup_options()

    def _load_replays(self) -> None:
        """Load available replay files and their metadata."""
        if os.path.exists(self.replays_dir):
            self.replay_files = [
                f for f in os.listdir(self.replays_dir)
                if f.endswith('.json')
            ]
            # Sort by modification time (newest first)
            self.replay_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.replays_dir, f)),
                reverse=True
            )

            # Pre-load replay info for all files
            for replay_file in self.replay_files:
                filepath = os.path.join(self.replays_dir, replay_file)
                self.replay_info_cache[filepath] = self._load_replay_info(filepath)

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
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    date_str = timestamp[:16] if len(timestamp) >= 16 else timestamp

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
            # No replays available
            lang = get_language()
            self.add_option(lang.get('replay.no_replays', 'No replays found'), lambda: None)
        else:
            for replay_file in self.replay_files:
                filepath = os.path.join(self.replays_dir, replay_file)
                info = self.replay_info_cache.get(filepath, {})

                # Create a more informative display name
                p1 = info.get('p1_name', 'P1')[:12]
                p2 = info.get('p2_name', 'P2')[:12]
                winner = info.get('winner')
                turns = info.get('total_turns', '?')
                date = info.get('date', '')[:10]  # Just the date part

                # Winner indicator
                winner_str = ""
                if winner == 1:
                    winner_str = f" [P1 wins]"
                elif winner == 2:
                    winner_str = f" [P2 wins]"

                display_name = f"{p1} vs {p2} ({turns}T){winner_str}"
                if date:
                    display_name = f"{date}: {display_name}"

                self.add_option(display_name, lambda p=filepath: p)

        self.add_option(get_language().get('common.back', 'Back'), lambda: None)

    def draw(self) -> None:
        """Draw the menu with replay info panel."""
        super().draw()

        # Draw info panel for hovered/selected replay
        self._draw_info_panel()

        pygame.display.flip()

    def _draw_info_panel(self) -> None:
        """Draw detailed info panel for selected replay."""
        # Determine which replay to show info for
        display_index = self.hover_index if self.hover_index >= 0 else self.selected_index

        # Make sure we have a valid index that corresponds to a replay file
        if display_index < 0 or display_index >= len(self.replay_files):
            return

        filepath = os.path.join(self.replays_dir, self.replay_files[display_index])
        info = self.replay_info_cache.get(filepath)
        if not info:
            return

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Info panel dimensions
        panel_width = 280
        panel_height = 140
        panel_x = screen_width - panel_width - 20
        panel_y = screen_height - panel_height - 20

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (45, 45, 60), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100, 100, 150), panel_rect, 2, border_radius=8)

        # Draw info text
        font = get_font(18)
        small_font = get_font(16)
        y_offset = panel_y + 12

        # Title
        title_text = "Replay Info"
        title_surface = font.render(title_text, True, (200, 200, 255))
        self.screen.blit(title_surface, (panel_x + 10, y_offset))
        y_offset += 25

        # Player names
        p1_color = (100, 255, 100) if info.get('winner') == 1 else (255, 255, 255)
        p2_color = (100, 255, 100) if info.get('winner') == 2 else (255, 255, 255)

        p1_text = f"P1: {info.get('p1_name', 'Unknown')[:20]}"
        p1_surface = small_font.render(p1_text, True, p1_color)
        self.screen.blit(p1_surface, (panel_x + 10, y_offset))
        y_offset += 20

        p2_text = f"P2: {info.get('p2_name', 'Unknown')[:20]}"
        p2_surface = small_font.render(p2_text, True, p2_color)
        self.screen.blit(p2_surface, (panel_x + 10, y_offset))
        y_offset += 20

        # Game stats
        turns_text = f"Turns: {info.get('total_turns', '?')}"
        turns_surface = small_font.render(turns_text, True, (200, 200, 200))
        self.screen.blit(turns_surface, (panel_x + 10, y_offset))

        actions_text = f"Actions: {info.get('num_actions', 0)}"
        actions_surface = small_font.render(actions_text, True, (200, 200, 200))
        self.screen.blit(actions_surface, (panel_x + 140, y_offset))
        y_offset += 20

        # Date
        if info.get('date'):
            date_text = f"Date: {info.get('date')}"
            date_surface = small_font.render(date_text, True, (180, 180, 180))
            self.screen.blit(date_surface, (panel_x + 10, y_offset))

    def run(self) -> Optional[str]:
        """
        Run replay selection menu.

        Returns:
            Path to selected replay file, or None if cancelled
        """
        return super().run()
