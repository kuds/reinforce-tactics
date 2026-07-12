"""
Replay player for watching recorded games
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pygame

from reinforcetactics.constants import MIN_MAP_SIZE, PLAYER_COLORS
from reinforcetactics.ui import theme
from reinforcetactics.ui.icons import (
    get_arrow_left_icon,
    get_arrow_right_icon,
    get_pause_icon,
    get_play_icon,
    get_restart_icon,
    get_skip_back_icon,
    get_skip_forward_icon,
    get_x_icon,
)
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.replay_actions import execute_replay_action, get_schema_version

# Default border size for replay padding (same as UI)
REPLAY_BORDER_SIZE = 2


class ReplayPlayer:
    """Plays back recorded game replays."""

    def __init__(self, replay_data, initial_map_data):
        """
        Initialize replay player.

        Args:
            replay_data: Dictionary with replay information
            initial_map_data: Initial map data for creating game state
        """
        self.replay_data = replay_data
        self.actions = replay_data.get("actions", [])
        self.game_info = replay_data.get("game_info", {})

        # Pad the map for UI display (same as gameplay)
        padded_map, offset_x, offset_y = self._pad_map_for_replay(initial_map_data)
        self.initial_map_data = padded_map
        self.padding_offset_x = offset_x
        self.padding_offset_y = offset_y

        self.current_action_index = 0
        self.paused = True
        # True while the progress bar is being dragged (scrubbing)
        self.scrubbing = False

        # v2 replays carry per-action outcome data so we can apply
        # recorded effects directly instead of re-calling the engine.
        # v1 (legacy) replays fall through to the recompute path.
        self.schema_version = get_schema_version(self.game_info)

        # Actions record 0-based turn numbers; the in-game HUD shows 1-based
        # ("Turn 1" on the opening move), so the info panel displays 1-based
        # too. The total is derived from the recorded actions so both
        # numbers share the same convention.
        if self.actions:
            self._total_turns = max(a.get("turn", 0) for a in self.actions) + 1
        else:
            self._total_turns = self.game_info.get("total_turns", "?")
        self.playback_speed = 1.0  # 1x speed
        self.last_action_time = time.time()

        # Video export state (True while an export is in progress)
        self.exporting = False

        # On-screen notification state
        self.notification_text = ""
        self.notification_time = 0
        self.notification_duration = 2.0  # seconds

        # Current action description for display
        self.current_action_description = ""

        # Create initial game state with padded map
        from reinforcetactics.core.game_state import GameState

        self.game_state = GameState(padded_map, num_players=self.game_info.get("num_players", 2))

        # Create renderer (replay mode hides End Turn and Resign buttons)
        from reinforcetactics.ui.renderer import Renderer

        self.renderer = Renderer(self.game_state, replay_mode=True)

        # UI elements
        self.setup_ui()

    def _pad_map_for_replay(self, map_data):
        """
        Pad the map for replay display (same as UI gameplay).

        Applies minimum size padding and water border, matching the
        behavior of FileIO.load_map(for_ui=True).

        Args:
            map_data: Initial map data (DataFrame or array-like)

        Returns:
            Tuple of (padded_map_dataframe, offset_x, offset_y)
        """
        # Convert to DataFrame if needed
        if isinstance(map_data, pd.DataFrame):
            df = map_data.copy()
        elif isinstance(map_data, np.ndarray):
            df = pd.DataFrame(map_data)
        else:
            df = pd.DataFrame(map_data)

        height, width = df.shape
        offset_x = 0
        offset_y = 0

        # Apply minimum size padding if needed
        if height < MIN_MAP_SIZE or width < MIN_MAP_SIZE:
            min_height = max(height, MIN_MAP_SIZE)
            min_width = max(width, MIN_MAP_SIZE)

            pad_width = max(0, min_width - width)
            pad_height = max(0, min_height - height)

            if pad_width > 0 or pad_height > 0:
                padded = pd.DataFrame(np.full((min_height, min_width), "o", dtype=object))
                start_y = pad_height // 2
                start_x = pad_width // 2
                end_y = start_y + height
                end_x = start_x + width
                padded.iloc[start_y:end_y, start_x:end_x] = df.values
                df = padded
                offset_x = start_x
                offset_y = start_y

        # Add water border
        border_size = REPLAY_BORDER_SIZE
        height, width = df.shape
        new_height = height + 2 * border_size
        new_width = width + 2 * border_size

        bordered = pd.DataFrame(np.full((new_height, new_width), "o", dtype=object))
        bordered.iloc[border_size : border_size + height, border_size : border_size + width] = df.values

        # Update offsets to include border
        offset_x += border_size
        offset_y += border_size

        return bordered, offset_x, offset_y

    def _translate_coords(self, x, y):
        """
        Translate original coordinates to padded coordinates.

        Args:
            x: Original x coordinate
            y: Original y coordinate

        Returns:
            Tuple of (padded_x, padded_y)
        """
        return (x + self.padding_offset_x, y + self.padding_offset_y)

    def setup_ui(self):
        """Setup UI elements for replay controls."""
        from reinforcetactics.constants import TILE_SIZE

        screen_width = self.game_state.grid.width * TILE_SIZE
        screen_height = self.game_state.grid.height * TILE_SIZE

        # Control panel at bottom
        self.control_y = screen_height - 60

        # Layout buttons with proper spacing
        btn_y = self.control_y + 10
        btn_height = 40

        # Row 1: Playback controls
        x_pos = 10
        self.step_back_button = pygame.Rect(x_pos, btn_y, 40, btn_height)
        x_pos += 50
        self.play_pause_button = pygame.Rect(x_pos, btn_y, 60, btn_height)
        x_pos += 70
        self.step_forward_button = pygame.Rect(x_pos, btn_y, 40, btn_height)
        x_pos += 50
        self.restart_button = pygame.Rect(x_pos, btn_y, 40, btn_height)

        # Skip turn buttons
        x_pos += 50
        self.prev_turn_button = pygame.Rect(x_pos, btn_y, 40, btn_height)
        x_pos += 50
        self.next_turn_button = pygame.Rect(x_pos, btn_y, 40, btn_height)

        # Speed controls
        x_pos += 60
        self.speed_down_button = pygame.Rect(x_pos, btn_y, 35, btn_height)
        x_pos += 40
        self.speed_display_x = x_pos + 25  # Center of speed display area
        x_pos += 55
        self.speed_up_button = pygame.Rect(x_pos, btn_y, 35, btn_height)

        # Save video button
        x_pos += 50
        self.save_video_button = pygame.Rect(x_pos, btn_y, 80, btn_height)

        # Exit button at the right
        self.exit_button = pygame.Rect(screen_width - 70, btn_y, 60, btn_height)

        # Progress bar - positioned after the buttons
        progress_start = x_pos + 90
        progress_end = screen_width - 80
        progress_width = max(progress_end - progress_start, 100)
        self.progress_bar_rect = pygame.Rect(progress_start, self.control_y + 20, progress_width, 20)

        # Hover tooltips for the icon-only controls (label + shortcut)
        self._control_tooltips = [
            (self.step_back_button, "Step back (Left)"),
            (self.play_pause_button, "Play / Pause (Space)"),
            (self.step_forward_button, "Step forward (Right)"),
            (self.restart_button, "Restart (R)"),
            (self.prev_turn_button, "Previous turn (PgUp)"),
            (self.next_turn_button, "Next turn (PgDn)"),
            (self.speed_down_button, "Slower (-)"),
            (self.speed_up_button, "Faster (+)"),
            (self.save_video_button, "Export MP4 (V)"),
            (self.exit_button, "Exit (Esc)"),
        ]

    def execute_action(self, action):
        """
        Execute a single replay action.

        Coordinates in the action are in original (unpadded) space and are
        translated to padded coordinates before execution. Dispatch lives
        in :func:`reinforcetactics.utils.replay_actions.execute_replay_action`,
        shared with the headless video recorder so the two playback paths
        can't drift apart.

        Args:
            action: Action dictionary to execute
        """
        execute_replay_action(self.game_state, action, self._translate_coords, self.schema_version)

    def update(self):
        """Update replay playback."""
        if not self.paused and self.current_action_index < len(self.actions):
            current_time = time.time()
            time_per_action = 0.5 / self.playback_speed  # 0.5 seconds per action at 1x speed

            if current_time - self.last_action_time >= time_per_action:
                action = self.actions[self.current_action_index]
                self.current_action_description = self._get_action_description(action)
                self.execute_action(action)
                self.current_action_index += 1
                self.last_action_time = current_time

                if self.current_action_index >= len(self.actions):
                    self.paused = True
                    self.show_notification("Replay finished!")

    def restart(self):
        """Restart the replay from the beginning."""
        self._reset_game_state()
        self.current_action_index = 0
        self.paused = True
        self.current_action_description = ""

        self.show_notification("Replay restarted")

    def _reset_game_state(self):
        """Rebuild the game state at action 0, reusing the live renderer.

        The map (and therefore the window size) never changes for a given
        replay, so the existing renderer is retargeted rather than
        recreated -- constructing a Renderer re-runs display set_mode and
        reloads every sprite from disk, which made each backward step,
        turn skip, and seek visibly hitch.
        """
        from reinforcetactics.core.game_state import GameState

        self.game_state = GameState(self.initial_map_data, num_players=self.game_info.get("num_players", 2))
        self.renderer.game_state = self.game_state

    def toggle_pause(self):
        """Toggle pause/play."""
        self.paused = not self.paused
        if not self.paused:
            self.last_action_time = time.time()

    def change_speed(self, delta):
        """Change playback speed."""
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
        try:
            current_idx = speeds.index(self.playback_speed)
            new_idx = max(0, min(len(speeds) - 1, current_idx + delta))
            self.playback_speed = speeds[new_idx]
            self.show_notification(f"Speed: {self.playback_speed}x")
        except ValueError:
            self.playback_speed = 1.0

    def show_notification(self, text):
        """Show an on-screen notification."""
        self.notification_text = text
        self.notification_time = time.time()

    def step_forward(self):
        """Step forward one action."""
        if self.current_action_index < len(self.actions):
            action = self.actions[self.current_action_index]
            self.execute_action(action)
            self.current_action_index += 1
            self.paused = True
            if self.current_action_index >= len(self.actions):
                self.show_notification("Replay finished!")

    def step_backward(self):
        """Step backward one action by replaying from start."""
        if self.current_action_index > 0:
            target = self.current_action_index - 1
            self._replay_to_action(target)
            self.paused = True

    def _replay_to_action(self, target_index):
        """Replay from beginning to target action index."""
        self._reset_game_state()

        # Replay to target
        target_index = max(0, min(target_index, len(self.actions)))
        for i in range(target_index):
            self.execute_action(self.actions[i])
        self.current_action_index = target_index

        # Refresh playback state so the seek doesn't leave a stale action
        # description on screen or an old timer that fires immediately.
        if target_index > 0:
            self.current_action_description = self._get_action_description(self.actions[target_index - 1])
        else:
            self.current_action_description = ""
        self.last_action_time = time.time()

    def skip_to_next_turn(self):
        """Skip to the next turn's first action."""
        if self.current_action_index >= len(self.actions):
            return

        current_turn = self._get_action_turn(self.current_action_index)

        # Find next turn's first action
        for i in range(self.current_action_index + 1, len(self.actions)):
            if self._get_action_turn(i) > current_turn:
                self._replay_to_action(i)
                self.show_notification(f"Turn {self._get_action_turn(i) + 1}")
                return

        # No next turn found, go to end
        self._replay_to_action(len(self.actions))
        self.show_notification("Replay finished!")

    def skip_to_prev_turn(self):
        """Skip to the start of the current turn, or the previous turn.

        Media-player semantics: when playback is mid-turn, the first press
        rewinds to the start of the current turn; pressing again at a turn
        boundary goes back one full turn. Turn numbers are 0-based, so the
        first turn is reachable (the old ``max(1, ...)`` floor made it
        impossible to rewind into turn 0 and could even jump forward).
        """
        if self.current_action_index <= 0:
            return

        current_turn = self._get_action_turn(max(0, self.current_action_index - 1))

        # Index of the current turn's first action
        first_of_current = 0
        for i in range(len(self.actions)):
            if self._get_action_turn(i) >= current_turn:
                first_of_current = i
                break

        if self.current_action_index > first_of_current:
            target_turn = current_turn
        else:
            target_turn = max(0, current_turn - 1)

        # Find the first action of the target turn
        for i in range(len(self.actions)):
            if self._get_action_turn(i) >= target_turn:
                self._replay_to_action(i)
                self.show_notification(f"Turn {self._get_action_turn(i) + 1}")
                return

        # Fallback to beginning
        self._replay_to_action(0)
        self.show_notification("Turn 1")

    def _seek_to_mouse(self, mouse_pos):
        """Seek playback to the progress-bar position under the mouse."""
        if not self.actions or self.progress_bar_rect.width <= 0:
            return
        relative_x = mouse_pos[0] - self.progress_bar_rect.x
        progress = max(0.0, min(1.0, relative_x / self.progress_bar_rect.width))
        target_action = int(progress * len(self.actions))
        if target_action != self.current_action_index:
            self._replay_to_action(target_action)

    def _get_action_turn(self, action_index):
        """Get the turn number for an action index."""
        if action_index < 0 or action_index >= len(self.actions):
            return 0
        return self.actions[action_index].get("turn", 0)

    def _get_action_description(self, action):
        """Get a human-readable description of an action."""
        action_type = action.get("type", "unknown")
        player = action.get("player", "?")

        if action_type == "create_unit":
            unit_type = action.get("unit_type", "unit")
            return f"P{player} creates {unit_type}"
        elif action_type == "move":
            return f"P{player} moves unit"
        elif action_type == "attack":
            return f"P{player} attacks"
        elif action_type == "paralyze":
            return f"P{player} paralyzes target"
        elif action_type == "heal":
            return f"P{player} heals unit"
        elif action_type == "cure":
            return f"P{player} cures unit"
        elif action_type == "haste":
            return f"P{player} hastes ally"
        elif action_type == "defence_buff":
            return f"P{player} defence buffs ally"
        elif action_type == "attack_buff":
            return f"P{player} attack buffs ally"
        elif action_type == "seize":
            return f"P{player} seizes structure"
        elif action_type == "resign":
            return f"P{player} resigns"
        elif action_type == "end_turn":
            return f"P{player} ends turn"
        else:
            return f"P{player}: {action_type}"

    def export_video(self):
        """Export the full replay to an MP4 file via headless rendering.

        Renders the entire action history (not just what has been watched)
        through :func:`reinforcetactics.utils.video.record_replay_to_video`,
        streaming frames straight to the encoder so memory stays bounded.
        An on-screen progress bar tracks the export; playback resumes when
        it finishes.

        Returns:
            Path to the saved video file, or None on failure.
        """
        if self.exporting:
            return None

        from reinforcetactics.utils.video import record_replay_to_video

        self.exporting = True
        was_paused = self.paused
        self.paused = True

        try:
            videos_dir = Path("videos")
            videos_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = videos_dir / f"replay_{timestamp}.mp4"

            last_drawn = 0.0

            def _on_progress(done, total):
                # Keep the window responsive and show progress, but don't
                # redraw more than ~10x per second.
                nonlocal last_drawn
                now = time.time()
                if now - last_drawn < 0.1 and done < total:
                    return
                last_drawn = now
                pygame.event.pump()
                self._draw_export_progress(done, total)
                pygame.display.flip()

            path = record_replay_to_video(
                self.replay_data,
                str(output_path),
                fps=4,
                # None = follow settings.json, so the export matches the
                # sprite style the viewer is currently looking at.
                use_pixel_art=None,
                progress_callback=_on_progress,
            )
            self.show_notification(f"Video saved: {path}")
            return path

        except Exception as e:
            self.show_notification(f"Export failed: {e}")
            return None

        finally:
            self.exporting = False
            self.paused = was_paused
            self.last_action_time = time.time()

    def _draw_export_progress(self, done, total):
        """Draw a modal progress bar while a video export runs."""
        screen = self.renderer.screen
        width, height = screen.get_width(), screen.get_height()

        bar_width = max(200, width // 2)
        bar_height = 22
        bar_x = (width - bar_width) // 2
        bar_y = height // 2

        panel = pygame.Rect(bar_x - 20, bar_y - 50, bar_width + 40, 100)
        pygame.draw.rect(screen, theme.PANEL_BG, panel, border_radius=theme.BORDER_RADIUS)
        pygame.draw.rect(screen, theme.PANEL_BORDER, panel, 2, border_radius=theme.BORDER_RADIUS)

        label = get_font(20).render("Exporting video...", True, theme.TEXT)
        screen.blit(label, label.get_rect(centerx=width // 2, y=bar_y - 36))

        bar_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(screen, theme.REPLAY_PROGRESS_BG, bar_rect, border_radius=4)
        if total > 0 and done > 0:
            fill = pygame.Rect(bar_x, bar_y, int(bar_width * done / total), bar_height)
            pygame.draw.rect(screen, theme.REPLAY_PROGRESS_FILL, fill, border_radius=4)
        pygame.draw.rect(screen, theme.REPLAY_BTN_BORDER, bar_rect, 2, border_radius=4)

    def run(self):
        """Run the replay player."""
        clock = pygame.time.Clock()
        running = True

        # Show initial notification with the full matchup (2-4 players)
        player_configs = self.game_info.get("player_configs", [])
        if player_configs:
            num_players = self.game_info.get("num_players", 2)
            names = [player_configs[i].get("name", f"P{i + 1}") for i in range(min(num_players, len(player_configs)))]
            self.show_notification(" vs ".join(names))

        while running:
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.toggle_pause()
                    elif event.key == pygame.K_r:
                        self.restart()
                    elif event.key == pygame.K_v:
                        self.export_video()
                    elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                        self.change_speed(1)
                    elif event.key == pygame.K_MINUS:
                        self.change_speed(-1)
                    # New keyboard shortcuts
                    elif event.key == pygame.K_LEFT:
                        self.step_backward()
                    elif event.key == pygame.K_RIGHT:
                        self.step_forward()
                    elif event.key == pygame.K_PAGEUP:
                        self.skip_to_prev_turn()
                    elif event.key == pygame.K_PAGEDOWN:
                        self.skip_to_next_turn()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.play_pause_button.collidepoint(mouse_pos):
                            self.toggle_pause()
                        elif self.restart_button.collidepoint(mouse_pos):
                            self.restart()
                        elif self.step_back_button.collidepoint(mouse_pos):
                            self.step_backward()
                        elif self.step_forward_button.collidepoint(mouse_pos):
                            self.step_forward()
                        elif self.prev_turn_button.collidepoint(mouse_pos):
                            self.skip_to_prev_turn()
                        elif self.next_turn_button.collidepoint(mouse_pos):
                            self.skip_to_next_turn()
                        elif self.speed_up_button.collidepoint(mouse_pos):
                            self.change_speed(1)
                        elif self.speed_down_button.collidepoint(mouse_pos):
                            self.change_speed(-1)
                        elif self.save_video_button.collidepoint(mouse_pos):
                            self.export_video()
                        elif self.exit_button.collidepoint(mouse_pos):
                            running = False
                        elif self.progress_bar_rect.collidepoint(mouse_pos):
                            # Seek, then keep scrubbing while the button is held
                            self.scrubbing = True
                            self._seek_to_mouse(mouse_pos)

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.scrubbing = False

                elif event.type == pygame.MOUSEMOTION and self.scrubbing:
                    self._seek_to_mouse(event.pos)

            # Update replay
            self.update()

            # Draw
            self.draw(mouse_pos)

            pygame.display.flip()
            clock.tick(60)

    def draw(self, mouse_pos):
        """Draw the replay player."""
        # Draw game state
        self.renderer.render()

        screen = self.renderer.screen
        screen_width = screen.get_width()

        # Draw info panel at top
        self._draw_info_panel(screen, screen_width)

        # Draw control panel background
        panel_rect = pygame.Rect(0, self.control_y, screen_width, 60)
        pygame.draw.rect(screen, theme.PANEL_BG, panel_rect)
        pygame.draw.line(screen, theme.REPLAY_PANEL_DIVIDER, (0, self.control_y), (screen_width, self.control_y), 2)

        # Draw buttons
        font = get_font(24)
        small_font = get_font(20)
        icon_size = 24
        icon_color = theme.TEXT

        # Step back button
        step_back_icon = get_arrow_left_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.step_back_button, step_back_icon, mouse_pos, theme.REPLAY_BTN_STEP)

        # Play/Pause button
        if self.paused:
            play_pause_icon = get_play_icon(size=icon_size, color=icon_color)
        else:
            play_pause_icon = get_pause_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.play_pause_button, play_pause_icon, mouse_pos, theme.REPLAY_BTN_PLAY)

        # Step forward button
        step_forward_icon = get_arrow_right_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.step_forward_button, step_forward_icon, mouse_pos, theme.REPLAY_BTN_STEP)

        # Restart button
        restart_icon = get_restart_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.restart_button, restart_icon, mouse_pos, theme.REPLAY_BTN_RESTART)

        # Skip to previous turn button
        skip_back_icon = get_skip_back_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.prev_turn_button, skip_back_icon, mouse_pos, theme.REPLAY_BTN_TURN_SKIP)

        # Skip to next turn button
        skip_forward_icon = get_skip_forward_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.next_turn_button, skip_forward_icon, mouse_pos, theme.REPLAY_BTN_TURN_SKIP)

        # Speed down button
        self._draw_button(self.speed_down_button, "-", mouse_pos, theme.REPLAY_BTN_SPEED, font)

        # Speed display (between speed buttons)
        speed_text = f"{self.playback_speed:g}x"
        speed_surface = small_font.render(speed_text, True, theme.REPLAY_SPEED_TEXT)
        speed_rect = speed_surface.get_rect(center=(self.speed_display_x, self.control_y + 30))
        screen.blit(speed_surface, speed_rect)

        # Speed up button
        self._draw_button(self.speed_up_button, "+", mouse_pos, theme.REPLAY_BTN_SPEED, font)

        # Export video button
        export_text = "..." if self.exporting else "Export"
        export_color = theme.REPLAY_BTN_EXPORT_ACTIVE if self.exporting else theme.REPLAY_BTN_EXPORT
        self._draw_button(self.save_video_button, export_text, mouse_pos, export_color, small_font)

        # Progress bar (only if there's space)
        if self.progress_bar_rect.width > 50:
            pygame.draw.rect(screen, theme.REPLAY_PROGRESS_BG, self.progress_bar_rect, border_radius=4)
            if len(self.actions) > 0:
                progress = self.current_action_index / len(self.actions)
                progress_width = int(self.progress_bar_rect.width * progress)
                if progress_width > 0:
                    progress_rect = pygame.Rect(
                        self.progress_bar_rect.x, self.progress_bar_rect.y, progress_width, self.progress_bar_rect.height
                    )
                    pygame.draw.rect(screen, theme.REPLAY_PROGRESS_FILL, progress_rect, border_radius=4)
            pygame.draw.rect(screen, theme.REPLAY_BTN_BORDER, self.progress_bar_rect, 2, border_radius=4)

        # Exit button
        exit_icon = get_x_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.exit_button, exit_icon, mouse_pos, theme.REPLAY_BTN_EXIT)

        # Unit stats on hover, same as in-game (only while the cursor is
        # over the board, not the info/control panels)
        info_panel_top = self.control_y - 40
        if mouse_pos[1] < info_panel_top:
            self.renderer.draw_unit_tooltip(mouse_pos)

        # Label + shortcut tooltip for the hovered control
        self._draw_control_tooltip(mouse_pos)

        # Draw on-screen notification
        self._draw_notification(screen, screen_width)

    def _draw_control_tooltip(self, mouse_pos):
        """Draw a small label/shortcut tooltip above the hovered control."""
        for rect, label in self._control_tooltips:
            if not rect.collidepoint(mouse_pos):
                continue
            screen = self.renderer.screen
            text_surface = get_font(16).render(label, True, theme.TEXT)
            padding = 6
            bg_rect = pygame.Rect(0, 0, text_surface.get_width() + 2 * padding, text_surface.get_height() + 2 * padding)
            bg_rect.midbottom = (rect.centerx, rect.top - 6)
            bg_rect.x = max(4, min(bg_rect.x, screen.get_width() - bg_rect.width - 4))
            pygame.draw.rect(screen, theme.TOOLTIP_BG, bg_rect, border_radius=theme.BORDER_RADIUS_SMALL)
            pygame.draw.rect(screen, theme.PANEL_BORDER, bg_rect, 1, border_radius=theme.BORDER_RADIUS_SMALL)
            screen.blit(text_surface, (bg_rect.x + padding, bg_rect.y + padding))
            return

    def _draw_info_panel(self, screen, screen_width):
        """Draw the info panel above the control panel at the bottom."""
        # Panel height and position (above the control panel)
        panel_height = 40
        panel_y = self.control_y - panel_height

        # Semi-transparent background
        info_panel = pygame.Surface((screen_width, panel_height), pygame.SRCALPHA)
        info_panel.fill((0, 0, 0, 150))
        screen.blit(info_panel, (0, panel_y))

        font = get_font(18)
        small_font = get_font(16)

        # Get current turn
        current_turn = 0
        if self.current_action_index > 0 and self.current_action_index <= len(self.actions):
            current_turn = self._get_action_turn(self.current_action_index - 1)
        elif self.current_action_index == 0 and len(self.actions) > 0:
            current_turn = self._get_action_turn(0)

        total_turns = self._total_turns

        # Layout: Turn/Action on left, Action description in center, Player info on right
        # All on a single row for compact display

        # Turn and action info on left
        turn_text = f"Turn {current_turn + 1}/{total_turns}"
        action_text = f"Action {self.current_action_index}/{len(self.actions)}"
        combined_left = f"{turn_text}  |  {action_text}"
        left_surface = small_font.render(combined_left, True, theme.TEXT_SECONDARY)
        screen.blit(left_surface, (10, panel_y + (panel_height - left_surface.get_height()) // 2))

        # Current action description in center
        if self.current_action_description:
            desc_surface = font.render(self.current_action_description, True, theme.SELECTED)
            desc_rect = desc_surface.get_rect(center=(screen_width // 2, panel_y + panel_height // 2))
            screen.blit(desc_surface, desc_rect)

        # Player info on right
        player_configs = self.game_info.get("player_configs", [])
        winner = self.game_info.get("winner", None)
        num_players = self.game_info.get("num_players", 2)

        if player_configs:
            # Build player name segments with their colors
            segments = []
            for p_idx in range(min(num_players, len(player_configs))):
                p_num = p_idx + 1
                p_name = player_configs[p_idx].get("name", f"P{p_num}")[:12]
                if winner == p_num:
                    p_color = theme.STATUS_VALID
                else:
                    p_color = PLAYER_COLORS.get(p_num, (200, 200, 200))
                segments.append((f"{p_name} (P{p_num})", p_color))

            # Draw segments right-aligned with " vs " separator
            right_x = screen_width - 10
            text_y = panel_y + (panel_height - small_font.get_height()) // 2
            for i in range(len(segments) - 1, -1, -1):
                text, color = segments[i]
                text_surface = small_font.render(text, True, color)
                right_x -= text_surface.get_width()
                screen.blit(text_surface, (right_x, text_y))
                if i > 0:
                    vs_surface = small_font.render(" vs ", True, theme.TEXT_SECONDARY)
                    right_x -= vs_surface.get_width()
                    screen.blit(vs_surface, (right_x, text_y))

    def _draw_notification(self, screen, screen_width):
        """Draw on-screen notification if active."""
        if self.notification_text:
            elapsed = time.time() - self.notification_time
            if elapsed < self.notification_duration:
                # Fade out effect
                alpha = 255
                if elapsed > self.notification_duration - 0.5:
                    alpha = int(255 * (self.notification_duration - elapsed) / 0.5)

                font = get_font(24)
                text_surface = font.render(self.notification_text, True, theme.TEXT)

                # Create background
                padding = 15
                bg_width = text_surface.get_width() + 2 * padding
                bg_height = text_surface.get_height() + 2 * padding
                bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                bg_surface.fill((0, 0, 0, min(180, alpha)))

                # Position at center (above info panel which is above control panel)
                info_panel_height = 40
                x = (screen_width - bg_width) // 2
                y = self.control_y - info_panel_height - bg_height - 15

                screen.blit(bg_surface, (x, y))

                # Draw text with alpha
                text_surface.set_alpha(alpha)
                screen.blit(text_surface, (x + padding, y + padding))
            else:
                self.notification_text = ""

    def _draw_button(self, rect, text, mouse_pos, color, font):
        """Draw a button with text and rounded corners."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 40, 255) for c in color) if is_hover else color
        border_color = theme.REPLAY_BTN_BORDER_HOVER if is_hover else theme.REPLAY_BTN_BORDER

        pygame.draw.rect(self.renderer.screen, button_color, rect, border_radius=theme.BORDER_RADIUS_SMALL)
        pygame.draw.rect(self.renderer.screen, border_color, rect, width=2, border_radius=theme.BORDER_RADIUS_SMALL)

        text_surface = font.render(text, True, theme.TEXT)
        text_rect = text_surface.get_rect(center=rect.center)
        self.renderer.screen.blit(text_surface, text_rect)

    def _draw_icon_button(self, rect, icon, mouse_pos, color):
        """Draw a button with an icon surface and rounded corners."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 40, 255) for c in color) if is_hover else color
        border_color = theme.REPLAY_BTN_BORDER_HOVER if is_hover else theme.REPLAY_BTN_BORDER

        pygame.draw.rect(self.renderer.screen, button_color, rect, border_radius=theme.BORDER_RADIUS_SMALL)
        pygame.draw.rect(self.renderer.screen, border_color, rect, width=2, border_radius=theme.BORDER_RADIUS_SMALL)

        icon_rect = icon.get_rect(center=rect.center)
        self.renderer.screen.blit(icon, icon_rect)
