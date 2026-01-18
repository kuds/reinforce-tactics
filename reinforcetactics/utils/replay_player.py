"""
Replay player for watching recorded games
"""
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pygame

from reinforcetactics.utils.fonts import get_font
from reinforcetactics.constants import MIN_MAP_SIZE
from reinforcetactics.ui.icons import (
    get_play_icon, get_pause_icon, get_arrow_left_icon, get_arrow_right_icon,
    get_restart_icon, get_skip_back_icon, get_skip_forward_icon, get_x_icon
)

# Optional OpenCV import for video export
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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
        self.actions = replay_data.get('actions', [])
        self.game_info = replay_data.get('game_info', {})

        # Pad the map for UI display (same as gameplay)
        padded_map, offset_x, offset_y = self._pad_map_for_replay(initial_map_data)
        self.initial_map_data = padded_map
        self.padding_offset_x = offset_x
        self.padding_offset_y = offset_y

        self.current_action_index = 0
        self.playing = False
        self.paused = True
        self.playback_speed = 1.0  # 1x speed
        self.last_action_time = time.time()

        # Video recording state
        self.recording = False
        self.recorded_frames = []

        # On-screen notification state
        self.notification_text = ""
        self.notification_time = 0
        self.notification_duration = 2.0  # seconds

        # Current action description for display
        self.current_action_description = ""

        # Create initial game state with padded map
        from reinforcetactics.core.game_state import GameState
        self.game_state = GameState(padded_map,
                                    num_players=self.game_info.get('num_players', 2))

        # Create renderer
        from reinforcetactics.ui.renderer import Renderer
        self.renderer = Renderer(self.game_state)

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
                padded = pd.DataFrame(
                    np.full((min_height, min_width), 'o', dtype=object)
                )
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

        bordered = pd.DataFrame(
            np.full((new_height, new_width), 'o', dtype=object)
        )
        bordered.iloc[border_size:border_size + height,
                      border_size:border_size + width] = df.values

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
        self.progress_bar_rect = pygame.Rect(progress_start, self.control_y + 20,
                                             progress_width, 20)

    def execute_action(self, action):
        """
        Execute a single replay action.

        Coordinates in the action are in original (unpadded) space and are
        translated to padded coordinates before execution.

        Args:
            action: Action dictionary to execute
        """
        action_type = action.get('type')

        try:
            if action_type == 'create_unit':
                # Translate coordinates from original to padded
                padded_x, padded_y = self._translate_coords(action['x'], action['y'])
                self.game_state.create_unit(
                    action['unit_type'],
                    padded_x,
                    padded_y,
                    action['player']
                )

            elif action_type == 'move':
                # Translate coordinates from original to padded
                from_x, from_y = self._translate_coords(action['from_x'], action['from_y'])
                to_x, to_y = self._translate_coords(action['to_x'], action['to_y'])
                # Find the unit at the translated position
                unit = self.game_state.get_unit_at_position(from_x, from_y)
                if unit and unit.player == action['player']:
                    self.game_state.move_unit(unit, to_x, to_y)

            elif action_type == 'attack':
                # Translate coordinates from original to padded
                orig_attacker_pos = action['attacker_pos']
                orig_target_pos = action['target_pos']
                attacker_pos = self._translate_coords(*orig_attacker_pos)
                target_pos = self._translate_coords(*orig_target_pos)
                attacker = self.game_state.get_unit_at_position(*attacker_pos)
                target = self.game_state.get_unit_at_position(*target_pos)

                if attacker and target:
                    self.game_state.attack(attacker, target)

            elif action_type == 'seize':
                # Translate coordinates from original to padded
                orig_position = action['position']
                position = self._translate_coords(*orig_position)
                unit = self.game_state.get_unit_at_position(*position)
                if unit:
                    self.game_state.seize(unit)

            elif action_type == 'paralyze':
                # Translate coordinates from original to padded
                orig_paralyzer_pos = action['paralyzer_pos']
                orig_target_pos = action['target_pos']
                paralyzer_pos = self._translate_coords(*orig_paralyzer_pos)
                target_pos = self._translate_coords(*orig_target_pos)
                paralyzer = self.game_state.get_unit_at_position(*paralyzer_pos)
                target = self.game_state.get_unit_at_position(*target_pos)
                if paralyzer and target:
                    self.game_state.paralyze(paralyzer, target)

            elif action_type == 'heal':
                # Translate coordinates from original to padded
                orig_healer_pos = action['healer_pos']
                orig_target_pos = action['target_pos']
                healer_pos = self._translate_coords(*orig_healer_pos)
                target_pos = self._translate_coords(*orig_target_pos)
                healer = self.game_state.get_unit_at_position(*healer_pos)
                target = self.game_state.get_unit_at_position(*target_pos)
                if healer and target:
                    self.game_state.heal(healer, target)

            elif action_type == 'cure':
                # Translate coordinates from original to padded
                orig_curer_pos = action['curer_pos']
                orig_target_pos = action['target_pos']
                curer_pos = self._translate_coords(*orig_curer_pos)
                target_pos = self._translate_coords(*orig_target_pos)
                curer = self.game_state.get_unit_at_position(*curer_pos)
                target = self.game_state.get_unit_at_position(*target_pos)
                if curer and target:
                    self.game_state.cure(curer, target)

            elif action_type == 'resign':
                self.game_state.resign(action['player'])

            elif action_type == 'end_turn':
                # Don't record this action again
                old_history = self.game_state.action_history
                self.game_state.action_history = []
                self.game_state.end_turn()
                self.game_state.action_history = old_history

        except Exception as e:
            print(f"⚠️  Error executing action: {e}")
            print(f"   Action: {action}")

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
        self.current_action_index = 0
        self.paused = True
        self.current_action_description = ""

        # Recreate game state
        from reinforcetactics.core.game_state import GameState
        self.game_state = GameState(self.initial_map_data,
                                    num_players=self.game_info.get('num_players', 2))

        # Recreate renderer
        from reinforcetactics.ui.renderer import Renderer
        self.renderer = Renderer(self.game_state)
        self.setup_ui()

        self.show_notification("Replay restarted")

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
        # Reset game state
        from reinforcetactics.core.game_state import GameState
        self.game_state = GameState(self.initial_map_data,
                                    num_players=self.game_info.get('num_players', 2))

        # Recreate renderer
        from reinforcetactics.ui.renderer import Renderer
        self.renderer = Renderer(self.game_state)
        self.setup_ui()

        # Replay to target
        for i in range(target_index):
            if i < len(self.actions):
                self.execute_action(self.actions[i])
        self.current_action_index = target_index

    def skip_to_next_turn(self):
        """Skip to the next turn's first action."""
        if self.current_action_index >= len(self.actions):
            return

        current_turn = self._get_action_turn(self.current_action_index)

        # Find next turn's first action
        for i in range(self.current_action_index + 1, len(self.actions)):
            if self._get_action_turn(i) > current_turn:
                self._replay_to_action(i)
                self.show_notification(f"Turn {self._get_action_turn(i)}")
                return

        # No next turn found, go to end
        self._replay_to_action(len(self.actions))
        self.show_notification("Replay finished!")

    def skip_to_prev_turn(self):
        """Skip to the previous turn's first action."""
        if self.current_action_index <= 0:
            return

        current_turn = self._get_action_turn(max(0, self.current_action_index - 1))

        # Find the start of the previous turn
        target_turn = max(1, current_turn - 1)

        # Find the first action of the target turn
        for i in range(len(self.actions)):
            if self._get_action_turn(i) >= target_turn:
                self._replay_to_action(i)
                self.show_notification(f"Turn {self._get_action_turn(i)}")
                return

        # Fallback to beginning
        self._replay_to_action(0)
        self.show_notification("Turn 1")

    def _get_action_turn(self, action_index):
        """Get the turn number for an action index."""
        if action_index < 0 or action_index >= len(self.actions):
            return 0
        return self.actions[action_index].get('turn', 0)

    def _get_action_description(self, action):
        """Get a human-readable description of an action."""
        action_type = action.get('type', 'unknown')
        player = action.get('player', '?')

        if action_type == 'create_unit':
            unit_type = action.get('unit_type', 'unit')
            return f"P{player} creates {unit_type}"
        elif action_type == 'move':
            return f"P{player} moves unit"
        elif action_type == 'attack':
            return f"P{player} attacks"
        elif action_type == 'paralyze':
            return f"P{player} paralyzes target"
        elif action_type == 'heal':
            return f"P{player} heals unit"
        elif action_type == 'cure':
            return f"P{player} cures unit"
        elif action_type == 'seize':
            return f"P{player} seizes structure"
        elif action_type == 'resign':
            return f"P{player} resigns"
        elif action_type == 'end_turn':
            return f"P{player} ends turn"
        else:
            return f"P{player}: {action_type}"

    def start_recording(self):
        """Start recording video frames."""
        self.recording = True
        self.recorded_frames = []
        self.show_notification("Recording started...")

    def stop_recording(self):
        """Stop recording and prompt to save."""
        if self.recording:
            self.recording = False
            self.show_notification(f"Recording stopped ({len(self.recorded_frames)} frames)")

    def save_video(self):
        """Save recorded frames to video file."""
        if not self.recorded_frames:
            self.show_notification("No frames recorded!")
            return None

        if not CV2_AVAILABLE:
            self.show_notification("OpenCV not installed!")
            return None

        try:
            # Create videos directory
            videos_dir = Path("videos")
            videos_dir.mkdir(exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = videos_dir / f"replay_{timestamp}.mp4"

            # Get frame dimensions from first frame
            height, width, _ = self.recorded_frames[0].shape

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # 30 FPS for smooth playback
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Write frames
            for frame in self.recorded_frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr_frame)

            writer.release()
            duration = len(self.recorded_frames) / fps
            self.show_notification(f"Video saved! ({duration:.1f}s)")

            # Clear recorded frames to free memory
            self.recorded_frames = []
            return str(output_path)

        except Exception as e:
            self.show_notification(f"Error: {e}")
            return None

    def run(self):
        """Run the replay player."""
        clock = pygame.time.Clock()
        running = True

        # Show initial notification with game info
        player_configs = self.game_info.get('player_configs', [])
        if player_configs:
            p1_name = player_configs[0].get('name', 'P1') if len(player_configs) > 0 else 'P1'
            p2_name = player_configs[1].get('name', 'P2') if len(player_configs) > 1 else 'P2'
            self.show_notification(f"{p1_name} vs {p2_name}")

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
                        if not self.recording:
                            self.start_recording()
                        else:
                            self.stop_recording()
                            self.save_video()
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
                            if not self.recording:
                                self.start_recording()
                            else:
                                self.stop_recording()
                                self.save_video()
                        elif self.exit_button.collidepoint(mouse_pos):
                            running = False
                        elif self.progress_bar_rect.collidepoint(mouse_pos):
                            # Seek to position
                            relative_x = mouse_pos[0] - self.progress_bar_rect.x
                            progress = relative_x / self.progress_bar_rect.width
                            target_action = int(progress * len(self.actions))
                            self._replay_to_action(target_action)

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
        pygame.draw.rect(screen, (40, 40, 50), panel_rect)
        pygame.draw.line(screen, (200, 200, 220),
                        (0, self.control_y),
                        (screen_width, self.control_y), 2)

        # Draw buttons
        font = get_font(24)
        small_font = get_font(20)
        icon_size = 24
        icon_color = (255, 255, 255)

        # Step back button
        step_back_icon = get_arrow_left_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.step_back_button, step_back_icon, mouse_pos, (80, 80, 120))

        # Play/Pause button
        if self.paused:
            play_pause_icon = get_play_icon(size=icon_size, color=icon_color)
        else:
            play_pause_icon = get_pause_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.play_pause_button, play_pause_icon, mouse_pos, (100, 150, 100))

        # Step forward button
        step_forward_icon = get_arrow_right_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.step_forward_button, step_forward_icon, mouse_pos, (80, 80, 120))

        # Restart button
        restart_icon = get_restart_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.restart_button, restart_icon, mouse_pos, (150, 100, 100))

        # Skip to previous turn button
        skip_back_icon = get_skip_back_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.prev_turn_button, skip_back_icon, mouse_pos, (100, 100, 130))

        # Skip to next turn button
        skip_forward_icon = get_skip_forward_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.next_turn_button, skip_forward_icon, mouse_pos, (100, 100, 130))

        # Speed down button
        self._draw_button(self.speed_down_button, "-", mouse_pos, (100, 100, 150), font)

        # Speed display (between speed buttons)
        speed_text = f"{self.playback_speed}x"
        speed_surface = small_font.render(speed_text, True, (200, 200, 255))
        speed_rect = speed_surface.get_rect(center=(self.speed_display_x, self.control_y + 30))
        screen.blit(speed_surface, speed_rect)

        # Speed up button
        self._draw_button(self.speed_up_button, "+", mouse_pos, (100, 100, 150), font)

        # Save Video button
        save_video_text = "Rec" if not self.recording else "Save"
        save_video_color = (200, 50, 50) if self.recording else (70, 70, 150)
        self._draw_button(self.save_video_button, save_video_text, mouse_pos,
                         save_video_color, small_font)

        # Progress bar (only if there's space)
        if self.progress_bar_rect.width > 50:
            pygame.draw.rect(screen, (60, 60, 70), self.progress_bar_rect)
            if len(self.actions) > 0:
                progress = self.current_action_index / len(self.actions)
                progress_width = int(self.progress_bar_rect.width * progress)
                progress_rect = pygame.Rect(self.progress_bar_rect.x, self.progress_bar_rect.y,
                                            progress_width, self.progress_bar_rect.height)
                pygame.draw.rect(screen, (100, 200, 100), progress_rect)
            pygame.draw.rect(screen, (200, 200, 220), self.progress_bar_rect, 2)

        # Exit button
        exit_icon = get_x_icon(size=icon_size, color=icon_color)
        self._draw_icon_button(self.exit_button, exit_icon, mouse_pos, (150, 70, 70))

        # Recording indicator
        if self.recording:
            rec_text = f"REC ({len(self.recorded_frames)})"
            rec_surface = get_font(18).render(rec_text, True, (255, 50, 50))
            rec_x = screen_width - rec_surface.get_width() - 10
            screen.blit(rec_surface, (rec_x, 10))

        # Draw on-screen notification
        self._draw_notification(screen, screen_width)

        # Capture frame if recording
        if self.recording:
            # Get the pygame surface as a numpy array
            frame_data = pygame.surfarray.array3d(screen)
            # Transpose from (width, height, channels) to (height, width, channels)
            frame_data = np.transpose(frame_data, (1, 0, 2))
            self.recorded_frames.append(frame_data.copy())

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

        total_turns = self.game_info.get('total_turns', '?')

        # Layout: Turn/Action on left, Action description in center, Player info on right
        # All on a single row for compact display

        # Turn and action info on left
        turn_text = f"Turn {current_turn}/{total_turns}"
        action_text = f"Action {self.current_action_index}/{len(self.actions)}"
        combined_left = f"{turn_text}  |  {action_text}"
        left_surface = small_font.render(combined_left, True, (200, 200, 220))
        screen.blit(left_surface, (10, panel_y + (panel_height - left_surface.get_height()) // 2))

        # Current action description in center
        if self.current_action_description:
            desc_surface = font.render(self.current_action_description, True, (255, 220, 100))
            desc_rect = desc_surface.get_rect(center=(screen_width // 2, panel_y + panel_height // 2))
            screen.blit(desc_surface, desc_rect)

        # Player info on right
        player_configs = self.game_info.get('player_configs', [])
        winner = self.game_info.get('winner', None)

        if player_configs:
            p1_name = player_configs[0].get('name', 'P1')[:12] if len(player_configs) > 0 else 'P1'
            p2_name = player_configs[1].get('name', 'P2')[:12] if len(player_configs) > 1 else 'P2'

            # Color based on winner
            p1_color = (100, 255, 100) if winner == 1 else (200, 200, 255)
            p2_color = (100, 255, 100) if winner == 2 else (255, 200, 200)

            player_text = f"{p1_name} vs {p2_name}"
            player_surface = small_font.render(player_text, True, (255, 255, 255))
            screen.blit(player_surface, (screen_width - player_surface.get_width() - 10,
                                         panel_y + (panel_height - player_surface.get_height()) // 2))

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
                text_surface = font.render(self.notification_text, True, (255, 255, 255))

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
        """Draw a button with text."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 30, 255) for c in color) if is_hover else color

        pygame.draw.rect(self.renderer.screen, button_color, rect)
        pygame.draw.rect(self.renderer.screen, (200, 200, 220), rect, 2)

        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.renderer.screen.blit(text_surface, text_rect)

    def _draw_icon_button(self, rect, icon, mouse_pos, color):
        """Draw a button with an icon surface."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 30, 255) for c in color) if is_hover else color

        pygame.draw.rect(self.renderer.screen, button_color, rect)
        pygame.draw.rect(self.renderer.screen, (200, 200, 220), rect, 2)

        icon_rect = icon.get_rect(center=rect.center)
        self.renderer.screen.blit(icon, icon_rect)
