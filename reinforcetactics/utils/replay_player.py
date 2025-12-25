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

        self.play_pause_button = pygame.Rect(10, self.control_y + 10, 80, 40)
        self.restart_button = pygame.Rect(100, self.control_y + 10, 80, 40)
        self.speed_up_button = pygame.Rect(190, self.control_y + 10, 40, 40)
        self.speed_down_button = pygame.Rect(240, self.control_y + 10, 40, 40)
        self.save_video_button = pygame.Rect(290, self.control_y + 10, 100, 40)
        self.exit_button = pygame.Rect(screen_width - 90, self.control_y + 10, 80, 40)

        # Progress bar
        self.progress_bar_rect = pygame.Rect(400, self.control_y + 20,
                                             max(screen_width - 510, 100), 20)

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
                self.execute_action(action)
                self.current_action_index += 1
                self.last_action_time = current_time

                if self.current_action_index >= len(self.actions):
                    self.paused = True
                    print("✅ Replay finished!")

    def restart(self):
        """Restart the replay from the beginning."""
        self.current_action_index = 0
        self.paused = True

        # Recreate game state
        from reinforcetactics.core.game_state import GameState
        self.game_state = GameState(self.initial_map_data,
                                    num_players=self.game_info.get('num_players', 2))

        # Recreate renderer
        from reinforcetactics.ui.renderer import Renderer
        self.renderer = Renderer(self.game_state)
        self.setup_ui()

        print("🔄 Replay restarted")

    def toggle_pause(self):
        """Toggle pause/play."""
        self.paused = not self.paused
        if not self.paused:
            self.last_action_time = time.time()

    def change_speed(self, delta):
        """Change playback speed."""
        speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
        try:
            current_idx = speeds.index(self.playback_speed)
            new_idx = max(0, min(len(speeds) - 1, current_idx + delta))
            self.playback_speed = speeds[new_idx]
            print(f"⏩ Playback speed: {self.playback_speed}x")
        except ValueError:
            self.playback_speed = 1.0

    def start_recording(self):
        """Start recording video frames."""
        self.recording = True
        self.recorded_frames = []
        print("🔴 Started recording video...")

    def stop_recording(self):
        """Stop recording and prompt to save."""
        if self.recording:
            self.recording = False
            print(f"⏹️  Stopped recording. Captured {len(self.recorded_frames)} frames.")

    def save_video(self):
        """Save recorded frames to video file."""
        if not self.recorded_frames:
            print("⚠️  No frames recorded. Start recording first.")
            return None

        if not CV2_AVAILABLE:
            print("❌ opencv-python not installed. Install with: pip install opencv-python")
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
            print(f"✅ Video saved: {output_path}")
            print(f"   Duration: {len(self.recorded_frames) / fps:.1f} seconds")
            print(f"   Resolution: {width}x{height}")
            print(f"   Frames: {len(self.recorded_frames)}")

            # Clear recorded frames to free memory
            self.recorded_frames = []
            return str(output_path)

        except Exception as e:
            print(f"❌ Error saving video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """Run the replay player."""
        clock = pygame.time.Clock()
        running = True

        print("\n🎬 Starting replay playback")
        print(f"Total actions: {len(self.actions)}")
        print(f"Game info: {self.game_info}")
        print("\nControls:")
        print("  SPACE - Play/Pause")
        print("  R - Restart")
        print("  V - Start/Stop video recording")
        print("  + - Speed up")
        print("  - - Slow down")
        print("  ESC - Exit\n")

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

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.play_pause_button.collidepoint(mouse_pos):
                            self.toggle_pause()
                        elif self.restart_button.collidepoint(mouse_pos):
                            self.restart()
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

                            # Reset and play to target
                            self.restart()
                            for i in range(target_action):
                                if i < len(self.actions):
                                    self.execute_action(self.actions[i])
                            self.current_action_index = target_action

            # Update replay
            self.update()

            # Draw
            self.draw(mouse_pos)

            pygame.display.flip()
            clock.tick(60)

        print("👋 Exiting replay player")

    def draw(self, mouse_pos):
        """Draw the replay player."""
        # Draw game state
        self.renderer.render()

        # Draw control panel background
        panel_rect = pygame.Rect(0, self.control_y,
                                 self.renderer.screen.get_width(), 60)
        pygame.draw.rect(self.renderer.screen, (40, 40, 50), panel_rect)
        pygame.draw.line(self.renderer.screen, (200, 200, 220),
                        (0, self.control_y),
                        (self.renderer.screen.get_width(), self.control_y), 2)

        # Draw buttons
        font = get_font(28)

        # Play/Pause button
        play_text = "▶" if self.paused else "⏸"
        self._draw_button(self.play_pause_button, play_text, mouse_pos, (100, 150, 100), font)

        # Restart button
        self._draw_button(self.restart_button, "⟲", mouse_pos, (150, 100, 100), font)

        # Speed buttons
        speed_font = get_font(32)
        self._draw_button(self.speed_up_button, "+", mouse_pos, (100, 100, 150), speed_font)
        self._draw_button(self.speed_down_button, "-", mouse_pos, (100, 100, 150), speed_font)

        # Save Video button
        save_video_text = "⏺ Rec" if not self.recording else "⏹ Save"
        save_video_color = (200, 50, 50) if self.recording else (70, 70, 150)
        small_font = get_font(22)
        self._draw_button(self.save_video_button, save_video_text, mouse_pos,
                         save_video_color, small_font)

        # Progress bar
        pygame.draw.rect(self.renderer.screen, (60, 60, 70), self.progress_bar_rect)
        if len(self.actions) > 0:
            progress = self.current_action_index / len(self.actions)
            progress_width = int(self.progress_bar_rect.width * progress)
            progress_rect = pygame.Rect(self.progress_bar_rect.x, self.progress_bar_rect.y,
                                        progress_width, self.progress_bar_rect.height)
            pygame.draw.rect(self.renderer.screen, (100, 200, 100), progress_rect)
        pygame.draw.rect(self.renderer.screen, (200, 200, 220), self.progress_bar_rect, 2)

        # Progress text
        progress_text = f"{self.current_action_index} / {len(self.actions)}"
        progress_surface = get_font(20).render(progress_text, True, (255, 255, 255))
        progress_text_rect = progress_surface.get_rect(
            midleft=(self.progress_bar_rect.right + 10, self.progress_bar_rect.centery)
        )
        self.renderer.screen.blit(progress_surface, progress_text_rect)

        # Game outcome display (when replay is finished)
        if self.current_action_index >= len(self.actions) and len(self.actions) > 0:
            winner = self.game_info.get('winner')
            draw_reason = self.game_info.get('draw_reason')

            if winner == 0:
                # Draw outcome
                outcome_text = "Game ended in a DRAW"
                if draw_reason == "max_turns":
                    outcome_text += f" (max turns: {self.game_info.get('max_turns', '?')})"
                outcome_color = (255, 200, 50)  # Yellow/gold for draws
            elif winner is not None:
                outcome_text = f"Player {winner} WINS!"
                outcome_color = (100, 255, 100)  # Green for wins
            else:
                outcome_text = "Game Over"
                outcome_color = (200, 200, 200)

            # Draw outcome banner
            outcome_font = get_font(36)
            outcome_surface = outcome_font.render(outcome_text, True, outcome_color)
            banner_width = outcome_surface.get_width() + 40
            banner_height = outcome_surface.get_height() + 20
            banner_x = (self.renderer.screen.get_width() - banner_width) // 2
            banner_y = self.control_y - banner_height - 20

            # Semi-transparent background
            banner_surface = pygame.Surface((banner_width, banner_height), pygame.SRCALPHA)
            banner_surface.fill((0, 0, 0, 180))
            self.renderer.screen.blit(banner_surface, (banner_x, banner_y))

            # Border
            pygame.draw.rect(self.renderer.screen, outcome_color,
                           (banner_x, banner_y, banner_width, banner_height), 3)

            # Text
            text_rect = outcome_surface.get_rect(center=(banner_x + banner_width // 2,
                                                          banner_y + banner_height // 2))
            self.renderer.screen.blit(outcome_surface, text_rect)

        # Exit button
        self._draw_button(self.exit_button, "Exit", mouse_pos, (150, 70, 70), font)

        # Recording indicator
        if self.recording:
            rec_text = f"🔴 REC ({len(self.recorded_frames)} frames)"
            rec_surface = get_font(20).render(rec_text, True, (255, 50, 50))
            self.renderer.screen.blit(rec_surface, (10, 10))

        # Capture frame if recording
        if self.recording:
            # Get the pygame surface as a numpy array
            frame_data = pygame.surfarray.array3d(self.renderer.screen)
            # Transpose from (width, height, channels) to (height, width, channels)
            frame_data = np.transpose(frame_data, (1, 0, 2))
            self.recorded_frames.append(frame_data.copy())

    def _draw_button(self, rect, text, mouse_pos, color, font):
        """Draw a button."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 30, 255) for c in color) if is_hover else color

        pygame.draw.rect(self.renderer.screen, button_color, rect)
        pygame.draw.rect(self.renderer.screen, (200, 200, 220), rect, 2)

        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.renderer.screen.blit(text_surface, text_rect)
