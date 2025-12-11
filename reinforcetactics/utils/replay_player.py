"""
Replay player for watching recorded games
"""
import time
from pathlib import Path
import pygame

from reinforcetactics.utils.fonts import get_font


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
        self.initial_map_data = initial_map_data

        self.current_action_index = 0
        self.playing = False
        self.paused = True
        self.playback_speed = 1.0  # 1x speed
        self.last_action_time = time.time()

        # Video recording state
        self.recording = False
        self.recorded_frames = []

        # Create initial game state
        from reinforcetactics.core.game_state import GameState
        self.game_state = GameState(initial_map_data,
                                    num_players=self.game_info.get('num_players', 2))

        # Create renderer
        from reinforcetactics.ui.renderer import Renderer
        self.renderer = Renderer(self.game_state)

        # UI elements
        self.setup_ui()

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

        Args:
            action: Action dictionary to execute
        """
        action_type = action.get('type')

        try:
            if action_type == 'create_unit':
                self.game_state.create_unit(
                    action['unit_type'],
                    action['x'],
                    action['y'],
                    action['player']
                )

            elif action_type == 'move':
                # Find the unit
                unit = self.game_state.get_unit_at_position(action['from_x'], action['from_y'])
                if unit and unit.player == action['player']:
                    self.game_state.move_unit(unit, action['to_x'], action['to_y'])

            elif action_type == 'attack':
                # Find attacker and target
                attacker_pos = action['attacker_pos']
                target_pos = action['target_pos']
                attacker = self.game_state.get_unit_at_position(*attacker_pos)
                target = self.game_state.get_unit_at_position(*target_pos)

                if attacker and target:
                    self.game_state.attack(attacker, target)

            elif action_type == 'seize':
                position = action['position']
                unit = self.game_state.get_unit_at_position(*position)
                if unit:
                    self.game_state.seize(unit)

            elif action_type == 'paralyze':
                paralyzer_pos = action['paralyzer_pos']
                target_pos = action['target_pos']
                paralyzer = self.game_state.get_unit_at_position(*paralyzer_pos)
                target = self.game_state.get_unit_at_position(*target_pos)
                if paralyzer and target:
                    self.game_state.paralyze(paralyzer, target)

            elif action_type == 'heal':
                healer_pos = action['healer_pos']
                target_pos = action['target_pos']
                healer = self.game_state.get_unit_at_position(*healer_pos)
                target = self.game_state.get_unit_at_position(*target_pos)
                if healer and target:
                    self.game_state.heal(healer, target)

            elif action_type == 'cure':
                curer_pos = action['curer_pos']
                target_pos = action['target_pos']
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

        try:
            import cv2
            from pathlib import Path
            from datetime import datetime

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

        except ImportError:
            print("❌ opencv-python not installed. Install with: pip install opencv-python")
            return None
        except Exception as e:
            print(f"❌ Error saving video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """Run the replay player."""
        clock = pygame.time.Clock()
        running = True

        print(f"\n🎬 Starting replay playback")
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

        # Exit button
        self._draw_button(self.exit_button, "Exit", mouse_pos, (150, 70, 70), font)

        # Recording indicator
        if self.recording:
            rec_text = f"🔴 REC ({len(self.recorded_frames)} frames)"
            rec_surface = get_font(20).render(rec_text, True, (255, 50, 50))
            self.renderer.screen.blit(rec_surface, (10, 10))

        # Capture frame if recording
        if self.recording:
            import numpy as np
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
