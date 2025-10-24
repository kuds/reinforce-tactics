"""
Replay player for watching recorded games
"""
import pygame
import time
from pathlib import Path


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
        self.exit_button = pygame.Rect(screen_width - 90, self.control_y + 10, 80, 40)
        
        # Progress bar
        self.progress_bar_rect = pygame.Rect(290, self.control_y + 20, 
                                             screen_width - 400, 20)
    
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
        font = pygame.font.Font(None, 28)
        
        # Play/Pause button
        play_text = "▶" if self.paused else "⏸"
        self._draw_button(self.play_pause_button, play_text, mouse_pos, (100, 150, 100), font)
        
        # Restart button
        self._draw_button(self.restart_button, "⟲", mouse_pos, (150, 100, 100), font)
        
        # Speed buttons
        speed_font = pygame.font.Font(None, 32)
        self._draw_button(self.speed_up_button, "+", mouse_pos, (100, 100, 150), speed_font)
        self._draw_button(self.speed_down_button, "-", mouse_pos, (100, 100, 150), speed_font)
        
        # Speed indicator
        speed_text = f"{self.playback_speed}x"
        speed_surface = font.render(speed_text, True, (255, 255, 255))
        self.renderer.screen.blit(speed_surface, (250, self.control_y + 20))
        
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
        progress_surface = pygame.font.Font(None, 20).render(progress_text, True, (255, 255, 255))
        progress_text_rect = progress_surface.get_rect(
            midleft=(self.progress_bar_rect.right + 10, self.progress_bar_rect.centery)
        )
        self.renderer.screen.blit(progress_surface, progress_text_rect)
        
        # Exit button
        self._draw_button(self.exit_button, "Exit", mouse_pos, (150, 70, 70), font)
    
    def _draw_button(self, rect, text, mouse_pos, color, font):
        """Draw a button."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 30, 255) for c in color) if is_hover else color
        
        pygame.draw.rect(self.renderer.screen, button_color, rect)
        pygame.draw.rect(self.renderer.screen, (200, 200, 220), rect, 2)
        
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.renderer.screen.blit(text_surface, text_rect)
