"""
Menu classes for the game UI
"""
import pygame
from pathlib import Path
from reinforcetactics.constants import TILE_SIZE, UNIT_DATA, PLAYER_COLORS, UNIT_COLORS
from reinforcetactics.utils.language import get_language, LANGUAGE_NAMES
from reinforcetactics.utils.settings import get_settings


class MainMenu:
    """Main menu for game mode selection."""
    
    def __init__(self):
        """Initialize the main menu."""
        pygame.init()
        
        # Get language instance
        self.lang = get_language()
        
        # Menu dimensions
        self.width = 600
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Reinforce Tactics - Main Menu")
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.title_color = (255, 215, 0)
        self.button_color = (70, 70, 90)
        self.button_hover_color = (100, 100, 130)
        self.button_text_color = (255, 255, 255)
        self.border_color = (200, 200, 220)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 72)
        self.button_font = pygame.font.Font(None, 42)
        self.subtitle_font = pygame.font.Font(None, 28)
        
        # Button setup
        button_width = 400
        button_height = 60
        button_x = (self.width - button_width) // 2
        start_y = 200
        spacing = 80
        
        self.buttons = {
            '1v1_human': pygame.Rect(button_x, start_y, button_width, button_height),
            '1v1_computer': pygame.Rect(button_x, start_y + spacing, button_width, button_height),
            'replay': pygame.Rect(button_x, start_y + spacing * 2, button_width, button_height),
            'load': pygame.Rect(button_x, start_y + spacing * 3, button_width, button_height),
            'settings': pygame.Rect(button_x, start_y + spacing * 4, button_width, button_height),
            'exit': pygame.Rect(button_x, start_y + spacing * 5, button_width, button_height)
        }
        
        self.clock = pygame.time.Clock()
    
    def run(self):
        """Run the main menu and return the selected option."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return {'type': 'exit'}
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return {'type': 'exit'}
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        for key, rect in self.buttons.items():
                            if rect.collidepoint(mouse_pos):
                                if key == 'exit':
                                    return {'type': 'exit'}
                                elif key == '1v1_human':
                                    return {'type': 'new_game', 'mode': 'human_vs_human'}
                                elif key == '1v1_computer':
                                    return {'type': 'new_game', 'mode': 'human_vs_computer'}
                                elif key == 'replay':
                                    return {'type': 'watch_replay'}
                                elif key == 'load':
                                    return {'type': 'load_game'}
                                elif key == 'settings':
                                    # Show settings menu
                                    settings_menu = SettingsMenu()
                                    settings_menu.run()
                                    # Reload language after settings change
                                    self.lang = get_language()
            
            # Draw menu
            self.draw(mouse_pos)
            pygame.display.flip()
            self.clock.tick(60)
        
        return {'type': 'exit'}
    
    def draw(self, mouse_pos):
        """Draw the main menu."""
        # Background
        self.screen.fill(self.bg_color)
        
        # Title
        title_text = self.lang.get('main_title')
        title_surface = self.title_font.render(title_text, True, self.title_color)
        title_rect = title_surface.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_surface, title_rect)
        
        # Subtitle
        subtitle_text = self.lang.get('main_subtitle')
        subtitle_surface = self.subtitle_font.render(subtitle_text, True, (150, 150, 200))
        subtitle_rect = subtitle_surface.get_rect(center=(self.width // 2, 130))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Button labels with translations
        button_labels = {
            '1v1_human': self.lang.get('menu_1v1_human'),
            '1v1_computer': self.lang.get('menu_1v1_computer'),
            'replay': self.lang.get('menu_replay'),
            'load': self.lang.get('menu_load'),
            'settings': self.lang.get('menu_settings'),
            'exit': self.lang.get('menu_exit')
        }
        
        # Draw buttons
        for key, rect in self.buttons.items():
            # Check if mouse is hovering
            is_hover = rect.collidepoint(mouse_pos)
            button_color = self.button_hover_color if is_hover else self.button_color
            
            # Draw button background
            pygame.draw.rect(self.screen, button_color, rect)
            pygame.draw.rect(self.screen, self.border_color, rect, 3)
            
            # Draw button text
            label = button_labels[key]
            text_surface = self.button_font.render(label, True, self.button_text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
        
        # Footer
        footer_font = pygame.font.Font(None, 20)
        footer_text = self.lang.get('press_esc')
        footer_surface = footer_font.render(footer_text, True, (120, 120, 140))
        footer_rect = footer_surface.get_rect(center=(self.width // 2, self.height - 30))
        self.screen.blit(footer_surface, footer_rect)


class SettingsMenu:
    """Settings menu for configuring game options."""
    
    def __init__(self):
        """Initialize settings menu."""
        pygame.init()
        
        self.settings = get_settings()
        self.lang = get_language()
        
        # Menu dimensions
        self.width = 700
        self.height = 650
        self.screen = pygame.display.get_surface()
        if self.screen is None or self.screen.get_size() != (self.width, self.height):
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption(self.lang.get('settings_title'))
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.section_color = (50, 50, 60)
        self.button_color = (70, 70, 90)
        self.button_hover_color = (100, 100, 130)
        self.selected_color = (100, 150, 100)
        self.text_color = (255, 255, 255)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 56)
        self.section_font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 28)
        self.button_font = pygame.font.Font(None, 32)
        
        # Language buttons
        lang_y = 120
        lang_x_start = 50
        lang_width = 140
        lang_height = 50
        lang_spacing = 160
        
        self.language_buttons = {}
        languages = ['english', 'french', 'korean', 'spanish']
        for i, lang in enumerate(languages):
            x = lang_x_start + (i * lang_spacing)
            self.language_buttons[lang] = pygame.Rect(x, lang_y, lang_width, lang_height)
        
        # Path input fields
        self.path_fields = {}
        self.active_field = None
        self.input_text = {}
        
        path_y_start = 240
        path_spacing = 60
        path_types = ['maps', 'videos', 'replays', 'saves']
        
        for i, path_type in enumerate(path_types):
            y = path_y_start + (i * path_spacing)
            # Label area
            label_rect = pygame.Rect(50, y, 200, 40)
            # Input area
            input_rect = pygame.Rect(260, y, 390, 40)
            self.path_fields[path_type] = {
                'label': label_rect,
                'input': input_rect
            }
            self.input_text[path_type] = self.settings.get_path(path_type)
        
        # Bottom buttons
        button_y = 550
        self.reset_button = pygame.Rect(50, button_y, 180, 50)
        self.save_button = pygame.Rect(260, button_y, 180, 50)
        self.back_button = pygame.Rect(470, button_y, 180, 50)
        
        self.clock = pygame.time.Clock()
        self.saved_message_timer = 0
    
    def run(self):
        """Run the settings menu."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif self.active_field:
                        # Handle text input
                        if event.key == pygame.K_RETURN:
                            self.active_field = None
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text[self.active_field] = self.input_text[self.active_field][:-1]
                        else:
                            # Add character if printable
                            if event.unicode.isprintable():
                                self.input_text[self.active_field] += event.unicode
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check language buttons
                        for lang, rect in self.language_buttons.items():
                            if rect.collidepoint(mouse_pos):
                                self.settings.set_language(lang)
                                # Reload language instance
                                from reinforcetactics.utils.language import _language_instance
                                _language_instance = None
                                self.lang = get_language()
                                pygame.display.set_caption(self.lang.get('settings_title'))
                        
                        # Check path input fields
                        clicked_field = None
                        for path_type, rects in self.path_fields.items():
                            if rects['input'].collidepoint(mouse_pos):
                                clicked_field = path_type
                                break
                        self.active_field = clicked_field
                        
                        # Check bottom buttons
                        if self.reset_button.collidepoint(mouse_pos):
                            self.settings.reset_to_defaults()
                            # Reload paths
                            for path_type in self.input_text.keys():
                                self.input_text[path_type] = self.settings.get_path(path_type)
                            self.saved_message_timer = 120
                        
                        elif self.save_button.collidepoint(mouse_pos):
                            # Save all paths
                            for path_type, path in self.input_text.items():
                                self.settings.set_path(path_type, path)
                            self.settings.ensure_directories()
                            self.saved_message_timer = 120
                        
                        elif self.back_button.collidepoint(mouse_pos):
                            return
            
            # Update timers
            if self.saved_message_timer > 0:
                self.saved_message_timer -= 1
            
            self.draw(mouse_pos)
            pygame.display.flip()
            self.clock.tick(60)
    
    def draw(self, mouse_pos):
        """Draw the settings menu."""
        self.screen.fill(self.bg_color)
        
        # Title
        title_text = self.lang.get('settings_title')
        title_surface = self.title_font.render(title_text, True, (255, 215, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 40))
        self.screen.blit(title_surface, title_rect)
        
        # Language section
        lang_label = self.lang.get('settings_language')
        lang_surface = self.section_font.render(lang_label, True, (150, 200, 255))
        self.screen.blit(lang_surface, (50, 80))
        
        # Language buttons
        current_lang = self.settings.get_language()
        for lang, rect in self.language_buttons.items():
            is_hover = rect.collidepoint(mouse_pos)
            is_selected = lang == current_lang
            
            if is_selected:
                color = self.selected_color
            elif is_hover:
                color = self.button_hover_color
            else:
                color = self.button_color
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (200, 200, 220), rect, 2)
            
            # Button text
            text = LANGUAGE_NAMES.get(lang, lang.capitalize())
            text_surface = self.label_font.render(text, True, self.text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
        
        # Paths section
        paths_label = self.lang.get('settings_paths')
        paths_surface = self.section_font.render(paths_label, True, (150, 200, 255))
        self.screen.blit(paths_surface, (50, 200))
        
        # Path input fields
        path_labels = {
            'maps': self.lang.get('settings_maps_path'),
            'videos': self.lang.get('settings_videos_path'),
            'replays': self.lang.get('settings_replays_path'),
            'saves': self.lang.get('settings_saves_path')
        }
        
        for path_type, rects in self.path_fields.items():
            # Draw label
            label_text = path_labels[path_type]
            label_surface = self.label_font.render(label_text, True, (200, 200, 200))
            label_rect = label_surface.get_rect(midleft=(rects['label'].x, rects['label'].centery))
            self.screen.blit(label_surface, label_rect)
            
            # Draw input box
            is_active = self.active_field == path_type
            input_color = (80, 120, 80) if is_active else (60, 60, 70)
            pygame.draw.rect(self.screen, input_color, rects['input'])
            pygame.draw.rect(self.screen, (200, 200, 220) if is_active else (120, 120, 140), rects['input'], 2)
            
            # Draw text
            text = self.input_text[path_type]
            text_surface = self.label_font.render(text, True, self.text_color)
            text_rect = text_surface.get_rect(midleft=(rects['input'].x + 10, rects['input'].centery))
            
            # Clip text if too long
            if text_rect.width > rects['input'].width - 20:
                # Show end of text
                while text_rect.width > rects['input'].width - 20 and text:
                    text = text[1:]
                    text_surface = self.label_font.render("..." + text, True, self.text_color)
                    text_rect = text_surface.get_rect(midleft=(rects['input'].x + 10, rects['input'].centery))
            
            self.screen.blit(text_surface, text_rect)
            
            # Draw cursor if active
            if is_active and pygame.time.get_ticks() % 1000 < 500:
                cursor_x = text_rect.right + 2
                cursor_y1 = rects['input'].centery - 12
                cursor_y2 = rects['input'].centery + 12
                pygame.draw.line(self.screen, self.text_color, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)
        
        # Bottom buttons
        self._draw_button(self.reset_button, self.lang.get('settings_reset'), mouse_pos)
        self._draw_button(self.save_button, self.lang.get('settings_save'), mouse_pos)
        self._draw_button(self.back_button, self.lang.get('settings_back'), mouse_pos)
        
        # Saved message
        if self.saved_message_timer > 0:
            saved_text = self.lang.get('settings_saved')
            saved_surface = self.section_font.render(saved_text, True, (100, 255, 100))
            saved_rect = saved_surface.get_rect(center=(self.width // 2, 510))
            self.screen.blit(saved_surface, saved_rect)
    
    def _draw_button(self, rect, text, mouse_pos):
        """Draw a button."""
        is_hover = rect.collidepoint(mouse_pos)
        color = self.button_hover_color if is_hover else self.button_color
        
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (200, 200, 220), rect, 2)
        
        text_surface = self.button_font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)


class MapSelectionMenu:
    """Menu for selecting a map."""
    
    def __init__(self, map_type="1v1"):
        """Initialize map selection menu."""
        pygame.init()
        
        self.lang = get_language()
        self.settings = get_settings()
        
        self.width = 600
        self.height = 500
        self.screen = pygame.display.get_surface()
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption(self.lang.get('map_select_title'))
        
        # Get available maps from configured path
        from reinforcetactics.utils.file_io import FileIO
        maps_path = Path(self.settings.get_path('maps')) / map_type
        self.maps = list(maps_path.glob("*.csv")) if maps_path.exists() else []
        self.map_type = map_type
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.list_bg_color = (50, 50, 60)
        self.selected_color = (100, 150, 100)
        self.hover_color = (70, 70, 90)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.list_font = pygame.font.Font(None, 32)
        
        # List setup
        self.list_rect = pygame.Rect(50, 100, 500, 300)
        self.item_height = 50
        self.scroll_offset = 0
        self.selected_index = 0 if self.maps else -1
        
        # Buttons
        self.select_button = pygame.Rect(150, 420, 120, 50)
        self.random_button = pygame.Rect(280, 420, 120, 50)
        self.back_button = pygame.Rect(410, 420, 120, 50)
        
        self.clock = pygame.time.Clock()
    
    def run(self):
        """Run the map selection menu."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP and self.selected_index > 0:
                        self.selected_index -= 1
                    elif event.key == pygame.K_DOWN and self.selected_index < len(self.maps) - 1:
                        self.selected_index += 1
                    elif event.key == pygame.K_RETURN and self.selected_index >= 0:
                        return str(self.maps[self.selected_index])
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check buttons
                        if self.select_button.collidepoint(mouse_pos) and self.selected_index >= 0:
                            return str(self.maps[self.selected_index])
                        elif self.random_button.collidepoint(mouse_pos):
                            return "random"
                        elif self.back_button.collidepoint(mouse_pos):
                            return None
                        
                        # Check list items
                        if self.list_rect.collidepoint(mouse_pos):
                            relative_y = mouse_pos[1] - self.list_rect.y + self.scroll_offset
                            clicked_index = int(relative_y // self.item_height)
                            if 0 <= clicked_index < len(self.maps):
                                self.selected_index = clicked_index
                
                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_offset = max(0, min(
                        self.scroll_offset - event.y * 20,
                        max(0, len(self.maps) * self.item_height - self.list_rect.height)
                    ))
            
            self.draw(mouse_pos)
            pygame.display.flip()
            self.clock.tick(60)
        
        return None
    
    def draw(self, mouse_pos):
        """Draw the map selection menu."""
        self.screen.fill(self.bg_color)
        
        # Title
        title_text = self.lang.get('map_select_title')
        title_surface = self.title_font.render(title_text, True, (255, 215, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title_surface, title_rect)
        
        # Draw map list
        pygame.draw.rect(self.screen, self.list_bg_color, self.list_rect)
        pygame.draw.rect(self.screen, (200, 200, 220), self.list_rect, 2)
        
        # Create clipping rect for list
        clip_rect = self.screen.get_clip()
        self.screen.set_clip(self.list_rect)
        
        for i, map_path in enumerate(self.maps):
            y = self.list_rect.y + i * self.item_height - self.scroll_offset
            item_rect = pygame.Rect(self.list_rect.x, y, self.list_rect.width, self.item_height)
            
            if self.list_rect.colliderect(item_rect):
                # Draw background
                if i == self.selected_index:
                    pygame.draw.rect(self.screen, self.selected_color, item_rect)
                elif item_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, self.hover_color, item_rect)
                
                # Draw text
                map_name = map_path.stem
                text_surface = self.list_font.render(map_name, True, (255, 255, 255))
                text_rect = text_surface.get_rect(midleft=(item_rect.x + 10, item_rect.centery))
                self.screen.blit(text_surface, text_rect)
        
        self.screen.set_clip(clip_rect)
        
        # Draw buttons with translations
        self._draw_button(self.select_button, self.lang.get('map_select'), mouse_pos, self.selected_index >= 0)
        self._draw_button(self.random_button, self.lang.get('map_random'), mouse_pos, True)
        self._draw_button(self.back_button, self.lang.get('map_back'), mouse_pos, True)
    
    def _draw_button(self, rect, text, mouse_pos, enabled):
        """Draw a button."""
        if enabled:
            color = (100, 100, 130) if rect.collidepoint(mouse_pos) else (70, 70, 90)
            text_color = (255, 255, 255)
        else:
            color = (50, 50, 60)
            text_color = (100, 100, 100)
        
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (200, 200, 220), rect, 2)
        
        button_font = pygame.font.Font(None, 32)
        text_surface = button_font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)


class UnitStatsMenu:
    """Menu for displaying unit statistics on right-click."""

    def __init__(self, unit, grid_width, grid_height):
        """
        Initialize the unit stats menu.
        
        Args:
            unit: The unit to display stats for
            grid_width: Width of the game grid
            grid_height: Height of the game grid
        """
        self.unit = unit
        self.lang = get_language()
        
        # Menu dimensions
        self.width = 280
        self.height = 320
        
        # Calculate menu position (next to unit, adjusting for screen bounds)
        screen_x = unit.x * TILE_SIZE + TILE_SIZE + 10
        screen_y = unit.y * TILE_SIZE
        
        # Adjust if menu would go off screen
        if screen_x + self.width > grid_width * TILE_SIZE:
            screen_x = unit.x * TILE_SIZE - self.width - 10
        if screen_y + self.height > grid_height * TILE_SIZE:
            screen_y = grid_height * TILE_SIZE - self.height
        
        self.rect = pygame.Rect(screen_x, screen_y, self.width, self.height)
        
        # Get unit data from constants
        self.unit_info = UNIT_DATA[unit.type]
        
    def draw(self, screen):
        """Draw the stats menu."""
        # Main background
        pygame.draw.rect(screen, (40, 40, 50), self.rect)
        pygame.draw.rect(screen, (200, 200, 220), self.rect, 3)
        
        # Header section
        header_rect = pygame.Rect(self.rect.x, self.rect.y, self.width, 50)
        player_color = PLAYER_COLORS.get(self.unit.player, (255, 255, 255))
        pygame.draw.rect(screen, player_color, header_rect)
        pygame.draw.rect(screen, (200, 200, 220), header_rect, 2)
        
        # Unit name and type
        title_font = pygame.font.Font(None, 32)
        
        # Get translated unit name
        unit_names = {'W': 'warrior', 'M': 'mage', 'C': 'cleric'}
        unit_key = unit_names.get(self.unit.type, 'warrior')
        title_text = self.lang.get(unit_key).title()
        
        title_surface = title_font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=(header_rect.centerx, header_rect.centery - 8))
        screen.blit(title_surface, title_rect)
        
        # Player ownership
        owner_font = pygame.font.Font(None, 22)
        owner_text = f"{self.lang.get('player')} {self.unit.player}"
        owner_surface = owner_font.render(owner_text, True, (255, 255, 255))
        owner_rect = owner_surface.get_rect(center=(header_rect.centerx, header_rect.centery + 12))
        screen.blit(owner_surface, owner_rect)
        
        # Stats section
        stats_y = self.rect.y + 60
        line_height = 28
        label_font = pygame.font.Font(None, 24)
        value_font = pygame.font.Font(None, 24)
        
        # Health
        self._draw_stat_line(screen, self.lang.get('health') + ":", 
                            f"{self.unit.health} / {self.unit.max_health}", 
                            stats_y, label_font, value_font)
        
        # Health bar
        bar_x = self.rect.x + 20
        bar_y = stats_y + 22
        bar_width = self.width - 40
        bar_height = 8
        pygame.draw.rect(screen, (100, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        health_percentage = self.unit.health / self.unit.max_health
        current_bar_width = int(bar_width * health_percentage)
        if health_percentage > 0.5:
            health_color = (0, 200, 0)
        elif health_percentage > 0.25:
            health_color = (255, 165, 0)
        else:
            health_color = (255, 0, 0)
        pygame.draw.rect(screen, health_color, (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), 1)
        
        stats_y += 40
        
        # Attack
        if self.unit.type == 'M':
            attack_str = f"{self.unit_info['attack']['adjacent']} / {self.unit_info['attack']['range']}"
        else:
            attack_str = str(self.unit_info['attack'])
        self._draw_stat_line(screen, self.lang.get('attack') + ":", attack_str, stats_y, label_font, value_font)
        stats_y += line_height
        
        # Movement
        self._draw_stat_line(screen, self.lang.get('movement') + ":", str(self.unit.movement_range), 
                            stats_y, label_font, value_font)
        stats_y += line_height
        
        # Cost
        self._draw_stat_line(screen, self.lang.get('cost') + ":", f"${self.unit_info['cost']}", 
                            stats_y, label_font, value_font)
        stats_y += line_height
        
        # Status effects section (shortened to fit)
        if self.unit.is_paralyzed():
            stats_y += 10
            status_header = label_font.render(self.lang.get('status') + ":", True, (255, 200, 100))
            screen.blit(status_header, (self.rect.x + 20, stats_y))
            stats_y += line_height
            
            status_font = pygame.font.Font(None, 22)
            status_text = f"• Paralyzed ({self.unit.paralyzed_turns})"
            status_surface = status_font.render(status_text, True, (200, 100, 255))
            screen.blit(status_surface, (self.rect.x + 30, stats_y))
    
    def _draw_stat_line(self, screen, label, value, y, label_font, value_font):
        """Helper method to draw a stat line with label and value."""
        label_surface = label_font.render(label, True, (200, 200, 200))
        screen.blit(label_surface, (self.rect.x + 20, y))
        
        value_surface = value_font.render(str(value), True, (255, 255, 255))
        value_rect = value_surface.get_rect(right=self.rect.right - 20, y=y)
        screen.blit(value_surface, value_rect)
    
    def is_clicked(self, mouse_pos):
        """Check if the menu area was clicked."""
        return self.rect.collidepoint(mouse_pos)

"""
Add these classes to reinforcetactics/ui/menus.py
"""

class SaveGameMenu:
    """Menu for saving the current game."""
    
    def __init__(self, game_state):
        """Initialize save game menu."""
        pygame.init()
        
        self.game_state = game_state
        self.lang = get_language()
        self.settings = get_settings()
        
        self.width = 500
        self.height = 400
        self.screen = pygame.display.get_surface()
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption("Save Game")
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.input_color = (60, 60, 70)
        self.input_active_color = (80, 120, 80)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.label_font = pygame.font.Font(None, 32)
        
        # Input field for save name
        self.input_rect = pygame.Rect(50, 150, 400, 50)
        self.input_text = f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.input_active = True
        
        # Buttons
        self.save_button = pygame.Rect(100, 250, 140, 50)
        self.cancel_button = pygame.Rect(260, 250, 140, 50)
        
        self.clock = pygame.time.Clock()
        self.save_result = None
    
    def run(self):
        """Run the save menu and return save path or None."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_RETURN:
                        # Save game
                        saves_dir = Path(self.settings.get_path('saves'))
                        saves_dir.mkdir(parents=True, exist_ok=True)
                        filepath = saves_dir / f"{self.input_text}.json"
                        result = self.game_state.save_to_file(str(filepath))
                        return result
                    elif self.input_active:
                        if event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        elif event.unicode.isprintable():
                            self.input_text += event.unicode
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.input_rect.collidepoint(mouse_pos):
                            self.input_active = True
                        else:
                            self.input_active = False
                        
                        if self.save_button.collidepoint(mouse_pos):
                            saves_dir = Path(self.settings.get_path('saves'))
                            saves_dir.mkdir(parents=True, exist_ok=True)
                            filepath = saves_dir / f"{self.input_text}.json"
                            result = self.game_state.save_to_file(str(filepath))
                            return result
                        
                        elif self.cancel_button.collidepoint(mouse_pos):
                            return None
            
            self.draw(mouse_pos)
            pygame.display.flip()
            self.clock.tick(60)
        
        return None
    
    def draw(self, mouse_pos):
        """Draw the save menu."""
        self.screen.fill(self.bg_color)
        
        # Title
        title = self.title_font.render("Save Game", True, (255, 215, 0))
        title_rect = title.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title, title_rect)
        
        # Label
        label = self.label_font.render("Save Name:", True, (200, 200, 200))
        self.screen.blit(label, (50, 110))
        
        # Input field
        input_color = self.input_active_color if self.input_active else self.input_color
        pygame.draw.rect(self.screen, input_color, self.input_rect)
        pygame.draw.rect(self.screen, (200, 200, 220), self.input_rect, 2)
        
        # Input text
        text_surface = self.label_font.render(self.input_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(midleft=(self.input_rect.x + 10, self.input_rect.centery))
        self.screen.blit(text_surface, text_rect)
        
        # Cursor
        if self.input_active and pygame.time.get_ticks() % 1000 < 500:
            cursor_x = text_rect.right + 2
            pygame.draw.line(self.screen, (255, 255, 255),
                           (cursor_x, self.input_rect.centery - 15),
                           (cursor_x, self.input_rect.centery + 15), 2)
        
        # Buttons
        self._draw_button(self.save_button, "Save", mouse_pos, (100, 150, 100))
        self._draw_button(self.cancel_button, "Cancel", mouse_pos, (150, 70, 70))
    
    def _draw_button(self, rect, text, mouse_pos, color):
        """Draw a button."""
        is_hover = rect.collidepoint(mouse_pos)
        button_color = tuple(min(c + 30, 255) for c in color) if is_hover else color
        
        pygame.draw.rect(self.screen, button_color, rect)
        pygame.draw.rect(self.screen, (200, 200, 220), rect, 2)
        
        text_surface = self.label_font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)


class LoadGameMenu:
    """Menu for loading a saved game."""
    
    def __init__(self):
        """Initialize load game menu."""
        pygame.init()
        
        self.lang = get_language()
        self.settings = get_settings()
        
        self.width = 600
        self.height = 500
        self.screen = pygame.display.get_surface()
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption("Load Game")
        
        # Get available saves
        from reinforcetactics.utils.file_io import FileIO
        self.saves = FileIO.list_saves()
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.list_bg_color = (50, 50, 60)
        self.selected_color = (100, 150, 100)
        self.hover_color = (70, 70, 90)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.list_font = pygame.font.Font(None, 28)
        self.info_font = pygame.font.Font(None, 22)
        
        # List setup
        self.list_rect = pygame.Rect(50, 100, 500, 300)
        self.item_height = 60
        self.scroll_offset = 0
        self.selected_index = 0 if self.saves else -1
        
        # Buttons
        self.load_button = pygame.Rect(150, 420, 120, 50)
        self.delete_button = pygame.Rect(280, 420, 120, 50)
        self.back_button = pygame.Rect(410, 420, 120, 50)
        
        self.clock = pygame.time.Clock()
    
    def run(self):
        """Run the load menu and return save data or None."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP and self.selected_index > 0:
                        self.selected_index -= 1
                    elif event.key == pygame.K_DOWN and self.selected_index < len(self.saves) - 1:
                        self.selected_index += 1
                    elif event.key == pygame.K_RETURN and self.selected_index >= 0:
                        from reinforcetactics.utils.file_io import FileIO
                        return FileIO.load_game(str(self.saves[self.selected_index]))
                    elif event.key == pygame.K_DELETE and self.selected_index >= 0:
                        # Delete save
                        self.saves[self.selected_index].unlink()
                        self.saves.pop(self.selected_index)
                        if self.selected_index >= len(self.saves):
                            self.selected_index = len(self.saves) - 1
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.load_button.collidepoint(mouse_pos) and self.selected_index >= 0:
                            from reinforcetactics.utils.file_io import FileIO
                            return FileIO.load_game(str(self.saves[self.selected_index]))
                        
                        elif self.delete_button.collidepoint(mouse_pos) and self.selected_index >= 0:
                            self.saves[self.selected_index].unlink()
                            self.saves.pop(self.selected_index)
                            if self.selected_index >= len(self.saves):
                                self.selected_index = len(self.saves) - 1
                        
                        elif self.back_button.collidepoint(mouse_pos):
                            return None
                        
                        # Check list items
                        if self.list_rect.collidepoint(mouse_pos):
                            relative_y = mouse_pos[1] - self.list_rect.y + self.scroll_offset
                            clicked_index = int(relative_y // self.item_height)
                            if 0 <= clicked_index < len(self.saves):
                                self.selected_index = clicked_index
                
                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_offset = max(0, min(
                        self.scroll_offset - event.y * 20,
                        max(0, len(self.saves) * self.item_height - self.list_rect.height)
                    ))
            
            self.draw(mouse_pos)
            pygame.display.flip()
            self.clock.tick(60)
        
        return None
    
    def draw(self, mouse_pos):
        """Draw the load menu."""
        self.screen.fill(self.bg_color)
        
        # Title
        title = self.title_font.render("Load Game", True, (255, 215, 0))
        title_rect = title.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title, title_rect)
        
        # Draw saves list
        pygame.draw.rect(self.screen, self.list_bg_color, self.list_rect)
        pygame.draw.rect(self.screen, (200, 200, 220), self.list_rect, 2)
        
        if not self.saves:
            no_saves = self.list_font.render("No saved games found", True, (150, 150, 150))
            no_saves_rect = no_saves.get_rect(center=self.list_rect.center)
            self.screen.blit(no_saves, no_saves_rect)
        else:
            # Clip to list area
            clip_rect = self.screen.get_clip()
            self.screen.set_clip(self.list_rect)
            
            for i, save_path in enumerate(self.saves):
                y = self.list_rect.y + i * self.item_height - self.scroll_offset
                item_rect = pygame.Rect(self.list_rect.x, y, self.list_rect.width, self.item_height)
                
                if self.list_rect.colliderect(item_rect):
                    # Draw background
                    if i == self.selected_index:
                        pygame.draw.rect(self.screen, self.selected_color, item_rect)
                    elif item_rect.collidepoint(mouse_pos):
                        pygame.draw.rect(self.screen, self.hover_color, item_rect)
                    
                    # Draw save name
                    save_name = save_path.stem
                    name_surface = self.list_font.render(save_name, True, (255, 255, 255))
                    self.screen.blit(name_surface, (item_rect.x + 10, item_rect.y + 10))
                    
                    # Draw save info (modified time)
                    import time
                    mtime = save_path.stat().st_mtime
                    time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                    info_surface = self.info_font.render(time_str, True, (180, 180, 180))
                    self.screen.blit(info_surface, (item_rect.x + 10, item_rect.y + 35))
            
            self.screen.set_clip(clip_rect)
        
        # Buttons
        self._draw_button(self.load_button, "Load", mouse_pos, self.selected_index >= 0)
        self._draw_button(self.delete_button, "Delete", mouse_pos, self.selected_index >= 0)
        self._draw_button(self.back_button, "Back", mouse_pos, True)
    
    def _draw_button(self, rect, text, mouse_pos, enabled):
        """Draw a button."""
        if enabled:
            color = (100, 100, 130) if rect.collidepoint(mouse_pos) else (70, 70, 90)
            text_color = (255, 255, 255)
        else:
            color = (50, 50, 60)
            text_color = (100, 100, 100)
        
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (200, 200, 220), rect, 2)
        
        button_font = pygame.font.Font(None, 28)
        text_surface = button_font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)


class ReplaySelectionMenu:
    """Menu for selecting a replay to watch."""
    
    def __init__(self):
        """Initialize replay selection menu."""
        pygame.init()
        
        self.lang = get_language()
        self.settings = get_settings()
        
        self.width = 600
        self.height = 500
        self.screen = pygame.display.get_surface()
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
        
        pygame.display.set_caption("Watch Replay")
        
        # Get available replays
        from reinforcetactics.utils.file_io import FileIO
        self.replays = FileIO.list_replays()
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.list_bg_color = (50, 50, 60)
        self.selected_color = (100, 150, 100)
        self.hover_color = (70, 70, 90)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.list_font = pygame.font.Font(None, 28)
        self.info_font = pygame.font.Font(None, 22)
        
        # List setup
        self.list_rect = pygame.Rect(50, 100, 500, 300)
        self.item_height = 60
        self.scroll_offset = 0
        self.selected_index = 0 if self.replays else -1
        
        # Buttons
        self.watch_button = pygame.Rect(150, 420, 120, 50)
        self.delete_button = pygame.Rect(280, 420, 120, 50)
        self.back_button = pygame.Rect(410, 420, 120, 50)
        
        self.clock = pygame.time.Clock()
    
    def run(self):
        """Run the replay selection menu."""
        running = True
        
        while running:
            mouse_pos = pygame.mouse.get_pos()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key == pygame.K_UP and self.selected_index > 0:
                        self.selected_index -= 1
                    elif event.key == pygame.K_DOWN and self.selected_index < len(self.replays) - 1:
                        self.selected_index += 1
                    elif event.key == pygame.K_RETURN and self.selected_index >= 0:
                        return str(self.replays[self.selected_index])
                    elif event.key == pygame.K_DELETE and self.selected_index >= 0:
                        self.replays[self.selected_index].unlink()
                        self.replays.pop(self.selected_index)
                        if self.selected_index >= len(self.replays):
                            self.selected_index = len(self.replays) - 1
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.watch_button.collidepoint(mouse_pos) and self.selected_index >= 0:
                            return str(self.replays[self.selected_index])
                        
                        elif self.delete_button.collidepoint(mouse_pos) and self.selected_index >= 0:
                            self.replays[self.selected_index].unlink()
                            self.replays.pop(self.selected_index)
                            if self.selected_index >= len(self.replays):
                                self.selected_index = len(self.replays) - 1
                        
                        elif self.back_button.collidepoint(mouse_pos):
                            return None
                        
                        if self.list_rect.collidepoint(mouse_pos):
                            relative_y = mouse_pos[1] - self.list_rect.y + self.scroll_offset
                            clicked_index = int(relative_y // self.item_height)
                            if 0 <= clicked_index < len(self.replays):
                                self.selected_index = clicked_index
                
                elif event.type == pygame.MOUSEWHEEL:
                    self.scroll_offset = max(0, min(
                        self.scroll_offset - event.y * 20,
                        max(0, len(self.replays) * self.item_height - self.list_rect.height)
                    ))
            
            self.draw(mouse_pos)
            pygame.display.flip()
            self.clock.tick(60)
        
        return None
    
    def draw(self, mouse_pos):
        """Draw the replay selection menu."""
        self.screen.fill(self.bg_color)
        
        # Title
        title = self.title_font.render("Watch Replay", True, (255, 215, 0))
        title_rect = title.get_rect(center=(self.width // 2, 50))
        self.screen.blit(title, title_rect)
        
        # Draw replays list
        pygame.draw.rect(self.screen, self.list_bg_color, self.list_rect)
        pygame.draw.rect(self.screen, (200, 200, 220), self.list_rect, 2)
        
        if not self.replays:
            no_replays = self.list_font.render("No replays found", True, (150, 150, 150))
            no_replays_rect = no_replays.get_rect(center=self.list_rect.center)
            self.screen.blit(no_replays, no_replays_rect)
        else:
            clip_rect = self.screen.get_clip()
            self.screen.set_clip(self.list_rect)
            
            for i, replay_path in enumerate(self.replays):
                y = self.list_rect.y + i * self.item_height - self.scroll_offset
                item_rect = pygame.Rect(self.list_rect.x, y, self.list_rect.width, self.item_height)
                
                if self.list_rect.colliderect(item_rect):
                    if i == self.selected_index:
                        pygame.draw.rect(self.screen, self.selected_color, item_rect)
                    elif item_rect.collidepoint(mouse_pos):
                        pygame.draw.rect(self.screen, self.hover_color, item_rect)
                    
                    replay_name = replay_path.stem
                    name_surface = self.list_font.render(replay_name, True, (255, 255, 255))
                    self.screen.blit(name_surface, (item_rect.x + 10, item_rect.y + 10))
                    
                    import time
                    mtime = replay_path.stat().st_mtime
                    time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                    info_surface = self.info_font.render(time_str, True, (180, 180, 180))
                    self.screen.blit(info_surface, (item_rect.x + 10, item_rect.y + 35))
            
            self.screen.set_clip(clip_rect)
        
        # Buttons
        self._draw_button(self.watch_button, "Watch", mouse_pos, self.selected_index >= 0)
        self._draw_button(self.delete_button, "Delete", mouse_pos, self.selected_index >= 0)
        self._draw_button(self.back_button, "Back", mouse_pos, True)
    
    def _draw_button(self, rect, text, mouse_pos, enabled):
        """Draw a button."""
        if enabled:
            color = (100, 100, 130) if rect.collidepoint(mouse_pos) else (70, 70, 90)
            text_color = (255, 255, 255)
        else:
            color = (50, 50, 60)
            text_color = (100, 100, 100)
        
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, (200, 200, 220), rect, 2)
        
        button_font = pygame.font.Font(None, 28)
        text_surface = button_font.render(text, True, text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

class BuildingMenu:
    """Menu for creating units at buildings."""
    
    def __init__(self, game_state, building_pos):
        """
        Initialize building menu.
        
        Args:
            game_state: Current GameState
            building_pos: (x, y) tuple of building position
        """
        self.game_state = game_state
        self.building_pos = building_pos
        self.lang = get_language()
        
        # Menu dimensions
        self.width = 250
        self.height = 280
        
        # Position menu near building
        from reinforcetactics.constants import TILE_SIZE
        screen_x = building_pos[0] * TILE_SIZE + TILE_SIZE + 10
        screen_y = building_pos[1] * TILE_SIZE
        
        # Adjust if would go off screen
        if screen_x + self.width > game_state.grid.width * TILE_SIZE:
            screen_x = building_pos[0] * TILE_SIZE - self.width - 10
        if screen_y + self.height > game_state.grid.height * TILE_SIZE:
            screen_y = game_state.grid.height * TILE_SIZE - self.height
        
        self.rect = pygame.Rect(screen_x, screen_y, self.width, self.height)
        
        # Unit buttons
        button_width = 200
        button_height = 50
        button_x = self.rect.x + 25
        start_y = self.rect.y + 60
        
        self.unit_buttons = {
            'W': pygame.Rect(button_x, start_y, button_width, button_height),
            'M': pygame.Rect(button_x, start_y + 60, button_width, button_height),
            'C': pygame.Rect(button_x, start_y + 120, button_width, button_height)
        }
        
        self.close_button = pygame.Rect(button_x, start_y + 180, button_width, button_height)
    
    def handle_click(self, mouse_pos):
        """
        Handle mouse click on menu.
        
        Returns:
            dict with action type, or None if no action
        """
        if not self.rect.collidepoint(mouse_pos):
            # Clicked outside - close menu
            return {'type': 'close'}
        
        # Check unit buttons
        for unit_type, button_rect in self.unit_buttons.items():
            if button_rect.collidepoint(mouse_pos):
                return {
                    'type': 'create_unit',
                    'unit_type': unit_type,
                    'building_pos': self.building_pos
                }
        
        # Check close button
        if self.close_button.collidepoint(mouse_pos):
            return {'type': 'close'}
        
        return None
    
    def draw(self, screen):
        """Draw the building menu."""
        # Background
        pygame.draw.rect(screen, (40, 40, 50), self.rect)
        pygame.draw.rect(screen, (200, 200, 220), self.rect, 3)
        
        # Header
        header_rect = pygame.Rect(self.rect.x, self.rect.y, self.width, 50)
        pygame.draw.rect(screen, (70, 70, 90), header_rect)
        pygame.draw.rect(screen, (200, 200, 220), header_rect, 2)
        
        title_font = pygame.font.Font(None, 32)
        title_text = "Create Unit"
        title_surface = title_font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=header_rect.center)
        screen.blit(title_surface, title_rect)
        
        # Current gold display
        gold_font = pygame.font.Font(None, 24)
        player = self.game_state.current_player
        gold_text = f"Gold: ${self.game_state.player_gold[player]}"
        gold_surface = gold_font.render(gold_text, True, (255, 215, 0))
        gold_rect = gold_surface.get_rect(topright=(self.rect.right - 10, self.rect.y + 15))
        screen.blit(gold_surface, gold_rect)
        
        # Unit buttons
        mouse_pos = pygame.mouse.get_pos()
        
        unit_info = {
            'W': ('Warrior', '$200', (139, 69, 19)),
            'M': ('Mage', '$250', (0, 200, 0)),
            'C': ('Cleric', '$200', (128, 128, 128))
        }
        
        for unit_type, button_rect in self.unit_buttons.items():
            name, cost, color = unit_info[unit_type]
            
            # Check if can afford
            cost_value = int(cost[1:])
            can_afford = self.game_state.player_gold[player] >= cost_value
            
            # Button color
            if not can_afford:
                button_color = (60, 60, 70)
                text_color = (120, 120, 120)
            elif button_rect.collidepoint(mouse_pos):
                button_color = (90, 90, 110)
                text_color = (255, 255, 255)
            else:
                button_color = (70, 70, 90)
                text_color = (255, 255, 255)
            
            # Draw button
            pygame.draw.rect(screen, button_color, button_rect)
            pygame.draw.rect(screen, color if can_afford else (80, 80, 80), button_rect, 3)
            
            # Draw text
            name_font = pygame.font.Font(None, 28)
            name_surface = name_font.render(name, True, text_color)
            name_rect = name_surface.get_rect(midleft=(button_rect.x + 10, button_rect.centery - 8))
            screen.blit(name_surface, name_rect)
            
            cost_font = pygame.font.Font(None, 24)
            cost_surface = cost_font.render(cost, True, (255, 215, 0) if can_afford else (120, 120, 120))
            cost_rect = cost_surface.get_rect(midleft=(button_rect.x + 10, button_rect.centery + 12))
            screen.blit(cost_surface, cost_rect)
        
        # Close button
        close_color = (150, 70, 70) if self.close_button.collidepoint(mouse_pos) else (120, 50, 50)
        pygame.draw.rect(screen, close_color, self.close_button)
        pygame.draw.rect(screen, (200, 100, 100), self.close_button, 2)
        
        close_font = pygame.font.Font(None, 28)
        close_text = close_font.render("Cancel (ESC)", True, (255, 255, 255))
        close_rect = close_text.get_rect(center=self.close_button.center)
        screen.blit(close_text, close_rect)


class UnitActionMenu:
    """Menu for unit actions (attack, heal, seize, etc.)."""
    
    def __init__(self, unit, game_state):
        """
        Initialize unit action menu.
        
        Args:
            unit: Unit to show actions for
            game_state: Current GameState
        """
        self.unit = unit
        self.game_state = game_state
        self.lang = get_language()
        
        # Get available actions
        self.actions = self._get_available_actions()
        
        # Menu dimensions
        self.width = 200
        self.height = 60 + len(self.actions) * 45
        
        # Position menu near unit
        from reinforcetactics.constants import TILE_SIZE
        screen_x = unit.x * TILE_SIZE + TILE_SIZE + 10
        screen_y = unit.y * TILE_SIZE
        
        # Adjust if would go off screen
        if screen_x + self.width > game_state.grid.width * TILE_SIZE:
            screen_x = unit.x * TILE_SIZE - self.width - 10
        if screen_y + self.height > game_state.grid.height * TILE_SIZE:
            screen_y = game_state.grid.height * TILE_SIZE - self.height
        
        self.rect = pygame.Rect(screen_x, screen_y, self.width, self.height)
        
        # Action buttons
        self.action_buttons = {}
        button_y = self.rect.y + 50
        for action in self.actions:
            self.action_buttons[action] = pygame.Rect(
                self.rect.x + 10,
                button_y,
                self.width - 20,
                40
            )
            button_y += 45
    
    def _get_available_actions(self):
        """Get list of available actions for this unit."""
        actions = []
        
        # Always can wait
        actions.append('wait')
        
        # Check for adjacent enemies to attack
        if self.unit.can_attack:
            adjacent_enemies = self.game_state.mechanics.get_adjacent_enemies(
                self.unit, self.game_state.units
            )
            if adjacent_enemies:
                actions.append('attack')
        
        # Check for structure to seize
        tile = self.game_state.grid.get_tile(self.unit.x, self.unit.y)
        if tile.is_capturable() and tile.player != self.unit.player:
            actions.append('seize')
        
        # Cleric-specific actions
        if self.unit.type == 'C' and self.unit.can_attack:
            adjacent_allies = self.game_state.mechanics.get_adjacent_allies(
                self.unit, self.game_state.units
            )
            if adjacent_allies:
                actions.append('heal')
            
            adjacent_paralyzed = self.game_state.mechanics.get_adjacent_paralyzed_allies(
                self.unit, self.game_state.units
            )
            if adjacent_paralyzed:
                actions.append('cure')
        
        return actions
    
    def handle_click(self, mouse_pos):
        """Handle mouse click on menu."""
        if not self.rect.collidepoint(mouse_pos):
            return {'type': 'close'}
        
        for action, button_rect in self.action_buttons.items():
            if button_rect.collidepoint(mouse_pos):
                return {
                    'type': 'unit_action',
                    'action': action,
                    'unit': self.unit
                }
        
        return None
    
    def draw(self, screen):
        """Draw the action menu."""
        # Background
        pygame.draw.rect(screen, (40, 40, 50), self.rect)
        pygame.draw.rect(screen, (200, 200, 220), self.rect, 3)
        
        # Header
        header_rect = pygame.Rect(self.rect.x, self.rect.y, self.width, 40)
        player_color = PLAYER_COLORS.get(self.unit.player, (255, 255, 255))
        pygame.draw.rect(screen, player_color, header_rect)
        pygame.draw.rect(screen, (200, 200, 220), header_rect, 2)
        
        title_font = pygame.font.Font(None, 28)
        title_text = "Unit Actions"
        title_surface = title_font.render(title_text, True, (255, 255, 255))
        title_rect = title_surface.get_rect(center=header_rect.center)
        screen.blit(title_surface, title_rect)
        
        # Action buttons
        mouse_pos = pygame.mouse.get_pos()
        button_font = pygame.font.Font(None, 24)
        
        action_labels = {
            'wait': 'Wait',
            'attack': 'Attack',
            'seize': 'Seize',
            'heal': 'Heal',
            'cure': 'Cure'
        }
        
        for action, button_rect in self.action_buttons.items():
            is_hover = button_rect.collidepoint(mouse_pos)
            button_color = (90, 90, 110) if is_hover else (70, 70, 90)
            
            pygame.draw.rect(screen, button_color, button_rect)
            pygame.draw.rect(screen, (200, 200, 220), button_rect, 2)
            
            label = action_labels.get(action, action.capitalize())
            text_surface = button_font.render(label, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=button_rect.center)
            screen.blit(text_surface, text_rect)