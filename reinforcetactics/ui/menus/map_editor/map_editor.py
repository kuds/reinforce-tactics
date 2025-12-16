"""Main map editor class that coordinates all components."""
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import pygame
import pandas as pd
import numpy as np

from reinforcetactics.ui.menus.map_editor.tile_palette import TilePalette
from reinforcetactics.ui.menus.map_editor.editor_canvas import EditorCanvas
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.constants import MIN_MAP_SIZE


class MapEditor:
    """Main map editor class."""

    def __init__(self, screen: Optional[pygame.Surface] = None,
                 map_data: Optional[pd.DataFrame] = None,
                 map_filename: Optional[str] = None,
                 num_players: int = 2) -> None:
        """
        Initialize the map editor.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            map_data: Optional map data to edit. If None, creates empty map.
            map_filename: Optional filename for the map being edited
            num_players: Number of players (2-4)
        """
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()
        
        # Create screen if not provided
        self.owns_screen = screen is None
        if self.owns_screen:
            self.screen = pygame.display.set_mode((1500, 1000))
            pygame.display.set_caption("Map Editor - Reinforce Tactics")
        else:
            self.screen = screen
        
        self.map_data = map_data
        self.map_filename = map_filename
        self.num_players = num_players
        self.running = True
        self.modified = False
        
        # Initialize components
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Tile palette on the right side
        palette_width = 250
        palette_height = screen_height - 150
        self.palette = TilePalette(
            screen_width - palette_width - 20,
            100,
            palette_width,
            palette_height,
            num_players
        )
        
        # Canvas on the left side
        canvas_width = screen_width - palette_width - 60
        canvas_height = screen_height - 150
        self.canvas = EditorCanvas(
            20,
            100,
            canvas_width,
            canvas_height,
            self.map_data
        )
        
        # Fonts
        self.title_font = get_font(28)
        self.info_font = get_font(18)
        self.shortcut_font = get_font(14)
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.title_color = (100, 200, 255)
        
        # Keyboard state
        self.ctrl_pressed = False

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the map editor loop.

        Returns:
            Result dict or None
        """
        clock = pygame.time.Clock()
        
        # Clear any residual events
        pygame.event.clear()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if self.modified:
                        # TODO: Add save confirmation dialog
                        pass
                    return {'type': 'exit'}
                
                self._handle_event(event)
            
            self.draw()
            clock.tick(30)
        
        return None

    def _handle_event(self, event: pygame.event.Event) -> None:
        """
        Handle pygame event.

        Args:
            event: Pygame event
        """
        if event.type == pygame.KEYDOWN:
            self._handle_keydown(event)
        elif event.type == pygame.KEYUP:
            self._handle_keyup(event)
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._handle_mouse_up(event)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """
        Handle key down event.

        Args:
            event: Pygame event
        """
        # Track Ctrl key
        if event.key in (pygame.K_LCTRL, pygame.K_RCTRL):
            self.ctrl_pressed = True
        
        # Handle shortcuts
        if self.ctrl_pressed:
            if event.key == pygame.K_s:
                self._save_map()
            elif event.key == pygame.K_n:
                # TODO: New map with confirmation if modified
                pass
            elif event.key == pygame.K_o:
                # TODO: Open map with confirmation if modified
                pass
        elif event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_g:
            self.canvas.toggle_grid()
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.canvas.zoom_in()
        elif event.key == pygame.K_MINUS:
            self.canvas.zoom_out()
        # Quick tile selection with number keys
        elif event.key == pygame.K_1:
            self.palette.selected_tile = 'p'  # Grass
        elif event.key == pygame.K_2:
            self.palette.selected_tile = 'o'  # Ocean
        elif event.key == pygame.K_3:
            self.palette.selected_tile = 'w'  # Water
        elif event.key == pygame.K_4:
            self.palette.selected_tile = 'm'  # Mountain
        elif event.key == pygame.K_5:
            self.palette.selected_tile = 'f'  # Forest
        elif event.key == pygame.K_6:
            self.palette.selected_tile = 'r'  # Road
        elif event.key == pygame.K_7:
            self.palette.selected_tile = 't'  # Tower
        elif event.key == pygame.K_8:
            self.palette.selected_tile = 'b'  # Building
        elif event.key == pygame.K_9:
            self.palette.selected_tile = 'h'  # Headquarters

    def _handle_keyup(self, event: pygame.event.Event) -> None:
        """
        Handle key up event.

        Args:
            event: Pygame event
        """
        if event.key in (pygame.K_LCTRL, pygame.K_RCTRL):
            self.ctrl_pressed = False

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        """
        Handle mouse motion event.

        Args:
            event: Pygame event
        """
        self.canvas.handle_mouse_move(event.pos)
        
        # If painting and mouse is down, continue painting
        if self.canvas.is_painting and event.buttons[0]:  # Left button
            tile_code = self.palette.get_selected_tile()
            if self.canvas.handle_mouse_click(event.pos, tile_code):
                self.modified = True

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        """
        Handle mouse button down event.

        Args:
            event: Pygame event
        """
        if event.button == 1:  # Left click
            # Check palette first
            if self.palette.handle_click(event.pos):
                return
            
            # Then check canvas for painting
            tile_code = self.palette.get_selected_tile()
            if self.canvas.handle_mouse_click(event.pos, tile_code):
                self.modified = True
        
        elif event.button == 3:  # Right click - erase
            if self.canvas.handle_mouse_click(event.pos, 'p'):  # Replace with grass
                self.modified = True
        
        elif event.button == 4:  # Mouse wheel up
            self.canvas.handle_scroll(0, -32)
        
        elif event.button == 5:  # Mouse wheel down
            self.canvas.handle_scroll(0, 32)

    def _handle_mouse_up(self, event: pygame.event.Event) -> None:
        """
        Handle mouse button up event.

        Args:
            event: Pygame event
        """
        if event.button == 1:  # Left click
            self.canvas.handle_mouse_release()

    def _save_map(self) -> bool:
        """
        Save the current map.

        Returns:
            True if saved successfully, False otherwise
        """
        lang = get_language()
        
        # Validate map before saving
        validation_errors = self._validate_map()
        if validation_errors:
            print("Map validation failed:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        
        # Determine save path
        if self.map_filename:
            save_path = Path(self.map_filename)
        else:
            # Ask for filename (for now, use a default)
            # Determine directory based on player count
            map_type = "1v1" if self.num_players == 2 else "2v2"
            save_dir = Path(f"maps/{map_type}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"custom_map_{timestamp}.csv"
        
        # Save the map
        result = FileIO.save_map(self.map_data, str(save_path))
        if result:
            self.modified = False
            self.map_filename = str(save_path)
            print(f"✅ Map saved: {save_path}")
            return True
        else:
            print(f"❌ Failed to save map")
            return False

    def _validate_map(self) -> list:
        """
        Validate the map.

        Returns:
            List of validation errors (empty if valid)
        """
        lang = get_language()
        errors = []
        
        # Check minimum size
        height, width = self.map_data.shape
        if width < MIN_MAP_SIZE or height < MIN_MAP_SIZE:
            errors.append(
                lang.get('map_editor.validation.min_size', 'Map must be at least {size}x{size}')
                .format(size=MIN_MAP_SIZE)
            )
        
        # Check headquarters for each player
        for player_num in range(1, self.num_players + 1):
            hq_code = f"h_{player_num}"
            hq_count = (self.map_data == hq_code).sum().sum()
            
            if hq_count != 1:
                errors.append(
                    lang.get('map_editor.validation.invalid_hq',
                            'Player {player} needs exactly one Headquarters')
                    .format(player=player_num)
                )
        
        return errors

    def draw(self) -> None:
        """Draw the map editor."""
        self.screen.fill(self.bg_color)
        
        lang = get_language()
        screen_width = self.screen.get_width()
        
        # Draw title
        title_text = lang.get('map_editor.title', 'Map Editor')
        if self.map_filename:
            title_text += f" - {Path(self.map_filename).name}"
        if self.modified:
            title_text += " *"
        
        title_surface = self.title_font.render(title_text, True, self.title_color)
        title_rect = title_surface.get_rect(x=20, y=20)
        self.screen.blit(title_surface, title_rect)
        
        # Draw keyboard shortcuts
        shortcuts_x = 20
        shortcuts_y = 60
        shortcuts = [
            f"[Ctrl+S] {lang.get('map_editor.shortcuts.save', 'Save')}",
            f"[G] {lang.get('map_editor.shortcuts.grid', 'Toggle Grid')}",
            f"[Esc] {lang.get('map_editor.shortcuts.esc', 'Exit')}",
            f"[1-9] Quick tile select",
            f"[+/-] Zoom",
        ]
        
        current_x = shortcuts_x
        for shortcut in shortcuts:
            shortcut_surface = self.shortcut_font.render(shortcut, True, (180, 180, 180))
            self.screen.blit(shortcut_surface, (current_x, shortcuts_y))
            current_x += shortcut_surface.get_width() + 20
        
        # Draw canvas
        self.canvas.draw(self.screen)
        
        # Draw palette
        self.palette.draw(self.screen)
        
        # Draw selected tile info
        selected_tile = self.palette.get_selected_tile()
        info_text = f"Selected: {selected_tile}"
        info_surface = self.info_font.render(info_text, True, self.text_color)
        info_rect = info_surface.get_rect(x=screen_width - 270, y=20)
        self.screen.blit(info_surface, info_rect)
        
        pygame.display.flip()
