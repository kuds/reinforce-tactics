"""Dialog for creating a new map with custom dimensions."""
from typing import Optional, Dict, Any
import pygame
from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font
from reinforcetactics.constants import MIN_MAP_SIZE


class NewMapDialog(Menu):
    """Dialog for creating a new map with custom dimensions."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize the new map dialog.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        lang = get_language()
        super().__init__(screen, lang.get('map_editor.new_map_dialog.title', 'Create New Map'))
        
        # Input fields
        self.width_value = MIN_MAP_SIZE
        self.height_value = MIN_MAP_SIZE
        self.num_players = 2
        self.active_field = 'width'  # 'width', 'height', or 'players'
        
        # Fonts
        self.label_font = get_font(32)
        self.value_font = get_font(36)
        self.info_font = get_font(24)
        
        # Colors
        self.active_color = (255, 200, 50)
        self.inactive_color = (150, 150, 150)
        
        # Setup options
        self._setup_options()

    def _setup_options(self) -> None:
        """Setup menu options."""
        lang = get_language()
        self.add_option(lang.get('map_editor.new_map_dialog.create', 'Create'), self._create)
        self.add_option(lang.get('common.cancel', 'Cancel'), lambda: None)

    def _create(self) -> Dict[str, Any]:
        """Create a new map with specified dimensions."""
        return {
            'width': max(self.width_value, MIN_MAP_SIZE),
            'height': max(self.height_value, MIN_MAP_SIZE),
            'num_players': self.num_players
        }

    def handle_input(self, event: pygame.event.Event) -> Optional[Any]:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            Result of selected option callback, if any
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                # Cycle through fields
                if self.active_field == 'width':
                    self.active_field = 'height'
                elif self.active_field == 'height':
                    self.active_field = 'players'
                else:
                    self.active_field = 'width'
                return None
            elif event.key == pygame.K_UP:
                # Increase value
                if self.active_field == 'width':
                    self.width_value = min(100, self.width_value + 1)
                elif self.active_field == 'height':
                    self.height_value = min(100, self.height_value + 1)
                elif self.active_field == 'players':
                    self.num_players = min(4, self.num_players + 1)
                return None
            elif event.key == pygame.K_DOWN:
                # Decrease value
                if self.active_field == 'width':
                    self.width_value = max(MIN_MAP_SIZE, self.width_value - 1)
                elif self.active_field == 'height':
                    self.height_value = max(MIN_MAP_SIZE, self.height_value - 1)
                elif self.active_field == 'players':
                    self.num_players = max(2, self.num_players - 1)
                return None
            elif event.key == pygame.K_RETURN:
                # Only create if we're not on a button
                if self.active_field in ('width', 'height', 'players'):
                    return self._create()
        
        # Handle menu navigation for buttons
        return super().handle_input(event)

    def draw(self) -> None:
        """Draw the dialog."""
        self.screen.fill(self.bg_color)
        
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        lang = get_language()
        
        # Draw title
        title_surface = self.title_font.render(self.title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
        self.screen.blit(title_surface, title_rect)
        
        # Draw info text
        info_text = lang.get(
            'map_editor.new_map_dialog.min_size',
            'Minimum size: {size}x{size}'
        ).format(size=MIN_MAP_SIZE)
        info_surface = self.info_font.render(info_text, True, self.text_color)
        info_rect = info_surface.get_rect(centerx=screen_width // 2, y=120)
        self.screen.blit(info_surface, info_rect)
        
        # Draw input fields
        start_y = 180
        spacing = 80
        
        # Width field
        self._draw_field(
            lang.get('map_editor.new_map_dialog.width', 'Width:'),
            str(self.width_value),
            screen_width // 2,
            start_y,
            self.active_field == 'width'
        )
        
        # Height field
        self._draw_field(
            lang.get('map_editor.new_map_dialog.height', 'Height:'),
            str(self.height_value),
            screen_width // 2,
            start_y + spacing,
            self.active_field == 'height'
        )
        
        # Players field
        self._draw_field(
            lang.get('map_editor.new_map_dialog.players', 'Players:'),
            str(self.num_players),
            screen_width // 2,
            start_y + spacing * 2,
            self.active_field == 'players'
        )
        
        # Draw controls hint
        hint_text = "Use UP/DOWN to adjust, TAB to switch fields, ENTER to create"
        hint_surface = self.info_font.render(hint_text, True, (180, 180, 180))
        hint_rect = hint_surface.get_rect(centerx=screen_width // 2, y=start_y + spacing * 3)
        self.screen.blit(hint_surface, hint_rect)
        
        # Draw options (buttons)
        self._draw_options(start_y + spacing * 3 + 60)
        
        pygame.display.flip()

    def _draw_field(self, label: str, value: str, center_x: int, y: int, is_active: bool) -> None:
        """
        Draw an input field.

        Args:
            label: Field label
            value: Field value
            center_x: Center X position
            y: Y position
            is_active: Whether this field is active
        """
        color = self.active_color if is_active else self.inactive_color
        
        # Draw label
        label_surface = self.label_font.render(label, True, color)
        label_rect = label_surface.get_rect(right=center_x - 20, centery=y)
        self.screen.blit(label_surface, label_rect)
        
        # Draw value with background
        value_surface = self.value_font.render(value, True, self.text_color)
        value_rect = value_surface.get_rect(left=center_x + 20, centery=y)
        
        # Draw background rectangle
        bg_rect = pygame.Rect(
            value_rect.x - 10,
            value_rect.y - 5,
            value_rect.width + 20,
            value_rect.height + 10
        )
        bg_color = self.option_bg_selected_color if is_active else self.option_bg_color
        pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=5)
        
        if is_active:
            pygame.draw.rect(self.screen, color, bg_rect, width=2, border_radius=5)
        
        self.screen.blit(value_surface, value_rect)

    def _draw_options(self, start_y: int) -> None:
        """
        Draw menu options (buttons).

        Args:
            start_y: Starting Y position for options
        """
        screen_width = self.screen.get_width()
        spacing = 60
        self.option_rects = []
        
        # Calculate maximum option width for uniform sizing
        padding_x = 40
        padding_y = 10
        max_text_width = 0
        for text, _ in self.options:
            text_surface = self.option_font.render(text, True, self.text_color)
            max_text_width = max(max_text_width, text_surface.get_width())
        
        uniform_width = max_text_width + 2 * padding_x
        
        # Draw each option
        for i, (text, _) in enumerate(self.options):
            is_selected = i == self.selected_index
            is_hovered = i == self.hover_index
            
            # Choose colors
            if is_selected:
                text_color = self.selected_color
                bg_color = self.option_bg_selected_color
            elif is_hovered:
                text_color = self.hover_color
                bg_color = self.option_bg_hover_color
            else:
                text_color = self.text_color
                bg_color = self.option_bg_color
            
            # Render text
            text_surface = self.option_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(centerx=screen_width // 2, y=start_y + i * spacing)
            
            # Create background rectangle
            bg_rect = pygame.Rect(
                (screen_width - uniform_width) // 2,
                text_rect.y - padding_y,
                uniform_width,
                text_rect.height + 2 * padding_y
            )
            
            # Draw background
            pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=8)
            
            # Draw border for selected/hovered
            if is_selected or is_hovered:
                border_color = self.selected_color if is_selected else self.hover_color
                pygame.draw.rect(self.screen, border_color, bg_rect, width=2, border_radius=8)
            
            # Draw text
            self.screen.blit(text_surface, text_rect)
            
            # Store rect for click detection
            self.option_rects.append(bg_rect)
