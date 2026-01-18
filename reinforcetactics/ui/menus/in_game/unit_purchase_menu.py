"""In-game overlay menu for purchasing units on buildings."""
from typing import Optional, Dict, Any, List, Tuple

import pygame

from reinforcetactics.constants import TILE_SIZE, UNIT_DATA
from reinforcetactics.utils.fonts import get_font


class UnitPurchaseMenu:
    """In-game overlay menu for purchasing units on buildings."""

    def __init__(self, screen: pygame.Surface, game_state: Any, building_pos: Tuple[int, int]) -> None:
        """
        Initialize unit purchase menu.

        Args:
            screen: Pygame surface to draw on
            game_state: Game state object
            building_pos: (x, y) tuple of building position in grid coordinates
        """
        self.screen = screen
        self.game_state = game_state
        self.building_pos = building_pos
        self.running = True

        # Colors
        self.bg_color = (40, 40, 50)
        self.text_color = (255, 255, 255)
        self.hover_color = (200, 180, 100)
        self.disabled_color = (100, 100, 120)
        self.disabled_bg_color = (60, 60, 70)
        self.border_color = (100, 150, 200)
        self.close_button_color = (200, 50, 50)
        self.close_button_hover_color = (255, 80, 80)

        # Fonts
        self.title_font = get_font(28)
        self.option_font = get_font(24)

        # Unit types to display
        # Basic: Warrior, Mage, Cleric, Archer
        # Advanced: Knight, Rogue, Sorcerer, Barbarian
        self.unit_types = ['W', 'M', 'C', 'A', 'K', 'R', 'S', 'B']

        # Interactive elements
        self.interactive_elements: List[Dict[str, Any]] = []
        self.hover_element = None

        # Calculate menu position and size
        self._calculate_menu_rect()

    def _calculate_menu_rect(self) -> None:
        """Calculate the menu rectangle position and size."""
        # Menu dimensions - adjust height for number of units
        menu_width = 220
        menu_height = 50 + len(self.unit_types) * 35 + 20  # header + units + padding

        # Convert building position to screen coordinates
        building_screen_x = self.building_pos[0] * TILE_SIZE
        building_screen_y = self.building_pos[1] * TILE_SIZE

        # Position menu to the right of the building
        menu_x = building_screen_x + TILE_SIZE + 10
        menu_y = building_screen_y

        # Ensure menu stays within screen bounds
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        if menu_x + menu_width > screen_width:
            # Position to the left instead
            menu_x = building_screen_x - menu_width - 10

        if menu_y + menu_height > screen_height:
            # Position higher
            menu_y = screen_height - menu_height - 10

        # Ensure minimum position
        menu_x = max(10, menu_x)
        menu_y = max(10, menu_y)

        self.menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)

    def handle_click(self, mouse_pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Handle mouse clicks.

        Args:
            mouse_pos: (x, y) tuple of mouse position

        Returns:
            Dict with action result, or None
        """
        # Check if click is outside menu (close menu)
        if not self.menu_rect.collidepoint(mouse_pos):
            return {'type': 'close'}

        # Check interactive elements
        for element in self.interactive_elements:
            if element['rect'].collidepoint(mouse_pos):
                if element['type'] == 'close_button':
                    return {'type': 'close'}
                elif element['type'] == 'unit_button' and not element['disabled']:
                    unit_type = element['unit_type']
                    # Try to create the unit
                    unit = self.game_state.create_unit(
                        unit_type,
                        self.building_pos[0],
                        self.building_pos[1]
                    )
                    if unit:
                        return {'type': 'unit_created', 'unit': unit}
                    # Failed to create - position occupied or insufficient gold
                    # This shouldn't happen if UI logic is correct
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("Failed to create unit %s at %s", unit_type, self.building_pos)
                    return None

        return None

    def handle_mouse_motion(self, mouse_pos: Tuple[int, int]) -> None:
        """
        Handle mouse motion for hover effects.

        Args:
            mouse_pos: (x, y) tuple of mouse position
        """
        self.hover_element = None
        for element in self.interactive_elements:
            if element['rect'].collidepoint(mouse_pos):
                self.hover_element = element
                break

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the unit purchase menu.

        Args:
            screen: Pygame surface to draw on
        """
        self.interactive_elements = []

        # Draw semi-transparent background overlay for entire screen (to show menu is modal)
        overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        screen.blit(overlay, (0, 0))

        # Draw menu background
        pygame.draw.rect(screen, self.bg_color, self.menu_rect, border_radius=10)
        pygame.draw.rect(screen, self.border_color, self.menu_rect, width=2, border_radius=10)

        # Draw title
        title = "Purchase Unit"
        title_surface = self.title_font.render(title, True, self.text_color)
        title_rect = title_surface.get_rect(
            centerx=self.menu_rect.centerx,
            y=self.menu_rect.y + 10
        )
        screen.blit(title_surface, title_rect)

        # Draw close button (X) in upper right
        close_button_size = 20
        close_button_x = self.menu_rect.right - close_button_size - 10
        close_button_y = self.menu_rect.y + 10
        close_button_rect = pygame.Rect(
            close_button_x,
            close_button_y,
            close_button_size,
            close_button_size
        )

        # Check if hovering over close button
        is_close_hover = (self.hover_element and
                         self.hover_element.get('type') == 'close_button')
        close_color = self.close_button_hover_color if is_close_hover else self.close_button_color

        pygame.draw.rect(screen, close_color, close_button_rect, border_radius=3)

        # Draw X
        x_margin = 4
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (close_button_rect.left + x_margin, close_button_rect.top + x_margin),
            (close_button_rect.right - x_margin, close_button_rect.bottom - x_margin),
            2
        )
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (close_button_rect.right - x_margin, close_button_rect.top + x_margin),
            (close_button_rect.left + x_margin, close_button_rect.bottom - x_margin),
            2
        )

        self.interactive_elements.append({
            'type': 'close_button',
            'rect': close_button_rect
        })

        # Draw unit options
        start_y = self.menu_rect.y + 50
        spacing = 35
        current_player = self.game_state.current_player
        player_gold = self.game_state.player_gold[current_player]

        for i, unit_type in enumerate(self.unit_types):
            unit_data = UNIT_DATA[unit_type]
            unit_name = unit_data['name']
            unit_cost = unit_data['cost']

            # Check if player can afford
            can_afford = player_gold >= unit_cost

            y_pos = start_y + i * spacing

            # Draw unit option
            button_width = 190
            button_height = 28
            button_x = self.menu_rect.x + 15
            button_rect = pygame.Rect(button_x, y_pos, button_width, button_height)

            # Check if hovering
            is_hovered = (self.hover_element and
                         self.hover_element.get('rect') == button_rect and
                         can_afford)

            # Choose colors
            if not can_afford:
                bg_color = self.disabled_bg_color
                text_color = self.disabled_color
            elif is_hovered:
                bg_color = (70, 70, 90)
                text_color = self.hover_color
            else:
                bg_color = (50, 50, 65)
                text_color = self.text_color

            # Draw button background
            pygame.draw.rect(screen, bg_color, button_rect, border_radius=5)

            if is_hovered:
                pygame.draw.rect(screen, self.hover_color, button_rect, width=2, border_radius=5)

            # Draw text: "Unit Name - Cost"
            text = f"{unit_name} - {unit_cost}g"
            text_surface = self.option_font.render(text, True, text_color)
            text_rect = text_surface.get_rect(
                left=button_rect.left + 10,
                centery=button_rect.centery
            )
            screen.blit(text_surface, text_rect)

            # Register as interactive element
            self.interactive_elements.append({
                'type': 'unit_button',
                'rect': button_rect,
                'unit_type': unit_type,
                'disabled': not can_afford
            })
