"""In-game overlay menu for unit actions during gameplay."""

from typing import Any

import pygame

from reinforcetactics.constants import TILE_SIZE
from reinforcetactics.game.mechanics import GameMechanics
from reinforcetactics.ui import theme, widgets
from reinforcetactics.utils.fonts import get_display_font, get_font


class UnitActionMenu:
    """In-game overlay menu for unit actions."""

    def __init__(self, screen: pygame.Surface, game_state: Any, unit: Any) -> None:
        """
        Initialize unit action menu.

        Args:
            screen: Pygame surface to draw on
            game_state: Game state object
            unit: The unit to show actions for
        """
        self.screen = screen
        self.game_state = game_state
        self.unit = unit
        self.running = True

        # Colors (from shared theme)
        self.bg_color = theme.PANEL_BG
        self.text_color = theme.TEXT
        self.border_color = theme.BORDER

        # Fonts
        self.title_font = get_display_font(24)
        self.option_font = get_font(24)

        # Interactive elements
        self.interactive_elements: list[dict[str, Any]] = []
        self.hover_element: dict[str, Any] | None = None

        # Cached overlay surface to avoid per-frame allocation
        self._overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        self._overlay.fill((0, 0, 0, 100))

        # Calculate available actions
        self.actions = self._calculate_available_actions()

        # Calculate menu position and size
        self._calculate_menu_rect()

    def _calculate_available_actions(self) -> list[dict[str, Any]]:
        """
        Calculate which actions are available for this unit.

        Returns:
            List of action dictionaries with 'name', 'key', 'type', and optionally 'targets'
        """
        actions = []

        # Check for attackable enemies (using range-aware method)
        attackable_enemies = GameMechanics.get_attackable_enemies(self.unit, self.game_state.units, self.game_state.grid)

        # Attack - available if attackable enemies exist
        if attackable_enemies:
            actions.append({"name": "Attack (A)", "key": "a", "type": "attack", "targets": attackable_enemies})

        # Paralyze - only for Mages with adjacent enemies
        if self.unit.type == "M":
            adjacent_enemies = GameMechanics.get_adjacent_enemies(self.unit, self.game_state.units)
            if adjacent_enemies:
                actions.append({"name": "Paralyze (P)", "key": "p", "type": "paralyze", "targets": adjacent_enemies})

        # Heal and Cure - only for Clerics
        if self.unit.type == "C":
            # Heal - damaged allies within CLERIC_HEAL_RANGE
            healable_allies = GameMechanics.get_healable_allies(self.unit, self.game_state.units)
            if healable_allies:
                actions.append({"name": "Heal (H)", "key": "h", "type": "heal", "targets": healable_allies})

            # Cure - paralyzed allies within CLERIC_HEAL_RANGE
            curable_allies = GameMechanics.get_curable_allies(self.unit, self.game_state.units)
            if curable_allies:
                actions.append({"name": "Cure (C)", "key": "c", "type": "cure", "targets": curable_allies})

        # Haste - only for Sorcerers with ability off cooldown
        if self.unit.type == "S" and self.unit.can_use_haste():
            hasteable_allies = GameMechanics.get_hasteable_allies(self.unit, self.game_state.units)
            if hasteable_allies:
                actions.append({"name": "Haste (T)", "key": "t", "type": "haste", "targets": hasteable_allies})

        # Defence Buff - only for Sorcerers with ability off cooldown
        if self.unit.type == "S" and self.unit.can_use_defence_buff():
            buffable_allies = GameMechanics.get_defence_buffable_allies(self.unit, self.game_state.units)
            if buffable_allies:
                actions.append({"name": "Defence Buff (D)", "key": "d", "type": "defence_buff", "targets": buffable_allies})

        # Attack Buff - only for Sorcerers with ability off cooldown
        if self.unit.type == "S" and self.unit.can_use_attack_buff():
            buffable_allies = GameMechanics.get_attack_buffable_allies(self.unit, self.game_state.units)
            if buffable_allies:
                actions.append({"name": "Attack Buff (B)", "key": "b", "type": "attack_buff", "targets": buffable_allies})

        # Capture - only if on a capturable structure
        tile = self.game_state.grid.get_tile(self.unit.x, self.unit.y)
        if tile.is_capturable() and tile.player != self.unit.player:
            actions.append({"name": "Capture (S)", "key": "s", "type": "capture", "targets": None})

        # Cancel Move - only if unit has moved this turn
        if self.unit.has_moved:
            actions.append({"name": "Cancel Move (M)", "key": "m", "type": "cancel_move", "targets": None})

        # Wait/End Turn - always available
        actions.append({"name": "Wait/End Turn (W)", "key": "w", "type": "wait", "targets": None})

        return actions

    def _calculate_menu_rect(self) -> None:
        """Calculate the menu rectangle position and size."""
        # Menu dimensions based on number of actions
        menu_width = 240
        action_height = 35
        header_height = 50
        menu_height = header_height + len(self.actions) * action_height + 20

        # Convert unit position to screen coordinates
        unit_screen_x = self.unit.x * TILE_SIZE
        unit_screen_y = self.unit.y * TILE_SIZE

        # Position menu to the right of the unit
        menu_x = unit_screen_x + TILE_SIZE + 10
        menu_y = unit_screen_y

        # Ensure menu stays within screen bounds
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        if menu_x + menu_width > screen_width:
            # Position to the left instead
            menu_x = unit_screen_x - menu_width - 10

        if menu_y + menu_height > screen_height:
            # Position higher
            menu_y = screen_height - menu_height - 10

        # Ensure minimum position
        menu_x = max(10, menu_x)
        menu_y = max(10, menu_y)

        self.menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)

    def handle_click(self, mouse_pos: tuple[int, int]) -> dict[str, Any] | None:
        """
        Handle mouse clicks.

        Args:
            mouse_pos: (x, y) tuple of mouse position

        Returns:
            Dict with action result, or None
        """
        # Check if click is outside menu (close menu and cancel)
        if not self.menu_rect.collidepoint(mouse_pos):
            return {"type": "cancel"}

        # Check interactive elements
        for element in self.interactive_elements:
            if element["rect"].collidepoint(mouse_pos):
                if element["type"] == "close_button":
                    return {"type": "cancel"}
                if element["type"] == "action_button":
                    action = element["action"]
                    return {"type": "action_selected", "action": action}

        return None

    def handle_keydown(self, event: pygame.event.Event) -> dict[str, Any] | None:
        """
        Handle keyboard input.

        Args:
            event: Pygame keyboard event

        Returns:
            Dict with action result, or None
        """
        if event.key == pygame.K_ESCAPE:
            return {"type": "cancel"}

        # Check keyboard shortcuts
        key_char = pygame.key.name(event.key).lower()
        for action in self.actions:
            if action["key"] == key_char:
                return {"type": "action_selected", "action": action}

        return None

    def handle_mouse_motion(self, mouse_pos: tuple[int, int]) -> None:
        """
        Handle mouse motion for hover effects.

        Args:
            mouse_pos: (x, y) tuple of mouse position
        """
        self.hover_element = None
        for element in self.interactive_elements:
            if element["rect"].collidepoint(mouse_pos):
                self.hover_element = element
                break

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the unit action menu.

        Args:
            screen: Pygame surface to draw on
        """
        self.interactive_elements = []

        # Draw semi-transparent background overlay for entire screen (modal style)
        screen.blit(self._overlay, (0, 0))

        # Draw menu background
        pygame.draw.rect(screen, self.bg_color, self.menu_rect, border_radius=10)
        pygame.draw.rect(screen, self.border_color, self.menu_rect, width=2, border_radius=10)

        # Draw title
        title = "Unit Actions"
        title_surface = self.title_font.render(title, True, self.text_color)
        title_rect = title_surface.get_rect(centerx=self.menu_rect.centerx, y=self.menu_rect.y + 10)
        screen.blit(title_surface, title_rect)

        # Draw close button (X) in upper right
        close_button = widgets.CloseButton(self.menu_rect.right - widgets.CloseButton.SIZE - 10, self.menu_rect.y + 10)
        is_close_hover = bool(self.hover_element and self.hover_element.get("type") == "close_button")
        close_button.draw(screen, hovered=is_close_hover)
        self.interactive_elements.append({"type": "close_button", "rect": close_button.rect})

        # Draw action options
        start_y = self.menu_rect.y + 50
        spacing = 35

        for i, action in enumerate(self.actions):
            # Use the full panel width (minus margins); labels that still
            # don't fit are ellipsized by Button.draw instead of spilling.
            button_rect = pygame.Rect(self.menu_rect.x + 15, start_y + i * spacing, self.menu_rect.width - 30, 28)
            button = widgets.Button(
                button_rect,
                action["name"],
                self.option_font,
                style=widgets.MENU_OPTION_SMALL,
                text_align="left",
            )

            is_hovered = bool(self.hover_element and self.hover_element.get("rect") == button_rect)
            button.draw(screen, hovered=is_hovered)

            # Register as interactive element
            self.interactive_elements.append({"type": "action_button", "rect": button_rect, "action": action})
