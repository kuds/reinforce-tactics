"""
Input Handler for Reinforce Tactics.

This module manages user input state and event handling for the game loop.
"""

import pygame
from reinforcetactics.constants import TILE_SIZE
from reinforcetactics.ui.menus import UnitActionMenu, UnitPurchaseMenu, ConfirmationDialog
from game.action_executor import handle_action_menu_result


class InputHandler:
    """
    Manages input state and event handling for the game loop.

    Attributes:
        game: The GameState instance
        renderer: The Renderer instance
        bots: Dictionary mapping player numbers to bot instances
        num_players: Total number of players
        selected_unit: Currently selected unit
        active_menu: Currently active menu
        menu_opened_time: Timestamp when menu was opened (milliseconds)
        target_selection_mode: Whether we're in target selection mode
        target_selection_action: The action waiting for target selection
        target_selection_unit: The unit performing the action
    """

    def __init__(self, game, renderer, bots, num_players):
        """
        Initialize the InputHandler.

        Args:
            game: The GameState instance
            renderer: The Renderer instance
            bots: Dictionary mapping player numbers to bot instances
            num_players: Total number of players in the game
        """
        self.game = game
        self.renderer = renderer
        self.bots = bots
        self.num_players = num_players

        # Input state
        self.selected_unit = None
        self.active_menu = None
        self.menu_opened_time = 0
        self.target_selection_mode = False
        self.target_selection_action = None
        self.target_selection_unit = None

        # Right-click preview state
        self.right_click_preview_active = False
        self.preview_unit = None
        self.preview_positions = []

    def handle_keyboard_event(self, event):
        """
        Handle keyboard events.

        Args:
            event: pygame.KEYDOWN event

        Returns:
            'quit' if game should quit, 'save' if save requested, None otherwise
        """
        if event.key == pygame.K_ESCAPE:
            if self.target_selection_mode:
                # Cancel target selection and return to menu
                self.target_selection_mode = False
                self.target_selection_action = None
                # Menu should still be open
                return None
            elif self.active_menu:
                # Close menu with ESC
                if isinstance(self.active_menu, UnitActionMenu):
                    # Cancel move if unit has moved
                    if self.target_selection_unit and self.target_selection_unit.has_moved:
                        self.target_selection_unit.cancel_move()
                        print(f"Cancelled move for {self.target_selection_unit.type}")
                    self.target_selection_unit = None
                self.active_menu = None
                return None
            else:
                return 'quit'

        # Handle keyboard shortcuts for UnitActionMenu
        elif self.active_menu and isinstance(self.active_menu, UnitActionMenu):
            menu_result = self.active_menu.handle_keydown(event)
            if menu_result:
                active_menu_ref = [self.active_menu]
                target_selection_unit_ref = [self.target_selection_unit]
                selected_unit_ref = [self.selected_unit]

                result = handle_action_menu_result(
                    self.game, menu_result, active_menu_ref,
                    target_selection_unit_ref, selected_unit_ref
                )

                self.active_menu = active_menu_ref[0]
                self.target_selection_unit = target_selection_unit_ref[0]
                self.selected_unit = selected_unit_ref[0]

                if result:
                    self.target_selection_mode, self.target_selection_action = result

        elif event.key == pygame.K_s and not self.active_menu:
            # Save game
            return 'save'

        elif event.key == pygame.K_SPACE and not self.active_menu:
            # End turn
            print(f"\nPlayer {self.game.current_player} ended turn")
            self.selected_unit = None
            self.game.end_turn()

            # Process bot turns
            self._process_bot_turns()

        return None

    def handle_mouse_click(self, mouse_pos):
        """
        Handle mouse click events.

        Args:
            mouse_pos: Tuple of (x, y) mouse position

        Returns:
            'continue' if event was handled and should skip further processing
        """
        current_time = pygame.time.get_ticks()

        # Priority 0: Handle target selection mode
        if self.target_selection_mode and self.target_selection_action:
            return self._handle_target_selection_click(mouse_pos, current_time)

        # Priority 1: Handle active menu clicks
        if self.active_menu:
            # Ignore clicks for 200ms after menu opens
            if current_time - self.menu_opened_time < 200:
                return 'continue'

            menu_result = self.active_menu.handle_click(mouse_pos)
            if menu_result:
                return self._handle_menu_result(menu_result, current_time)
            return 'continue'

        # Priority 2: Check if clicking on UI buttons
        if self.renderer.end_turn_button.collidepoint(mouse_pos):
            print(f"\nPlayer {self.game.current_player} ended turn")
            self.selected_unit = None
            self.game.end_turn()
            self._process_bot_turns()
            return 'continue'

        if self.renderer.resign_button.collidepoint(mouse_pos):
            # Show confirmation dialog before resigning
            dialog = ConfirmationDialog(
                self.renderer.screen,
                "Resign Game",
                f"Player {self.game.current_player}, are you sure you want to resign?",
                confirm_text="Resign",
                cancel_text="Cancel"
            )
            if dialog.run():
                print(f"\nPlayer {self.game.current_player} resigned")
                self.game.resign()
            return 'continue'

        # Priority 3: Handle grid clicks
        return self._handle_grid_click(mouse_pos, current_time)

    def handle_mouse_motion(self, mouse_pos):
        """
        Handle mouse motion events.

        Args:
            mouse_pos: Tuple of (x, y) mouse position
        """
        if self.active_menu and hasattr(self.active_menu, 'handle_mouse_motion'):
            self.active_menu.handle_mouse_motion(mouse_pos)

    def handle_right_click_press(self, mouse_pos):
        """
        Handle right mouse button press.

        If a unit has been selected and moved (menu is open), cancel the move
        and deselect the unit. Otherwise, show attack range preview.

        Args:
            mouse_pos: Tuple of (x, y) mouse position
        """
        # Priority 1: Cancel target selection mode
        if self.target_selection_mode:
            self.target_selection_mode = False
            self.target_selection_action = None
            if self.target_selection_unit and self.target_selection_unit.has_moved:
                self.target_selection_unit.cancel_move()
                print(f"Cancelled move for {self.target_selection_unit.type}")
            self.target_selection_unit = None
            self.active_menu = None
            self.selected_unit = None
            return

        # Priority 2: Close menu and cancel move if unit has moved
        if self.active_menu and isinstance(self.active_menu, UnitActionMenu):
            if self.target_selection_unit and self.target_selection_unit.has_moved:
                self.target_selection_unit.cancel_move()
                print(f"Cancelled move for {self.target_selection_unit.type}")
            self.target_selection_unit = None
            self.active_menu = None
            self.selected_unit = None
            return

        # Priority 3: Show attack range preview (right-click no longer deselects units)
        grid_x = mouse_pos[0] // TILE_SIZE
        grid_y = mouse_pos[1] // TILE_SIZE

        # Check bounds
        if not (0 <= grid_x < self.game.grid.width and 0 <= grid_y < self.game.grid.height):
            return

        # Find unit at clicked position
        clicked_unit = self.game.get_unit_at_position(grid_x, grid_y)

        if clicked_unit:
            # Activate preview for this unit
            self.right_click_preview_active = True
            self.preview_unit = clicked_unit

            # Get all attackable positions (enemy unit positions)
            from reinforcetactics.game.mechanics import GameMechanics
            attackable_enemies = GameMechanics.get_attackable_enemies(
                clicked_unit, self.game.units, self.game.grid
            )

            # Convert to positions list
            self.preview_positions = [(enemy.x, enemy.y) for enemy in attackable_enemies]

    def handle_right_click_release(self):
        """
        Handle right mouse button release - end attack range preview.
        """
        self.right_click_preview_active = False
        self.preview_unit = None
        self.preview_positions = []

    def _handle_target_selection_click(self, mouse_pos, current_time):
        """Handle clicks during target selection mode."""
        grid_x = mouse_pos[0] // TILE_SIZE
        grid_y = mouse_pos[1] // TILE_SIZE

        clicked_unit = self.game.get_unit_at_position(grid_x, grid_y)
        if (clicked_unit and self.target_selection_action and
            clicked_unit in self.target_selection_action['targets']):
            # Execute the action on the clicked target
            action_type = self.target_selection_action['type']
            if action_type == 'attack':
                self.game.attack(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} attacked {clicked_unit.type}")
            elif action_type == 'paralyze':
                self.game.paralyze(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} paralyzed {clicked_unit.type}")
            elif action_type == 'heal':
                self.game.heal(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} healed {clicked_unit.type}")
            elif action_type == 'cure':
                self.game.cure(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} cured {clicked_unit.type}")
            elif action_type == 'haste':
                self.game.haste(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} hasted {clicked_unit.type}")
            elif action_type == 'defence_buff':
                self.game.defence_buff(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} granted defence buff to {clicked_unit.type}")
            elif action_type == 'attack_buff':
                self.game.attack_buff(self.target_selection_unit, clicked_unit)
                print(f"{self.target_selection_unit.type} granted attack buff to {clicked_unit.type}")

            # End unit's turn and reset selection
            can_still_act = self.target_selection_unit.end_unit_turn()
            self.target_selection_mode = False
            self.target_selection_action = None

            if can_still_act:
                # Unit has haste and can act again - keep it selected
                print(f"{self.target_selection_unit.type} used haste action (can act again)")
                self.selected_unit = self.target_selection_unit
                # FOW: Capture visible enemies for the new action
                self.game.capture_visible_enemies_for_unit(self.selected_unit)
                self.target_selection_unit = None
            else:
                self.target_selection_unit = None
                self.selected_unit = None
        else:
            # Clicked outside valid targets, cancel and return to menu
            self.target_selection_mode = False
            self.target_selection_action = None
            # Reopen the unit action menu
            self.active_menu = UnitActionMenu(
                self.renderer.screen, self.game, self.target_selection_unit
            )
            self.menu_opened_time = current_time
            print("Target selection cancelled, returning to menu")

        return 'continue'

    def _handle_menu_result(self, menu_result, current_time):
        """Handle menu interaction results."""
        if menu_result['type'] == 'close':
            self.active_menu = None
        elif menu_result['type'] == 'unit_created':
            unit = menu_result['unit']
            print(f"Created {unit.type} at ({unit.x}, {unit.y})")
            self.active_menu = None
        elif menu_result['type'] in ['cancel', 'action_selected']:
            # Handle UnitActionMenu results
            if isinstance(self.active_menu, UnitActionMenu):
                active_menu_ref = [self.active_menu]
                target_selection_unit_ref = [self.target_selection_unit]
                selected_unit_ref = [self.selected_unit]

                result = handle_action_menu_result(
                    self.game, menu_result, active_menu_ref,
                    target_selection_unit_ref, selected_unit_ref
                )

                self.active_menu = active_menu_ref[0]
                self.target_selection_unit = target_selection_unit_ref[0]
                self.selected_unit = selected_unit_ref[0]

                if result:
                    self.target_selection_mode, self.target_selection_action = result

        return 'continue'

    def _handle_grid_click(self, mouse_pos, current_time):
        """Handle clicks on the game grid."""
        grid_x = mouse_pos[0] // TILE_SIZE
        grid_y = mouse_pos[1] // TILE_SIZE

        # Check bounds
        if not (0 <= grid_x < self.game.grid.width and 0 <= grid_y < self.game.grid.height):
            return None

        clicked_unit = self.game.get_unit_at_position(grid_x, grid_y)
        clicked_tile = self.game.grid.get_tile(grid_x, grid_y)

        # Priority 1: Own unit clicked
        if clicked_unit and clicked_unit.player == self.game.current_player:
            if self.selected_unit == clicked_unit:
                # Open unit action menu if unit can perform actions
                if not clicked_unit.is_paralyzed() and (clicked_unit.can_move or clicked_unit.can_attack):
                    self.active_menu = UnitActionMenu(
                        self.renderer.screen, self.game, clicked_unit
                    )
                    self.target_selection_unit = clicked_unit
                    self.menu_opened_time = current_time
                    print(f"Opened unit action menu for {clicked_unit.type}")
                else:
                    print(f"{clicked_unit.type} cannot perform actions")
            else:
                # Select new unit
                self.selected_unit = clicked_unit
                # FOW: Capture visible enemies at the start of this unit's action
                self.game.capture_visible_enemies_for_unit(clicked_unit)
                print(f"Selected {clicked_unit.type} at ({grid_x}, {grid_y})")
            return 'continue'

        # Priority 2: Building clicked for unit purchase
        if (not clicked_unit and clicked_tile.player == self.game.current_player and
            clicked_tile.type == 'b'):
            self.active_menu = UnitPurchaseMenu(
                self.renderer.screen, self.game, (grid_x, grid_y)
            )
            self.menu_opened_time = current_time
            print(f"Opened unit purchase menu at ({grid_x}, {grid_y})")
            return 'continue'

        # Priority 3: Movement with selected unit
        if self.selected_unit and self.selected_unit.can_move:
            if self.game.move_unit(self.selected_unit, grid_x, grid_y):
                print(f"Moved {self.selected_unit.type} to ({grid_x}, {grid_y})")
                # After movement, open unit action menu
                self.active_menu = UnitActionMenu(
                    self.renderer.screen, self.game, self.selected_unit
                )
                self.target_selection_unit = self.selected_unit
                self.menu_opened_time = current_time
                self.selected_unit = None
            return 'continue'

        # Priority 4: Deselect
        self.selected_unit = None
        return 'continue'

    def _process_bot_turns(self):
        """Process consecutive bot turns."""
        # Safety counter to prevent infinite loops
        max_bot_turns = self.num_players * 2
        bot_turn_count = 0

        while (self.game.current_player in self.bots and
               not self.game.game_over and
               bot_turn_count < max_bot_turns):
            current_bot = self.bots[self.game.current_player]
            print(f"Bot (Player {self.game.current_player}) is thinking...")
            current_bot.take_turn()
            # Note: Bots call end_turn() internally, so we don't call it here
            bot_turn_count += 1
            print(f"Bot finished. Player {self.game.current_player}'s turn\n")
