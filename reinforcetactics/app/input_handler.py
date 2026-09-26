"""
Input Handler for Reinforce Tactics.

This module manages user input state and event handling for the game loop.
"""

import logging

import pygame

from reinforcetactics.app.action_executor import apply_targeted_action, handle_action_menu_result
from reinforcetactics.game.llm_bot import LLMBotError
from reinforcetactics.ui import widgets
from reinforcetactics.ui.assets import TILE_SIZE
from reinforcetactics.ui.menus import ConfirmationDialog, UnitActionMenu, UnitPurchaseMenu
from reinforcetactics.ui.menus.base import drain_events
from reinforcetactics.ui.widgets.dialog import Dialog

# The bot-replaced dialog shortens the LLM error (which can carry a whole
# HTTP error body) until the dialog fits the window, but not below this many
# characters. The full message is printed to the console.
_MIN_DIALOG_REASON_CHARS = 40

logger = logging.getLogger(__name__)

# How long an on-screen notice stays up.
NOTICE_DURATION_MS = 6000


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

        # Transient on-screen notice, drawn by GameSession. A GUI player
        # never sees stdout, so problems such as a crashed bot turn are
        # reported here as well as logged.
        self.notice_text = None
        self.notice_expires_at = 0

    def show_notice(self, text, duration_ms=NOTICE_DURATION_MS):
        """Show ``text`` on screen for ``duration_ms`` milliseconds."""
        self.notice_text = text
        self.notice_expires_at = pygame.time.get_ticks() + duration_ms

    def handle_keyboard_event(self, event):
        """
        Handle keyboard events.

        Args:
            event: pygame.KEYDOWN event

        Returns:
            'pause' if pause menu should open, 'save' if save requested, None otherwise
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
                        if self.game.cancel_move(self.target_selection_unit):
                            print(f"Cancelled move for {self.target_selection_unit.type}")
                    self.target_selection_unit = None
                self.active_menu = None
                return None
            else:
                return "pause"

        # Handle keyboard shortcuts for UnitActionMenu
        elif self.active_menu and isinstance(self.active_menu, UnitActionMenu):
            menu_result = self.active_menu.handle_keydown(event)
            if menu_result:
                active_menu_ref = [self.active_menu]
                target_selection_unit_ref = [self.target_selection_unit]
                selected_unit_ref = [self.selected_unit]

                result = handle_action_menu_result(
                    self.game, menu_result, active_menu_ref, target_selection_unit_ref, selected_unit_ref
                )

                self.active_menu = active_menu_ref[0]
                self.target_selection_unit = target_selection_unit_ref[0]
                self.selected_unit = selected_unit_ref[0]

                if result:
                    self.target_selection_mode, self.target_selection_action = result

        # Handle keyboard shortcuts for UnitPurchaseMenu
        elif self.active_menu and isinstance(self.active_menu, UnitPurchaseMenu):
            menu_result = self.active_menu.handle_keydown(event)
            if menu_result:
                return self._handle_menu_result(menu_result, pygame.time.get_ticks())

        elif event.key == pygame.K_s and not self.active_menu:
            # Save game
            return "save"

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
                return "continue"

            menu_result = self.active_menu.handle_click(mouse_pos)
            if menu_result:
                return self._handle_menu_result(menu_result, current_time)
            return "continue"

        # Priority 2: Check if clicking on UI buttons
        if self.renderer.end_turn_button.collidepoint(mouse_pos):
            print(f"\nPlayer {self.game.current_player} ended turn")
            self.selected_unit = None
            self.game.end_turn()
            self._process_bot_turns()
            return "continue"

        if self.renderer.resign_button.collidepoint(mouse_pos):
            # Show confirmation dialog before resigning
            dialog = ConfirmationDialog(
                self.renderer.screen,
                "Resign Game",
                f"Player {self.game.current_player}, are you sure you want to resign?",
                confirm_text="Resign",
                cancel_text="Cancel",
            )
            if dialog.run():
                player = self.game.current_player
                print(f"\nPlayer {player} resigned")
                self.game.resign()
                if not self.game.game_over and self.game.is_eliminated(player):
                    # Three or more seats: the others play on (review core-7).
                    # The resigned seat has nothing left to do, so hand the
                    # turn on for it, and let any bots that follow play.
                    self.selected_unit = None
                    self.game.end_turn()
                    self._process_bot_turns()
            return "continue"

        # Priority 3: Handle grid clicks
        return self._handle_grid_click(mouse_pos, current_time)

    def handle_mouse_motion(self, mouse_pos):
        """
        Handle mouse motion events.

        Args:
            mouse_pos: Tuple of (x, y) mouse position
        """
        if self.active_menu and hasattr(self.active_menu, "handle_mouse_motion"):
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
                if self.game.cancel_move(self.target_selection_unit):
                    print(f"Cancelled move for {self.target_selection_unit.type}")
            self.target_selection_unit = None
            self.active_menu = None
            self.selected_unit = None
            return

        # Priority 2: Close menu and cancel move if unit has moved
        if self.active_menu and isinstance(self.active_menu, UnitActionMenu):
            if self.target_selection_unit and self.target_selection_unit.has_moved:
                if self.game.cancel_move(self.target_selection_unit):
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
            from reinforcetactics.core.mechanics import GameMechanics

            attackable_enemies = GameMechanics.get_attackable_enemies(
                clicked_unit, self.game.units, self.game.grid, self.game.teams
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
        if clicked_unit and self.target_selection_action and clicked_unit in self.target_selection_action["targets"]:
            # Execute the action on the clicked target
            action_type = self.target_selection_action["type"]
            if not apply_targeted_action(self.game, action_type, self.target_selection_unit, clicked_unit):
                # The engine refused it and nothing changed, so the unit
                # still has its action: back to its menu, turn not spent.
                self.target_selection_mode = False
                self.target_selection_action = None
                self.active_menu = UnitActionMenu(self.renderer.screen, self.game, self.target_selection_unit)
                self.menu_opened_time = current_time
                return "continue"

            # End unit's turn and reset selection. A hasted unit was already
            # refreshed by the engine; end_unit_turn then keeps its extra
            # action and returns True (see GameState.end_unit_turn).
            can_still_act = self.game.end_unit_turn(self.target_selection_unit)
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
            self.active_menu = UnitActionMenu(self.renderer.screen, self.game, self.target_selection_unit)
            self.menu_opened_time = current_time
            print("Target selection cancelled, returning to menu")

        return "continue"

    def _handle_menu_result(self, menu_result, current_time):
        """Handle menu interaction results."""
        if menu_result["type"] == "close":
            self.active_menu = None
        elif menu_result["type"] == "unit_created":
            unit = menu_result["unit"]
            print(f"Created {unit.type} at ({unit.x}, {unit.y})")
            self.active_menu = None
        elif menu_result["type"] in ["cancel", "action_selected"]:
            # Handle UnitActionMenu results
            if isinstance(self.active_menu, UnitActionMenu):
                active_menu_ref = [self.active_menu]
                target_selection_unit_ref = [self.target_selection_unit]
                selected_unit_ref = [self.selected_unit]

                result = handle_action_menu_result(
                    self.game, menu_result, active_menu_ref, target_selection_unit_ref, selected_unit_ref
                )

                self.active_menu = active_menu_ref[0]
                self.target_selection_unit = target_selection_unit_ref[0]
                self.selected_unit = selected_unit_ref[0]

                if result:
                    self.target_selection_mode, self.target_selection_action = result

        return "continue"

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
                    self.active_menu = UnitActionMenu(self.renderer.screen, self.game, clicked_unit)
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
            return "continue"

        # Priority 2: Building clicked for unit purchase
        if not clicked_unit and clicked_tile.player == self.game.current_player and clicked_tile.type == "b":
            self.active_menu = UnitPurchaseMenu(self.renderer.screen, self.game, (grid_x, grid_y))
            self.menu_opened_time = current_time
            print(f"Opened unit purchase menu at ({grid_x}, {grid_y})")
            return "continue"

        # Priority 3: Movement with selected unit
        if self.selected_unit and self.selected_unit.can_move:
            if self.game.move_unit(self.selected_unit, grid_x, grid_y):
                unit = self.selected_unit
                # Under fog of war the unit may have been ambushed and stopped
                # short of the clicked tile (see GameState.move_unit).
                print(f"Moved {unit.type} to ({unit.x}, {unit.y})")
                if unit.ambushed:
                    self.show_notice(f"Ambushed! Your {unit.type} stopped at ({unit.x}, {unit.y})")
                # After movement, open unit action menu
                self.active_menu = UnitActionMenu(self.renderer.screen, self.game, self.selected_unit)
                self.target_selection_unit = self.selected_unit
                self.menu_opened_time = current_time
                self.selected_unit = None
            return "continue"

        # Priority 4: Deselect
        self.selected_unit = None
        return "continue"

    def _process_bot_turns(self):
        """Process consecutive bot turns.

        A bot that raises must not end the game: the exception used to
        unwind through GameSession.run, dropping the player to the main menu
        with nothing saved. Now the error is logged with its traceback,
        reported on screen, and the bot's turn is ended so play continues.
        """
        # Safety counter to prevent infinite loops
        max_bot_turns = self.num_players * 2
        bot_turn_count = 0

        while self.game.current_player in self.bots and not self.game.game_over and bot_turn_count < max_bot_turns:
            player = self.game.current_player
            current_bot = self.bots[player]
            print(f"Bot (Player {player}) is thinking...")
            try:
                current_bot.take_turn()
            except LLMBotError as exc:
                # The LLM bot can't reach its model (rejected key, unknown
                # model, an outage that outlasted its retries) and left its
                # turn un-ended. Letting this unwind ended the session with
                # nothing saved; SimpleBot takes the seat instead and plays
                # the turn on the next pass of this loop.
                self._replace_failed_llm_bot(player, current_bot, exc)
            except Exception:
                logger.exception("%s (player %d) raised during its turn", type(current_bot).__name__, player)
                self.show_notice(f"Player {player}'s bot hit an error; its turn was skipped")
                if not self._end_crashed_bot_turn(player):
                    break
            # Note: Bots call end_turn() internally, so we don't call it here
            bot_turn_count += 1
            print(f"Bot finished. Player {self.game.current_player}'s turn\n")

    def _end_crashed_bot_turn(self, player):
        """End ``player``'s turn after its bot raised part-way through it.

        Returns:
            False if the turn could not be ended; bot processing then stops
            instead of re-running a bot against a state it can't leave.
        """
        # The bot may have raised after its own end_turn() call already
        # handed the turn on; ending it again would skip the next player.
        if self.game.game_over or self.game.current_player != player:
            return True
        try:
            self.game.end_turn()
        except Exception:
            logger.exception("Could not end player %d's turn after its bot raised", player)
            return False
        return True

    def _replace_failed_llm_bot(self, player, bot, exc):
        """Hand ``player``'s seat to SimpleBot after its LLM bot raised LLMBotError.

        This is the fallback bot_factory uses when an LLM bot can't be built
        at all (missing SDK or key). The player is told in a dialog, since a
        GUI player doesn't see the console.
        """
        from reinforcetactics.game.bot import SimpleBot

        bot_name = f"{type(bot).__name__} ({getattr(bot, 'model', 'unknown model')})"
        print(f"❌ Player {player}'s {bot_name} stopped: {exc}")
        print(f"   SimpleBot takes over Player {player} for the rest of the game")
        self.bots[player] = SimpleBot(self.game, player=player)

        # Lead with the cause: the window is sized to the map, and on small
        # maps a "ClaudeBot (model-id):" prefix pushed the actual reason
        # (e.g. "HTTP 401 authentication failed") out of the dialog.
        reason = str(exc)
        prefix = f"{type(bot).__name__} ({getattr(bot, 'model', 'unknown model')}): "
        if reason.startswith(prefix):
            reason = reason[len(prefix) :]
        try:
            self._show_bot_replaced_dialog(
                f"Player {player}: LLM stopped",
                reason,
                f"{bot_name}. SimpleBot takes over Player {player}.",
            )
        except Exception as dialog_error:  # noqa: BLE001
            # The notice is best-effort: failing to draw it must not end the
            # game this fallback exists to keep going. The console has it.
            print(f"⚠️  Could not show the bot-replaced dialog: {dialog_error}")

    def _show_bot_replaced_dialog(self, title, reason, footer):
        """Show ``reason`` and ``footer`` in a modal notice with an OK button.

        Split out so tests can stub it. The game window is sized to the map
        and an LLM error can carry a whole HTTP error body, so ``reason`` is
        shortened until the dialog fits the window (on the smallest maps it
        can't entirely: the dialog stays centred and is clipped a little).
        """
        screen = self.renderer.screen
        while True:
            dialog = Dialog(
                screen,
                title,
                f"{reason}\n\n{footer}",
                buttons=[("OK", "ok", widgets.CONFIRM)],
                keymap={pygame.K_RETURN: "ok", pygame.K_KP_ENTER: "ok"},
                cancel_value="ok",
                quit_value="quit",
                min_width=min(500, screen.get_width() - 40),
            )
            if dialog.dialog_rect.height <= screen.get_height() or len(reason) <= _MIN_DIALOG_REASON_CHARS:
                break
            reason = reason[: len(reason) * 3 // 4].rstrip() + "…"
        result = dialog.run()
        # Drop clicks and keys queued while the dialog was up so they don't
        # land on the board. A window-close is kept, and one that closed the
        # dialog itself is re-posted, so the game loop still offers to save
        # before quitting.
        drain_events()
        if result == "quit":
            pygame.event.post(pygame.event.Event(pygame.QUIT))
