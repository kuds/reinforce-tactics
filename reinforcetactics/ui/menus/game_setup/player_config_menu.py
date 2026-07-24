"""Menu for configuring players."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pygame

from reinforcetactics.ui import theme, widgets
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.fonts import get_display_font, get_font
from reinforcetactics.utils.language import get_language

# Horizontal gap between the buttons that make up one player's row.
BUTTON_GAP = 16

# Import tkinter optionally for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class PlayerConfigMenu:
    """Menu for configuring players (Human vs Computer) with difficulty settings."""

    def __init__(self, screen: pygame.Surface | None = None, game_mode: str = "1v1") -> None:
        """
        Initialize player configuration menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
            game_mode: Game mode ("1v1" or "2v2")

        Raises:
            ValueError: If game_mode is not "1v1" or "2v2"
        """
        # Validate game_mode
        if game_mode not in ["1v1", "2v2"]:
            raise ValueError(f"Invalid game_mode: {game_mode}. Must be '1v1' or '2v2'")

        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Create screen if not provided
        self.owns_screen = screen is None
        if self.owns_screen:
            self.screen = pygame.display.set_mode((900, 700))
            pygame.display.set_caption("Reinforce Tactics")
        else:
            self.screen = screen

        self.game_mode = game_mode
        self.num_players = 2 if game_mode == "1v1" else 4
        self.running = True

        # Colors (from shared theme)
        self.bg_color = theme.BG
        self.text_color = theme.TEXT
        self.selected_color = theme.SELECTED
        self.hover_color = theme.HOVER
        self.title_color = theme.TITLE
        self.option_bg_color = theme.OPTION_BG
        self.option_bg_hover_color = theme.OPTION_BG_HOVER
        self.option_bg_selected_color = theme.OPTION_BG_SELECTED
        self.disabled_color = theme.DISABLED

        # Fonts
        self.title_font = get_display_font(theme.FONT_SIZE_TITLE)
        self.label_font = get_font(theme.FONT_SIZE_HEADING)
        self.option_font = get_font(theme.FONT_SIZE_SUBHEADING)

        # Player configurations
        # Default: Player 1 is Human, others are Computer (SimpleBot)
        self.player_configs = []
        for i in range(self.num_players):
            self.player_configs.append(
                {"type": "human" if i == 0 else "computer", "bot_type": None if i == 0 else "SimpleBot", "model_path": None}
            )

        # Game options
        self.fog_of_war = False  # Fog of war toggle

        # Model validation state: {player_idx: {'valid': bool, 'error': str}}
        self.model_validation: dict[int, dict[str, Any]] = {}

        # UI interaction tracking
        self.hover_element = None
        self.selected_element = None
        self.interactive_elements: list[dict[str, Any]] = []
        # Keyboard focus into ``interactive_elements``. -1 means "nothing
        # focused", which keeps Enter meaning "start the game" until the
        # player actually tabs into a control.
        self.focus_index = -1
        self._focused_rect: pygame.Rect | None = None
        # Message shown under the buttons instead of printing to stdout,
        # which a player running the GUI never sees.
        self.status_message: str | None = None

        # Get language instance
        self.lang = get_language()

        # Check which LLM providers have API keys configured
        from reinforcetactics.utils.settings import get_settings

        settings = get_settings()
        self.available_llm_bots = {
            "OpenAIBot": bool(settings.get_api_key("openai")),
            "ClaudeBot": bool(settings.get_api_key("anthropic")),
            "GeminiBot": bool(settings.get_api_key("google")),
        }

        # Check if stable-baselines3 is available for ModelBot
        self.modelbot_available = self._check_modelbot_available()

    def _check_modelbot_available(self) -> bool:
        """Check if ModelBot dependencies are available."""
        try:
            # pylint: disable=unused-import,import-outside-toplevel
            import stable_baselines3  # noqa: F401

            return True
        except ImportError:
            return False

    def _validate_model(self, model_path: str) -> dict[str, Any]:
        """
        Validate that a model file is compatible.

        Args:
            model_path: Path to the model .zip file

        Returns:
            Dict with 'valid' (bool) and 'error' (str or None)
        """
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                return {"valid": False, "error": "File not found"}

            if not model_file.suffix == ".zip":
                return {"valid": False, "error": "Must be a .zip file"}

            # Try to load the model with ModelBot
            from reinforcetactics.core.game_state import GameState
            from reinforcetactics.game.model_bot import ModelBot

            # Create a dummy game state for testing
            # Use a simple 6x6 map for validation
            dummy_map = [["p" for _ in range(6)] for _ in range(6)]
            dummy_map[0][0] = "h_1"  # HQ for player 1
            dummy_map[5][5] = "h_2"  # HQ for player 2

            dummy_state = GameState(dummy_map, num_players=2)

            # Try to create the bot - this will load the model
            bot = ModelBot(dummy_state, player=2, model_path=str(model_path))

            if bot.model is None:
                return {"valid": False, "error": "Failed to load model"}

            return {"valid": True, "error": None}

        except ImportError as e:
            return {"valid": False, "error": f"Missing dependency: {e}"}
        except Exception as e:
            return {"valid": False, "error": f"Load error: {str(e)[:50]}"}

    def _open_file_dialog(self) -> str | None:
        """
        Open a native OS file dialog to select a model file.

        Returns:
            Selected file path or None if cancelled
        """
        if not TKINTER_AVAILABLE:
            print("⚠️  tkinter not available - cannot open file dialog")
            return None

        try:
            # Create a hidden tkinter root window
            root = tk.Tk()
            root.withdraw()

            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Model File", filetypes=[("Model files", "*.zip"), ("All files", "*.*")], initialdir="."
            )

            # Clean up
            root.destroy()

            # Return the path or None if cancelled
            return file_path if file_path else None

        except Exception as e:
            print(f"Error opening file dialog: {e}")
            return None

    def handle_input(self, event: pygame.event.Event) -> dict[str, Any] | None:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            Result dict with player configurations, or None
        """
        if event.type == pygame.KEYDOWN:
            return self._handle_keydown(event)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos
                for i, element in enumerate(self.interactive_elements):
                    if element["rect"].collidepoint(mouse_pos):
                        if element.get("disabled"):
                            self.status_message = self._blocking_reason()
                            return None
                        # Clicking also moves keyboard focus, so switching
                        # between mouse and keyboard doesn't lose your place.
                        self.focus_index = i
                        return self._activate(element)

        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_element = None
            for element in self.interactive_elements:
                if element["rect"].collidepoint(mouse_pos):
                    self.hover_element = element
                    break

        return None

    def _handle_keydown(self, event: pygame.event.Event) -> dict[str, Any] | None:
        """Handle keyboard navigation and activation.

        Every control on this screen used to be mouse-only; arrow keys and
        Tab now walk the same element list the mouse hit-tests against.
        """
        if event.key == pygame.K_ESCAPE:
            self.running = False
            return None

        focusable = [i for i, e in enumerate(self.interactive_elements) if not e.get("disabled")]

        if event.key in (pygame.K_DOWN, pygame.K_RIGHT, pygame.K_TAB):
            self._move_focus(focusable, 1)
            return None
        if event.key in (pygame.K_UP, pygame.K_LEFT):
            self._move_focus(focusable, -1)
            return None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            if 0 <= self.focus_index < len(self.interactive_elements):
                element = self.interactive_elements[self.focus_index]
                if element.get("disabled"):
                    self.status_message = self._blocking_reason()
                    return None
                return self._activate(element)
            # Nothing focused: Enter still means "start with these settings".
            return self._get_result()

        return None

    def _move_focus(self, focusable: list[int], step: int) -> None:
        """Move keyboard focus to the next/previous enabled element."""
        if not focusable:
            self.focus_index = -1
            return
        if self.focus_index not in focusable:
            self.focus_index = focusable[0] if step > 0 else focusable[-1]
            return
        position = focusable.index(self.focus_index)
        self.focus_index = focusable[(position + step) % len(focusable)]

    def _activate(self, element: dict[str, Any]) -> dict[str, Any] | None:
        """Apply an element's action. Shared by mouse clicks and Enter."""
        element_type = element["type"]
        player_idx = element["player_idx"]

        if element_type == "type_toggle":
            config = self.player_configs[player_idx]
            if config["type"] == "human":
                config["type"] = "computer"
                config["bot_type"] = "SimpleBot"
                config["model_path"] = None
            else:
                config["type"] = "human"
                config["bot_type"] = None
                config["model_path"] = None
                self.model_validation.pop(player_idx, None)
            self.status_message = None

        elif element_type == "difficulty_select":
            config = self.player_configs[player_idx]
            if config["type"] == "computer":
                bot_types = self._available_bot_types()
                current_bot = config["bot_type"]
                if current_bot in bot_types:
                    next_bot = bot_types[(bot_types.index(current_bot) + 1) % len(bot_types)]
                    config["bot_type"] = next_bot
                    # Clear model path if switching away from ModelBot
                    if next_bot != "ModelBot":
                        config["model_path"] = None
                        self.model_validation.pop(player_idx, None)
                else:
                    # Unknown bot type (e.g. an LLM whose key was removed).
                    config["bot_type"] = "SimpleBot"
            self.status_message = None

        elif element_type == "browse_model":
            config = self.player_configs[player_idx]
            file_path = self._open_file_dialog()
            if file_path:
                config["model_path"] = file_path
                validation = self._validate_model(file_path)
                self.model_validation[player_idx] = validation
                self.status_message = None if validation.get("valid") else validation.get("error")

        elif element_type == "fog_of_war_toggle":
            self.fog_of_war = not self.fog_of_war

        elif element_type == "start_button":
            result = self._get_result()
            if result is None:
                self.status_message = self._blocking_reason()
            return result

        elif element_type == "back_button":
            self.running = False
            return None

        return None

    def _is_focused(self, rect: pygame.Rect) -> bool:
        """Whether ``rect`` belongs to the keyboard-focused element.

        Compared against the rect snapshotted at the start of :meth:`draw`,
        since the element list is being rebuilt while buttons are drawn.
        """
        return self._focused_rect is not None and self._focused_rect == rect

    def _available_bot_types(self) -> list[str]:
        """Bot types the player can cycle through, in display order."""
        # All built-in bots are always available; LLM bots need an API key
        # and ModelBot needs stable-baselines3.
        bot_types = ["SimpleBot", "MediumBot", "AdvancedBot"]
        bot_types.extend(name for name, available in self.available_llm_bots.items() if available)
        if self.modelbot_available:
            bot_types.append("ModelBot")
        return bot_types

    def _blocking_reason(self) -> str | None:
        """Why the game can't start yet, phrased for the player."""
        for i, config in enumerate(self.player_configs):
            if config["type"] != "computer" or config["bot_type"] != "ModelBot":
                continue
            if not config.get("model_path"):
                return f"Player {i + 1}: choose a model file (Browse...)"
            validation = self.model_validation.get(i, {})
            if not validation.get("valid", False):
                return f"Player {i + 1}: {validation.get('error', 'model could not be loaded')}"
        return None

    def _get_result(self) -> dict[str, Any] | None:
        """
        Get the configured player settings as a result dict.

        Returns:
            Dict with player configurations, or None if validation fails
        """
        # Refuse to start while any ModelBot is missing a usable model. The
        # reason is surfaced on screen by ``_blocking_reason`` rather than
        # printed, which a player running the GUI would never see.
        if self._blocking_reason() is not None:
            return None

        return {"players": self.player_configs, "fog_of_war": self.fog_of_war}

    def draw(self) -> None:
        """Draw the player configuration menu."""
        self.screen.fill(self.bg_color)
        # Remember which rect has focus before the element list is rebuilt:
        # buttons are drawn as they are appended, so mid-rebuild the list
        # can't answer "is this the focused one?" for itself.
        self._focused_rect = (
            self.interactive_elements[self.focus_index]["rect"]
            if 0 <= self.focus_index < len(self.interactive_elements)
            else None
        )
        self.interactive_elements = []

        screen_width = self.screen.get_width()

        # Draw title
        title = self.lang.get("player_config.title", "Configure Players")
        title_surface = self.title_font.render(title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=24)
        self.screen.blit(title_surface, title_rect)

        # Starting Y position for player configurations. Derived from the
        # title's actual height: the old hardcoded 80 sat *inside* the
        # display-font title, so "Player 1" collided with it.
        # 2v2 uses tighter row spacing so four players still fit.
        start_y = title_rect.bottom + 16
        spacing_y = 85 if self.num_players > 2 else 100

        # Draw each player's configuration
        for i in range(self.num_players):
            config = self.player_configs[i]
            y_pos = start_y + i * spacing_y

            # Player label
            player_label = self.lang.get("player_config.player", "Player {number}").format(number=i + 1)
            label_surface = self.label_font.render(player_label, True, self.text_color)
            label_rect = label_surface.get_rect(x=50, y=y_pos)
            self.screen.blit(label_surface, label_rect)

            # Type toggle button (Human/Computer)
            type_x = 200
            if config["type"] == "human":
                type_text = self.lang.get("player_config.type_human", "Human")
            else:
                type_text = self.lang.get("player_config.type_computer", "Computer")
            type_rect = self._draw_button(type_x, y_pos, type_text, "type_toggle", i)

            # Difficulty selection (only shown if computer)
            if config["type"] == "computer":
                # Buttons are sized to their labels, so the row is laid out
                # left to right from the previous button's edge. The old
                # fixed x positions (400, 590) made a long bot name overlap
                # the Browse button next to it.
                diff_x = type_rect.right + BUTTON_GAP
                bot_type = config.get("bot_type") or "SimpleBot"
                # Get display text for bot type
                bot_display_names = {
                    "SimpleBot": "Simple Bot",
                    "MediumBot": "Medium Bot",
                    "AdvancedBot": "Advanced Bot",
                    "OpenAIBot": "OpenAI (GPT)",
                    "ClaudeBot": "Claude",
                    "GeminiBot": "Gemini",
                    "ModelBot": "Custom Model",
                }
                diff_text = bot_display_names.get(bot_type, bot_type)

                # Add indicator if bot is unavailable (no API key)
                if bot_type in self.available_llm_bots and not self.available_llm_bots[bot_type]:
                    diff_text += " (No API Key)"

                diff_rect = self._draw_button(diff_x, y_pos, diff_text, "difficulty_select", i, disabled=False)

                # If ModelBot is selected, show browse button and model status
                if bot_type == "ModelBot":
                    browse_x = diff_rect.right + BUTTON_GAP
                    browse_rect = self._draw_button(browse_x, y_pos, "Browse...", "browse_model", i, disabled=False)

                    # Show model status below
                    model_path = config.get("model_path")
                    if model_path:
                        # Show filename
                        filename = Path(model_path).name
                        # Truncate if too long
                        if len(filename) > 30:
                            filename = filename[:27] + "..."

                        # Check validation status
                        validation = self.model_validation.get(i, {})
                        if validation.get("valid"):
                            status_text = f"✓ {filename}"
                            status_color = theme.STATUS_VALID
                        else:
                            error = validation.get("error", "Invalid")
                            status_text = f"✗ {error}"
                            status_color = theme.STATUS_INVALID
                    else:
                        status_text = "No model selected"
                        status_color = theme.STATUS_WARNING

                    # Status goes on its own line under the row (it used to
                    # be drawn at the row's own height, on top of Browse),
                    # and is ellipsized so a long error can't run off screen.
                    status_font = get_font(theme.FONT_SIZE_LABEL)
                    status_x = type_rect.x
                    status_text = ellipsize(status_text, status_font, screen_width - status_x - 50)
                    status_surface = status_font.render(status_text, True, status_color)
                    self.screen.blit(status_surface, (status_x, browse_rect.bottom + 6))

        # Draw Game Options section
        options_y = start_y + self.num_players * spacing_y + 10

        # Draw divider line
        divider_y = options_y
        pygame.draw.line(self.screen, theme.PANEL_BUTTON_BORDER, (50, divider_y), (screen_width - 50, divider_y), 2)

        # Game Options label
        options_label = self.lang.get("player_config.game_options", "Game Options")
        options_surface = self.label_font.render(options_label, True, self.title_color)
        options_rect = options_surface.get_rect(x=50, y=divider_y + 10)
        self.screen.blit(options_surface, options_rect)

        # Fog of War toggle
        fow_y = divider_y + 50
        fow_label = self.lang.get("player_config.fog_of_war", "Fog of War")
        fow_label_surface = self.option_font.render(fow_label, True, self.text_color)
        self.screen.blit(fow_label_surface, (50, fow_y))

        fow_status = (
            self.lang.get("common.enabled", "Enabled") if self.fog_of_war else self.lang.get("common.disabled", "Disabled")
        )
        fow_color = theme.STATUS_VALID if self.fog_of_war else theme.TEXT_PLACEHOLDER
        self._draw_toggle_button(200, fow_y - 5, fow_status, "fog_of_war_toggle", fow_color)

        # Draw Start Game button
        # Add extra spacing if any player uses ModelBot (for status text)
        extra_spacing = 30 if any(c["bot_type"] == "ModelBot" for c in self.player_configs) else 0
        start_y_pos = fow_y + 50 + extra_spacing
        start_text = self.lang.get("player_config.start_game", "Start Game")

        # Disabled whenever anything blocks the start; same check the Enter
        # key and the click handler use, so they can never disagree.
        blocking_reason = self._blocking_reason()
        self._draw_button(
            screen_width // 2 - 100,
            start_y_pos,
            start_text,
            "start_button",
            centered=True,
            disabled=blocking_reason is not None,
        )

        # Draw Back button
        back_text = self.lang.get("common.back", "Back")
        self._draw_button(screen_width // 2 - 100, start_y_pos + 60, back_text, "back_button", centered=True)

        # Explain a blocked start (or report the last failed action) on
        # screen rather than only on stdout.
        message = self.status_message or blocking_reason
        if message:
            hint_font = get_font(theme.FONT_SIZE_HINT)
            hint_surface = hint_font.render(message, True, theme.STATUS_WARNING)
            self.screen.blit(hint_surface, hint_surface.get_rect(centerx=screen_width // 2, y=start_y_pos + 124))

        # Keep keyboard focus valid after the element list is rebuilt.
        if self.focus_index >= len(self.interactive_elements):
            self.focus_index = -1

        self._draw_footer_hint()

        pygame.display.flip()

    def _draw_footer_hint(self) -> None:
        """Draw the shared control hint along the bottom of the screen."""
        hint = self.lang.get("common.menu_hint", "Arrows: Move   Enter: Select   Esc: Back")
        hint_font = get_font(theme.FONT_SIZE_HINT)
        hint_surface = hint_font.render(hint, True, theme.TEXT_MUTED)
        self.screen.blit(
            hint_surface,
            hint_surface.get_rect(x=theme.SCREEN_MARGIN_X, bottom=self.screen.get_height() - 8),
        )

    def _draw_button(
        self,
        x: int,
        y: int,
        text: str,
        element_type: str,
        player_idx: int = -1,
        centered: bool = False,
        disabled: bool = False,
    ) -> pygame.Rect:
        """
        Draw a button and register it as an interactive element.

        Args:
            x: X position
            y: Y position
            text: Button text
            element_type: Type of element ('type_toggle', 'difficulty_select',
                         'browse_model', 'start_button', 'back_button')
            player_idx: Player index for player-specific buttons
            centered: Whether to center the button at x position
            disabled: Whether the button is disabled

        Returns:
            Button rect
        """
        # Container width for centered buttons
        button_container_width = 200

        button = widgets.Button.with_label(x, y, text, self.option_font, enabled=not disabled)
        if centered:
            button.rect.x = x + (button_container_width - button.rect.width) // 2

        is_hovered = bool(self.hover_element and self.hover_element.get("rect") == button.rect and not disabled)
        is_focused = self._is_focused(button.rect)
        button.draw(self.screen, hovered=is_hovered, selected=is_focused)

        # Registered even when disabled so a disabled Start can still be
        # clicked/focused to learn *why* it is disabled, and so the element
        # ordering keyboard focus walks stays stable.
        self.interactive_elements.append(
            {"type": element_type, "rect": button.rect, "player_idx": player_idx, "disabled": disabled}
        )

        return button.rect

    def _draw_toggle_button(self, x: int, y: int, text: str, element_type: str, text_color: tuple = theme.TEXT) -> pygame.Rect:
        """
        Draw a toggle button for game options.

        Args:
            x: X position
            y: Y position
            text: Button text (current state)
            element_type: Type of element for click handling
            text_color: Color of the text

        Returns:
            Button rect
        """
        style = replace(widgets.PANEL_BUTTON, text_color=text_color, text_hover_color=text_color)
        button = widgets.Button.with_label(x, y, text, self.option_font, padding_x=15, padding_y=8, style=style)

        is_hovered = bool(self.hover_element and self.hover_element.get("rect") == button.rect)
        button.draw(self.screen, hovered=is_hovered, selected=self._is_focused(button.rect))

        # Register as interactive element
        self.interactive_elements.append({"type": element_type, "rect": button.rect, "player_idx": -1, "disabled": False})

        return button.rect

    def run(self) -> dict[str, Any] | None:
        """
        Run the player configuration menu loop.

        Returns:
            Dict with player configurations, or None if cancelled
        """
        result = None
        clock = pygame.time.Clock()

        # Draw once before event loop to populate interactive_elements
        self.draw()

        # Clear any residual events AFTER draw populates interactive_elements
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                result = self.handle_input(event)
                if result is not None or not self.running:
                    return result

            self.draw()
            clock.tick(theme.MENU_FRAMERATE)

        return result
