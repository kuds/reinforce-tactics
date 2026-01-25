"""Menu for configuring players."""
from pathlib import Path
from typing import Optional, List, Dict, Any

import pygame

from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font

# Import tkinter optionally for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


class PlayerConfigMenu:
    """Menu for configuring players (Human vs Computer) with difficulty settings."""

    def __init__(self, screen: Optional[pygame.Surface] = None, game_mode: str = "1v1") -> None:
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

        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.selected_color = (255, 200, 50)
        self.hover_color = (200, 180, 100)
        self.title_color = (100, 200, 255)
        self.option_bg_color = (50, 50, 65)
        self.option_bg_hover_color = (70, 70, 90)
        self.option_bg_selected_color = (80, 80, 100)
        self.disabled_color = (100, 100, 120)

        # Fonts
        self.title_font = get_font(48)
        self.label_font = get_font(32)
        self.option_font = get_font(28)

        # Player configurations
        # Default: Player 1 is Human, others are Computer (SimpleBot)
        self.player_configs = []
        for i in range(self.num_players):
            self.player_configs.append({
                'type': 'human' if i == 0 else 'computer',
                'bot_type': None if i == 0 else 'SimpleBot',
                'model_path': None
            })

        # Game options
        self.fog_of_war = False  # Fog of war toggle

        # Model validation state: {player_idx: {'valid': bool, 'error': str}}
        self.model_validation = {}

        # UI interaction tracking
        self.hover_element = None
        self.selected_element = None
        self.interactive_elements: List[Dict[str, Any]] = []

        # Get language instance
        self.lang = get_language()

        # Check which LLM providers have API keys configured
        from reinforcetactics.utils.settings import get_settings
        settings = get_settings()
        self.available_llm_bots = {
            'OpenAIBot': bool(settings.get_api_key('openai')),
            'ClaudeBot': bool(settings.get_api_key('anthropic')),
            'GeminiBot': bool(settings.get_api_key('google'))
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

    def _validate_model(self, model_path: str) -> Dict[str, Any]:
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
                return {'valid': False, 'error': 'File not found'}

            if not model_file.suffix == '.zip':
                return {'valid': False, 'error': 'Must be a .zip file'}

            # Try to load the model with ModelBot
            from reinforcetactics.game.model_bot import ModelBot
            from reinforcetactics.core.game_state import GameState

            # Create a dummy game state for testing
            # Use a simple 6x6 map for validation
            dummy_map = [['p' for _ in range(6)] for _ in range(6)]
            dummy_map[0][0] = 'h_1'  # HQ for player 1
            dummy_map[5][5] = 'h_2'  # HQ for player 2

            dummy_state = GameState(dummy_map, num_players=2)

            # Try to create the bot - this will load the model
            bot = ModelBot(dummy_state, player=2, model_path=str(model_path))

            if bot.model is None:
                return {'valid': False, 'error': 'Failed to load model'}

            return {'valid': True, 'error': None}

        except ImportError as e:
            return {'valid': False, 'error': f'Missing dependency: {e}'}
        except Exception as e:
            return {'valid': False, 'error': f'Load error: {str(e)[:50]}'}

    def _open_file_dialog(self) -> Optional[str]:
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
                title="Select Model File",
                filetypes=[("Model files", "*.zip"), ("All files", "*.*")],
                initialdir="."
            )

            # Clean up
            root.destroy()

            # Return the path or None if cancelled
            return file_path if file_path else None

        except Exception as e:
            print(f"Error opening file dialog: {e}")
            return None

    def handle_input(self, event: pygame.event.Event) -> Optional[Dict[str, Any]]:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            Result dict with player configurations, or None
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return None
            elif event.key == pygame.K_RETURN:
                # Start game with current configuration
                return self._get_result()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos
                for element in self.interactive_elements:
                    if element['rect'].collidepoint(mouse_pos):
                        if element['type'] == 'type_toggle':
                            # Toggle between human and computer
                            player_idx = element['player_idx']
                            config = self.player_configs[player_idx]
                            if config['type'] == 'human':
                                config['type'] = 'computer'
                                config['bot_type'] = 'SimpleBot'
                                config['model_path'] = None
                            else:
                                config['type'] = 'human'
                                config['bot_type'] = None
                                config['model_path'] = None
                                # Clear validation
                                if player_idx in self.model_validation:
                                    del self.model_validation[player_idx]

                        elif element['type'] == 'difficulty_select':
                            # Cycle through available bot types (only those with API keys)
                            player_idx = element['player_idx']
                            config = self.player_configs[player_idx]
                            if config['type'] == 'computer':
                                # Build list of available bot types
                                # All built-in bots are always available
                                bot_types = ['SimpleBot', 'MediumBot', 'AdvancedBot']
                                for bot_name, is_available in self.available_llm_bots.items():
                                    if is_available:
                                        bot_types.append(bot_name)
                                # Add ModelBot if dependencies are available
                                if self.modelbot_available:
                                    bot_types.append('ModelBot')

                                current_bot = config['bot_type']
                                try:
                                    current_idx = bot_types.index(current_bot)
                                    next_idx = (current_idx + 1) % len(bot_types)
                                    config['bot_type'] = bot_types[next_idx]
                                    # Clear model path if switching away from ModelBot
                                    if bot_types[next_idx] != 'ModelBot':
                                        config['model_path'] = None
                                        if player_idx in self.model_validation:
                                            del self.model_validation[player_idx]
                                except ValueError:
                                    # If current bot type is not in list, default to SimpleBot
                                    config['bot_type'] = 'SimpleBot'

                        elif element['type'] == 'browse_model':
                            # Open file dialog to select model
                            player_idx = element['player_idx']
                            config = self.player_configs[player_idx]

                            file_path = self._open_file_dialog()
                            if file_path:
                                config['model_path'] = file_path
                                # Validate the model
                                validation = self._validate_model(file_path)
                                self.model_validation[player_idx] = validation

                        elif element['type'] == 'fog_of_war_toggle':
                            # Toggle fog of war
                            self.fog_of_war = not self.fog_of_war

                        elif element['type'] == 'start_button':
                            return self._get_result()

                        elif element['type'] == 'back_button':
                            self.running = False
                            return None

        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_element = None
            for element in self.interactive_elements:
                if element['rect'].collidepoint(mouse_pos):
                    self.hover_element = element
                    break

        return None

    def _get_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the configured player settings as a result dict.

        Returns:
            Dict with player configurations, or None if validation fails
        """
        # Check if any ModelBot has invalid or missing model
        for i, config in enumerate(self.player_configs):
            if config['type'] == 'computer' and config['bot_type'] == 'ModelBot':
                if not config.get('model_path'):
                    # Show error message
                    print(f"⚠️  Player {i+1}: ModelBot requires a model file")
                    return None

                # Check validation status
                validation = self.model_validation.get(i, {})
                if not validation.get('valid', False):
                    error = validation.get('error', 'Unknown error')
                    print(f"⚠️  Player {i+1}: Model validation failed - {error}")
                    return None

        return {
            'players': self.player_configs,
            'fog_of_war': self.fog_of_war
        }

    def draw(self) -> None:
        """Draw the player configuration menu."""
        self.screen.fill(self.bg_color)
        self.interactive_elements = []

        screen_width = self.screen.get_width()

        # Draw title
        title = self.lang.get('player_config.title', 'Configure Players')
        title_surface = self.title_font.render(title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=30)
        self.screen.blit(title_surface, title_rect)

        # Starting Y position for player configurations
        # Use more compact spacing for 2v2 to fit all elements on screen
        start_y = 80
        spacing_y = 85 if self.num_players > 2 else 100

        # Draw each player's configuration
        for i in range(self.num_players):
            config = self.player_configs[i]
            y_pos = start_y + i * spacing_y

            # Player label
            player_label = self.lang.get(
                'player_config.player',
                'Player {number}'
            ).format(number=i + 1)
            label_surface = self.label_font.render(player_label, True, self.text_color)
            label_rect = label_surface.get_rect(x=50, y=y_pos)
            self.screen.blit(label_surface, label_rect)

            # Type toggle button (Human/Computer)
            type_x = 200
            if config['type'] == 'human':
                type_text = self.lang.get('player_config.type_human', 'Human')
            else:
                type_text = self.lang.get('player_config.type_computer', 'Computer')
            self._draw_button(type_x, y_pos, type_text, 'type_toggle', i)

            # Difficulty selection (only shown if computer)
            if config['type'] == 'computer':
                diff_x = 400
                bot_type = config.get('bot_type', 'SimpleBot')
                # Get display text for bot type
                bot_display_names = {
                    'SimpleBot': 'Simple Bot',
                    'MediumBot': 'Medium Bot',
                    'AdvancedBot': 'Advanced Bot',
                    'OpenAIBot': 'OpenAI (GPT)',
                    'ClaudeBot': 'Claude',
                    'GeminiBot': 'Gemini',
                    'ModelBot': 'Custom Model'
                }
                diff_text = bot_display_names.get(bot_type, bot_type)

                # Add indicator if bot is unavailable (no API key)
                if bot_type in self.available_llm_bots and not self.available_llm_bots[bot_type]:
                    diff_text += ' (No API Key)'

                self._draw_button(
                    diff_x, y_pos, diff_text,
                    'difficulty_select', i, disabled=False
                )

                # If ModelBot is selected, show browse button and model status
                if bot_type == 'ModelBot':
                    # Browse button
                    browse_x = 590
                    self._draw_button(
                        browse_x, y_pos, 'Browse...',
                        'browse_model', i, disabled=False
                    )

                    # Show model status below
                    model_path = config.get('model_path')
                    if model_path:
                        # Show filename
                        filename = Path(model_path).name
                        # Truncate if too long
                        if len(filename) > 30:
                            filename = filename[:27] + '...'

                        # Check validation status
                        validation = self.model_validation.get(i, {})
                        if validation.get('valid'):
                            status_text = f"✓ {filename}"
                            status_color = (100, 255, 100)  # Green
                        else:
                            error = validation.get('error', 'Invalid')
                            status_text = f"✗ {error}"
                            status_color = (255, 100, 100)  # Red
                    else:
                        status_text = "No model selected"
                        status_color = (255, 200, 100)  # Yellow

                    # Draw status text
                    status_surface = self.option_font.render(status_text, True, status_color)
                    status_rect = status_surface.get_rect(x=browse_x, y=y_pos + 30)
                    self.screen.blit(status_surface, status_rect)

        # Draw Game Options section
        options_y = start_y + self.num_players * spacing_y + 10

        # Draw divider line
        divider_y = options_y
        pygame.draw.line(
            self.screen,
            (60, 60, 80),
            (50, divider_y),
            (screen_width - 50, divider_y),
            2
        )

        # Game Options label
        options_label = self.lang.get('player_config.game_options', 'Game Options')
        options_surface = self.label_font.render(options_label, True, self.title_color)
        options_rect = options_surface.get_rect(x=50, y=divider_y + 10)
        self.screen.blit(options_surface, options_rect)

        # Fog of War toggle
        fow_y = divider_y + 50
        fow_label = self.lang.get('player_config.fog_of_war', 'Fog of War')
        fow_label_surface = self.option_font.render(fow_label, True, self.text_color)
        self.screen.blit(fow_label_surface, (50, fow_y))

        fow_status = self.lang.get('common.enabled', 'Enabled') if self.fog_of_war else self.lang.get('common.disabled', 'Disabled')
        fow_color = (100, 255, 100) if self.fog_of_war else (150, 150, 150)
        self._draw_toggle_button(200, fow_y - 5, fow_status, 'fog_of_war_toggle', fow_color)

        # Draw Start Game button
        # Add extra spacing if any player uses ModelBot (for status text)
        extra_spacing = 30 if any(
            c['bot_type'] == 'ModelBot' for c in self.player_configs
        ) else 0
        start_y_pos = fow_y + 50 + extra_spacing
        start_text = self.lang.get('player_config.start_game', 'Start Game')

        # Disable start button if any ModelBot has invalid/missing model
        start_disabled = False
        for i, config in enumerate(self.player_configs):
            if config['type'] == 'computer' and config['bot_type'] == 'ModelBot':
                if not config.get('model_path'):
                    start_disabled = True
                    break
                validation = self.model_validation.get(i, {})
                if not validation.get('valid', False):
                    start_disabled = True
                    break

        self._draw_button(
            screen_width // 2 - 100, start_y_pos, start_text,
            'start_button', centered=True, disabled=start_disabled
        )

        # Draw Back button
        back_text = self.lang.get('common.back', 'Back')
        self._draw_button(
            screen_width // 2 - 100, start_y_pos + 60,
            back_text, 'back_button', centered=True
        )

        pygame.display.flip()

    def _draw_button(
            self, x: int, y: int, text: str, element_type: str,
            player_idx: int = -1, centered: bool = False,
            disabled: bool = False) -> pygame.Rect:
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
        padding_x = 20
        padding_y = 10
        # Container width for centered buttons
        button_container_width = 200

        # Render text
        text_color = self.disabled_color if disabled else self.text_color
        text_surface = self.option_font.render(text, True, text_color)
        text_rect = text_surface.get_rect()

        # Calculate button dimensions
        button_width = text_rect.width + 2 * padding_x
        button_height = text_rect.height + 2 * padding_y

        # Adjust position if centered
        if centered:
            button_x = x + (button_container_width - button_width) // 2
        else:
            button_x = x

        button_rect = pygame.Rect(button_x, y, button_width, button_height)

        # Determine styling
        is_hovered = (
            self.hover_element
            and self.hover_element.get('rect') == button_rect
            and not disabled
        )

        if is_hovered:
            bg_color = self.option_bg_hover_color
            border_color = self.hover_color
        else:
            bg_color = self.option_bg_color
            border_color = self.option_bg_color if disabled else (60, 60, 80)

        # Draw button background
        pygame.draw.rect(self.screen, bg_color, button_rect, border_radius=8)

        # Draw border
        if is_hovered:
            pygame.draw.rect(self.screen, border_color, button_rect, width=2, border_radius=8)

        # Draw text
        text_rect.center = button_rect.center
        self.screen.blit(text_surface, text_rect)

        # Register as interactive element if not disabled
        if not disabled:
            self.interactive_elements.append({
                'type': element_type,
                'rect': button_rect,
                'player_idx': player_idx
            })

        return button_rect

    def _draw_toggle_button(
            self, x: int, y: int, text: str, element_type: str,
            text_color: tuple = (255, 255, 255)) -> pygame.Rect:
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
        padding_x = 15
        padding_y = 8

        # Render text
        text_surface = self.option_font.render(text, True, text_color)
        text_rect = text_surface.get_rect()

        # Calculate button dimensions
        button_width = text_rect.width + 2 * padding_x
        button_height = text_rect.height + 2 * padding_y

        button_rect = pygame.Rect(x, y, button_width, button_height)

        # Determine styling
        is_hovered = (
            self.hover_element
            and self.hover_element.get('rect') == button_rect
        )

        if is_hovered:
            bg_color = self.option_bg_hover_color
            border_color = self.hover_color
        else:
            bg_color = self.option_bg_color
            border_color = (60, 60, 80)

        # Draw button background
        pygame.draw.rect(self.screen, bg_color, button_rect, border_radius=6)
        pygame.draw.rect(self.screen, border_color, button_rect, width=1, border_radius=6)

        # Draw text
        text_rect.center = button_rect.center
        self.screen.blit(text_surface, text_rect)

        # Register as interactive element
        self.interactive_elements.append({
            'type': element_type,
            'rect': button_rect,
            'player_idx': -1
        })

        return button_rect

    def run(self) -> Optional[Dict[str, Any]]:
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
            clock.tick(30)

        return result
