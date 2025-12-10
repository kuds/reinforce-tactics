"""Menu for configuring players."""
import sys
from typing import Optional, List, Dict, Any

import pygame

from reinforcetactics.utils.language import get_language


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
            self.screen = pygame.display.set_mode((800, 600))
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
        self.title_font = pygame.font.Font(None, 48)
        self.label_font = pygame.font.Font(None, 32)
        self.option_font = pygame.font.Font(None, 28)

        # Player configurations
        # Default: Player 1 is Human, others are Computer (SimpleBot)
        self.player_configs = []
        for i in range(self.num_players):
            self.player_configs.append({
                'type': 'human' if i == 0 else 'computer',
                'bot_type': None if i == 0 else 'SimpleBot'
            })

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
                            else:
                                config['type'] = 'human'
                                config['bot_type'] = None

                        elif element['type'] == 'difficulty_select':
                            # Cycle through available bot types (only those with API keys)
                            player_idx = element['player_idx']
                            config = self.player_configs[player_idx]
                            if config['type'] == 'computer':
                                # Build list of available bot types
                                bot_types = ['SimpleBot']  # SimpleBot is always available
                                for bot_name, is_available in self.available_llm_bots.items():
                                    if is_available:
                                        bot_types.append(bot_name)
                                
                                current_bot = config['bot_type']
                                try:
                                    current_idx = bot_types.index(current_bot)
                                    next_idx = (current_idx + 1) % len(bot_types)
                                    config['bot_type'] = bot_types[next_idx]
                                except ValueError:
                                    # If current bot type is not in list, default to SimpleBot
                                    config['bot_type'] = 'SimpleBot'

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

    def _get_result(self) -> Dict[str, Any]:
        """Get the configured player settings as a result dict."""
        return {
            'players': self.player_configs
        }

    def draw(self) -> None:
        """Draw the player configuration menu."""
        self.screen.fill(self.bg_color)
        self.interactive_elements = []

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

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
            player_label = self.lang.get('player_config.player', 'Player {number}').format(number=i + 1)
            label_surface = self.label_font.render(player_label, True, self.text_color)
            label_rect = label_surface.get_rect(x=50, y=y_pos)
            self.screen.blit(label_surface, label_rect)

            # Type toggle button (Human/Computer)
            type_x = 200
            type_text = self.lang.get('player_config.type_human', 'Human') if config['type'] == 'human' else self.lang.get('player_config.type_computer', 'Computer')
            type_rect = self._draw_button(type_x, y_pos, type_text, 'type_toggle', i)

            # Difficulty selection (only shown if computer)
            if config['type'] == 'computer':
                diff_x = 400
                bot_type = config.get('bot_type', 'SimpleBot')
                # Get display text for bot type
                bot_display_names = {
                    'SimpleBot': 'SimpleBot',
                    'OpenAIBot': 'OpenAI (GPT)',
                    'ClaudeBot': 'Claude',
                    'GeminiBot': 'Gemini'
                }
                diff_text = bot_display_names.get(bot_type, bot_type)
                
                # Add indicator if bot is unavailable (no API key)
                if bot_type in self.available_llm_bots and not self.available_llm_bots[bot_type]:
                    diff_text += ' (No API Key)'
                
                self._draw_button(diff_x, y_pos, diff_text, 'difficulty_select', i, disabled=False)

        # Draw Start Game button
        start_y_pos = start_y + self.num_players * spacing_y + 20
        start_text = self.lang.get('player_config.start_game', 'Start Game')
        self._draw_button(screen_width // 2 - 100, start_y_pos, start_text, 'start_button', centered=True)

        # Draw Back button
        back_text = self.lang.get('common.back', 'Back')
        self._draw_button(screen_width // 2 - 100, start_y_pos + 60, back_text, 'back_button', centered=True)

        pygame.display.flip()

    def _draw_button(self, x: int, y: int, text: str, element_type: str, 
                     player_idx: int = -1, centered: bool = False, disabled: bool = False) -> pygame.Rect:
        """
        Draw a button and register it as an interactive element.

        Args:
            x: X position
            y: Y position
            text: Button text
            element_type: Type of element ('type_toggle', 'difficulty_select', 'start_button', 'back_button')
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
        is_hovered = self.hover_element and self.hover_element.get('rect') == button_rect and not disabled

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
                    pygame.quit()
                    sys.exit()

                result = self.handle_input(event)
                if result is not None or not self.running:
                    return result

            self.draw()
            clock.tick(30)

        return result
