"""Menu for configuring LLM API keys."""
from typing import Optional

import pygame

from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.fonts import get_font


class APIKeysMenu:
    """Menu for configuring LLM API keys."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize API keys configuration menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        # Initialize pygame if not already done
        if not pygame.get_init():
            pygame.init()

        # Create screen if not provided
        self.owns_screen = screen is None
        if self.owns_screen:
            self.screen = pygame.display.set_mode((900, 700))
            pygame.display.set_caption("Reinforce Tactics - API Keys")
            # Initialize clipboard support when we own the screen
            try:
                pygame.scrap.init()
            except pygame.error:
                # Clipboard not available on this platform
                pass
        else:
            self.screen = screen

        self.running = True

        # Colors
        self.bg_color = (30, 30, 40)
        self.text_color = (255, 255, 255)
        self.title_color = (100, 200, 255)
        self.input_bg_color = (50, 50, 65)
        self.input_active_color = (70, 70, 90)
        self.button_color = (60, 60, 80)
        self.button_hover_color = (80, 80, 100)

        # Fonts
        self.title_font = get_font(48)
        self.label_font = get_font(28)
        self.input_font = get_font(24)

        # Get language instance
        self.lang = get_language()

        # Load current API keys from settings
        from reinforcetactics.utils.settings import get_settings
        self.settings = get_settings()

        self.api_keys = {
            'openai': self.settings.get_api_key('openai'),
            'anthropic': self.settings.get_api_key('anthropic'),
            'google': self.settings.get_api_key('google')
        }

        # Input tracking
        self.active_input = None
        self.input_rects = {}
        self.button_rects = {}
        self.hover_element = None

        # Test connection status
        self.test_status = {
            'openai': None,  # None, 'testing', 'success', 'failed'
            'anthropic': None,
            'google': None
        }
        self.test_messages = {
            'openai': '',
            'anthropic': '',
            'google': ''
        }

    def handle_input(self, event: pygame.event.Event) -> Optional[bool]:
        """
        Handle input events.

        Args:
            event: Pygame event

        Returns:
            True if settings were saved, False if cancelled, None to continue
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
                return False
            elif event.key == pygame.K_RETURN and self.active_input is None:
                # Save and exit
                return True
            elif self.active_input is not None:
                # Typing in an input field
                # Check for paste first (before other key handlers)
                if event.key == pygame.K_v and (event.mod & pygame.KMOD_CTRL or event.mod & pygame.KMOD_META):
                    # Handle Ctrl+V (Windows/Linux) or Cmd+V (macOS) for paste
                    try:
                        clipboard_text = pygame.scrap.get(pygame.SCRAP_TEXT)
                        if clipboard_text:
                            # Decode bytes to string and strip null characters
                            pasted_text = clipboard_text.decode('utf-8').rstrip('\x00')
                            # Filter to only include printable characters
                            filtered = ''.join(c for c in pasted_text if c.isprintable())
                            self.api_keys[self.active_input] += filtered
                    except (pygame.error, UnicodeDecodeError, AttributeError):
                        # Clipboard operation failed or clipboard not available
                        pass
                elif event.key == pygame.K_RETURN or event.key == pygame.K_TAB:
                    # Move to next field or finish
                    self.active_input = None
                elif event.key == pygame.K_BACKSPACE:
                    self.api_keys[self.active_input] = self.api_keys[self.active_input][:-1]
                elif event.unicode and event.unicode.isprintable():
                    # Only add regular characters if no modifier keys are pressed
                    # This prevents Cmd+V from adding 'v' on macOS
                    if not (event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META | pygame.KMOD_ALT)):
                        self.api_keys[self.active_input] += event.unicode

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos

                # Check input fields
                for key, rect in self.input_rects.items():
                    if rect.collidepoint(mouse_pos):
                        self.active_input = key
                        return None

                # Check buttons
                if 'save' in self.button_rects and self.button_rects['save'].collidepoint(mouse_pos):
                    return True
                if 'back' in self.button_rects and self.button_rects['back'].collidepoint(mouse_pos):
                    self.running = False
                    return False

                # Check test buttons
                for provider in ['openai', 'anthropic', 'google']:
                    test_button_key = f'test_{provider}'
                    if test_button_key in self.button_rects and self.button_rects[test_button_key].collidepoint(mouse_pos):
                        self._test_connection(provider)
                        return None

                # Clicked outside any input
                self.active_input = None

        elif event.type == pygame.MOUSEMOTION:
            # Update hover state
            mouse_pos = event.pos
            self.hover_element = None
            for name, rect in self.button_rects.items():
                if rect.collidepoint(mouse_pos):
                    self.hover_element = name
                    break

        return None

    def draw(self) -> None:
        """Draw the API keys configuration menu."""
        self.screen.fill(self.bg_color)
        self.input_rects = {}
        self.button_rects = {}

        screen_width = self.screen.get_width()

        # Draw title
        title = self.lang.get('api_keys.title', 'LLM API Keys Configuration')
        title_surface = self.title_font.render(title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=30)
        self.screen.blit(title_surface, title_rect)

        # Instructions
        instructions = self.lang.get('api_keys.instructions',
                                     'Enter your API keys for LLM providers (leave blank to use environment variables)')
        inst_surface = self.label_font.render(instructions, True, self.text_color)
        inst_rect = inst_surface.get_rect(centerx=screen_width // 2, y=80)
        # Ensure the instruction text doesn't overflow
        if inst_rect.width > screen_width - 40:
            # Split into two lines
            line1 = 'Enter your API keys for LLM providers'
            line2 = '(leave blank to use environment variables)'
            inst1_surface = self.input_font.render(line1, True, self.text_color)
            inst2_surface = self.input_font.render(line2, True, self.text_color)
            inst1_rect = inst1_surface.get_rect(centerx=screen_width // 2, y=80)
            inst2_rect = inst2_surface.get_rect(centerx=screen_width // 2, y=105)
            self.screen.blit(inst1_surface, inst1_rect)
            self.screen.blit(inst2_surface, inst2_rect)
            start_y = 140
        else:
            self.screen.blit(inst_surface, inst_rect)
            start_y = 120

        # Draw input fields for each provider
        providers = [
            ('openai', 'OpenAI API Key (GPT)'),
            ('anthropic', 'Anthropic API Key (Claude)'),
            ('google', 'Google API Key (Gemini)')
        ]

        y_pos = start_y
        for provider_key, provider_label in providers:
            # Label
            label_surface = self.label_font.render(provider_label, True, self.text_color)
            label_rect = label_surface.get_rect(x=50, y=y_pos)
            self.screen.blit(label_surface, label_rect)

            # Input field
            input_y = y_pos + 35
            input_width = 700
            input_height = 35
            input_x = 50
            input_rect = pygame.Rect(input_x, input_y, input_width, input_height)
            self.input_rects[provider_key] = input_rect

            # Background color based on active state
            bg_color = self.input_active_color if self.active_input == provider_key else self.input_bg_color
            pygame.draw.rect(self.screen, bg_color, input_rect, border_radius=5)
            pygame.draw.rect(self.screen, self.button_color, input_rect, width=2, border_radius=5)

            # Display masked key (show only last 4 chars)
            display_text = self.api_keys[provider_key]
            if len(display_text) > 8 and self.active_input != provider_key:
                display_text = '*' * (len(display_text) - 4) + display_text[-4:]

            # Render text
            if display_text or self.active_input == provider_key:
                text_surface = self.input_font.render(display_text, True, self.text_color)
                text_rect = text_surface.get_rect(x=input_x + 10, centery=input_rect.centery)
                # Clip text if too long
                if text_rect.width > input_width - 20:
                    self.screen.set_clip(pygame.Rect(input_x + 10, input_rect.y, input_width - 20, input_height))
                    text_rect.right = input_x + input_width - 10
                self.screen.blit(text_surface, text_rect)
                self.screen.set_clip(None)

            # Draw cursor if active
            if self.active_input == provider_key:
                cursor_x = input_x + 10 + self.input_font.size(display_text)[0] + 2
                cursor_y1 = input_rect.centery - 10
                cursor_y2 = input_rect.centery + 10
                pygame.draw.line(self.screen, self.text_color, (cursor_x, cursor_y1), (cursor_x, cursor_y2), 2)

            # Draw Test Connection button
            test_button_x = input_x + input_width - 100
            test_button_y = input_y + input_height + 5
            test_button_key = f'test_{provider_key}'

            # Determine button text and color based on test status
            status = self.test_status[provider_key]
            if status == 'testing':
                test_text = 'Testing...'
                test_color = (150, 150, 150)
            elif status == 'success':
                test_text = '✓ Test'
                test_color = (50, 150, 50)
            elif status == 'failed':
                test_text = '✗ Test'
                test_color = (150, 50, 50)
            else:
                test_text = 'Test'
                test_color = self.button_color

            # Draw test button
            test_rect = self._draw_test_button(test_button_x, test_button_y, test_text, test_button_key, test_color)
            self.button_rects[test_button_key] = test_rect

            # Draw status message if available
            if self.test_messages[provider_key]:
                status_surface = self.input_font.render(self.test_messages[provider_key], True, self.text_color)
                status_rect = status_surface.get_rect(x=input_x, y=test_button_y + 35)
                self.screen.blit(status_surface, status_rect)

            y_pos += 110

        # Draw buttons
        button_y = y_pos + 20

        # Save button
        save_text = self.lang.get('common.save', 'Save')
        save_rect = self._draw_button(screen_width // 2 - 120, button_y, save_text, 'save')
        self.button_rects['save'] = save_rect

        # Back button
        back_text = self.lang.get('common.back', 'Back')
        back_rect = self._draw_button(screen_width // 2 + 20, button_y, back_text, 'back')
        self.button_rects['back'] = back_rect

        pygame.display.flip()

    def _draw_button(self, x: int, y: int, text: str, button_name: str) -> pygame.Rect:
        """Draw a button and return its rect."""
        padding_x = 20
        padding_y = 10

        text_surface = self.label_font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect()

        button_width = text_rect.width + 2 * padding_x
        button_height = text_rect.height + 2 * padding_y
        button_rect = pygame.Rect(x, y, button_width, button_height)

        # Button color based on hover state
        bg_color = self.button_hover_color if self.hover_element == button_name else self.button_color
        pygame.draw.rect(self.screen, bg_color, button_rect, border_radius=8)

        # Draw text
        text_rect.center = button_rect.center
        self.screen.blit(text_surface, text_rect)

        return button_rect

    def _draw_test_button(self, x: int, y: int, text: str, button_name: str, bg_color: tuple) -> pygame.Rect:
        """Draw a test button with custom color and return its rect."""
        padding_x = 15
        padding_y = 8

        text_surface = self.input_font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect()

        button_width = text_rect.width + 2 * padding_x
        button_height = text_rect.height + 2 * padding_y
        button_rect = pygame.Rect(x, y, button_width, button_height)

        # Use custom color or hover color
        if self.hover_element == button_name:
            final_color = tuple(min(c + 30, 255) for c in bg_color)
        else:
            final_color = bg_color

        pygame.draw.rect(self.screen, final_color, button_rect, border_radius=5)

        # Draw text
        text_rect.center = button_rect.center
        self.screen.blit(text_surface, text_rect)

        return button_rect

    def _test_connection(self, provider: str) -> None:
        """
        Test connection to an LLM provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google')
        """
        api_key = self.api_keys[provider]
        if not api_key:
            self.test_status[provider] = 'failed'
            self.test_messages[provider] = 'No API key provided'
            return

        self.test_status[provider] = 'testing'
        self.test_messages[provider] = 'Testing...'
        self.draw()  # Redraw to show testing status
        pygame.display.flip()

        try:
            if provider == 'openai':
                self._test_openai(api_key)
            elif provider == 'anthropic':
                self._test_anthropic(api_key)
            elif provider == 'google':
                self._test_google(api_key)

            self.test_status[provider] = 'success'
            self.test_messages[provider] = 'Connection successful!'
        except Exception as e:
            self.test_status[provider] = 'failed'
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 50:
                error_msg = error_msg[:47] + '...'
            self.test_messages[provider] = f'Error: {error_msg}'

    def _test_openai(self, api_key: str) -> None:
        """Test OpenAI API connection."""
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package not installed") from exc

        client = openai.OpenAI(api_key=api_key)
        # Make a minimal API call to test the connection
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'Hello'}],
            max_tokens=5
        )
        if not response.choices:
            raise ValueError("Invalid response from OpenAI")

    def _test_anthropic(self, api_key: str) -> None:
        """Test Anthropic API connection."""
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package not installed") from exc

        client = anthropic.Anthropic(api_key=api_key)
        # Make a minimal API call to test the connection
        response = client.messages.create(
            model='claude-3-haiku-20240307',
            max_tokens=5,
            messages=[{'role': 'user', 'content': 'Hello'}]
        )
        if not response.content:
            raise ValueError("Invalid response from Anthropic")

    def _test_google(self, api_key: str) -> None:
        """Test Google Gemini API connection."""
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError("google-generativeai package not installed") from exc

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        # Make a minimal API call to test the connection
        response = model.generate_content(
            'Hello',
            generation_config=genai.types.GenerationConfig(max_output_tokens=5)
        )
        if not response.text:
            raise ValueError("Invalid response from Google")

    def run(self) -> bool:
        """
        Run the API keys configuration menu.

        Returns:
            True if settings were saved, False if cancelled
        """
        clock = pygame.time.Clock()

        # Initial draw
        self.draw()
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return False

                result = self.handle_input(event)
                if result is not None:
                    if result:
                        # Save the API keys
                        for provider, key in self.api_keys.items():
                            self.settings.set_api_key(provider, key)
                        print("✅ API keys saved")
                    self.running = False
                    return result

            self.draw()
            clock.tick(60)

        return False
