"""Menu for configuring LLM API keys."""

import pygame

from reinforcetactics.ui import theme, widgets
from reinforcetactics.ui.icons import get_checkmark_icon, get_x_icon
from reinforcetactics.ui.widgets import TextInput
from reinforcetactics.utils.clipboard import init_clipboard
from reinforcetactics.utils.fonts import get_display_font, get_font
from reinforcetactics.utils.language import get_language


class APIKeysMenu:
    """Menu for configuring LLM API keys."""

    def __init__(self, screen: pygame.Surface | None = None) -> None:
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
            init_clipboard()
        else:
            self.screen = screen

        self.running = True

        # Colors (from shared theme)
        self.bg_color = theme.BG
        self.text_color = theme.TEXT
        self.title_color = theme.TITLE
        self.button_color = (60, 60, 80)

        # Fonts
        self.title_font = get_display_font(theme.FONT_SIZE_TITLE)
        self.label_font = get_font(28)
        self.input_font = get_font(24)

        # Get language instance
        self.lang = get_language()

        # Load current API keys from settings
        from reinforcetactics.utils.settings import get_settings

        self.settings = get_settings()

        self.key_inputs = {
            "openai": TextInput(text=self.settings.get_api_key("openai")),
            "anthropic": TextInput(text=self.settings.get_api_key("anthropic")),
            "google": TextInput(text=self.settings.get_api_key("google")),
        }

        # Input tracking
        self.active_input = None
        self.input_rects: dict[str, pygame.Rect] = {}
        self.button_rects: dict[str, pygame.Rect] = {}
        self.hover_element = None

        # Test connection status
        self.test_status: dict[str, str | None] = {
            "openai": None,  # None, 'testing', 'success', 'failed'
            "anthropic": None,
            "google": None,
        }
        self.test_messages = {"openai": "", "anthropic": "", "google": ""}

    def handle_input(self, event: pygame.event.Event) -> bool | None:
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
                if event.key in (pygame.K_RETURN, pygame.K_TAB):
                    # Move to next field or finish
                    self.active_input = None
                else:
                    self.key_inputs[self.active_input].handle_key(event)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                mouse_pos = event.pos

                # Check input fields
                for key, rect in self.input_rects.items():
                    if rect.collidepoint(mouse_pos):
                        self.active_input = key
                        return None

                # Check buttons
                if "save" in self.button_rects and self.button_rects["save"].collidepoint(mouse_pos):
                    return True
                if "back" in self.button_rects and self.button_rects["back"].collidepoint(mouse_pos):
                    self.running = False
                    return False

                # Check test buttons
                for provider in ["openai", "anthropic", "google"]:
                    test_button_key = f"test_{provider}"
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
        title = self.lang.get("api_keys.title", "LLM API Keys Configuration")
        title_surface = self.title_font.render(title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=30)
        self.screen.blit(title_surface, title_rect)

        # Instructions
        instructions = self.lang.get(
            "api_keys.instructions", "Enter your API keys for LLM providers (leave blank to use environment variables)"
        )
        inst_surface = self.label_font.render(instructions, True, self.text_color)
        inst_rect = inst_surface.get_rect(centerx=screen_width // 2, y=80)
        # Ensure the instruction text doesn't overflow
        if inst_rect.width > screen_width - 40:
            # Split into two lines
            line1 = "Enter your API keys for LLM providers"
            line2 = "(leave blank to use environment variables)"
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
            ("openai", "OpenAI API Key (GPT)"),
            ("anthropic", "Anthropic API Key (Claude)"),
            ("google", "Google API Key (Gemini)"),
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

            # Display masked key (show only last 4 chars) when not editing
            key_input = self.key_inputs[provider_key]
            is_active = self.active_input == provider_key
            display_text = None
            if len(key_input.text) > 8 and not is_active:
                display_text = "*" * (len(key_input.text) - 4) + key_input.text[-4:]

            key_input.draw(self.screen, input_rect, self.input_font, active=is_active, display_text=display_text)

            # Draw Test Connection button
            test_button_x = input_x + input_width - 100
            test_button_y = input_y + input_height + 5
            test_button_key = f"test_{provider_key}"

            # Determine button text, color, and icon based on test status
            status = self.test_status[provider_key]
            test_icon = None
            if status == "testing":
                test_text = "Testing..."
                test_color = (150, 150, 150)
            elif status == "success":
                test_text = "Test"
                test_color = (50, 150, 50)
                test_icon = get_checkmark_icon(size=16, color=(255, 255, 255))
            elif status == "failed":
                test_text = "Test"
                test_color = (150, 50, 50)
                test_icon = get_x_icon(size=16, color=(255, 255, 255))
            else:
                test_text = "Test"
                test_color = self.button_color

            # Draw test button
            test_rect = self._draw_test_button(test_button_x, test_button_y, test_text, test_button_key, test_color, test_icon)
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
        save_text = self.lang.get("common.save", "Save")
        save_rect = self._draw_button(screen_width // 2 - 120, button_y, save_text, "save")
        self.button_rects["save"] = save_rect

        # Back button
        back_text = self.lang.get("common.back", "Back")
        back_rect = self._draw_button(screen_width // 2 + 20, button_y, back_text, "back")
        self.button_rects["back"] = back_rect

        pygame.display.flip()

    def _draw_button(self, x: int, y: int, text: str, button_name: str) -> pygame.Rect:
        """Draw a button and return its rect."""
        button = widgets.Button.with_label(x, y, text, self.label_font)
        button.draw(self.screen, hovered=self.hover_element == button_name)
        return button.rect

    def _draw_test_button(
        self, x: int, y: int, text: str, button_name: str, bg_color: tuple, icon: pygame.Surface = None
    ) -> pygame.Rect:
        """Draw a test button with custom color and optional icon, return its rect."""
        padding_x = 15
        padding_y = 8
        icon_spacing = 4

        text_surface = self.input_font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect()

        # Calculate button width including icon if present
        icon_width = icon.get_width() + icon_spacing if icon else 0
        button_width = text_rect.width + icon_width + 2 * padding_x
        button_height = text_rect.height + 2 * padding_y
        button_rect = pygame.Rect(x, y, button_width, button_height)

        # Use custom color or hover color
        if self.hover_element == button_name:
            final_color = tuple(min(c + 30, 255) for c in bg_color)
        else:
            final_color = bg_color

        pygame.draw.rect(self.screen, final_color, button_rect, border_radius=5)

        # Draw icon and text
        if icon:
            # Calculate positions for icon + text centered in button
            total_content_width = icon.get_width() + icon_spacing + text_rect.width
            content_start_x = button_rect.centerx - total_content_width // 2
            icon_rect = icon.get_rect(x=content_start_x, centery=button_rect.centery)
            self.screen.blit(icon, icon_rect)
            text_rect.x = content_start_x + icon.get_width() + icon_spacing
            text_rect.centery = button_rect.centery
        else:
            text_rect.center = button_rect.center

        self.screen.blit(text_surface, text_rect)

        return button_rect

    def _test_connection(self, provider: str) -> None:
        """
        Test connection to an LLM provider.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google')
        """
        api_key = self.key_inputs[provider].text
        if not api_key:
            self.test_status[provider] = "failed"
            self.test_messages[provider] = "No API key provided"
            return

        self.test_status[provider] = "testing"
        self.test_messages[provider] = "Testing..."
        self.draw()  # Redraw to show testing status
        pygame.display.flip()

        try:
            if provider == "openai":
                self._test_openai(api_key)
            elif provider == "anthropic":
                self._test_anthropic(api_key)
            elif provider == "google":
                self._test_google(api_key)

            self.test_status[provider] = "success"
            self.test_messages[provider] = "Connection successful!"
        except Exception as e:
            self.test_status[provider] = "failed"
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 50:
                error_msg = error_msg[:47] + "..."
            self.test_messages[provider] = f"Error: {error_msg}"

    def _test_openai(self, api_key: str) -> None:
        """Test OpenAI API connection."""
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai package not installed") from exc

        client = openai.OpenAI(api_key=api_key)
        # Make a minimal API call to test the connection. Uses the same
        # default model as OpenAIBot so the test can't pass/fail against a
        # different model than actual games use.
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07", messages=[{"role": "user", "content": "Hello"}], max_completion_tokens=5
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
        # Minimal call on the current cheapest model (the previous
        # claude-3-haiku-20240307 is deprecated and slated for retirement,
        # which made the test fail with valid keys).
        response = client.messages.create(
            model="claude-haiku-4-5", max_tokens=5, messages=[{"role": "user", "content": "Hello"}]
        )
        if not response.content:
            raise ValueError("Invalid response from Anthropic")

    def _test_google(self, api_key: str) -> None:
        """Test Google Gemini API connection."""
        # The legacy google.generativeai SDK is not installed by the [llm]
        # extra -- GeminiBot uses the google-genai SDK, so test with the
        # same package and default model.
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("google-genai package not installed") from exc

        client = genai.Client(api_key=api_key)
        # Make a minimal API call to test the connection
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Hello",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        if response.text is None and not response.candidates:
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
                        for provider, key_input in self.key_inputs.items():
                            self.settings.set_api_key(provider, key_input.text)
                        print("✅ API keys saved")
                    self.running = False
                    return result

            self.draw()
            clock.tick(60)

        return False
