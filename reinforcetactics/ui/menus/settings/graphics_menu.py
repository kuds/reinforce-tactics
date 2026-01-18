"""Graphics settings menu for configuring sprite paths."""
from typing import Optional

import pygame

from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.settings import get_settings
from reinforcetactics.utils.fonts import get_font


class GraphicsMenu(Menu):
    """Graphics settings menu for sprite configuration."""

    def __init__(self, screen: Optional[pygame.Surface] = None) -> None:
        """
        Initialize graphics menu.

        Args:
            screen: Optional pygame surface. If None, creates its own.
        """
        super().__init__(screen, get_language().get('graphics.title', 'Graphics Settings'))
        self.settings = get_settings()
        self.editing_path = None  # Which path is being edited ('unit', 'tile', or 'animation')
        self.input_text = ""
        self.cursor_visible = True
        self.cursor_timer = 0
        self._setup_options()

    def _setup_options(self) -> None:
        """Setup menu options."""
        lang = get_language()

        # Get current settings
        disable_animations = self.settings.get('graphics.disable_animations', False)
        disable_unit_sprites = self.settings.get('graphics.disable_unit_sprites', False)
        use_tile_sprites = self.settings.get('graphics.use_tile_sprites', False)
        animation_path = self.settings.get('graphics.animation_sprites_path', '')
        unit_path = self.settings.get('graphics.unit_sprites_path', '')
        tile_path = self.settings.get('graphics.tile_sprites_path', '')

        # --- Unit Animations (sprite sheets) ---
        # Animation sprites path
        anim_path_display = animation_path if animation_path else lang.get('graphics.not_set', '(not set)')
        self.add_option(
            f"{lang.get('graphics.animation_path', 'Animation Sheets Path')}: {anim_path_display}",
            self._edit_animation_path
        )

        # Toggle to disable animations
        anim_status = "YES" if disable_animations else "NO"
        self.add_option(
            f"Disable Animations: {anim_status}",
            self._toggle_animations
        )

        # --- Static Unit Sprites ---
        # Unit sprites path
        unit_path_display = unit_path if unit_path else lang.get('graphics.not_set', '(not set)')
        self.add_option(
            f"{lang.get('graphics.unit_path', 'Static Sprites Path')}: {unit_path_display}",
            self._edit_unit_path
        )

        # Toggle to disable static sprites
        static_status = "YES" if disable_unit_sprites else "NO"
        self.add_option(
            f"Disable Static Sprites: {static_status}",
            self._toggle_unit_sprites
        )

        # --- Tile Sprites ---
        # Toggle for tile sprites
        tile_status = "ON" if use_tile_sprites else "OFF"
        self.add_option(
            f'Use Tile Sprites: {tile_status}',
            self._toggle_tile_sprites
        )

        # Tile sprites path
        tile_path_display = tile_path if tile_path else lang.get('graphics.not_set', '(not set)')
        self.add_option(
            f"{lang.get('graphics.tile_path', 'Tile Sprites Path')}: {tile_path_display}",
            self._edit_tile_path
        )

        # Back option
        self.add_option(lang.get('common.back', 'Back'), lambda: None)

    def _refresh_options(self) -> None:
        """Refresh menu options after settings change."""
        self.options.clear()
        self._setup_options()

    def _toggle_unit_sprites(self) -> str:
        """Toggle disabling static unit sprites."""
        current = self.settings.get('graphics.disable_unit_sprites', False)
        self.settings.set('graphics.disable_unit_sprites', not current)
        self._refresh_options()
        return 'toggled'

    def _toggle_tile_sprites(self) -> str:
        """Toggle tile sprites on/off."""
        current = self.settings.get('graphics.use_tile_sprites', False)
        self.settings.set('graphics.use_tile_sprites', not current)
        self._refresh_options()
        return 'toggled'

    def _edit_unit_path(self) -> str:
        """Start editing unit sprites path."""
        self.editing_path = 'unit'
        self.input_text = self.settings.get('graphics.unit_sprites_path', '')
        return 'editing'

    def _edit_tile_path(self) -> str:
        """Start editing tile sprites path."""
        self.editing_path = 'tile'
        self.input_text = self.settings.get('graphics.tile_sprites_path', '')
        return 'editing'

    def _toggle_animations(self) -> str:
        """Toggle disabling animations."""
        current = self.settings.get('graphics.disable_animations', False)
        self.settings.set('graphics.disable_animations', not current)
        self._refresh_options()
        return 'toggled'

    def _edit_animation_path(self) -> str:
        """Start editing animation sprites path."""
        self.editing_path = 'animation'
        self.input_text = self.settings.get('graphics.animation_sprites_path', '')
        return 'editing'

    def _save_path(self) -> None:
        """Save the currently edited path."""
        if self.editing_path == 'unit':
            self.settings.set('graphics.unit_sprites_path', self.input_text)
        elif self.editing_path == 'tile':
            self.settings.set('graphics.tile_sprites_path', self.input_text)
        elif self.editing_path == 'animation':
            self.settings.set('graphics.animation_sprites_path', self.input_text)
        self.editing_path = None
        self.input_text = ""
        self._refresh_options()

    def _cancel_edit(self) -> None:
        """Cancel path editing."""
        self.editing_path = None
        self.input_text = ""

    def handle_input(self, event: pygame.event.Event) -> Optional[str]:
        """Handle input events, including text input for path editing."""
        if self.editing_path:
            return self._handle_text_input(event)
        return super().handle_input(event)

    def _handle_text_input(self, event: pygame.event.Event) -> Optional[str]:
        """Handle text input for path editing."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self._save_path()
                return 'saved'
            elif event.key == pygame.K_ESCAPE:
                self._cancel_edit()
                return 'cancelled'
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            elif event.key == pygame.K_v and (event.mod & pygame.KMOD_CTRL):
                # Paste from clipboard
                try:
                    if pygame.scrap.get_init():
                        clipboard_text = pygame.scrap.get(pygame.SCRAP_TEXT)
                        if clipboard_text:
                            # Decode and clean the clipboard text
                            if isinstance(clipboard_text, bytes):
                                clipboard_text = clipboard_text.decode('utf-8', errors='ignore')
                            clipboard_text = clipboard_text.rstrip('\x00').strip()
                            self.input_text += clipboard_text
                except Exception:
                    pass
            else:
                # Add typed character
                if event.unicode and event.unicode.isprintable():
                    self.input_text += event.unicode
        return None

    def draw(self) -> None:
        """Draw the menu, with special handling for path editing mode."""
        if self.editing_path:
            self._draw_path_editor()
        else:
            super().draw()

    def _draw_path_editor(self) -> None:
        """Draw the path editing interface."""
        self.screen.fill(self.bg_color)

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        lang = get_language()

        # Draw title
        if self.editing_path == 'unit':
            title = lang.get('graphics.edit_unit_path', 'Edit Unit Sprites Path')
        elif self.editing_path == 'animation':
            title = lang.get('graphics.edit_animation_path', 'Edit Animation Sprites Path')
        else:
            title = lang.get('graphics.edit_tile_path', 'Edit Tile Sprites Path')

        title_surface = self.title_font.render(title, True, self.title_color)
        title_rect = title_surface.get_rect(centerx=screen_width // 2, y=50)
        self.screen.blit(title_surface, title_rect)

        # Draw instructions
        instructions_font = get_font(24)
        instructions = [
            lang.get('graphics.path_hint', 'Enter the path to your sprites folder'),
            lang.get('graphics.path_example', 'Example: images/sprites/units'),
            "",
            lang.get('graphics.press_enter', 'Press ENTER to save, ESC to cancel'),
            lang.get('graphics.paste_hint', 'Ctrl+V to paste from clipboard')
        ]

        y_offset = 120
        for instruction in instructions:
            inst_surface = instructions_font.render(instruction, True, (180, 180, 180))
            inst_rect = inst_surface.get_rect(centerx=screen_width // 2, y=y_offset)
            self.screen.blit(inst_surface, inst_rect)
            y_offset += 30

        # Draw input box
        input_box_width = min(600, screen_width - 100)
        input_box_height = 50
        input_box = pygame.Rect(
            (screen_width - input_box_width) // 2,
            screen_height // 2 - input_box_height // 2,
            input_box_width,
            input_box_height
        )

        # Background
        pygame.draw.rect(self.screen, (50, 50, 65), input_box, border_radius=8)
        pygame.draw.rect(self.screen, self.selected_color, input_box, width=2, border_radius=8)

        # Input text with cursor
        input_font = get_font(28)

        # Update cursor blink
        self.cursor_timer += 1
        if self.cursor_timer >= 30:
            self.cursor_timer = 0
            self.cursor_visible = not self.cursor_visible

        display_text = self.input_text
        if self.cursor_visible:
            display_text += "|"

        # Truncate display if too long
        max_chars = (input_box_width - 20) // 14  # Approximate char width
        if len(display_text) > max_chars:
            display_text = "..." + display_text[-(max_chars - 3):]

        text_surface = input_font.render(display_text, True, self.text_color)
        text_rect = text_surface.get_rect(midleft=(input_box.left + 15, input_box.centery))
        self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def run(self) -> Optional[str]:
        """Run the graphics menu loop."""
        result = None
        clock = pygame.time.Clock()

        self._populate_option_rects()
        pygame.event.clear()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None

                result = self.handle_input(event)
                if result is not None:
                    # Handle special results
                    if result in ('toggled', 'saved', 'cancelled', 'editing'):
                        # Stay in menu, refresh display
                        self._populate_option_rects()
                        result = None
                    else:
                        return result

            self.draw()
            clock.tick(30)

        return result
