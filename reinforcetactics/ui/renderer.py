"""
Pygame rendering for the strategy game.
"""

import math
import os
import random
import time

import numpy as np
import pygame

from reinforcetactics.core.visibility import SHROUDED, UNEXPLORED, VISIBLE
from reinforcetactics.rules import TILE_TYPES, UNIT_DATA
from reinforcetactics.ui import theme
from reinforcetactics.ui.assets import (
    BASE_SPRITE_COLORS,
    NEUTRAL_STRUCTURE_PALETTE,
    PLAYER_COLORS,
    STRUCTURE_TILE_TYPES,
    TEAM_PALETTES,
    TILE_IMAGES,
    TILE_SIZE,
    UNIT_ASSETS,
    tile_color,
)
from reinforcetactics.ui.sprite_animator import SpriteAnimator, scale_unit_sprite
from reinforcetactics.ui.widgets.text import ellipsize
from reinforcetactics.utils.clipboard import init_clipboard
from reinforcetactics.utils.fonts import get_display_font, get_font
from reinforcetactics.utils.language import get_language
from reinforcetactics.utils.settings import get_settings


def _pulse(period_ms):
    """Return a 0..1 sine pulse based on the current tick count."""
    return 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / period_ms)


def _lerp_color(a, b, t):
    """Linearly interpolate between two RGB colors."""
    return tuple(int(ca + (cb - ca) * t) for ca, cb in zip(a, b))


def _resolve_bundled_sprites_path():
    """Locate the bundled ``assets/sprites`` directory.

    Walks up from this file (the repo ships ``assets/sprites/`` at the
    root) and also checks the current working directory. Returns ``None``
    if the directory can't be found.
    """
    from pathlib import Path

    candidates = [Path.cwd() / "assets" / "sprites"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "assets" / "sprites")

    for c in candidates:
        if c.is_dir():
            return str(c)
    return None


class Renderer:
    """Handles all Pygame rendering."""

    def __init__(self, game_state, replay_mode=False, viewing_player=None, headless=False, pixel_art=None):
        """
        Initialize the renderer.

        Args:
            game_state: GameState instance to render
            replay_mode: If True, skip rendering gameplay controls (End Turn, Resign)
            viewing_player: Player whose perspective to render (for fog of war).
                           If None, shows current player's view (or omniscient if no FOW).
            headless: If True, render to an offscreen surface without opening a window.
                     Useful for recording videos in notebooks or CI environments.
            pixel_art: Authoritative override for pixel-art rendering.
                     ``None`` (default) uses ``settings.json``: sprites from
                     the configured paths, or the bundled ``assets/sprites/``
                     when none is set, unless ``graphics.pixel_art`` is off.
                     ``True`` always uses the bundled directory and
                     force-enables tile/unit sprites. ``False`` force-disables
                     sprite loading and uses the fallback (coloured rects +
                     unit letters), regardless of settings.
        """
        self.game_state = game_state
        self.replay_mode = replay_mode
        self.viewing_player = viewing_player  # For FOW perspective
        self.headless = headless
        self._pixel_art = pixel_art
        self._sprites_override = _resolve_bundled_sprites_path() if pixel_art is True else None

        # Initialize Pygame if not already initialized
        if not pygame.get_init():
            pygame.init()

        # The board is drawn at the window's top-left corner, one TILE_SIZE
        # square per tile. In play the HUD (gold, turn, End Turn, Resign) is
        # a panel to the right of it: drawn over the board, as it used to be,
        # it hid units in the top rows of a 20x20 map and took their clicks.
        # Replays and headless capture keep a board-sized surface (the replay
        # player lays out its own controls, and videos are the board alone).
        self.board_rect = pygame.Rect(0, 0, game_state.grid.width * TILE_SIZE, game_state.grid.height * TILE_SIZE)
        self.has_hud_panel = not (headless or replay_mode)

        # UI elements (lays out the HUD panel, which sets the window size)
        self._setup_ui_elements()

        # Setup display
        screen_width, screen_height = self.window_size
        if headless:
            # Headless: prefer a dummy SDL display so the surface lives on
            # a properly initialised SDL backbuffer. On some environments
            # (notably Google Colab) a bare ``pygame.Surface`` can leave
            # parts of the buffer uninitialised and show through as
            # garbage in captured frames. Fall back to a plain Surface if
            # the display can't be initialised, or if one already exists
            # (so we don't clobber a live game window).
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            existing = pygame.display.get_surface()
            if existing is None:
                try:
                    if not pygame.display.get_init():
                        pygame.display.init()
                    self.screen = pygame.display.set_mode((screen_width, screen_height))
                except pygame.error:
                    self.screen = pygame.Surface((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.screen.fill((0, 0, 0))
        else:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("Reinforce Tactics")

        # Initialize clipboard support
        if not headless:
            init_clipboard()

        # Load settings
        self.settings = get_settings()

        # Load tile images
        self.tile_images = self._load_tile_images()

        # Load unit images
        self.unit_images = self._load_unit_images()

        # Initialize sprite animator for animations
        self.animator = self._init_animator()

        # Animation timing
        self.last_frame_time = time.time()
        self.delta_time = 0.0

        # Pre-allocate tile-sized overlay surfaces used every frame.
        # Fog is two-tier: unexplored tiles are darker than shrouded ones
        # (seen before, currently out of sight).
        self._fog_unexplored = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        self._fog_unexplored.fill(theme.OVERLAY_FOG_UNEXPLORED)
        self._fog_shrouded = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
        self._fog_shrouded.fill(theme.OVERLAY_FOG_SHROUDED)

        self._move_overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
        self._move_overlay.set_alpha(theme.OVERLAY_MOVEMENT_ALPHA)
        self._move_overlay.fill(theme.OVERLAY_MOVEMENT)

        self._target_overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
        self._target_overlay.set_alpha(theme.OVERLAY_TARGET_ALPHA)
        self._target_overlay.fill(theme.OVERLAY_TARGET)

        self._attack_range_overlay = pygame.Surface((TILE_SIZE, TILE_SIZE))
        self._attack_range_overlay.set_alpha(theme.OVERLAY_ATTACK_RANGE_ALPHA)
        self._attack_range_overlay.fill(theme.OVERLAY_ATTACK_RANGE)

        # Blend overlay cache (for unit tinting)
        self._overlay_cache = {}

        # Rendered-text caches: pygame text rendering is expensive, so HUD
        # text, tooltips, status indicators, and letter fallbacks are only
        # re-rendered when their content changes.
        self._text_cache = {}
        self._tooltip_cache = None
        self._letter_cache = {}

        # Per-overlay fade-in state: kind -> (signature, start_ticks)
        self._overlay_anim = {}

    def _use_sprites(self):
        """Whether to load sprites at all (False draws coloured tiles and letters)."""
        if self._pixel_art is not None:
            return self._pixel_art
        return bool(self.settings.get("graphics.pixel_art", True))

    def _resolve_sprites_path(self, category):
        """Resolve the sprite directory for a category.

        ``pixel_art=True`` always uses the bundled ``assets/sprites/``.
        Otherwise a path from the Graphics settings wins, and the bundled
        directory is the default when none is set.
        """
        base = self._sprites_override
        if not base:
            configured = self.settings.get_sprites_path(category)
            if configured:
                return configured
            # Custom static unit sprites with no sheets configured: the
            # bundled sheets would take priority and hide them.
            if category == "animation" and self.settings.get_sprites_path("units"):
                return ""
            base = _resolve_bundled_sprites_path()
            if not base:
                return ""
        subdir = {"units": "units", "tiles": "tiles", "animation": "units"}.get(category, category)
        return os.path.join(base, subdir)

    def _load_tile_images(self):
        """Load tile images, discover variants, and generate team-coloured
        structure variants.

        Variant discovery: for each base filename (e.g. ``grass.png``)
        we also look for ``grass_2.png``, ``grass_3.png``, etc.  Each
        position on the map deterministically picks one variant so the
        terrain looks varied but stays stable across frames.
        """
        tile_images = {}  # type_name -> base surface (single)
        tile_variants = {}  # type_name -> [surface, ...]

        # Letter mode (pixel_art=False or graphics.pixel_art off)
        if not self._use_sprites():
            self.tile_variants = {}
            self.team_tile_variants = {}
            return tile_images

        use_tile_sprites = bool(self._sprites_override) or self.settings.get("graphics.use_tile_sprites", True)
        tile_sprites_path = self._resolve_sprites_path("tiles")

        for tile_type, filename in TILE_IMAGES.items():
            variants = []

            # --- base image ---
            try:
                if use_tile_sprites and tile_sprites_path:
                    full_path = os.path.join(tile_sprites_path, filename)
                else:
                    full_path = filename

                image = pygame.image.load(full_path)
                if not self.headless:
                    image = image.convert_alpha()
                base_surface = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
                tile_images[tile_type] = base_surface
                variants.append(base_surface)
            except (pygame.error, FileNotFoundError):
                tile_images[tile_type] = None

            # --- numbered variants (_2, _3, _4, ...) ---
            if use_tile_sprites and tile_sprites_path and variants:
                stem, ext = os.path.splitext(filename)
                num = 2
                while True:
                    variant_name = f"{stem}_{num}{ext}"
                    variant_path = os.path.join(tile_sprites_path, variant_name)
                    try:
                        img = pygame.image.load(variant_path)
                        if not self.headless:
                            img = img.convert_alpha()
                        variants.append(pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE)))
                        num += 1
                    except (pygame.error, FileNotFoundError):
                        break

            if variants:
                tile_variants[tile_type] = variants

        self.tile_variants = tile_variants

        # Generate team-coloured variants for structure tiles
        self.team_tile_variants = {}
        self._generate_team_tile_variants(tile_variants)

        return tile_images

    def _generate_team_tile_variants(self, tile_variants):
        """
        Create team-coloured copies of structure tile sprite variants.

        For every visual variant of each structure type, replaces the
        base blue palette with each team's colours and a neutral gray
        palette for unowned structures.  Results are stored in
        ``self.team_tile_variants[(tile_type_name, player)]`` as lists
        matching the base variant order.
        """
        if not BASE_SPRITE_COLORS:
            return

        for tile_type_name in STRUCTURE_TILE_TYPES:
            base_list = tile_variants.get(tile_type_name)
            if not base_list:
                continue

            # Neutral variants (player=None)
            self.team_tile_variants[(tile_type_name, None)] = [
                self._recolor_tile(s, BASE_SPRITE_COLORS, NEUTRAL_STRUCTURE_PALETTE) for s in base_list
            ]

            # Per-team variants
            for player, palette in TEAM_PALETTES.items():
                if palette is None:
                    self.team_tile_variants[(tile_type_name, player)] = base_list
                else:
                    self.team_tile_variants[(tile_type_name, player)] = [
                        self._recolor_tile(s, BASE_SPRITE_COLORS, palette) for s in base_list
                    ]

    @staticmethod
    def _recolor_tile(surface, base_colors, team_colors):
        """
        Replace base palette colours with team colours in a tile surface.

        Args:
            surface: Source pygame.Surface (not mutated)
            base_colors: List of (R, G, B) colours to find
            team_colors: List of (R, G, B) replacement colours

        Returns:
            New pygame.Surface with swapped colours
        """
        recoloured = surface.copy()
        pxa = pygame.PixelArray(recoloured)
        for src, dst in zip(base_colors, team_colors):
            pxa.replace(src, dst, 0.01)
        del pxa
        return recoloured

    def _load_unit_images(self):
        """Load unit images from configured sprites path."""
        unit_images = {}

        # Letter mode (pixel_art=False or graphics.pixel_art off)
        if not self._use_sprites():
            return unit_images

        # Get the configured unit sprites path
        unit_sprites_path = self._resolve_sprites_path("units")
        if not unit_sprites_path:
            return unit_images

        # Load sprite for each unit type
        for unit_type, unit_assets in UNIT_ASSETS.items():
            static_path = unit_assets.get("static_path", "")
            if static_path:
                try:
                    full_path = os.path.join(unit_sprites_path, static_path)
                    image = pygame.image.load(full_path)
                    if not self.headless:
                        image = image.convert_alpha()
                    unit_images[unit_type] = scale_unit_sprite(image, TILE_SIZE)
                except (pygame.error, FileNotFoundError):
                    unit_images[unit_type] = None

        return unit_images

    def _init_animator(self):
        """Initialize the sprite animator for unit animations."""
        # Letter mode (pixel_art=False or graphics.pixel_art off)
        if not self._use_sprites():
            return None

        animation_path = self._resolve_sprites_path("animation")
        if not animation_path:
            return None

        return SpriteAnimator(animation_path, headless=self.headless)

    def reload_sprites(self):
        """Reload sprites after settings change."""
        self.tile_images = self._load_tile_images()
        self.unit_images = self._load_unit_images()
        self.animator = self._init_animator()

    def _setup_ui_elements(self):
        """Lay out the HUD and pre-render its static labels (localized).

        In play the HUD is a panel to the right of the board: the player
        card, gold and turn at the top, End Turn and Resign at the bottom.
        Without the panel (replays, headless capture) there are no buttons.
        """
        lang = get_language()

        self._hud_font = get_font(theme.FONT_SIZE_SUBHEADING)
        self._hud_caption_font = get_font(theme.FONT_SIZE_HINT)
        self._hud_title_font = get_display_font(theme.FONT_SIZE_BODY)
        self._hud_button_font = get_display_font(theme.FONT_SIZE_BODY)
        self._hud_badge_font = get_font(theme.FONT_SIZE_LABEL)

        self._hud_player_label = lang.get("player", "Player")
        self._hud_gold_label = lang.get("gold", "Gold")
        self._hud_turn_label = lang.get("turn", "Turn")

        pad = theme.HUD_PANEL_PADDING
        content_width = theme.HUD_PANEL_WIDTH - 2 * pad
        text_width = content_width - 16

        def label(font, text, color):
            return font.render(ellipsize(text, font, text_width), True, color)

        self._end_turn_label = label(self._hud_button_font, lang.get("end_turn", "End Turn"), theme.TEXT)
        self._resign_label = label(self._hud_button_font, lang.get("resign", "Resign"), theme.TEXT)
        self._fow_label = label(self._hud_badge_font, lang.get("player_config.fog_of_war", "Fog of War"), theme.HUD_FOW_TEXT)
        self._hud_gold_caption = label(self._hud_caption_font, self._hud_gold_label, theme.HUD_LABEL_TEXT)
        self._hud_turn_caption = label(self._hud_caption_font, self._hud_turn_label, theme.HUD_LABEL_TEXT)

        if not self.has_hud_panel:
            self.hud_rect = pygame.Rect(self.board_rect.right, 0, 0, self.board_rect.height)
            self.end_turn_button = pygame.Rect(0, 0, 0, 0)
            self.resign_button = pygame.Rect(0, 0, 0, 0)
            return

        x = self.board_rect.right + pad
        stat_height = self._hud_caption_font.get_height() + self._hud_font.get_height()
        self._hud_player_card = pygame.Rect(x, pad, content_width, 40)
        self._hud_gold_pos = (x, self._hud_player_card.bottom + 16)
        self._hud_turn_pos = (x, self._hud_gold_pos[1] + stat_height + 12)
        self._hud_fow_pos = (x, self._hud_turn_pos[1] + stat_height + 16)
        info_bottom = self._hud_fow_pos[1] + self._fow_label.get_height() + 4

        # End Turn and Resign sit at the bottom of the panel. A board too
        # short to fit the panel beside it gets a taller window instead.
        button_height, gap = 40, 10
        height = max(self.board_rect.height, info_bottom + 16 + 2 * button_height + gap + pad)
        self.hud_rect = pygame.Rect(self.board_rect.right, 0, theme.HUD_PANEL_WIDTH, height)
        self.resign_button = pygame.Rect(x, height - pad - button_height, content_width, button_height)
        self.end_turn_button = pygame.Rect(x, self.resign_button.y - gap - button_height, content_width, button_height)

    @property
    def window_size(self):
        """(width, height) of the window: the board, plus the HUD panel in play."""
        return self.hud_rect.right, max(self.board_rect.height, self.hud_rect.height)

    def render(self):
        """Render the entire game state."""
        # Update animation timing
        current_time = time.time()
        self.delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        self.screen.fill((0, 0, 0))

        # Draw grid
        self._draw_grid()

        # Draw units
        self._draw_units()

        # Draw UI
        self._draw_ui()

        # Note: pygame.display.flip() is called by the main game loop

    def _draw_grid(self):
        """Draw the tile grid."""
        # Determine which player's perspective to use for fog of war
        fow_player = self._get_fow_player()

        for y in range(self.game_state.grid.height):
            for x in range(self.game_state.grid.width):
                tile = self.game_state.grid.tiles[y][x]
                self._draw_tile(tile, fow_player)

    def _get_fow_player(self):
        """Get the player whose fog of war perspective to render."""
        if not self.game_state.fog_of_war:
            return None  # No FOW, show everything

        if self.viewing_player is not None:
            return self.viewing_player

        # Default to current player
        return self.game_state.current_player

    def _get_visibility_state(self, x, y, player):
        """Get the visibility state of a tile for a player."""
        if player is None or not self.game_state.fog_of_war:
            return VISIBLE

        vis_map = self.game_state.visibility_maps.get(player)
        if vis_map is None:
            return VISIBLE

        return vis_map.get_visibility_state(x, y)

    def _draw_tile(self, tile, fow_player=None):
        """Draw a single tile with improved visuals and fog of war."""
        rect = pygame.Rect(tile.x * TILE_SIZE, tile.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

        # Check visibility state for fog of war
        vis_state = self._get_visibility_state(tile.x, tile.y, fow_player)

        tile_type_name = TILE_TYPES.get(tile.type, "OCEAN")

        # Under fog of war a structure is drawn with the owner the viewer
        # knows (GameState.known_structure: live in sight, else as last
        # seen, HQs known from the start), never the live one: sprite
        # variants and the fallback colour are both owner-coloured and show
        # through the fog overlay (review pygame-12).
        shown_owner = self._shown_structure_owner(tile, fow_player) if tile.is_capturable() else tile.player

        # Draw tile image or color (always draw terrain, even in fog)
        # For structures, pick from team-coloured variants; for terrain,
        # pick from tile variants.  Selection is deterministic per
        # position so each tile always shows the same variant.
        variants = None
        if tile_type_name in STRUCTURE_TILE_TYPES:
            variants = self.team_tile_variants.get((tile_type_name, shown_owner))
        if variants is None:
            variants = self.tile_variants.get(tile_type_name)

        tile_surface = None
        if variants:
            idx = (tile.x * 7 + tile.y * 13) % len(variants)
            tile_surface = variants[idx]
        if tile_surface is None:
            tile_surface = self.tile_images.get(tile_type_name)

        if tile_surface:
            self.screen.blit(tile_surface, rect)
        else:
            color = tile_color(tile.type, shown_owner)
            pygame.draw.rect(self.screen, color, rect)

            # Add visual variety to tiles
            if tile.type == "p":  # Grass - checkerboard pattern
                if (tile.x + tile.y) % 2 == 0:
                    lighter = tuple(min(c + 15, 255) for c in color)
                    pygame.draw.rect(self.screen, lighter, rect)

            elif tile.type == "o":  # Ocean - checkerboard pattern
                if (tile.x + tile.y) % 2 == 0:
                    lighter = tuple(min(c + 15, 255) for c in color)
                    pygame.draw.rect(self.screen, lighter, rect)

            elif tile.type == "w":  # Water - darker edges
                darker = tuple(max(c - 30, 0) for c in color)
                pygame.draw.line(self.screen, darker, rect.topleft, rect.topright, 2)
                pygame.draw.line(self.screen, darker, rect.topleft, rect.bottomleft, 2)

            elif tile.type == "m":  # Mountain - lighter top
                lighter = tuple(min(c + 40, 255) for c in color)
                top_rect = pygame.Rect(rect.x, rect.y, rect.width, rect.height // 2)
                pygame.draw.rect(self.screen, lighter, top_rect)

            elif tile.type == "f":  # Forest - random dots for texture
                # Local RNG: reseeding the global random module here would
                # make gameplay rolls (e.g. Rogue evade) deterministic after
                # every rendered frame.
                rng = random.Random(tile.x * 1000 + tile.y)
                for _ in range(3):
                    x = rect.x + rng.randint(5, rect.width - 5)
                    y = rect.y + rng.randint(5, rect.height - 5)
                    darker = tuple(max(c - 20, 0) for c in color)
                    pygame.draw.circle(self.screen, darker, (x, y), 2)

            elif tile.type == "r":  # Road - center stripe
                stripe_color = tuple(min(c + 30, 255) for c in color)
                center_y = rect.centery
                pygame.draw.line(self.screen, stripe_color, (rect.left, center_y), (rect.right, center_y), 2)

            elif tile.type in ["h", "b", "t"]:  # Structures - border highlight
                # Border in the known owner's colour (every HQ is known from
                # the start; other structures once seen)
                if shown_owner:
                    player_color = PLAYER_COLORS.get(shown_owner, (255, 255, 255))
                    pygame.draw.rect(self.screen, player_color, rect, 3)

            # Tile border
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

        # Draw structure health bar (only for capturable structures and visible tiles)
        if tile.is_capturable() and tile.health is not None:
            if vis_state == VISIBLE:
                self._draw_structure_health_bar(tile)

        # Apply fog overlay: unexplored tiles are darker than shrouded ones
        if vis_state == UNEXPLORED:
            self.screen.blit(self._fog_unexplored, rect)
        elif vis_state == SHROUDED:
            self.screen.blit(self._fog_shrouded, rect)

    def _shown_structure_owner(self, tile, fow_player):
        """The owner of structure ``tile`` as ``fow_player`` knows it (live without fog)."""
        if fow_player is None:
            return tile.player
        known = self.game_state.known_structure(fow_player, tile.x, tile.y)
        return known.owner if known is not None else None

    def _draw_structure_health_bar(self, tile):
        """Draw health bar for structures."""
        margin = theme.HEALTH_BAR_MARGIN
        bar_width = TILE_SIZE - 2 * margin
        bar_height = theme.HEALTH_BAR_STRUCTURE_HEIGHT
        bar_x = tile.x * TILE_SIZE + margin
        bar_y = tile.y * TILE_SIZE + margin

        # Background
        pygame.draw.rect(self.screen, theme.HEALTH_STRUCTURE_BG, (bar_x, bar_y, bar_width, bar_height))

        # Foreground
        health_percentage = tile.health / tile.max_health
        current_bar_width = int(bar_width * health_percentage)

        if tile.type == "h":
            health_color = theme.HEALTH_STRUCTURE_HQ
        elif tile.type == "b":
            health_color = theme.HEALTH_STRUCTURE_BUILDING
        else:
            health_color = theme.HEALTH_STRUCTURE_TOWER

        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 1)

    def _draw_units(self):
        """Draw all units.

        Sprites are taller than a tile and overflow it upward, so units are
        drawn top to bottom (by ``y``, then ``x``): a unit lower on the
        screen stands in front of the one above it. Health bars and status
        badges go on in a second pass so no neighbour's sprite covers them.
        """
        fow_player = self._get_fow_player()

        shown = []
        for unit in self.game_state.units:
            # With fog of war, only draw visible units (own units + units in visible tiles)
            if fow_player is not None:
                # Always show own units
                if unit.player != fow_player:
                    # Check if enemy unit is in a visible tile
                    vis_state = self._get_visibility_state(unit.x, unit.y, fow_player)
                    if vis_state != VISIBLE:
                        continue

            shown.append(unit)

        shown.sort(key=lambda u: (u.y, u.x))
        for unit in shown:
            self._draw_unit(unit)
        for unit in shown:
            self._draw_unit_status(unit)

    def _draw_unit(self, unit):
        """
        Draw a single unit's body (status overlays are :meth:`_draw_unit_status`).

        Rendering priority (cascading fallback):
        1. Animated sprite sheet (if available and not disabled)
        2. Static sprite image (if available and not disabled)
        3. Letter representation (always available as fallback)
        """
        # Check settings for what's disabled
        animations_disabled = self.settings.get("graphics.disable_animations", False)
        static_sprites_disabled = self.settings.get("graphics.disable_unit_sprites", False)

        # Try animated sprite first (if not disabled)
        if not animations_disabled:
            if self.animator and self.animator.has_animations(unit.type):
                animated_frame = self.animator.get_frame(unit, self.delta_time)
                if animated_frame:
                    self._draw_unit_sprite(unit, animated_frame)
                    return

        # Fall back to static sprite (if not disabled)
        if not static_sprites_disabled:
            static_sprite = self.unit_images.get(unit.type)
            if static_sprite:
                self._draw_unit_sprite(unit, static_sprite)
                return

        # Fall back to letter representation
        self._draw_unit_letter(unit)

    def _draw_unit_status(self, unit):
        """Draw a unit's status badges and health bar, anchored to its tile."""
        self._draw_paralysis_indicator(unit)
        self._draw_haste_indicator(unit)
        self._draw_unit_health_bar(unit)

    def _draw_status_badge(self, cache_key, text, color, *, topright=None, bottomleft=None):
        """Draw a compact status badge: dark pill, colored border and text.

        Sized to stay inside a single tile so badges never spill onto
        neighbouring tiles or cover the unit sprite.
        """
        text_surface = self._cached_text(cache_key, text, get_font(14), color)
        if topright is not None:
            rect = text_surface.get_rect(topright=topright)
        else:
            rect = text_surface.get_rect(bottomleft=bottomleft)
        bg_rect = rect.inflate(6, 2)
        pygame.draw.rect(self.screen, theme.TOOLTIP_BG, bg_rect, border_radius=theme.BORDER_RADIUS_SMALL)
        pygame.draw.rect(self.screen, color, bg_rect, 1, border_radius=theme.BORDER_RADIUS_SMALL)
        self.screen.blit(text_surface, rect)

    def _draw_paralysis_indicator(self, unit):
        """Draw paralysis indicator if unit is paralyzed (pulsing border)."""
        if not unit.is_paralyzed():
            return

        tile_rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        border_color = _lerp_color(theme.STATUS_PARALYSIS, theme.STATUS_PARALYSIS_TINT, _pulse(theme.STATUS_PULSE_MS))
        pygame.draw.rect(self.screen, border_color, tile_rect, 3)

        self._draw_status_badge(
            f"paralysis_{unit.paralyzed_turns}",
            f"P{unit.paralyzed_turns}",
            theme.STATUS_PARALYSIS_TINT,
            topright=(tile_rect.right - 3, tile_rect.top + 3),
        )

    def _draw_haste_indicator(self, unit):
        """Draw haste indicator if unit is hasted."""
        if not unit.is_hasted:
            return

        # Anchored just above the health bar so neither covers the other.
        bar_top = unit.y * TILE_SIZE + TILE_SIZE - theme.HEALTH_BAR_UNIT_HEIGHT - theme.HEALTH_BAR_MARGIN
        self._draw_status_badge(
            "haste",
            "H",
            theme.STATUS_HASTE,
            bottomleft=(unit.x * TILE_SIZE + 3, bar_top - 2),
        )

    def _draw_unit_sprite(self, unit, sprite):
        """Draw a unit using its sprite image."""
        needs_effect = (not unit.can_move and not unit.can_attack) or unit.is_paralyzed()

        if needs_effect:
            # Only copy when we actually need to tint
            display_sprite = sprite.copy()

            if not unit.can_move and not unit.can_attack:
                display_sprite.blit(
                    self._get_overlay(sprite.get_size(), theme.STATUS_DISABLED_TINT),
                    (0, 0),
                    special_flags=pygame.BLEND_MULT,
                )

            if unit.is_paralyzed():
                display_sprite.blit(
                    self._get_overlay(sprite.get_size(), theme.STATUS_PARALYSIS_TINT),
                    (0, 0),
                    special_flags=pygame.BLEND_MULT,
                )
        else:
            display_sprite = sprite

        # Draw player-colored border around sprite
        player_color = PLAYER_COLORS.get(unit.player, (255, 255, 255))
        border_rect = pygame.Rect(unit.x * TILE_SIZE + 1, unit.y * TILE_SIZE + 1, TILE_SIZE - 2, TILE_SIZE - 2)
        pygame.draw.rect(self.screen, player_color, border_rect, 2)

        # Stand the sprite on the tile: its bottom edge (the feet line) on the
        # tile's bottom edge, centred horizontally. A sprite larger than the
        # tile overflows it upward and sideways.
        sprite_rect = display_sprite.get_rect(midbottom=(unit.x * TILE_SIZE + TILE_SIZE // 2, (unit.y + 1) * TILE_SIZE))
        self.screen.blit(display_sprite, sprite_rect)

    def _get_overlay(self, size, color):
        """Return a cached solid-colour overlay surface for blend effects."""
        key = (size, color)
        surface = self._overlay_cache.get(key)
        if surface is None:
            surface = pygame.Surface(size)
            surface.fill(color)
            self._overlay_cache[key] = surface
        return surface

    def _draw_unit_letter(self, unit):
        """Draw a unit using its letter representation (fallback)."""
        color = UNIT_ASSETS[unit.type]["color"]

        # Gray out if can't act
        if not unit.can_move and not unit.can_attack:
            color = tuple(c // 2 for c in color)

        # Purple tint for paralyzed
        if unit.is_paralyzed():
            color = tuple(min(int(c * 0.6 + 128 * 0.4), 255) for c in color)

        # Letter + outline surfaces are cached per (type, color); rendering
        # them fresh every frame for every unit is expensive.
        key = (unit.type, color)
        cached = self._letter_cache.get(key)
        if cached is None:
            font = get_font(28)
            cached = (font.render(unit.type, True, color), font.render(unit.type, True, (0, 0, 0)))
            self._letter_cache[key] = cached
        text, outline_text = cached

        text_rect = text.get_rect(center=(unit.x * TILE_SIZE + TILE_SIZE // 2, unit.y * TILE_SIZE + TILE_SIZE // 2))

        # Black outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            outline_rect = text_rect.copy()
            outline_rect.x += dx
            outline_rect.y += dy
            self.screen.blit(outline_text, outline_rect)

        self.screen.blit(text, text_rect)

    def _draw_unit_health_bar(self, unit):
        """Draw health bar for a unit."""
        margin = theme.HEALTH_BAR_MARGIN
        bar_width = TILE_SIZE - 2 * margin
        bar_height = theme.HEALTH_BAR_UNIT_HEIGHT
        bar_x = unit.x * TILE_SIZE + margin
        bar_y = unit.y * TILE_SIZE + TILE_SIZE - bar_height - margin

        # Background
        pygame.draw.rect(self.screen, theme.HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))

        # Foreground
        health_percentage = unit.health / unit.max_health
        current_bar_width = int(bar_width * health_percentage)

        if health_percentage > 0.5:
            health_color = theme.HEALTH_GOOD
        elif health_percentage > 0.25:
            health_color = theme.HEALTH_INJURED
        else:
            health_color = theme.HEALTH_CRITICAL

        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 1)

    def _cached_text(self, key, text, font, color):
        """Render text through a cache, re-rendering only when it changes."""
        cached = self._text_cache.get(key)
        if cached is not None and cached[0] == (text, color):
            return cached[1]
        surface = font.render(text, True, color)
        self._text_cache[key] = ((text, color), surface)
        return surface

    def _gold_text(self):
        """The current player's gold, written as the unit shop writes prices."""
        return f"{self.game_state.player_gold[self.game_state.current_player]}g"

    def _turn_text(self):
        """The turn number, out of the turn limit when there is one."""
        text = str(self.game_state.turn_number + 1)
        if self.game_state.max_turns:
            text += f" / {self.game_state.max_turns}"
        return text

    def _draw_ui(self):
        """Draw the HUD: the side panel in play, two labels in a replay."""
        # Headless capture (videos, notebooks) is the board alone.
        if self.headless:
            return
        if self.has_hud_panel:
            self._draw_hud_panel()
        else:
            self._draw_replay_hud()

    def _draw_player_card(self, rect, text_surface, player_color):
        """Draw ``text_surface`` on a dark card with a player-colour accent strip and border."""
        radius = theme.BORDER_RADIUS_SMALL
        pygame.draw.rect(self.screen, theme.HUD_CARD_BG, rect, border_radius=radius)
        accent = pygame.Rect(rect.x, rect.y, theme.HUD_ACCENT_WIDTH, rect.height)
        pygame.draw.rect(self.screen, player_color, accent, border_top_left_radius=radius, border_bottom_left_radius=radius)
        pygame.draw.rect(self.screen, player_color, rect, 2, border_radius=radius)
        self.screen.blit(text_surface, text_surface.get_rect(midleft=(accent.right + 8, rect.centery)))

    def _draw_hud_panel(self):
        """Draw the HUD panel to the right of the board."""
        panel = self.hud_rect
        pygame.draw.rect(self.screen, theme.HUD_PANEL_BG, panel)
        pygame.draw.line(self.screen, theme.HUD_PANEL_BORDER, panel.topleft, (panel.left, panel.bottom - 1), 2)

        player = self.game_state.current_player
        name = self._cached_text("hud_player", f"{self._hud_player_label} {player}", self._hud_title_font, theme.TEXT)
        self._draw_player_card(self._hud_player_card, name, PLAYER_COLORS.get(player, theme.TEXT))

        for key, caption, text, color, (x, y) in (
            ("hud_gold", self._hud_gold_caption, self._gold_text(), theme.HUD_GOLD_TEXT, self._hud_gold_pos),
            ("hud_turn", self._hud_turn_caption, self._turn_text(), theme.TEXT, self._hud_turn_pos),
        ):
            self.screen.blit(caption, (x, y))
            self.screen.blit(self._cached_text(key, text, self._hud_font, color), (x, y + caption.get_height()))

        if self.game_state.fog_of_war:
            fow_rect = self._fow_label.get_rect(topleft=(self._hud_fow_pos[0] + 4, self._hud_fow_pos[1] + 2))
            fow_bg = fow_rect.inflate(8, 4)
            pygame.draw.rect(self.screen, theme.HUD_FOW_BG, fow_bg, border_radius=theme.BORDER_RADIUS_SMALL)
            pygame.draw.rect(self.screen, theme.HUD_FOW_BORDER, fow_bg, width=1, border_radius=theme.BORDER_RADIUS_SMALL)
            self.screen.blit(self._fow_label, fow_rect)

        mouse_pos = pygame.mouse.get_pos()
        for button, label, color, hover_color, border_color in (
            (self.end_turn_button, self._end_turn_label, theme.BTN_END_TURN, theme.BTN_END_TURN_HOVER, theme.TEXT),
            (self.resign_button, self._resign_label, theme.BTN_RESIGN, theme.BTN_RESIGN_HOVER, theme.BTN_RESIGN_BORDER),
        ):
            fill = hover_color if button.collidepoint(mouse_pos) else color
            pygame.draw.rect(self.screen, fill, button, border_radius=theme.BORDER_RADIUS)
            pygame.draw.rect(self.screen, border_color, button, 2, border_radius=theme.BORDER_RADIUS)
            self.screen.blit(label, label.get_rect(center=button.center))

    def _draw_replay_hud(self):
        """Draw the gold and turn labels over the board's top-left corner.

        A replay keeps a board-sized window (the replay player lays out its
        own controls) and takes no clicks on the board, so these stay on it.
        """
        player = self.game_state.current_player
        gold_text = f"{self._hud_player_label} {player} {self._hud_gold_label}: {self._gold_text()}"
        gold_surface = self._cached_text("gold", gold_text, self._hud_font, theme.HUD_GOLD_TEXT)
        card = pygame.Rect(5, 5, theme.HUD_ACCENT_WIDTH + gold_surface.get_width() + 16, gold_surface.get_height() + 6)
        self._draw_player_card(card, gold_surface, PLAYER_COLORS.get(player, theme.TEXT))

        turn_text = f"{self._hud_turn_label}: {self._turn_text()}"
        turn_surface = self._cached_text("turn", turn_text, self._hud_font, theme.TEXT)
        turn_rect = turn_surface.get_rect(topleft=(10, card.bottom + 5))
        turn_bg_rect = turn_rect.inflate(10, 5)

        pygame.draw.rect(self.screen, theme.HUD_TURN_BG, turn_bg_rect)
        pygame.draw.rect(self.screen, theme.HUD_TURN_BORDER, turn_bg_rect, 2)
        self.screen.blit(turn_surface, turn_rect)

    def _overlay_alpha(self, kind, signature, base_alpha):
        """Fade an overlay in over OVERLAY_FADE_MS when its target changes.

        Args:
            kind: Overlay identifier ('move', 'target', 'attack_range').
            signature: Hashable describing what the overlay currently shows;
                a change restarts the fade.
            base_alpha: Alpha once fully faded in.

        Returns:
            Alpha (0..base_alpha) to use this frame.
        """
        now = pygame.time.get_ticks()
        previous = self._overlay_anim.get(kind)
        if previous is None or previous[0] != signature:
            self._overlay_anim[kind] = (signature, now)
            return 0
        progress = (now - previous[1]) / theme.OVERLAY_FADE_MS
        if progress >= 1.0:
            return base_alpha
        return int(base_alpha * progress)

    def draw_selected_unit_highlight(self, unit):
        """Draw a pulsing highlight around the currently selected unit."""
        rect = pygame.Rect(unit.x * TILE_SIZE, unit.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        color = _lerp_color(theme.OVERLAY_MOVEMENT_BORDER, theme.TEXT, _pulse(theme.SELECTION_PULSE_MS))
        pygame.draw.rect(self.screen, color, rect, 3)

    def draw_movement_overlay(self, unit):
        """Draw movement range overlay for a unit (fades in on selection)."""
        if not unit.can_move:
            return

        # The engine's search: terrain move costs, and the units the player
        # knows of (all of them without fog of war), so a hidden enemy neither
        # punches a hole in the overlay, revealing it, nor hides a tile the
        # engine allows.
        movement_positions = self.game_state.get_reachable_positions(unit)

        if movement_positions:
            alpha = self._overlay_alpha("move", (id(unit), unit.x, unit.y), theme.OVERLAY_MOVEMENT_ALPHA)
            self._move_overlay.set_alpha(alpha)
            for x, y in movement_positions:
                self.screen.blit(self._move_overlay, (x * TILE_SIZE, y * TILE_SIZE))
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, theme.OVERLAY_MOVEMENT_BORDER, rect, 1)

    def draw_target_overlay(self, valid_targets):
        """
        Draw target selection overlay highlighting valid target positions.

        Args:
            valid_targets: List of unit objects that are valid targets
        """
        signature = tuple(sorted((t.x, t.y) for t in valid_targets))
        alpha = self._overlay_alpha("target", signature, theme.OVERLAY_TARGET_ALPHA)
        self._target_overlay.set_alpha(alpha)
        for target in valid_targets:
            self.screen.blit(self._target_overlay, (target.x * TILE_SIZE, target.y * TILE_SIZE))
            rect = pygame.Rect(target.x * TILE_SIZE, target.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.screen, theme.OVERLAY_TARGET_BORDER, rect, 3)

    def draw_attack_range_overlay(self, positions):
        """
        Draw attack range preview overlay highlighting attackable positions.

        Args:
            positions: List of (x, y) tuples for positions that can be attacked
        """
        alpha = self._overlay_alpha("attack_range", tuple(sorted(positions)), theme.OVERLAY_ATTACK_RANGE_ALPHA)
        self._attack_range_overlay.set_alpha(alpha)
        for x, y in positions:
            self.screen.blit(self._attack_range_overlay, (x * TILE_SIZE, y * TILE_SIZE))
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(self.screen, theme.OVERLAY_ATTACK_RANGE_BORDER, rect, 2)

    def draw_unit_tooltip(self, mouse_pos):
        """
        Draw a tooltip showing unit stats when hovering over a unit.

        Args:
            mouse_pos: Tuple of (x, y) mouse position
        """
        # Convert mouse position to grid coordinates
        grid_x = mouse_pos[0] // TILE_SIZE
        grid_y = mouse_pos[1] // TILE_SIZE

        # Check bounds
        if not (0 <= grid_x < self.game_state.grid.width and 0 <= grid_y < self.game_state.grid.height):
            return

        # Find unit at position
        unit = self.game_state.get_unit_at_position(grid_x, grid_y)
        if not unit:
            return

        # Under fog of war, don't reveal stats of enemy units the viewing
        # player can't currently see (mirrors the visibility rule used when
        # drawing units).
        fow_player = self._get_fow_player()
        if fow_player is not None and unit.player != fow_player:
            if self._get_visibility_state(unit.x, unit.y, fow_player) != VISIBLE:
                return

        # Get unit data
        unit_data = UNIT_DATA.get(unit.type, {})
        unit_name = unit_data.get("name", unit.type)

        # Build tooltip lines
        # Handle attack display - can be int or dict (for ranged units like Mage/Sorcerer)
        attack_data = unit.attack_data
        if isinstance(attack_data, dict):
            attack_str = f"{attack_data.get('adjacent', 0)}/{attack_data.get('range', 0)}"
        else:
            attack_str = str(attack_data)

        lines = [
            f"{unit_name} (P{unit.player})",
            f"HP: {unit.health}/{unit.max_health}",
            f"ATK: {attack_str}  DEF: {unit.defence}",
            f"MOV: {unit.movement_range}",
        ]

        # Add status indicators
        status_parts = []
        if unit.can_move:
            status_parts.append("Can Move")
        if unit.can_attack:
            status_parts.append("Can Act")
        if unit.is_paralyzed():
            status_parts.append(f"Paralyzed ({unit.paralyzed_turns})")

        if status_parts:
            lines.append(" | ".join(status_parts))

        # Render the lines through a cache: re-render only when the hovered
        # unit (or its stats) changes, not on every mouse-motion frame.
        player_color = PLAYER_COLORS.get(unit.player, theme.TEXT)
        font = get_font(20)
        cache_key = (tuple(lines), unit.player)
        if self._tooltip_cache is not None and self._tooltip_cache[0] == cache_key:
            line_surfaces = self._tooltip_cache[1]
        else:
            line_surfaces = [
                font.render(line, True, player_color if i == 0 else theme.TEXT_SECONDARY) for i, line in enumerate(lines)
            ]
            self._tooltip_cache = (cache_key, line_surfaces)

        padding = 8
        line_height = font.get_linesize()
        max_width = max(surface.get_width() for surface in line_surfaces)

        tooltip_width = max_width + 2 * padding
        tooltip_height = len(lines) * line_height + 2 * padding

        # Position tooltip near mouse but avoid going off-screen
        tooltip_x = mouse_pos[0] + 15
        tooltip_y = mouse_pos[1] + 15

        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        if tooltip_x + tooltip_width > screen_width:
            tooltip_x = mouse_pos[0] - tooltip_width - 5
        if tooltip_y + tooltip_height > screen_height:
            tooltip_y = mouse_pos[1] - tooltip_height - 5

        # Ensure tooltip stays on screen
        tooltip_x = max(5, tooltip_x)
        tooltip_y = max(5, tooltip_y)

        # Draw tooltip background
        tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, theme.TOOLTIP_BG, tooltip_rect, border_radius=6)

        # Draw player-colored border
        pygame.draw.rect(self.screen, player_color, tooltip_rect, width=2, border_radius=6)

        # Draw text lines
        for i, text_surface in enumerate(line_surfaces):
            text_y = tooltip_y + padding + i * line_height
            self.screen.blit(text_surface, (tooltip_x + padding, text_y))

    def get_rgb_array(self):
        """Get the current screen as RGB array."""
        # array3d copies and does not lock the surface; pixels3d returns a
        # locked view that can silently block subsequent blits.
        return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def set_unit_animation_state(self, unit, state):
        """
        Set the animation state for a unit.

        Args:
            unit: Unit object
            state: Animation state ('idle', 'move_down', 'move_up',
                   'move_left', 'move_right')
        """
        if self.animator:
            self.animator.set_unit_state(unit, state)

    def update_unit_animation_from_movement(self, unit, from_pos, to_pos):
        """
        Update unit animation based on movement direction.

        Args:
            unit: Unit object
            from_pos: Tuple (x, y) of starting position
            to_pos: Tuple (x, y) of ending position
        """
        if self.animator:
            self.animator.update_unit_state_from_movement(unit, from_pos, to_pos)

    def set_unit_idle(self, unit):
        """Set a unit to idle animation state."""
        if self.animator:
            self.animator.set_idle(unit)

    def queue_movement_path_animation(self, unit, path):
        """
        Queue a multi-step movement path for animation transitions.

        Each path segment triggers the correct walking direction animation
        (left, right, up, down). After the full path plays through, the
        unit returns to idle.

        Args:
            unit: Unit object
            path: List of (x, y) positions including start position
        """
        if self.animator:
            self.animator.queue_movement_path(unit, path)

    def cleanup_unit_animation(self, unit):
        """Clean up animation data for a removed unit."""
        if self.animator:
            self.animator.cleanup_unit(unit)

    def set_viewing_player(self, player):
        """
        Set which player's perspective to render for fog of war.

        Args:
            player: Player number (1 or 2), or None for current player's view
        """
        self.viewing_player = player

    def toggle_fow_perspective(self):
        """Toggle between player perspectives and omniscient view (for replays)."""
        if not self.game_state.fog_of_war:
            return

        if self.viewing_player is None:
            self.viewing_player = 1
        elif self.viewing_player < self.game_state.num_players:
            self.viewing_player += 1
        else:
            self.viewing_player = None  # Omniscient view

    def close(self):
        """Close the renderer."""
        pygame.quit()
