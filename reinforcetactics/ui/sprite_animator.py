"""
Sprite animation system for unit animations.

Handles loading sprite sheets and managing frame-by-frame animations
including directional walking, smooth transitions between animation
states during multi-step movement paths, and per-team palette swaps.
"""
import os
from collections import deque
import pygame
from reinforcetactics.constants import (
    TILE_SIZE, UNIT_DATA, ANIMATION_CONFIG,
    BASE_SPRITE_COLORS, TEAM_PALETTES,
)


class SpriteAnimator:
    """
    Manages sprite sheet animations for units.

    Sprite sheets use a grid of fixed-size frames (default 32x32).
    Each animation state is defined by a list of (row, col) frame
    coordinates that may span multiple rows.

    Directional mirroring (e.g. move_right from move_left) is handled
    automatically via the ``mirror_states`` config.

    Team colours are applied at load time by replacing a fixed set of
    base blue pixels with each team's palette, so ``get_frame`` has
    zero per-frame overhead for colouring.

    Movement path animation is supported: when a unit follows a
    multi-tile path, segments are queued so the walking direction
    updates correctly at each waypoint.
    """

    def __init__(self, sprites_path):
        """
        Initialize the sprite animator.

        Args:
            sprites_path: Base path to sprite sheet directory
        """
        self.sprites_path = sprites_path

        # Base (uncoloured) frames: unit_type -> {state -> [frames]}
        self.sprite_sheets = {}

        # Team-coloured frames: (unit_type, player) -> {state -> [frames]}
        self.team_sheets = {}

        self.animation_timers = {}  # unit_id -> {current_time, current_frame}
        self.unit_states = {}       # unit_id -> current animation state

        # Movement path queues for multi-step animation transitions
        self.movement_queues = {}   # unit_id -> deque of state strings

        # Frame dimensions (can be overridden per unit type)
        self.frame_width = ANIMATION_CONFIG.get('frame_width', 32)
        self.frame_height = ANIMATION_CONFIG.get('frame_height', 32)

        self._load_all_sprite_sheets()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all_sprite_sheets(self):
        """Load sprite sheets for all unit types."""
        if not self.sprites_path:
            return

        for unit_type, unit_data in UNIT_DATA.items():
            animation_path = unit_data.get('animation_path', '')
            if animation_path:
                self._load_sprite_sheet(unit_type, animation_path)

    def _load_sprite_sheet(self, unit_type, animation_path):
        """
        Load a sprite sheet for a unit type and generate team colour
        variants.

        Args:
            unit_type: Single character unit type (e.g., 'W' for Warrior)
            animation_path: Base name for the sprite sheet file
        """
        possible_names = [
            f"{animation_path}_sheet.png",
            f"{animation_path}_spritesheet.png",
            f"{animation_path}.png",
        ]

        sheet_surface = None
        for filename in possible_names:
            full_path = os.path.join(self.sprites_path, filename)
            try:
                sheet_surface = pygame.image.load(full_path).convert_alpha()
                break
            except (pygame.error, FileNotFoundError):
                continue

        if sheet_surface is None:
            return

        base_frames = self._parse_sprite_sheet(sheet_surface, unit_type)
        self.sprite_sheets[unit_type] = base_frames

        # Generate team-coloured variants
        self._generate_team_variants(unit_type, base_frames)

    def _parse_sprite_sheet(self, sheet_surface, unit_type):
        """
        Parse a sprite sheet into individual animation frames using
        the coordinate-based frame_map from ANIMATION_CONFIG.

        Each source frame (``frame_width`` x ``frame_height``, e.g. 64x64)
        is centre-cropped to ``crop_width`` x ``crop_height`` (e.g. 32x32)
        before being scaled to the display size.

        Args:
            sheet_surface: Pygame surface of the loaded sprite sheet
            unit_type: Unit type for getting per-unit config overrides

        Returns:
            Dictionary mapping animation states to lists of Pygame surfaces
        """
        frames = {}

        unit_cfg = ANIMATION_CONFIG.get('units', {}).get(unit_type, {})
        fw = unit_cfg.get('frame_width', self.frame_width)
        fh = unit_cfg.get('frame_height', self.frame_height)

        # Crop dimensions (defaults to full frame if not configured)
        cw = unit_cfg.get('crop_width',
                          ANIMATION_CONFIG.get('crop_width', fw))
        ch = unit_cfg.get('crop_height',
                          ANIMATION_CONFIG.get('crop_height', fh))

        sprite_size = TILE_SIZE - 4

        frame_map = ANIMATION_CONFIG.get('frame_map', {})

        for state_name, coords in frame_map.items():
            state_frames = []
            for row, col in coords:
                rect = pygame.Rect(col * fw, row * fh, fw, fh)
                if (rect.right <= sheet_surface.get_width() and
                        rect.bottom <= sheet_surface.get_height()):
                    frame = sheet_surface.subsurface(rect).copy()
                    # Centre-crop to the target crop size
                    if cw < fw or ch < fh:
                        cx = (fw - cw) // 2
                        cy = (fh - ch) // 2
                        frame = frame.subsurface(
                            pygame.Rect(cx, cy, cw, ch)
                        ).copy()
                    frame = pygame.transform.scale(frame, (sprite_size, sprite_size))
                    state_frames.append(frame)

            if state_frames:
                frames[state_name] = state_frames

        # Generate mirrored states (e.g. move_right from move_left)
        mirror_states = ANIMATION_CONFIG.get('mirror_states', {})
        for target_state, source_state in mirror_states.items():
            if source_state in frames and target_state not in frames:
                frames[target_state] = [
                    pygame.transform.flip(f, True, False)
                    for f in frames[source_state]
                ]

        return frames

    # ------------------------------------------------------------------
    # Team colour palette swap
    # ------------------------------------------------------------------

    def _generate_team_variants(self, unit_type, base_frames):
        """
        Create team-coloured copies of the base frames for every
        configured team palette.  Recolouring is done once at load
        time so ``get_frame`` has no per-frame cost.

        Args:
            unit_type: Unit type key (e.g. 'W')
            base_frames: {state -> [pygame.Surface]} dict of base frames
        """
        if not BASE_SPRITE_COLORS:
            return

        for player, palette in TEAM_PALETTES.items():
            if palette is None:
                # This team uses the base colours as-is
                self.team_sheets[(unit_type, player)] = base_frames
                continue

            recoloured = {}
            for state, frames in base_frames.items():
                recoloured[state] = [
                    self._recolor_frame(f, BASE_SPRITE_COLORS, palette)
                    for f in frames
                ]
            self.team_sheets[(unit_type, player)] = recoloured

    @staticmethod
    def _recolor_frame(frame, base_colors, team_colors):
        """
        Replace base palette colours with team colours in a single frame.

        Uses ``pygame.PixelArray.replace`` for efficient exact-match
        colour swapping (pixel art friendly).

        Args:
            frame: Source pygame.Surface (will not be mutated)
            base_colors: List of (R, G, B) colours to find
            team_colors: List of (R, G, B) replacement colours

        Returns:
            New pygame.Surface with swapped colours
        """
        recoloured = frame.copy()
        pxa = pygame.PixelArray(recoloured)
        for src, dst in zip(base_colors, team_colors):
            pxa.replace(src, dst, 0.01)
        del pxa  # unlock surface
        return recoloured

    # ------------------------------------------------------------------
    # Frame retrieval
    # ------------------------------------------------------------------

    def get_frame(self, unit, delta_time=None):
        """
        Get the current animation frame for a unit.

        Automatically selects team-coloured frames based on
        ``unit.player`` if available, otherwise falls back to the
        base (uncoloured) sprite sheet.

        Handles movement queue advancement: when the current movement
        segment finishes its allotted frames, the next queued segment's
        direction is activated automatically.

        Args:
            unit: Unit object with ``type`` and ``player`` attributes
            delta_time: Time since last frame in seconds

        Returns:
            Pygame surface of the current frame, or None if unavailable
        """
        unit_type = unit.type

        if unit_type not in self.sprite_sheets:
            return None

        # Prefer team-coloured frames, fall back to base
        player = getattr(unit, 'player', None)
        unit_frames = self.team_sheets.get((unit_type, player))
        if unit_frames is None:
            unit_frames = self.sprite_sheets.get(unit_type)
        if not unit_frames:
            return None

        unit_id = id(unit)
        state = self.unit_states.get(unit_id, 'idle')

        # Fallback chain
        if state not in unit_frames:
            state = 'idle'
            if state not in unit_frames:
                first_state = next(iter(unit_frames.keys()))
                return unit_frames[first_state][0]

        anim_frames = unit_frames[state]
        if not anim_frames:
            return None

        state_config = ANIMATION_CONFIG.get('states', {}).get(state, {})
        frame_duration = state_config.get('speed', 0.15)

        # Initialise timer
        if unit_id not in self.animation_timers:
            self.animation_timers[unit_id] = {
                'current_time': 0.0,
                'current_frame': 0,
            }

        timer = self.animation_timers[unit_id]

        if delta_time is not None:
            timer['current_time'] += delta_time
            if timer['current_time'] >= frame_duration:
                timer['current_time'] = 0.0
                next_frame = (timer['current_frame'] + 1) % len(anim_frames)
                timer['current_frame'] = next_frame

                # If we looped back to 0, check movement queue
                if next_frame == 0:
                    self._advance_movement_queue(unit)

        return anim_frames[timer['current_frame']]

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_unit_state(self, unit, state):
        """
        Set the animation state for a unit.

        Resets the frame timer when the state actually changes.

        Args:
            unit: Unit object
            state: Animation state name ('idle', 'move_down', 'move_up',
                   'move_left', 'move_right')
        """
        unit_id = id(unit)
        old_state = self.unit_states.get(unit_id)

        if old_state != state:
            self.unit_states[unit_id] = state
            if unit_id in self.animation_timers:
                self.animation_timers[unit_id] = {
                    'current_time': 0.0,
                    'current_frame': 0,
                }

    def update_unit_state_from_movement(self, unit, from_pos, to_pos):
        """
        Update unit animation state based on movement direction.

        Args:
            unit: Unit object
            from_pos: Tuple (x, y) of starting position
            to_pos: Tuple (x, y) of ending position
        """
        state = self._direction_state(from_pos, to_pos)
        self.set_unit_state(unit, state)

    def set_idle(self, unit):
        """Set a unit to idle animation state and clear any movement queue."""
        unit_id = id(unit)
        self.movement_queues.pop(unit_id, None)
        self.set_unit_state(unit, 'idle')

    # ------------------------------------------------------------------
    # Movement path animation
    # ------------------------------------------------------------------

    def queue_movement_path(self, unit, path):
        """
        Queue a multi-step movement path for smooth animation transitions.

        Each segment of the path produces a walking direction that plays
        for one full animation cycle before advancing to the next segment.
        After all segments complete, the unit returns to idle.

        This is intended for UI-mode animated movement where the unit
        visually walks along its path.

        Args:
            unit: Unit object
            path: List of (x, y) positions the unit travels through,
                  including the starting position.  Minimum 2 positions.
        """
        if len(path) < 2:
            return

        unit_id = id(unit)
        queue = deque()

        for i in range(len(path) - 1):
            state = self._direction_state(path[i], path[i + 1])
            queue.append(state)

        self.movement_queues[unit_id] = queue

        # Start the first segment immediately
        first_state = queue.popleft()
        self.set_unit_state(unit, first_state)

    def _advance_movement_queue(self, unit):
        """
        Advance to the next segment in the movement queue.

        Called internally when the current animation cycle loops.
        If no more segments remain, sets the unit back to idle.
        """
        unit_id = id(unit)
        queue = self.movement_queues.get(unit_id)
        if not queue:
            if unit_id in self.movement_queues:
                del self.movement_queues[unit_id]
                self.set_unit_state(unit, 'idle')
            return

        next_state = queue.popleft()
        self.set_unit_state(unit, next_state)

    # ------------------------------------------------------------------
    # Queries & cleanup
    # ------------------------------------------------------------------

    def has_animations(self, unit_type):
        """Check if animations are loaded for a unit type."""
        return unit_type in self.sprite_sheets and bool(self.sprite_sheets[unit_type])

    def cleanup_unit(self, unit):
        """Clean up all animation data for a removed unit."""
        unit_id = id(unit)
        self.animation_timers.pop(unit_id, None)
        self.unit_states.pop(unit_id, None)
        self.movement_queues.pop(unit_id, None)

    def reload(self, sprites_path=None):
        """
        Reload all sprite sheets, clearing existing animation state.

        Args:
            sprites_path: Optional new base path for sprites
        """
        if sprites_path is not None:
            self.sprites_path = sprites_path

        self.sprite_sheets.clear()
        self.team_sheets.clear()
        self.animation_timers.clear()
        self.unit_states.clear()
        self.movement_queues.clear()

        self._load_all_sprite_sheets()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _direction_state(from_pos, to_pos):
        """
        Determine the walking animation state for a movement vector.

        Args:
            from_pos: (x, y) origin
            to_pos: (x, y) destination

        Returns:
            Animation state string
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if abs(dx) > abs(dy):
            return 'move_right' if dx > 0 else 'move_left'
        elif dy != 0:
            return 'move_down' if dy > 0 else 'move_up'
        return 'idle'
