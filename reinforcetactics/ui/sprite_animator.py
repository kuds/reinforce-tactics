"""
Sprite animation system for unit animations.

Handles loading sprite sheets and managing frame-by-frame animations.
"""
import os
import pygame
from reinforcetactics.constants import TILE_SIZE, UNIT_DATA, ANIMATION_CONFIG


class SpriteAnimator:
    """
    Manages sprite sheet animations for units.

    Sprite sheets are organized with animation states in rows:
    - Row 0: Idle animation
    - Row 1: Move down
    - Row 2: Move up
    - Row 3: Move left
    - Row 4: Move right (optional - can be mirrored from left)

    Each row contains frames for that animation state.
    """

    def __init__(self, sprites_path):
        """
        Initialize the sprite animator.

        Args:
            sprites_path: Base path to sprite sheet directory
        """
        self.sprites_path = sprites_path
        self.sprite_sheets = {}  # unit_type -> {animation_state -> [frames]}
        self.animation_timers = {}  # unit_id -> {current_time, current_frame}
        self.unit_states = {}  # unit_id -> current animation state

        # Frame dimensions (can be overridden per unit type)
        self.frame_width = ANIMATION_CONFIG.get('frame_width', 32)
        self.frame_height = ANIMATION_CONFIG.get('frame_height', 32)

        # Load sprite sheets for all unit types
        self._load_all_sprite_sheets()

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
        Load a sprite sheet for a unit type.

        Args:
            unit_type: Single character unit type (e.g., 'W' for Warrior)
            animation_path: Base name for the sprite sheet file
        """
        # Try different file naming conventions
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

        # Parse the sprite sheet into animation frames
        self.sprite_sheets[unit_type] = self._parse_sprite_sheet(
            sheet_surface, unit_type
        )

    def _parse_sprite_sheet(self, sheet_surface, unit_type):
        """
        Parse a sprite sheet into individual animation frames.

        Args:
            sheet_surface: Pygame surface of the loaded sprite sheet
            unit_type: Unit type for getting animation config

        Returns:
            Dictionary mapping animation states to lists of frames
        """
        frames = {}
        sheet_width = sheet_surface.get_width()
        sheet_height = sheet_surface.get_height()

        # Get unit-specific config or use defaults
        unit_anim_config = ANIMATION_CONFIG.get('units', {}).get(unit_type, {})
        frame_width = unit_anim_config.get('frame_width', self.frame_width)
        frame_height = unit_anim_config.get('frame_height', self.frame_height)

        # Calculate number of frames per row
        frames_per_row = sheet_width // frame_width
        num_rows = sheet_height // frame_height

        # Animation state mapping (row index -> state name)
        state_mapping = ANIMATION_CONFIG.get('state_rows', {
            0: 'idle',
            1: 'move_down',
            2: 'move_up',
            3: 'move_left',
            4: 'move_right',
        })

        # Target sprite size for rendering (slightly smaller than tile)
        sprite_size = TILE_SIZE - 4

        # Extract frames for each row
        for row in range(num_rows):
            state_name = state_mapping.get(row)
            if state_name is None:
                continue

            state_frames = []
            # Get frame count for this state (default to all frames in row)
            state_config = ANIMATION_CONFIG.get('states', {}).get(state_name, {})
            frame_count = state_config.get('frames', frames_per_row)

            for col in range(min(frame_count, frames_per_row)):
                rect = pygame.Rect(
                    col * frame_width,
                    row * frame_height,
                    frame_width,
                    frame_height
                )
                frame = sheet_surface.subsurface(rect).copy()
                # Scale frame to target size
                frame = pygame.transform.scale(frame, (sprite_size, sprite_size))
                state_frames.append(frame)

            if state_frames:
                frames[state_name] = state_frames

        # Generate mirrored right frames from left if not present
        if 'move_left' in frames and 'move_right' not in frames:
            frames['move_right'] = [
                pygame.transform.flip(frame, True, False)
                for frame in frames['move_left']
            ]

        return frames

    def get_frame(self, unit, delta_time=None):
        """
        Get the current animation frame for a unit.

        Args:
            unit: Unit object with id, type, and position
            delta_time: Time since last frame in seconds (for animation timing)

        Returns:
            Pygame surface of the current frame, or None if no animation available
        """
        unit_type = unit.type

        # Check if we have animations for this unit type
        if unit_type not in self.sprite_sheets:
            return None

        unit_frames = self.sprite_sheets[unit_type]
        if not unit_frames:
            return None

        # Get current animation state for this unit
        unit_id = id(unit)
        state = self.unit_states.get(unit_id, 'idle')

        # Fall back to idle if state not available
        if state not in unit_frames:
            state = 'idle'
            if state not in unit_frames:
                # Return first available frame
                first_state = next(iter(unit_frames.keys()))
                return unit_frames[first_state][0]

        frames = unit_frames[state]
        if not frames:
            return None

        # Get animation timing
        state_config = ANIMATION_CONFIG.get('states', {}).get(state, {})
        frame_duration = state_config.get('speed', 0.15)  # seconds per frame

        # Initialize or update animation timer
        if unit_id not in self.animation_timers:
            self.animation_timers[unit_id] = {
                'current_time': 0.0,
                'current_frame': 0
            }

        timer = self.animation_timers[unit_id]

        # Update animation if delta_time provided
        if delta_time is not None:
            timer['current_time'] += delta_time
            if timer['current_time'] >= frame_duration:
                timer['current_time'] = 0.0
                timer['current_frame'] = (timer['current_frame'] + 1) % len(frames)

        return frames[timer['current_frame']]

    def set_unit_state(self, unit, state):
        """
        Set the animation state for a unit.

        Args:
            unit: Unit object
            state: Animation state name ('idle', 'move_down', 'move_up',
                   'move_left', 'move_right')
        """
        unit_id = id(unit)
        old_state = self.unit_states.get(unit_id)

        if old_state != state:
            self.unit_states[unit_id] = state
            # Reset animation timer when state changes
            if unit_id in self.animation_timers:
                self.animation_timers[unit_id] = {
                    'current_time': 0.0,
                    'current_frame': 0
                }

    def update_unit_state_from_movement(self, unit, from_pos, to_pos):
        """
        Update unit animation state based on movement direction.

        Args:
            unit: Unit object
            from_pos: Tuple (x, y) of starting position
            to_pos: Tuple (x, y) of ending position
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        # Determine primary direction (prioritize larger movement)
        if abs(dx) > abs(dy):
            state = 'move_right' if dx > 0 else 'move_left'
        elif dy != 0:
            state = 'move_down' if dy > 0 else 'move_up'
        else:
            state = 'idle'

        self.set_unit_state(unit, state)

    def set_idle(self, unit):
        """Set a unit to idle animation state."""
        self.set_unit_state(unit, 'idle')

    def has_animations(self, unit_type):
        """
        Check if animations are loaded for a unit type.

        Args:
            unit_type: Single character unit type

        Returns:
            True if animations are available
        """
        return unit_type in self.sprite_sheets and bool(self.sprite_sheets[unit_type])

    def cleanup_unit(self, unit):
        """
        Clean up animation data for a removed unit.

        Args:
            unit: Unit object being removed
        """
        unit_id = id(unit)
        self.animation_timers.pop(unit_id, None)
        self.unit_states.pop(unit_id, None)

    def reload(self, sprites_path=None):
        """
        Reload all sprite sheets.

        Args:
            sprites_path: Optional new path for sprites
        """
        if sprites_path is not None:
            self.sprites_path = sprites_path

        self.sprite_sheets.clear()
        self.animation_timers.clear()
        self.unit_states.clear()

        self._load_all_sprite_sheets()
