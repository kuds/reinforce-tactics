"""
Cross-platform UI icons module.

Generates simple geometric icons programmatically using pygame drawing,
ensuring consistent appearance across Windows, macOS, and Linux.
"""
from typing import Tuple, Dict
import pygame


# Icon cache to avoid regenerating icons
_icon_cache: Dict[str, pygame.Surface] = {}


def _ensure_pygame_init() -> None:
    """Ensure pygame is initialized for surface creation."""
    if not pygame.get_init():
        pygame.init()


def get_arrow_up_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (200, 180, 100),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate an upward-pointing arrow icon.

    Args:
        size: Icon size in pixels (square)
        color: Arrow color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the arrow icon
    """
    cache_key = f"arrow_up_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    # Create surface with alpha channel for transparency
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Calculate triangle points for upward arrow
    padding = size // 6
    top_point = (size // 2, padding)
    bottom_left = (padding, size - padding)
    bottom_right = (size - padding, size - padding)

    # Draw filled triangle
    pygame.draw.polygon(surface, color, [top_point, bottom_left, bottom_right])

    _icon_cache[cache_key] = surface
    return surface


def get_arrow_down_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (200, 180, 100),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a downward-pointing arrow icon.

    Args:
        size: Icon size in pixels (square)
        color: Arrow color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the arrow icon
    """
    cache_key = f"arrow_down_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    # Create surface with alpha channel for transparency
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Calculate triangle points for downward arrow
    padding = size // 6
    bottom_point = (size // 2, size - padding)
    top_left = (padding, padding)
    top_right = (size - padding, padding)

    # Draw filled triangle
    pygame.draw.polygon(surface, color, [bottom_point, top_left, top_right])

    _icon_cache[cache_key] = surface
    return surface


def get_checkmark_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (50, 200, 50),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    thickness: int = 0
) -> pygame.Surface:
    """
    Generate a checkmark icon.

    Args:
        size: Icon size in pixels (square)
        color: Checkmark color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)
        thickness: Line thickness (0 = auto based on size)

    Returns:
        pygame.Surface with the checkmark icon
    """
    cache_key = f"checkmark_{size}_{color}_{bg_color}_{thickness}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    # Create surface with alpha channel for transparency
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Auto thickness based on size
    line_thickness = thickness if thickness > 0 else max(2, size // 6)

    # Calculate checkmark points
    padding = size // 5
    # Start point (left)
    start = (padding, size // 2)
    # Middle point (bottom of check)
    middle = (size * 2 // 5, size - padding)
    # End point (top right)
    end = (size - padding, padding)

    # Draw checkmark as two connected lines
    pygame.draw.line(surface, color, start, middle, line_thickness)
    pygame.draw.line(surface, color, middle, end, line_thickness)

    _icon_cache[cache_key] = surface
    return surface


def get_x_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (200, 50, 50),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    thickness: int = 0
) -> pygame.Surface:
    """
    Generate an X (cross) icon.

    Args:
        size: Icon size in pixels (square)
        color: X color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)
        thickness: Line thickness (0 = auto based on size)

    Returns:
        pygame.Surface with the X icon
    """
    cache_key = f"x_{size}_{color}_{bg_color}_{thickness}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    # Create surface with alpha channel for transparency
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Auto thickness based on size
    line_thickness = thickness if thickness > 0 else max(2, size // 6)

    # Calculate X points
    padding = size // 5
    top_left = (padding, padding)
    top_right = (size - padding, padding)
    bottom_left = (padding, size - padding)
    bottom_right = (size - padding, size - padding)

    # Draw X as two diagonal lines
    pygame.draw.line(surface, color, top_left, bottom_right, line_thickness)
    pygame.draw.line(surface, color, top_right, bottom_left, line_thickness)

    _icon_cache[cache_key] = surface
    return surface


def get_play_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a play icon (right-pointing triangle).

    Args:
        size: Icon size in pixels (square)
        color: Icon color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the play icon
    """
    cache_key = f"play_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Calculate triangle points for right-pointing arrow
    padding = size // 5
    left_top = (padding, padding)
    left_bottom = (padding, size - padding)
    right_point = (size - padding, size // 2)

    pygame.draw.polygon(surface, color, [left_top, left_bottom, right_point])

    _icon_cache[cache_key] = surface
    return surface


def get_pause_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a pause icon (two vertical bars).

    Args:
        size: Icon size in pixels (square)
        color: Icon color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the pause icon
    """
    cache_key = f"pause_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Calculate bar dimensions
    padding = size // 5
    bar_width = size // 5
    bar_height = size - 2 * padding
    gap = size // 5

    # Left bar
    left_bar_x = (size - 2 * bar_width - gap) // 2
    left_bar = pygame.Rect(left_bar_x, padding, bar_width, bar_height)
    pygame.draw.rect(surface, color, left_bar)

    # Right bar
    right_bar_x = left_bar_x + bar_width + gap
    right_bar = pygame.Rect(right_bar_x, padding, bar_width, bar_height)
    pygame.draw.rect(surface, color, right_bar)

    _icon_cache[cache_key] = surface
    return surface


def get_arrow_left_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a left-pointing arrow icon.

    Args:
        size: Icon size in pixels (square)
        color: Arrow color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the arrow icon
    """
    cache_key = f"arrow_left_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    # Calculate triangle points for left-pointing arrow
    padding = size // 5
    right_top = (size - padding, padding)
    right_bottom = (size - padding, size - padding)
    left_point = (padding, size // 2)

    pygame.draw.polygon(surface, color, [right_top, right_bottom, left_point])

    _icon_cache[cache_key] = surface
    return surface


def get_arrow_right_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a right-pointing arrow icon.

    Args:
        size: Icon size in pixels (square)
        color: Arrow color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the arrow icon
    """
    # Same as play icon
    return get_play_icon(size, color, bg_color)


def get_restart_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
    thickness: int = 0
) -> pygame.Surface:
    """
    Generate a restart/refresh icon (circular arrow).

    Args:
        size: Icon size in pixels (square)
        color: Icon color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)
        thickness: Line thickness (0 = auto based on size)

    Returns:
        pygame.Surface with the restart icon
    """
    cache_key = f"restart_{size}_{color}_{bg_color}_{thickness}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    import math

    line_thickness = thickness if thickness > 0 else max(2, size // 7)
    padding = size // 5
    center_x = size // 2
    center_y = size // 2
    radius = (size // 2) - padding

    # Draw a smooth circular arc using anti-aliased circles
    # Arc spans from ~60 degrees to ~300 degrees (leaving gap at top-right for arrow)
    start_angle = 60
    end_angle = 300
    num_segments = 32

    # Draw the arc as connected line segments for smoother appearance
    points = []
    for i in range(num_segments + 1):
        angle_deg = start_angle + (end_angle - start_angle) * i / num_segments
        angle_rad = math.radians(angle_deg)
        x = center_x + radius * math.cos(angle_rad)
        y = center_y - radius * math.sin(angle_rad)
        points.append((x, y))

    # Draw thick anti-aliased lines for the arc
    if len(points) >= 2:
        pygame.draw.lines(surface, color, False, points, line_thickness)

        # Draw circles at each point for smoother joints and ends
        half_thick = line_thickness // 2
        for point in points:
            pygame.draw.circle(surface, color, (int(point[0]), int(point[1])), half_thick)

    # Draw arrowhead at the start of the arc (pointing clockwise/down-right)
    arrow_angle_rad = math.radians(start_angle)
    arrow_tip_x = center_x + radius * math.cos(arrow_angle_rad)
    arrow_tip_y = center_y - radius * math.sin(arrow_angle_rad)

    # Arrow size proportional to icon
    arrow_len = size // 3
    arrow_width = size // 4

    # Arrow points downward-right (clockwise direction)
    # The arrow tip is at the arc start, pointing in the direction of rotation
    arrow_dir = start_angle - 90  # Perpendicular to radius, pointing clockwise
    arrow_dir_rad = math.radians(arrow_dir)

    # Calculate arrow base center (behind the tip)
    base_x = arrow_tip_x - arrow_len * 0.6 * math.cos(arrow_dir_rad)
    base_y = arrow_tip_y + arrow_len * 0.6 * math.sin(arrow_dir_rad)

    # Calculate perpendicular for arrow width
    perp_rad = arrow_dir_rad + math.pi / 2

    # Arrow wing points
    wing1_x = base_x + arrow_width * 0.5 * math.cos(perp_rad)
    wing1_y = base_y - arrow_width * 0.5 * math.sin(perp_rad)
    wing2_x = base_x - arrow_width * 0.5 * math.cos(perp_rad)
    wing2_y = base_y + arrow_width * 0.5 * math.sin(perp_rad)

    arrow_points = [
        (arrow_tip_x, arrow_tip_y),
        (wing1_x, wing1_y),
        (wing2_x, wing2_y)
    ]
    pygame.draw.polygon(surface, color, arrow_points)

    _icon_cache[cache_key] = surface
    return surface


def get_skip_back_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a skip back icon (vertical bar + two left triangles).

    Args:
        size: Icon size in pixels (square)
        color: Icon color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the skip back icon
    """
    cache_key = f"skip_back_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    padding = size // 6
    bar_width = max(2, size // 8)
    triangle_width = (size - 2 * padding - bar_width) // 2

    # Vertical bar on the left
    bar_rect = pygame.Rect(padding, padding, bar_width, size - 2 * padding)
    pygame.draw.rect(surface, color, bar_rect)

    # First triangle (left)
    t1_left = padding + bar_width + 2
    t1_right = t1_left + triangle_width
    triangle1 = [
        (t1_right, padding),
        (t1_right, size - padding),
        (t1_left, size // 2)
    ]
    pygame.draw.polygon(surface, color, triangle1)

    # Second triangle (right)
    t2_left = t1_right
    t2_right = t2_left + triangle_width
    triangle2 = [
        (t2_right, padding),
        (t2_right, size - padding),
        (t2_left, size // 2)
    ]
    pygame.draw.polygon(surface, color, triangle2)

    _icon_cache[cache_key] = surface
    return surface


def get_skip_forward_icon(
    size: int = 16,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 0)
) -> pygame.Surface:
    """
    Generate a skip forward icon (two right triangles + vertical bar).

    Args:
        size: Icon size in pixels (square)
        color: Icon color as RGB tuple
        bg_color: Background color as RGBA tuple (transparent by default)

    Returns:
        pygame.Surface with the skip forward icon
    """
    cache_key = f"skip_forward_{size}_{color}_{bg_color}"
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    _ensure_pygame_init()

    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill(bg_color)

    padding = size // 6
    bar_width = max(2, size // 8)
    triangle_width = (size - 2 * padding - bar_width) // 2

    # First triangle (left)
    t1_left = padding
    t1_right = t1_left + triangle_width
    triangle1 = [
        (t1_left, padding),
        (t1_left, size - padding),
        (t1_right, size // 2)
    ]
    pygame.draw.polygon(surface, color, triangle1)

    # Second triangle (right)
    t2_left = t1_right
    t2_right = t2_left + triangle_width
    triangle2 = [
        (t2_left, padding),
        (t2_left, size - padding),
        (t2_right, size // 2)
    ]
    pygame.draw.polygon(surface, color, triangle2)

    # Vertical bar on the right
    bar_x = size - padding - bar_width
    bar_rect = pygame.Rect(bar_x, padding, bar_width, size - 2 * padding)
    pygame.draw.rect(surface, color, bar_rect)

    _icon_cache[cache_key] = surface
    return surface


def clear_icon_cache() -> None:
    """Clear the icon cache to free memory."""
    global _icon_cache
    _icon_cache = {}
