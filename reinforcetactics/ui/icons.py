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

    # Step 1: Define and draw the arrowhead first (triangle pointing clockwise)
    # Position the arrowhead at the top-right of the circle, pointing downward
    arrow_angle = 45  # Position on circle (degrees from right, counter-clockwise)
    arrow_angle_rad = math.radians(arrow_angle)

    # Arrowhead tip position on the circle
    arrow_tip_x = center_x + radius * math.cos(arrow_angle_rad)
    arrow_tip_y = center_y - radius * math.sin(arrow_angle_rad)

    # Arrow points in clockwise direction (tangent to circle, going "down")
    # Tangent direction for clockwise motion is perpendicular to radius, pointing down-right
    arrow_dir = arrow_angle - 90  # Direction arrowhead points (clockwise tangent)
    arrow_dir_rad = math.radians(arrow_dir)

    # Arrow dimensions proportional to icon size
    arrow_len = size // 3
    arrow_width = size // 3

    # Calculate the two base points of the triangle
    # Base center is behind the tip
    base_center_x = arrow_tip_x - arrow_len * 0.7 * math.cos(arrow_dir_rad)
    base_center_y = arrow_tip_y + arrow_len * 0.7 * math.sin(arrow_dir_rad)

    # Perpendicular direction for the arrow width
    perp_rad = arrow_dir_rad + math.pi / 2

    # Wing points (base corners of the triangle)
    wing1_x = base_center_x + arrow_width * 0.5 * math.cos(perp_rad)
    wing1_y = base_center_y - arrow_width * 0.5 * math.sin(perp_rad)
    wing2_x = base_center_x - arrow_width * 0.5 * math.cos(perp_rad)
    wing2_y = base_center_y + arrow_width * 0.5 * math.sin(perp_rad)

    # Draw the arrowhead triangle
    arrow_points = [
        (arrow_tip_x, arrow_tip_y),
        (wing1_x, wing1_y),
        (wing2_x, wing2_y)
    ]
    pygame.draw.polygon(surface, color, arrow_points)

    # Step 2: Draw the circular arc connecting into the arrowhead
    # Arc starts just after where the arrowhead base is and wraps around
    # The arc should connect smoothly to one of the wing points (the outer one)

    # Calculate where the arc should start (at the outer wing of the arrow)
    # This creates a smooth connection from the arc into the arrowhead
    arc_start_angle = arrow_angle + 25  # Start arc slightly past the arrowhead
    arc_end_angle = arrow_angle + 320  # Go most of the way around

    num_segments = 32

    # Draw the arc as connected line segments
    points = []
    for i in range(num_segments + 1):
        angle_deg = arc_start_angle + (arc_end_angle - arc_start_angle) * i / num_segments
        angle_rad = math.radians(angle_deg)
        x = center_x + radius * math.cos(angle_rad)
        y = center_y - radius * math.sin(angle_rad)
        points.append((x, y))

    # Draw thick lines for the arc
    if len(points) >= 2:
        pygame.draw.lines(surface, color, False, points, line_thickness)

        # Draw circles at joints for smoother appearance
        half_thick = line_thickness // 2
        for point in points:
            pygame.draw.circle(surface, color, (int(point[0]), int(point[1])), half_thick)

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
