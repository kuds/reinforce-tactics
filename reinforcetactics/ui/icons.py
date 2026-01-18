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


def clear_icon_cache() -> None:
    """Clear the icon cache to free memory."""
    global _icon_cache
    _icon_cache = {}
