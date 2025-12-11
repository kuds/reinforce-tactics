"""Font utility for cross-platform Unicode support including Korean/CJK characters."""
import pygame
from typing import Dict, Optional

# Priority list of fonts that support Korean/CJK characters
CJK_FONT_CANDIDATES = [
    "Noto Sans CJK",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Malgun Gothic",      # Windows Korean font
    "Microsoft YaHei",    # Windows Chinese font
    "Apple SD Gothic Neo", # macOS Korean font
    "AppleGothic",        # macOS Korean font (older)
    "PingFang SC",        # macOS Chinese font
    "DejaVu Sans",        # Good Unicode coverage
    "Arial Unicode MS",   # Wide Unicode support
    "FreeSans",           # Linux
]

_font_cache: Dict[int, pygame.font.Font] = {}
_system_font_name: Optional[str] = None


def _find_cjk_font() -> Optional[str]:
    """Find a system font that supports CJK characters."""
    global _system_font_name
    if _system_font_name is not None:
        return _system_font_name

    available_fonts = pygame.font.get_fonts()
    available_fonts_lower = [f.lower().replace(" ", "") for f in available_fonts]

    for candidate in CJK_FONT_CANDIDATES:
        candidate_normalized = candidate.lower().replace(" ", "")
        if candidate_normalized in available_fonts_lower:
            _system_font_name = candidate
            return candidate

    _system_font_name = ""  # Empty string means no CJK font found
    return None


def get_font(size: int) -> pygame.font.Font:
    """
    Get a font that supports Korean/CJK characters.

    Args:
        size: Font size in points

    Returns:
        pygame.font.Font instance with Unicode/CJK support if available,
        otherwise falls back to default pygame font
    """
    # Check if pygame.font is initialized; if not, initialize it
    if not pygame.font.get_init():
        pygame.font.init()

    # Check if we have a cached font for this size
    # But verify it's still valid by checking if pygame.font is initialized
    if size in _font_cache:
        try:
            # Test if the cached font is still valid
            _font_cache[size].get_height()
            return _font_cache[size]
        except pygame.error:
            # Font is invalid, remove from cache
            del _font_cache[size]

    cjk_font = _find_cjk_font()
    if cjk_font:
        try:
            font = pygame.font.SysFont(cjk_font, size)
            _font_cache[size] = font
            return font
        except Exception:
            pass

    # Fallback to default
    font = pygame.font.Font(None, size)
    _font_cache[size] = font
    return font

