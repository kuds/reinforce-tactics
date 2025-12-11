"""Font utility for cross-platform Unicode support.

Includes support for Latin accents, CJK characters, and symbols.
"""
from typing import Dict, Optional

import pygame

# Priority list of fonts supporting comprehensive Unicode
# Includes Latin, CJK, and symbols
CJK_FONT_CANDIDATES = [
    # Comprehensive Unicode fonts with good Latin + CJK + symbol coverage
    "Arial Unicode MS",   # Wide Unicode support - excellent coverage
    "DejaVu Sans",        # Good Unicode coverage - Latin + many symbols
    # macOS standard fonts (must come before CJK-specific fonts for Latin support)
    "Helvetica Neue",     # macOS default - excellent Latin support
    "Helvetica",          # macOS fallback - excellent Latin support
    # CJK-specific fonts with good overall Unicode support
    "Noto Sans CJK",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Malgun Gothic",      # Windows Korean font
    "Microsoft YaHei",    # Windows Chinese font
    "Apple SD Gothic Neo", # macOS Korean font - good Unicode coverage
    "AppleGothic",        # macOS Korean font (older)
    "PingFang SC",        # macOS Chinese font
    "FreeSans",           # Linux
    # Specialized symbol fonts (lower priority to avoid issues with Latin characters)
    "Symbola",            # Cross-platform - excellent Unicode coverage
    "Noto Sans Symbols",  # Cross-platform - Google's symbol font
    "Noto Sans Symbols2", # Additional symbols
    "Segoe UI Symbol",    # Windows - good symbol coverage
    "Segoe UI Emoji",     # Windows - emoji support
    "Apple Symbols",      # macOS - symbol support
    "Apple Color Emoji",  # macOS - emoji support
    "Noto Color Emoji",   # Emoji support
]

_font_cache: Dict[int, pygame.font.Font] = {}
_available_fonts_cache: Optional[list] = None


def _get_available_fonts() -> list:
    """Get list of available system fonts (cached).

    Returns:
        List of available font names in lowercase without spaces
    """
    global _available_fonts_cache
    if _available_fonts_cache is None:
        available_fonts = pygame.font.get_fonts()
        _available_fonts_cache = [f.lower().replace(" ", "") for f in available_fonts]
    return _available_fonts_cache


def get_font(size: int) -> pygame.font.Font:
    """
    Get a font that supports comprehensive Unicode.

    Includes Latin accents, CJK characters, and symbols.
    Uses font fallback list for pygame.font.SysFont to ensure proper rendering.

    Args:
        size: Font size in points

    Returns:
        pygame.font.Font instance with comprehensive Unicode support if available,
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

    # Try to use SysFont with a fallback list
    # This allows pygame to use multiple fonts for different character ranges
    try:
        # Build a font fallback list with both Latin and CJK support
        font_fallback_list = []

        # Check which fonts are available (cached)
        available_fonts_lower = _get_available_fonts()

        for candidate in CJK_FONT_CANDIDATES:
            candidate_normalized = candidate.lower().replace(" ", "")
            if candidate_normalized in available_fonts_lower:
                font_fallback_list.append(candidate)
                # Stop after finding first 3 fonts for fallback
                if len(font_fallback_list) >= 3:
                    break

        if font_fallback_list:
            # Use comma-separated list for pygame SysFont fallback
            font_names = ",".join(font_fallback_list)
            font = pygame.font.SysFont(font_names, size)
            _font_cache[size] = font
            return font
    except Exception:
        pass

    # Fallback to default
    font = pygame.font.Font(None, size)
    _font_cache[size] = font
    return font
