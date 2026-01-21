"""Font utility for cross-platform Unicode support.

Includes support for Latin accents, CJK characters, and symbols.
"""
from typing import Dict, Optional

import pygame

# Priority list of fonts supporting comprehensive Unicode
# Includes Latin, CJK, and symbols
CJK_FONT_CANDIDATES = [
    # Comprehensive Unicode fonts with BOTH Latin accents AND CJK support
    "Arial Unicode MS",   # Wide Unicode support - excellent coverage for both
    # macOS fonts with both Latin and CJK support
    "Apple SD Gothic Neo", # macOS - has both Latin extended and CJK support
    "PingFang SC",        # macOS Chinese font - also has Latin support
    "PingFang TC",        # macOS Traditional Chinese - also has Latin support
    "Hiragino Sans",      # macOS Japanese - also has Latin support
    # Comprehensive cross-platform fonts
    "DejaVu Sans",        # Good Unicode coverage - Latin + many symbols
    "Noto Sans CJK",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    # Windows fonts with CJK support
    "Malgun Gothic",      # Windows Korean font
    "Microsoft YaHei",    # Windows Chinese font
    # macOS fonts (Latin-focused, CJK-limited)
    "Helvetica Neue",     # macOS default - excellent Latin, NO CJK
    "Helvetica",          # macOS fallback - excellent Latin, NO CJK
    "AppleGothic",        # macOS Korean font (older)
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
    Selects a single font with the best comprehensive Unicode coverage.

    Args:
        size: Font size in points

    Returns:
        pygame.font.Font instance with comprehensive Unicode support if available,
        otherwise falls back to default pygame font
    """
    global _available_fonts_cache

    # Check if pygame.font is initialized; if not, initialize it
    # Also clear caches when reinitializing to avoid stale state
    if not pygame.font.get_init():
        pygame.font.init()
        # Clear caches after reinitialization to avoid stale fonts
        _font_cache.clear()
        _available_fonts_cache = None

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

    # Find the best available font with comprehensive Unicode support
    try:
        # Check which fonts are available (cached)
        available_fonts_lower = _get_available_fonts()

        # Try each candidate in priority order
        for candidate in CJK_FONT_CANDIDATES:
            candidate_normalized = candidate.lower().replace(" ", "")
            if candidate_normalized in available_fonts_lower:
                # Found an available font - use it
                font = pygame.font.SysFont(candidate, size)
                _font_cache[size] = font
                return font
    except (pygame.error, OSError):
        # Font loading failed, will fallback to default
        pass

    # Fallback to default
    font = pygame.font.Font(None, size)
    _font_cache[size] = font
    return font
