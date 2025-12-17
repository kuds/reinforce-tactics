"""Font utility for cross-platform Unicode support.

Includes support for Latin accents, CJK characters, and symbols.
"""
from pathlib import Path
from typing import Dict, Optional

import pygame

# Path to bundled fonts directory
FONTS_DIR = Path(__file__).parent.parent / "assets" / "fonts"

# Bundled font files to look for (in priority order)
# These fonts should support Latin extended + CJK characters
BUNDLED_FONT_FILES = [
    "NotoSansCJKsc-Regular.ttf",  # Noto Sans CJK Simplified Chinese (includes Latin)
    "NotoSansCJK-Regular.ttc",     # Noto Sans CJK collection
    "NotoSans-Regular.ttf",        # Noto Sans (Latin + extended)
]

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
_bundled_font_path: Optional[Path] = None
_bundled_font_checked: bool = False


def _find_bundled_font() -> Optional[Path]:
    """Find a bundled font file in the assets directory.

    Returns:
        Path to the bundled font file, or None if not found
    """
    global _bundled_font_path, _bundled_font_checked

    if _bundled_font_checked:
        return _bundled_font_path

    _bundled_font_checked = True

    if not FONTS_DIR.exists():
        return None

    # Check for each bundled font in priority order
    for font_file in BUNDLED_FONT_FILES:
        font_path = FONTS_DIR / font_file
        if font_path.exists():
            _bundled_font_path = font_path
            return _bundled_font_path

    return None


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
    Priority order:
    1. Bundled font from assets/fonts directory (most reliable)
    2. System fonts known to support CJK + Latin
    3. Default pygame font (limited Unicode support)

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

    # Priority 1: Try to use bundled font (most reliable for Unicode)
    bundled_font_path = _find_bundled_font()
    if bundled_font_path:
        try:
            font = pygame.font.Font(str(bundled_font_path), size)
            _font_cache[size] = font
            return font
        except (pygame.error, OSError, FileNotFoundError):
            # Bundled font loading failed, try system fonts
            pass

    # Priority 2: Find the best available system font with Unicode support
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

    # Priority 3: Fallback to default (WARNING: limited Unicode support)
    font = pygame.font.Font(None, size)
    _font_cache[size] = font
    return font


def get_font_info() -> dict:
    """
    Get information about the current font configuration.

    Returns:
        Dictionary with font status information:
        - bundled_font: Path to bundled font if available, None otherwise
        - using_bundled: True if bundled font will be used
        - system_fonts_available: List of available CJK-capable system fonts
        - unicode_support: 'full', 'partial', or 'limited'
        - recommendation: String with setup recommendation if needed
    """
    info = {
        "bundled_font": None,
        "using_bundled": False,
        "system_fonts_available": [],
        "unicode_support": "limited",
        "recommendation": None,
    }

    # Check for bundled font
    bundled = _find_bundled_font()
    if bundled:
        info["bundled_font"] = str(bundled)
        info["using_bundled"] = True
        info["unicode_support"] = "full"
        return info

    # Check for system fonts
    if not pygame.font.get_init():
        pygame.font.init()

    available_fonts_lower = _get_available_fonts()
    for candidate in CJK_FONT_CANDIDATES:
        candidate_normalized = candidate.lower().replace(" ", "")
        if candidate_normalized in available_fonts_lower:
            info["system_fonts_available"].append(candidate)

    if info["system_fonts_available"]:
        # Check if any are CJK-capable (not just Latin)
        cjk_fonts = [
            "Arial Unicode MS", "Noto Sans CJK", "Noto Sans CJK SC",
            "Noto Sans CJK JP", "Noto Sans CJK KR", "Microsoft YaHei",
            "Malgun Gothic", "PingFang SC", "PingFang TC",
            "Apple SD Gothic Neo", "Hiragino Sans"
        ]
        has_cjk = any(
            f.lower().replace(" ", "") in [c.lower().replace(" ", "") for c in cjk_fonts]
            for f in info["system_fonts_available"]
        )
        if has_cjk:
            info["unicode_support"] = "full"
        else:
            info["unicode_support"] = "partial"
            info["recommendation"] = (
                "System fonts support Latin characters but may not render "
                "Chinese/Korean text. Consider installing a Noto Sans CJK font."
            )
    else:
        info["unicode_support"] = "limited"
        info["recommendation"] = (
            "No Unicode-capable fonts found. Special characters (French accents, "
            "Chinese, Korean) may not render correctly. "
            "To fix this, download NotoSansCJKsc-Regular.ttf and place it in:\n"
            f"  {FONTS_DIR}\n"
            "Download from: https://fonts.google.com/noto/specimen/Noto+Sans+SC"
        )

    return info


def check_font_support() -> bool:
    """
    Check if the current font configuration supports all game languages.

    Returns:
        True if full Unicode support is available, False otherwise.
        Prints a warning message if support is limited.
    """
    info = get_font_info()

    if info["unicode_support"] == "full":
        return True

    if info["recommendation"]:
        print(f"⚠️  Font Warning: {info['recommendation']}")

    return False
