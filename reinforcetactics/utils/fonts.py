"""Font utility for cross-platform Unicode support.

The game bundles its own fonts (in ``assets/fonts/``) so the UI looks the
same on every platform:

- **Noto Sans** (``get_font``) — body/UI text. Full Latin coverage,
  including the accented characters used by the French and Spanish
  translations.
- **Pixelify Sans** (``get_display_font``) — pixel-styled display font for
  titles and headings, matching the game's pixel-art style.

Neither bundled font covers CJK, so when the active language is Korean or
Chinese both helpers fall back to a system font with CJK coverage. The
system-font chain is also the fallback when the bundled files are missing
(e.g. a stripped-down install).
"""

from pathlib import Path

import pygame

# Bundled fonts, relative to the assets/fonts directory.
BODY_FONT_FILE = "NotoSans-Regular.ttf"
DISPLAY_FONT_FILE = "PixelifySans-Regular.ttf"

# Languages whose glyphs the bundled fonts cannot render.
CJK_LANGUAGES = ("korean", "chinese")

# Priority list of system fonts supporting comprehensive Unicode.
# Includes Latin, CJK, and symbols. Used for CJK languages and as a
# fallback when the bundled fonts are unavailable.
CJK_FONT_CANDIDATES = [
    # Comprehensive Unicode fonts with BOTH Latin accents AND CJK support
    "Arial Unicode MS",  # Wide Unicode support - excellent coverage for both
    # macOS fonts with both Latin and CJK support
    "Apple SD Gothic Neo",  # macOS - has both Latin extended and CJK support
    "PingFang SC",  # macOS Chinese font - also has Latin support
    "PingFang TC",  # macOS Traditional Chinese - also has Latin support
    "Hiragino Sans",  # macOS Japanese - also has Latin support
    # Comprehensive cross-platform fonts
    "DejaVu Sans",  # Good Unicode coverage - Latin + many symbols
    "Noto Sans CJK",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    # Windows fonts with CJK support
    "Malgun Gothic",  # Windows Korean font
    "Microsoft YaHei",  # Windows Chinese font
    # macOS fonts (Latin-focused, CJK-limited)
    "Helvetica Neue",  # macOS default - excellent Latin, NO CJK
    "Helvetica",  # macOS fallback - excellent Latin, NO CJK
    "AppleGothic",  # macOS Korean font (older)
    "FreeSans",  # Linux
    # Specialized symbol fonts (lower priority to avoid issues with Latin characters)
    "Symbola",  # Cross-platform - excellent Unicode coverage
    "Noto Sans Symbols",  # Cross-platform - Google's symbol font
    "Noto Sans Symbols2",  # Additional symbols
    "Segoe UI Symbol",  # Windows - good symbol coverage
    "Segoe UI Emoji",  # Windows - emoji support
    "Apple Symbols",  # macOS - symbol support
    "Apple Color Emoji",  # macOS - emoji support
    "Noto Color Emoji",  # Emoji support
]

# Cache key is (kind, size) where kind is "body", "display", or "system".
_font_cache: dict[tuple[str, int], pygame.font.Font] = {}
_available_fonts_cache: list | None = None
_bundled_fonts_dir: Path | None = None
_bundled_fonts_dir_resolved = False
_quit_hook_registered = False


def _clear_caches_on_quit() -> None:
    """Drop cached fonts when pygame quits.

    Font objects do not survive a ``pygame.quit()`` / ``pygame.init()``
    cycle — rendering with a stale font can crash the interpreter — so this
    is registered via ``pygame.register_quit`` whenever the cache is in use.
    """
    global _available_fonts_cache, _quit_hook_registered
    _font_cache.clear()
    _available_fonts_cache = None
    _quit_hook_registered = False


def _resolve_bundled_fonts_dir() -> Path | None:
    """Locate the bundled ``assets/fonts`` directory.

    Walks up from this file (the repo ships ``assets/fonts/`` at the root)
    and also checks the current working directory. Returns ``None`` if the
    directory can't be found. Result is cached.
    """
    global _bundled_fonts_dir, _bundled_fonts_dir_resolved
    if _bundled_fonts_dir_resolved:
        return _bundled_fonts_dir

    candidates = [Path.cwd() / "assets" / "fonts"]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "assets" / "fonts")

    _bundled_fonts_dir = next((c for c in candidates if c.is_dir()), None)
    _bundled_fonts_dir_resolved = True
    return _bundled_fonts_dir


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


def _needs_cjk_font() -> bool:
    """Whether the active language needs glyphs the bundled fonts lack."""
    # Imported lazily: language settings load at startup and this avoids any
    # import-order coupling between utils modules.
    from reinforcetactics.utils.language import get_language

    try:
        return get_language().get_current_language() in CJK_LANGUAGES
    except Exception:  # pylint: disable=broad-except
        return False


def _ensure_font_init() -> None:
    """Initialize pygame.font if needed, clearing stale caches."""
    global _available_fonts_cache, _quit_hook_registered
    if not pygame.font.get_init():
        pygame.font.init()
        # Clear caches after reinitialization to avoid stale fonts
        _font_cache.clear()
        _available_fonts_cache = None
    if not _quit_hook_registered:
        pygame.register_quit(_clear_caches_on_quit)
        _quit_hook_registered = True


def _get_cached(kind: str, size: int) -> pygame.font.Font | None:
    """Return a cached font if present and still valid."""
    font = _font_cache.get((kind, size))
    if font is None:
        return None
    try:
        font.get_height()
        return font
    except pygame.error:
        del _font_cache[(kind, size)]
        return None


def _load_bundled(kind: str, filename: str, size: int) -> pygame.font.Font | None:
    """Load a bundled font file, or return None if unavailable."""
    fonts_dir = _resolve_bundled_fonts_dir()
    if fonts_dir is None:
        return None
    path = fonts_dir / filename
    if not path.is_file():
        return None
    try:
        font = pygame.font.Font(str(path), size)
    except (pygame.error, OSError, FileNotFoundError):
        return None
    _font_cache[(kind, size)] = font
    return font


def _get_system_font(size: int) -> pygame.font.Font:
    """Get the best available system font with comprehensive Unicode support."""
    cached = _get_cached("system", size)
    if cached is not None:
        return cached

    try:
        available_fonts_lower = _get_available_fonts()
    except (pygame.error, OSError):
        available_fonts_lower = set()

    for candidate in CJK_FONT_CANDIDATES:
        candidate_normalized = candidate.lower().replace(" ", "")
        if candidate_normalized not in available_fonts_lower:
            continue
        # One broken candidate must not abort the whole chain -- try the
        # next one instead of dropping straight to the default font.
        try:
            font = pygame.font.SysFont(candidate, size)
        except (pygame.error, OSError):
            continue
        _font_cache[("system", size)] = font
        return font

    font = pygame.font.Font(None, size)
    _font_cache[("system", size)] = font
    return font


def get_font(size: int) -> pygame.font.Font:
    """
    Get the game's body/UI font at the given size.

    Returns the bundled Noto Sans font so text looks identical on every
    platform. Falls back to a system font with comprehensive Unicode
    coverage when the active language is CJK (Korean/Chinese) or the
    bundled font file is missing.

    Args:
        size: Font size in points

    Returns:
        pygame.font.Font instance
    """
    _ensure_font_init()

    if _needs_cjk_font():
        return _get_system_font(size)

    cached = _get_cached("body", size)
    if cached is not None:
        return cached

    font = _load_bundled("body", BODY_FONT_FILE, size)
    if font is not None:
        return font
    return _get_system_font(size)


def get_display_font(size: int) -> pygame.font.Font:
    """
    Get the game's display font (titles/headings) at the given size.

    Returns the bundled Pixelify Sans font, a pixel-styled face matching
    the game's pixel-art look. Falls back to :func:`get_font` when the
    active language is CJK or the bundled font file is missing.

    Args:
        size: Font size in points

    Returns:
        pygame.font.Font instance
    """
    _ensure_font_init()

    if _needs_cjk_font():
        return _get_system_font(size)

    cached = _get_cached("display", size)
    if cached is not None:
        return cached

    font = _load_bundled("display", DISPLAY_FONT_FILE, size)
    if font is not None:
        return font
    return get_font(size)
