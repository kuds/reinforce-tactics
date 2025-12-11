"""Tests for font utility module."""
import pytest
import pygame
from reinforcetactics.utils.fonts import get_font, _find_cjk_font, _font_cache


@pytest.fixture
def pygame_init():
    """Initialize pygame for tests."""
    pygame.init()
    # Clear cache before each test to avoid test interference
    _font_cache.clear()
    yield
    pygame.quit()


def test_get_font_returns_font_object(pygame_init):
    """Test that get_font returns a pygame Font object."""
    font = get_font(24)
    assert isinstance(font, pygame.font.Font)


def test_get_font_caching(pygame_init):
    """Test that fonts are cached and reused."""
    font1 = get_font(24)
    font2 = get_font(24)
    # Same size should return the same cached object
    assert font1 is font2


def test_get_font_different_sizes(pygame_init):
    """Test that different sizes return different font objects."""
    font_small = get_font(12)
    font_large = get_font(48)

    assert font_small is not font_large


def test_get_font_various_sizes(pygame_init):
    """Test that get_font works with various typical sizes."""
    sizes = [12, 16, 20, 24, 28, 32, 36, 40, 48]

    for size in sizes:
        font = get_font(size)
        assert isinstance(font, pygame.font.Font)


def test_find_cjk_font_returns_string_or_none(pygame_init):
    """Test that _find_cjk_font returns a string or None."""
    result = _find_cjk_font()
    assert result is None or isinstance(result, str)


def test_font_renders_text(pygame_init):
    """Test that fonts can render text successfully."""
    font = get_font(24)

    # Test English text
    english_surface = font.render("Hello", True, (255, 255, 255))
    assert isinstance(english_surface, pygame.Surface)
    assert english_surface.get_width() > 0

    # Test Korean text (may not render perfectly without CJK font, but shouldn't crash)
    korean_surface = font.render("안녕하세요", True, (255, 255, 255))
    assert isinstance(korean_surface, pygame.Surface)
    assert korean_surface.get_width() > 0


def test_font_initialization_without_pygame():
    """Test that get_font initializes pygame.font if needed."""
    # Start fresh - quit and re-init pygame
    pygame.quit()
    pygame.init()

    # Quit font system to test auto-initialization
    pygame.font.quit()

    # get_font should auto-initialize
    font = get_font(24)
    assert isinstance(font, pygame.font.Font)

    # Clean up
    pygame.quit()


def test_get_font_edge_cases(pygame_init):
    """Test get_font with edge case sizes."""
    # Very small font
    small_font = get_font(8)
    assert isinstance(small_font, pygame.font.Font)

    # Very large font
    large_font = get_font(200)
    assert isinstance(large_font, pygame.font.Font)

    # Ensure they're different
    assert small_font is not large_font

