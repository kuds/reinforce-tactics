"""Tests for the language menu."""
import pygame
import pytest
from reinforcetactics.ui.menus.settings.language_menu import LanguageMenu
from reinforcetactics.utils.language import get_language, reset_language


@pytest.fixture
def pygame_init():
    """Initialize pygame for tests."""
    pygame.init()
    # Reset to English before each test for consistency
    reset_language('en')
    yield
    pygame.quit()


def test_language_menu_creation(pygame_init):
    """Test that language menu can be created."""
    menu = LanguageMenu()
    assert menu is not None
    assert len(menu.options) > 0


def test_language_menu_has_all_languages(pygame_init):
    """Test that language menu has options for all supported languages."""
    menu = LanguageMenu()

    # Should have 5 language options + 1 back button
    assert len(menu.options) == 6

    # Check that all language names are present
    option_texts = [text for text, _ in menu.options]
    assert 'English' in option_texts
    assert 'Français' in option_texts
    assert '한국어' in option_texts
    assert 'Español' in option_texts
    assert '中文' in option_texts


def test_language_change_updates_menu_title(pygame_init):
    """Test that changing language updates the menu title immediately."""
    menu = LanguageMenu()

    # Initial title should be in English
    initial_title = menu.title
    assert initial_title == 'Select Language'

    # Change language to French
    menu._set_language('fr')

    # Title should now be in French
    assert menu.title == 'Choisir la Langue'
    assert menu.title != initial_title


def test_language_change_updates_back_button(pygame_init):
    """Test that changing language updates the Back button text."""
    menu = LanguageMenu()

    # Initial back button should be in English
    back_option = menu.options[-1][0]
    assert back_option == 'Back'

    # Change language to French
    menu._set_language('fr')

    # Back button should now be in French
    back_option = menu.options[-1][0]
    assert back_option == 'Retour'


def test_language_change_updates_global_language(pygame_init):
    """Test that changing language updates the global language instance."""
    # Reset to English
    reset_language('en')
    lang = get_language()
    assert lang.get('common.back') == 'Back'

    # Create menu and change language
    menu = LanguageMenu()
    menu._set_language('es')

    # Global language should be updated
    lang = get_language()
    assert lang.get('common.back') == 'Atrás'


def test_language_change_rebuilds_options(pygame_init):
    """Test that changing language rebuilds all menu options."""
    menu = LanguageMenu()

    # Record initial number of options
    initial_option_count = len(menu.options)

    # Change language
    menu._set_language('ko')

    # Should still have the same number of options
    assert len(menu.options) == initial_option_count

    # Back button should be in Korean
    back_option = menu.options[-1][0]
    assert back_option == '뒤로'


def test_multiple_language_changes(pygame_init):
    """Test multiple consecutive language changes."""
    menu = LanguageMenu()

    # Change to Spanish
    menu._set_language('es')
    assert menu.title == 'Seleccionar Idioma'
    assert menu.options[-1][0] == 'Atrás'

    # Change to Korean
    menu._set_language('ko')
    assert menu.title == '언어 선택'
    assert menu.options[-1][0] == '뒤로'

    # Change to Chinese
    menu._set_language('zh')
    assert menu.title == '选择语言'
    assert menu.options[-1][0] == '返回'

    # Change back to English
    menu._set_language('en')
    assert menu.title == 'Select Language'
    assert menu.options[-1][0] == 'Back'
