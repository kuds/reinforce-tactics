"""Tests for the menu system."""
import os
import pygame
import pytest
from reinforcetactics.ui.menus import Menu, MapSelectionMenu


@pytest.fixture
def pygame_init():
    """Initialize pygame for tests."""
    pygame.init()
    yield
    pygame.quit()


class TestMenuMouseInput:
    """Test menu mouse input handling."""

    def test_menu_has_hover_tracking(self, pygame_init):
        """Test that menu initializes with hover tracking."""
        menu = Menu(title="Test Menu")
        assert hasattr(menu, 'hover_index')
        assert menu.hover_index == -1
        assert hasattr(menu, 'option_rects')
        assert isinstance(menu.option_rects, list)

    def test_menu_handles_mousebuttondown(self, pygame_init):
        """Test that menu handles mouse click events."""
        menu = Menu(title="Test Menu")
        callback_called = []

        def test_callback():
            callback_called.append(True)
            return "result"

        menu.add_option("Test Option", test_callback)

        # Draw menu to create option rects
        menu.draw()

        # Simulate mouse click on option (need to create the event)
        if menu.option_rects:
            rect = menu.option_rects[0]
            # Create a fake mouse button down event at the center of the rect
            event = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                {'button': 1, 'pos': rect.center}
            )

            result = menu.handle_input(event)
            assert result == "result"
            assert len(callback_called) == 1

    def test_menu_handles_mousemotion(self, pygame_init):
        """Test that menu updates hover state on mouse motion."""
        menu = Menu(title="Test Menu")
        menu.add_option("Option 1", lambda: "result1")
        menu.add_option("Option 2", lambda: "result2")

        # Draw menu to create option rects
        menu.draw()

        # Simulate mouse motion over first option
        if menu.option_rects:
            rect = menu.option_rects[0]
            event = pygame.event.Event(
                pygame.MOUSEMOTION,
                {'pos': rect.center}
            )

            menu.handle_input(event)
            assert menu.hover_index == 0

    def test_menu_keyboard_still_works(self, pygame_init):
        """Test that keyboard navigation still works after mouse support."""
        menu = Menu(title="Test Menu")
        callback_results = []

        menu.add_option("Option 1", lambda: callback_results.append(1) or "result1")
        menu.add_option("Option 2", lambda: callback_results.append(2) or "result2")

        # Test DOWN key
        event_down = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN})
        menu.handle_input(event_down)
        assert menu.selected_index == 1

        # Test UP key
        event_up = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP})
        menu.handle_input(event_up)
        assert menu.selected_index == 0

        # Test RETURN key
        event_return = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN})
        result = menu.handle_input(event_return)
        assert result == "result1"
        assert callback_results == [1]


class TestMapSelectionMenu:
    """Test map selection menu."""

    @staticmethod
    def _assert_valid_map_path(map_path):
        """Helper to assert that a map path is valid."""
        # Should start with "maps/"
        assert map_path.startswith("maps/") or map_path.startswith("maps" + os.sep), \
            f"Map path '{map_path}' should start with 'maps/'"
        # Should end with .csv
        assert map_path.endswith(".csv"), \
            f"Map path '{map_path}' should end with '.csv'"

    def test_map_paths_include_maps_prefix(self, pygame_init):
        """Test that map paths include the maps/ prefix."""
        # Create a map selection menu
        menu = MapSelectionMenu(maps_dir="maps")

        # Check that maps include the full path
        for map_path in menu.available_maps:
            if map_path != "random":
                self._assert_valid_map_path(map_path)

    def test_map_files_exist(self, pygame_init):
        """Test that the map files referenced actually exist."""
        menu = MapSelectionMenu(maps_dir="maps")

        for map_path in menu.available_maps:
            if map_path != "random":
                # The full path should exist
                assert os.path.exists(map_path), \
                    f"Map file should exist at: {map_path}"

    def test_map_selection_returns_full_path(self, pygame_init):
        """Test that selecting a map returns the full path."""
        menu = MapSelectionMenu(maps_dir="maps")

        # Find a non-random map option
        for i, map_path in enumerate(menu.available_maps):
            if map_path != "random":
                # The callback should return the full path
                _, callback = menu.options[i]
                result = callback()
                self._assert_valid_map_path(result)
                break
