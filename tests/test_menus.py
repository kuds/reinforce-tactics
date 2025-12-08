"""Tests for the menu system."""
import os
import pygame
import pytest
from reinforcetactics.ui.menus import Menu, MapSelectionMenu, GameModeMenu


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

    def test_map_selection_with_game_mode(self, pygame_init):
        """Test that MapSelectionMenu filters maps by game_mode."""
        # Test with 1v1 mode
        menu_1v1 = MapSelectionMenu(maps_dir="maps", game_mode="1v1")
        for map_path in menu_1v1.available_maps:
            if map_path != "random":
                assert "1v1" in map_path, f"Map {map_path} should be from 1v1 folder"
                assert "2v2" not in map_path, f"Map {map_path} should not be from 2v2 folder"

        # Test with 2v2 mode
        menu_2v2 = MapSelectionMenu(maps_dir="maps", game_mode="2v2")
        for map_path in menu_2v2.available_maps:
            if map_path != "random":
                assert "2v2" in map_path, f"Map {map_path} should be from 2v2 folder"
                assert "1v1" not in map_path, f"Map {map_path} should not be from 1v1 folder"

    def test_map_selection_display_names_without_folder(self, pygame_init):
        """Test that display names don't include folder prefix when game_mode is set."""
        menu = MapSelectionMenu(maps_dir="maps", game_mode="1v1")

        # Check that display names don't include the folder prefix
        for text, _ in menu.options:
            if text != "Random Map" and text != "Back":
                # Display name should not contain "1v1/"
                assert "1v1" not in text, f"Display name '{text}' should not contain '1v1'"
                assert "/" not in text, f"Display name '{text}' should not contain '/'"


class TestGameModeMenu:
    """Test game mode selection menu."""

    def test_game_mode_menu_discovers_modes(self, pygame_init):
        """Test that GameModeMenu discovers available game modes."""
        menu = GameModeMenu(maps_dir="maps")

        # Should find both 1v1 and 2v2 modes
        assert "1v1" in menu.available_modes, "Should discover 1v1 mode"
        assert "2v2" in menu.available_modes, "Should discover 2v2 mode"

    def test_game_mode_menu_creates_options(self, pygame_init):
        """Test that GameModeMenu creates menu options for discovered modes."""
        menu = GameModeMenu(maps_dir="maps")

        # Check that options include the modes (excluding Back button)
        option_texts = [text for text, _ in menu.options]
        assert "1v1" in option_texts, "1v1 should be in menu options"
        assert "2v2" in option_texts, "2v2 should be in menu options"
        assert "Back" in option_texts, "Back button should be in menu options"

    def test_game_mode_menu_returns_selected_mode(self, pygame_init):
        """Test that GameModeMenu returns the selected mode."""
        menu = GameModeMenu(maps_dir="maps")

        # Find the 1v1 option and test its callback
        for text, callback in menu.options:
            if text == "1v1":
                result = callback()
                assert result == "1v1", "Selecting 1v1 should return '1v1'"
                break

        # Find the 2v2 option and test its callback
        for text, callback in menu.options:
            if text == "2v2":
                result = callback()
                assert result == "2v2", "Selecting 2v2 should return '2v2'"
                break

    def test_game_mode_menu_back_returns_none(self, pygame_init):
        """Test that selecting Back in GameModeMenu returns None."""
        menu = GameModeMenu(maps_dir="maps")

        # Find the Back option and test its callback
        for text, callback in menu.options:
            if text == "Back":
                result = callback()
                assert result is None, "Selecting Back should return None"
                break


class TestUniformMenuWidths:
    """Test uniform menu option widths."""

    def test_menu_options_have_uniform_width(self, pygame_init):
        """Test that all menu option backgrounds have the same width."""
        menu = Menu(title="Test Menu")
        menu.add_option("Short", lambda: None)
        menu.add_option("Much Longer Option Text", lambda: None)
        menu.add_option("Medium Length", lambda: None)

        # Draw menu to create option rects
        menu.draw()

        # Check that all option rects have the same width
        if len(menu.option_rects) > 1:
            first_width = menu.option_rects[0].width
            for rect in menu.option_rects[1:]:
                assert rect.width == first_width, \
                    "All option background rectangles should have the same width"

    def test_uniform_width_based_on_longest_option(self, pygame_init):
        """Test that uniform width is based on the longest option text."""
        menu = Menu(title="Test Menu")
        menu.add_option("Short", lambda: None)
        menu.add_option("This is a very long option text", lambda: None)
        menu.add_option("Medium", lambda: None)

        # Draw menu to create option rects
        menu.draw()

        # The width should accommodate the longest option
        # Calculate what the width should be for the longest option
        padding_x = 40
        longest_text = "> This is a very long option text"  # With selection indicator
        text_surface = menu.option_font.render(longest_text, True, menu.text_color)
        expected_width = text_surface.get_width() + 2 * padding_x

        # All rects should have this width
        for rect in menu.option_rects:
            assert rect.width == expected_width, \
                "All option rects should have width based on longest option"

    def test_uniform_width_boxes_are_centered(self, pygame_init):
        """Test that uniform width boxes are centered on screen."""
        menu = Menu(title="Test Menu")
        menu.add_option("Option 1", lambda: None)
        menu.add_option("Option 2", lambda: None)

        # Draw menu to create option rects
        menu.draw()

        screen_width = menu.screen.get_width()

        # Check that all option rects are centered
        for rect in menu.option_rects:
            expected_x = (screen_width - rect.width) // 2
            assert rect.x == expected_x, \
                f"Option rect should be centered at x={expected_x}, but is at x={rect.x}"
