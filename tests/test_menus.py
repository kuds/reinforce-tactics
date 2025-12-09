"""Tests for the menu system."""
import os
import numpy as np
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


class TestPlayerConfigMenu:
    """Test player configuration menu."""

    def test_player_config_menu_initialization_1v1(self, pygame_init):
        """Test that PlayerConfigMenu initializes correctly for 1v1 mode."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        assert menu.num_players == 2
        assert len(menu.player_configs) == 2
        
        # Player 1 should be human by default
        assert menu.player_configs[0]['type'] == 'human'
        assert menu.player_configs[0]['bot_type'] is None
        
        # Player 2 should be computer (SimpleBot) by default
        assert menu.player_configs[1]['type'] == 'computer'
        assert menu.player_configs[1]['bot_type'] == 'SimpleBot'

    def test_player_config_menu_initialization_2v2(self, pygame_init):
        """Test that PlayerConfigMenu initializes correctly for 2v2 mode."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="2v2")
        
        assert menu.num_players == 4
        assert len(menu.player_configs) == 4
        
        # Player 1 should be human by default
        assert menu.player_configs[0]['type'] == 'human'
        assert menu.player_configs[0]['bot_type'] is None
        
        # Players 2-4 should be computer (SimpleBot) by default
        for i in range(1, 4):
            assert menu.player_configs[i]['type'] == 'computer'
            assert menu.player_configs[i]['bot_type'] == 'SimpleBot'

    def test_player_config_toggle_type(self, pygame_init):
        """Test toggling player type between human and computer."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        # Draw menu to populate interactive elements
        menu.draw()
        
        # Find the type toggle button for player 1
        type_toggle_elements = [e for e in menu.interactive_elements 
                                if e['type'] == 'type_toggle' and e['player_idx'] == 0]
        
        assert len(type_toggle_elements) > 0, "Should have type toggle button for player 1"
        
        # Simulate click on player 1's type toggle
        element = type_toggle_elements[0]
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {'button': 1, 'pos': element['rect'].center}
        )
        
        # Player 1 starts as human
        assert menu.player_configs[0]['type'] == 'human'
        
        # Toggle to computer
        menu.handle_input(event)
        assert menu.player_configs[0]['type'] == 'computer'
        assert menu.player_configs[0]['bot_type'] == 'SimpleBot'
        
        # Toggle back to human
        menu.handle_input(event)
        assert menu.player_configs[0]['type'] == 'human'
        assert menu.player_configs[0]['bot_type'] is None

    def test_player_config_result_structure(self, pygame_init):
        """Test that the result has the correct structure."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        result = menu._get_result()
        
        assert 'players' in result
        assert isinstance(result['players'], list)
        assert len(result['players']) == 2
        
        # Check each player config structure
        for config in result['players']:
            assert 'type' in config
            assert config['type'] in ['human', 'computer']
            assert 'bot_type' in config
            if config['type'] == 'computer':
                assert config['bot_type'] in ['SimpleBot', 'NormalBot', 'HardBot', None]
            else:
                assert config['bot_type'] is None

    def test_player_config_start_game_button(self, pygame_init):
        """Test that start game button returns configuration."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        # Draw menu to populate interactive elements
        menu.draw()
        
        # Find the start button
        start_button_elements = [e for e in menu.interactive_elements 
                                 if e['type'] == 'start_button']
        
        assert len(start_button_elements) > 0, "Should have start game button"
        
        # Simulate click on start button
        element = start_button_elements[0]
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {'button': 1, 'pos': element['rect'].center}
        )
        
        result = menu.handle_input(event)
        
        assert result is not None
        assert 'players' in result
        assert len(result['players']) == 2

    def test_player_config_back_button(self, pygame_init):
        """Test that back button cancels and returns None."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        # Draw menu to populate interactive elements
        menu.draw()
        
        # Find the back button
        back_button_elements = [e for e in menu.interactive_elements 
                                if e['type'] == 'back_button']
        
        assert len(back_button_elements) > 0, "Should have back button"
        
        # Simulate click on back button
        element = back_button_elements[0]
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {'button': 1, 'pos': element['rect'].center}
        )
        
        result = menu.handle_input(event)
        
        assert result is None
        assert not menu.running

    def test_player_config_keyboard_escape(self, pygame_init):
        """Test that ESC key cancels the menu."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_ESCAPE})
        result = menu.handle_input(event)
        
        assert result is None
        assert not menu.running

    def test_player_config_keyboard_enter(self, pygame_init):
        """Test that ENTER key starts the game with current configuration."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RETURN})
        result = menu.handle_input(event)
        
        assert result is not None
        assert 'players' in result
        assert len(result['players']) == 2

    def test_player_config_all_players_can_be_human(self, pygame_init):
        """Test that all players can be set to human."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        # Set both players to human
        menu.player_configs[0]['type'] = 'human'
        menu.player_configs[0]['bot_type'] = None
        menu.player_configs[1]['type'] = 'human'
        menu.player_configs[1]['bot_type'] = None
        
        result = menu._get_result()
        
        # Verify all players are human
        for config in result['players']:
            assert config['type'] == 'human'
            assert config['bot_type'] is None

    def test_player_config_all_players_can_be_computer(self, pygame_init):
        """Test that all players can be set to computer."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        menu = PlayerConfigMenu(game_mode="1v1")
        
        # Set both players to computer
        menu.player_configs[0]['type'] = 'computer'
        menu.player_configs[0]['bot_type'] = 'SimpleBot'
        menu.player_configs[1]['type'] = 'computer'
        menu.player_configs[1]['bot_type'] = 'SimpleBot'
        
        result = menu._get_result()
        
        # Verify all players are computer
        for config in result['players']:
            assert config['type'] == 'computer'
            assert config['bot_type'] == 'SimpleBot'

    def test_player_config_invalid_game_mode(self, pygame_init):
        """Test that invalid game mode raises ValueError."""
        from reinforcetactics.ui.menus import PlayerConfigMenu
        
        with pytest.raises(ValueError) as excinfo:
            PlayerConfigMenu(game_mode="3v3")
        
        assert "Invalid game_mode" in str(excinfo.value)
        assert "Must be '1v1' or '2v2'" in str(excinfo.value)


class TestUnitPurchaseMenu:
    """Test unit purchase menu."""

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state for testing."""
        # Create a simple 10x10 map with an HQ at (5, 5)
        map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[5][5] = 'h_1'  # HQ owned by player 1
        map_data[6][6] = 'b_1'  # Building owned by player 1
        
        from reinforcetactics.core.game_state import GameState
        game = GameState(map_data, num_players=2)
        game.current_player = 1
        game.player_gold[1] = 300  # Enough to buy any unit
        return game

    @pytest.fixture
    def poor_game_state(self):
        """Create a mock game state with low gold."""
        # Create a simple 10x10 map with an HQ at (5, 5)
        map_data = np.array([['p' for _ in range(10)] for _ in range(10)], dtype=object)
        map_data[5][5] = 'h_1'  # HQ owned by player 1
        
        from reinforcetactics.core.game_state import GameState
        game = GameState(map_data, num_players=2)
        game.current_player = 1
        game.player_gold[1] = 50  # Not enough to buy any unit
        return game

    def test_unit_purchase_menu_initialization(self, pygame_init, mock_game_state):
        """Test that UnitPurchaseMenu initializes correctly."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, mock_game_state, (5, 5))
        
        assert menu.game_state == mock_game_state
        assert menu.building_pos == (5, 5)
        assert menu.unit_types == ['W', 'M', 'C']
        assert hasattr(menu, 'menu_rect')
        assert isinstance(menu.interactive_elements, list)

    def test_unit_purchase_menu_affordability(self, pygame_init, poor_game_state):
        """Test that units are disabled when player cannot afford them."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, poor_game_state, (5, 5))
        
        # Draw the menu to populate interactive elements
        menu.draw(screen)
        
        # All unit buttons should be disabled
        unit_buttons = [el for el in menu.interactive_elements if el['type'] == 'unit_button']
        assert len(unit_buttons) == 3  # W, M, C
        
        for button in unit_buttons:
            assert button['disabled'] is True

    def test_unit_purchase_menu_close_button(self, pygame_init, mock_game_state):
        """Test that close button is present and functional."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, mock_game_state, (5, 5))
        
        # Draw the menu to populate interactive elements
        menu.draw(screen)
        
        # Check for close button
        close_buttons = [el for el in menu.interactive_elements if el['type'] == 'close_button']
        assert len(close_buttons) == 1
        
        # Simulate clicking the close button
        close_button = close_buttons[0]
        result = menu.handle_click(close_button['rect'].center)
        assert result is not None
        assert result['type'] == 'close'

    def test_unit_purchase_menu_click_outside_closes(self, pygame_init, mock_game_state):
        """Test that clicking outside menu closes it."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, mock_game_state, (5, 5))
        
        # Click far outside the menu
        result = menu.handle_click((10, 10))
        assert result is not None
        assert result['type'] == 'close'

    def test_unit_purchase_creates_unit(self, pygame_init, mock_game_state):
        """Test that purchasing a unit creates it in the game state."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, mock_game_state, (5, 5))
        
        # Draw the menu to populate interactive elements
        menu.draw(screen)
        
        # Find an affordable unit button (Warrior)
        unit_buttons = [el for el in menu.interactive_elements 
                       if el['type'] == 'unit_button' and not el['disabled']]
        assert len(unit_buttons) > 0
        
        # Click the first affordable unit button
        warrior_button = unit_buttons[0]
        initial_gold = mock_game_state.player_gold[1]
        result = menu.handle_click(warrior_button['rect'].center)
        
        assert result is not None
        assert result['type'] == 'unit_created'
        assert result['unit'] is not None
        
        # Check that unit was created at the building position
        created_unit = mock_game_state.get_unit_at_position(5, 5)
        assert created_unit is not None
        assert created_unit.player == 1
        
        # Check that gold was deducted
        assert mock_game_state.player_gold[1] < initial_gold

    def test_unit_purchase_menu_disabled_button_not_clickable(self, pygame_init, poor_game_state):
        """Test that disabled unit buttons don't create units."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, poor_game_state, (5, 5))
        
        # Draw the menu to populate interactive elements
        menu.draw(screen)
        
        # Find a disabled unit button
        unit_buttons = [el for el in menu.interactive_elements 
                       if el['type'] == 'unit_button' and el['disabled']]
        assert len(unit_buttons) > 0
        
        # Try to click a disabled button
        disabled_button = unit_buttons[0]
        result = menu.handle_click(disabled_button['rect'].center)
        
        # Should return None (no action taken)
        assert result is None
        
        # No unit should be created
        created_unit = poor_game_state.get_unit_at_position(5, 5)
        assert created_unit is None

    def test_unit_purchase_menu_hover_effects(self, pygame_init, mock_game_state):
        """Test that hovering updates hover state."""
        from reinforcetactics.ui.menus import UnitPurchaseMenu
        
        screen = pygame.display.set_mode((640, 640))
        menu = UnitPurchaseMenu(screen, mock_game_state, (5, 5))
        
        # Draw the menu to populate interactive elements
        menu.draw(screen)
        
        # Initially no hover
        assert menu.hover_element is None
        
        # Hover over a button
        unit_buttons = [el for el in menu.interactive_elements 
                       if el['type'] == 'unit_button']
        if unit_buttons:
            button = unit_buttons[0]
            menu.handle_mouse_motion(button['rect'].center)
            assert menu.hover_element is not None
            assert menu.hover_element['type'] == 'unit_button'
