"""Tests for menu layout, scrolling and navigation behaviour.

These cover the guarantees that are easy to regress by hand-tuning a
constant: every option stays inside the window (however small the window
is), the click targets match what was drawn, and the keyboard can reach
every control.
"""

import pygame
import pytest

from reinforcetactics.ui import theme
from reinforcetactics.ui.menus.base import Menu
from reinforcetactics.ui.menus.game_setup.map_selection_menu import MapSelectionMenu
from reinforcetactics.ui.menus.in_game.game_over_menu import GameOverMenu
from reinforcetactics.ui.menus.in_game.pause_menu import PauseMenu
from reinforcetactics.ui.menus.save_load.load_game_menu import LoadGameMenu
from reinforcetactics.ui.menus.save_load.replay_selection_menu import ReplaySelectionMenu
from reinforcetactics.ui.menus.settings.units_menu import UnitsMenu

# The in-game screen is sized to the map (grid * TILE_SIZE), so menus opened
# over it can be far smaller than the 900x700 stand-alone menu window.
# maps/1v1/beginner.csv is 6x6 and loads with a 2-tile border -> 320x320.
SMALL_GAME_SCREEN = (320, 320)
MEDIUM_GAME_SCREEN = (448, 448)
MENU_SCREEN = (900, 700)


@pytest.fixture
def pygame_init():
    """Initialize pygame for tests."""
    pygame.init()
    yield
    pygame.quit()


def _menu_with(screen, labels, title="Test Menu"):
    menu = Menu(screen, title)
    for label in labels:
        menu.add_option(label, lambda: None)
    return menu


class TestOptionsFitTheWindow:
    """Options must never be drawn past the edge of the window."""

    @pytest.mark.parametrize("size", [SMALL_GAME_SCREEN, MEDIUM_GAME_SCREEN, MENU_SCREEN])
    @pytest.mark.parametrize("count", [1, 4, 5, 8, 14])
    def test_no_option_falls_below_the_bottom_edge(self, pygame_init, size, count):
        screen = pygame.display.set_mode(size)
        menu = _menu_with(screen, [f"Option {i}" for i in range(count)])

        for _, _, rect in menu._layout_visible_options():
            assert rect.bottom <= size[1], f"option drawn past the bottom of a {size[0]}x{size[1]} window"
            assert rect.top >= 0

    @pytest.mark.parametrize("size", [SMALL_GAME_SCREEN, MENU_SCREEN])
    def test_rows_stay_within_the_window_width(self, pygame_init, size):
        screen = pygame.display.set_mode(size)
        menu = _menu_with(screen, ["Short", "An extremely long option label that will not fit a narrow window at all"])

        for _, _, rect in menu._layout_visible_options():
            assert rect.left >= 0
            assert rect.right <= size[0]

    def test_visible_count_shrinks_with_the_window(self, pygame_init):
        labels = [f"Option {i}" for i in range(14)]

        screen = pygame.display.set_mode(MENU_SCREEN)
        big = _menu_with(screen, labels)
        big._layout_visible_options()

        screen = pygame.display.set_mode(SMALL_GAME_SCREEN)
        small = _menu_with(screen, labels)
        small._layout_visible_options()

        assert big.max_visible_options == theme.MAX_VISIBLE_OPTIONS
        assert 1 <= small.max_visible_options < big.max_visible_options

    def test_every_option_is_reachable_by_scrolling(self, pygame_init):
        """A short window must scroll, not silently drop the tail."""
        screen = pygame.display.set_mode(SMALL_GAME_SCREEN)
        menu = _menu_with(screen, [f"Option {i}" for i in range(5)])
        menu._layout_visible_options()

        seen = set()
        for offset in range(len(menu.options)):
            menu.scroll_offset = offset
            seen.update(i for i, _, _ in menu._layout_visible_options())

        assert seen == set(range(5))

    def test_scroll_hint_and_readout_stay_on_screen(self, pygame_init):
        """The 'Scroll Down' hint used to render below the window bottom."""
        screen = pygame.display.set_mode(MENU_SCREEN)
        menu = _menu_with(screen, [f"Option {i}" for i in range(14)])
        menu._draw_content()

        # The hint is drawn just under the last row; that has to leave room.
        last_row = menu.option_rects[-1]
        assert last_row.bottom + 6 + 20 <= MENU_SCREEN[1]


class TestRealMenusFit:
    """The concrete screens that used to overflow."""

    def test_pause_menu_fits_a_small_game_window(self, pygame_init):
        screen = pygame.display.set_mode(SMALL_GAME_SCREEN)
        menu = PauseMenu(screen, game=None)

        for _, _, rect in menu._layout_visible_options():
            assert rect.bottom <= SMALL_GAME_SCREEN[1]
        assert menu.max_visible_options >= 1

    def test_units_menu_shows_no_clipped_row(self, pygame_init):
        screen = pygame.display.set_mode(MENU_SCREEN)
        menu = UnitsMenu(screen)

        assert len(menu.options) > menu.max_visible_options, "expected this menu to need scrolling"
        for _, _, rect in menu._layout_visible_options():
            assert rect.bottom <= MENU_SCREEN[1]

    def test_game_over_options_clear_the_winner_banner(self, pygame_init):
        """The banner is painted over the list, so the list must start below it."""

        class _FakeState:
            turn_number = 12

        for size in (SMALL_GAME_SCREEN, MENU_SCREEN):
            screen = pygame.display.set_mode(size)
            menu = GameOverMenu(winner=1, game_state=_FakeState(), screen=screen)
            banner_bottom = theme.TITLE_MARGIN_Y + menu.title_font.get_height() * 2

            layout = menu._layout_visible_options()
            assert layout, "at least one option must be visible"
            assert layout[0][2].top >= banner_bottom
            for _, _, rect in layout:
                assert rect.bottom <= size[1]


class TestClickTargetsMatchTheDrawing:
    """``_populate_option_rects`` and ``draw`` must agree on the geometry."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda screen: MapSelectionMenu(screen, maps_dir="maps", game_mode="1v1"),
            lambda screen: LoadGameMenu(screen),
            lambda screen: ReplaySelectionMenu(screen),
        ],
        ids=["map_selection", "load_game", "replay_selection"],
    )
    def test_split_panel_hit_boxes_match_drawn_rows(self, pygame_init, factory):
        screen = pygame.display.set_mode(MENU_SCREEN)
        menu = factory(screen)

        # Before the first frame: this is what run() hit-tests against.
        menu._populate_option_rects()
        pre_draw = list(menu.option_rects)

        menu.draw()
        drawn = list(menu.option_rects)

        assert pre_draw == drawn, "click targets differ from the rows actually drawn"

    def test_split_panel_rows_stay_inside_their_panel(self, pygame_init):
        screen = pygame.display.set_mode(MENU_SCREEN)
        menu = MapSelectionMenu(screen, maps_dir="maps", game_mode="1v1")
        menu.draw()

        left_panel, _ = menu._panels()
        for rect in menu.option_rects:
            assert left_panel.contains(rect)


class TestPlayerConfigKeyboard:
    """The player setup screen used to be reachable by mouse only."""

    @staticmethod
    def _key(key):
        return pygame.event.Event(pygame.KEYDOWN, {"key": key})

    def _menu(self):
        from reinforcetactics.ui.menus.game_setup.player_config_menu import PlayerConfigMenu

        screen = pygame.display.set_mode(MENU_SCREEN)
        menu = PlayerConfigMenu(screen, game_mode="1v1")
        menu.draw()  # populates interactive_elements
        return menu

    def test_tab_focuses_the_first_control(self, pygame_init):
        menu = self._menu()
        assert menu.focus_index == -1

        menu.handle_input(self._key(pygame.K_TAB))
        assert menu.focus_index == 0

    def test_focus_wraps_around(self, pygame_init):
        menu = self._menu()
        count = len(menu.interactive_elements)

        menu.handle_input(self._key(pygame.K_UP))
        assert menu.focus_index == count - 1

        menu.handle_input(self._key(pygame.K_DOWN))
        assert menu.focus_index == 0

    def test_enter_activates_the_focused_control(self, pygame_init):
        menu = self._menu()
        menu.handle_input(self._key(pygame.K_TAB))  # player 1 type toggle
        assert menu.interactive_elements[menu.focus_index]["type"] == "type_toggle"

        assert menu.player_configs[0]["type"] == "human"
        menu.handle_input(self._key(pygame.K_RETURN))
        assert menu.player_configs[0]["type"] == "computer"

    def test_enter_without_focus_still_starts_the_game(self, pygame_init):
        menu = self._menu()
        result = menu.handle_input(self._key(pygame.K_RETURN))
        assert result is not None
        assert "players" in result

    def test_blocked_start_reports_a_reason_on_screen(self, pygame_init):
        menu = self._menu()
        menu.player_configs[1].update(type="computer", bot_type="ModelBot", model_path=None)

        reason = menu._blocking_reason()
        assert reason and "Player 2" in reason
        assert menu._get_result() is None

        # Pressing Enter surfaces it instead of failing silently.
        menu.focus_index = -1
        assert menu.handle_input(self._key(pygame.K_RETURN)) is None

    def test_row_buttons_never_overlap(self, pygame_init):
        """Buttons are label-sized, so the row must flow, not use fixed x."""
        menu = self._menu()
        # The widest configuration: Computer + Custom Model + Browse...
        menu.player_configs[1].update(type="computer", bot_type="ModelBot", model_path=None)
        menu.draw()

        row = [e["rect"] for e in menu.interactive_elements if e["player_idx"] == 1]
        assert len(row) == 3, "expected type, bot-type and browse buttons"
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                assert not row[i].colliderect(row[j]), "player row buttons overlap"

    def test_row_buttons_stay_inside_the_window(self, pygame_init):
        menu = self._menu()
        menu.player_configs[1].update(type="computer", bot_type="ModelBot", model_path=None)
        menu.draw()

        for element in menu.interactive_elements:
            assert element["rect"].right <= MENU_SCREEN[0]
            assert element["rect"].bottom <= MENU_SCREEN[1]

    def test_disabled_start_button_is_still_focusable(self, pygame_init):
        """So the player can ask *why* it is disabled instead of guessing."""
        menu = self._menu()
        menu.player_configs[1].update(type="computer", bot_type="ModelBot", model_path=None)
        menu.draw()

        start = [e for e in menu.interactive_elements if e["type"] == "start_button"]
        assert len(start) == 1
        assert start[0]["disabled"] is True

        menu.handle_input(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": start[0]["rect"].center}))
        assert menu.status_message and "Player 2" in menu.status_message


class TestDrainEvents:
    """Flushing leftover input must not swallow a window-close request."""

    def test_quit_survives_the_flush(self, pygame_init):
        from reinforcetactics.ui.menus.base import drain_events

        pygame.display.set_mode(MENU_SCREEN)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (0, 0)}))
        pygame.event.post(pygame.event.Event(pygame.QUIT))

        drain_events()

        remaining = pygame.event.get()
        assert [e.type for e in remaining] == [pygame.QUIT]

    def test_other_events_are_dropped(self, pygame_init):
        from reinforcetactics.ui.menus.base import drain_events

        pygame.display.set_mode(MENU_SCREEN)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (0, 0)}))

        drain_events()

        assert pygame.event.get() == []

    def test_submenu_reposts_quit_for_its_parent(self, pygame_init):
        """A menu drawn on a borrowed screen propagates the close request."""
        screen = pygame.display.set_mode(MENU_SCREEN)
        pygame.event.clear()

        child = Menu(screen, "Child")  # screen provided -> owns_screen False
        assert child.owns_screen is False
        stop, value = child._on_quit_event()

        assert (stop, value) == (True, None)
        assert [e.type for e in pygame.event.get()] == [pygame.QUIT]


class TestResultHook:
    """The base loop's ``_on_result`` hook replaces per-menu run() copies."""

    def test_absorbed_result_keeps_the_menu_running(self, pygame_init):
        screen = pygame.display.set_mode(MENU_SCREEN)

        class _ToggleMenu(Menu):
            def _on_result(self, result):
                return (False, None) if result == "toggled" else (True, result)

        menu = _ToggleMenu(screen, "Toggles")
        menu.add_option("Toggle", lambda: "toggled")
        menu.add_option("Done", lambda: "done")

        assert menu._on_result("toggled") == (False, None)
        assert menu._on_result("done") == (True, "done")

    def test_units_menu_absorbs_its_sentinels(self, pygame_init):
        screen = pygame.display.set_mode(MENU_SCREEN)
        menu = UnitsMenu(screen)

        for sentinel in ("toggled", "separator", "cannot_disable_last"):
            assert menu._on_result(sentinel) == (False, None)
        assert menu._on_result(None) == (True, None)
