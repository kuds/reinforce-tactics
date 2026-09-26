"""Every game mode the New Game flow offers must be startable (review §1.4).

GameModeMenu used to list every ``maps/`` subfolder, so it offered the bundled
``1v1v1`` folder; PlayerConfigMenu then raised ``ValueError`` for anything but
1v1/2v2 and the uncaught exception closed the app. These tests drive the menus
headlessly (SDL dummy driver) and check that a 3-player game on a
``maps/1v1v1`` map starts and cycles through all three players' turns.
"""

from pathlib import Path

import pygame
import pytest

from reinforcetactics.ui.menus import GameModeMenu, MainMenu, PlayerConfigMenu
from reinforcetactics.ui.menus import main_menu as main_menu_module
from reinforcetactics.utils import settings as settings_module

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = REPO_ROOT / "maps"
THREE_PLAYER_MAP = MAPS_DIR / "1v1v1" / "triangle_arena.csv"
EXPECTED_SEATS = {"1v1": 2, "1v1v1": 3, "2v2": 4}


@pytest.fixture
def pygame_init():
    """Initialize pygame for tests."""
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """A settings instance backed by a temp file, so tests never write the repo's settings.json."""
    monkeypatch.setattr(settings_module, "_settings_instance", settings_module.Settings(str(tmp_path / "settings.json")))


def _space_event():
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0, unicode=" ", scancode=0)


class TestOfferedModesAreStartable:
    def test_bundled_modes_are_offered(self, pygame_init):
        menu = GameModeMenu(maps_dir=str(MAPS_DIR))
        assert menu.available_modes == ["1v1", "1v1v1", "2v2"]

    @pytest.mark.parametrize("mode", ["1v1", "1v1v1", "2v2"])
    def test_every_offered_mode_has_a_player_config_screen(self, pygame_init, isolated_settings, mode):
        assert mode in GameModeMenu(maps_dir=str(MAPS_DIR)).available_modes

        menu = PlayerConfigMenu(game_mode=mode)

        assert menu.num_players == EXPECTED_SEATS[mode]
        assert len(menu.player_configs) == EXPECTED_SEATS[mode]

    def test_folders_that_are_not_modes_are_not_offered(self, pygame_init, tmp_path):
        for folder in ("1v1", "scenarios", "3v3"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "map.csv").write_text("p,p\np,p\n")

        menu = GameModeMenu(maps_dir=str(tmp_path))

        assert menu.available_modes == ["1v1"]
        assert [text for text, _ in menu.options] == ["1v1", "Back"]


class TestThreePlayerConfig:
    def test_1v1v1_builds_a_three_player_config(self, pygame_init, isolated_settings):
        menu = PlayerConfigMenu(game_mode="1v1v1")

        assert menu.num_players == 3
        assert [c["type"] for c in menu.player_configs] == ["human", "computer", "computer"]
        assert [c["bot_type"] for c in menu.player_configs] == [None, "SimpleBot", "SimpleBot"]

        result = menu._get_result()
        assert result is not None
        assert len(result["players"]) == 3
        assert result["num_players"] == 3

    @pytest.mark.parametrize("mode", ["1v1v1", "2v2"])
    def test_every_row_and_button_fits_the_window(self, pygame_init, isolated_settings, mode):
        menu = PlayerConfigMenu(game_mode=mode)
        menu.draw()

        toggles = [e for e in menu.interactive_elements if e["type"] == "type_toggle"]
        assert sorted(e["player_idx"] for e in toggles) == list(range(menu.num_players))

        screen_rect = menu.screen.get_rect()
        for element in menu.interactive_elements:
            assert screen_rect.contains(element["rect"]), f"{element['type']} is drawn off screen"

    def test_start_button_returns_the_three_player_config(self, pygame_init, isolated_settings):
        menu = PlayerConfigMenu(game_mode="1v1v1")
        menu.draw()
        start = next(e for e in menu.interactive_elements if e["type"] == "start_button")

        result = menu.handle_input(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": start["rect"].center}))

        assert result is not None
        assert len(result["players"]) == 3


class _FakeMapMenu:
    def __init__(self, screen=None, maps_dir="maps", game_mode=None):
        self.game_mode = game_mode

    def run(self):
        return str(THREE_PLAYER_MAP)


def test_main_menu_new_game_flow_passes_the_seat_count(pygame_init, isolated_settings, monkeypatch):
    """New Game -> 1v1v1 -> map -> Start must hand the game loop a 3-seat game."""
    monkeypatch.setattr(main_menu_module.GameModeMenu, "run", lambda self: "1v1v1")
    monkeypatch.setattr(main_menu_module, "MapSelectionMenu", _FakeMapMenu)
    monkeypatch.setattr(main_menu_module.PlayerConfigMenu, "run", lambda self: self._get_result())

    result = MainMenu()._new_game()

    assert result is not None
    assert result["type"] == "new_game"
    assert result["mode"] == "1v1v1"
    assert result["num_players"] == 3
    assert len(result["players"]) == 3


def test_three_player_game_starts_and_cycles_through_every_seat(pygame_init, isolated_settings, monkeypatch):
    """A 1v1v1 map game starts with 3 seats and each SPACE runs both bots' turns."""
    from reinforcetactics.app import game_loop

    observed = {}

    def fake_run(session):
        game = session.game
        observed["num_players"] = game.num_players
        observed["gold_seats"] = sorted(game.player_gold)
        observed["hq_owners"] = sorted({t.player for row in game.grid.tiles for t in row if t.type == "h"})
        turns = []
        for _ in range(2):
            session.input_handler.handle_keyboard_event(_space_event())
            turns.append((game.current_player, game.turn_number))
            session._render_frame()
        observed["turns"] = turns
        observed["players_who_acted"] = sorted({a.get("player") for a in game.action_history})
        return "main_menu"

    monkeypatch.setattr(game_loop.GameSession, "run", fake_run)

    result = game_loop.start_new_game(
        mode="1v1v1",
        selected_map=str(THREE_PLAYER_MAP),
        player_configs=[
            {"type": "human", "bot_type": None},
            {"type": "computer", "bot_type": "SimpleBot"},
            {"type": "computer", "bot_type": "SimpleBot"},
        ],
        num_players=3,
    )

    assert result == "main_menu"
    assert observed["num_players"] == 3
    assert observed["gold_seats"] == [1, 2, 3]
    assert observed["hq_owners"] == [1, 2, 3]
    # Each SPACE ends the human's turn, both bots play theirs, and it is
    # player 1's turn again one full round later.
    assert observed["turns"] == [(1, 1), (1, 2)]
    assert observed["players_who_acted"] == [1, 2, 3]


def test_2v2_still_starts_with_four_seats(pygame_init, isolated_settings, monkeypatch):
    """2v2 team rules are out of scope, but it must keep starting as before."""
    from reinforcetactics.app import game_loop

    observed = {}

    def fake_run(session):
        observed["num_players"] = session.game.num_players
        return "main_menu"

    monkeypatch.setattr(game_loop.GameSession, "run", fake_run)
    menu = PlayerConfigMenu(game_mode="2v2")
    config = menu._get_result()

    result = game_loop.start_new_game(
        mode="2v2",
        selected_map=str(MAPS_DIR / "2v2" / "beginner.csv"),
        player_configs=config["players"],
        num_players=config["num_players"],
    )

    assert result == "main_menu"
    assert observed["num_players"] == 4


def test_map_editor_files_a_three_player_map_under_1v1v1(pygame_init, isolated_settings, tmp_path, monkeypatch):
    """New 3-player maps used to be saved into maps/2v2 and get four seats."""
    import numpy as np
    import pandas as pd

    from reinforcetactics.ui.menus.map_editor.map_editor import MapEditor

    monkeypatch.chdir(tmp_path)
    grid = np.full((20, 20), "p", dtype=object)
    grid[1, 1], grid[1, 18], grid[18, 9] = "h_1", "h_2", "h_3"
    grid[2, 1], grid[2, 18], grid[17, 9] = "b_1", "b_2", "b_3"
    editor = MapEditor(pygame.display.set_mode((900, 700)), pd.DataFrame(grid), None, 3)

    assert editor._save_map()
    assert Path(editor.map_filename).parent == Path("maps/1v1v1")
