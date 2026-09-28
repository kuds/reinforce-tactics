"""The in-game HUD never covers or intercepts a board tile.

The window used to be exactly the board, with the gold and turn labels and
the End Turn and Resign buttons drawn over its top rows. On a 20x20 map
(padded with a 2-tile ocean border) Resign covered playable tiles in row 2,
and since the input handler checks the buttons before the grid, clicking a
unit under it opened the resign dialog. The HUD is now a panel to the right
of the board.
"""

from pathlib import Path

import numpy as np
import pygame
import pytest

from reinforcetactics.app import game_loop
from reinforcetactics.app import input_handler as input_handler_module
from reinforcetactics.app.input_handler import InputHandler
from reinforcetactics.constants import TILE_SIZE
from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui import theme
from reinforcetactics.ui.assets import PLAYER_COLORS
from reinforcetactics.ui.renderer import Renderer
from reinforcetactics.utils import settings as settings_module
from reinforcetactics.utils.file_io import FileIO

REPO_ROOT = Path(__file__).resolve().parents[1]
MOUNTAIN_SNIPERS = REPO_ROOT / "maps" / "1v1" / "mountain_snipers.csv"


@pytest.fixture
def display(tmp_path, monkeypatch):
    """A dummy SDL display, with settings kept out of the working tree."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, "_settings_instance", settings_module.Settings(str(tmp_path / "settings.json")))
    pygame.init()
    yield
    pygame.quit()


def _mountain_snipers(fog_of_war=False):
    """The map as the GUI loads it, and the (x, y) of every tile of the map file."""
    meta = FileIO.load_map_with_metadata(str(MOUNTAIN_SNIPERS), for_ui=True)
    game = GameState(meta["map_data"], num_players=2, fog_of_war=fog_of_war)
    ox, oy = meta["padding_offset_x"], meta["padding_offset_y"]
    playable = [(ox + x, oy + y) for y in range(meta["original_height"]) for x in range(meta["original_width"])]
    return game, playable


def _tile_rect(x, y):
    return pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)


def _hud_rects(renderer):
    return {
        "panel": renderer.hud_rect,
        "player": renderer._hud_player_card,
        "end_turn": renderer.end_turn_button,
        "resign": renderer.resign_button,
    }


class TestHudLayout:
    def test_the_map_leaves_playable_tiles_in_the_top_rows(self, display):
        # The 2-tile border is all that separates the map from the window's
        # top edge, where the HUD used to be drawn.
        _, playable = _mountain_snipers()
        assert min(y for _, y in playable) == 2

    def test_no_hud_element_overlaps_the_board(self, display):
        game, playable = _mountain_snipers(fog_of_war=True)
        renderer = Renderer(game)

        assert renderer.board_rect == pygame.Rect(0, 0, game.grid.width * TILE_SIZE, game.grid.height * TILE_SIZE)
        assert renderer.screen.get_size() == renderer.window_size
        for name, rect in _hud_rects(renderer).items():
            assert rect.width and rect.height, name
            assert not rect.colliderect(renderer.board_rect), name
            assert renderer.screen.get_rect().contains(rect), name
            assert not any(rect.colliderect(_tile_rect(x, y)) for x, y in playable), name
        assert renderer.hud_rect.contains(renderer.end_turn_button)
        assert renderer.hud_rect.contains(renderer.resign_button)

    def test_the_hud_draws_nothing_on_the_board(self, display):
        game, _ = _mountain_snipers(fog_of_war=True)
        game.place_unit("A", 20, 2, player=1)  # where Resign used to be drawn
        game.place_unit("W", 3, 2, player=1)  # where the turn label used to be drawn
        in_play = Renderer(game, pixel_art=False)
        board_only = Renderer(game, headless=True, pixel_art=False)

        in_play.render()
        board_only.render()

        board = in_play.board_rect
        drawn = pygame.surfarray.array3d(in_play.screen.subsurface(board))
        assert np.array_equal(drawn, pygame.surfarray.array3d(board_only.screen))
        # ...and the panel is drawn beside it.
        assert in_play.screen.get_at(in_play.hud_rect.center)[:3] == theme.HUD_PANEL_BG

    def test_every_playable_tile_click_reaches_the_grid(self, display, monkeypatch):
        game, playable = _mountain_snipers()
        renderer = Renderer(game)
        handler = InputHandler(game, renderer, bots={}, num_players=2)

        def not_a_hud_click(*args, **kwargs):
            raise AssertionError("a click on a board tile reached a HUD button")

        # The resign dialog would block waiting for input; either button
        # firing fails the test instead.
        monkeypatch.setattr(input_handler_module, "ConfirmationDialog", not_a_hud_click)
        monkeypatch.setattr(game, "end_turn", not_a_hud_click)
        reached = []
        monkeypatch.setattr(handler, "_handle_grid_click", lambda pos, now: reached.append(pos) or "continue")

        corners = [(0, 0), (TILE_SIZE - 1, 0), (0, TILE_SIZE - 1), (TILE_SIZE - 1, TILE_SIZE - 1)]
        clicks = [(x * TILE_SIZE + dx, y * TILE_SIZE + dy) for x, y in playable for dx, dy in corners]
        for pos in clicks:
            assert handler.handle_mouse_click(pos) == "continue"

        assert reached == clicks

    def test_the_buttons_still_work_from_the_panel(self, display, monkeypatch):
        game, _ = _mountain_snipers()
        renderer = Renderer(game)
        handler = InputHandler(game, renderer, bots={}, num_players=2)
        dialogs = []

        class Dialog:
            def __init__(self, *args, **kwargs):
                dialogs.append(args)

            def run(self):
                return False  # cancelled

        monkeypatch.setattr(input_handler_module, "ConfirmationDialog", Dialog)

        handler.handle_mouse_click(renderer.end_turn_button.center)
        assert game.current_player == 2

        handler.handle_mouse_click(renderer.resign_button.center)
        assert len(dialogs) == 1 and not game.game_over

    def test_space_still_ends_the_turn(self, display):
        game, _ = _mountain_snipers()
        handler = InputHandler(game, Renderer(game), bots={}, num_players=2)

        handler.handle_keyboard_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        assert game.current_player == 2

    def test_a_short_board_gets_a_window_tall_enough_for_the_panel(self, display):
        map_data = np.array([["p"] * 6 for _ in range(6)], dtype=object)
        map_data[0][0], map_data[5][5] = "h_1", "h_2"
        game = GameState(map_data, num_players=2, fog_of_war=True)
        renderer = Renderer(game)

        width, height = renderer.screen.get_size()
        assert (width, height) == (6 * TILE_SIZE + theme.HUD_PANEL_WIDTH, renderer.hud_rect.height)
        assert height > 6 * TILE_SIZE
        fow_bottom = renderer._hud_fow_pos[1] + renderer._fow_label.get_height() + 4
        assert renderer.end_turn_button.top > fow_bottom
        assert renderer.resign_button.bottom <= height
        renderer.render()

        # Below the board is not the board.
        handler = InputHandler(game, renderer, bots={}, num_players=2)
        assert handler.handle_mouse_click((10, height - 5)) is None

    def test_saving_keeps_the_window_size(self, display, monkeypatch):
        # Saving with S used to reopen the display at 900x700, which cut off
        # End Turn and Resign at the bottom of the panel.
        game, _ = _mountain_snipers()
        renderer = Renderer(game)
        session = game_loop.GameSession(game, renderer, bots={}, num_players=2)
        monkeypatch.setattr(game_loop.SaveGameMenu, "run", lambda self: None)

        session._handle_save_game()

        assert pygame.display.get_surface().get_size() == renderer.window_size

    @pytest.mark.parametrize("kwargs", [{"headless": True}, {"replay_mode": True}])
    def test_replays_and_headless_capture_keep_a_board_sized_window(self, display, kwargs):
        game, _ = _mountain_snipers()
        renderer = Renderer(game, **kwargs)

        assert renderer.screen.get_size() == renderer.board_rect.size
        assert renderer.end_turn_button.width == renderer.resign_button.width == 0
        renderer.render()


class TestHudText:
    def test_gold_is_written_as_the_shop_writes_prices(self, display):
        game, _ = _mountain_snipers()
        renderer = Renderer(game)

        assert renderer._gold_text() == f"{game.player_gold[1]}g"

    @staticmethod
    def _contrast(a, b):
        def luminance(rgb):
            channels = [c / 255 for c in rgb]
            linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        high, low = sorted((luminance(a), luminance(b)), reverse=True)
        return (high + 0.05) / (low + 0.05)

    def test_hud_text_is_legible_whatever_the_player_colour(self):
        # Text sits on dark fills; the player colour is only an accent. Gold
        # text on the player's colour measured 1.03:1 on green.
        for text in (theme.HUD_GOLD_TEXT, theme.TEXT):
            for background in (theme.HUD_PANEL_BG, theme.HUD_CARD_BG):
                assert self._contrast(text, background) >= 4.5
        assert self._contrast(theme.HUD_LABEL_TEXT, theme.HUD_PANEL_BG) >= 4.5
        assert self._contrast(theme.HUD_GOLD_TEXT, PLAYER_COLORS[3]) < 1.1  # the old pairing
