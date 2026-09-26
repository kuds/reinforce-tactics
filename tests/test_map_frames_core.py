"""One coordinate frame per game, and one display padder (review core-19, anim-18, core-22).

GameState records every coordinate on the grid it was built from, and a
replay stores that grid as its ``initial_map``. A GUI game is played on the
UI-padded map (``FileIO.load_map(for_ui=True)``), so its replay carries the
padding and still plays back exactly, whichever padding the viewer or the
video exporter adds for display. That padding is ``FileIO.pad_for_display``
everywhere; it used to be written out three times.
"""

import logging
import os

import numpy as np
import pandas as pd
import pygame
import pytest

from reinforcetactics.constants import MIN_MAP_SIZE
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.grid import TileGrid
from reinforcetactics.utils import video
from reinforcetactics.utils.file_io import FileIO

BEGINNER_MAP = "maps/1v1/beginner.csv"


def _old_replay_viewer_padding(map_data, border_size=2):
    """ReplayPlayer._pad_map_for_replay as it was, the reference for the shared padder."""
    df = pd.DataFrame(map_data)
    height, width = df.shape
    offset_x = offset_y = 0
    if height < MIN_MAP_SIZE or width < MIN_MAP_SIZE:
        min_height, min_width = max(height, MIN_MAP_SIZE), max(width, MIN_MAP_SIZE)
        pad_width, pad_height = min_width - width, min_height - height
        padded = pd.DataFrame(np.full((min_height, min_width), "o", dtype=object))
        offset_x, offset_y = pad_width // 2, pad_height // 2
        padded.iloc[offset_y : offset_y + height, offset_x : offset_x + width] = df.values
        df = padded
    height, width = df.shape
    bordered = pd.DataFrame(np.full((height + 2 * border_size, width + 2 * border_size), "o", dtype=object))
    bordered.iloc[border_size : border_size + height, border_size : border_size + width] = df.values
    return bordered, offset_x + border_size, offset_y + border_size


def _terrain(height, width):
    """A map whose every tile is distinguishable by where it came from."""
    codes = np.array([["p", "f", "m", "r"][(x + 2 * y) % 4] for y in range(height) for x in range(width)], dtype=object)
    codes = codes.reshape(height, width)
    codes[0][0] = "h_1"
    codes[height - 1][width - 1] = "h_2"
    return codes


class TestPadForDisplay:
    @pytest.mark.parametrize("shape", [(6, 6), (7, 11), (19, 20), (20, 20), (25, 25), (10, 30), (30, 10)])
    def test_matches_the_replay_viewers_old_padding(self, shape):
        terrain = _terrain(*shape)
        expected, ex, ey = _old_replay_viewer_padding(terrain)

        padded, ox, oy = FileIO.pad_for_display(terrain, MIN_MAP_SIZE, 2)

        assert (ox, oy) == (ex, ey)
        assert padded.values.tolist() == expected.values.tolist()

    def test_video_framing_is_a_plain_border(self):
        terrain = _terrain(6, 8)

        padded, ox, oy = FileIO.pad_for_display(pd.DataFrame(terrain), min_size=0, border_size=2)

        assert (ox, oy) == (2, 2)
        assert padded.shape == (10, 12)
        assert padded.values[2:8, 2:10].tolist() == terrain.tolist()
        assert set(padded.values[:2].ravel()) == {"o"}

    @pytest.mark.parametrize("shape", [(6, 6), (10, 25)])
    def test_ui_load_offsets_locate_the_map_file(self, tmp_path, shape):
        terrain = _terrain(*shape)
        path = tmp_path / "map.csv"
        pd.DataFrame(terrain).to_csv(path, header=False, index=False)

        loaded = FileIO.load_map_with_metadata(str(path), for_ui=True, border_size=2)

        ox, oy = loaded["padding_offset_x"], loaded["padding_offset_y"]
        height, width = shape
        assert loaded["map_data"].values[oy : oy + height, ox : ox + width].tolist() == terrain.tolist()
        assert loaded["original_map_data"] == terrain.tolist()
        # A 25-wide, 10-high map used to fail to load for the UI: the
        # minimum-size padding assumed both sides were short.
        assert loaded["map_data"].shape == (max(height, MIN_MAP_SIZE) + 4, max(width, MIN_MAP_SIZE) + 4)

    def test_a_gui_map_is_still_padded_to_the_minimum_with_a_border(self):
        loaded = FileIO.load_map_with_metadata(BEGINNER_MAP, for_ui=True, border_size=2)

        assert loaded["map_data"].shape == (24, 24)
        assert (loaded["padding_offset_x"], loaded["padding_offset_y"]) == (9, 9)


class TestMapInput:
    """TileGrid and GameState take a DataFrame, an array or a list of rows alike (core-22)."""

    def test_a_list_of_rows_builds_the_same_game(self):
        rows = [["h_1", "b_1", "p"], ["p", "t", "p"], ["p", "b_2", "h_2"]]

        from_list = GameState(rows)
        from_frame = GameState(pd.DataFrame(rows))

        assert from_list.initial_map_data == from_frame.initial_map_data == rows
        assert [[t.type for t in row] for row in TileGrid(rows).tiles] == [["h", "b", "p"], ["p", "t", "p"], ["p", "b", "h"]]
        assert from_list.to_dict()["tiles"] == from_frame.to_dict()["tiles"]

    def test_anything_but_a_2d_grid_is_refused(self):
        # The RL encoding (grid.to_numpy()) used to build an all-ocean map.
        with pytest.raises(ValueError, match="2D grid"):
            TileGrid(GameState([["h_1", "h_2"]]).grid.to_numpy())
        with pytest.raises(ValueError, match="2D grid"):
            TileGrid([["p", "p"], ["p"]])


def _play_gui_game():
    """A short GUI-style game on the padded beginner map: creates, a Haste and
    the hasted unit's two moves, a Defence Buff, melee and ranged attacks.

    Beginner's buildings sit at grid (10, 9)/(9, 10) and (14, 13)/(13, 14):
    the map is centred in 20x20 and bordered, so file (0, 0) is grid (9, 9).
    """
    game = GameState(
        FileIO.load_map(BEGINNER_MAP, for_ui=True, border_size=2),
        num_players=2,
        engine_overrides={"starting_gold": 3000},
        seed=7,
    )
    game.map_file_used = BEGINNER_MAP
    sorcerer = game.create_unit("S", 10, 9)
    warrior = game.create_unit("W", 9, 10)
    game.end_turn()
    enemy_warrior = game.create_unit("W", 14, 13)
    enemy_archer = game.create_unit("A", 13, 14)
    game.end_turn()
    assert game.haste(sorcerer, warrior)
    assert game.move_unit(warrior, 10, 11)
    game.end_unit_turn(warrior)  # the GUI's Wait: the hasted extra action starts
    assert game.move_unit(warrior, 11, 12)
    game.end_turn()
    assert game.move_unit(enemy_warrior, 12, 12)
    assert game.attack(enemy_warrior, warrior)["damage"]
    game.end_turn()
    assert game.move_unit(sorcerer, 10, 11)
    assert game.defence_buff(sorcerer, warrior)
    assert game.attack(warrior, enemy_warrior)["damage"]
    game.end_turn()
    assert game.move_unit(enemy_archer, 13, 13)
    assert game.attack(enemy_archer, warrior)["damage"]
    game.end_turn()
    return game


def _final_state(game, offset=(0, 0)):
    """What playback must reproduce, in the recorded game's coordinates."""
    ox, oy = offset
    units = sorted(
        (
            u.unit_id,
            u.type,
            u.player,
            u.x - ox,
            u.y - oy,
            u.health,
            u.haste_cooldown,
            u.defence_buff_cooldown,
            u.defence_buff_turns,
        )
        for u in game.units
    )
    return units, dict(game.player_gold), game.turn_number, game.current_player


class TestGuiReplayRoundTrip:
    @pytest.fixture(autouse=True)
    def headless_pygame(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        yield
        pygame.quit()

    @pytest.fixture
    def recorded(self, tmp_path):
        game = _play_gui_game()
        replay = FileIO.load_replay(game.save_replay_to_file(str(tmp_path / "gui_game.json")))
        return game, replay

    def test_the_replay_records_the_grid_its_actions_are_on(self, recorded):
        game, replay = recorded
        initial_map = replay["game_info"]["initial_map"]

        assert initial_map == game.initial_map_data
        assert (len(initial_map), len(initial_map[0])) == (24, 24)
        for action in replay["actions"]:
            if action["type"] == "create_unit":
                assert initial_map[action["y"]][action["x"]] == f"b_{action['player']}"
        # Sorcerer abilities record the caster where it stood (the key the old
        # padded-to-original conversion list missed).
        casts = [(a["type"], tuple(a["sorcerer_pos"])) for a in replay["actions"] if "sorcerer_pos" in a]
        assert casts == [("haste", (10, 9)), ("defence_buff", (10, 11))]

    def test_the_replay_viewer_plays_it_back_exactly(self, recorded, caplog):
        from reinforcetactics.utils.replay_player import ReplayPlayer

        game, replay = recorded
        player = ReplayPlayer(replay, FileIO.load_replay_map(replay["game_info"]))
        with caplog.at_level(logging.WARNING, logger="reinforcetactics.utils.replay_actions"):
            while player.current_action_index < len(player.actions):
                player.step_forward()

        assert not caplog.records  # no action refused or unit not found

        offset = (player.padding_offset_x, player.padding_offset_y)
        assert offset == (2, 2)  # already 24x24: only the viewer's own border
        assert _final_state(player.game_state, offset) == _final_state(game)

    def test_video_export_plays_it_back_exactly(self, recorded, tmp_path, monkeypatch, caplog):
        game, replay = recorded
        played = {}

        class _NoWriter:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, frame):
                pass

        def _capture(frame, game_info, game_state):
            played["game_state"] = game_state
            return frame

        monkeypatch.setattr(video, "_VideoWriter", _NoWriter)
        monkeypatch.setattr(video, "_overlay_game_over", _capture)

        with caplog.at_level(logging.WARNING, logger="reinforcetactics.utils.replay_actions"):
            video.record_replay_to_video(replay, str(tmp_path / "gui_game.mp4"), fps=1)

        assert not caplog.records
        assert _final_state(played["game_state"], (2, 2)) == _final_state(game)
