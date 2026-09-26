"""Watch Replay / Load Game must survive any file in their folders (review §1.4, menus-1).

Random-map replays record ``"map_file": null`` and ``os.path.basename(None)``
crashed the replay picker every time it opened; a save or replay that is not a
JSON object, or has fields of the wrong type, crashed the pickers the same way.
Each such file now gets fallback metadata instead, and every row and detail
panel still draws.
"""

import json
from pathlib import Path

import pygame
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui.menus.list_detail import ListDetailMenu
from reinforcetactics.ui.menus.save_load.load_game_menu import LoadGameMenu
from reinforcetactics.ui.menus.save_load.replay_selection_menu import ReplaySelectionMenu
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.language import get_language

# Content that parses as JSON (or doesn't) but is not a valid replay/save.
MALFORMED_FILES = {
    "list.json": "[1, 2, 3]",
    "string.json": '"just a string"',
    "truncated.json": '{"game_info": {',
    "wrong_types.json": json.dumps(
        {
            "timestamp": 12,
            "game_info": {
                "num_players": "three",
                "winner": [1],
                "game_over": "yes",
                "player_configs": {"type": "human"},
                "map_file": 5,
                "total_turns": "many",
                "initial_map": "not a grid",
                "max_turns": {},
            },
        }
    ),
    "odd_players.json": json.dumps(
        {
            "game_info": {
                "num_players": 9,
                "winner": 7,
                "game_over": True,
                "player_configs": [5, {"type": 3}, {"type": "llm", "name": 42}],
            }
        }
    ),
    "wrong_save_types.json": json.dumps(
        {
            "timestamp": ["2026"],
            "player_gold": [250, 250],
            "units": [7, {"player": [1], "type": 3, "health": "full", "x": "a"}],
            "tiles": ["x", {"x": "1", "y": None}],
            "turn_number": "3",
            "current_player": [2],
            "num_players": 0,
            "winner": {"p": 1},
            "player_configs": "Human",
        }
    ),
}
NOT_UTF8 = b'{"game_info": "\xff\xfe\xfd"}'


def _random_map_label():
    """The (localised) "Random Map" label the pickers show for map_file null."""
    return get_language().get("map_random", "Random Map")


@pytest.fixture
def screen(tmp_path, monkeypatch):
    """Headless 900x700 display in a temp cwd (the replay picker also scans ./tournament_results)."""
    monkeypatch.chdir(tmp_path)
    pygame.init()
    yield pygame.display.set_mode((900, 700))
    pygame.quit()


def _write_malformed(folder, prefix):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in MALFORMED_FILES.items():
        path = folder / f"{prefix}_{name}"
        path.write_text(content, encoding="utf-8")
        paths.append(str(path))
    path = folder / f"{prefix}_not_utf8.json"
    path.write_bytes(NOT_UTF8)
    paths.append(str(path))
    return paths


def _draw_every_row_and_detail(menu, items):
    """Draw the list and each item's detail panel, as hovering down the list does."""
    for index in range(len(items)):
        menu.hover_index = -1
        menu.selected_index = index
        menu.draw()


def _random_map_replay(folder):
    game = GameState(FileIO.generate_random_map(20, 20, num_players=2), num_players=2)
    game.end_turn()
    return game.save_replay_to_file(str(folder / "replay_random.json"))


class TestReplaySelectionMenu:
    def test_random_map_replay_is_listed_as_random_map(self, screen, tmp_path):
        replay_path = _random_map_replay(tmp_path / "replays")
        assert json.loads(Path(replay_path).read_text())["game_info"]["map_file"] is None

        menu = ReplaySelectionMenu(screen, replays_dir=str(tmp_path / "replays"))

        assert menu.replay_files == [replay_path]
        metadata = menu.replay_metadata[replay_path]
        assert metadata["map_name"] == _random_map_label()
        assert metadata["initial_map"], "the preview should come from the recorded terrain"
        _draw_every_row_and_detail(menu, menu.replay_files)

    def test_malformed_replays_do_not_crash_the_menu(self, screen, tmp_path):
        replays_dir = tmp_path / "replays"
        good = _random_map_replay(replays_dir)
        bad = _write_malformed(replays_dir, "replay")

        menu = ReplaySelectionMenu(screen, replays_dir=str(replays_dir))

        assert sorted(menu.replay_files) == sorted([good, *bad])
        # Unreadable or non-object files get the minimal "Unknown" metadata.
        for name in ("list.json", "string.json", "truncated.json", "not_utf8.json"):
            assert menu.replay_metadata[str(replays_dir / f"replay_{name}")]["result"] == "Unknown"
        _draw_every_row_and_detail(menu, menu.replay_files)

    def test_wrong_field_types_fall_back_to_defaults(self, screen, tmp_path):
        replays_dir = tmp_path / "replays"
        _write_malformed(replays_dir, "replay")

        menu = ReplaySelectionMenu(screen, replays_dir=str(replays_dir))

        wrong = menu.replay_metadata[str(replays_dir / "replay_wrong_types.json")]
        assert wrong["num_players"] == 2
        assert wrong["winner"] is None
        assert wrong["total_turns"] == 0
        assert wrong["map_name"] == "Unknown Map"
        odd = menu.replay_metadata[str(replays_dir / "replay_odd_players.json")]
        assert odd["num_players"] == 2  # 9 seats can't be drawn; fall back
        assert all(isinstance(odd[key], str) for key in ("player1", "player2"))


class TestLoadGameMenu:
    def test_random_map_save_is_listed_as_random_map(self, screen, tmp_path):
        game = GameState(FileIO.generate_random_map(20, 20, num_players=2), num_players=2)
        save_path = game.save_to_file(str(tmp_path / "saves" / "save_random.json"))

        menu = LoadGameMenu(screen, saves_dir=str(tmp_path / "saves"))

        assert menu.save_metadata[save_path]["map_name"] == _random_map_label()
        _draw_every_row_and_detail(menu, menu.save_files)

    def test_malformed_saves_do_not_crash_the_menu(self, screen, tmp_path):
        saves_dir = tmp_path / "saves"
        bad = _write_malformed(saves_dir, "save")

        menu = LoadGameMenu(screen, saves_dir=str(saves_dir))

        assert sorted(menu.save_files) == sorted(bad)
        wrong = menu.save_metadata[str(saves_dir / "save_wrong_save_types.json")]
        assert wrong["player_gold"] == {}
        assert wrong["turn_number"] == 0
        assert wrong["num_players"] == 2
        assert wrong["unit_counts"] == {0: 1}
        _draw_every_row_and_detail(menu, menu.save_files)

    def test_selecting_a_non_object_save_returns_nothing_to_load(self, screen, tmp_path, monkeypatch):
        saves_dir = tmp_path / "saves"
        _write_malformed(saves_dir, "save")
        menu = LoadGameMenu(screen, saves_dir=str(saves_dir))
        monkeypatch.setattr(ListDetailMenu, "run", lambda self: str(saves_dir / "save_list.json"), raising=False)

        assert menu.run() is None
