"""Random-map saves and replays keep their terrain (review §1.5, pygame-2).

Saves only stored capturable tiles plus a ``map_file`` that is ``null`` for
random maps, and the load path's ``"map_file" in save_data`` check was always
true, so every random-map save failed to load (and the fallback would have
generated different terrain). Saves now record the terrain; ``to_dict`` also
writes ``max_turns`` and ``end_reason``, which ``from_dict`` always read back
(core-6 / persist-9). Replays and video export build their map from what the
replay recorded instead of a fresh random map.
"""

import json
from pathlib import Path

import pandas as pd
import pygame
import pytest

from reinforcetactics.app import game_loop
from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui.menus.list_detail import ListDetailMenu
from reinforcetactics.ui.menus.save_load.load_game_menu import LoadGameMenu
from reinforcetactics.ui.menus.save_load.save_game_menu import SaveGameMenu
from reinforcetactics.utils import settings as settings_module
from reinforcetactics.utils import video
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.language import get_language

REPO_ROOT = Path(__file__).resolve().parents[1]
BEGINNER_MAP = REPO_ROOT / "maps" / "1v1" / "beginner.csv"
SKIRMISH_MAP = REPO_ROOT / "maps" / "1v1" / "skirmish.csv"


@pytest.fixture
def gui_env(tmp_path, monkeypatch):
    """Headless pygame in a temp working dir (saves/ and replays/ land there)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, "_settings_instance", settings_module.Settings(str(tmp_path / "settings.json")))
    pygame.init()
    yield tmp_path
    pygame.quit()


def _space_event():
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0, unicode=" ", scancode=0)


def _terrain(game):
    return [[(t.type, t.player, t.health) for t in row] for row in game.grid.tiles]


def _units(game):
    return sorted((u.type, u.x, u.y, u.player, u.health, u.unit_id) for u in game.units)


def _random_game():
    return GameState(FileIO.generate_random_map(20, 20, num_players=2), num_players=2)


def _play_session_then(monkeypatch, action):
    """Patch GameSession.run: play two rounds against SimpleBot, then ``action(session)``."""
    captured = {}

    def fake_run(session):
        for _ in range(2):
            session.input_handler.handle_keyboard_event(_space_event())
        captured["game"] = session.game
        captured["result"] = action(session)
        return "main_menu"

    monkeypatch.setattr(game_loop.GameSession, "run", fake_run)
    return captured


def _load_via_gui(monkeypatch, save_data):
    """Run load_saved_game as the main menu does, capturing the restored game."""
    captured = {}

    def fake_run(session):
        captured["game"] = session.game
        return "main_menu"

    monkeypatch.setattr(game_loop.GameSession, "run", fake_run)
    monkeypatch.setattr(game_loop.LoadGameMenu, "run", lambda self: save_data)
    captured["result"] = game_loop.load_saved_game()
    return captured


class TestSaveFormat:
    def test_to_dict_persists_max_turns_end_reason_and_terrain(self):
        game = GameState(FileIO.load_map(str(BEGINNER_MAP)), num_players=2, max_turns=30)
        game.resign(1)

        data = json.loads(json.dumps(game.to_dict()))

        assert data["max_turns"] == 30
        assert data["end_reason"] == "resign"
        assert data["map_data"] == game.initial_map_data

    def test_max_turns_and_end_reason_round_trip(self):
        game = GameState(FileIO.load_map(str(BEGINNER_MAP)), num_players=2, max_turns=30)
        game.resign(2)

        restored = GameState.from_dict(json.loads(json.dumps(game.to_dict())), FileIO.load_map(str(BEGINNER_MAP)))

        assert restored.max_turns == 30
        assert restored.end_reason == "resign"
        assert restored.game_over and restored.winner == 1

    def test_from_dict_rebuilds_the_grid_from_recorded_terrain(self):
        game = _random_game()
        game.create_unit("W", 1, 3, 1)
        game.end_turn()

        restored = GameState.from_dict(json.loads(json.dumps(game.to_dict())))

        assert _terrain(restored) == _terrain(game)
        assert _units(restored) == _units(game)


class TestRandomMapSaveLoadThroughTheGui:
    def test_random_map_save_round_trips_through_save_and_load_menus(self, gui_env, monkeypatch):
        """New Game on "Random Map" -> in-game save -> Load Game restores the same game."""

        def save_like_the_gui(session):
            # The GUI never sets a turn limit; set one to check it survives.
            session.game.max_turns = 40
            return SaveGameMenu(session.game, screen=session.renderer.screen)._save_game()

        played = _play_session_then(monkeypatch, save_like_the_gui)
        result = game_loop.start_new_game(
            mode="1v1",
            selected_map="random",
            player_configs=[{"type": "human", "bot_type": None}, {"type": "computer", "bot_type": "SimpleBot"}],
            num_players=2,
        )
        assert result == "main_menu"
        original = played["game"]
        save_path = played["result"]
        assert original.map_file_used is None
        assert original.turn_number == 2 and original.units, "the bot should have built units"

        # Load Game lists the save and hands back its parsed contents.
        pygame.init()
        load_menu = LoadGameMenu(pygame.display.set_mode((900, 700)), saves_dir="saves")
        assert load_menu.save_metadata[save_path]["map_name"] == get_language().get("map_random", "Random Map")
        monkeypatch.setattr(ListDetailMenu, "run", lambda self: save_path, raising=False)
        save_data = load_menu.run()
        assert isinstance(save_data, dict)

        loaded = _load_via_gui(monkeypatch, save_data)

        restored = loaded["game"]
        assert _terrain(restored) == _terrain(original)
        assert _units(restored) == _units(original)
        assert restored.player_gold == original.player_gold
        assert restored.turn_number == original.turn_number
        assert restored.current_player == original.current_player
        assert restored.max_turns == 40

    def test_recorded_terrain_wins_over_an_edited_map_file(self, gui_env, monkeypatch):
        """A save must not silently move onto different terrain if its map file changes."""
        game = GameState(FileIO.load_map(str(BEGINNER_MAP), for_ui=True, border_size=2), num_players=2)
        game.map_file_used = str(SKIRMISH_MAP)  # as if beginner.csv had since been replaced
        save_data = json.loads(json.dumps(game.to_dict()))

        loaded = _load_via_gui(monkeypatch, save_data)

        assert _terrain(loaded["game"]) == _terrain(game)

    def test_saves_without_recorded_terrain_still_load_from_their_map_file(self, gui_env, monkeypatch):
        """The shipped scenario saves predate the terrain field."""
        scenario = json.loads((REPO_ROOT / "saves" / "skirmish_scenario.json").read_text())
        assert "map_data" not in scenario
        scenario["map_file"] = str(REPO_ROOT / scenario["map_file"])

        loaded = _load_via_gui(monkeypatch, scenario)

        expected = GameState(FileIO.load_map(scenario["map_file"], for_ui=True, border_size=2), num_players=2)
        assert [[t.type for t in row] for row in loaded["game"].grid.tiles] == [
            [t.type for t in row] for row in expected.grid.tiles
        ]

    def test_old_random_map_save_is_refused_not_rebuilt_on_new_terrain(self, gui_env, monkeypatch):
        save_data = json.loads(json.dumps(_random_game().to_dict()))
        del save_data["map_data"]  # as written before the terrain was recorded

        loaded = _load_via_gui(monkeypatch, save_data)

        assert loaded["result"] == "main_menu"
        assert "game" not in loaded, "no session should start on made-up terrain"


class TestRandomMapReplays:
    def test_random_map_replay_records_its_terrain(self, gui_env):
        game = _random_game()
        game.end_turn()

        replay = json.loads(Path(game.save_replay_to_file()).read_text())

        assert replay["game_info"]["map_file"] is None
        assert replay["game_info"]["initial_map"] == game.initial_map_data

    def test_watch_replay_uses_the_recorded_terrain(self, gui_env, monkeypatch):
        game = _random_game()
        game.end_turn()
        replay_path = game.save_replay_to_file()
        captured = {}

        class FakePlayer:
            def __init__(self, replay_data, initial_map_data):
                captured["map"] = pd.DataFrame(initial_map_data).values.tolist()

            def run(self):
                pass

        monkeypatch.setattr(game_loop, "ReplayPlayer", FakePlayer)

        assert game_loop.watch_replay(replay_path) == "main_menu"
        assert captured["map"] == game.initial_map_data

    def test_replay_without_terrain_uses_its_map_file_not_a_random_map(self, gui_env, monkeypatch):
        replay_path = gui_env / "replay_no_terrain.json"
        replay_path.write_text(
            json.dumps({"game_info": {"num_players": 2, "map_file": str(BEGINNER_MAP)}, "actions": []}),
        )
        captured = {}

        class FakePlayer:
            def __init__(self, replay_data, initial_map_data):
                captured["map"] = pd.DataFrame(initial_map_data).values.tolist()

            def run(self):
                pass

        monkeypatch.setattr(game_loop, "ReplayPlayer", FakePlayer)

        assert game_loop.watch_replay(str(replay_path)) == "main_menu"
        assert captured["map"] == FileIO.load_map(str(BEGINNER_MAP)).values.tolist()

    def test_replay_with_no_map_at_all_is_refused(self, gui_env, monkeypatch):
        replay_path = gui_env / "replay_no_map.json"
        replay_path.write_text(json.dumps({"game_info": {"num_players": 2, "map_file": None}, "actions": []}))
        constructed = []
        monkeypatch.setattr(game_loop, "ReplayPlayer", lambda *args: constructed.append(args))

        assert game_loop.watch_replay(str(replay_path)) == "main_menu"
        assert constructed == [], "the replay must not be played on a made-up map"

    def test_video_export_of_snapshots_uses_their_recorded_terrain(self, gui_env, monkeypatch):
        game = _random_game()
        game.create_unit("W", 1, 3, 1)
        snapshots = [json.loads(json.dumps(game.to_dict()))]
        rendered = []

        def consume(frames, output_path, fps, scale=None):
            for _ in frames:
                pass
            return output_path

        monkeypatch.setattr(video, "_write_frames_to_video", consume)
        monkeypatch.setattr(video, "_draw_video_hud", lambda screen, gs: rendered.append(gs))

        video.record_game_to_video(snapshots, output_path=str(gui_env / "out.mp4"))

        assert len(rendered) == 1
        assert _terrain(rendered[0]) == _terrain(game)
        assert _units(rendered[0]) == _units(game)

    def test_replay_video_export_falls_back_to_the_map_file(self, gui_env, monkeypatch):
        written = []

        class FakeWriter:
            def __init__(self, output_path, fps, scale=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, frame):
                written.append(frame.shape)

        monkeypatch.setattr(video, "_VideoWriter", FakeWriter)
        replay = {"game_info": {"num_players": 2, "map_file": str(BEGINNER_MAP)}, "actions": []}

        video.record_replay_to_video(replay, output_path=str(gui_env / "out.mp4"), fps=1)

        assert written, "frames should be rendered from the map file"
