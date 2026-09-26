"""A bug in a bot or a menu must not kill the GUI or lose the game (review §1.4, pygame-14).

A bot exception used to unwind through GameSession.run into start_new_game's
blanket handler: the game ended, no replay was written, pygame was left
initialised, and the only report was a console traceback. A menu exception
(e.g. the old 1v1v1 crash) escaped play_mode and closed the app.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pygame
import pytest

import reinforcetactics.ui.menus as menus_package
from reinforcetactics.app import bot_factory, game_loop
from reinforcetactics.app.input_handler import InputHandler
from reinforcetactics.cli import commands
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils import settings as settings_module
from reinforcetactics.utils.file_io import FileIO

REPO_ROOT = Path(__file__).resolve().parents[1]
BEGINNER_MAP = REPO_ROOT / "maps" / "1v1" / "beginner.csv"
TWO_PLAYERS = [{"type": "human", "bot_type": None}, {"type": "computer", "bot_type": "SimpleBot"}]


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


def _broken_take_turn(self):
    raise RuntimeError("bot bug")


class TestBotTurnIsContained:
    def _handler(self, bot, game=None):
        game = game or GameState(FileIO.load_map(str(BEGINNER_MAP)), num_players=2)
        renderer = Mock()
        renderer.end_turn_button = pygame.Rect(0, 0, 10, 10)
        renderer.resign_button = pygame.Rect(0, 20, 10, 10)
        return InputHandler(game, renderer, {2: bot}, num_players=2), game

    def test_raising_bot_turn_is_logged_reported_and_ended(self, caplog):
        bot = Mock(spec=SimpleBot)
        bot.take_turn = Mock(side_effect=RuntimeError("bot bug"))
        handler, game = self._handler(bot)
        game.end_turn()  # human ends turn 0 -> player 2 (the bot) to move

        with caplog.at_level(logging.ERROR, logger="reinforcetactics.app.input_handler"):
            handler._process_bot_turns()

        assert game.current_player == 1, "the bot's turn should have been ended"
        assert game.turn_number == 1
        bot.take_turn.assert_called_once()
        assert "error" in handler.notice_text and "Player 2" in handler.notice_text
        assert any(record.exc_info and "bot bug" in str(record.exc_info[1]) for record in caplog.records)

    def test_bot_that_raises_after_ending_its_turn_does_not_skip_the_next_player(self):
        game = GameState(FileIO.load_map(str(BEGINNER_MAP)), num_players=2)

        def end_then_raise():
            game.end_turn()
            raise RuntimeError("post-turn bug")

        bot = Mock(spec=SimpleBot)
        bot.take_turn = Mock(side_effect=end_then_raise)
        handler, _ = self._handler(bot, game)
        game.end_turn()

        handler._process_bot_turns()

        assert game.current_player == 1
        assert game.turn_number == 1

    def test_gui_session_keeps_running_after_a_bot_crash(self, gui_env, monkeypatch, caplog):
        """Real GameSession loop: SPACE -> SimpleBot raises -> play returns to the human."""
        monkeypatch.setattr(SimpleBot, "take_turn", _broken_take_turn)
        observed = {}
        frames = [0]
        original_render = game_loop.GameSession._render_frame

        def render_then_stop(session):
            original_render(session)
            frames[0] += 1
            if frames[0] == 1:
                pygame.event.post(_space_event())
            elif frames[0] >= 3:
                observed["current_player"] = session.game.current_player
                observed["turn_number"] = session.game.turn_number
                observed["notice"] = session.input_handler.notice_text
                session.running = False

        monkeypatch.setattr(game_loop.GameSession, "_render_frame", render_then_stop)

        with caplog.at_level(logging.ERROR):
            result = game_loop.start_new_game(mode="1v1", selected_map=str(BEGINNER_MAP), player_configs=TWO_PLAYERS)

        assert result == "quit"  # the loop ended normally (running=False), not via a crash
        assert observed["current_player"] == 1
        assert observed["turn_number"] == 1
        assert "Player 2" in observed["notice"]
        assert list((gui_env / "replays").glob("*.json")), "a mid-game exit saves the replay"
        assert not pygame.get_init(), "the display is released for the main menu"


class TestSessionCrashIsContained:
    def test_session_crash_autosaves_replay_and_crash_save(self, gui_env, monkeypatch):
        state = {}

        def crash_mid_game(session):
            session.input_handler.handle_keyboard_event(_space_event())  # a full round is played
            state["turn_number"] = session.game.turn_number
            state["units"] = len(session.game.units)
            raise RuntimeError("renderer bug")

        monkeypatch.setattr(game_loop.GameSession, "run", crash_mid_game)

        result = game_loop.start_new_game(mode="1v1", selected_map="random", player_configs=TWO_PLAYERS)

        assert result == "main_menu"
        assert not pygame.get_init()
        replays = list((gui_env / "replays").glob("*.json"))
        crash_saves = list((gui_env / "saves").glob("crash_*.json"))
        assert len(replays) == 1 and len(crash_saves) == 1

        # The crash save is a normal save the player can load and continue.
        pygame.init()
        monkeypatch.setattr(game_loop.GameSession, "run", lambda session: state.update(loaded=session.game) or "main_menu")
        assert game_loop.load_saved_game(json.loads(crash_saves[0].read_text())) == "main_menu"
        assert state["loaded"].turn_number == state["turn_number"]
        assert len(state["loaded"].units) == state["units"]

    def test_keyboard_interrupt_is_not_swallowed(self, gui_env, monkeypatch):
        def interrupt(session):
            raise KeyboardInterrupt

        monkeypatch.setattr(game_loop.GameSession, "run", interrupt)

        with pytest.raises(KeyboardInterrupt):
            game_loop.start_new_game(mode="1v1", selected_map=str(BEGINNER_MAP), player_configs=TWO_PLAYERS)

    def test_bot_that_cannot_be_built_falls_back_to_simplebot(self, gui_env, monkeypatch):
        """E.g. a loaded save whose ModelBot model file has since moved."""
        real_create_bot = bot_factory.create_bot

        def create_bot(game, player_num, bot_type, settings, model_path=None):
            if bot_type == "ModelBot":
                raise FileNotFoundError(f"Model file not found: {model_path}")
            return real_create_bot(game, player_num, bot_type, settings, model_path)

        monkeypatch.setattr(bot_factory, "create_bot", create_bot)
        game = GameState(FileIO.load_map(str(BEGINNER_MAP)), num_players=2)

        def configs():
            return [{"type": "human"}, {"type": "computer", "bot_type": "ModelBot", "model_path": "/moved/agent.zip"}]

        bots = bot_factory.create_bots_from_config(game, configs(), settings_module.get_settings())
        assert isinstance(bots[2], SimpleBot)

        # The GUI collects the reason to show it on screen.
        notices = []
        bot_factory.create_bots_from_config(game, configs(), settings_module.get_settings(), notices=notices)
        assert len(notices) == 1 and "Model file not found" in notices[0]


class _ScriptedMainMenu:
    """Stands in for MainMenu: each run() pops the next scripted step."""

    steps: list = []

    def run(self):
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


class TestPlayModeTopLevel:
    @pytest.fixture(autouse=True)
    def scripted_menu(self, gui_env, monkeypatch):
        monkeypatch.setattr(menus_package, "MainMenu", _ScriptedMainMenu)
        return _ScriptedMainMenu

    def test_menu_exception_returns_to_the_main_menu(self, scripted_menu):
        scripted_menu.steps = [ValueError("Invalid game_mode: 1v1v1"), {"type": "exit"}]

        commands.play_mode(None)

        assert scripted_menu.steps == [], "the menu should have been shown again after the error"

    def test_session_exception_returns_to_the_main_menu(self, scripted_menu, monkeypatch):
        scripted_menu.steps = [{"type": "watch_replay", "replay_path": "x.json"}, {"type": "exit"}]

        def broken_watch_replay(replay_path=None):
            raise RuntimeError("replay bug")

        monkeypatch.setattr(game_loop, "watch_replay", broken_watch_replay)

        commands.play_mode(None)

        assert scripted_menu.steps == []

    def test_load_game_uses_the_save_picked_in_the_main_menu(self, scripted_menu, monkeypatch):
        save_data = {"turn_number": 3}
        scripted_menu.steps = [{"type": "load_game", "save_data": save_data}, {"type": "exit"}]
        calls = []
        monkeypatch.setattr(game_loop, "load_saved_game", lambda *args: calls.append(args) or "main_menu")

        commands.play_mode(None)

        assert calls == [(save_data,)], "the player must not be asked to pick the save a second time"

    def test_new_game_passes_the_seat_count(self, scripted_menu, monkeypatch):
        scripted_menu.steps = [
            {"type": "new_game", "map": "m.csv", "mode": "1v1v1", "num_players": 3, "players": [{}, {}, {}]},
            {"type": "exit"},
        ]
        calls = []
        monkeypatch.setattr(game_loop, "start_new_game", lambda **kwargs: calls.append(kwargs) or "main_menu")

        commands.play_mode(None)

        assert calls[0]["num_players"] == 3

    def test_keyboard_interrupt_still_quits(self, scripted_menu):
        scripted_menu.steps = [KeyboardInterrupt()]

        with pytest.raises(KeyboardInterrupt):
            commands.play_mode(None)

    def test_an_error_on_every_attempt_eventually_exits(self, scripted_menu):
        scripted_menu.steps = [RuntimeError("no display")] * commands.MAX_CONSECUTIVE_PLAY_ERRORS

        with pytest.raises(RuntimeError, match="no display"):
            commands.play_mode(None)
