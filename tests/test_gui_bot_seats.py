"""Bots in a GUI game move without waiting for a human to end a turn (review pygame-5).

Bot turns used to run only from the End Turn handlers, so a game with a bot in
seat 1 (or only bots) sat at turn 0 forever and the human could move the bot's
units. GameSession now lets bots act at the start of each frame.
"""

from unittest.mock import Mock

import pygame
import pytest

from reinforcetactics.app.game_loop import GameSession
from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import SimpleBot
from reinforcetactics.utils.file_io import FileIO

MAP = "maps/1v1/beginner.csv"
MAX_FRAMES = 200


@pytest.fixture(autouse=True)
def _display():
    pygame.init()
    pygame.display.set_mode((64, 64))
    yield


def _session(game, bots, monkeypatch):
    """A GameSession that stops itself after MAX_FRAMES rendered frames."""
    session = GameSession(game, Mock(), bots, num_players=2)
    frames = [0]

    def render_frame():
        frames[0] += 1
        if frames[0] >= MAX_FRAMES:
            session.running = False

    monkeypatch.setattr(session, "_render_frame", render_frame)
    monkeypatch.setattr(session, "_handle_game_over", lambda: "main_menu")
    return session, frames


def _game(max_turns=None):
    return GameState(FileIO.load_map(MAP), num_players=2, max_turns=max_turns)


def test_all_bot_game_plays_to_the_end(monkeypatch):
    game = _game(max_turns=4)
    bots = {1: SimpleBot(game, player=1), 2: SimpleBot(game, player=2)}
    session, _frames = _session(game, bots, monkeypatch)

    assert session.run() == "main_menu"
    assert game.game_over
    assert any(a.get("player") == 1 and a.get("type") != "end_turn" for a in game.action_history)


def test_bot_in_seat_one_moves_before_the_human(monkeypatch):
    game = _game()
    bots = {1: SimpleBot(game, player=1)}
    session, frames = _session(game, bots, monkeypatch)
    monkeypatch.setattr(pygame.event, "get", lambda: [])

    def stop_after_first_frame():
        frames[0] += 1
        session.running = False

    monkeypatch.setattr(session, "_render_frame", stop_after_first_frame)
    session.run()

    assert game.current_player == 2  # the human's turn, after the bot played turn 1
    assert any(a.get("player") == 1 and a.get("type") == "create_unit" for a in game.action_history)


def test_a_bot_that_never_ends_its_turn_is_not_rerun_every_frame(monkeypatch):
    game = _game()
    idle_bot = Mock()  # take_turn() returns without ending the turn
    session, _frames = _session(game, {1: idle_bot}, monkeypatch)
    monkeypatch.setattr(pygame.event, "get", lambda: [])

    session.run()

    # One batch of attempts (the input handler caps it), not one per frame.
    assert 1 <= idle_bot.take_turn.call_count <= 2 * 2
    assert "did not finish its turn" in session.input_handler.notice_text
