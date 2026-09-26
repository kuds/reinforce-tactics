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


def test_a_bot_that_never_ends_its_turn_has_it_ended_not_handed_to_the_human(monkeypatch):
    game = _game()
    idle_bot = Mock()  # take_turn() returns without ending the turn
    session, _frames = _session(game, {1: idle_bot}, monkeypatch)
    monkeypatch.setattr(pygame.event, "get", lambda: [])

    session.run()

    # Its turn was ended for it once, so the human (player 2) is to move and
    # the bot is not re-run every frame. The seat used to stay the bot's
    # while the human's clicks played its units.
    assert idle_bot.take_turn.call_count == 1
    assert game.current_player == 2
    assert "did not finish its turn" in session.input_handler.notice_text


def _only_bots_left_games():
    all_bots = _game()
    yield "all-bot 1v1", all_bots, {1: SimpleBot(all_bots, player=1), 2: SimpleBot(all_bots, player=2)}
    ffa = GameState(FileIO.load_map("maps/1v1v1/triangle_arena.csv"), num_players=3)
    ffa.resign(3)  # the human left a free-for-all; the two bots play on
    ffa.end_turn() if ffa.current_player == 3 else None
    yield "1v1v1 after the human resigned", ffa, {1: SimpleBot(ffa, player=1), 2: SimpleBot(ffa, player=2)}


@pytest.mark.parametrize("case", ["all-bot 1v1", "1v1v1 after the human resigned"])
def test_a_game_with_only_bots_to_move_can_still_be_quit(monkeypatch, case):
    """Bot turns used to run instead of the frame, so no QUIT was ever read (and nothing ends a GUI game)."""
    name, game, bots = next(g for g in _only_bots_left_games() if g[0] == case)
    session, frames = _session(game, bots, monkeypatch)
    calls = [0]

    def events():
        calls[0] += 1
        return [pygame.event.Event(pygame.QUIT)] if calls[0] == 3 else []

    monkeypatch.setattr(pygame.event, "get", events)
    monkeypatch.setattr(session, "_handle_pause", lambda: "quit")

    assert session.run() == "quit", name
    assert frames[0] < MAX_FRAMES and not game.game_over


def test_human_input_is_ignored_while_a_bot_seat_is_to_move(monkeypatch):
    game = GameState(FileIO.load_map("maps/1v1v1/triangle_arena.csv"), num_players=3)
    bots = {1: SimpleBot(game, player=1), 2: SimpleBot(game, player=2)}  # player 3 is the human
    session, _frames = _session(game, bots, monkeypatch)
    handler = session.input_handler
    seen = []
    monkeypatch.setattr(handler, "handle_mouse_click", lambda pos: seen.append(("click", game.current_player)))
    monkeypatch.setattr(handler, "handle_keyboard_event", lambda ev: seen.append(("key", game.current_player)))

    def events():
        if len(seen) >= 6:
            session.running = False
        return [
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(40, 40)),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0, unicode=" "),
        ]

    monkeypatch.setattr(pygame.event, "get", events)
    session.run()

    assert seen and all(player == 3 for _kind, player in seen)
