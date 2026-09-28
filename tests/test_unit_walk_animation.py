"""Units walk their moves in the pygame client (review pygame-17, anim-7, anim-12).

The engine applies a move at once and hands ``GameState.move_listeners`` the
path the unit walked. The renderer tweens the sprite along it, a tile per
``theme.UNIT_WALK_MS_PER_TILE``, facing the way each step goes. The board
ignores the player until their unit arrives, a bot's moves play out one at a
time, and animation state follows the engine's ``unit_id`` and is dropped
when the unit leaves the game.
"""

from unittest.mock import Mock

import numpy as np
import pygame
import pytest

from reinforcetactics.app.game_loop import GameSession
from reinforcetactics.constants import TILE_SIZE
from reinforcetactics.core.game_state import GameState
from reinforcetactics.ui import theme
from reinforcetactics.ui.menus.in_game.unit_action_menu import UnitActionMenu
from reinforcetactics.ui.renderer import Renderer
from reinforcetactics.ui.sprite_animator import animation_key
from reinforcetactics.utils import settings as settings_module

STEP_MS = theme.UNIT_WALK_MS_PER_TILE


def _plains(size=10, fog_of_war=False):
    grid = np.array([["p"] * size for _ in range(size)], dtype=object)
    grid[0][0] = "h_1"
    grid[size - 1][size - 1] = "h_2"
    return GameState(grid, num_players=2, fog_of_war=fog_of_war)


def _direction(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return {(1, 0): "move_right", (-1, 0): "move_left", (0, 1): "move_down", (0, -1): "move_up"}[(dx, dy)]


def _click(handler, x, y):
    return handler.handle_mouse_click((x * TILE_SIZE + 1, y * TILE_SIZE + 1))


@pytest.fixture
def headless(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, "_settings_instance", settings_module.Settings(str(tmp_path / "settings.json")))
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def clock(monkeypatch):
    """A fake ``pygame.time.get_ticks``: ``clock[0]`` is the time in ms."""
    now = [1000]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: now[0])
    return now


class TestTheEngineReportsThePath:
    def test_listeners_get_the_path_walked(self):
        game = _plains()
        warrior = game.place_unit("W", 2, 2, player=1)
        seen = []
        game.move_listeners.append(lambda unit, path: seen.append((unit, path)))

        assert game.move_unit(warrior, 4, 3)

        [(unit, path)] = seen
        assert unit is warrior
        assert path[0] == (2, 2) and path[-1] == (4, 3)
        assert len(path) == 4  # the shortest path: three steps
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))

    def test_a_refused_move_is_not_reported(self):
        game = _plains()
        warrior = game.place_unit("W", 2, 2, player=1)
        seen = []
        game.move_listeners.append(lambda unit, path: seen.append(path))

        assert not game.move_unit(warrior, 9, 2)  # out of range

        assert seen == []

    def test_an_ambushed_move_reports_the_path_to_where_it_stopped(self):
        game = _plains(size=12, fog_of_war=True)
        barbarian = game.place_unit("B", 4, 7, player=1)
        game.place_unit("W", 7, 7, player=2)  # hidden from player 1, on the path
        seen = []
        game.move_listeners.append(lambda unit, path: seen.append(path))

        assert game.move_unit(barbarian, 9, 7)

        assert barbarian.ambushed and (barbarian.x, barbarian.y) == (6, 7)
        assert seen == [[(4, 7), (5, 7), (6, 7)]]

    def test_search_clones_do_not_call_the_listeners(self):
        game = _plains()
        game.place_unit("W", 2, 2, player=1)
        seen = []
        game.move_listeners.append(lambda unit, path: seen.append(path))

        clone = game.clone_for_search()
        assert clone.move_listeners == []
        assert clone.move_unit(clone.units[0], 3, 2)
        assert seen == []


class TestAnimatorState:
    def test_state_is_keyed_by_the_units_id(self, headless):
        game = _plains()
        animator = Renderer(game, headless=True, pixel_art=True).animator
        warrior = game.place_unit("W", 2, 2, player=1)
        animator.set_unit_state(warrior, "move_up")

        assert animation_key(warrior) == warrior.unit_id
        assert animator.unit_states == {warrior.unit_id: "move_up"}

        # A unit created later has its own key, whatever memory it reuses.
        newcomer = game.place_unit("W", 3, 2, player=1)
        assert animation_key(newcomer) != animation_key(warrior)

        animator.cleanup_unit(warrior)
        assert animator.unit_states == {}

    def test_a_unit_from_an_old_save_falls_back_to_its_object_identity(self):
        game = _plains()
        unit = game.place_unit("W", 2, 2, player=1)
        unit.unit_id = None

        assert animation_key(unit) == ("object", id(unit))

    def test_turning_a_corner_keeps_the_stride(self, headless):
        game = _plains()
        animator = Renderer(game, headless=True, pixel_art=True).animator
        warrior = game.place_unit("W", 2, 2, player=1)
        speed = 0.1  # ANIMATION_CONFIG seconds per walking frame

        animator.set_unit_state(warrior, "move_right")
        animator.get_frame(warrior, 0.0)
        for _ in range(3):
            animator.get_frame(warrior, speed)
        animator.set_unit_state(warrior, "move_down")
        assert animator.animation_timers[warrior.unit_id]["current_frame"] == 3

        animator.set_unit_state(warrior, "idle")
        assert animator.animation_timers[warrior.unit_id]["current_frame"] == 0


class TestTheSpriteWalks:
    def test_it_crosses_the_path_tile_by_tile_facing_each_step(self, headless, clock):
        game = _plains()
        renderer = Renderer(game, headless=True, pixel_art=True)
        game.move_listeners.append(renderer.queue_movement_path_animation)
        warrior = game.place_unit("W", 2, 2, player=1)
        paths = []
        game.move_listeners.append(lambda unit, path: paths.append(path))

        assert game.move_unit(warrior, 3, 3)
        [path] = paths
        assert len(path) == 3

        def state():
            return renderer.animator.unit_states[warrior.unit_id]

        renderer.render()
        assert renderer._unit_origin(warrior) == (2 * TILE_SIZE, 2 * TILE_SIZE)
        assert renderer.is_unit_moving(warrior)
        assert state() == _direction(path[0], path[1])

        clock[0] += STEP_MS + STEP_MS // 2  # half way along the second step
        renderer.render()
        (x0, y0), (x1, y1) = path[1], path[2]
        assert renderer._unit_origin(warrior) == (
            round((x0 + x1) / 2 * TILE_SIZE),
            round((y0 + y1) / 2 * TILE_SIZE),
        )
        assert state() == _direction(path[1], path[2])

        clock[0] += STEP_MS // 2  # arrived
        assert not renderer.is_unit_moving(warrior)
        renderer.render()
        assert renderer._unit_origin(warrior) == (3 * TILE_SIZE, 3 * TILE_SIZE)
        assert state() == "idle"

    def test_it_walks_without_sprite_sheets_too(self, headless, clock):
        game = _plains()
        renderer = Renderer(game, headless=True, pixel_art=False)
        warrior = game.place_unit("W", 4, 2, player=1)

        assert renderer.queue_movement_path_animation(warrior, [(2, 2), (3, 2), (4, 2)])
        clock[0] += STEP_MS // 2
        renderer.render()

        assert renderer._unit_origin(warrior) == (round(2.5 * TILE_SIZE), 2 * TILE_SIZE)

    def test_a_path_without_a_step_is_not_animated(self, headless):
        game = _plains()
        renderer = Renderer(game, headless=True, pixel_art=False)
        warrior = game.place_unit("W", 2, 2, player=1)

        assert not renderer.queue_movement_path_animation(warrior, [(2, 2)])
        assert not renderer.is_unit_moving(warrior)

    def test_a_dead_units_animation_state_is_dropped(self, headless, clock):
        game = _plains()
        renderer = Renderer(game, headless=True, pixel_art=True)
        attacker = game.place_unit("W", 2, 2, player=1)
        victim = game.place_unit("W", 3, 2, player=2)
        victim.health = 1
        renderer.queue_movement_path_animation(victim, [(4, 2), (3, 2)])
        renderer.render()
        assert victim.unit_id in renderer.animator.unit_states

        assert game.attack(attacker, victim)["target_alive"] is False
        renderer.render()

        assert victim.unit_id not in renderer.animator.unit_states
        assert victim.unit_id not in renderer.animator.animation_timers
        assert not renderer.is_unit_moving(victim)
        assert attacker.unit_id in renderer.animator.animation_timers


class TestFogOfWar:
    def test_an_enemy_walk_is_drawn_only_between_tiles_in_sight(self, headless, clock):
        game = _plains(size=12, fog_of_war=True)
        renderer = Renderer(game, headless=True, viewing_player=1)
        in_sight = [(1, 1), (2, 1), (3, 1)]
        fogged = [(9, 8), (9, 9), (10, 9)]
        assert all(game.is_position_visible(x, y, player=1) for x, y in in_sight)
        assert not any(game.is_position_visible(x, y, player=1) for x, y in fogged)

        hidden = game.place_unit("W", *fogged[-1], player=2)
        assert not renderer.queue_movement_path_animation(hidden, fogged)
        renderer.render()
        assert not renderer._unit_in_sight(hidden, 1)

        seen = game.place_unit("W", *in_sight[-1], player=2)
        assert renderer.queue_movement_path_animation(seen, in_sight)
        renderer.render()
        assert renderer._unit_in_sight(seen, 1)

        # Stepping out of the fog: hidden until both ends of the step are in sight.
        emerging = game.place_unit("W", 4, 4, player=2)
        assert not game.is_position_visible(5, 4, player=1) and game.is_position_visible(4, 4, player=1)
        renderer.queue_movement_path_animation(emerging, [(5, 4), (4, 4)])
        renderer.render()
        assert not renderer._unit_in_sight(emerging, 1)


class TestHumanMoves:
    @staticmethod
    def _session(game):
        renderer = Renderer(game, headless=True, pixel_art=False)
        return GameSession(game, renderer, bots={}, num_players=2)

    def test_the_board_waits_for_the_walk_then_opens_the_action_menu(self, headless, clock):
        game = _plains()
        session = self._session(game)
        handler = session.input_handler
        warrior = game.place_unit("W", 2, 2, player=1)
        game.place_unit("W", 6, 6, player=1)

        _click(handler, 2, 2)  # select
        _click(handler, 4, 2)  # move

        assert (warrior.x, warrior.y) == (4, 2)  # the engine moved it at once
        assert handler.walking_unit is warrior and handler.active_menu is None

        # Nothing on the board answers while it walks.
        _click(handler, 6, 6)
        handler.handle_right_click_press((6 * TILE_SIZE, 6 * TILE_SIZE))
        handler.handle_keyboard_event(Mock(key=pygame.K_SPACE))
        assert handler.selected_unit is None and not handler.right_click_preview_active
        assert game.current_player == 1

        handler.update(clock[0])
        assert handler.active_menu is None

        clock[0] += 2 * STEP_MS
        handler.update(clock[0])
        assert isinstance(handler.active_menu, UnitActionMenu)
        assert handler.target_selection_unit is warrior and handler.walking_unit is None
        assert handler.menu_opened_time == clock[0]

    def test_escape_still_pauses_during_the_walk(self, headless, clock):
        game = _plains()
        handler = self._session(game).input_handler
        game.place_unit("W", 2, 2, player=1)
        _click(handler, 2, 2)
        _click(handler, 4, 2)

        assert handler.handle_keyboard_event(Mock(key=pygame.K_ESCAPE)) == "pause"


class _ScriptedBot:
    """Moves each of its ``moves`` in turn, noting which earlier walks were still on, then ends its turn."""

    def __init__(self, game, renderer, moves):
        self.game, self.renderer, self.moves = game, renderer, moves
        self.walking_before_move = []

    def take_turn(self):
        moved = []
        for unit, x, y in self.moves:
            self.walking_before_move.append([self.renderer.is_unit_moving(u) for u in moved])
            assert self.game.move_unit(unit, x, y)
            moved.append(unit)
        self.game.end_turn()


class TestBotMoves:
    @staticmethod
    def _bot_turn(game, moves, clock, monkeypatch):
        """Play one turn of a scripted bot for player 2 in a GameSession; return (bot, frames rendered)."""
        renderer = Renderer(game, headless=True, pixel_art=False)
        game.end_turn()  # player 2 (the bot) to move
        bot = _ScriptedBot(game, renderer, [(game.get_unit_at_position(*src), *dst) for src, dst in moves])
        session = GameSession(game, renderer, {2: bot}, num_players=2)
        frames = [0]
        original = session._render_frame

        def render_frame():
            frames[0] += 1
            original()

        monkeypatch.setattr(session, "_render_frame", render_frame)
        # Each frame takes 16 ms of the fake clock.
        monkeypatch.setattr(session, "clock", Mock(tick=lambda fps: clock.__setitem__(0, clock[0] + 16)))
        session.input_handler._process_bot_turns(max_turns=1)
        return session, bot, frames[0]

    def test_each_move_plays_out_before_the_next(self, headless, clock, monkeypatch):
        game = _plains()
        game.place_unit("W", 7, 7, player=2)
        game.place_unit("W", 8, 6, player=2)

        _session, bot, frames = self._bot_turn(game, [((7, 7), (5, 7)), ((8, 6), (8, 4))], clock, monkeypatch)

        assert bot.walking_before_move == [[], [False]]
        assert frames >= 2 * (2 * STEP_MS) // 16
        assert game.current_player == 1

    def test_the_human_sees_the_bot_turn_through_their_own_fog(self, headless, clock, monkeypatch):
        game = _plains(size=12, fog_of_war=True)
        game.place_unit("W", 10, 10, player=2)  # its whole move stays out of player 1's sight

        session, _bot, frames = self._bot_turn(game, [((10, 10), (10, 8))], clock, monkeypatch)

        assert session.renderer.viewing_player == 1  # the move was judged from the human's side
        assert frames == 0  # nothing to watch, so nothing waited for

    def test_pausing_during_a_bot_walk_stops_the_waiting(self, headless, clock, monkeypatch):
        game = _plains()
        game.place_unit("W", 7, 7, player=2)
        game.place_unit("W", 8, 6, player=2)
        pygame.event.clear()
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode="\x1b"))

        session, bot, frames = self._bot_turn(game, [((7, 7), (5, 7)), ((8, 6), (8, 4))], clock, monkeypatch)

        assert frames == 1  # one frame of the first walk, then no more waiting
        assert bot.walking_before_move == [[], [True]]
        # The key is kept for the main loop, which opens the pause menu.
        assert [e.key for e in pygame.event.get(pygame.KEYDOWN)] == [pygame.K_ESCAPE]

    def test_the_board_keeps_the_humans_view_while_a_bot_moves(self, headless):
        game = _plains(size=12, fog_of_war=True)
        renderer = Renderer(game, headless=True, pixel_art=False)
        session = GameSession(game, renderer, {2: Mock()}, num_players=2)

        session._update_view()
        assert renderer._get_fow_player() == 1

        game.end_turn()
        session._update_view()
        assert game.current_player == 2 and renderer._get_fow_player() == 1

    def test_with_only_bots_each_bot_keeps_its_own_view(self, headless):
        game = _plains(size=12, fog_of_war=True)
        renderer = Renderer(game, headless=True, pixel_art=False)
        session = GameSession(game, renderer, {1: Mock(), 2: Mock()}, num_players=2)

        game.end_turn()
        session._update_view()
        assert renderer._get_fow_player() == 2
