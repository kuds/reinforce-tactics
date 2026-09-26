"""Fog of war shows each player only what it knows (review core-5, core-12, core-21,
critic-integration-3, pygame-12).

``GameState.known_structure`` is the one engine view of what a player knows
about a structure: live while in sight, as last seen once out of sight, and
every HQ from the start. The RL observation, the renderer and the LLM prompt
all read it, so a capture made out of sight reaches none of them. Pathfinding
plans around the units a player can see, so a hidden enemy no longer shapes
the move mask; a move that runs into one is ambushed and stops short.
"""

from unittest.mock import Mock

import numpy as np
import pygame
import pytest

from reinforcetactics.app.input_handler import InputHandler
from reinforcetactics.constants import BUILDING_MAX_HEALTH, TILE_SIZE, TOWER_MAX_HEALTH
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.visibility import SHROUDED, UNEXPLORED, VISIBLE, calculate_vision_radius
from reinforcetactics.game.llm_bot import LLMBot
from reinforcetactics.rl.observation import build_observation
from reinforcetactics.ui.menus.in_game.unit_action_menu import UnitActionMenu
from reinforcetactics.ui.renderer import Renderer
from reinforcetactics.utils import settings as settings_module

# 12x12 grass. P1's HQ (vision 4) sees x, y <= 4; P2's HQ sees x, y >= 7.
BUILDING = (8, 1)  # neutral, out of both HQs' sight
TOWER = (1, 8)  # neutral, out of both HQs' sight


def _map():
    grid = np.array([["p"] * 12 for _ in range(12)], dtype=object)
    grid[0][0] = "h_1"
    grid[11][11] = "h_2"
    grid[BUILDING[1]][BUILDING[0]] = "b"
    grid[TOWER[1]][TOWER[0]] = "t"
    return grid


@pytest.fixture
def fow_game():
    game = GameState(_map(), num_players=2, fog_of_war=True)
    game.player_gold = {1: 10000, 2: 10000}
    return game


def _tile(game, pos):
    return game.grid.get_tile(*pos)


def _scout_building_then_leave(game):
    """P1's Archer sees BUILDING from (6, 1), then walks out of sight to (3, 1)."""
    archer = game.place_unit("A", 6, 1, player=1)
    assert game.is_position_visible(*BUILDING, player=1)
    assert game.move_unit(archer, 3, 1)
    assert game.visibility_maps[1].get_visibility_state(*BUILDING) == SHROUDED
    return archer


def _moves(game, player):
    return sorted((a["from_x"], a["from_y"], a["to_x"], a["to_y"]) for a in game.get_legal_actions(player)["move"])


def _same_arrays(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=key)


class TestKnownStructure:
    def test_without_fog_every_structure_is_known_live(self):
        game = GameState(_map(), num_players=2)
        _tile(game, BUILDING).player = 2

        known = game.known_structure(1, *BUILDING)

        assert (known.owner, known.health, known.turn_seen) == (2, BUILDING_MAX_HEALTH, 0)
        assert game.known_structure(1, 5, 5) is None  # grass

    def test_every_hq_is_known_from_the_start(self, fow_game):
        """The documented rule "enemy HQ is always known": location and owner, not its later state."""
        known = fow_game.known_structure(1, 11, 11)

        assert not fow_game.is_position_visible(11, 11, player=1)
        assert fow_game.is_position_explored(11, 11, player=1)
        assert (known.tile_type, known.owner, known.turn_seen) == ("h", 2, 0)
        obs = fow_game.to_numpy(for_player=1)
        assert obs["grid"][11, 11, 0] == 6  # HQ terrain code
        assert obs["grid"][11, 11, 1] == 2

    def test_a_structure_never_seen_is_unknown(self, fow_game):
        assert fow_game.known_structure(1, *BUILDING) is None
        assert fow_game.known_structure(1, *TOWER) is None

    def test_a_structure_out_of_sight_is_known_as_last_seen(self, fow_game):
        _scout_building_then_leave(fow_game)
        _tile(fow_game, BUILDING).player = 2  # captured out of sight
        _tile(fow_game, BUILDING).health = 12

        known = fow_game.known_structure(1, *BUILDING)

        assert (known.owner, known.health, known.turn_seen) == (None, BUILDING_MAX_HEALTH, 0)
        assert fow_game.known_structure(2, *BUILDING) is None  # P2 has never seen it

    def test_a_change_watched_before_losing_sight_is_remembered(self, fow_game):
        """Memory is taken when the structure leaves sight, not at the previous update."""
        archer = fow_game.place_unit("A", 6, 1, player=1)
        tile = _tile(fow_game, BUILDING)
        tile.player, tile.health = 2, 12  # happens in plain sight of the Archer

        assert fow_game.move_unit(archer, 3, 1)

        known = fow_game.known_structure(1, *BUILDING)
        assert (known.owner, known.health) == (2, 12)
        obs = fow_game.to_numpy(for_player=1)
        assert obs["grid"][BUILDING[1], BUILDING[0], 1] == 2
        assert obs["grid"][BUILDING[1], BUILDING[0], 2] == pytest.approx(100 * 12 / BUILDING_MAX_HEALTH)


class TestObservationHidesWhatThePlayerCannotSee:
    def test_a_capture_out_of_sight_does_not_change_the_observation(self, fow_game):
        _scout_building_then_leave(fow_game)
        before = fow_game.to_numpy(for_player=1)
        before_rl = build_observation(fow_game, 1)

        tile = _tile(fow_game, BUILDING)
        tile.player, tile.health = 2, 12

        _same_arrays(before, fow_game.to_numpy(for_player=1))
        _same_arrays(before_rl, build_observation(fow_game, 1))
        # ... and it shows the building as last seen: neutral, full HP.
        assert before["grid"][BUILDING[1], BUILDING[0], 1] == 0
        assert before["grid"][BUILDING[1], BUILDING[0], 2] == 100

    def test_an_engine_capture_out_of_sight_does_not_reach_the_observation(self, fow_game):
        """The same, driven through seize (which refreshes every player's visibility)."""
        _scout_building_then_leave(fow_game)
        raider = fow_game.place_unit("W", *BUILDING, player=2)
        _tile(fow_game, BUILDING).health = 5  # so one seize captures it
        fow_game.end_turn()
        before = build_observation(fow_game, 1)

        result = fow_game.seize(raider)

        assert result["captured"] and _tile(fow_game, BUILDING).player == 2
        _same_arrays(before, build_observation(fow_game, 1))
        assert fow_game.known_structure(1, *BUILDING).owner is None

    def test_a_hidden_unit_changes_neither_observation_nor_move_mask(self, fow_game):
        barbarian = fow_game.place_unit("B", 4, 7, player=1)  # moves 5, sees 2
        before_obs = fow_game.to_numpy(for_player=1)
        before_moves = _moves(fow_game, 1)
        assert (9, 7) in {(m[2], m[3]) for m in before_moves}

        fow_game.place_unit("W", 7, 7, player=2)  # on the Barbarian's path, out of its sight
        assert not fow_game.is_position_visible(7, 7, player=1)

        _same_arrays(before_obs, fow_game.to_numpy(for_player=1))
        assert _moves(fow_game, 1) == before_moves
        assert barbarian.can_move

    def test_a_visible_enemy_still_blocks(self, fow_game):
        fow_game.place_unit("B", 4, 7, player=1)
        fow_game.place_unit("W", 5, 7, player=2)  # adjacent, so in sight

        destinations = {(m[2], m[3]) for m in _moves(fow_game, 1)}

        assert (5, 7) not in destinations
        assert (9, 7) not in destinations  # the only 5-step route runs through it

    def test_never_explored_tiles_are_blank(self, fow_game):
        obs = fow_game.to_numpy(for_player=1)
        unexplored = obs["visibility"] == UNEXPLORED

        assert unexplored[TOWER[1], TOWER[0]]
        assert not obs["grid"][unexplored].any()


class TestAmbush:
    def test_a_move_through_a_hidden_enemy_stops_on_the_tile_before_it(self, fow_game):
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        ambusher = fow_game.place_unit("W", 7, 7, player=2)

        assert fow_game.move_unit(barbarian, 9, 7)

        assert (barbarian.x, barbarian.y) == (6, 7)
        assert not barbarian.can_move  # the move is spent
        assert fow_game.is_position_visible(7, 7, player=1)  # the ambusher is revealed
        record = fow_game.action_history[-1]
        assert (record["type"], record["to_x"], record["to_y"], record["ambushed"]) == ("move", 6, 7, True)
        # Discovered by moving, so not attackable this action (the FOW pre-move snapshot rule).
        assert ambusher not in [a["target"] for a in fow_game.get_legal_actions(1)["attack"]]

    def test_a_hidden_enemy_on_the_destination_stops_the_move_next_to_it(self, fow_game):
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        fow_game.place_unit("W", 7, 7, player=2)

        assert fow_game.move_unit(barbarian, 7, 7)

        assert (barbarian.x, barbarian.y) == (6, 7)

    def test_a_clear_path_is_not_an_ambush(self, fow_game):
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        fow_game.place_unit("W", 7, 9, player=2)  # hidden, but off the path

        assert fow_game.move_unit(barbarian, 9, 7)

        assert (barbarian.x, barbarian.y) == (9, 7)
        assert "ambushed" not in fow_game.action_history[-1]

    def test_without_fog_enemies_block_as_before(self):
        game = GameState(_map(), num_players=2)
        barbarian = game.place_unit("B", 4, 7, player=1)
        game.place_unit("W", 7, 7, player=2)

        assert not game.move_unit(barbarian, 9, 7)
        assert (barbarian.x, barbarian.y) == (4, 7) and barbarian.can_move

    def test_an_ambushed_move_cannot_be_cancelled(self, fow_game):
        """The move is spent: cancelling it would be free scouting (move, see the ambusher, undo, re-plan)."""
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        fow_game.place_unit("W", 7, 7, player=2)
        assert fow_game.move_unit(barbarian, 9, 7)
        assert barbarian.ambushed and barbarian.has_moved

        assert not fow_game.can_cancel_move(barbarian)
        assert fow_game.cancel_move(barbarian) is False

        assert (barbarian.x, barbarian.y) == (6, 7)
        assert not barbarian.can_move
        assert not any(m[:2] == (6, 7) for m in _moves(fow_game, 1))  # no second move either

    def test_reselecting_an_ambushed_unit_does_not_make_the_ambusher_a_target(self, fow_game):
        """The GUI re-captures the attack snapshot when a unit is selected; after a move it must keep the old one."""
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        ambusher = fow_game.place_unit("W", 7, 7, player=2)
        assert fow_game.move_unit(barbarian, 9, 7)

        fow_game.capture_visible_enemies_for_unit(barbarian)  # what selecting it in the GUI does

        assert (7, 7) not in barbarian.visible_enemies_at_action_start
        assert ambusher not in [a["target"] for a in fow_game.get_legal_actions(1)["attack"]]

    def test_the_units_next_action_is_cancellable_again(self, fow_game):
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        fow_game.place_unit("W", 7, 7, player=2)
        assert fow_game.move_unit(barbarian, 9, 7)
        fow_game.end_unit_turn(barbarian)
        assert not barbarian.ambushed
        fow_game.end_turn()
        fow_game.end_turn()

        assert fow_game.move_unit(barbarian, 6, 9)  # a clear path this time

        assert fow_game.can_cancel_move(barbarian)
        assert fow_game.cancel_move(barbarian)
        assert (barbarian.x, barbarian.y) == (6, 7) and barbarian.can_move


class TestVisibilityIsAlwaysCurrent:
    """core-12: the engine refreshes visibility itself whenever vision can change."""

    def test_a_new_game_starts_with_computed_visibility(self):
        game = GameState(_map(), num_players=2, fog_of_war=True)  # no update_visibility() call

        assert game.is_position_visible(4, 4, player=1)
        assert game.is_position_visible(7, 7, player=2)

    def test_create_unit_gives_vision_at_once(self):
        grid = _map()
        grid[3][5] = "b_1"  # P1 building; its own vision reaches x <= 8
        game = GameState(grid, num_players=2, fog_of_war=True)
        game.player_gold[1] = 1000
        assert not game.is_position_visible(9, 3, player=1)

        assert game.create_unit("A", 5, 3, player=1) is not None

        assert game.is_position_visible(9, 3, player=1)  # Archer vision 4

    def test_a_capture_moves_the_structures_vision_to_the_capturer(self):
        grid = _map()
        grid[TOWER[1]][TOWER[0]] = "t_2"
        game = GameState(grid, num_players=2, fog_of_war=True)
        warrior = game.place_unit("W", *TOWER, player=1)
        _tile(game, TOWER).health = 5
        far = (TOWER[0] + 5, TOWER[1])  # tower vision 5, Warrior vision 3
        assert game.is_position_visible(*far, player=2)
        assert not game.is_position_visible(*far, player=1)

        assert game.seize(warrior)["captured"]

        assert game.is_position_visible(*far, player=1)
        assert not game.is_position_visible(*far, player=2)

    def test_a_dead_unit_stops_giving_vision(self, fow_game):
        scout = fow_game.place_unit("C", 8, 8, player=1)
        scout.health = 1
        killer = fow_game.place_unit("W", 9, 8, player=2)
        assert fow_game.is_position_visible(10, 10, player=1)
        fow_game.end_turn()

        assert not fow_game.attack(killer, scout)["target_alive"]

        assert not fow_game.is_position_visible(10, 10, player=1)
        assert not fow_game.is_position_visible(9, 8, player=1)

    def test_cancel_move_takes_back_the_vision_the_move_gave(self, fow_game):
        """A cancelled scout left its tiles VISIBLE, so known_structure served (and memorised) live state."""
        archer = fow_game.place_unit("A", 3, 1, player=1)  # vision 4: sees x <= 7
        assert not fow_game.is_position_visible(*BUILDING, player=1)
        assert fow_game.move_unit(archer, 5, 1)
        assert fow_game.is_position_visible(*BUILDING, player=1)

        assert fow_game.cancel_move(archer)

        assert fow_game.visibility_maps[1].get_visibility_state(*BUILDING) == SHROUDED  # seen, now out of sight
        fow_game.end_turn()
        tile = _tile(fow_game, BUILDING)
        tile.player, tile.health = 2, 5  # captured out of P1's sight
        assert fow_game.to_numpy(for_player=1)["grid"][BUILDING[1], BUILDING[0], 1] == 0
        fow_game.end_turn()
        assert fow_game.known_structure(1, *BUILDING).owner is None

    def test_a_bare_update_visibility_invalidates_the_legal_actions(self, fow_game):
        """Legality under fog reads visibility, so any refresh (not just the engine's own) must drop the cache."""
        barbarian = fow_game.place_unit("B", 5, 1, player=1)  # moves 5, sees 2
        fow_game.place_unit("W", 8, 3, player=2)  # hidden, so a legal destination
        assert (5, 1, 8, 3) in _moves(fow_game, 1)
        _tile(fow_game, BUILDING).player = 1  # e.g. a scenario script; building vision 3 covers (8, 3)

        fow_game.update_visibility()

        assert fow_game.is_position_visible(8, 3, player=1)
        assert (5, 1, 8, 3) not in _moves(fow_game, 1)
        assert barbarian.can_move


class TestVisibilityUpdateCost:
    """core-21: update() visits units and structures, never every tile."""

    def test_update_matches_a_brute_force_chebyshev_scan(self, fow_game):
        rng = np.random.default_rng(3)
        for _ in range(12):
            x, y = (int(v) for v in rng.integers(0, 12, 2))
            if fow_game.get_unit_at_position(x, y) is None:
                fow_game.place_unit(str(rng.choice(list("WMCAKRSB"))), x, y, player=int(rng.integers(1, 3)))

        for player in (1, 2):
            expected = np.zeros((12, 12), dtype=bool)
            sources = [
                (u.x, u.y, calculate_vision_radius(u.type, _tile(fow_game, (u.x, u.y)).type))
                for u in fow_game.units
                if u.player == player
            ]
            sources += [
                (t.x, t.y, calculate_vision_radius(t.type, is_structure=True))
                for row in fow_game.grid.tiles
                for t in row
                if t.is_capturable() and t.player == player
            ]
            for sx, sy, r in sources:
                for y in range(12):
                    for x in range(12):
                        expected[y, x] |= max(abs(x - sx), abs(y - sy)) <= r
            np.testing.assert_array_equal(fow_game.visibility_maps[player].state == VISIBLE, expected)

    def test_update_does_not_scan_the_board(self, fow_game, monkeypatch):
        fow_game.place_unit("W", 5, 5, player=1)
        calls = []
        original = fow_game.grid.get_tile
        monkeypatch.setattr(fow_game.grid, "get_tile", lambda x, y: calls.append((x, y)) or original(x, y))

        fow_game.visibility_maps[1].update(fow_game)

        assert len(calls) <= len(fow_game.units)  # one terrain lookup per own unit at most


class _SilentLLMBot(LLMBot):
    def _get_api_key_from_env(self):
        return "test-key"

    def _get_env_var_name(self):
        return "TEST_API_KEY"

    def _get_default_model(self):
        return "test-model"

    def _get_supported_models(self):
        return ["test-model"]

    def _call_llm(self, messages):
        return '{"actions": []}'

    def _get_llm_sdk_version(self):
        return "test-sdk-1.0.0"


def _buildings(state, key):
    return {tuple(b["position"]): b for b in state[key]}


class TestLLMPromptUsesTheSameView:
    """critic-integration-3: the LLM serializer reported the live owner of shrouded structures."""

    def test_a_capture_out_of_sight_is_not_reported(self, fow_game):
        _scout_building_then_leave(fow_game)
        tile = _tile(fow_game, BUILDING)
        tile.player, tile.health = 2, 12

        state = _SilentLLMBot(fow_game, player=1, api_key="test-key")._serialize_game_state()

        assert BUILDING not in _buildings(state, "enemy_buildings")
        entry = _buildings(state, "neutral_buildings")[BUILDING]
        assert (entry["last_seen"], entry["turn_seen"], entry["hp"]) == (True, 0, BUILDING_MAX_HEALTH)

    def test_the_enemy_hq_is_known_from_the_start(self, fow_game):
        state = _SilentLLMBot(fow_game, player=1, api_key="test-key")._serialize_game_state()

        assert _buildings(state, "enemy_buildings")[(11, 11)]["last_seen"] is True
        assert TOWER not in _buildings(state, "neutral_buildings")  # never seen

    def test_without_fog_the_prompt_is_unchanged(self):
        game = GameState(_map(), num_players=2)
        _tile(game, TOWER).player = 2

        state = _SilentLLMBot(game, player=1, api_key="test-key")._serialize_game_state()

        assert _buildings(state, "enemy_buildings")[TOWER] == {"type": "t", "position": [1, 8], "income": 50}

    @staticmethod
    def _seize_offers(game):
        state = _SilentLLMBot(game, player=1, api_key="test-key")._serialize_game_state()
        return {tuple(a["move_to"]) for a in state["legal_actions"]["move_then_seize"]}

    def test_move_then_seize_is_not_offered_on_a_structure_never_seen(self, fow_game):
        """The move set reaches tiles the bot has never seen; a then_seize hint there revealed the structure."""
        fow_game.place_unit("B", 4, 7, player=1)  # moves 5, sees 2: TOWER is reachable but unexplored
        assert fow_game.known_structure(1, *TOWER) is None
        assert TOWER in {(m[2], m[3]) for m in _moves(fow_game, 1)}

        assert TOWER not in self._seize_offers(fow_game)

    def test_move_then_seize_follows_the_known_owner(self, fow_game):
        fow_game.place_unit("B", 4, 7, player=1)
        fow_game.place_unit("A", TOWER[0], TOWER[1] - 3, player=1)  # vision 4: sees the tower
        assert fow_game.known_structure(1, *TOWER).owner is None

        assert TOWER in self._seize_offers(fow_game)  # a known neutral tower
        _tile(fow_game, TOWER).player = 1
        assert TOWER not in self._seize_offers(fow_game)  # its own

    def test_without_fog_move_then_seize_is_unchanged(self):
        game = GameState(_map(), num_players=2)
        game.place_unit("B", 4, 7, player=1)

        assert TOWER in self._seize_offers(game)


@pytest.fixture
def headless(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, "_settings_instance", settings_module.Settings(str(tmp_path / "settings.json")))
    pygame.init()
    yield
    pygame.quit()


class TestRendererDrawsTheKnownOwner:
    """pygame-12 (structure owners): a fogged structure's live owner showed through the fog."""

    @staticmethod
    def _tile_pixels(renderer, tile):
        renderer.screen.fill((0, 0, 0))
        renderer._draw_tile(tile, 1)
        rect = pygame.Rect(tile.x * TILE_SIZE, tile.y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        return pygame.surfarray.array3d(renderer.screen.subsurface(rect)).copy()

    @pytest.mark.parametrize("pixel_art", [False, True])
    def test_a_fogged_structure_looks_the_same_whoever_owns_it(self, headless, fow_game, pixel_art):
        renderer = Renderer(fow_game, headless=True, pixel_art=pixel_art, viewing_player=1)
        tower = _tile(fow_game, TOWER)
        assert fow_game.visibility_maps[1].get_visibility_state(*TOWER) == UNEXPLORED
        assert tower.health == TOWER_MAX_HEALTH

        neutral = self._tile_pixels(renderer, tower)
        tower.player = 2
        captured = self._tile_pixels(renderer, tower)

        assert (neutral == captured).all()

    @pytest.mark.parametrize("pixel_art", [False, True])
    def test_a_visible_structure_shows_its_owner(self, headless, fow_game, pixel_art):
        renderer = Renderer(fow_game, headless=True, pixel_art=pixel_art, viewing_player=1)
        fow_game.place_unit("A", TOWER[0], TOWER[1] - 2, player=1)
        tower = _tile(fow_game, TOWER)
        assert fow_game.is_position_visible(*TOWER, player=1)

        neutral = self._tile_pixels(renderer, tower)
        tower.player = 2
        captured = self._tile_pixels(renderer, tower)

        assert not (neutral == captured).all()


class TestGuiFollowsTheAmbushRule:
    """The GUI must not undo an ambush, nor draw moves by units its player can't see."""

    @staticmethod
    def _click(handler, x, y):
        return handler._handle_grid_click((x * TILE_SIZE + 1, y * TILE_SIZE + 1), current_time=0)

    def test_an_ambushed_unit_gets_a_notice_and_no_cancel_option(self, headless, fow_game):
        screen = pygame.display.set_mode((TILE_SIZE * 12, TILE_SIZE * 12))
        renderer = Mock()
        renderer.screen = screen
        handler = InputHandler(fow_game, renderer, bots={}, num_players=2)
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        ambusher = fow_game.place_unit("W", 7, 7, player=2)

        self._click(handler, 4, 7)  # select
        self._click(handler, 9, 7)  # move: ambushed at (6, 7)

        assert (barbarian.x, barbarian.y) == (6, 7)
        assert handler.notice_text.startswith("Ambushed!")
        assert isinstance(handler.active_menu, UnitActionMenu)
        assert "cancel_move" not in {a["type"] for a in handler.active_menu.actions}

        # ESC closes the menu; the move stays spent and the unit where it stopped.
        handler.handle_keyboard_event(Mock(key=pygame.K_ESCAPE))
        assert handler.active_menu is None
        assert (barbarian.x, barbarian.y) == (6, 7) and not barbarian.can_move

        # Selecting it again and opening its menu offers no attack on the ambusher.
        self._click(handler, 6, 7)
        self._click(handler, 6, 7)
        assert isinstance(handler.active_menu, UnitActionMenu)
        attack = [a for a in handler.active_menu.actions if a["type"] == "attack"]
        assert not attack or ambusher not in attack[0]["targets"]

    def test_the_movement_overlay_ignores_units_the_player_cannot_see(self, headless, fow_game, monkeypatch):
        renderer = Renderer(fow_game, headless=True, viewing_player=1)
        barbarian = fow_game.place_unit("B", 4, 7, player=1)
        drawn = []
        original = barbarian.get_reachable_positions
        monkeypatch.setattr(
            barbarian, "get_reachable_positions", lambda *a, **k: drawn.append(sorted(original(*a, **k))) or drawn[-1]
        )

        renderer.draw_movement_overlay(barbarian)
        fow_game.place_unit("W", 7, 7, player=2)  # hidden on the Barbarian's path
        renderer.draw_movement_overlay(barbarian)

        assert drawn[0] == drawn[1]  # no hole where the hidden enemy stands
        assert {(m[2], m[3]) for m in _moves(fow_game, 1)} <= set(drawn[1])  # every legal move is drawn
