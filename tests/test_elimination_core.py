"""Multi-player elimination (review core-7).

With more than two teams, capturing a player's (last) HQ, killing its last
unit or its resignation eliminates that player: its units go, its structures
turn neutral, end_turn skips its seat (no turns, income or new units), and the
game goes on until one team is left. Before this, any HQ capture ended a
free-for-all outright while resigned or wiped-out players kept taking turns,
collecting income and rebuilding. Two-team games are unchanged.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.utils import replay_actions
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.replay_actions import execute_replay_action

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"


def _ffa_map():
    """10x10, three players each with an HQ and a building."""
    grid = np.full((10, 10), "p", dtype=object)
    grid[0, 0], grid[0, 1] = "h_1", "b_1"
    grid[0, 9], grid[0, 8] = "h_2", "b_2"
    grid[9, 5], grid[9, 4] = "h_3", "b_3"
    return grid


@pytest.fixture
def ffa():
    game = GameState(_ffa_map(), num_players=3)
    return game


def _capture_hq(game, capturer_player, hq_xy):
    """Seize the HQ at ``hq_xy`` with a fresh Warrior of ``capturer_player`` (HQ at 1 HP)."""
    tile = game.grid.get_tile(*hq_xy)
    tile.health = 1
    unit = game.place_unit("W", hq_xy[0], hq_xy[1], capturer_player)
    return game.seize(unit)


class TestHqCaptureInAFreeForAll:
    def test_eliminates_the_owner_and_play_goes_on(self, ffa):
        victim_unit = ffa.place_unit("W", 5, 5, 2)
        ffa.place_unit("W", 6, 6, 3)

        result = _capture_hq(ffa, 1, (9, 0))

        assert result["captured"] and not result["game_over"]
        assert not ffa.game_over
        assert ffa.eliminated_players == {2}
        assert victim_unit not in ffa.units
        # The captured HQ is the capturer's; the rest of player 2's holdings are neutral.
        assert ffa.grid.get_tile(9, 0).player == 1
        assert ffa.grid.get_tile(8, 0).player is None
        assert [a["type"] for a in ffa.action_history] == ["seize", "eliminate"]
        assert ffa.action_history[-1]["eliminated_player"] == 2

    def test_end_turn_skips_the_eliminated_seat(self, ffa):
        ffa.place_unit("W", 6, 6, 3)
        _capture_hq(ffa, 1, (9, 0))
        gold_2 = ffa.player_gold[2]

        seats = []
        for _ in range(4):
            ffa.end_turn()
            seats.append(ffa.current_player)

        assert seats == [3, 1, 3, 1]
        assert ffa.player_gold[2] == gold_2  # no income for a seat out of the game
        assert ffa.get_legal_actions(2)["create_unit"] == []

    def test_the_last_hq_capture_ends_the_game(self, ffa):
        _capture_hq(ffa, 1, (9, 0))
        ffa.end_turn()  # -> player 3
        ffa.end_turn()  # -> player 1

        result = _capture_hq(ffa, 1, (5, 9))

        assert result["game_over"]
        assert ffa.game_over and ffa.winner == 1 and ffa.end_reason == "hq_capture"
        assert ffa.eliminated_players == {2, 3}

    def test_a_neutral_hq_is_just_a_structure(self, ffa):
        ffa.place_unit("W", 6, 6, 3)
        ffa.resign(2)  # player 2's HQ turns neutral
        assert ffa.grid.get_tile(9, 0).player is None

        result = _capture_hq(ffa, 1, (9, 0))

        assert result["captured"] and not ffa.game_over
        assert ffa.eliminated_players == {2}


class TestOtherEliminations:
    def test_resignation_eliminates_without_ending_a_free_for_all(self, ffa):
        ffa.place_unit("W", 5, 5, 2)

        ffa.resign(2)

        assert not ffa.game_over and ffa.eliminated_players == {2}
        assert all(u.player != 2 for u in ffa.units)
        assert all(t.player != 2 for row in ffa.grid.tiles for t in row)
        ffa.end_turn()  # player 1 -> player 3, skipping player 2
        assert ffa.current_player == 3
        ffa.resign()  # player 3, on its own turn
        assert ffa.game_over and ffa.winner == 1 and ffa.end_reason == "resign"

    def test_losing_the_last_unit_eliminates(self, ffa):
        attacker = ffa.place_unit("K", 4, 4, 1)
        victim = ffa.place_unit("W", 5, 4, 2)
        victim.health = 1
        ffa.place_unit("W", 8, 8, 3)

        ffa.attack(attacker, victim)

        assert ffa.eliminated_players == {2} and not ffa.game_over
        assert ffa.grid.get_tile(9, 0).player is None

    def test_game_ends_when_one_team_is_left(self):
        grid = np.full((10, 10), "p", dtype=object)
        grid[0, 0], grid[0, 9], grid[9, 0], grid[9, 9] = "h_1_1", "h_2_2", "h_4_2", "h_3_1"
        game = GameState(grid, num_players=4)
        game.resign(2)
        assert not game.game_over  # player 4 still holds team 2's side

        game.resign(4)

        assert game.game_over and game.end_reason == "resign"
        assert game.winner in (1, 3) and game.are_allies(game.winner, 1)


class TestTwoTeamGamesUnchanged:
    def test_1v1_hq_capture_still_ends_the_game_with_no_extra_record(self):
        game = GameState(FileIO.load_map(str(MAPS_DIR / "1v1" / "beginner.csv")), num_players=2)
        hq = next(t for row in game.grid.tiles for t in row if t.type == "h" and t.player == 2)

        _capture_hq(game, 1, (hq.x, hq.y))

        assert game.game_over and game.winner == 1 and game.end_reason == "hq_capture"
        assert [a["type"] for a in game.action_history] == ["seize"]

    def test_1v1_resign_and_elimination_winners(self):
        game = GameState(FileIO.load_map(str(MAPS_DIR / "1v1" / "beginner.csv")), num_players=2)
        game.resign(1)
        assert game.game_over and game.winner == 2 and game.end_reason == "resign"
        assert [a["type"] for a in game.action_history] == ["resign"]

        game = GameState(np.full((6, 6), "p", dtype=object), num_players=2)
        attacker = game.place_unit("W", 2, 2, 1)
        victim = game.place_unit("W", 3, 2, 2)
        attacker.health = 1  # dies to the counter
        game.attack(attacker, victim)
        assert game.game_over and game.winner == 2 and game.end_reason == "elimination"
        assert game.game_over_action_index == len(game.action_history) - 1


class TestPersistenceAndReplay:
    def test_save_round_trip(self, ffa):
        ffa.place_unit("W", 6, 6, 3)
        _capture_hq(ffa, 1, (9, 0))

        restored = GameState.from_dict(json.loads(json.dumps(ffa.to_dict())))

        assert restored.eliminated_players == {2}
        # Player 2's building stays neutral after the reload.
        assert restored.grid.get_tile(8, 0).player is None
        restored.end_turn()
        assert restored.current_player == 3

    def test_replay_reproduces_eliminations(self):
        game = GameState(FileIO.load_map(str(MAPS_DIR / "1v1v1" / "triangle_arena.csv")), num_players=3)
        # Build, fight and capture through the engine only, so the log holds everything.
        hq2 = next(t for row in game.grid.tiles for t in row if t.type == "h" and t.player == 2)
        spawn = next(t for row in game.grid.tiles for t in row if t.type == "b" and t.player == 1)
        game.create_unit("W", spawn.x, spawn.y)
        game.end_turn()
        spawn2 = next(t for row in game.grid.tiles for t in row if t.type == "b" and t.player == 2)
        game.create_unit("W", spawn2.x, spawn2.y)
        game.end_turn()
        game.end_turn()
        # Walk player 1's Warrior next to player 2's HQ via direct moves along legal destinations.
        warrior = next(u for u in game.units if u.player == 1)
        for _ in range(40):
            if game.game_over or (warrior.x, warrior.y) == (hq2.x, hq2.y):
                break
            if game.current_player == 1:
                dests = [(a["to_x"], a["to_y"]) for a in game.get_legal_actions(1)["move"] if a["unit"] is warrior]
                if dests:
                    best = min(dests, key=lambda p: abs(p[0] - hq2.x) + abs(p[1] - hq2.y))
                    game.move_unit(warrior, *best)
            game.end_turn()
        while game.current_player != 1:
            game.end_turn()
        hq2.health = 1
        game.seize(warrior)
        assert game.eliminated_players == {2}
        game.end_turn()
        game.resign(3)
        assert game.game_over and game.winner == 1

        info = {"num_players": 3, "replay_schema_version": 3, "teams": game.teams}
        replay = GameState(
            FileIO.load_map(str(MAPS_DIR / "1v1v1" / "triangle_arena.csv")), **replay_actions.replay_game_state_kwargs(info)
        )
        for action in game.action_history:
            execute_replay_action(replay, action, lambda x, y: (x, y), schema_version=3)

        assert replay.eliminated_players == game.eliminated_players
        assert replay.game_over and replay.winner == game.winner and replay.end_reason == game.end_reason
        assert sorted((u.player, u.x, u.y) for u in replay.units) == sorted((u.player, u.x, u.y) for u in game.units)
