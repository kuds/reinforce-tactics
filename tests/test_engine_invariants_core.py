"""Seeded random play with engine invariants checked after every action (review core-17).

Drives bot mixes (RandomBot, BalancedRandomBot, AdvancedBot, plus a random
player that also cancels moves and ends unit actions early the way the GUI
does) on 1v1, 1v1v1 and 2v2 maps, with fog of war on and off, and after every
engine call asserts:

* no two units share a tile, and no unit stands on an unwalkable one;
* living units have 0 < health <= max_health (dead ones are removed);
* per turn, each unit moves and acts at most once, plus once per haste it
  received that turn (derived from the action history);
* gold is never negative;
* no turn starts on an eliminated seat, an eliminated player acts no more
  and holds no units or structures;
* the legal-action cache equals a fresh recompute;
* the game ends within max_turns, and replaying its action log reproduces
  the final units, gold, eliminations and result.
"""

import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.core.game_state import GameState
from reinforcetactics.game.bot import AdvancedBot, BalancedRandomBot, RandomBot
from reinforcetactics.utils import replay_actions
from reinforcetactics.utils.file_io import FileIO
from reinforcetactics.utils.replay_actions import execute_replay_action

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

# Engine calls that change state; each is wrapped to check invariants after it.
_MUTATORS = (
    "create_unit",
    "move_unit",
    "attack",
    "paralyze",
    "heal",
    "cure",
    "haste",
    "defence_buff",
    "attack_buff",
    "seize",
    "end_turn",
    "resign",
    "cancel_move",
    "end_unit_turn",
)
# Recorded actions that spend a unit's action (the actor field of each record).
_ACTS = {
    "attack": "attacker_unit_id",
    "seize": "actor_unit_id",
    "paralyze": "actor_unit_id",
    "heal": "actor_unit_id",
    "cure": "actor_unit_id",
    "haste": "actor_unit_id",
    "defence_buff": "actor_unit_id",
    "attack_buff": "actor_unit_id",
}


class GuiLikeRandomBot(RandomBot):
    """RandomBot that sometimes takes a move back or ends a unit's action early, as a GUI player can."""

    def _execute(self, action_key, action):
        super()._execute(action_key, action)
        unit = action.get("unit") or action.get("attacker") or action.get("sorcerer") or action.get("healer")
        if unit is None or unit not in self.game_state.units:
            return
        roll = self._rng.random()
        if action_key == "move" and roll < 0.3:
            self.game_state.cancel_move(unit)
        elif roll < 0.15:
            self.game_state.end_unit_turn(unit)


def _check_invariants(game):
    positions = [(u.x, u.y) for u in game.units]
    assert len(positions) == len(set(positions)), "two units on one tile"
    for unit in game.units:
        assert game.grid.get_tile(unit.x, unit.y).is_walkable(), f"{unit.type} on unwalkable ({unit.x}, {unit.y})"
        assert 0 < unit.health <= unit.max_health, f"{unit.type} at ({unit.x}, {unit.y}) has {unit.health} HP"
    assert all(gold >= 0 for gold in game.player_gold.values()), game.player_gold

    if not game.game_over:
        for player in getattr(game, "eliminated_players", ()):
            assert not any(u.player == player for u in game.units)
            assert not any(t.player == player for row in game.grid.tiles for t in row)

    for player in (game.current_player,):
        if game._legal_actions_cache_valid and player in game._legal_actions_cache:
            assert game._legal_actions_cache[player] == game._compute_legal_actions(player), "stale legal-action cache"


def _check_history(game):
    """Per turn segment (between end_turns): moves and acts per unit <= 1 + hastes it received."""
    segment_moves: Counter = Counter()
    segment_acts: Counter = Counter()
    hastes: Counter = Counter()
    eliminated: set[int] = set()
    for action in game.action_history:
        kind = action["type"]
        if kind != "end_turn":
            assert action["player"] not in eliminated or kind == "eliminate", f"eliminated player acted: {action}"
        if kind == "end_turn":
            segment_moves.clear()
            segment_acts.clear()
            hastes.clear()
        elif kind == "move":
            segment_moves[action["actor_unit_id"]] += 1
        elif kind == "cancel_move":
            segment_moves[action["actor_unit_id"]] -= 1
        elif kind == "eliminate":
            eliminated.add(action["eliminated_player"])
        if kind == "haste":
            hastes[action["target_unit_id"]] += 1
        if kind in _ACTS:
            segment_acts[action[_ACTS[kind]]] += 1
        for unit_id, n in list(segment_moves.items()) + list(segment_acts.items()):
            assert n <= 1 + hastes[unit_id], f"unit {unit_id} acted {n} times with {hastes[unit_id]} haste(s): {action}"


def _instrument(game):
    starts = []

    def wrap(name):
        method = getattr(game, name)

        def checked(*args, **kwargs):
            result = method(*args, **kwargs)
            _check_invariants(game)
            if name == "end_turn" and not game.game_over:
                starts.append(game.current_player)
                assert game.current_player not in getattr(game, "eliminated_players", ()), "turn given to an eliminated seat"
            return result

        return checked

    for name in _MUTATORS:
        setattr(game, name, wrap(name))
    return starts


# A small free-for-all arena, so three seats meet (and capture HQs) within
# the test's turn budget; the bundled 1v1v1 maps are 17-20 tiles across.
SMALL_FFA = [
    "h_1,b_1,p,p,p,p,p,b_2,h_2",
    "b_1,p,p,f,p,f,p,p,b_2",
    "p,p,t,p,p,p,t,p,p",
    "p,f,p,p,m,p,p,f,p",
    "p,p,p,b,p,b,p,p,p",
    "p,p,t,p,p,p,t,p,p",
    "p,p,p,p,r,p,p,p,p",
    "p,p,p,b_3,p,b_3,p,p,p",
    "p,p,p,p,h_3,p,p,p,p",
]

SCENARIOS = [
    # (map, seats, bot classes by seat, fog, seed, max_turns)
    ("1v1/beginner.csv", 2, (GuiLikeRandomBot, AdvancedBot), False, 1, 20),
    ("1v1/beginner.csv", 2, (BalancedRandomBot, GuiLikeRandomBot), True, 2, 20),
    ("1v1/crossroads.csv", 2, (AdvancedBot, GuiLikeRandomBot), True, 7, 20),
    ("1v1v1/triangle_arena.csv", 3, (GuiLikeRandomBot, BalancedRandomBot, RandomBot), True, 3, 12),
    ("1v1v1/three_islands.csv", 3, (RandomBot, AdvancedBot, GuiLikeRandomBot), False, 4, 10),
    ("small_ffa", 3, (AdvancedBot, GuiLikeRandomBot, AdvancedBot), False, 8, 30),
    ("small_ffa", 3, (GuiLikeRandomBot, AdvancedBot, BalancedRandomBot), True, 9, 30),
    ("small_ffa", 3, (AdvancedBot, AdvancedBot, RandomBot), False, 10, 30),
    ("2v2/beginner.csv", 4, (GuiLikeRandomBot, BalancedRandomBot, AdvancedBot, RandomBot), False, 5, 16),
    ("2v2/beginner.csv", 4, (RandomBot, GuiLikeRandomBot, BalancedRandomBot, AdvancedBot), True, 6, 16),
    ("2v2/beginner.csv", 4, (AdvancedBot, AdvancedBot, GuiLikeRandomBot, BalancedRandomBot), False, 11, 20),
]


def _load(map_name):
    if map_name == "small_ffa":
        return np.array([row.split(",") for row in SMALL_FFA], dtype=object)
    return FileIO.load_map(str(MAPS_DIR / map_name))


def _make_bot(cls, game, player, seed):
    rng = random.Random(seed * 10 + player)
    if cls is AdvancedBot:
        return cls(game, player, rng=rng)
    return cls(game, player=player, rng=rng)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: f"{s[0]}-fog{s[3]}-seed{s[4]}")
def test_random_play_keeps_engine_invariants(scenario):
    map_name, seats, bot_classes, fog, seed, max_turns = scenario
    map_data = _load(map_name)
    game = GameState(map_data, num_players=seats, max_turns=max_turns, fog_of_war=fog, rng=random.Random(seed))
    if fog:
        game.update_visibility()
    bots = {p: _make_bot(cls, game, p, seed) for p, cls in zip(range(1, seats + 1), bot_classes)}
    turn_starts = _instrument(game)

    for _ in range(max_turns * seats + 1):
        if game.game_over:
            break
        player = game.current_player
        bots[player].take_turn()
        if not game.game_over and game.current_player == player:
            # A bot whose seat was eliminated mid-turn still ends its turn.
            game.end_turn()

    assert game.game_over, "the game outlived max_turns"
    assert game.turn_number <= max_turns
    assert turn_starts, "no turn was ever handed on"
    _check_history(game)

    # The action log alone reproduces the game.
    info = {
        "num_players": seats,
        "teams": game.teams,
        "max_turns": max_turns,
        "eliminated_players": sorted(game.eliminated_players),
    }
    replay = GameState(map_data, **replay_actions.replay_game_state_kwargs(info))
    for action in game.action_history:
        execute_replay_action(replay, action, lambda x, y: (x, y), schema_version=3)
    assert sorted((u.unit_id, u.player, u.type, u.x, u.y, u.health) for u in replay.units) == sorted(
        (u.unit_id, u.player, u.type, u.x, u.y, u.health) for u in game.units
    )
    assert replay.player_gold == game.player_gold
    assert replay.eliminated_players == game.eliminated_players
    assert (replay.winner, replay.end_reason) == (game.winner, game.end_reason)
