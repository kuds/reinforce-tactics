"""GameState.apply_action / is_legal: the one engine entry point for named actions (review core-14).

Seeded play on 1v1, 1v1v1 and 2v2 maps, fog of war on and off, with the
current seat driven by random legal actions and the others by AdvancedBot,
checks at every step that:

* ``is_legal(kind, a)`` is exactly membership in ``get_legal_actions()[kind]``
  over candidates holding every legal entry plus illegal ones: every
  actor/target pair for every targeted kind (spent units, wrong sides, out
  of range, other players' units, hidden targets), moves to tiles off the
  list, seizes by every unit, creates on every structure for every unit
  type and for other players, stale unit references, and every other
  player's legal actions;
* ``apply_action(kind, a).accepted == is_legal(kind, a)``, and a refused
  action changes nothing (``to_dict()`` without timestamps);
* ``is_legal`` changes neither ``to_dict()`` (the fog-of-war attack snapshot
  included) nor the legal-action cache, whether that is filled or not.

The cases random play does not reliably reach have their own tests: a
target hidden by fog of war, an eliminated player, a finished game.
"""

import copy
import dataclasses
import random
from pathlib import Path

import numpy as np
import pytest

from reinforcetactics.core import ActionResult, GameState
from reinforcetactics.core.actions import ACTION_KINDS, ACTOR_KEYS
from reinforcetactics.game.bot import AdvancedBot
from reinforcetactics.rules import ALL_UNIT_TYPES
from reinforcetactics.utils.file_io import FileIO

MAPS_DIR = Path(__file__).resolve().parents[1] / "maps"

TARGETED = ("attack", "paralyze", "heal", "cure", "haste", "defence_buff", "attack_buff")
UNIT_KINDS = ("create_unit", "move", *TARGETED, "seize")


def _strip(obj):
    """``obj`` without the wall-clock ``timestamp`` fields."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "timestamp"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _state(game):
    return _strip(game.to_dict())


def _cache(game):
    """The legal-action cache: its flag, and each player's cached lists (identity and contents)."""
    return game._legal_actions_cache_valid, {
        player: (id(actions), {k: list(v) if isinstance(v, list) else v for k, v in actions.items()})
        for player, actions in game._legal_actions_cache.items()
    }


def _key(kind, action):
    """What identifies a payload: the units involved (by identity) and the tile or unit type."""
    if kind == "create_unit":
        return action["unit_type"], action["x"], action["y"]
    if kind == "move":
        return id(action["unit"]), action["to_x"], action["to_y"]
    if kind == "seize":
        return (id(action["unit"]),)
    if kind == "end_turn":
        return ()
    return id(action[ACTOR_KEYS[kind]]), id(action["target"])


def _candidates(game, rng):
    """(kind, payload) pairs around the current state: every legal entry and many illegal ones."""
    player = game.current_player
    legal = game.get_legal_actions()
    candidates = [(kind, entry) for kind in UNIT_KINDS for entry in legal[kind]]
    candidates.append(("end_turn", {}))

    # A unit reference the engine no longer holds (a bot's stale copy).
    units = list(game.units)
    if units:
        units.append(copy.copy(rng.choice(units)))

    # Every actor/target pair, for every targeted kind.
    for kind in TARGETED:
        for actor in units:
            for target in units:
                candidates.append((kind, {ACTOR_KEYS[kind]: actor, "target": target}))
    # Every unit seizing where it stands, and moving to a few tiles.
    width, height = game.grid.width, game.grid.height
    for unit in units:
        candidates.append(("seize", {"unit": unit}))
        tiles = [(unit.x, unit.y), (unit.x + 1, unit.y), (-1, 0), (width, height)]
        tiles += [(rng.randrange(width), rng.randrange(height)) for _ in range(4)]
        candidates.extend(("move", {"unit": unit, "to_x": x, "to_y": y}) for x, y in tiles)
    # Every unit type on every structure (owned or not, free or occupied),
    # for the current player (implicitly and by name) and for the others.
    structures = [(t.x, t.y) for t in game.grid.get_capturable_tiles()] + [(0, 0), (-1, -1)]
    for x, y in structures:
        for unit_type in ALL_UNIT_TYPES:
            candidates.append(("create_unit", {"unit_type": unit_type, "x": x, "y": y}))
            candidates.append(("create_unit", {"unit_type": unit_type, "x": x, "y": y, "player": player}))
            other = rng.choice([p for p in range(1, game.num_players + 1)])
            candidates.append(("create_unit", {"unit_type": unit_type, "x": x, "y": y, "player": other}))
    # Everything the other players could do on their own turn.
    for other in range(1, game.num_players + 1):
        if other != player:
            other_legal = game._compute_legal_actions(other)
            candidates.extend((kind, entry) for kind in UNIT_KINDS for entry in other_legal[kind])
    return legal, candidates


def _expected(game, legal_keys, kind, action):
    """Whether the payload is one of the current player's listed legal actions."""
    if game.game_over:
        return False
    if kind == "end_turn":
        return True
    if kind == "create_unit" and action.get("player", game.current_player) != game.current_player:
        return False
    return _key(kind, action) in legal_keys[kind]


def _why_illegal(game, kind, action):
    """A coarse label for why a refused candidate is illegal (for the coverage check)."""
    if kind == "create_unit":
        tile = game.grid.get_tile(action["x"], action["y"])
        owned = tile is not None and tile.type == "b" and tile.player == game.current_player
        if action.get("player", game.current_player) != game.current_player:
            return "create_for_other_player"
        if owned and game.get_unit_at_position(action["x"], action["y"]) is not None:
            return "create_occupied"
        if owned and game.player_gold[game.current_player] < game.unit_data[action["unit_type"]]["cost"]:
            return "create_unaffordable"
        return "create_other"
    if kind == "end_turn":
        return "game_over"
    actor = action["unit"] if kind in ("move", "seize") else action[ACTOR_KEYS[kind]]
    if actor not in game.units or ("target" in action and action["target"] not in game.units):
        return "stale_unit"
    if actor.player != game.current_player:
        return "wrong_player"
    if not (actor.can_move if kind == "move" else actor.can_attack):
        return "spent_unit"
    if kind == "attack":
        target = action["target"]
        if game.are_enemies(actor.player, target.player):
            if not game.mechanics.can_reach(actor, target.x, target.y, game.grid):
                return "out_of_range"
            return "hidden_target"
    return "other"


def check_contract(game, rng, seen=None):
    """Check the is_legal / apply_action contract on ``game`` as it stands; changes nothing."""
    before = _state(game)
    legal, candidates = _candidates(game, rng)
    legal_keys = {kind: {_key(kind, entry) for entry in legal[kind]} for kind in UNIT_KINDS}
    if rng.random() < 0.3:
        game._invalidate_cache()  # is_legal must not fill it either
    cache = _cache(game)

    refused = []
    for kind, action in candidates:
        legal_now = game.is_legal(kind, action)
        assert legal_now == _expected(game, legal_keys, kind, action), (kind, action)
        if not legal_now:
            refused.append((kind, action))
            if seen is not None:
                seen.add(_why_illegal(game, kind, action))
    assert _cache(game) == cache, "is_legal changed the legal-action cache"
    assert _state(game) == before, "is_legal changed the game"

    for kind, action in rng.sample(refused, min(10, len(refused))):
        outcome = game.apply_action(kind, action)
        assert outcome.kind == kind and outcome.accepted is False, (kind, action)
        assert _state(game) == before, f"a refused {kind} changed the game"
    return legal


def _play_step(game, legal, rng):
    """One random legal action (or end_turn) for the current player, through apply_action.

    The kind is drawn first, so the many moves don't crowd out attacks, abilities and seizes.
    """
    kinds = [kind for kind in UNIT_KINDS if legal[kind]]
    if not kinds or rng.random() < 0.08:
        kind, action = "end_turn", {}
    else:
        kind = rng.choice(kinds)
        action = rng.choice(legal[kind])
    assert game.is_legal(kind, action)
    history = len(game.action_history)
    outcome = game.apply_action(kind, action)
    assert isinstance(outcome, ActionResult) and outcome.kind == kind and outcome.accepted is True, (kind, action)
    assert len(game.action_history) > history or game.game_over


SCENARIOS = [
    # (map, seats, fog, seed, random-driven seats, max_turns, checks)
    ("1v1/crossroads.csv", 2, False, 1, {1}, 20, 120),
    ("1v1/crossroads.csv", 2, True, 2, {1, 2}, 20, 120),
    ("1v1v1/triangle_arena.csv", 3, False, 3, {1, 3}, 16, 90),
    ("1v1v1/triangle_arena.csv", 3, True, 4, {2}, 16, 90),
    ("2v2/beginner.csv", 4, False, 5, {1, 4}, 16, 90),
    ("2v2/beginner.csv", 4, True, 6, {1, 2, 3}, 16, 90),
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: f"{s[0]}-fog{s[2]}-seed{s[3]}")
def test_is_legal_matches_the_legal_actions_and_apply_action(scenario):
    map_name, seats, fog, seed, random_seats, max_turns, budget = scenario
    game = GameState(
        FileIO.load_map(str(MAPS_DIR / map_name)),
        num_players=seats,
        max_turns=max_turns,
        fog_of_war=fog,
        engine_overrides={"starting_gold": 1000},
        seed=seed,
    )
    rng = random.Random(seed)
    bots = {p: AdvancedBot(game, p, rng=random.Random(seed * 10 + p)) for p in range(1, seats + 1) if p not in random_seats}
    seen: set[str] = set()
    checks = 0
    while not game.game_over and checks < budget:
        player = game.current_player
        if player in bots:
            turn = game.turn_number
            bots[player].take_turn()
            if not game.game_over and (game.current_player, game.turn_number) == (player, turn):
                game.end_turn()
            continue
        legal = check_contract(game, rng, seen)
        checks += 1
        _play_step(game, legal, rng)
    check_contract(game, rng, seen)  # the final state too, over or not
    assert checks >= budget // 2
    assert {"stale_unit", "wrong_player", "spent_unit", "out_of_range", "create_occupied"} <= seen, seen


# --- Targets hidden by fog of war -----------------------------------------

CONCEALMENT_MAP = np.array(
    [
        ["h_1", "b_1", "p", "p", "p", "p"],
        ["p", "p", "p", "p", "f", "p"],
        ["p", "p", "p", "p", "p", "p"],
        ["p", "p", "p", "p", "p", "p"],
        ["p", "p", "p", "p", "p", "p"],
        ["p", "p", "p", "p", "b_2", "h_2"],
    ],
    dtype=object,
)


def test_a_hidden_target_is_not_legal_and_asking_takes_no_snapshot():
    """An enemy in reach but hidden is illegal; is_legal leaves the attack snapshot untaken.

    Forest concealment hides an enemy standing in forest from anyone not next
    to it, so it can be in an Archer's reach unseen. Asking about it must not
    freeze the Archer's targets: once a Warrior walks up and reveals it, the
    Archer (which has not moved, so it has no snapshot) may shoot it, as
    get_legal_actions says. The Warrior itself may not: it only saw the enemy
    by moving.
    """
    game = GameState(CONCEALMENT_MAP, num_players=2, fog_of_war=True, engine_overrides={"forest_concealment": True}, seed=1)
    archer = game.place_unit("A", 2, 1, 1)
    scout = game.place_unit("W", 4, 3, 1)
    enemy = game.place_unit("W", 4, 1, 2)
    assert not game.is_position_visible(enemy.x, enemy.y, 1)

    shot = {"attacker": archer, "target": enemy}
    before = _state(game)
    assert not game.is_legal("attack", shot)
    assert archer.visible_enemies_at_action_start is None
    assert _state(game) == before
    refused = game.apply_action("attack", shot)
    assert refused.accepted is False and refused.result["damage"] == 0
    assert _state(game) == before

    assert game.apply_action("move", {"unit": scout, "to_x": 4, "to_y": 2}).accepted
    assert game.is_position_visible(enemy.x, enemy.y, 1)
    lunge = {"attacker": scout, "target": enemy}
    assert not game.is_legal("attack", lunge)
    assert not game.apply_action("attack", lunge).accepted

    assert game.is_legal("attack", shot)
    assert any(a["attacker"] is archer and a["target"] is enemy for a in game.get_legal_actions()["attack"])
    assert game.apply_action("attack", shot).result["damage"] > 0


# --- Eliminated players and finished games ----------------------------------


def test_an_eliminated_player_may_only_end_its_turn():
    game = GameState(
        FileIO.load_map(str(MAPS_DIR / "1v1v1/triangle_arena.csv")),
        num_players=3,
        engine_overrides={"starting_gold": 2500},
        seed=7,
    )
    others = {p: game._compute_legal_actions(p) for p in (2, 3)}
    game.resign(1)
    assert game.current_player == 1 and game.is_eliminated(1) and not game.game_over

    # Its structures went neutral with it; hand one back to test the gate itself.
    building = next(t for t in game.grid.get_capturable_tiles() if t.type == "b")
    building.player = 1
    create = {"unit_type": "W", "x": building.x, "y": building.y}
    assert game._is_free_spawn_tile(1, building.x, building.y) and game._can_afford(1, "W")
    before = _state(game)
    assert not game.is_legal("create_unit", create)
    assert not game.apply_action("create_unit", create).accepted
    for player, legal in others.items():
        for kind in UNIT_KINDS:
            for entry in legal[kind]:
                assert not game.is_legal(kind, entry), (player, kind, entry)
    assert _state(game) == before

    assert game.is_legal("end_turn", {})
    ended = game.apply_action("end_turn", {})
    assert ended.accepted and game.current_player == 2
    assert game.is_legal("create_unit", {**create, "player": 2}) is game._is_free_spawn_tile(2, building.x, building.y)


def test_nothing_is_legal_once_the_game_is_over():
    game = GameState(
        FileIO.load_map(str(MAPS_DIR / "1v1/crossroads.csv")), num_players=2, engine_overrides={"starting_gold": 2500}, seed=3
    )
    rng = random.Random(3)
    for _ in range(40):
        _play_step(game, game.get_legal_actions(), rng)
    game.resign(2)
    assert game.game_over

    legal = game.get_legal_actions()  # still enumerated: the gates apply on execution
    entries = [(kind, entry) for kind in UNIT_KINDS for entry in legal[kind]] + [("end_turn", {})]
    assert len(entries) > 1
    before = _state(game)
    for kind, entry in entries:
        assert not game.is_legal(kind, entry)
        assert game.apply_action(kind, entry).accepted is False
    assert _state(game) == before
    check_contract(game, rng)


# --- The entry point itself ---------------------------------------------------


def test_unknown_kinds_are_rejected():
    game = GameState(FileIO.load_map(str(MAPS_DIR / "1v1/crossroads.csv")), num_players=2, seed=1)
    for call in (game.apply_action, game.is_legal):
        with pytest.raises(ValueError, match="Unknown action kind"):
            call("wait", {})
    assert set(ACTION_KINDS) == set(game.get_legal_actions())


def test_apply_action_goes_through_the_instance_methods_and_returns_their_result():
    """A wrapper installed on the instance (the imitation recorder's) sees every action."""
    game = GameState(
        FileIO.load_map(str(MAPS_DIR / "1v1/crossroads.csv")), num_players=2, engine_overrides={"starting_gold": 2500}, seed=1
    )
    calls = []
    original = game.create_unit

    def spy(unit_type, x, y, player=None):
        calls.append((unit_type, x, y, player))
        return original(unit_type, x, y, player=player)

    setattr(game, "create_unit", spy)
    entry = game.get_legal_actions()["create_unit"][0]
    outcome = game.apply_action("create_unit", entry)
    assert calls == [(entry["unit_type"], entry["x"], entry["y"], None)]
    assert outcome.accepted and outcome.result is game.get_unit_at_position(entry["x"], entry["y"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.accepted = False
