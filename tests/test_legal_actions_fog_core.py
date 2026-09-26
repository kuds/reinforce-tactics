"""The legal-action enumerator and the fog-of-war component, on their own (review core-16).

``core/legal_actions.py`` holds the legality rules and the enumeration built
from them; ``core/fog.py`` holds ``FogOfWar``, which ``GameState`` keeps as
``game.fog``. The engine-level behaviour (masks, validation, saves, search
clones, cancel_move, the ambush rule) is covered by the tests of those
features; these pin what the extraction must keep true: the enumeration is
the rule table applied to every unit, in order, and the component is copied,
saved and restored with the game it belongs to.
"""

import copy
import json
import random

import numpy as np
import pytest

from reinforcetactics.core import legal_actions
from reinforcetactics.core.actions import ACTION_KINDS, ACTOR_KEYS
from reinforcetactics.core.fog import FogOfWar
from reinforcetactics.core.game_state import GameState
from reinforcetactics.core.legal_actions import TARGET_RULES, enumerate_legal_actions
from reinforcetactics.game.bot import AdvancedBot, RandomBot
from reinforcetactics.utils.file_io import FileIO

CASTERS = ["W", "M", "C", "S", "K", "R", "A", "B"]


def _states(fog_of_war: bool, seed: int, turns: int = 16) -> list[GameState]:
    """Search clones of a seeded caster-heavy game at the start of every turn."""
    game = GameState(
        FileIO.load_map("maps/1v1/crossroads.csv"),
        num_players=2,
        enabled_units=CASTERS,
        fog_of_war=fog_of_war,
        engine_overrides={"starting_gold": 3000},
        seed=seed,
    )
    bots = {1: AdvancedBot(game, player=1, rng=random.Random(seed)), 2: RandomBot(game, player=2, rng=random.Random(seed + 1))}
    states = []
    while not game.game_over and game.turn_number < turns:
        states.append(game.clone_for_search())
        player = game.current_player
        bots[player].take_turn()
        if not game.game_over and game.current_player == player:
            game.end_turn()
    return states


@pytest.fixture(scope="module", params=[False, True], ids=["no-fog", "fog"])
def states(request):
    return _states(request.param, seed=3)


def _board(rows: list[str], **kwargs) -> GameState:
    return GameState(np.array([row.split() for row in rows], dtype=object), **{"num_players": 2, "seed": 1, **kwargs})


PLAINS = ["h_1 p p p p p p p p p"] + ["p p p p p p p p p p"] * 4 + ["p p p p p p p p p h_2"]


# --- The enumeration ----------------------------------------------------------


def test_enumerate_legal_actions_is_get_legal_actions_uncached(states):
    for state in states:
        for player in (1, 2):
            state._invalidate_cache()
            fresh = enumerate_legal_actions(state, player)
            assert not state._legal_actions_cache_valid, "enumerating must not fill the cache"
            assert list(fresh) == list(ACTION_KINDS) and fresh["end_turn"] is True
            assert fresh == state.get_legal_actions(player) == state._compute_legal_actions(player)
            # A new result every call: callers may keep or mutate it.
            again = enumerate_legal_actions(state, player)
            assert again == fresh and again is not fresh and again["move"] is not fresh["move"]


def test_targeted_actions_are_the_rule_table_applied_to_every_unit_in_order(states):
    """The enumerator's shortcuts (actor part first, paralyze within attack) change nothing.

    Every targeted list must be exactly: for each ready unit of the player
    with its action left, in ``units`` order, every unit its kind's
    ``TARGET_RULES`` rule accepts, in ``units`` order. Nothing else may
    decide what is offered, or the mask and the engine could drift apart.
    """
    offered = set()
    for state in states:
        player = state.current_player
        actions = enumerate_legal_actions(state, player)
        for kind, rule in TARGET_RULES.items():
            expected = [
                (id(unit), id(target))
                for unit in state.units
                if legal_actions.is_ready_unit(unit, player) and unit.can_attack
                for target in state.units
                if rule(state, unit, target)
            ]
            got = [(id(entry[ACTOR_KEYS[kind]]), id(entry["target"])) for entry in actions[kind]]
            assert got == expected, kind
            if got:
                offered.add(kind)
    assert {"attack", "paralyze", "heal", "haste", "defence_buff", "attack_buff"} <= offered, offered


def test_a_rule_is_its_actor_part_and_its_target_part(states):
    """``TargetRule`` asks both parts; a paralyze target is always an attack target (the enumerator relies on it)."""
    for state in states[::3]:
        for unit in state.units:
            for target in state.units:
                for rule in TARGET_RULES.values():
                    assert rule(state, unit, target) == (rule.actor(unit) and rule.target(state, unit, target))
                if TARGET_RULES["paralyze"](state, unit, target):
                    assert TARGET_RULES["attack"](state, unit, target)


def test_paths_are_the_move_rule(states):
    state = states[len(states) // 2]
    for unit in state.units:
        reachable = legal_actions.find_paths(state, unit)
        destinations = legal_actions.move_paths(state, unit)
        assert list(reachable) == state.get_reachable_positions(unit)
        assert list(destinations) == state.get_move_destinations(unit)
        assert set(destinations) <= set(reachable)
        known = {(u.x, u.y) for u in state.pathing_units(unit.player)}
        assert not known & set(destinations)


def test_creation_rules_answer_for_any_player():
    game = _board(["h_1 b_1 p p", "p p p p", "p p b_2 h_2"])
    assert legal_actions.is_free_spawn_tile(game, 1, 1, 0) and not legal_actions.is_free_spawn_tile(game, 1, 2, 2)
    assert not legal_actions.is_free_spawn_tile(game, 1, 0, 0)  # an HQ never spawns
    assert legal_actions.can_afford(game, 2, "W") and legal_actions.under_unit_cap(game, 2)
    # Player 2's creates are listed although it is player 1's turn.
    assert {(a["x"], a["y"]) for a in enumerate_legal_actions(game, 2)["create_unit"]} == {(2, 2)}
    game.place_unit("W", 2, 2, 1)
    assert not legal_actions.is_free_spawn_tile(game, 2, 2, 2)
    assert enumerate_legal_actions(game, 2)["create_unit"] == []


# --- The fog-of-war component --------------------------------------------------


def _views(game: GameState):
    return {p: (m.state.tobytes(), m.to_dict()) for p, m in game.fog.maps.items()}


@pytest.fixture
def fog_game():
    game = _board(PLAINS, fog_of_war=True)
    game.place_unit("W", 1, 1, 1)
    game.place_unit("A", 8, 4, 2)
    return game


def test_game_state_reads_the_fog_through_its_old_names(fog_game):
    fog = fog_game.fog
    assert isinstance(fog, FogOfWar) and fog.game is fog_game
    assert fog_game.fog_of_war is True and fog_game.fog_of_war_method == "simple_radius"
    assert fog_game.visibility_maps is fog.maps and set(fog.maps) == {1, 2}
    assert fog_game.is_position_visible(1, 2, 1) == fog.is_visible(1, 2, 1)
    assert fog_game.pathing_units(1) == fog.pathing_units(1)

    plain = _board(PLAINS)
    assert plain.fog_of_war is False and plain.fog_of_war_method == "none" and plain.visibility_maps == {}
    unit = plain.place_unit("W", 1, 1, 1)
    assert plain.is_position_visible(9, 5, 1) and plain.pathing_units(1) is plain.units
    plain.capture_visible_enemies_for_unit(unit)
    assert unit.visible_enemies_at_action_start is None


def test_a_fog_refresh_invalidates_the_legal_action_cache(fog_game):
    fog_game.get_legal_actions(1)
    assert fog_game._legal_actions_cache_valid
    fog_game.fog.update(2)
    assert not fog_game._legal_actions_cache_valid


def test_the_fog_state_round_trips_through_a_save(fog_game):
    unit = fog_game.units[0]
    assert fog_game.move_unit(unit, 4, 1)  # reveals tiles, and keeps a pre-move view for cancel_move
    loaded = GameState.from_dict(json.loads(json.dumps(fog_game.to_dict())))

    assert loaded.fog.game is loaded and loaded.fog is not fog_game.fog
    assert loaded.fog_of_war_method == "simple_radius"
    assert _views(loaded) == _views(fog_game)
    loaded_unit = loaded.units[0]
    assert loaded_unit.pre_move_visibility.to_dict() == unit.pre_move_visibility.to_dict()
    assert loaded.cancel_move(loaded_unit) and fog_game.cancel_move(unit)
    assert _views(loaded) == _views(fog_game)


def test_restore_rebuilds_the_fog_when_a_player_is_missing(fog_game):
    saved = fog_game.fog.to_dict()
    assert set(saved) == {"1", "2"}  # str keys: the same before and after JSON

    fresh = FogOfWar(fog_game, enabled=True)
    fresh.restore(saved)
    assert {p: m.to_dict() for p, m in fresh.maps.items()} == {p: m.to_dict() for p, m in fog_game.fog.maps.items()}

    partial = FogOfWar(fog_game, enabled=True)
    partial.restore({"1": saved["1"]})  # e.g. a version 1 save: rebuilt from the board
    assert set(partial.maps) == {1, 2} and partial.maps[2].get_visible_mask().any()

    off = FogOfWar(fog_game, enabled=False)
    off.restore(saved)
    assert off.maps == {}


def test_a_search_clone_has_its_own_fog_bound_to_the_clone(fog_game):
    before = _views(fog_game)
    clone = fog_game.clone_for_search()

    assert clone.fog is not fog_game.fog and clone.fog.game is clone
    assert all(clone.fog.maps[p] is not fog_game.fog.maps[p] for p in (1, 2))
    assert _views(clone) == before

    # Moving in the clone refreshes the clone's view from the clone's units only.
    clone_unit = clone.units[0]
    assert clone.move_unit(clone_unit, 4, 1)
    assert clone.fog.is_visible(7, 2, 1) and not fog_game.fog.is_visible(7, 2, 1)
    clone.fog.maps[2].state[:] = 0
    assert _views(fog_game) == before
    assert fog_game.units[0].x == 1


def test_a_deepcopy_binds_the_copied_fog_to_the_copy(fog_game):
    copied = copy.deepcopy(fog_game)
    assert copied.fog.game is copied and copied.fog.maps[1] is not fog_game.fog.maps[1]
    assert _views(copied) == _views(fog_game)


# --- Re-selecting a unit re-takes its attack snapshot ---------------------------
# The GUI takes a unit's fog-of-war attack snapshot every time a unit that
# has not moved is selected. The snapshot decides which enemies it may
# attack, so rewriting it must drop the cached legal actions: otherwise the
# action menu, built from them, lists an attack the engine refuses or hides
# one it accepts.

FOREST = [
    "h_1 b_1 p p p p",
    "p p p p f p",
    "p p p p p p",
    "p p p p p p",
    "p p p p p p",
    "p p p p b_2 h_2",
]


def _forest_game(scout_at: tuple[int, int]):
    """An Archer in range of an enemy hidden in a forest (seen only from next to it) and a scout."""
    game = _board(FOREST, fog_of_war=True, engine_overrides={"forest_concealment": True})
    archer = game.place_unit("A", 2, 1, 1)
    scout = game.place_unit("W", *scout_at, 1)
    enemy = game.place_unit("W", 4, 1, 2)
    return game, archer, scout, enemy


def _listed_as_legal(game, archer, enemy) -> bool:
    """Whether the (cached) legal actions list the shot, checked against a fresh enumeration and is_legal."""
    cached = game.get_legal_actions()
    assert cached == game._compute_legal_actions(game.current_player), "stale legal-action cache"
    listed = any(a["attacker"] is archer and a["target"] is enemy for a in cached["attack"])
    assert listed == game.is_legal("attack", {"attacker": archer, "target": enemy})
    return listed


def test_reselecting_after_an_enemy_is_revealed_offers_the_attack():
    game, archer, scout, enemy = _forest_game(scout_at=(4, 3))
    game.capture_visible_enemies_for_unit(archer)  # the enemy is hidden in the forest
    assert game.move_unit(scout, 4, 2)  # next to the forest: the enemy is revealed
    assert not _listed_as_legal(game, archer, enemy)  # the archer's action began before

    game.capture_visible_enemies_for_unit(archer)  # the archer is selected again
    assert _listed_as_legal(game, archer, enemy)


def test_reselecting_after_vision_shrinks_withdraws_the_attack():
    game, archer, scout, enemy = _forest_game(scout_at=(4, 2))
    game.capture_visible_enemies_for_unit(archer)  # the scout next to the forest sees the enemy
    assert _listed_as_legal(game, archer, enemy)
    assert game.move_unit(scout, 4, 4)  # the scout walks off: the enemy is hidden again
    assert _listed_as_legal(game, archer, enemy)  # the archer's snapshot still has it

    game.capture_visible_enemies_for_unit(archer)  # the archer is selected again
    assert not _listed_as_legal(game, archer, enemy)


def test_reselecting_without_a_change_keeps_the_cache():
    game, archer, _scout, enemy = _forest_game(scout_at=(4, 2))
    game.capture_visible_enemies_for_unit(archer)
    assert _listed_as_legal(game, archer, enemy)
    game.capture_visible_enemies_for_unit(archer)  # the same snapshot again
    assert game._legal_actions_cache_valid
